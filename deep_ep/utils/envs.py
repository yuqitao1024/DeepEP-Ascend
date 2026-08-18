import functools
import inspect
import json
import os
import random
import re
import subprocess
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional, Tuple

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from ..platform import comm_handle_value, get_comm_handle, is_cuda, require_cuda

AscendHybridPreflightRecord = Tuple[str, int, str, str]

_local_rank = None
_local_seed = 0
_global_seed = 0


class _AscendTopologyConfigError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _positive_env_integer(name: str, default: int, error_code: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isascii() or not raw.isdigit() or raw.startswith('0'):
        raise _AscendTopologyConfigError(error_code)
    value = int(raw)
    if value <= 0 or value > 0x7fffffff:
        raise _AscendTopologyConfigError(error_code)
    return value


def _parse_ascend_topology(world_size: int) -> Tuple[str, int, int, int]:
    simulation_raw = os.environ.get(
        'DEEP_EP_ASCEND_LOGICAL_SIMULATION', '0')
    if simulation_raw not in ('0', '1'):
        raise _AscendTopologyConfigError('invalid_logical_simulation')
    logical_simulation = simulation_raw == '1'
    if (logical_simulation and
            'DEEP_EP_ASCEND_SCALE_UP_SIZE' not in os.environ):
        raise _AscendTopologyConfigError('missing_scale_up_size')
    scale_up_default = world_size if not logical_simulation else 0
    scale_up_size = _positive_env_integer(
        'DEEP_EP_ASCEND_SCALE_UP_SIZE', scale_up_default,
        'invalid_scale_up_size')
    if world_size % scale_up_size != 0:
        raise _AscendTopologyConfigError('scale_up_size_not_divisor')
    if logical_simulation:
        if scale_up_size >= world_size:
            raise _AscendTopologyConfigError('insufficient_scale_out_ranks')
        kind = 'logical_simulation'
    else:
        if scale_up_size != world_size:
            raise _AscendTopologyConfigError(
                'non_flat_requires_logical_simulation')
        kind = 'flat_scale_up'

    epoch = _positive_env_integer(
        'DEEP_EP_ASCEND_TOPOLOGY_EPOCH', 1, 'invalid_topology_epoch')
    return kind, scale_up_size, epoch, world_size


def _encode_ascend_preflight_contract(contract: Optional[Dict[str, Any]]) -> str:
    if not isinstance(contract, dict):
        return ""
    try:
        return json.dumps(contract, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return ""


def _decode_ascend_preflight_record(
        value: Any, stage: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Read the fixed primitive record exchanged by Ascend process groups."""
    if (not isinstance(value, tuple) or len(value) != 4 or
            not isinstance(value[0], str) or not isinstance(value[1], int) or
            not isinstance(value[2], str) or not isinstance(value[3], str) or
            value[0] != stage or value[1] not in (0, 1)):
        return False, "invalid_preflight_record", None
    if value[1] == 0:
        return False, value[2] or "invalid_preflight_record", None
    try:
        contract = json.loads(value[3])
    except (TypeError, ValueError):
        return False, "invalid_preflight_record", None
    if not isinstance(contract, dict):
        return False, "invalid_preflight_record", None
    return True, "", contract


def _first_ascend_contract_mismatch(reference: Dict[str, Any],
                                    candidate: Dict[str, Any]) -> str:
    for field in sorted(set(reference) | set(candidate)):
        if reference.get(field) != candidate.get(field):
            return f"{field}_mismatch"
    return "contract_mismatch"


def preflight_ascend_topology(group: dist.ProcessGroup) -> Tuple[str, int, int, int]:
    """Validate and aggregate the explicit Ascend topology configuration."""
    world_size = group.size()
    if world_size < 2:
        raise RuntimeError(
            "DeepEP Ascend backend: topology requires at least two ranks")

    try:
        local_config = _parse_ascend_topology(world_size)
        local = (
            "topology", 1, "", _encode_ascend_preflight_contract({
                "kind": local_config[0],
                "scale_up_size": local_config[1],
                "topology_epoch": local_config[2],
                "world_size": local_config[3],
            }))
    except _AscendTopologyConfigError as error:
        local_config = None
        local = ("topology", 0, error.code, "")

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local, group)
    decoded = []
    for rank, value in enumerate(gathered):
        valid, error_code, config = _decode_ascend_preflight_record(
            value, "topology")
        if not valid:
            raise RuntimeError(
                "DeepEP Ascend backend: topology preflight failed on rank "
                f"{rank} ({error_code})")
        decoded.append(config)

    first_config = decoded[0]
    for rank, config in enumerate(decoded[1:], 1):
        if config == first_config:
            continue
        reason = ("topology_epoch_mismatch" if
                  config.get("topology_epoch") !=
                  first_config.get("topology_epoch") else
                  "topology_shape_mismatch")
        raise RuntimeError(
            "DeepEP Ascend backend: topology preflight failed on rank "
            f"{rank} ({reason})")
    return local_config


def preflight_ascend_contract(group: dist.ProcessGroup, stage: str,
                              contract: Optional[Dict[str, Any]],
                              error_code: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate a rank-local Ascend contract before collective runtime work."""
    world_size = group.size()
    encoded_contract = _encode_ascend_preflight_contract(contract)
    local = (stage, int(error_code is None and bool(encoded_contract)),
             "" if error_code is None else error_code, encoded_contract)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local, group)
    decoded = []
    for rank, value in enumerate(gathered):
        valid_record, code, decoded_contract = _decode_ascend_preflight_record(
            value, stage)
        if not valid_record:
            raise RuntimeError(
                f"DeepEP Ascend backend: {stage} preflight failed on rank "
                f"{rank} ({code})")
        decoded.append(decoded_contract)
    first_contract = decoded[0]
    for rank, candidate in enumerate(decoded[1:], 1):
        if candidate != first_contract:
            raise RuntimeError(
                f"DeepEP Ascend backend: {stage} preflight failed on rank "
                f"{rank} ({_first_ascend_contract_mismatch(first_contract, candidate)})")
    return contract

# Default NIC name for RDMA operations, configurable via environment variable
_DEFAULT_NIC_NAME = os.getenv('EP_NIC_NAME', 'mlx5_0')


def init_seed(global_seed: int) -> None:
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()
    _global_seed = global_seed
    torch.manual_seed(_local_seed)
    random.seed(_local_seed)


def get_local_seed() -> int:
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed


def get_global_seed() -> int:
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.

    Returns:
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.
    """
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    # Set local rank
    global _local_rank
    _local_rank = local_rank

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    init_seed(seed)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def get_physical_domain_size(group: dist.ProcessGroup) -> Tuple[int, int]:
    """
    Get the physical domain sizes (RDMA ranks and NVLink ranks).

    Arguments:
        group: the communication group.

    Returns:
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
    """
    if not is_cuda():
        group_size = group.size()
        if group_size < 2:
            raise RuntimeError(
                "DeepEP Ascend backend: physical domain requires at least two ranks")
        return 1, group_size
    return _C.get_physical_domain_size(
        comm_handle_value(get_comm_handle(group)))


def get_logical_domain_size(group: dist.ProcessGroup, allow_hybrid_mode: bool = True) -> Tuple[int, int]:
    """
    Get the logical domain sizes (scaleout ranks and scaleup ranks).

    Arguments:
        group: the communication group.
        allow_hybrid_mode: whether to enable hybrid mode.

    Returns:
        num_scaleout_ranks: the number of logical scaleout ranks.
        num_scaleup_ranks: the number of logical scaleup ranks.
    """
    if not is_cuda():
        if allow_hybrid_mode:
            raise NotImplementedError(
                "DeepEP Ascend backend: logical domain does not support hybrid mode")
        group_size = group.size()
        if group_size < 2:
            raise RuntimeError(
                "DeepEP Ascend backend: logical domain requires at least two ranks")
        return 1, group_size
    return _C.get_logical_domain_size(
        comm_handle_value(get_comm_handle(group)), allow_hybrid_mode)


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'

        # noinspection PyUnresolvedReferences
        import pynvml
        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [0, ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                assert status == pynvml.NVML_P2P_STATUS_OK, \
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'

        # Close NVML
        pynvml.nvmlShutdown()


def check_torch_deterministic() -> None:
    """
    Ensure PyTorch deterministic algorithms and fill_uninitialized_memory are not both enabled.
    When both are on, `torch.empty()` calls an initialization kernel that may overlap with communication streams,
    causing errors.
    """
    assert not (torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory)


@functools.lru_cache()
def get_nvlink_gbs(factor: float = 0.9) -> float:
    """
    Get the total NVLink bandwidth in GB/s, cached.

    Arguments:
        factor: the bandwidth efficiency factor.

    Returns:
        gbs: the total NVLink bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                                capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
        assert match

        gpu_block = match.group(0)
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'
        link_matches = re.findall(link_pattern, gpu_block)
        assert link_matches
        return sum(float(bw) for bw in link_matches) * factor
    except Exception as e:
        print(f'Failed to get NVLink connection speed: {e}')
        return 0


@functools.lru_cache()
def check_fast_rdma_atomic_support(nic_name: str = _DEFAULT_NIC_NAME) -> bool:
    """
    Check whether the NIC supports fast RDMA atomic operations (MT4131 or newer).

    Arguments:
        nic_name: the NIC device name.

    Returns:
        supported: `True` if fast RDMA atomics are supported.
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = rf"CA '{nic_name}'.*?CA type:\s*(\S+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        return match.group(1) == 'MT4131'
    except Exception:
        return False


@functools.lru_cache()
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout

        pattern = rf"CA '{nic_name}'.*?Port \d+:\s*.*?Rate:\s*(\d+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        rate = int(match.group(1))
        return rate / 8
    except Exception as e:
        print(f'Failed to get RDMA connection speed: {e}')
        return 0
