import ast
import importlib.abc
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ELASTIC_SOURCE = ROOT / "deep_ep/buffers/elastic.py"
ENVS_SOURCE = ROOT / "deep_ep/utils/envs.py"
API_SURFACE_PROBE = ROOT / "tests/ascend/production/api_surface.py"

API_SURFACE_SPEC = importlib.util.spec_from_file_location(
    "ascend_api_surface", API_SURFACE_PROBE)
API_SURFACE = importlib.util.module_from_spec(API_SURFACE_SPEC)
API_SURFACE_SPEC.loader.exec_module(API_SURFACE)

TRANSPORT_ERROR = (
    "DeepEP Ascend backend: {} is unavailable until the Ascend device "
    "transport is implemented")
GATED_METHODS = {
    "get_comm_stream": "get_comm_stream",
    "get_engram_storage_size_hint": "get_engram_storage_size_hint",
    "get_pp_buffer_size_hint": "get_pp_buffer_size_hint",
    "get_agrs_num_max_session_bytes": "get_agrs_num_max_session_bytes",
    "get_agrs_buffer_size_hint": "get_agrs_buffer_size_hint",
    "engram_write": "engram_write",
    "engram_fetch": "engram_fetch",
    "pp_set_config": "pp_set_config",
    "pp_send": "pp_send",
    "pp_recv": "pp_recv",
    "create_agrs_session": "create_agrs_session",
    "destroy_agrs_session": "destroy_agrs_session",
    "agrs_new_session": "agrs_new_session",
    "agrs_set_config": "agrs_set_config",
    "agrs_get_inplace_tensor": "agrs_get_inplace_tensor",
    "all_gather": "all_gather",
    "get_theoretical_num_sms": "get_theoretical_num_sms",
    "get_theoretical_num_qps": "get_theoretical_num_qps",
}


class _FakeDType:
    def __init__(self, itemsize):
        self.itemsize = itemsize


class _FakeTensor:
    def __init__(self, device_type="cpu", shape=()):
        self.device = types.SimpleNamespace(type=device_type)
        self.shape = shape


class _FakeSize(tuple):
    pass


class _FakeStream:
    def __init__(self, stream_id=0, device_index=0, device_type=1):
        self.stream_id = stream_id
        self.device_index = device_index
        self.device_type = device_type


class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"argument inspection reached {name}")


class _ForbiddenImportFinder(importlib.abc.MetaPathFinder):
    FORBIDDEN = {
        "deep_ep.buffers.legacy",
        "deep_ep.utils.comm",
        "deep_ep.utils.find_pkgs",
        "pynvml",
        "torch_npu",
        "cann",
        "hccl",
    }

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(f"{name}.")
               for name in self.FORBIDDEN):
            raise AssertionError(f"forbidden Ascend import: {fullname}")
        return None


class _FakeGroup:
    def __init__(self, events, rank=0, size=1, comm_pointer=4242):
        self.events = events
        self._rank = rank
        self._size = size
        self.comm_pointer = comm_pointer

    def rank(self):
        self.events.append("group.rank")
        return self._rank

    def size(self):
        self.events.append("group.size")
        return self._size

    def barrier(self):
        self.events.append("group.barrier")

    def _get_backend(self, device):
        self.events.append(("group.backend", device))
        return types.SimpleNamespace(
            _comm_ptr=lambda: self.comm_pointer,
            get_hccl_comm=lambda local_rank: self.comm_pointer)


def _transport_error(operation):
    return NotImplementedError(TRANSPORT_ERROR.format(operation))


def _install_fake_torch(platform, events):
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.Tensor = _FakeTensor
    torch.Size = _FakeSize
    torch.Stream = _FakeStream
    torch.dtype = _FakeDType
    torch.bfloat16 = _FakeDType(2)
    torch.float8_e4m3fn = _FakeDType(1)
    torch.float32 = torch.float = _FakeDType(4)
    torch.int64 = torch.long = _FakeDType(8)
    torch.int32 = torch.int = _FakeDType(4)
    torch.uint8 = torch.bool = _FakeDType(1)

    def compile_decorator(function=None, **kwargs):
        if function is not None:
            return function
        return lambda decorated: decorated

    torch.compile = compile_decorator
    torch.device = lambda device_type, index=None: (
        device_type if index is None else f"{device_type}:{index}")
    torch.manual_seed = lambda seed: events.append(("manual_seed", seed))
    torch.set_default_dtype = lambda dtype: events.append(("default_dtype", dtype))
    torch.set_default_device = lambda device: events.append(("default_device", device))
    torch.are_deterministic_algorithms_enabled = lambda: False
    torch.utils = types.SimpleNamespace(
        deterministic=types.SimpleNamespace(fill_uninitialized_memory=False))

    class FakeCuda:
        def _record(self, name, *args, **kwargs):
            if platform != "cuda":
                raise AssertionError(f"Ascend import touched torch.cuda.{name}")
            events.append((f"torch.cuda.{name}", args, kwargs))

        def synchronize(self):
            self._record("synchronize")

        def get_device_name(self):
            self._record("get_device_name")
            return "Fake H100"

        def current_device(self):
            self._record("current_device")
            return 0

        def get_device_properties(self, device):
            self._record("get_device_properties", device)
            return types.SimpleNamespace(
                multi_processor_count=80, total_memory=1 << 30)

        def set_device(self, device):
            self._record("set_device", device)

        def Stream(self, **kwargs):
            self._record("Stream", **kwargs)
            return _FakeStream(**kwargs)

    torch.cuda = FakeCuda()

    class FakeNpu:
        def synchronize(self):
            if platform != "ascend":
                raise AssertionError("CUDA import touched torch.npu.synchronize")
            events.append("torch.npu.synchronize")

    torch.npu = FakeNpu()

    distributed = types.ModuleType("torch.distributed")
    distributed.ProcessGroup = _FakeGroup
    distributed.get_rank = lambda: 0
    distributed.get_world_size = lambda: 1
    distributed.new_group = lambda ranks: None
    distributed.init_process_group = lambda **kwargs: None
    distributed.barrier = lambda: events.append("dist.barrier")
    def all_gather_object(output, value, group):
        output[:] = [value] * len(output)

    distributed.all_gather_object = all_gather_object
    torch.distributed = distributed

    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = distributed
    return torch


def _install_fake_extension(platform, events):
    extension = types.ModuleType("deep_ep._C")
    extension.get_platform = lambda: platform
    extension.topk_idx_t = _FakeDType(8)
    extension.runtime_args = []
    extension.runtime_instances = []
    extension.size_calls = []

    class EventHandle:
        def current_stream_wait(self):
            if platform == "ascend":
                raise _transport_error("current_stream_wait")
            events.append("event.current_stream_wait")

    class ElasticRuntime:
        def __init__(self, *args):
            extension.runtime_args.append(args)
            extension.runtime_instances.append(self)
            self.dispatch_calls = []
            self.combine_calls = []
            events.append(("runtime.construct", args))

        def destroy(self):
            events.append("runtime.destroy")

        def get_comm_stream(self):
            events.append("runtime.get_comm_stream")
            return _FakeStream(17, 2, 1)

        def get_logical_domain_size(self):
            events.append("runtime.get_logical_domain_size")
            return (1, 2) if platform == "ascend" else (2, 4)

        def get_physical_domain_size(self):
            events.append("runtime.get_physical_domain_size")
            return (1, 2) if platform == "ascend" else (2, 4)

        def barrier(self, use_comm_stream, with_cpu_sync, sequential):
            events.append(("runtime.barrier", use_comm_stream, with_cpu_sync, sequential))

        def create_agrs_session(self):
            events.append("runtime.create_agrs_session")

        def destroy_agrs_session(self):
            events.append("runtime.destroy_agrs_session")

        def dispatch(self, *args):
            self.dispatch_calls.append(args)
            recv_src_metadata = _FakeTensor(shape=(1,))
            cloned_topk_idx = _FakeTensor(
                args[2].device.type, args[2].shape)
            token_metadata_at_forward = (
                _FakeTensor(shape=(96,)) if platform == "ascend" else None)
            return (args[0], args[1], args[2], args[3], cloned_topk_idx,
                    1, 1, [], _FakeTensor(), _FakeTensor(), _FakeTensor(),
                    recv_src_metadata, _FakeTensor(), token_metadata_at_forward,
                    None, None if platform == "ascend" else EventHandle())

        def combine(self, *args):
            self.combine_calls.append(args)
            return args[0], args[1], None if platform == "ascend" else EventHandle()

    def calculate_elastic_buffer_size(*args):
        extension.size_calls.append(args)
        if platform == "ascend" and args[5]:
            raise _transport_error("calculate_elastic_buffer_size")
        return 2 * 1024 * 1024 if platform == "ascend" else 8192

    extension.EventHandle = EventHandle
    extension.ElasticBuffer = ElasticRuntime
    extension.calculate_elastic_buffer_size = calculate_elastic_buffer_size

    if platform == "cuda":
        class Config:
            pass

        def init_jit(*args):
            events.append(("extension.init_jit", args))

        extension.Config = Config
        extension.init_jit = init_jit
        extension.destroy_nccl_comm = lambda handle: events.append(
            ("extension.destroy_nccl_comm", handle))
        extension.get_local_nccl_unique_id = lambda: "unused"
        def create_nccl_comm(unique_id, size, rank):
            events.append((
                "extension.create_nccl_comm", unique_id, size, rank,
                os.environ.get("NCCL_SYM_REUSE_SYSMEM_HANDLES"),
                os.environ.get("NCCL_WIN_STRIDE")))
            return 1234

        extension.create_nccl_comm = create_nccl_comm
        extension.create_cpu_handle = lambda num_bytes: (
            events.append(("extension.create_cpu_handle", num_bytes)) or (101, 202))
        extension.get_physical_domain_size = lambda handle: (2, 4)
        extension.get_logical_domain_size = lambda handle, hybrid: (2, 4)

    sys.modules["deep_ep._C"] = extension
    return extension


def _install_fake_find_pkgs(events):
    find_pkgs = types.ModuleType("deep_ep.utils.find_pkgs")

    def find_nccl_root(optional=False):
        events.append(("find_nccl_root", optional))
        return "/fake/nccl"

    find_pkgs.find_nccl_root = find_nccl_root
    sys.modules["deep_ep.utils.find_pkgs"] = find_pkgs


def _load_low_level_extension(path):
    """Load a candidate extension without executing deep_ep.__init__."""
    if path.suffix != ".py":
        import torch  # noqa: F401

    spec = importlib.util.spec_from_file_location("deep_ep._C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create an extension loader for: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("deep_ep._C")
    sys.modules["deep_ep._C"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("deep_ep._C", None)
        else:
            sys.modules["deep_ep._C"] = previous
    return module


def _find_in_place_extension():
    extensions = sorted((ROOT / "deep_ep").glob("_C*.so"))
    return extensions[0] if extensions else None


def _probe_low_level_extension(path, forbid_package=False):
    command = [sys.executable, str(pathlib.Path(__file__).resolve()),
               "--probe-low-level-extension", str(path)]
    if forbid_package:
        command.append("--forbid-package")
    result = subprocess.run(
        command,
        cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"low-level extension probe failed\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}")
    return result.stdout.strip()


def _is_ascend_extension_available(path, forbid_package=False):
    return _probe_low_level_extension(path, forbid_package) == "ascend"


def _load_package(platform, block_accelerator_imports=False):
    events = []
    _install_fake_torch(platform, events)
    extension = _install_fake_extension(platform, events)
    if platform == "cuda":
        _install_fake_find_pkgs(events)
        os.environ["EP_SUPPRESS_NCCL_CHECK"] = "1"
        os.environ["EP_REUSE_NCCL_COMM"] = "1"
        os.environ["CUDA_HOME"] = "/fake/cuda"

    blocker = _ForbiddenImportFinder() if block_accelerator_imports else None
    if blocker is not None:
        sys.meta_path.insert(0, blocker)
    sys.path.insert(0, str(ROOT))
    try:
        import deep_ep
    finally:
        if blocker is not None:
            sys.meta_path.remove(blocker)
    return deep_ep, extension, events


def _assert_transport_error(operation, call):
    try:
        call()
    except Exception as error:
        assert type(error) is NotImplementedError, (operation, type(error), error)
        assert str(error) == TRANSPORT_ERROR.format(operation), (operation, error)
    else:
        raise AssertionError(f"{operation} did not raise")


def _scenario_ascend_import():
    deep_ep, extension, events = _load_package("ascend", True)
    platform = importlib.import_module("deep_ep.platform")

    assert extension.get_platform() == "ascend"
    assert deep_ep.ElasticBuffer is not None
    assert deep_ep.EPHandle is not None
    assert deep_ep.EventOverlap is not None
    assert deep_ep.EventHandle is extension.EventHandle
    assert deep_ep.topk_idx_t is extension.topk_idx_t
    assert not hasattr(deep_ep, "Buffer")
    assert not hasattr(deep_ep, "Config")
    assert not hasattr(deep_ep, "get_physical_domain_size")
    assert not hasattr(deep_ep, "get_logical_domain_size")
    assert not hasattr(extension, "init_jit")

    assert platform.COMPILED_PLATFORM == "ascend"
    assert not platform.is_cuda()
    group = _FakeGroup(events, rank=1, size=2, comm_pointer=0x1234)
    assert platform.get_comm_handle(group) == 0x1234
    assert platform.comm_handle_value(0x1234) == 0x1234
    assert platform.comm_handle_value(None) == 0

    for wrapped in ((0x1234,), [0x1234]):
        wrapped_group = _FakeGroup(
            events, rank=1, size=2, comm_pointer=wrapped)
        assert platform.get_comm_handle(wrapped_group) == 0x1234

    for invalid in (0, None, (), (1, 2)):
        invalid_group = _FakeGroup(
            events, rank=1, size=2, comm_pointer=invalid)
        try:
            platform.get_comm_handle(invalid_group)
        except RuntimeError as error:
            assert "HCCL communicator" in str(error), error
        else:
            raise AssertionError(f"invalid HCCL communicator accepted: {invalid!r}")

    class MissingAccessorGroup(_FakeGroup):
        def _get_backend(self, device):
            return types.SimpleNamespace()

    try:
        platform.get_comm_handle(MissingAccessorGroup(events, rank=1, size=2))
    except RuntimeError as error:
        assert str(error) == "ProcessGroupHCCL.get_hccl_comm is unavailable"
    else:
        raise AssertionError("missing HCCL communicator accessor was accepted")

    platform.synchronize()
    assert "torch.npu.synchronize" in events
    assert not any(str(event).startswith("torch.cuda") for event in events)
    _assert_transport_error("adapter", lambda: platform.require_cuda("adapter"))
    platform.validate_device_type(_FakeTensor("npu"), "validate_device_type")
    try:
        platform.validate_device_type(_FakeTensor("cpu"), "validate_device_type")
    except ValueError as error:
        assert str(error) == (
            "DeepEP Ascend backend: validate_device_type requires an NPU tensor")
    else:
        raise AssertionError("CPU tensor was accepted by Ascend validation")
    _assert_transport_error("get_comm_stream", lambda: platform.wrap_stream(_Poison()))

    event = platform.capture_event()
    wrapped = types.SimpleNamespace(event=event)
    assert platform.unwrap_event(None) is None
    assert platform.unwrap_event(event) is event
    assert platform.unwrap_event(wrapped) is event
    assert deep_ep.ElasticBuffer.capture().__class__ is extension.EventHandle
    _assert_transport_error("current_stream_wait", event.current_stream_wait)

    for name in _ForbiddenImportFinder.FORBIDDEN:
        assert name not in sys.modules, name


def _scenario_invalid_platform():
    _install_fake_torch("rocm", [])
    _install_fake_extension("rocm", [])
    sys.path.insert(0, str(ROOT))
    try:
        import deep_ep  # noqa: F401
    except RuntimeError as error:
        assert str(error) == "Unsupported compiled DeepEP platform: rocm"
    else:
        raise AssertionError("unsupported compiled platform was accepted")


def _scenario_ascend_construction():
    deep_ep, extension, events = _load_package("ascend", True)
    group = _FakeGroup(events, rank=1, size=2)
    os.environ["EP_OVERRIDE_RDMA_SL"] = "11"
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True, num_allocated_qps=77)

    assert buffer.group is group
    assert buffer.rank_idx == 1
    assert buffer.num_ranks == 2
    assert buffer.num_bytes == 2 * 1024 * 1024
    assert buffer.num_allocated_qps == 0
    assert buffer.comm_handle == 4242
    assert not hasattr(buffer, "nccl_comm_handle")
    assert (buffer.num_scaleout_ranks, buffer.num_scaleup_ranks) == (1, 2)
    assert (buffer.scaleout_rank_idx, buffer.scaleup_rank_idx) == (0, 1)
    assert (buffer.num_rdma_ranks, buffer.num_nvlink_ranks) == (1, 2)
    assert len(extension.runtime_args) == 1
    runtime_args = extension.runtime_args[0]
    assert runtime_args[:6] == (
        1, 2, 4242, [], 2 * 1024 * 1024, 0)
    assert runtime_args[9:11] == (3, 0)
    assert "group.barrier" not in events
    assert events.count("runtime.get_logical_domain_size") == 1
    assert events.count("runtime.get_physical_domain_size") == 1
    assert events.count("torch.npu.synchronize") == 2
    assert not any(isinstance(event, tuple) and str(event[0]).startswith("torch.cuda")
                   for event in events)

    buffer.destroy()
    assert buffer.runtime is None
    assert buffer.comm_handle is None

    for num_bytes in (0, -1):
        try:
            deep_ep.ElasticBuffer(
                group, num_bytes=num_bytes, allow_hybrid_mode=False,
                explicitly_destroy=True)
        except ValueError as error:
            assert str(error) == "num_bytes must be positive"
        else:
            raise AssertionError(f"non-positive buffer size {num_bytes} was accepted")


def _scenario_ascend_implicit_size():
    deep_ep, extension, events = _load_package("ascend", True)
    group = _FakeGroup(events, rank=0, size=2)
    buffer = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=1, hidden=16,
        allow_hybrid_mode=False, explicitly_destroy=True)
    assert buffer.num_bytes == 2 * 1024 * 1024
    assert extension.size_calls == [(4242, 1, 16, 0, False, False, True)]
    buffer.destroy()

    _assert_transport_error(
        "calculate_elastic_buffer_size",
        lambda: deep_ep.ElasticBuffer.get_buffer_size_hint(group, 2, 32, 4))
    assert extension.size_calls[-1] == (4242, 2, 32, 4, False, True, True)


def _scenario_ascend_method_gates():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=1, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    buffer.barrier(use_comm_stream=False, with_cpu_sync=True, sequential=True)
    assert events[-1] == ("runtime.barrier", False, True, True)
    assert buffer.get_logical_domain_size() == (1, 2)
    assert buffer.get_physical_domain_size() == (1, 2)
    runtime_barriers = [event for event in events
                        if isinstance(event, tuple) and
                        event[0] == "runtime.barrier"]
    try:
        buffer.barrier(sequential=False)
    except RuntimeError as error:
        assert str(error) == (
            "DeepEP Ascend backend: barrier requires sequential=True")
    else:
        raise AssertionError("non-sequential Ascend barrier was accepted")
    assert [event for event in events
            if isinstance(event, tuple) and event[0] == "runtime.barrier"] == \
        runtime_barriers
    poison = _Poison()
    calls = {
        "get_comm_stream": buffer.get_comm_stream,
        "get_engram_storage_size_hint": lambda: buffer.get_engram_storage_size_hint(
            poison, poison, poison),
        "get_pp_buffer_size_hint": lambda: buffer.get_pp_buffer_size_hint(
            poison, poison),
        "get_agrs_num_max_session_bytes": lambda: buffer.get_agrs_num_max_session_bytes(
            poison, poison, poison),
        "get_agrs_buffer_size_hint": lambda: buffer.get_agrs_buffer_size_hint(
            poison, poison),
        "engram_write": lambda: buffer.engram_write(poison),
        "engram_fetch": lambda: buffer.engram_fetch(poison),
        "pp_set_config": lambda: buffer.pp_set_config(poison, poison),
        "pp_send": lambda: buffer.pp_send(poison, poison),
        "pp_recv": lambda: buffer.pp_recv(poison, poison),
        "create_agrs_session": buffer.create_agrs_session,
        "destroy_agrs_session": buffer.destroy_agrs_session,
        "agrs_new_session": lambda: buffer.agrs_new_session(False),
        "agrs_set_config": lambda: buffer.agrs_set_config(poison, poison),
        "agrs_get_inplace_tensor": lambda: buffer.agrs_get_inplace_tensor(poison, poison),
        "all_gather": lambda: buffer.all_gather(poison),
        "get_theoretical_num_sms": lambda: buffer.get_theoretical_num_sms(
            [poison], [poison]),
        "get_theoretical_num_qps": lambda: buffer.get_theoretical_num_qps(poison),
    }
    assert calls.keys() == GATED_METHODS.keys()
    for method, operation in GATED_METHODS.items():
        _assert_transport_error(operation, calls[method])

    envs = importlib.import_module("deep_ep.utils.envs")
    group = _FakeGroup(events, rank=1, size=2)
    assert envs.get_physical_domain_size(group) == (1, 2)
    assert envs.get_logical_domain_size(group, False) == (1, 2)
    buffer.destroy()


def _scenario_ascend_contextmanager_gate():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    _assert_transport_error(
        "agrs_new_session", lambda: buffer.agrs_new_session(False))


def _scenario_ascend_dispatch():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    x = _FakeTensor("npu", (1, 16))
    topk_idx = _FakeTensor("npu", (1, 1))
    topk_weights = _FakeTensor("npu", (1, 1))

    _, _, recv_topk_weights, handle, event = buffer.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=2, num_max_tokens_per_rank=1)
    dispatch_args = runtime.dispatch_calls[-1]
    assert len(dispatch_args) == 29
    assert dispatch_args[2] is topk_idx
    assert dispatch_args[3] is topk_weights
    assert dispatch_args[18:20] == (1, 0)
    assert dispatch_args[20:24] == (None, None, False, False)
    assert dispatch_args[25] is True
    assert recv_topk_weights is topk_weights
    assert isinstance(handle, deep_ep.EPHandle)
    assert handle.num_sms == 1
    assert handle.topk_idx is not topk_idx
    assert handle.token_metadata_at_forward is not None
    assert event.event is None

    cached_topk_weights = _FakeTensor("npu", (1, 1))
    _, _, recv_topk_weights, cached_handle, cached_event = buffer.dispatch(
        x, topk_weights=cached_topk_weights, handle=handle)
    cached_args = runtime.dispatch_calls[-1]
    expected_cached_fields = (
        1, 1, [],
        handle.psum_num_recv_tokens_per_scaleup_rank,
        handle.psum_num_recv_tokens_per_expert,
        handle.num_unaligned_recv_tokens_per_expert,
        handle.dst_buffer_slot_idx,
        handle.token_metadata_at_forward,
        handle.recv_src_metadata,
        handle.channel_linked_list,
    )
    assert cached_args[2] is handle.topk_idx
    assert cached_args[3] is cached_topk_weights
    assert cached_args[5:15] == expected_cached_fields
    assert cached_args[25] is False
    assert recv_topk_weights is cached_topk_weights
    assert cached_handle is handle
    assert cached_event.event is None

    for name, kwargs in (("num_sms", {"num_sms": 2}),
                         ("num_qps", {"num_qps": 1})):
        try:
            buffer.dispatch(x, topk_idx=topk_idx, num_experts=2,
                            num_max_tokens_per_rank=1, **kwargs)
        except RuntimeError as error:
            assert str(error) == (
                "DeepEP Ascend backend: dispatch requires num_sms=1 and num_qps=0")
        else:
            raise AssertionError(f"Ascend dispatch accepted {name}={kwargs[name]}")

    no_weights_idx = _FakeTensor("npu", (1, 1))
    recv_x, recv_topk_idx, recv_topk_weights, no_weights_handle, no_weights_event = \
        buffer.dispatch(
            x, topk_idx=no_weights_idx, num_experts=2,
            num_max_tokens_per_rank=1, do_handle_copy=False)
    no_weights_args = runtime.dispatch_calls[-1]
    assert no_weights_args[0] is x
    assert no_weights_args[2] is no_weights_idx
    assert no_weights_args[3] is None
    assert no_weights_args[18:20] == (1, 0)
    assert no_weights_args[24] is False
    assert recv_x is x
    assert recv_topk_idx is no_weights_idx
    assert recv_topk_weights is None
    assert no_weights_handle.topk_idx is no_weights_idx
    assert no_weights_event.event is None

    empty_x = _FakeTensor("npu", (0, 16))
    empty_topk_idx = _FakeTensor("npu", (0, 1))
    recv_x, recv_topk_idx, recv_topk_weights, empty_handle, empty_event = \
        buffer.dispatch(
            empty_x, topk_idx=empty_topk_idx, num_experts=2,
            num_max_tokens_per_rank=1, num_sms=1, num_qps=0)
    empty_args = runtime.dispatch_calls[-1]
    assert empty_args[0] is empty_x
    assert empty_args[2] is empty_topk_idx
    assert empty_args[3] is None
    assert empty_args[18:20] == (1, 0)
    assert empty_args[25] is True
    assert recv_x is empty_x
    assert recv_topk_idx is empty_topk_idx
    assert recv_topk_weights is None
    assert empty_handle.topk_idx is not empty_topk_idx
    assert empty_event.event is None

    buffer.destroy()


def _scenario_ascend_dispatch_optimized():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    x = _FakeTensor("npu", (1, 16))
    topk_idx = _FakeTensor("npu", (1, 1))
    expected_error = (
        "DeepEP Ascend backend: dispatch requires num_sms=1 and num_qps=0")

    for kwargs in ({"num_sms": 2}, {"num_qps": 1}):
        call_count = len(runtime.dispatch_calls)
        try:
            buffer.dispatch(_Poison(), **kwargs)
        except RuntimeError as error:
            if str(error) != expected_error:
                raise AssertionError(f"unexpected validation error: {error}")
        else:
            raise AssertionError(f"Ascend dispatch accepted {kwargs}")
        if len(runtime.dispatch_calls) != call_count:
            raise AssertionError("invalid Ascend dispatch reached runtime")

    buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=2,
        num_max_tokens_per_rank=1, num_sms=1, num_qps=0)
    if len(runtime.dispatch_calls) != 1:
        raise AssertionError("valid explicit Ascend dispatch did not reach runtime")
    if runtime.dispatch_calls[-1][18:20] != (1, 0):
        raise AssertionError("valid explicit Ascend counts changed before runtime")
    buffer.destroy()


def _scenario_ascend_combine():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    x = _FakeTensor("npu", (2, 16))
    topk_weights = _FakeTensor("npu", (2, 2))
    bias_0 = _FakeTensor("npu", (1, 16))
    bias_1 = _FakeTensor("npu", (1, 16))
    recv_src_metadata = _FakeTensor("npu", (2, 4))
    topk_idx = _FakeTensor("npu", (1, 2))
    rank_prefix = _FakeTensor("npu", (2,))
    descriptor = _FakeTensor("npu", (96,))
    handle = deep_ep.EPHandle(
        False, 2, 4, 4, 1, topk_idx, 2, 2, [], rank_prefix,
        _FakeTensor("npu", (1,)), _FakeTensor("npu", (1,)),
        recv_src_metadata, _FakeTensor("npu", (1, 2)), descriptor, None)

    combined_x, combined_weights, event = buffer.combine(
        x, handle, topk_weights=topk_weights, bias=(bias_0, bias_1))
    assert combined_x is x
    assert combined_weights is topk_weights
    assert event.event is None
    assert runtime.combine_calls[-1] == (
        x, topk_weights, bias_0, bias_1, recv_src_metadata, topk_idx,
        rank_prefix, descriptor, None, 2, 4, 1, 0,
        None, None, False, False, False)

    _, _, one_bias_event = buffer.combine(x, handle, bias=bias_0, num_sms=1)
    assert runtime.combine_calls[-1][2:4] == (bias_0, None)
    assert one_bias_event.event is None
    buffer.combine(x, handle)
    assert runtime.combine_calls[-1][2:4] == (None, None)

    invalid_calls = (
        ("num_sms", {"num_sms": 2}),
        ("num_qps", {"num_qps": 1}),
        ("previous_event", {"previous_event": deep_ep.EventOverlap(
            extension.EventHandle())}),
        ("previous_event_before_epilogue", {
            "previous_event_before_epilogue": extension.EventHandle()}),
        ("async_with_compute_stream", {"async_with_compute_stream": True}),
        ("allocate_on_comm_stream", {"allocate_on_comm_stream": True}),
    )
    calls_before = len(runtime.combine_calls)
    for name, kwargs in invalid_calls:
        try:
            buffer.combine(x, handle, **kwargs)
        except RuntimeError as error:
            assert "DeepEP Ascend backend: combine" in str(error), (name, error)
        else:
            raise AssertionError(f"Ascend combine accepted unsupported {name}")
    assert len(runtime.combine_calls) == calls_before
    buffer.destroy()


def _scenario_ascend_weak_lru_gate():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    poison = _Poison()
    _assert_transport_error(
        "get_theoretical_num_sms",
        lambda: buffer.get_theoretical_num_sms([poison], [poison]))


def _scenario_cuda_preservation():
    profile_events = []

    def profile(frame, event, arg):
        if event != "call":
            return
        filename = pathlib.Path(frame.f_code.co_filename)
        if filename == ROOT / "deep_ep/__init__.py" and frame.f_code.co_name in {
                "check_nccl_so", "init_jit"}:
            profile_events.append(frame.f_code.co_name)
        if filename == ROOT / "deep_ep/buffers/legacy.py" and frame.f_code.co_name == "Buffer":
            profile_events.append("legacy.Buffer")
        if filename == ROOT / "deep_ep/buffers/elastic.py" and frame.f_code.co_name == "ElasticBuffer":
            profile_events.append("elastic.ElasticBuffer")

    sys.setprofile(profile)
    try:
        deep_ep, extension, events = _load_package("cuda")
    finally:
        sys.setprofile(None)

    assert profile_events[:4] == [
        "check_nccl_so", "init_jit", "legacy.Buffer", "elastic.ElasticBuffer"]
    assert hasattr(deep_ep, "Buffer")
    assert deep_ep.Config is extension.Config
    assert callable(deep_ep.get_physical_domain_size)
    assert callable(deep_ep.get_logical_domain_size)
    init_event = next(event for event in events
                      if isinstance(event, tuple) and event[0] == "extension.init_jit")
    assert init_event[1] == (str(ROOT / "deep_ep"), "/fake/cuda", "/fake/nccl")

    elastic = importlib.import_module("deep_ep.buffers.elastic")
    elastic.check_nvlink_connections = lambda group: events.append("check_nvlink_connections")
    elastic.check_fast_rdma_atomic_support = lambda: events.append(
        "check_fast_rdma_atomic_support") or False
    elastic.get_rdma_gbs = lambda: 100
    elastic.get_nvlink_gbs = lambda: 100
    os.environ["EP_OVERRIDE_RDMA_SL"] = "7"
    group = _FakeGroup(events, rank=5, size=8)

    zero_buffer = deep_ep.ElasticBuffer(
        group, num_bytes=0, explicitly_destroy=True)
    assert zero_buffer.num_bytes == 0
    zero_runtime_args = extension.runtime_args[-1]
    assert zero_runtime_args[:6] == (5, 8, 4242, [], 0, 0)
    assert zero_runtime_args[9:11] == (7, 129)
    zero_buffer.destroy()
    events.clear()

    os.environ["NCCL_GIN_CROSS_NIC"] = "0"
    os.environ.pop("NCCL_SYM_REUSE_SYSMEM_HANDLES", None)
    os.environ.pop("NCCL_WIN_STRIDE", None)
    cpu_buffer = deep_ep.ElasticBuffer(
        group, num_bytes=8192, num_cpu_bytes=4096, explicitly_destroy=True)
    cpu_runtime_args = extension.runtime_args[-1]
    assert cpu_runtime_args[:3] == (5, 8, 1234)
    assert cpu_runtime_args[3] == [(101, 202)] * 8
    assert cpu_runtime_args[4:6] == (8192, 4096)
    create_comm_event = next(
        event for event in events if event[0] == "extension.create_nccl_comm")
    assert create_comm_event[:4] == (
        "extension.create_nccl_comm", "unused", 8, 5)
    assert create_comm_event[4] == "1"
    assert create_comm_event[5] is not None
    assert ("extension.create_cpu_handle", 4096) in events
    cpu_buffer.destroy()
    events.clear()

    buffer = deep_ep.ElasticBuffer(group, num_bytes=4096, explicitly_destroy=True)

    runtime_args = extension.runtime_args[-1]
    assert runtime_args[:6] == (5, 8, 4242, [], 4096, 0)
    assert runtime_args[9:11] == (7, 129)
    assert buffer.nccl_comm_handle is buffer.comm_handle
    assert buffer.num_allocated_qps == 129
    assert (buffer.num_scaleout_ranks, buffer.num_scaleup_ranks) == (2, 4)
    assert (buffer.scaleout_rank_idx, buffer.scaleup_rank_idx) == (1, 1)
    assert (buffer.num_rdma_ranks, buffer.num_nvlink_ranks) == (2, 4)
    assert events.count("check_nvlink_connections") == 1
    assert events.count("check_fast_rdma_atomic_support") == 1
    assert events.count("runtime.get_logical_domain_size") == 1
    assert events.count("runtime.get_physical_domain_size") == 1
    assert events.count("group.barrier") == 1
    sync_events = [event for event in events
                   if isinstance(event, tuple) and event[0] == "torch.cuda.synchronize"]
    assert len(sync_events) == 2

    agrs_start = len(events)
    with buffer.agrs_new_session(False):
        events.append("agrs.disabled.body")
    assert events[agrs_start:] == ["agrs.disabled.body"]

    agrs_start = len(events)
    with buffer.agrs_new_session():
        events.append("agrs.enabled.body")
    assert events[agrs_start:] == [
        "runtime.create_agrs_session", "agrs.enabled.body",
        "runtime.destroy_agrs_session"]

    properties_before = sum(
        isinstance(event, tuple) and event[0] == "torch.cuda.get_device_properties"
        for event in events)
    first_num_sms = buffer.get_theoretical_num_sms(
        8, 1, rdma_gbs=100, nvlink_gbs=100)
    second_num_sms = buffer.get_theoretical_num_sms(
        8, 1, rdma_gbs=100, nvlink_gbs=100)
    assert first_num_sms == second_num_sms
    properties_after = sum(
        isinstance(event, tuple) and event[0] == "torch.cuda.get_device_properties"
        for event in events)
    assert properties_after - properties_before == 1

    stream = buffer.get_comm_stream()
    assert (stream.stream_id, stream.device_index, stream.device_type) == (17, 2, 1)
    platform = importlib.import_module("deep_ep.platform")
    raw_event = platform.capture_event()
    assert platform.unwrap_event(deep_ep.EventOverlap(raw_event)) is raw_event
    assert platform.unwrap_event(raw_event) is raw_event

    runtime = extension.runtime_instances[-1]
    topk_idx = _FakeTensor("cuda", (1, 1))
    previous_event = extension.EventHandle()
    previous_event_before_epilogue = extension.EventHandle()
    _, _, _, handle, _ = buffer.dispatch(
        _FakeTensor("cuda", (1, 16)), topk_idx=topk_idx,
        num_experts=1, num_max_tokens_per_rank=1, num_sms=2, num_qps=1,
        previous_event=deep_ep.EventOverlap(previous_event),
        previous_event_before_epilogue=previous_event_before_epilogue)
    assert len(runtime.dispatch_calls[-1]) == 29
    assert runtime.dispatch_calls[-1][20] is previous_event
    assert runtime.dispatch_calls[-1][21] is previous_event_before_epilogue

    _, _, _, auto_handle, _ = buffer.dispatch(
        _FakeTensor("cuda", (1, 16)), topk_idx=_FakeTensor("cuda", (1, 1)),
        num_experts=8, num_max_tokens_per_rank=1)
    auto_args = runtime.dispatch_calls[-1]
    assert auto_args[18] > 1
    assert auto_args[19] > 0
    assert auto_args[19] <= buffer.num_allocated_qps
    assert auto_handle.num_sms == auto_args[18]

    combine_previous_event = extension.EventHandle()
    combine_previous_event_before_epilogue = extension.EventHandle()
    buffer.combine(
        _FakeTensor("cuda", (1, 16)), handle, num_sms=2, num_qps=1,
        previous_event=deep_ep.EventOverlap(combine_previous_event),
        previous_event_before_epilogue=combine_previous_event_before_epilogue)
    assert runtime.combine_calls[-1][13] is combine_previous_event
    assert runtime.combine_calls[-1][14] is combine_previous_event_before_epilogue

    size = deep_ep.ElasticBuffer.get_buffer_size_hint(group, 2, 32, 4)
    assert size == 8192
    assert extension.size_calls[-1] == (4242, 2, 32, 4, False, True, True)
    buffer.destroy()
    assert buffer.comm_handle is None
    assert buffer.nccl_comm_handle is None


def _scenario_stale_cuda_extension_guard():
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "_C_stale.py"
        path.write_text("def get_platform():\n    return 'cuda'\n")
        assert not _is_ascend_extension_available(path, forbid_package=True)
        assert "deep_ep" not in sys.modules
        assert "deep_ep._C" not in sys.modules


SCENARIOS = {
    "ascend_import": _scenario_ascend_import,
    "invalid_platform": _scenario_invalid_platform,
    "ascend_construction": _scenario_ascend_construction,
    "ascend_implicit_size": _scenario_ascend_implicit_size,
    "ascend_method_gates": _scenario_ascend_method_gates,
    "ascend_contextmanager_gate": _scenario_ascend_contextmanager_gate,
    "ascend_dispatch": _scenario_ascend_dispatch,
    "ascend_dispatch_optimized": _scenario_ascend_dispatch_optimized,
    "ascend_combine": _scenario_ascend_combine,
    "ascend_weak_lru_gate": _scenario_ascend_weak_lru_gate,
    "cuda_preservation": _scenario_cuda_preservation,
    "stale_cuda_extension_guard": _scenario_stale_cuda_extension_guard,
}


class PythonApiIsolationTest(unittest.TestCase):
    def run_scenario(self, scenario, optimize=False):
        command = [sys.executable]
        if optimize:
            command.append("-O")
        command.extend([str(pathlib.Path(__file__).resolve()), "--isolation", scenario])
        result = subprocess.run(
            command,
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(
            result.returncode, 0,
            f"scenario {scenario} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    def test_ascend_import_and_platform_adapter_avoid_cuda_dependencies(self):
        self.run_scenario("ascend_import")

    def test_unknown_compiled_platform_is_rejected(self):
        self.run_scenario("invalid_platform")

    def test_explicit_size_constructs_without_cuda_or_topology(self):
        self.run_scenario("ascend_construction")

    def test_implicit_size_reaches_backend_transport_error_without_nccl(self):
        self.run_scenario("ascend_implicit_size")

    def test_all_runtime_methods_fail_before_argument_or_symbol_access(self):
        self.run_scenario("ascend_method_gates")

    def test_contextmanager_method_gates_when_called(self):
        self.run_scenario("ascend_contextmanager_gate")

    def test_dispatch_routes_to_the_synchronous_ascend_runtime(self):
        self.run_scenario("ascend_dispatch")

    def test_dispatch_count_validation_survives_optimized_python(self):
        self.run_scenario("ascend_dispatch_optimized", optimize=True)

    def test_combine_routes_to_the_synchronous_ascend_runtime(self):
        self.run_scenario("ascend_combine")

    def test_cached_method_gates_before_hashing_arguments(self):
        self.run_scenario("ascend_weak_lru_gate")

    def test_cuda_initialization_and_constructor_behavior_are_preserved(self):
        self.run_scenario("cuda_preservation")

    def test_stale_cuda_extension_skips_before_package_initialization(self):
        self.run_scenario("stale_cuda_extension_guard")

    def test_testing_diagnostic_surface_probe_rejects_injection_attributes(self):
        class CleanBuffer:
            pass

        class CleanExtension:
            ElasticBuffer = CleanBuffer

        API_SURFACE.assert_no_testing_diagnostic_surface(CleanExtension)

        class ModuleDiagnosticExtension:
            ElasticBuffer = CleanBuffer
            diagnostic = None

        with self.assertRaisesRegex(AssertionError, "module.diagnostic"):
            API_SURFACE.assert_no_testing_diagnostic_surface(
                ModuleDiagnosticExtension)

        class DiagnosticBuffer:
            diagnostic = None

        class DiagnosticExtension:
            ElasticBuffer = DiagnosticBuffer

        with self.assertRaisesRegex(AssertionError, "ElasticBuffer.diagnostic"):
            API_SURFACE.assert_no_testing_diagnostic_surface(DiagnosticExtension)


class PythonGateSourceTest(unittest.TestCase):
    @staticmethod
    def first_executable_statement(function):
        statements = function.body
        if statements and isinstance(statements[0], ast.Expr) and isinstance(
                statements[0].value, ast.Constant) and isinstance(
                    statements[0].value.value, str):
            statements = statements[1:]
        return statements[0]

    def assert_first_require_cuda(self, function, operation):
        statement = self.first_executable_statement(function)
        self.assertIsInstance(statement, ast.Expr)
        self.assertIsInstance(statement.value, ast.Call)
        self.assertIsInstance(statement.value.func, ast.Name)
        self.assertEqual(statement.value.func.id, "require_cuda")
        self.assertEqual([ast.literal_eval(arg) for arg in statement.value.args],
                         [operation])

    def test_every_runtime_method_gates_as_its_first_executable_line(self):
        tree = ast.parse(ELASTIC_SOURCE.read_text())
        elastic = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "ElasticBuffer")
        methods = {node.name: node for node in elastic.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertEqual(GATED_METHODS.keys() - methods.keys(), set())
        for method, operation in GATED_METHODS.items():
            with self.subTest(method=method):
                self.assert_first_require_cuda(methods[method], operation)

    def test_module_topology_helpers_select_platform_before_communicator_access(self):
        tree = ast.parse(ENVS_SOURCE.read_text())
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        for name in ("get_physical_domain_size", "get_logical_domain_size"):
            with self.subTest(function=name):
                statement = self.first_executable_statement(functions[name])
                self.assertIsInstance(statement, ast.If)


class RealAscendPythonApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("real Ascend package tests require PyTorch")
        extension_path = _find_in_place_extension()
        if extension_path is None:
            raise unittest.SkipTest(
                "real Ascend package tests require an in-place built extension")
        if not _is_ascend_extension_available(extension_path):
            raise unittest.SkipTest(
                "real Ascend package tests require an Ascend extension")

        import torch
        import deep_ep
        import deep_ep._C as extension

        cls.torch = torch
        cls.deep_ep = deep_ep
        cls.extension = extension
        if extension.get_platform() != "ascend":
            raise AssertionError("low-level extension platform changed before package import")

    def setUp(self):
        self.assertEqual(self.extension.get_platform(), "ascend")

    class FakeGroup:
        def rank(self):
            return 0

        def size(self):
            return 1

        def barrier(self):
            raise AssertionError(
                "Ascend construction must not enter a CUDA-era barrier")

        def _get_backend(self, device):
            return types.SimpleNamespace(get_hccl_comm=lambda local_rank: 0x1234)

    def assert_transport_error(self, operation, call):
        with self.assertRaises(NotImplementedError) as context:
            call()
        self.assertIs(type(context.exception), NotImplementedError)
        self.assertEqual(str(context.exception), TRANSPORT_ERROR.format(operation))

    def test_import_skips_cuda_exports(self):
        self.assertTrue(hasattr(self.deep_ep, "ElasticBuffer"))
        self.assertFalse(hasattr(self.deep_ep, "Buffer"))
        self.assertFalse(hasattr(self.extension, "init_jit"))
        API_SURFACE.assert_no_testing_diagnostic_surface(self.extension)

    def test_constructor_rejects_non_two_rank_group_before_hccl_use(self):
        with self.assertRaisesRegex(
                RuntimeError, "exactly two ranks are required"):
            self.deep_ep.ElasticBuffer(
                self.FakeGroup(), num_bytes=2 * 1024 * 1024,
                allow_hybrid_mode=False, explicitly_destroy=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--isolation":
        SCENARIOS[sys.argv[2]]()
    elif (len(sys.argv) in (3, 4) and
          sys.argv[1] == "--probe-low-level-extension"):
        finder = None
        if len(sys.argv) == 4:
            if sys.argv[3] != "--forbid-package":
                raise RuntimeError(f"unknown probe argument: {sys.argv[3]}")
            finder = _ForbiddenImportFinder()
            finder.FORBIDDEN = {"deep_ep"}
            sys.meta_path.insert(0, finder)
        try:
            print(_load_low_level_extension(pathlib.Path(sys.argv[2])).get_platform())
        finally:
            if finder is not None:
                sys.meta_path.remove(finder)
    else:
        unittest.main()
