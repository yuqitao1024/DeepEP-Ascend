import ast
import importlib.abc
import importlib.util
import json
import os
import pathlib
import copy
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ELASTIC_SOURCE = ROOT / "deep_ep/buffers/elastic.py"
ENVS_SOURCE = ROOT / "deep_ep/utils/envs.py"
API_SURFACE_PROBE = ROOT / "tests/ascend/production/api_surface.py"
FP8_RUNTIME_MATRIX = \
    ROOT / "tests/ascend/production/run_fp8_dispatch_combine.py"

API_SURFACE_SPEC = importlib.util.spec_from_file_location(
    "ascend_api_surface", API_SURFACE_PROBE)
API_SURFACE = importlib.util.module_from_spec(API_SURFACE_SPEC)
API_SURFACE_SPEC.loader.exec_module(API_SURFACE)

TRANSPORT_ERROR = (
    "DeepEP Ascend backend: {} is unavailable until the Ascend device "
    "transport is implemented")
GATED_METHODS = {
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
    def __init__(self, device_type="cpu", shape=(), dtype=None,
                 contiguous=True, values=None, strides=None, device_index=0):
        self.device = types.SimpleNamespace(type=device_type, index=device_index)
        self.shape = shape
        self.dtype = dtype
        self._contiguous = contiguous
        if strides is None:
            running = 1
            computed = []
            for extent in reversed(shape):
                computed.append(running)
                running *= extent
            strides = tuple(reversed(computed))
        self._strides = tuple(strides)
        numel = 1
        for extent in shape:
            numel *= extent
        self._values = list(values) if values is not None else [0] * numel
        self.cpu_calls = 0

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def stride(self, dimension=None):
        return self._strides if dimension is None else self._strides[dimension]

    def detach(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self

    def reshape(self, *shape):
        return self

    def tolist(self):
        return list(self._values)


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
    def __init__(self, events, rank=0, size=1, comm_pointer=4242,
                 gathered_objects=None):
        self.events = events
        self._rank = rank
        self._size = size
        self.comm_pointer = comm_pointer
        self.gathered_objects = gathered_objects

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


def _fixed_preflight_record(stage, contract=None, error_code=None):
    return (
        stage, int(error_code is None), error_code or "",
        "" if error_code is not None else json.dumps(
            contract, separators=(",", ":"), sort_keys=True))


def _fixed_preflight_contract(record):
    return json.loads(record[3])


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
        def __init__(self):
            self.device_index = 0
            self.capturing = False

        def current_device(self):
            if platform != "ascend":
                raise AssertionError("CUDA import touched torch.npu.current_device")
            events.append("torch.npu.current_device")
            return self.device_index

        def Stream(self, **kwargs):
            if platform != "ascend":
                raise AssertionError("CUDA import touched torch.npu.Stream")
            events.append(("torch.npu.Stream", (), kwargs))
            return _FakeStream(**kwargs)

        def synchronize(self):
            if platform != "ascend":
                raise AssertionError("CUDA import touched torch.npu.synchronize")
            events.append("torch.npu.synchronize")

        def is_current_stream_capturing(self):
            if platform != "ascend":
                raise AssertionError(
                    "CUDA import touched torch.npu.is_current_stream_capturing")
            events.append("torch.npu.is_current_stream_capturing")
            return self.capturing

    torch.npu = FakeNpu()

    distributed = types.ModuleType("torch.distributed")
    distributed.ProcessGroup = _FakeGroup
    distributed.get_rank = lambda: 0
    distributed.get_world_size = lambda: 1
    distributed.new_group = lambda ranks: None
    distributed.init_process_group = lambda **kwargs: None
    distributed.barrier = lambda: events.append("dist.barrier")
    def all_gather_object(output, value, group):
        events.append(("dist.all_gather_object", value))
        gathered = group.gathered_objects
        output[:] = (gathered(value) if callable(gathered) else gathered
                     if gathered is not None else [value] * len(output))

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
        def __init__(self, state=None):
            if state is None:
                state = types.SimpleNamespace(
                    device_index=(sys.modules["torch"].npu.current_device()
                                  if platform == "ascend" else 0),
                    wait_count=0, owners=0, completed=False,
                    completion_error=None, on_finish=None)
                events.append(("event.capture", state.device_index))
            self._state = state
            self._state.owners += 1

        def __copy__(self):
            return EventHandle(self._state)

        def _finish(self):
            if self._state.completion_error is not None:
                raise RuntimeError(self._state.completion_error)
            if not self._state.completed:
                if self._state.on_finish is not None:
                    self._state.on_finish()
                self._state.completed = True

        def current_stream_wait(self):
            if (platform == "ascend" and
                    sys.modules["torch"].npu.current_device() !=
                    self._state.device_index):
                raise RuntimeError(
                    "DeepEP Ascend backend: current_stream_wait current NPU "
                    "device does not belong to the event device")
            self._finish()
            self._state.wait_count += 1
            events.append(("event.current_stream_wait", self._state.wait_count))

        def __del__(self):
            try:
                self._state.owners -= 1
                if self._state.owners == 0:
                    self._finish()
            except Exception:
                pass

    class ElasticRuntime:
        def __init__(self, *args):
            extension.runtime_args.append(args)
            extension.runtime_instances.append(self)
            self.world_size = args[1]
            self.dispatch_calls = []
            self.combine_calls = []
            self.combine_events = []
            self.destroyed = False
            self.destroy_error = None
            self.destroy_error_destroys_runtime = False
            self.next_dispatch_generation = 0
            self.last_dispatch_generation = 0
            self.committed_dispatch_tensor = None
            self.committed_dispatch_fingerprint = None
            self.dispatch_generation_queries = 0
            self.dispatch_identity_prefix = None
            self.fail_next_dispatch_completion = False
            events.append(("runtime.construct", args))

        def destroy(self):
            events.append("runtime.destroy")
            if self.destroy_error is not None:
                if self.destroy_error_destroys_runtime:
                    self.destroyed = True
                raise RuntimeError(self.destroy_error)
            self.destroyed = True

        def is_destroyed(self):
            events.append("runtime.is_destroyed")
            return self.destroyed

        def get_comm_stream(self):
            events.append("runtime.get_comm_stream")
            return _FakeStream(17, 2, 1)

        def get_dispatch_handle_generation(self, tensor):
            self.dispatch_generation_queries += 1
            observed = tuple(tensor._values)
            expected = self.committed_dispatch_fingerprint
            if self.dispatch_identity_prefix is not None:
                observed = observed[:self.dispatch_identity_prefix]
                expected = expected[:self.dispatch_identity_prefix]
            if (tensor is not self.committed_dispatch_tensor or
                    observed != expected):
                return 0
            return self.last_dispatch_generation

        def get_logical_domain_size(self):
            events.append("runtime.get_logical_domain_size")
            if (platform == "ascend" and
                    os.environ.get("DEEP_EP_ASCEND_LOGICAL_SIMULATION") == "1"):
                return (2, 2)
            return (1, 2) if platform == "ascend" else (2, 4)

        def get_physical_domain_size(self):
            events.append("runtime.get_physical_domain_size")
            return (1, self.world_size) if platform == "ascend" else (2, 4)

        def barrier(self, use_comm_stream, with_cpu_sync, sequential):
            events.append(("runtime.barrier", use_comm_stream, with_cpu_sync, sequential))

        def create_agrs_session(self):
            events.append("runtime.create_agrs_session")

        def destroy_agrs_session(self):
            events.append("runtime.destroy_agrs_session")

        def dispatch(self, *args):
            self.dispatch_calls.append(args)
            device = args[0].device
            recv_src_metadata = _FakeTensor(
                device.type, (1,), device_index=device.index)
            cloned_topk_idx = _FakeTensor(
                args[2].device.type, args[2].shape, args[2].dtype,
                device_index=args[2].device.index)
            token_metadata_at_forward = (
                args[12] if platform == "ascend" and args[12] is not None else
                _FakeTensor(device.type, (120,), device_index=device.index)
                if platform == "ascend" else None)
            event = EventHandle() if args[22] or platform != "ascend" else None
            if platform == "ascend":
                self.next_dispatch_generation += 1
                generation = self.next_dispatch_generation

                def commit_dispatch():
                    token_metadata_at_forward._values[0] = generation
                    self.last_dispatch_generation = generation
                    self.committed_dispatch_tensor = token_metadata_at_forward
                    self.committed_dispatch_fingerprint = tuple(
                        token_metadata_at_forward._values)

                if args[22]:
                    if self.fail_next_dispatch_completion:
                        event._state.completion_error = (
                            "DeepEP Ascend backend: dispatch completion failed")
                        self.fail_next_dispatch_completion = False
                    else:
                        event._state.on_finish = commit_dispatch
                else:
                    commit_dispatch()
            return (args[0], args[1], None if args[26] else args[2], args[3],
                    cloned_topk_idx,
                    1, 1, [], _FakeTensor(device.type, device_index=device.index),
                    _FakeTensor(device.type, device_index=device.index),
                    _FakeTensor(device.type, device_index=device.index),
                    recv_src_metadata,
                    _FakeTensor(device.type, device_index=device.index),
                    token_metadata_at_forward,
                    None,
                    event)

        def combine(self, *args):
            self.combine_calls.append(args)
            event = EventHandle() if args[15] or platform != "ascend" else None
            self.combine_events.append(event)
            return args[0], args[1], event

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
    stream = platform.wrap_stream(_FakeStream(17, 2, 1))
    assert (stream.stream_id, stream.device_index, stream.device_type) == (17, 2, 1)
    assert ("torch.npu.Stream", (), {
        "stream_id": 17, "device_index": 2, "device_type": 1}) in events

    event = platform.capture_event()
    wrapped = types.SimpleNamespace(event=event)
    assert platform.unwrap_event(None) is None
    assert platform.unwrap_event(event) is event
    assert platform.unwrap_event(wrapped) is event
    assert deep_ep.ElasticBuffer.capture().__class__ is extension.EventHandle
    event.current_stream_wait()
    copied_event = copy.copy(event)
    copied_event.current_stream_wait()
    assert event._state is copied_event._state
    assert event._state.wait_count == 2

    overlap = deep_ep.EventOverlap(platform.capture_event())
    overlap.current_stream_wait(release_handle=True)
    assert overlap.event is None
    with deep_ep.EventOverlap(platform.capture_event()) as context_event:
        assert context_event.event is not None
    assert context_event.event is not None

    torch = sys.modules["torch"]
    wrong_device_event = platform.capture_event()
    torch.npu.device_index = 3
    try:
        wrong_device_event.current_stream_wait()
    except RuntimeError as error:
        assert str(error) == (
            "DeepEP Ascend backend: current_stream_wait current NPU device "
            "does not belong to the event device")
    else:
        raise AssertionError("cross-device Ascend event wait was accepted")
    torch.npu.device_index = 0

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
        except RuntimeError as error:
            assert "construction preflight failed" in str(error)
            assert "invalid_num_bytes" in str(error)
        else:
            raise AssertionError(f"non-positive buffer size {num_bytes} was accepted")


def _scenario_ascend_topology_preflight():
    deep_ep, extension, events = _load_package("ascend", True)
    os.environ["DEEP_EP_ASCEND_SCALE_UP_SIZE"] = "2"
    os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "1"
    os.environ["DEEP_EP_ASCEND_TOPOLOGY_EPOCH"] = "17"
    group = _FakeGroup(events, rank=3, size=4)
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True)
    assert ("dist.all_gather_object", _fixed_preflight_record(
        "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                     "topology_epoch": 17, "world_size": 4})) in events
    assert (buffer.num_scaleout_ranks, buffer.num_scaleup_ranks) == (2, 2)
    assert (buffer.scaleout_rank_idx, buffer.scaleup_rank_idx) == (1, 1)
    assert (buffer.num_rdma_ranks, buffer.num_nvlink_ranks) == (1, 4)
    assert len(extension.runtime_args) == 1
    buffer.destroy()


def _scenario_ascend_destroy_state_publication():
    deep_ep, extension, events = _load_package("ascend", True)
    group = _FakeGroup(events, rank=0, size=2)

    destroyed_buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True)
    destroyed_runtime = destroyed_buffer.runtime
    destroyed_runtime.destroy_error = "terminal operation failure"
    destroyed_runtime.destroy_error_destroys_runtime = True
    try:
        destroyed_buffer.destroy()
    except RuntimeError as error:
        assert "terminal operation failure" in str(error), error
    else:
        raise AssertionError("terminal destroy failure was not propagated")
    assert destroyed_buffer.runtime is None
    assert destroyed_buffer.comm_handle is None
    destroy_calls = events.count("runtime.destroy")
    destroyed_buffer.destroy()
    assert events.count("runtime.destroy") == destroy_calls

    retryable_buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True)
    retryable_runtime = retryable_buffer.runtime
    retryable_runtime.destroy_error = "retryable completion timeout"
    try:
        retryable_buffer.destroy()
    except RuntimeError as error:
        assert "retryable completion timeout" in str(error), error
    else:
        raise AssertionError("retryable destroy failure was not propagated")
    assert retryable_buffer.runtime is retryable_runtime
    assert retryable_buffer.comm_handle == 4242
    assert not retryable_runtime.is_destroyed()

    retryable_runtime.destroy_error = None
    retryable_buffer.destroy()
    assert retryable_buffer.runtime is None
    assert retryable_buffer.comm_handle is None


def _scenario_ascend_topology_preflight_mismatch():
    deep_ep, extension, events = _load_package("ascend", True)
    os.environ["DEEP_EP_ASCEND_SCALE_UP_SIZE"] = "2"
    os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "1"
    group = _FakeGroup(
        events, rank=0, size=4,
        gathered_objects=[
            _fixed_preflight_record(
                "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                             "topology_epoch": 1, "world_size": 4}),
            _fixed_preflight_record(
                "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                             "topology_epoch": 1, "world_size": 4}),
            _fixed_preflight_record(
                "topology", {"kind": "logical_simulation", "scale_up_size": 4,
                             "topology_epoch": 1, "world_size": 4}),
            _fixed_preflight_record(
                "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                             "topology_epoch": 1, "world_size": 4}),
        ])
    try:
        deep_ep.ElasticBuffer(
            group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
            explicitly_destroy=True)
    except RuntimeError as error:
        assert "topology preflight failed on rank 2 (topology_shape_mismatch)" \
               in str(error), error
    else:
        raise AssertionError("asymmetric Ascend topology was accepted")
    assert not extension.runtime_args


def _scenario_ascend_topology_preflight_local_parse_failure():
    deep_ep, extension, events = _load_package("ascend", True)
    os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "1"
    os.environ.pop("DEEP_EP_ASCEND_SCALE_UP_SIZE", None)
    gathered = [
        _fixed_preflight_record(
            "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                         "topology_epoch": 1, "world_size": 4}),
        _fixed_preflight_record("topology", error_code="missing_scale_up_size"),
        _fixed_preflight_record(
            "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                         "topology_epoch": 1, "world_size": 4}),
        _fixed_preflight_record(
            "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                         "topology_epoch": 1, "world_size": 4}),
    ]
    group = _FakeGroup(
        events, rank=1, size=4, gathered_objects=gathered)
    try:
        deep_ep.ElasticBuffer(
            group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
            explicitly_destroy=True)
    except RuntimeError as error:
        assert str(error) == (
            "DeepEP Ascend backend: topology preflight failed on rank 1 "
            "(missing_scale_up_size)"), error
    else:
        raise AssertionError("invalid local Ascend topology was accepted")
    assert ("dist.all_gather_object", gathered[1]) in events
    assert not extension.runtime_args


def _scenario_ascend_topology_preflight_remote_parse_failure():
    deep_ep, extension, events = _load_package("ascend", True)
    os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "1"
    os.environ["DEEP_EP_ASCEND_SCALE_UP_SIZE"] = "2"
    gathered = [
        _fixed_preflight_record(
            "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                         "topology_epoch": 1, "world_size": 4}),
        _fixed_preflight_record("topology", error_code="invalid_topology_epoch"),
        _fixed_preflight_record(
            "topology", {"kind": "logical_simulation", "scale_up_size": 2,
                         "topology_epoch": 1, "world_size": 4}),
        _fixed_preflight_record("topology", error_code="invalid_scale_up_size"),
    ]
    errors = []
    for rank in (0, 3):
        group = _FakeGroup(
            events, rank=rank, size=4, gathered_objects=gathered)
        try:
            deep_ep.ElasticBuffer(
                group, num_bytes=2 * 1024 * 1024,
                allow_hybrid_mode=False, explicitly_destroy=True)
        except RuntimeError as error:
            errors.append(str(error))
        else:
            raise AssertionError("remote Ascend topology failure was ignored")
    assert errors == [
        "DeepEP Ascend backend: topology preflight failed on rank 1 "
        "(invalid_topology_epoch)",
    ] * 2
    assert sum(event[0] == "dist.all_gather_object"
               for event in events if isinstance(event, tuple)) == 2
    assert not extension.runtime_args


def _scenario_ascend_collective_contract_preflight():
    deep_ep, extension, events = _load_package("ascend", True)

    def valid_record(stage, contract):
        return _fixed_preflight_record(stage, contract)

    def gather_local_communicator_failure(value):
        if value[0] == "topology":
            return [value, value]
        assert value[0] == "construction_communicator", value
        assert value[1] == 0, value
        assert value[2] == "invalid_communicator_handle", value
        return [value, valid_record(
            "construction_communicator", {"communicator_valid": True})]

    local_invalid_group = _FakeGroup(
        events, rank=0, size=2, comm_pointer=0,
        gathered_objects=gather_local_communicator_failure)
    try:
        deep_ep.ElasticBuffer(
            local_invalid_group, num_bytes=2 * 1024 * 1024,
            allow_hybrid_mode=False, explicitly_destroy=True)
    except RuntimeError as error:
        assert "construction_communicator preflight failed on rank 0 " \
               "(invalid_communicator_handle)" in str(error), error
    else:
        raise AssertionError("local null communicator bypassed collective preflight")
    assert not extension.runtime_args
    assert not extension.size_calls

    def gather_remote_communicator_failure(value):
        if value[0] == "topology":
            return [value, value]
        assert value[0] == "construction_communicator", value
        assert _fixed_preflight_contract(value) == {"communicator_valid": True}, value
        remote = _fixed_preflight_record(
            "construction_communicator", error_code="invalid_communicator_handle")
        return [value, remote]

    remote_invalid_group = _FakeGroup(
        events, rank=0, size=2,
        gathered_objects=gather_remote_communicator_failure)
    try:
        deep_ep.ElasticBuffer(
            remote_invalid_group, num_bytes=2 * 1024 * 1024,
            allow_hybrid_mode=False, explicitly_destroy=True)
    except RuntimeError as error:
        assert "construction_communicator preflight failed on rank 1 " \
               "(invalid_communicator_handle)" in str(error), error
    else:
        raise AssertionError("remote null communicator bypassed collective preflight")
    assert not extension.runtime_args
    assert not extension.size_calls

    def mismatch_construction(value):
        if value[0] == "topology":
            return [value, value]
        if value[0] != "construction":
            return [value, value]
        remote_contract = _fixed_preflight_contract(value)
        remote_contract["num_bytes"] = 4 * 1024 * 1024
        remote = valid_record("construction", remote_contract)
        return [value, remote]

    mismatched_group = _FakeGroup(
        events, rank=0, size=2, gathered_objects=mismatch_construction)
    try:
        deep_ep.ElasticBuffer(
            mismatched_group, num_bytes=2 * 1024 * 1024,
            allow_hybrid_mode=False, explicitly_destroy=True)
    except RuntimeError as error:
        assert "construction preflight failed on rank 1 (num_bytes_mismatch)" \
               in str(error), error
    else:
        raise AssertionError("mismatched construction layout was accepted")
    assert not extension.runtime_args

    group = _FakeGroup(events, rank=0, size=2)
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (1, 16), torch.bfloat16)
    topk_idx = _FakeTensor("npu", (1, 1), torch.int64)

    def remote_failure(error_code):
        def gather(value):
            remote = _fixed_preflight_record(value[0], error_code=error_code)
            return [value, remote]
        return gather

    group.gathered_objects = remote_failure("invalid_x_dtype")
    try:
        buffer.dispatch(
            x, topk_idx=topk_idx, num_experts=2,
            num_max_tokens_per_rank=1)
    except RuntimeError as error:
        assert "dispatch preflight failed on rank 1 (invalid_x_dtype)" in str(error), error
    else:
        raise AssertionError("peer dispatch validation failure was ignored")
    assert not runtime.dispatch_calls

    def local_dispatch_failure(expected_error):
        def gather(value):
            assert value[0] == "dispatch", value
            assert value[1] == 0, value
            assert value[2] == expected_error, value
            return [value, valid_record("dispatch", {})]
        return gather

    group.gathered_objects = local_dispatch_failure(
        "invalid_dispatch_shape")
    try:
        buffer.dispatch(
            _FakeTensor("npu", (2, 16), torch.bfloat16),
            topk_idx=_FakeTensor("npu", (2, 1), torch.int64),
            num_experts=2, num_max_tokens_per_rank=1)
    except RuntimeError as error:
        assert "dispatch preflight failed on rank 0 " \
               "(invalid_dispatch_shape)" in str(error), error
    else:
        raise AssertionError("token count above capacity reached dispatch")
    assert not runtime.dispatch_calls

    group.gathered_objects = local_dispatch_failure(
        "invalid_dispatch_shape")
    try:
        buffer.dispatch(
            x, topk_idx=_FakeTensor("npu", (1, 3), torch.int64),
            num_experts=2, num_max_tokens_per_rank=1)
    except RuntimeError as error:
        assert "dispatch preflight failed on rank 0 " \
               "(invalid_dispatch_shape)" in str(error), error
    else:
        raise AssertionError("top-k width above expert count reached dispatch")
    assert not runtime.dispatch_calls

    def gather_valid_dispatch(value):
        assert value[0] == "dispatch", value
        assert value[1] == 1, value
        contract = _fixed_preflight_contract(value)
        assert contract["x_shape"] == [None, 16], value
        assert contract["topk_shape"] == [None, 1], value
        assert "token_count" not in contract, value
        return [value, value]

    group.gathered_objects = gather_valid_dispatch
    _, _, _, handle, _ = buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=2,
        num_max_tokens_per_rank=1)
    group.gathered_objects = None

    descriptor = handle.token_metadata_at_forward
    descriptor._values[0] ^= 1
    dispatch_calls = len(runtime.dispatch_calls)
    try:
        buffer.dispatch(x, handle=handle)
    except RuntimeError as error:
        assert "dispatch preflight failed on rank 0 (invalid_dispatch_handle)" in str(error), error
    else:
        raise AssertionError("in-place dispatch descriptor corruption was accepted")
    assert len(runtime.dispatch_calls) == dispatch_calls
    combine_calls = len(runtime.combine_calls)
    try:
        buffer.combine(x, handle)
    except RuntimeError as error:
        assert "combine preflight failed on rank 0 (invalid_dispatch_handle)" in str(error), error
    else:
        raise AssertionError("in-place combine descriptor corruption was accepted")
    assert len(runtime.combine_calls) == combine_calls
    descriptor._values[0] ^= 1

    group.gathered_objects = remote_failure("invalid_dispatch_handle")
    combine_calls = len(runtime.combine_calls)
    try:
        buffer.combine(x, handle)
    except RuntimeError as error:
        assert "combine preflight failed on rank 1 (invalid_dispatch_handle)" in str(error), error
    else:
        raise AssertionError("peer combine handle failure was ignored")
    assert len(runtime.combine_calls) == combine_calls
    buffer.destroy()


def _scenario_ascend_hybrid_collective_preflight():
    deep_ep, extension, events = _load_package("ascend", True)
    os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "1"
    os.environ["DEEP_EP_ASCEND_SCALE_UP_SIZE"] = "2"
    os.environ["DEEP_EP_ASCEND_TOPOLOGY_EPOCH"] = "9"

    def mutate_record(value, field, replacement):
        assert isinstance(value, tuple), value
        assert len(value) == 4, value
        assert all(isinstance(item, (str, int)) for item in value), value
        stage, ok, error_code, encoded_contract = value
        assert ok == 1 and error_code == "", value
        contract = json.loads(encoded_contract)
        assert field in contract, (field, contract)
        contract[field] = replacement
        return stage, ok, error_code, json.dumps(
            contract, separators=(",", ":"), sort_keys=True)

    def construction_gather(field, replacement):
        def gather(value):
            if value[0] == "topology":
                return [value] * 4
            if value[0] == "construction_communicator":
                return [value] * 4
            assert value[0] == "construction", value
            remote = mutate_record(value, field, replacement)
            return [value, remote, value, value]
        return gather

    construction_cases = (
        ("hybrid_mode", 0, "hybrid_mode_mismatch"),
        ("symmetric_bytes", 4 * 1024 * 1024, "symmetric_bytes_mismatch"),
        ("route_capacity", 17, "route_capacity_mismatch"),
        ("scale_up_team_available", 0, "scale_up_team_available_mismatch"),
        ("scale_out_team_available", 0, "scale_out_team_available_mismatch"),
        ("transport_capabilities", 0, "transport_capabilities_mismatch"),
        ("num_cpu_bytes", 2 * 1024 * 1024, "num_cpu_bytes_mismatch"),
    )
    for field, replacement, error_code in construction_cases:
        errors = []
        for rank in (0, 3):
            group = _FakeGroup(
                events, rank=rank, size=4,
                gathered_objects=construction_gather(field, replacement))
            try:
                deep_ep.ElasticBuffer(
                    group, num_bytes=2 * 1024 * 1024,
                    allow_hybrid_mode=True, explicitly_destroy=True)
            except RuntimeError as error:
                errors.append(str(error))
            else:
                raise AssertionError(f"asymmetric {field} was accepted")
        assert errors == [
            f"DeepEP Ascend backend: construction preflight failed on rank 1 "
            f"({error_code})",
        ] * 2, errors
        assert not extension.runtime_args

    def topology_gather(field, replacement):
        def gather(value):
            assert isinstance(value, tuple), value
            assert value[0] == "topology", value
            contract = json.loads(value[3])
            contract[field] = replacement
            remote = _fixed_preflight_record("topology", contract)
            return [value, remote, value, value]
        return gather

    topology_cases = (
        ("topology_epoch", 10, "topology_epoch_mismatch"),
        ("scale_up_size", 4, "topology_shape_mismatch"),
    )
    for field, replacement, error_code in topology_cases:
        topology_errors = []
        for rank in (0, 3):
            group = _FakeGroup(events, rank=rank, size=4,
                               gathered_objects=topology_gather(
                                   field, replacement))
            try:
                deep_ep.ElasticBuffer(
                    group, num_bytes=2 * 1024 * 1024,
                    allow_hybrid_mode=True, explicitly_destroy=True)
            except RuntimeError as error:
                topology_errors.append(str(error))
            else:
                raise AssertionError(f"asymmetric {field} was accepted")
        assert topology_errors == [
            "DeepEP Ascend backend: topology preflight failed on rank 1 "
            f"({error_code})",
        ] * 2, topology_errors
        assert not extension.runtime_args

    def valid_gather(value):
        assert isinstance(value, tuple), value
        assert len(value) == 4, value
        return [value] * 4

    cpu_errors = []
    for rank in (0, 3):
        group = _FakeGroup(events, rank=rank, size=4,
                           gathered_objects=valid_gather)
        try:
            deep_ep.ElasticBuffer(
                group, num_bytes=4 * 1024 * 1024,
                num_cpu_bytes=2 * 1024 * 1024, allow_hybrid_mode=True,
                explicitly_destroy=True)
        except RuntimeError as error:
            cpu_errors.append(str(error))
        else:
            raise AssertionError("mapped CPU request was accepted without capability")
    assert cpu_errors == [
        "DeepEP Ascend backend: construction preflight failed on rank 0 "
        "(cpu_buffer_unsupported)",
    ] * 2, cpu_errors
    assert not extension.runtime_args

    def rank_invariant_gather(world_size, hybrid, normalized):
        def gather(value):
            if value[0] in ("dispatch", "combine"):
                contract = _fixed_preflight_contract(value)
                assert "cached_handle_descriptor" not in contract, contract
                assert "dispatch_handle_descriptor" not in contract, contract
                field = ("cached_descriptor_contract" if
                         value[0] == "dispatch" else
                         "combine_descriptor_contract")
                descriptor_contract = contract[field]
                assert descriptor_contract == {
                    "layout": "regular",
                    "operation": value[0],
                    "routing_mode": "hybrid" if hybrid else "direct",
                    "schema_version": 1,
                    "valid": 1,
                }, descriptor_contract
                normalized[value[0]].append(json.dumps(
                    descriptor_contract, separators=(",", ":"),
                    sort_keys=True))
            return [value] * world_size
        return gather

    def run_distinct_rank_handles(world_size, hybrid):
        saved_simulation = os.environ.get("DEEP_EP_ASCEND_LOGICAL_SIMULATION")
        saved_scale_up = os.environ.get("DEEP_EP_ASCEND_SCALE_UP_SIZE")
        if hybrid:
            os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "1"
            os.environ["DEEP_EP_ASCEND_SCALE_UP_SIZE"] = "2"
        else:
            os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = "0"
            os.environ.pop("DEEP_EP_ASCEND_SCALE_UP_SIZE", None)
        try:
            buffers = []
            handles = []
            for rank in range(world_size):
                group = _FakeGroup(events, rank=rank, size=world_size,
                                   gathered_objects=valid_gather)
                buffer = deep_ep.ElasticBuffer(
                    group, num_bytes=2 * 1024 * 1024,
                    allow_hybrid_mode=hybrid, explicitly_destroy=True)
                torch = sys.modules["torch"]
                x = _FakeTensor("npu", (1, 16), torch.bfloat16)
                topk_idx = _FakeTensor("npu", (1, 1), torch.int64)
                _, _, _, handle, _ = buffer.dispatch(
                    x, topk_idx=topk_idx, num_experts=world_size,
                    num_max_tokens_per_rank=1)
                handle.token_metadata_at_forward._values[0] = rank + 1
                handle._ascend_descriptor_fingerprint = tuple(
                    handle.token_metadata_at_forward._values)
                buffers.append((buffer, group, x))
                handles.append(handle)

            assert len({tuple(handle.token_metadata_at_forward._values)
                        for handle in handles}) == world_size
            normalized = {"dispatch": [], "combine": []}
            for (buffer, group, x), handle in zip(buffers, handles):
                runtime = buffer.runtime
                dispatch_calls = len(runtime.dispatch_calls)
                combine_calls = len(runtime.combine_calls)
                group.gathered_objects = rank_invariant_gather(
                    world_size, hybrid, normalized)
                buffer.dispatch(x, handle=handle)
                buffer.combine(x, handle)
                assert len(runtime.dispatch_calls) == dispatch_calls + 1
                assert len(runtime.combine_calls) == combine_calls + 1
            assert len(set(normalized["dispatch"])) == 1, normalized
            assert len(set(normalized["combine"])) == 1, normalized
            return buffers, handles
        finally:
            if saved_simulation is None:
                os.environ.pop("DEEP_EP_ASCEND_LOGICAL_SIMULATION", None)
            else:
                os.environ["DEEP_EP_ASCEND_LOGICAL_SIMULATION"] = saved_simulation
            if saved_scale_up is None:
                os.environ.pop("DEEP_EP_ASCEND_SCALE_UP_SIZE", None)
            else:
                os.environ["DEEP_EP_ASCEND_SCALE_UP_SIZE"] = saved_scale_up

    direct_buffers, direct_handles = run_distinct_rank_handles(2, False)
    hybrid_buffers, _ = run_distinct_rank_handles(4, True)

    def invalid_handle_gather(rank):
        def gather(value):
            if rank == 1:
                assert value[1:3] == (0, "invalid_dispatch_handle"), value
                assert _fixed_preflight_contract(value)[
                    "cached_descriptor_contract"]["valid"] == 0, value
            else:
                assert value[1] == 1, value
                assert _fixed_preflight_contract(value)[
                    "cached_descriptor_contract"]["valid"] == 1, value
            return [
                _fixed_preflight_record("dispatch", {}),
                _fixed_preflight_record(
                    "dispatch", error_code="invalid_dispatch_handle"),
            ]
        return gather

    direct_handles[1].token_metadata_at_forward._values[1] = 99
    invalid_errors = []
    for buffer, group, x in direct_buffers:
        runtime = buffer.runtime
        dispatch_calls = len(runtime.dispatch_calls)
        generation = buffer._ascend_handle_generation
        group.gathered_objects = invalid_handle_gather(buffer.rank_idx)
        try:
            buffer.dispatch(x, handle=direct_handles[buffer.rank_idx])
        except RuntimeError as error:
            invalid_errors.append(str(error))
        else:
            raise AssertionError("invalid local descriptor was accepted")
        assert len(runtime.dispatch_calls) == dispatch_calls
        assert buffer._ascend_handle_generation == generation
    assert invalid_errors == [
        "DeepEP Ascend backend: dispatch preflight failed on rank 1 "
        "(invalid_dispatch_handle)",
    ] * 2, invalid_errors

    for buffer, _, _ in direct_buffers + hybrid_buffers:
        buffer.destroy()


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

    extension.size_failure = None
    assert deep_ep.ElasticBuffer.get_buffer_size_hint(
        group, 2, 32, 4, use_fp8_dispatch=True,
        allow_hybrid_mode=False) == 2 * 1024 * 1024
    assert extension.size_calls[-1] == (4242, 2, 32, 4, True, False, True)


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
    stream = buffer.get_comm_stream()
    assert (stream.stream_id, stream.device_index, stream.device_type) == (17, 2, 1)
    assert ("torch.npu.Stream", (), {
        "stream_id": 17, "device_index": 2, "device_type": 1}) in events
    calls = {
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
    group = _FakeGroup(events, rank=2, size=3)
    assert envs.get_physical_domain_size(group) == (1, 3)
    assert envs.get_logical_domain_size(group, False) == (1, 3)
    undersized = _FakeGroup(events, rank=0, size=1)
    for query in (
            lambda: envs.get_physical_domain_size(undersized),
            lambda: envs.get_logical_domain_size(undersized, False)):
        try:
            query()
        except RuntimeError as error:
            assert "at least two ranks" in str(error)
        else:
            raise AssertionError("single-rank Ascend domain was accepted")
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
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (1, 16), torch.bfloat16)
    topk_idx = _FakeTensor("npu", (1, 1), torch.int64)
    topk_weights = _FakeTensor("npu", (1, 1), torch.float32)

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

    cached_topk_weights = _FakeTensor("npu", (1, 1), torch.float32)
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
    cached_preflight = next(
        event[1] for event in reversed(events)
        if isinstance(event, tuple) and len(event) == 2 and
        event[0] == "dist.all_gather_object" and event[1][0] == "dispatch")
    assert _fixed_preflight_contract(cached_preflight)["cpu_sync"] == 0

    _, _, _, async_handle, async_event = buffer.dispatch(
        x, topk_weights=cached_topk_weights, handle=handle,
        async_with_compute_stream=True)
    async_args = runtime.dispatch_calls[-1]
    assert async_args[20:24] == (None, None, True, False)
    assert async_handle is handle
    assert async_event.event is not None
    async_event.current_stream_wait()
    async_event.current_stream_wait()

    deterministic_calls = []
    buffer.deterministic = True
    handle.deterministic_sort = lambda *args: deterministic_calls.append(args)
    generation_before_wait = handle._ascend_generation
    _, _, _, deterministic_handle, deterministic_event = buffer.dispatch(
        x, topk_weights=cached_topk_weights, handle=handle,
        async_with_compute_stream=True)
    assert deterministic_handle is handle
    assert handle._ascend_generation == generation_before_wait
    deterministic_event.current_stream_wait()
    assert handle._ascend_generation == generation_before_wait + 1
    assert len(deterministic_calls) == 1
    buffer.deterministic = False

    previous = deep_ep.ElasticBuffer.capture()
    _, _, _, previous_handle, previous_event = buffer.dispatch(
        x, topk_weights=cached_topk_weights, handle=handle,
        previous_event=previous, async_with_compute_stream=True,
        allocate_on_comm_stream=True)
    previous_args = runtime.dispatch_calls[-1]
    assert previous_args[20] is previous
    assert previous_args[21:24] == (None, True, True)
    assert previous_handle is handle
    previous_event.current_stream_wait()

    _, _, _, sync_allocated_handle, sync_allocated_event = buffer.dispatch(
        x, topk_weights=cached_topk_weights, handle=handle,
        allocate_on_comm_stream=True)
    assert runtime.dispatch_calls[-1][20:24] == (None, None, False, True)
    assert sync_allocated_handle is handle
    assert sync_allocated_event.event is None

    rejected_calls = len(runtime.dispatch_calls)
    scale_factors = _FakeTensor("npu", (1, 1), torch.float32)
    try:
        buffer.dispatch(
            (x, scale_factors), topk_idx=topk_idx, num_experts=2,
            num_max_tokens_per_rank=1)
    except RuntimeError as error:
        assert "invalid_fp8_pairing" in str(error), error
    else:
        raise AssertionError("Ascend dispatch accepted BF16 scale factors")
    assert len(runtime.dispatch_calls) == rejected_calls

    generation_queries_before_capture = runtime.dispatch_generation_queries
    descriptor_cpu_calls_before_capture = \
        handle.token_metadata_at_forward.cpu_calls
    torch.npu.capturing = True
    try:
        capture_modes = (
            ("sync", False, False, None),
            ("sync allocated", False, True, None),
            ("async", True, False, None),
            ("async allocated", True, True, None),
            ("sync predecessor", False, True, previous),
            ("async predecessor", True, True, previous),
        )
        for name, async_mode, allocate, predecessor in capture_modes:
            try:
                buffer.dispatch(
                    x, topk_weights=cached_topk_weights, handle=handle,
                    previous_event=predecessor,
                    async_with_compute_stream=async_mode,
                    allocate_on_comm_stream=allocate)
            except RuntimeError as error:
                assert "unsupported_graph_capture" in str(error), (name, error)
            else:
                raise AssertionError(
                    f"Ascend dispatch accepted graph capture for {name}")
        try:
            buffer.dispatch(
                x, topk_idx=topk_idx, num_experts=2,
                num_max_tokens_per_rank=1)
        except RuntimeError as error:
            assert "unsupported_graph_capture" in str(error), error
        else:
            raise AssertionError("Ascend uncached dispatch accepted graph capture")
    finally:
        torch.npu.capturing = False
    assert len(runtime.dispatch_calls) == rejected_calls
    assert runtime.dispatch_generation_queries == \
        generation_queries_before_capture
    assert handle.token_metadata_at_forward.cpu_calls == \
        descriptor_cpu_calls_before_capture

    capture_query = torch.npu.is_current_stream_capturing
    torch.npu.is_current_stream_capturing = None
    try:
        for name, kwargs in (
                ("cached", {
                    "topk_weights": cached_topk_weights,
                    "handle": handle,
                }),
                ("uncached", {
                    "topk_idx": topk_idx,
                    "num_experts": 2,
                    "num_max_tokens_per_rank": 1,
                })):
            try:
                buffer.dispatch(x, **kwargs)
            except RuntimeError as error:
                assert "unsupported_graph_capture" in str(error), (name, error)
            else:
                raise AssertionError(
                    f"Ascend dispatch accepted missing capture probe for {name}")
    finally:
        torch.npu.is_current_stream_capturing = capture_query
    assert len(runtime.dispatch_calls) == rejected_calls
    assert runtime.dispatch_generation_queries == \
        generation_queries_before_capture
    assert handle.token_metadata_at_forward.cpu_calls == \
        descriptor_cpu_calls_before_capture

    buffer.allow_hybrid_mode = True
    try:
        buffer.dispatch(
            x, topk_weights=cached_topk_weights, handle=handle,
            async_with_compute_stream=True)
    except RuntimeError as error:
        assert "unsupported_dispatch_mode" in str(error), error
    else:
        raise AssertionError("Ascend dispatch accepted hybrid async mode")
    finally:
        buffer.allow_hybrid_mode = False
    assert len(runtime.dispatch_calls) == rejected_calls

    for expanded in (False, True):
        for async_mode in (False, True):
            for allocate in (False, True):
                _, recv_topk_idx, _, uncached_handle, uncached_event = \
                    buffer.dispatch(
                        x, topk_idx=topk_idx, num_experts=2,
                        num_max_tokens_per_rank=1, do_cpu_sync=True,
                        do_expand=expanded,
                        async_with_compute_stream=async_mode,
                        allocate_on_comm_stream=allocate)
                uncached_args = runtime.dispatch_calls[-1]
                assert uncached_args[22:27] == (
                    async_mode, allocate, True, True, expanded)
                assert (recv_topk_idx is None) == expanded
                assert uncached_handle.do_expand == expanded
                if async_mode:
                    assert uncached_event.event is not None
                    uncached_event.current_stream_wait()
                else:
                    assert uncached_event.event is None
                assert uncached_handle._ascend_owner is buffer

    predecessor = deep_ep.EventOverlap(extension.EventHandle())
    _, _, _, predecessor_handle, predecessor_event = buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=2,
        num_max_tokens_per_rank=1, do_cpu_sync=True,
        previous_event=predecessor, async_with_compute_stream=True,
        allocate_on_comm_stream=True)
    predecessor_args = runtime.dispatch_calls[-1]
    assert predecessor_args[20] is predecessor.event
    assert predecessor_args[22:26] == (True, True, True, True)
    predecessor_event.current_stream_wait()
    assert predecessor_handle._ascend_owner is buffer

    rejected_calls = len(runtime.dispatch_calls)
    for name, kwargs in (
            ("previous without comm allocation", {
                "topk_weights": cached_topk_weights,
                "handle": predecessor_handle,
                "previous_event": previous}),
            ("pre-epilogue predecessor", {
                "topk_weights": cached_topk_weights,
                "handle": predecessor_handle,
                "previous_event_before_epilogue": previous,
                "allocate_on_comm_stream": True})):
        try:
            buffer.dispatch(x, **kwargs)
        except RuntimeError as error:
            assert "unsupported_dispatch_mode" in str(error), (name, error)
        else:
            raise AssertionError(f"Ascend dispatch accepted {name}")
        assert len(runtime.dispatch_calls) == rejected_calls

    for name, kwargs in (("num_sms", {"num_sms": 2}),
                         ("num_qps", {"num_qps": 1})):
        try:
            buffer.dispatch(x, topk_idx=topk_idx, num_experts=2,
                            num_max_tokens_per_rank=1, **kwargs)
        except RuntimeError as error:
            assert "invalid_launch_configuration" in str(error)
        else:
            raise AssertionError(f"Ascend dispatch accepted {name}={kwargs[name]}")

    no_weights_idx = _FakeTensor("npu", (1, 1), torch.int64)
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

    empty_x = _FakeTensor("npu", (0, 16), torch.bfloat16)
    empty_topk_idx = _FakeTensor("npu", (0, 1), torch.int64)
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


def _scenario_ascend_fp8_dispatch():
    deep_ep, extension, events = _load_package("ascend", True)
    group = _FakeGroup(events, rank=0, size=2)
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (2, 64), torch.float8_e4m3fn)
    topk_idx = _FakeTensor("npu", (2, 1), torch.int64)

    def gather_valid(value):
        contract = _fixed_preflight_contract(value)
        assert contract["x_dtype"] == str(torch.float8_e4m3fn), contract
        assert contract["sf_dtype"] == str(torch.float32), contract
        assert contract["sf_shape"] == [None, 2], contract
        assert contract["sf_layout"] == "row_major", contract
        assert contract["column_major_sf"] == 0, contract
        return [value, value]

    sf = _FakeTensor("npu", (2, 2), torch.float32, strides=(2, 1))
    group.gathered_objects = gather_valid
    recv_x, _, _, handle, _ = buffer.dispatch(
        (x, sf), topk_idx=topk_idx, num_experts=2,
        num_max_tokens_per_rank=2)
    assert isinstance(recv_x, tuple) and recv_x == (x, sf), recv_x
    assert runtime.dispatch_calls[-1][1] is sf

    packed_sf = _FakeTensor(
        "npu", (2, 2), torch.int32, contiguous=False, strides=(1, 4))

    def gather_column_major(value):
        contract = _fixed_preflight_contract(value)
        assert contract["sf_dtype"] == str(torch.int32), contract
        assert contract["sf_layout"] == "column_major", contract
        assert contract["column_major_sf"] == 1, contract
        return [value, value]

    group.gathered_objects = gather_column_major
    recv_x, _, _, _, _ = buffer.dispatch(
        (x, packed_sf), handle=handle,
        use_tma_aligned_col_major_sf=True)
    assert recv_x == (x, packed_sf), recv_x
    assert runtime.dispatch_calls[-1][28] is True

    invalid_cases = (
        (x, None),
        (_FakeTensor("npu", (2, 64), torch.bfloat16), sf),
        (x, _FakeTensor("npu", (2, 2), torch.bfloat16)),
        (x, _FakeTensor("cpu", (2, 2), torch.float32)),
        (x, _FakeTensor("npu", (1, 2), torch.float32)),
        (x, _FakeTensor("npu", (2, 0), torch.float32)),
        (x, _FakeTensor("npu", (2, 2), torch.float32,
                        contiguous=False, strides=(0, 1))),
    )
    group.gathered_objects = None
    for invalid_x, invalid_sf in invalid_cases:
        calls = len(runtime.dispatch_calls)
        value = invalid_x if invalid_sf is None else (invalid_x, invalid_sf)
        try:
            buffer.dispatch(
                value, topk_idx=topk_idx, num_experts=2,
                num_max_tokens_per_rank=2)
        except RuntimeError as error:
            assert "dispatch preflight failed" in str(error), error
        else:
            raise AssertionError("invalid FP8 dispatch reached runtime")
        assert len(runtime.dispatch_calls) == calls

    def gather_mismatched_layout(value):
        remote_contract = _fixed_preflight_contract(value)
        remote_contract["sf_layout"] = "strided"
        return [value, _fixed_preflight_record("dispatch", remote_contract)]

    group.gathered_objects = gather_mismatched_layout
    calls = len(runtime.dispatch_calls)
    try:
        buffer.dispatch(
            (x, sf), topk_idx=topk_idx, num_experts=2,
            num_max_tokens_per_rank=2)
    except RuntimeError as error:
        assert "sf_layout_mismatch" in str(error), error
    else:
        raise AssertionError("asymmetric SF layout reached runtime")
    assert len(runtime.dispatch_calls) == calls
    buffer.destroy()


def _scenario_ascend_owner_device_preflight():
    deep_ep, extension, events = _load_package("ascend", True)
    group = _FakeGroup(events, rank=0, size=2)
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=2 * 1024 * 1024, allow_hybrid_mode=False,
        explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    torch = sys.modules["torch"]
    assert buffer._ascend_owner_device == 0
    x = _FakeTensor("npu", (1, 16), torch.bfloat16, device_index=1)
    topk_idx = _FakeTensor("npu", (1, 1), torch.int64, device_index=1)

    def one_rank_mismatch(value):
        assert value[0] == "dispatch", value
        assert value[1] == 0 and value[2] == "invalid_buffer_device", value
        return [value, _fixed_preflight_record(
            "dispatch", _fixed_preflight_contract(value))]

    group.gathered_objects = one_rank_mismatch
    try:
        buffer.dispatch(
            x, topk_idx=topk_idx, num_experts=2,
            num_max_tokens_per_rank=1)
    except RuntimeError as error:
        assert "dispatch preflight failed on rank 0 (invalid_buffer_device)" in str(error)
    else:
        raise AssertionError("owner-device mismatch reached runtime")
    assert not runtime.dispatch_calls

    group.gathered_objects = None
    healthy_x = _FakeTensor("npu", (1, 16), torch.bfloat16)
    healthy_topk = _FakeTensor("npu", (1, 1), torch.int64)
    buffer.dispatch(
        healthy_x, topk_idx=healthy_topk, num_experts=2,
        num_max_tokens_per_rank=1)
    assert len(runtime.dispatch_calls) == 1
    buffer.destroy()


def _scenario_ascend_dispatch_optimized():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (1, 16), torch.bfloat16)
    topk_idx = _FakeTensor("npu", (1, 1), torch.int64)
    expected_error = "invalid_launch_configuration"

    for kwargs in ({"num_sms": 2}, {"num_qps": 1}):
        call_count = len(runtime.dispatch_calls)
        try:
            buffer.dispatch(_Poison(), **kwargs)
        except RuntimeError as error:
            if expected_error not in str(error):
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


def _new_ascend_publication_fixture():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (1, 16), torch.bfloat16)
    topk_idx = _FakeTensor("npu", (1, 1), torch.int64)
    _, _, _, handle, _ = buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=2,
        num_max_tokens_per_rank=1)
    return buffer, extension.runtime_instances[-1], x, handle


def _scenario_ascend_copied_event_publication():
    buffer, runtime, x, handle = _new_ascend_publication_fixture()
    initial_generation = handle._ascend_generation
    _, _, _, _, overlap = buffer.dispatch(
        x, handle=handle, async_with_compute_stream=True)
    copied_event = copy.copy(overlap.event)
    copied_event.current_stream_wait()
    assert runtime.last_dispatch_generation == initial_generation + 1
    assert buffer._ascend_handle_generation == initial_generation

    _, _, _, reused_handle, reused_event = buffer.dispatch(x, handle=handle)
    assert reused_handle is handle
    assert reused_event.event is None
    assert handle._ascend_generation == initial_generation + 2
    buffer.destroy()


def _scenario_ascend_dropped_event_publication():
    buffer, runtime, x, handle = _new_ascend_publication_fixture()
    initial_generation = handle._ascend_generation
    _, _, _, _, overlap = buffer.dispatch(
        x, handle=handle, async_with_compute_stream=True)
    del overlap
    import gc
    gc.collect()
    assert runtime.last_dispatch_generation == initial_generation + 1
    assert buffer._ascend_handle_generation == initial_generation

    _, _, _, reused_handle, reused_event = buffer.dispatch(x, handle=handle)
    assert reused_handle is handle
    assert reused_event.event is None
    assert handle._ascend_generation == initial_generation + 2
    buffer.destroy()


def _scenario_ascend_failed_event_does_not_publish():
    buffer, runtime, x, handle = _new_ascend_publication_fixture()
    initial_generation = handle._ascend_generation
    initial_fingerprint = handle._ascend_descriptor_fingerprint
    runtime.fail_next_dispatch_completion = True
    _, _, _, _, overlap = buffer.dispatch(
        x, handle=handle, async_with_compute_stream=True)
    try:
        overlap.current_stream_wait()
    except RuntimeError as error:
        assert "dispatch completion failed" in str(error), error
    else:
        raise AssertionError("failed Ascend completion unexpectedly succeeded")
    assert runtime.last_dispatch_generation == initial_generation
    assert buffer._ascend_handle_generation == initial_generation
    assert handle._ascend_generation == initial_generation
    assert handle._ascend_descriptor_fingerprint == initial_fingerprint
    buffer.destroy()


def _new_ascend_hybrid_route_identity_fixture(deep_ep, extension, events):
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (1, 16), torch.bfloat16)
    topk_idx = _FakeTensor("npu", (1, 1), torch.int64)
    _, _, _, handle, _ = buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=2,
        num_max_tokens_per_rank=1)
    runtime = extension.runtime_instances[-1]
    buffer.allow_hybrid_mode = True
    descriptor = handle.token_metadata_at_forward
    descriptor._values.extend([37])
    descriptor.shape = (len(descriptor._values),)
    committed_fingerprint = tuple(descriptor._values)
    runtime.committed_dispatch_fingerprint = committed_fingerprint
    runtime.dispatch_identity_prefix = 120
    handle._ascend_descriptor_fingerprint = committed_fingerprint
    descriptor._values[-1] = 41
    return buffer, runtime, x, handle


def _scenario_ascend_hybrid_route_mutation_is_rejected():
    deep_ep, extension, events = _load_package("ascend", True)
    dispatch_buffer, dispatch_runtime, dispatch_x, dispatch_handle = \
        _new_ascend_hybrid_route_identity_fixture(deep_ep, extension, events)
    dispatch_calls = len(dispatch_runtime.dispatch_calls)
    try:
        dispatch_buffer.dispatch(dispatch_x, handle=dispatch_handle)
    except RuntimeError as error:
        assert "invalid_dispatch_handle" in str(error), error
    else:
        raise AssertionError("cached dispatch accepted mutated hybrid route payload")
    assert len(dispatch_runtime.dispatch_calls) == dispatch_calls
    dispatch_buffer.destroy()

    combine_buffer, combine_runtime, combine_x, combine_handle = \
        _new_ascend_hybrid_route_identity_fixture(deep_ep, extension, events)
    combine_calls = len(combine_runtime.combine_calls)
    try:
        combine_buffer.combine(combine_x, combine_handle)
    except RuntimeError as error:
        assert "invalid_dispatch_handle" in str(error), error
    else:
        raise AssertionError("combine accepted mutated hybrid route payload")
    assert len(combine_runtime.combine_calls) == combine_calls
    combine_buffer.destroy()


def _scenario_ascend_combine():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events, rank=0, size=2), num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False, explicitly_destroy=True)
    runtime = extension.runtime_instances[-1]
    torch = sys.modules["torch"]
    x = _FakeTensor("npu", (2, 16), torch.bfloat16)
    topk_weights = _FakeTensor("npu", (2, 2), torch.float32)
    bias_0 = _FakeTensor("npu", (1, 16), torch.bfloat16)
    bias_1 = _FakeTensor("npu", (1, 16), torch.bfloat16)
    recv_src_metadata = _FakeTensor("npu", (2, 4))
    topk_idx = _FakeTensor("npu", (1, 2))
    rank_prefix = _FakeTensor("npu", (2,))
    descriptor = _FakeTensor("npu", (120,))
    handle = deep_ep.EPHandle(
        False, 2, 4, 4, 1, topk_idx, 2, 2, [], rank_prefix,
        _FakeTensor("npu", (1,)), _FakeTensor("npu", (1,)),
        recv_src_metadata, _FakeTensor("npu", (1, 2)), descriptor, None)
    buffer._ascend_handle_generation = 1
    handle._ascend_owner = buffer
    handle._ascend_generation = 1
    handle._ascend_descriptor_fingerprint = tuple(descriptor._values)

    combined_x, combined_weights, event = buffer.combine(
        x, handle, topk_weights=topk_weights, bias=(bias_0, bias_1))
    assert combined_x is x
    assert combined_weights is topk_weights
    assert event.event is None
    assert runtime.combine_calls[-1] == (
        x, topk_weights, bias_0, bias_1, recv_src_metadata, topk_idx,
        rank_prefix, descriptor, None, 2, 4, 1, 0,
        None, None, False, False, False)

    def expect_owner_rejection(call):
        calls = len(runtime.combine_calls)
        try:
            call()
        except RuntimeError as error:
            assert "combine preflight failed on rank 0 (invalid_buffer_device)" in str(error)
        else:
            raise AssertionError("owner-device mismatch reached combine runtime")
        assert len(runtime.combine_calls) == calls

    expect_owner_rejection(lambda: buffer.combine(
        x, handle, topk_weights=_FakeTensor(
            "npu", (2, 2), torch.float32, device_index=1)))
    recv_src_metadata.device.index = 1
    expect_owner_rejection(lambda: buffer.combine(x, handle))
    recv_src_metadata.device.index = 0
    buffer.combine(x, handle)

    _, _, one_bias_event = buffer.combine(
        x, handle, bias=bias_0, num_sms=1, allocate_on_comm_stream=True)
    assert runtime.combine_calls[-1][2:4] == (bias_0, None)
    assert one_bias_event.event is None
    assert runtime.combine_calls[-1][13:17] == (None, None, False, True)

    _, _, async_event = buffer.combine(
        x, handle, async_with_compute_stream=True)
    assert async_event.event is not None
    assert async_event.event is runtime.combine_events[-1]
    assert runtime.combine_calls[-1][13:17] == (None, None, True, False)
    async_event.current_stream_wait()
    async_event.current_stream_wait()

    previous = deep_ep.ElasticBuffer.capture()
    _, _, predecessor_event = buffer.combine(
        x, handle, previous_event=previous,
        async_with_compute_stream=True, allocate_on_comm_stream=True)
    assert predecessor_event.event is not None
    assert runtime.combine_calls[-1][13:17] == (
        previous, None, True, True)
    predecessor_event.current_stream_wait()
    _, _, predecessor_sync_event = buffer.combine(
        x, handle, previous_event=previous,
        allocate_on_comm_stream=True)
    assert predecessor_sync_event.event is None
    assert runtime.combine_calls[-1][13:17] == (
        previous, None, False, True)

    handle.do_expand = True
    expanded_weights = _FakeTensor("npu", (2,), torch.float32)
    _, returned_expanded_weights, expanded_event = buffer.combine(
        x, handle, topk_weights=expanded_weights,
        async_with_compute_stream=True)
    assert returned_expanded_weights is expanded_weights
    assert runtime.combine_calls[-1][17] is True
    expanded_event.current_stream_wait()
    _, returned_unweighted, unweighted_event = buffer.combine(x, handle)
    assert returned_unweighted is None
    assert unweighted_event.event is None
    handle.do_expand = False

    empty_x = _FakeTensor("npu", (0, 16), torch.bfloat16)
    _, _, empty_event = buffer.combine(
        empty_x, handle, async_with_compute_stream=True)
    assert runtime.combine_calls[-1][0] is empty_x
    empty_event.current_stream_wait()

    buffer.combine(x, handle)
    assert runtime.combine_calls[-1][2:4] == (None, None)

    invalid_calls = (
        ("num_sms", {"num_sms": 2}),
        ("num_qps", {"num_qps": 1}),
        ("previous_event_without_allocation", {
            "previous_event": deep_ep.EventOverlap(extension.EventHandle())}),
        ("previous_event_before_epilogue", {
            "previous_event_before_epilogue": extension.EventHandle()}),
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

    generation_queries_before_capture = runtime.dispatch_generation_queries
    descriptor_cpu_calls_before_capture = descriptor.cpu_calls
    torch.npu.capturing = True
    try:
        capture_modes = (
            ("sync", False, False, None),
            ("sync allocated", False, True, None),
            ("async", True, False, None),
            ("async allocated", True, True, None),
            ("sync predecessor", False, True, previous),
            ("async predecessor", True, True, previous),
        )
        for name, async_mode, allocate, predecessor in capture_modes:
            try:
                buffer.combine(
                    x, handle, previous_event=predecessor,
                    async_with_compute_stream=async_mode,
                    allocate_on_comm_stream=allocate)
            except RuntimeError as error:
                assert "unsupported_graph_capture" in str(error), (name, error)
            else:
                raise AssertionError(
                    f"Ascend combine accepted graph capture for {name}")
    finally:
        torch.npu.capturing = False
    assert len(runtime.combine_calls) == calls_before
    assert runtime.dispatch_generation_queries == \
        generation_queries_before_capture
    assert descriptor.cpu_calls == descriptor_cpu_calls_before_capture

    capture_query = torch.npu.is_current_stream_capturing
    torch.npu.is_current_stream_capturing = None
    try:
        try:
            buffer.combine(x, handle)
        except RuntimeError as error:
            assert "unsupported_graph_capture" in str(error), error
        else:
            raise AssertionError("Ascend combine accepted missing capture probe")
    finally:
        torch.npu.is_current_stream_capturing = capture_query
    assert len(runtime.combine_calls) == calls_before
    assert runtime.dispatch_generation_queries == \
        generation_queries_before_capture
    assert descriptor.cpu_calls == descriptor_cpu_calls_before_capture

    buffer.allow_hybrid_mode = True
    try:
        buffer.combine(x, handle, async_with_compute_stream=True)
    except RuntimeError as error:
        assert "unsupported_combine_mode" in str(error), error
    else:
        raise AssertionError("Ascend async combine accepted hybrid mode")
    finally:
        buffer.allow_hybrid_mode = False
    assert len(runtime.combine_calls) == calls_before

    saved_topology = buffer._ascend_topology
    saved_scaleout = buffer.num_scaleout_ranks
    buffer._ascend_topology = (
        saved_topology[0], saved_topology[1], saved_topology[2],
        saved_topology[1] * 2)
    buffer.num_scaleout_ranks = 2
    try:
        buffer.combine(x, handle, async_with_compute_stream=True)
    except RuntimeError as error:
        assert "unsupported_combine_mode" in str(error), error
    else:
        raise AssertionError("Ascend async combine accepted physical scale-out")
    finally:
        buffer._ascend_topology = saved_topology
        buffer.num_scaleout_ranks = saved_scaleout
    assert len(runtime.combine_calls) == calls_before

    for name, invalid_x in (
            ("mapped CPU", _FakeTensor("cpu", (2, 16), torch.bfloat16)),
            ("FP8", _FakeTensor("npu", (2, 16), torch.float8_e4m3fn))):
        try:
            buffer.combine(
                invalid_x, handle, async_with_compute_stream=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Ascend async combine accepted {name} input")
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
    "ascend_destroy_state_publication":
        _scenario_ascend_destroy_state_publication,
    "ascend_topology_preflight": _scenario_ascend_topology_preflight,
    "ascend_topology_preflight_mismatch":
        _scenario_ascend_topology_preflight_mismatch,
    "ascend_topology_preflight_local_parse_failure":
        _scenario_ascend_topology_preflight_local_parse_failure,
    "ascend_topology_preflight_remote_parse_failure":
        _scenario_ascend_topology_preflight_remote_parse_failure,
    "ascend_collective_contract_preflight":
        _scenario_ascend_collective_contract_preflight,
    "ascend_hybrid_collective_preflight":
        _scenario_ascend_hybrid_collective_preflight,
    "ascend_implicit_size": _scenario_ascend_implicit_size,
    "ascend_method_gates": _scenario_ascend_method_gates,
    "ascend_contextmanager_gate": _scenario_ascend_contextmanager_gate,
    "ascend_dispatch": _scenario_ascend_dispatch,
    "ascend_fp8_dispatch": _scenario_ascend_fp8_dispatch,
    "ascend_owner_device_preflight": _scenario_ascend_owner_device_preflight,
    "ascend_dispatch_optimized": _scenario_ascend_dispatch_optimized,
    "ascend_copied_event_publication":
        _scenario_ascend_copied_event_publication,
    "ascend_dropped_event_publication":
        _scenario_ascend_dropped_event_publication,
    "ascend_failed_event_does_not_publish":
        _scenario_ascend_failed_event_does_not_publish,
    "ascend_hybrid_route_mutation_is_rejected":
        _scenario_ascend_hybrid_route_mutation_is_rejected,
    "ascend_combine": _scenario_ascend_combine,
    "ascend_weak_lru_gate": _scenario_ascend_weak_lru_gate,
    "cuda_preservation": _scenario_cuda_preservation,
    "stale_cuda_extension_guard": _scenario_stale_cuda_extension_guard,
}


class PythonApiIsolationTest(unittest.TestCase):
    def test_fp8_runtime_matrix_has_independent_public_contract(self):
        self.assertTrue(FP8_RUNTIME_MATRIX.is_file())
        source = FP8_RUNTIME_MATRIX.read_text()
        tree = ast.parse(source)
        case_names = set(ast.literal_eval(next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "CASE_NAMES"
                for target in node.targets))))
        self.assertTrue({
            "normal-fp32", "packed-int32", "expanded", "zero-padded",
            "cached-representation-change", "weighted", "empty-input",
            "asymmetric-routing", "negative-one-route", "bf16-combine",
            "sequential-100-generations", "malformed-inputs",
        }.issubset(case_names))
        for marker in (
                "buffer.dispatch(", "buffer.combine(", "dist.all_gather(",
                "torch.equal(", "view(torch.uint8)", "view(torch.int32)",
                "dist.all_reduce(", "finally:", "buffer.destroy()"):
            self.assertIn(marker, source)
        result = subprocess.run(
            [sys.executable, str(FP8_RUNTIME_MATRIX), "--contract"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = json.loads(result.stdout)
        self.assertEqual(contract["reference"], "gathered-fp8-bytes-and-sf-packs")
        self.assertEqual(contract["supported_world_sizes"], [2, 4, 8])
        self.assertIn("exact-payload-bytes", contract["contract_checks"])
        self.assertIn("exact-scale-factor-packs", contract["contract_checks"])
        self.assertIn("exact-column-major-scale-stride", contract["contract_checks"])
        self.assertIn("cached-bf16-to-fp8-and-fp8-to-bf16", contract["contract_checks"])
        self.assertIn("independent-fp8-dequantization", contract["contract_checks"])
        self.assertIn("one-rank-malformed-recovery", contract["contract_checks"])

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

    def test_failed_destroy_publishes_only_confirmed_destroyed_state(self):
        self.run_scenario("ascend_destroy_state_publication")

    def test_ascend_collective_contract_preflight_precedes_runtime_work(self):
        self.run_scenario("ascend_collective_contract_preflight")

    def test_hybrid_collective_preflight_rejects_asymmetric_records_before_runtime(self):
        self.run_scenario("ascend_hybrid_collective_preflight")

    def test_explicit_logical_topology_is_aggregated_before_construction(self):
        self.run_scenario("ascend_topology_preflight")

    def test_asymmetric_logical_topology_fails_before_construction(self):
        self.run_scenario("ascend_topology_preflight_mismatch")

    def test_local_topology_parse_failure_is_gathered_before_rejection(self):
        self.run_scenario("ascend_topology_preflight_local_parse_failure")

    def test_remote_topology_parse_failure_is_rejected_identically(self):
        self.run_scenario("ascend_topology_preflight_remote_parse_failure")

    def test_implicit_size_reaches_backend_transport_error_without_nccl(self):
        self.run_scenario("ascend_implicit_size")

    def test_all_runtime_methods_fail_before_argument_or_symbol_access(self):
        self.run_scenario("ascend_method_gates")

    def test_contextmanager_method_gates_when_called(self):
        self.run_scenario("ascend_contextmanager_gate")

    def test_dispatch_routes_to_the_synchronous_ascend_runtime(self):
        self.run_scenario("ascend_dispatch")

    def test_fp8_dispatch_collectively_validates_scale_factor_contract(self):
        self.run_scenario("ascend_fp8_dispatch")

    def test_owner_device_mismatch_is_collective_and_allows_retry(self):
        self.run_scenario("ascend_owner_device_preflight")

    def test_dispatch_count_validation_survives_optimized_python(self):
        self.run_scenario("ascend_dispatch_optimized", optimize=True)

    def test_copied_raw_event_wait_publishes_cached_handle(self):
        self.run_scenario("ascend_copied_event_publication")

    def test_dropped_event_publishes_cached_handle_before_reuse(self):
        self.run_scenario("ascend_dropped_event_publication")

    def test_failed_event_completion_does_not_publish_cached_handle(self):
        self.run_scenario("ascend_failed_event_does_not_publish")

    def test_hybrid_route_mutation_is_rejected_before_dispatch_or_combine(self):
        self.run_scenario("ascend_hybrid_route_mutation_is_rejected")

    def test_combine_async_routes_native_events_and_rejects_deferred_modes(self):
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
            return 0

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

    def test_constructor_rejects_empty_group_before_hccl_use(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "DeepEP Ascend backend: topology requires at least two ranks"):
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
