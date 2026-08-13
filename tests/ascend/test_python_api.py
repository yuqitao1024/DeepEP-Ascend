import ast
import importlib.abc
import importlib.util
import os
import pathlib
import subprocess
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ELASTIC_SOURCE = ROOT / "deep_ep/buffers/elastic.py"
ENVS_SOURCE = ROOT / "deep_ep/utils/envs.py"

PHASE_ERROR = "DeepEP Ascend backend: {} is not implemented in phase 1"
GATED_METHODS = {
    "barrier": "barrier",
    "get_comm_stream": "get_comm_stream",
    "get_physical_domain_size": "get_physical_domain_size",
    "get_logical_domain_size": "get_logical_domain_size",
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
    "dispatch": "dispatch",
    "combine": "combine",
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
        return types.SimpleNamespace(_comm_ptr=lambda: self.comm_pointer)


def _phase_error(operation):
    return NotImplementedError(PHASE_ERROR.format(operation))


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
    torch.device = lambda name: name
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
            return types.SimpleNamespace(multi_processor_count=80)

        def set_device(self, device):
            self._record("set_device", device)

        def Stream(self, **kwargs):
            self._record("Stream", **kwargs)
            return _FakeStream(**kwargs)

    torch.cuda = FakeCuda()

    distributed = types.ModuleType("torch.distributed")
    distributed.ProcessGroup = _FakeGroup
    distributed.get_rank = lambda: 0
    distributed.get_world_size = lambda: 1
    distributed.new_group = lambda ranks: None
    distributed.init_process_group = lambda **kwargs: None
    distributed.barrier = lambda: events.append("dist.barrier")
    distributed.all_gather_object = lambda output, value, group: None
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
                raise _phase_error("current_stream_wait")
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
            return 2, 4

        def get_physical_domain_size(self):
            events.append("runtime.get_physical_domain_size")
            return 2, 4

        def barrier(self, use_comm_stream, with_cpu_sync, sequential):
            events.append(("runtime.barrier", use_comm_stream, with_cpu_sync, sequential))

        def dispatch(self, *args):
            self.dispatch_calls.append(args)
            recv_src_metadata = _FakeTensor(shape=(1,))
            return (args[0], args[1], args[2], args[3], args[2],
                    1, 1, [], _FakeTensor(), _FakeTensor(), _FakeTensor(),
                    recv_src_metadata, _FakeTensor(), None, None, EventHandle())

        def combine(self, *args):
            self.combine_calls.append(args)
            return args[0], args[1], EventHandle()

    def calculate_elastic_buffer_size(*args):
        extension.size_calls.append(args)
        if platform == "ascend":
            raise _phase_error("calculate_elastic_buffer_size")
        return 8192

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
        extension.create_nccl_comm = lambda unique_id, size, rank: 1234
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


def _assert_phase_error(operation, call):
    try:
        call()
    except Exception as error:
        assert type(error) is NotImplementedError, (operation, type(error), error)
        assert str(error) == PHASE_ERROR.format(operation), (operation, error)
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
    assert platform.get_comm_handle(_Poison()) is None
    assert platform.comm_handle_value(None) == 0
    platform.synchronize()
    assert not any(str(event).startswith("torch.cuda") for event in events)
    _assert_phase_error("adapter", lambda: platform.require_cuda("adapter"))
    _assert_phase_error(
        "validate_device_type",
        lambda: platform.validate_device_type(_Poison(), "validate_device_type"))
    _assert_phase_error("get_comm_stream", lambda: platform.wrap_stream(_Poison()))

    event = platform.capture_event()
    wrapped = types.SimpleNamespace(event=event)
    assert platform.unwrap_event(None) is None
    assert platform.unwrap_event(event) is event
    assert platform.unwrap_event(wrapped) is event
    assert deep_ep.ElasticBuffer.capture().__class__ is extension.EventHandle
    _assert_phase_error("current_stream_wait", event.current_stream_wait)

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
    group = _FakeGroup(events, rank=3, size=8)
    os.environ["EP_OVERRIDE_RDMA_SL"] = "11"
    buffer = deep_ep.ElasticBuffer(
        group, num_bytes=4096, explicitly_destroy=True, num_allocated_qps=77)

    assert buffer.rank_idx == 3
    assert buffer.num_ranks == 8
    assert buffer.num_bytes == 4096
    assert buffer.num_allocated_qps == 0
    assert buffer.comm_handle is None
    assert buffer.num_scaleout_ranks is None
    assert buffer.num_scaleup_ranks is None
    assert buffer.scaleout_rank_idx is None
    assert buffer.scaleup_rank_idx is None
    assert buffer.num_rdma_ranks is None
    assert buffer.num_nvlink_ranks is None
    assert len(extension.runtime_args) == 1
    runtime_args = extension.runtime_args[0]
    assert runtime_args[:6] == (3, 8, 0, [], 4096, 0)
    assert runtime_args[9:11] == (3, 0)
    assert "group.barrier" not in events
    assert "runtime.get_logical_domain_size" not in events
    assert "runtime.get_physical_domain_size" not in events
    assert not any(isinstance(event, tuple) and str(event[0]).startswith("torch.cuda")
                   for event in events)

    buffer.destroy()
    assert buffer.runtime is None
    assert buffer.comm_handle is None

    try:
        deep_ep.ElasticBuffer(group, num_bytes=0, explicitly_destroy=True)
    except ValueError as error:
        assert str(error) == "num_bytes must be positive"
    else:
        raise AssertionError("zero-sized buffer was accepted")


def _scenario_ascend_implicit_size():
    deep_ep, extension, events = _load_package("ascend", True)
    group = _FakeGroup(events)
    _assert_phase_error(
        "calculate_elastic_buffer_size",
        lambda: deep_ep.ElasticBuffer(
            group, num_max_tokens_per_rank=1, hidden=16))
    assert extension.size_calls == [(0, 1, 16, 0, False, True, True)]

    _assert_phase_error(
        "calculate_elastic_buffer_size",
        lambda: deep_ep.ElasticBuffer.get_buffer_size_hint(group, 2, 32, 4))
    assert extension.size_calls[-1] == (0, 2, 32, 4, False, True, True)


def _scenario_ascend_method_gates():
    deep_ep, extension, events = _load_package("ascend", True)
    buffer = deep_ep.ElasticBuffer(
        _FakeGroup(events), num_bytes=4096, explicitly_destroy=True)
    poison = _Poison()
    calls = {
        "barrier": buffer.barrier,
        "get_comm_stream": buffer.get_comm_stream,
        "get_physical_domain_size": buffer.get_physical_domain_size,
        "get_logical_domain_size": buffer.get_logical_domain_size,
        "engram_write": lambda: buffer.engram_write(poison),
        "engram_fetch": lambda: buffer.engram_fetch(poison),
        "pp_set_config": lambda: buffer.pp_set_config(poison, poison),
        "pp_send": lambda: buffer.pp_send(poison, poison),
        "pp_recv": lambda: buffer.pp_recv(poison, poison),
        "create_agrs_session": buffer.create_agrs_session,
        "destroy_agrs_session": buffer.destroy_agrs_session,
        "agrs_new_session": lambda: buffer.agrs_new_session(False).__enter__(),
        "agrs_set_config": lambda: buffer.agrs_set_config(poison, poison),
        "agrs_get_inplace_tensor": lambda: buffer.agrs_get_inplace_tensor(poison, poison),
        "all_gather": lambda: buffer.all_gather(poison),
        "get_theoretical_num_sms": lambda: buffer.get_theoretical_num_sms(poison, poison),
        "get_theoretical_num_qps": lambda: buffer.get_theoretical_num_qps(poison),
        "dispatch": lambda: buffer.dispatch(poison),
        "combine": lambda: buffer.combine(poison, poison),
    }
    assert calls.keys() == GATED_METHODS.keys()
    for method, operation in GATED_METHODS.items():
        _assert_phase_error(operation, calls[method])

    envs = importlib.import_module("deep_ep.utils.envs")
    _assert_phase_error(
        "get_physical_domain_size",
        lambda: envs.get_physical_domain_size(poison))
    _assert_phase_error(
        "get_logical_domain_size",
        lambda: envs.get_logical_domain_size(poison))
    buffer.destroy()


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
    os.environ["EP_OVERRIDE_RDMA_SL"] = "7"
    group = _FakeGroup(events, rank=5, size=8)
    buffer = deep_ep.ElasticBuffer(group, num_bytes=4096, explicitly_destroy=True)

    runtime_args = extension.runtime_args[-1]
    assert runtime_args[:6] == (5, 8, 4242, [], 4096, 0)
    assert runtime_args[9:11] == (7, 129)
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


SCENARIOS = {
    "ascend_import": _scenario_ascend_import,
    "invalid_platform": _scenario_invalid_platform,
    "ascend_construction": _scenario_ascend_construction,
    "ascend_implicit_size": _scenario_ascend_implicit_size,
    "ascend_method_gates": _scenario_ascend_method_gates,
    "cuda_preservation": _scenario_cuda_preservation,
}


class PythonApiIsolationTest(unittest.TestCase):
    def run_scenario(self, scenario):
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--isolation", scenario],
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

    def test_implicit_size_reaches_backend_phase_error_without_nccl(self):
        self.run_scenario("ascend_implicit_size")

    def test_all_runtime_methods_fail_before_argument_or_symbol_access(self):
        self.run_scenario("ascend_method_gates")

    def test_cuda_initialization_and_constructor_behavior_are_preserved(self):
        self.run_scenario("cuda_preservation")


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

    def test_module_topology_helpers_gate_before_communicator_access(self):
        tree = ast.parse(ENVS_SOURCE.read_text())
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        for name in ("get_physical_domain_size", "get_logical_domain_size"):
            with self.subTest(function=name):
                self.assert_first_require_cuda(functions[name], name)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ASCEND_EXTENSION_AVAILABLE = TORCH_AVAILABLE and any(
    (ROOT / "deep_ep").glob("_C*.so"))


@unittest.skipUnless(
    ASCEND_EXTENSION_AVAILABLE,
    "real Ascend package tests require PyTorch and an in-place built extension")
class RealAscendPythonApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import deep_ep
        import deep_ep._C as extension

        cls.torch = torch
        cls.deep_ep = deep_ep
        cls.extension = extension
        if extension.get_platform() != "ascend":
            raise unittest.SkipTest(
                "real Ascend package tests require an Ascend extension")

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

    def assert_phase_error(self, operation, call):
        with self.assertRaises(NotImplementedError) as context:
            call()
        self.assertIs(type(context.exception), NotImplementedError)
        self.assertEqual(str(context.exception), PHASE_ERROR.format(operation))

    def test_import_skips_cuda_exports(self):
        self.assertTrue(hasattr(self.deep_ep, "ElasticBuffer"))
        self.assertFalse(hasattr(self.deep_ep, "Buffer"))
        self.assertFalse(hasattr(self.extension, "init_jit"))

    def test_explicit_size_constructs_without_cuda_or_topology(self):
        buffer = self.deep_ep.ElasticBuffer(
            self.FakeGroup(), num_bytes=4096, explicitly_destroy=True)
        self.assertEqual(buffer.num_bytes, 4096)
        self.assertIsNone(buffer.num_scaleout_ranks)
        self.assertIsNone(buffer.num_scaleup_ranks)
        buffer.destroy()

    def test_implicit_size_raises_phase_error(self):
        self.assert_phase_error(
            "calculate_elastic_buffer_size",
            lambda: self.deep_ep.ElasticBuffer(
                self.FakeGroup(), num_max_tokens_per_rank=1, hidden=16))

    def test_core_operations_raise_before_cuda_helpers(self):
        buffer = self.deep_ep.ElasticBuffer(
            self.FakeGroup(), num_bytes=4096, explicitly_destroy=True)
        self.assert_phase_error(
            "dispatch",
            lambda: buffer.dispatch(
                self.torch.empty((1, 16), dtype=self.torch.bfloat16),
                self.torch.zeros((1, 1), dtype=self.torch.int64),
                num_experts=1))
        self.assert_phase_error("barrier", buffer.barrier)
        buffer.destroy()

    def test_cuda_only_methods_fail_by_name(self):
        buffer = self.deep_ep.ElasticBuffer(
            self.FakeGroup(), num_bytes=4096, explicitly_destroy=True)
        calls = {
            "engram_write": lambda: buffer.engram_write(self.torch.empty(1)),
            "pp_set_config": lambda: buffer.pp_set_config(32, 1),
            "create_agrs_session": buffer.create_agrs_session,
            "all_gather": lambda: buffer.all_gather(self.torch.empty(1)),
        }
        for operation, call in calls.items():
            with self.subTest(operation=operation):
                self.assert_phase_error(operation, call)
        buffer.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--isolation":
        SCENARIOS[sys.argv[2]]()
    else:
        unittest.main()
