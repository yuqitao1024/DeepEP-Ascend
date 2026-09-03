import ast
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ascend.benchmark.report import (
    BenchmarkReport,
    validate_comparable,
    write_report_atomic,
)
from tests.ascend.benchmark.compare import compare_reports
from tests.ascend.benchmark.bench_ep import build_parser, _selected_case_ids
from tests.ascend.benchmark.runtime import (
    AscendRuntime,
    PreparedCase,
    _aggregate_rank_operations,
    _aggregate_stage_profiles,
    _benchmark_timeout_seconds,
    _build_operation_work_counts,
    _configure_stage_profile_environment,
    _derive_host_envelope_samples,
    _derive_stage_timeline,
    _early_route_plan_observability,
    _payload_rows,
)
from tests.ascend.benchmark.timing import logical_gbps, summarize_samples
from tests.ascend.benchmark.timing import NpuEventTimer
from tests.ascend.benchmark.timeline import (
    operation_stage_semantics,
    stage_semantic,
)
from tests.ascend.benchmark.timeline_report import (
    build_timeline_report,
    render_timeline_markdown,
)
from tests.ascend.benchmark import workloads
from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_core import build_dispatch_arguments
from tests.utils.ep_benchmark_manifest import (
    BenchmarkManifest,
    EPModeCase,
    RankWorkload,
    WorkloadSpec,
    case_suite,
    enumerate_ep_mode_cases,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_EP = ROOT / "tests/ascend/benchmark/bench_ep.py"
RUNTIME = ROOT / "tests/ascend/benchmark/runtime.py"
COMPARE = ROOT / "tests/ascend/benchmark/compare.py"
ASYNC_OVERLAP = ROOT / "tests/ascend/production/run_async_overlap.py"

FP8_ASYNC_CASES = (
    "fp8-cached-normal-fp32-row-async-allocate-false",
    "fp8-cached-normal-packed-column-async-allocate-true",
    "fp8-cached-expanded-fp32-row-async-allocate-false",
    "fp8-cached-expanded-packed-column-async-allocate-true",
    "fp8-uncached-normal-fp32-row-async-allocate-false",
    "fp8-uncached-normal-packed-column-async-allocate-true",
    "fp8-uncached-expanded-fp32-row-async-allocate-false",
    "fp8-uncached-expanded-packed-column-async-allocate-true",
    "fp8-previous-event-allocate-true",
    "fp8-empty-route",
    "fp8-asymmetric-route",
    "fp8-10-generations",
    "fp8-cached-representation-changes",
    "fp8-completion-mismatch",
    "fp8-drop-event",
    "fp8-destroy-pending-retry",
)

WORK_COUNT_KEYS = {
    "input_tokens",
    "valid_routes",
    "received_records",
    "expanded_slots",
    "input_rows",
    "output_tokens",
    "hidden_elements",
    "topk_elements",
}


def _literal_work_counts(value=1):
    return {key: value for key in WORK_COUNT_KEYS}


def test_benchmark_device_timeout_defaults_and_allows_debug_override():
    assert _benchmark_timeout_seconds({}) == 300
    assert _benchmark_timeout_seconds({
        "DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS": "10",
    }) == 10
    with pytest.raises(ValueError, match="positive integer"):
        _benchmark_timeout_seconds({
            "DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS": "0",
        })
    with pytest.raises(ValueError, match="positive integer"):
        _benchmark_timeout_seconds({
            "DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS": "invalid",
        })


def test_case_matrix_matches_upstream_order_and_size():
    cases = enumerate_ep_mode_cases()
    case_id_sequence = "\n".join(case.case_id for case in cases)

    assert len(cases) == 144
    assert len({case.case_id for case in cases}) == 144
    assert hashlib.sha256(case_id_sequence.encode("ascii")).hexdigest() == (
        "eaf24b66d22f6f7ee6e293b5635006343d88c206d764b0a6cc6d21f826594c96"
    )
    assert cases[0].case_id == (
        "ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0")
    assert cases[-1].case_id == (
        "ep-bf16-align1-bias2-hcopy0-prev1-async1-alloc1")


def test_case_suite_counts_are_exhaustive():
    classified = [
        classify_ascend_case(case) for case in enumerate_ep_mode_cases()
    ]
    assert Counter(
        (classification.suite, classification.supported)
        for classification in classified
    ) == {
        ("performance", True): 144,
    }
    assert all(not classification.reason for classification in classified)


def test_fp8_async_matrix_promotes_all_full_rows():
    assert workloads.phase_3e1_acceptance_operations() == (
        "cached-bf16-dispatch-sync",
        "cached-bf16-dispatch-async",
        "bf16-combine-sync",
        "bf16-combine-async",
        "previous-event-with-comm-allocation",
        "comm-stream-allocation",
    )
    full_functional_rows = [
        case for case in enumerate_ep_mode_cases()
        if case_suite(case) == "functional"
    ]
    assert len(full_functional_rows) == 120
    assert sum(
        classify_ascend_case(case).supported
        for case in full_functional_rows
    ) == 120
    assert {case.use_fp8_dispatch for case in full_functional_rows} == {
        False, True,
    }
    assert all(
        classify_ascend_case(case).suite == "performance"
        for case in full_functional_rows
    )


def test_supported_ascend_cases_include_all_functional_modes():
    supported = [
        case
        for case in enumerate_ep_mode_cases()
        if classify_ascend_case(case).supported
    ]

    assert len(supported) == 144
    assert {case.use_fp8_dispatch for case in supported} == {False, True}
    functional = [case for case in supported if case_suite(case) == "functional"]
    assert len(functional) == 120
    assert {case.use_fp8_dispatch for case in functional} == {False, True}
    assert {case.with_previous_event for case in functional} == {False, True}
    assert {case.async_with_compute_stream for case in functional} == {False, True}
    assert {case.allocate_on_comm_stream for case in functional} == {False, True}
    assert {
        (case.expert_alignment, case.num_bias, case.do_handle_copy)
        for case in supported
    } == {
        (alignment, num_bias, do_handle_copy)
        for alignment in (128, 1)
        for num_bias in (0, 1, 2)
        for do_handle_copy in (True, False)
    }


def test_public_async_runner_declares_fp8_production_matrix():
    result = subprocess.run(
        [sys.executable, str(ASYNC_OVERLAP), "--contract"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert tuple(contract["fp8_async"]["case_names"]) == FP8_ASYNC_CASES
    assert contract["fp8_async"]["payload_dtype"] == "float8_e4m3fn"
    assert contract["fp8_async"]["scale_factor_dtypes"] == [
        "float32", "int32",
    ]
    assert contract["fp8_async"]["output_scale_layouts"] == [
        "row-major", "column-major",
    ]
    assert contract["fp8_async"]["dispatch_modes"] == [
        "cached", "non-cached",
    ]
    assert contract["fp8_async"]["completion"] == "native-event"
    assert contract["fp8_async"]["combine"] == "bf16-only"
    assert contract["fp8_async"]["supported_world_sizes"] == [2, 4, 8]


def test_runtime_launch_applies_predecessor_and_waits_async_event():
    calls = []
    waits = []
    predecessor = object()

    class Event:
        event = object()

        @staticmethod
        def current_stream_wait():
            waits.append("waited")

    class Buffer:
        @staticmethod
        def capture():
            calls.append(("capture", {}))
            return predecessor

        @staticmethod
        def dispatch(**arguments):
            calls.append(("dispatch", arguments))
            return ("x", "routes", "weights", "handle", Event())

    case = EPModeCase(
        do_handle_copy=True,
        expert_alignment=1,
        use_fp8_dispatch=False,
        num_bias=0,
        with_previous_event=True,
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
    )
    runtime = AscendRuntime.__new__(AscendRuntime)
    runtime.buffer = Buffer()
    arguments = {"x": "payload"}

    result = runtime._launch(case, "dispatch", arguments)

    assert result[:4] == ("x", "routes", "weights", "handle")
    assert calls == [
        ("capture", {}),
        ("dispatch", {"x": "payload", "previous_event": predecessor}),
    ]
    assert waits == ["waited"]
    assert arguments == {"x": "payload"}


def test_cached_dispatch_arguments_preserve_stream_modes():
    case = EPModeCase(
        do_handle_copy=True,
        expert_alignment=1,
        use_fp8_dispatch=False,
        num_bias=0,
        with_previous_event=False,
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
    )
    arguments = build_dispatch_arguments(
        case=case,
        x="payload",
        topk_idx="routes",
        topk_weights="weights",
        num_max_tokens_per_rank=64,
        num_experts=16,
        num_sms=1,
        num_qps=0,
    )

    cached = arguments.cached("handle")
    assert cached["async_with_compute_stream"] is True
    assert cached["allocate_on_comm_stream"] is True


class _ContractTensor:
    shape = (1, 3)

    def data_ptr(self):
        return 1

    def view(self, _dtype):
        return self

    def __getitem__(self, _key):
        return self

    def __lt__(self, _other):
        return self

    def __ge__(self, _other):
        return self

    def __eq__(self, _other):
        return self

    def __invert__(self):
        return self

    def masked_fill(self, _mask, _value):
        return self

    def to(self, _dtype):
        return self

    def argmax(self, dim):
        assert dim == 1
        return self

    def flatten(self):
        return self

    def float(self):
        return self

    def all(self):
        return self

    def item(self):
        return 1


class _CollapsedExpandedWeights(_ContractTensor):
    pass


class _GatheredExpandedWeights(_ContractTensor):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return _CollapsedExpandedWeights()
        return self


class _ExpandedWeights(_ContractTensor):
    def __getitem__(self, _key):
        return _GatheredExpandedWeights()


class _IndexableByteTensor(_ContractTensor):
    pass


class _NoIndexFloat8Tensor(_ContractTensor):
    def __getitem__(self, _key):
        raise AssertionError("FP8 payload was indexed without byte view")

    def view(self, dtype):
        assert dtype == "uint8"
        return _IndexableByteTensor()


class _ContractTorch:
    int32 = "int32"
    uint8 = "uint8"
    testing = SimpleNamespace(assert_close=lambda *_args, **_kwargs: None)

    @staticmethod
    def equal(left, right):
        if left is None or right is None:
            raise TypeError("equal() operands must be tensors")
        if isinstance(left, _CollapsedExpandedWeights):
            raise AssertionError("expanded weight lanes were collapsed")
        return True

    @staticmethod
    def argsort(_tensor):
        return _ContractTensor()

    @staticmethod
    def arange(_length, device):
        assert device == "npu"
        return _ContractTensor()


def _run_correctness_contract_check(expanded_weights=None, fp8=False):
    tensor = _ContractTensor()
    payload = _NoIndexFloat8Tensor() if fp8 else tensor
    expanded_weights = expanded_weights or tensor
    handle = SimpleNamespace(
        topk_idx=tensor,
        recv_src_metadata=tensor,
        psum_num_recv_tokens_per_expert=tensor,
    )
    event = SimpleNamespace(event=None)
    x = (payload, tensor) if fp8 else tensor
    normal = (x, tensor, tensor, handle, event)
    expanded = (x, None, expanded_weights, handle, event)
    cached = (x, tensor, None, handle, event)
    cached_expanded = (x, None, expanded_weights, handle, event)
    combined = (tensor, tensor, event)
    case = EPModeCase(
        do_handle_copy=False, expert_alignment=1, use_fp8_dispatch=fp8,
        num_bias=0, with_previous_event=False,
        async_with_compute_stream=False, allocate_on_comm_stream=False)
    runtime = AscendRuntime.__new__(AscendRuntime)
    runtime.torch = _ContractTorch()
    runtime.device = "npu"
    runtime.world_size = 1
    runtime.args = SimpleNamespace(allow_multiple_reduction=False)
    runtime.manifest = SimpleNamespace(
        spec=SimpleNamespace(num_experts=1),
    )

    runtime._check_case(
        case,
        tensor,
        tensor,
        normal,
        expanded,
        cached,
        cached_expanded,
        combined,
        combined,
        (((payload, tensor) if fp8 else tensor, tensor, tensor, tensor, None),
         tensor, tensor),
        num_recv_tokens=1,
    )


def test_correctness_check_accepts_missing_cached_weights_like_upstream():
    _run_correctness_contract_check()


def test_correctness_check_preserves_all_expanded_weight_lanes():
    _run_correctness_contract_check(_ExpandedWeights())


def test_correctness_check_indexes_fp8_payloads_as_bytes():
    _run_correctness_contract_check(fp8=True)


def test_summary_uses_linear_percentiles_and_decimal_gbps():
    summary = summarize_samples((1e-6, 2e-6, 3e-6, 4e-6))

    assert summary.minimum == 1e-6
    assert summary.mean == pytest.approx(2.5e-6)
    assert summary.p50 == pytest.approx(2.5e-6)
    assert summary.p95 == pytest.approx(3.85e-6)
    assert summary.maximum == 4e-6
    assert logical_gbps(1000, 2e-6) == 0.5


@pytest.mark.parametrize("samples", [(), (0.0,), (-1.0,), (float("nan"),)])
def test_summary_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        summarize_samples(samples)


def test_default_report_contains_all_current_cases(tmp_path):
    current_performance = tuple(
        case
        for case in enumerate_ep_mode_cases()
        if classify_ascend_case(case).supported
    )
    report = BenchmarkReport.empty_for_cases(
        platform="ascend",
        cases=current_performance,
        classify=classify_ascend_case,
        workload_fingerprint="a" * 64,
        world_size=2,
        allow_multiple_reduction=1,
    )
    output = tmp_path / "benchmark.json"

    write_report_atomic(output, report)
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 3
    assert payload["execution_protocol"] == {
        "allow_multiple_reduction": 1,
        "stage_profile": 0,
    }
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    expected_git_commit = (
        git_result.stdout.strip() if git_result.returncode == 0 else "unknown"
    )
    assert payload["git_commit"] == expected_git_commit
    assert len(payload["cases"]) == 144
    assert payload["case_summary"] == {
        "total": 144,
        "pending": 144,
        "passed": 0,
        "failed": 0,
    }


def test_performance_report_accepts_every_inventory_case():
    report = BenchmarkReport.empty_for_cases(
        platform="ascend",
        cases=tuple(enumerate_ep_mode_cases()),
        classify=classify_ascend_case,
        workload_fingerprint="a" * 64,
        world_size=2,
        allow_multiple_reduction=1,
    )

    assert len(report.cases) == 144


def test_comparison_rejects_incompatible_report_identity():
    cases = tuple(
        case
        for case in enumerate_ep_mode_cases()
        if classify_ascend_case(case).supported
    )
    left = BenchmarkReport.empty_for_cases(
        platform="cuda",
        cases=cases,
        classify=classify_ascend_case,
        workload_fingerprint="a" * 64,
        world_size=2,
        allow_multiple_reduction=1,
    )
    right = BenchmarkReport.empty_for_cases(
        platform="ascend",
        cases=cases,
        classify=classify_ascend_case,
        workload_fingerprint="b" * 64,
        world_size=2,
        allow_multiple_reduction=1,
    )

    with pytest.raises(ValueError, match="workload_fingerprint"):
        validate_comparable(left, right)


@pytest.mark.parametrize("allow_multiple_reduction", (0, 1))
def test_report_schema_v3_serializes_exact_execution_protocol(
    allow_multiple_reduction,
):
    report = BenchmarkReport.empty_for_cases(
        platform="ascend",
        cases=(),
        classify=classify_ascend_case,
        workload_fingerprint="a" * 64,
        world_size=2,
        allow_multiple_reduction=allow_multiple_reduction,
    )

    payload = report.to_dict()

    assert payload["schema_version"] == 3
    assert payload["execution_protocol"] == {
        "allow_multiple_reduction": allow_multiple_reduction,
        "stage_profile": 0,
    }


def test_report_comparison_rejects_execution_protocol_mismatch():
    reports = [
        BenchmarkReport.empty_for_cases(
            platform=platform,
            cases=(),
            classify=classify_ascend_case,
            workload_fingerprint="a" * 64,
            world_size=2,
            allow_multiple_reduction=allow_multiple_reduction,
        )
        for platform, allow_multiple_reduction in (
            ("cuda", 1),
            ("ascend", 0),
        )
    ]

    with pytest.raises(ValueError, match="execution_protocol"):
        validate_comparable(*reports)


def test_report_comparison_rejects_stage_profile_mismatch():
    reports = [
        BenchmarkReport.empty_for_cases(
            platform=platform,
            cases=(),
            classify=classify_ascend_case,
            workload_fingerprint="a" * 64,
            world_size=2,
            allow_multiple_reduction=1,
            stage_profile=stage_profile,
        )
        for platform, stage_profile in (("cuda", 0), ("ascend", 1))
    ]

    with pytest.raises(ValueError, match="execution_protocol"):
        validate_comparable(*reports)


def test_report_comparison_rejects_schema_v1_pair():
    reports = [
        BenchmarkReport.empty_for_cases(
            platform=platform,
            cases=(),
            classify=classify_ascend_case,
            workload_fingerprint="a" * 64,
            world_size=2,
            allow_multiple_reduction=1,
        )
        for platform in ("cuda", "ascend")
    ]
    for report in reports:
        report.schema_version = 1

    with pytest.raises(ValueError, match="schema_version"):
        validate_comparable(*reports)


def test_report_comparison_rejects_boolean_execution_protocol_value():
    reports = [
        BenchmarkReport.empty_for_cases(
            platform=platform,
            cases=(),
            classify=classify_ascend_case,
            workload_fingerprint="a" * 64,
            world_size=2,
            allow_multiple_reduction=1,
        )
        for platform in ("cuda", "ascend")
    ]
    for report in reports:
        report.execution_protocol["allow_multiple_reduction"] = True

    with pytest.raises(
        ValueError,
        match="execution_protocol.allow_multiple_reduction",
    ):
        validate_comparable(*reports)


def test_benchmark_parser_preserves_production_size_defaults():
    args = build_parser().parse_args([])

    assert args.num_tokens == 4096
    assert args.hidden == 7168
    assert args.num_topk == 6
    assert args.num_experts == 256
    assert args.warmups == 30
    assert args.iterations == 30
    assert args.allow_multiple_reduction == 1
    assert args.num_sms == 72
    assert args.profile_stages is False


def test_benchmark_parser_enables_stage_profile_explicitly():
    args = build_parser().parse_args(["--profile-stages"])

    assert args.profile_stages is True


def test_stage_profile_environment_is_enabled_only_on_request(monkeypatch):
    name = "DEEP_EP_ASCEND_PROFILE_STAGES"
    monkeypatch.setenv(name, "1")

    _configure_stage_profile_environment(False)
    assert name not in os.environ

    _configure_stage_profile_environment(True)
    assert os.environ[name] == "1"


def test_benchmark_parser_accepts_one_and_72_data_blocks():
    parser = build_parser()

    assert parser.parse_args(["--num-sms", "1"]).num_sms == 1
    assert parser.parse_args(["--num-sms", "72"]).num_sms == 72
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-sms", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-sms", "73"])


def test_default_selection_contains_all_current_supported_cases():
    selected = _selected_case_ids(build_parser(), None)

    assert len(selected) == 144
    assert any("-fp8-" in case_id for case_id in selected)
    assert any("-bf16-" in case_id for case_id in selected)
    assert any("-prev1-" in case_id for case_id in selected)
    assert any("-async1-" in case_id for case_id in selected)
    assert any("-alloc1" in case_id for case_id in selected)


@pytest.mark.parametrize(
    ("suite", "expected_count"),
    (
        ("all", 144),
        ("performance", 144),
        ("functional", 0),
    ),
)
def test_list_cases_filters_by_suite_without_runtime_imports(
    suite, expected_count,
):
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH_EP),
            "--list-cases",
            "--suite",
            suite,
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    payload = json.loads(result.stdout)

    assert len(payload["cases"]) == expected_count
    assert all(case["status"] == "supported" for case in payload["cases"])
    assert all(not case["reason"] for case in payload["cases"])


def test_cli_rejects_unknown_case_before_runtime_import():
    result = subprocess.run(
        [sys.executable, str(BENCH_EP), "--cases", "not-a-case"],
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 2
    assert "unknown case IDs: not-a-case" in result.stderr
    assert "torch_npu" not in result.stderr


def test_cli_lists_fp8_async_case_as_performance_without_runtime_import():
    case_id = "ep-fp8-align128-bias0-hcopy1-prev1-async0-alloc1"
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH_EP),
            "--list-cases",
            "--suite",
            "performance",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 0, result.stderr
    cases = {case["case_id"]: case for case in json.loads(result.stdout)["cases"]}
    assert cases[case_id]["status"] == "supported"
    assert cases[case_id]["reason"] == ""
    assert "torch_npu" not in result.stderr


class RecordingEvent:
    def __init__(self, backend, name):
        self.backend = backend
        self.name = name

    def record(self):
        self.backend.operations.append(f"{self.name}.record")

    def elapsed_time(self, other):
        assert other.name == "end"
        return self.backend.elapsed_ms


class RecordingEventBackend:
    def __init__(self, elapsed_ms):
        self.elapsed_ms = elapsed_ms
        self.operations = []

    def synchronize(self):
        self.operations.append("synchronize")

    def new_event(self, name):
        return RecordingEvent(self, name)


def test_npu_timer_synchronizes_and_returns_seconds():
    backend = RecordingEventBackend(elapsed_ms=1.25)
    timer = NpuEventTimer(backend)

    sample = timer.measure(lambda: backend.operations.append("operation"))

    assert sample.device_seconds == 0.00125
    assert sample.wall_seconds > 0
    assert backend.operations == [
        "synchronize",
        "start.record",
        "operation",
        "end.record",
        "synchronize",
    ]


def test_runtime_source_pins_supported_ascend_contract():
    source = RUNTIME.read_text()

    assert 'backend="hccl"' in source
    assert 'torch.device("npu", local_rank)' in source
    assert "allow_hybrid_mode=False" in source
    assert "explicitly_destroy=True" in source
    assert "num_sms=args.num_sms" in source
    assert '"num_sms": args.num_sms' in source
    assert "num_qps=0" in source
    assert source.index("buffer.destroy()") < source.index(
        "dist.destroy_process_group()")


def test_timed_handle_operations_prepare_current_handles_outside_measurement():
    class Handle:
        def __init__(self, generation):
            self.generation = generation

    class Buffer:
        def barrier(self, **_kwargs):
            return None

    state = {"generation": 0, "prepared": [], "measured": []}
    handles = {
        "cached": Handle(0),
        "combine": Handle(0),
        "reduced": Handle(0),
    }

    def fresh(operation_id):
        def operation():
            state["generation"] += 1
            state["measured"].append(operation_id)
        return operation

    def handle_operation(operation_id, slot, updates_generation=False):
        def operation():
            assert handles[slot].generation == state["generation"], (
                f"{operation_id} received stale handle")
            state["measured"].append(operation_id)
            if updates_generation:
                state["generation"] += 1
                handles[slot].generation = state["generation"]
        return operation

    def prepare(slot):
        def operation():
            state["generation"] += 1
            handles[slot] = Handle(state["generation"])
            state["prepared"].append(slot)
        return operation

    prepared = PreparedCase(
        case=SimpleNamespace(case_id="stateful"), x=None, topk_idx=None,
        topk_weights=None, bias=None,
        launches={
            "dispatch": fresh("dispatch"),
            "expanded_dispatch": fresh("expanded_dispatch"),
            "cached_dispatch": handle_operation(
                "cached_dispatch", "cached", updates_generation=True),
            "combine": handle_operation("combine", "combine"),
            "reduced_combine": handle_operation("reduced_combine", "reduced"),
        },
        prepare_launches={},
        traffic={operation_id: {} for operation_id in (
            "dispatch", "expanded_dispatch", "cached_dispatch", "combine",
            "reduced_combine")},
        work_counts={
            operation_id: _literal_work_counts()
            for operation_id in (
                "dispatch", "expanded_dispatch", "cached_dispatch", "combine",
                "reduced_combine")
        },
    )
    prepared.prepare_launches = {
        "cached_dispatch": prepare("cached"),
        "combine": prepare("combine"),
        "reduced_combine": prepare("reduced"),
    }
    runtime = AscendRuntime.__new__(AscendRuntime)
    runtime.buffer = Buffer()
    runtime.args = SimpleNamespace(
        warmups=0, iterations=1, profile_stages=False)
    runtime._prepare_case = lambda _case: prepared
    runtime.synchronized_step = lambda operation, _label: operation()
    runtime.timer = SimpleNamespace(
        measure=lambda operation: (
            operation() or SimpleNamespace(device_seconds=1.0, wall_seconds=1.0)))

    runtime.run_case(prepared.case)

    assert state["prepared"] == ["cached", "combine", "reduced"]
    assert state["measured"] == [
        "dispatch", "expanded_dispatch", "cached_dispatch", "combine",
        "reduced_combine",
    ]


def test_stage_profile_capture_runs_outside_event_timing():
    events = []
    inside_timing = {"value": False}

    class Buffer:
        def barrier(self, **_kwargs):
            events.append("barrier")

        def reset_stage_profile(self):
            assert not inside_timing["value"]
            events.append("reset")

        def get_stage_profile(self):
            assert not inside_timing["value"]
            events.append("read")
            return {
                "available": True,
                "operation": "dispatch",
                "generation": 7,
                "completion_generation": 7,
                "stages": [],
                "phase_cycles": {},
            }

    def launch():
        events.append(
            "timed-launch" if inside_timing["value"] else "profile-launch")

    def measure(operation):
        inside_timing["value"] = True
        try:
            operation()
        finally:
            inside_timing["value"] = False
        return SimpleNamespace(device_seconds=1.0, wall_seconds=1.0)

    operation_ids = (
        "dispatch", "expanded_dispatch", "cached_dispatch", "combine",
        "reduced_combine",
    )
    prepared = PreparedCase(
        case=SimpleNamespace(case_id="profiled"),
        x=None,
        topk_idx=None,
        topk_weights=None,
        bias=None,
        launches={operation_id: launch for operation_id in operation_ids},
        prepare_launches={},
        traffic={operation_id: {} for operation_id in operation_ids},
        work_counts={
            operation_id: _literal_work_counts()
            for operation_id in operation_ids
        },
    )
    runtime = AscendRuntime.__new__(AscendRuntime)
    runtime.buffer = Buffer()
    runtime.args = SimpleNamespace(
        warmups=0, iterations=1, profile_stages=True)
    runtime._prepare_case = lambda _case: prepared
    runtime.synchronized_step = lambda operation, _label: operation()
    runtime.timer = SimpleNamespace(measure=measure)
    runtime.torch = SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: events.append("synchronize")))

    records = runtime.run_case(prepared.case)

    assert all(record["stage_profile"]["available"] for record in records)
    assert events.count("timed-launch") == len(operation_ids)
    assert events.count("profile-launch") == len(operation_ids)
    assert events.count("reset") == len(operation_ids)
    assert events.count("read") == len(operation_ids)
    assert events.count("synchronize") == len(operation_ids)
    assert all(record["work_counts"] == _literal_work_counts()
               for record in records)
    assert all(record["logical_byte_components"] == {}
               for record in records)


def test_work_counts_and_logical_components_use_rank_max_and_byte_sum():
    ranks = []
    for rank in range(2):
        ranks.append([
            {
                "operation_id": operation_id,
                "device_samples": [0.001 + rank * 0.0001],
                "wall_samples": [0.002 + rank * 0.0001],
                "logical_bytes": {"scaleup": 100 + rank},
                "logical_byte_components": {"scaleup": 100 + rank},
                "work_counts": _literal_work_counts(7 + rank),
            }
            for operation_id in (
                "dispatch", "expanded_dispatch", "cached_dispatch", "combine",
                "reduced_combine",
            )
        ])

    operations = _aggregate_rank_operations(ranks)

    assert len(operations) == 5
    assert operations[0]["work_counts"] == _literal_work_counts(8)
    assert operations[0]["logical_bytes"] == {"scaleup": 201}
    assert operations[0]["logical_byte_components"] == {"scaleup": 201}
    assert operations[0]["per_rank"][0]["work_counts"] == (
        _literal_work_counts(7)
    )


def test_operation_work_counts_describe_literal_dispatch_and_combine_rows():
    counts = _build_operation_work_counts(
        SimpleNamespace(num_tokens=8, hidden=16, num_topk=2),
        valid_routes=12,
        normal_input_rows=8,
        normal_received_records=10,
        expanded_slots=20,
        combine_input_rows=10,
        reduced_input_rows=20,
    )

    assert counts["dispatch"] == {
        "input_tokens": 8,
        "valid_routes": 12,
        "received_records": 10,
        "expanded_slots": 0,
        "input_rows": 8,
        "output_tokens": 10,
        "hidden_elements": 128,
        "topk_elements": 16,
    }
    assert counts["expanded_dispatch"] == (
        counts["dispatch"] | {"expanded_slots": 20, "output_tokens": 20}
    )
    assert counts["cached_dispatch"] == counts["dispatch"]
    assert counts["combine"] == {
        "input_tokens": 8,
        "valid_routes": 12,
        "received_records": 10,
        "expanded_slots": 0,
        "input_rows": 10,
        "output_tokens": 8,
        "hidden_elements": 160,
        "topk_elements": 16,
    }
    assert counts["reduced_combine"] == (
        counts["combine"] | {
            "expanded_slots": 20,
            "input_rows": 20,
            "hidden_elements": 320,
        }
    )


def test_payload_rows_uses_fp8_payload_instead_of_scale_tuple():
    bf16 = SimpleNamespace(shape=(8, 7168))
    fp8_payload = SimpleNamespace(shape=(9, 7168))
    fp8_scales = SimpleNamespace(shape=(9, 56))

    assert _payload_rows(bf16) == 8
    assert _payload_rows((fp8_payload, fp8_scales)) == 9


def test_early_route_plan_observability_counts_records_and_expert_lanes():
    manifest = BenchmarkManifest(
        schema_version=1,
        generator_version=1,
        spec=WorkloadSpec(
            world_size=2,
            num_tokens=2,
            hidden=7168,
            num_topk=4,
            num_experts=8,
        ),
        ranks=(
            RankWorkload(
                rank=0,
                num_tokens=2,
                topk_idx=((0, 0, 4, -1), (1, 5, 6, 5)),
                topk_weights=((1.0,) * 4, (1.0,) * 4),
            ),
            RankWorkload(
                rank=1,
                num_tokens=1,
                topk_idx=((4, 0, -1, -1),),
                topk_weights=((1.0,) * 4,),
            ),
        ),
        fingerprint="literal",
    )

    observed = _early_route_plan_observability(manifest, rank=0, enabled=True)

    assert observed == {
        "enabled": True,
        "signal_index": 2,
        "slot_bytes": 1056,
        "window_bytes": 2112,
        "published_rank_counts": [2, 2],
        "published_expert_counts": [
            [2, 1, 0, 0],
            [1, 2, 1, 0],
        ],
        "commands": {
            "remote_put": 1,
            "flush": 2,
            "remote_signal": 1,
        },
        "kernel_checks": {
            "route_payload_count_parity": True,
            "route_payload_generation_parity": True,
        },
    }


def test_fp8_materialization_pads_partial_scale_group_and_restores_payload_width():
    pad_calls = []

    class ShapeTensor:
        def __init__(self, shape, dtype=None):
            self.shape = tuple(shape)
            self.dtype = dtype

        def float(self):
            return ShapeTensor(self.shape, "float32")

        def reshape(self, *shape):
            resolved = list(shape)
            if resolved.count(-1) == 1:
                known = 1
                for value in resolved:
                    if value != -1:
                        known *= value
                total = 1
                for value in self.shape:
                    total *= value
                resolved[resolved.index(-1)] = total // known
            return ShapeTensor(resolved, self.dtype)

        def abs(self):
            return ShapeTensor(self.shape, self.dtype)

        def amax(self, dim):
            return ShapeTensor(
                self.shape[:dim] + self.shape[dim + 1:], self.dtype)

        def clamp(self, **_kwargs):
            return ShapeTensor(self.shape, self.dtype)

        def unsqueeze(self, dim):
            shape = list(self.shape)
            shape.insert(dim, 1)
            return ShapeTensor(shape, self.dtype)

        def to(self, dtype):
            return ShapeTensor(self.shape, dtype)

        def narrow(self, dim, _start, length):
            shape = list(self.shape)
            shape[dim] = length
            return ShapeTensor(shape, self.dtype)

        def contiguous(self):
            return self

        def __truediv__(self, _other):
            return ShapeTensor(self.shape, self.dtype)

    class FakeFunctional:
        @staticmethod
        def pad(tensor, padding):
            pad_calls.append(padding)
            return ShapeTensor(
                (tensor.shape[0], tensor.shape[1] + padding[1]), tensor.dtype)

    class FakeTorch:
        bfloat16 = "bfloat16"
        float8_e4m3fn = "float8_e4m3fn"
        int64 = "int64"
        float32 = "float32"
        nn = SimpleNamespace(functional=FakeFunctional())

        @staticmethod
        def randn(shape, dtype, device):
            assert device == "npu"
            return ShapeTensor(shape, dtype)

        @staticmethod
        def tensor(values, dtype, device):
            assert device == "npu"
            return ShapeTensor((len(values), len(values[0])), dtype)

    case = EPModeCase(
        do_handle_copy=True,
        expert_alignment=128,
        use_fp8_dispatch=True,
        num_bias=0,
        with_previous_event=False,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )
    runtime = AscendRuntime.__new__(AscendRuntime)
    runtime.torch = FakeTorch()
    runtime.device = "npu"
    runtime.rank = 0
    runtime.manifest = SimpleNamespace(
        spec=SimpleNamespace(hidden=7184, num_topk=2),
        ranks=(SimpleNamespace(
            num_tokens=2,
            topk_idx=((0, 1), (1, -1)),
            topk_weights=((0.75, 0.25), (1.0, 0.0)),
        ),),
    )

    x, topk_idx, topk_weights, bias = runtime._materialize(case)

    assert x[0].shape == (2, 7184)
    assert x[1].shape == (2, 57)
    assert topk_idx.shape == (2, 2)
    assert topk_weights.shape == (2, 2)
    assert bias is None
    assert pad_calls == [(0, 112)]


def _literal_stage_profile(rank, *, generation=9, operation="dispatch"):
    starts = (100 + rank * 10, 90 + rank * 10)
    ends = (130 + rank * 10, 150 + rank * 10)
    return {
        "available": True,
        "abi_version": 1,
        "operation": operation,
        "generation": generation,
        "completion_generation": generation,
        "stages": [{
            "id": 1,
            "name": "producer_control",
            "block_count": 2,
            "blocks": [
                {"block": block, "start": start, "end": end}
                for block, (start, end) in enumerate(zip(starts, ends))
            ],
        }],
        "phase_cycles": {
            "producer": 60 + rank * 10,
            "publication": 20,
            "service_submit": 30,
            "cq_wait": 10,
            "barrier_wait": 20,
            "consumer_wait": 40,
            "consumer_compute": 50,
            "epilogue": 10,
        },
        "service": {
            "start": 100,
            "end": 200 + rank * 10,
            "cycles": 100 + rank * 10,
            "wait_cycles": 10,
            "payload_command_cycles": 40 + rank * 5,
            "control_command_cycles": 15,
            "flush_command_cycles": 9,
            "barrier_command_cycles": 30,
            "barrier_poll_cycles": 20 + rank * 2,
        },
    }


@pytest.mark.parametrize(
    ("operation_id", "raw_name", "stage_id"),
    (
        ("dispatch", "producer_control", "D0"),
        ("expanded_dispatch", "producer_group", "D1"),
        ("cached_dispatch", "producer_prefix", "D2"),
        ("dispatch", "producer_record", "D3"),
        ("dispatch", "release_payload", "D4"),
        ("dispatch", "release_control", "D4"),
        ("dispatch", "release_barrier", "D4"),
        ("dispatch", "epilogue_acquire", "D5"),
        ("dispatch", "epilogue_validate", "D5"),
        ("dispatch", "epilogue_validate_reduce", "D5"),
        ("dispatch", "epilogue_expert_count", "D5"),
        ("dispatch", "epilogue_expert_prefix", "D6"),
        ("dispatch", "epilogue_metadata", "D7"),
        ("dispatch", "epilogue_copy", "D8"),
        ("dispatch", "epilogue_complete", "F0"),
        ("combine", "producer_control", "C0"),
        ("reduced_combine", "producer_plan", "C1"),
        ("combine", "producer_plan_prefix", "C1"),
        ("combine", "producer_record", "C2"),
        ("combine", "producer_local_copy", "C3"),
        ("combine", "release_payload", "C4"),
        ("combine", "release_control", "C4"),
        ("combine", "release_barrier", "C4"),
        ("combine", "epilogue_acquire", "C5"),
        ("combine", "epilogue_validate", "C5"),
        ("combine", "epilogue_validate_reduce", "C5"),
        ("combine", "epilogue_reduce", "C6"),
        ("combine", "epilogue_weights", "C7"),
        ("combine", "epilogue_complete", "F0"),
    ),
)
def test_stage_semantic_maps_raw_runtime_stages_to_stable_ids(
    operation_id, raw_name, stage_id,
):
    semantic = stage_semantic(operation_id, raw_name)

    assert semantic.stage_id == stage_id
    assert semantic.short_name
    assert semantic.ascend_functions
    assert semantic.cuda_counterpart
    assert isinstance(semantic.work_count_keys, tuple)


def test_stage_semantic_marks_combine_local_staging_as_independently_timed():
    semantic = stage_semantic("combine", "producer_local_copy")

    assert semantic.stage_id == "C3"
    assert semantic.independently_timed is True
    assert semantic.ascend_functions == (
        "direct_combine_producer_local_copy_vf",
    )


@pytest.mark.parametrize(
    ("operation_id", "raw_name"),
    (("dispatch", "mystery"), ("combine", "mystery"), ("unknown", "full")),
)
def test_stage_semantic_rejects_unknown_operations_and_stages(
    operation_id, raw_name,
):
    with pytest.raises(ValueError, match="stage semantic"):
        stage_semantic(operation_id, raw_name)


def test_stage_profile_rank_aggregation_derives_literal_block_span():
    aggregated = _aggregate_stage_profiles(
        "dispatch", [_literal_stage_profile(0), _literal_stage_profile(1)])

    assert aggregated["operation"] == "dispatch"
    assert aggregated["generation"] == 9
    assert aggregated["stage_spans_cycles"] == {"producer_control": 60}
    assert aggregated["phase_cycles"]["producer"] == 70
    assert aggregated["phase_cycles"]["barrier_wait"] == 20
    assert aggregated["pipeline_cycles"]["network"] == 80
    assert aggregated["service_cycles"] == {
        "cycles": 110,
        "wait_cycles": 10,
        "payload_command_cycles": 45,
        "control_command_cycles": 15,
        "flush_command_cycles": 9,
        "barrier_command_cycles": 30,
        "barrier_poll_cycles": 22,
    }
    assert "barrier_diagnostics" not in aggregated
    assert aggregated["optimistic_speedup_ceiling"] == pytest.approx(2.5)
    assert aggregated["per_rank"][0]["stages"][0] == {
        "id": 1,
        "name": "producer_control",
        "block_count": 2,
        "blocks": [
            {"block": 0, "start": 100, "end": 130},
            {"block": 1, "start": 90, "end": 150},
        ],
        "stage_id": "D0",
        "short_name": "control",
        "ascend_functions": ("direct_dispatch_producer_control_vf",),
        "cuda_counterpart": (
            "dispatch_impl prologue and notify-warps setup"
        ),
        "work_count_keys": ("input_tokens",),
        "independently_timed": True,
        "start": 90,
        "end": 150,
        "span_cycles": 60,
    }


def test_stage_profile_rank_aggregation_reports_barrier_diagnostics():
    profiles = [_literal_stage_profile(0), _literal_stage_profile(1)]
    for rank, profile in enumerate(profiles):
        profile["service"]["barrier_diagnostics"] = {
            "issue_cycles": 100 + rank,
            "drain_cycles": 200 + rank,
            "poll_iterations": 3 + rank,
            "peer_count": 7,
            "first_observation_cycles": 20 + rank,
            "completion_cycles": 4 + rank,
            "poll_elapsed_cycles": 500 + rank,
        }

    aggregated = _aggregate_stage_profiles("dispatch", profiles)

    assert aggregated["barrier_diagnostics"] == {
        "issue_cycles": 101,
        "drain_cycles": 201,
        "poll_iterations": 4,
        "peer_count": 7,
        "first_observation_cycles": 21,
        "completion_cycles": 5,
        "poll_elapsed_cycles": 501,
    }


def test_stage_timeline_derives_idle_and_overlap_from_literal_intervals():
    stages = [
        {
            "name": "producer",
            "start": 90,
            "end": 150,
            "span_cycles": 60,
        },
        {
            "name": "release",
            "start": 170,
            "end": 200,
            "span_cycles": 30,
        },
        {
            "name": "consumer",
            "start": 190,
            "end": 230,
            "span_cycles": 40,
        },
    ]

    assert _derive_stage_timeline(stages) == {
        "start": 90,
        "end": 230,
        "envelope_cycles": 140,
        "active_cycles": 120,
        "idle_cycles": 20,
        "overlap_cycles": 10,
    }


def test_host_envelope_overhead_uses_paired_wall_and_device_samples():
    assert _derive_host_envelope_samples(
        wall_samples=[0.002, 0.004],
        device_samples=[0.0015, 0.003],
    ) == pytest.approx([0.0005, 0.001])


def test_stage_profile_rank_aggregation_maps_full_only_span_to_a_phase():
    profile = _literal_stage_profile(0)
    profile["stages"] = [{
        "id": 0,
        "name": "full",
        "block_count": 1,
        "blocks": [{"block": 0, "start": 100, "end": 240}],
    }]
    profile["phase_cycles"] = {
        "producer": 0,
        "publication": 0,
        "service_submit": 0,
        "cq_wait": 0,
        "barrier_wait": 0,
        "consumer_wait": 0,
        "consumer_compute": 0,
        "epilogue": 0,
    }

    aggregated = _aggregate_stage_profiles("dispatch", [profile])

    assert aggregated["stage_spans_cycles"] == {"full": 140}
    assert aggregated["per_rank"][0]["stages"][0]["stage_id"] == "FULL"
    assert aggregated["per_rank"][0]["stages"][0]["cuda_counterpart"] == ""
    assert aggregated["phase_cycles"]["producer"] == 140
    assert aggregated["pipeline_cycles"] == {
        "producer": 140,
        "network": 0,
        "consumer": 0,
    }
    assert aggregated["optimistic_speedup_ceiling"] == pytest.approx(1.0)


def test_stage_profile_rank_aggregation_rejects_unmapped_runtime_stage():
    profile = _literal_stage_profile(0)
    profile["stages"][0]["name"] = "mystery"

    with pytest.raises(ValueError, match="stage semantic"):
        _aggregate_stage_profiles("dispatch", [profile])


def test_stage_profile_rank_aggregation_includes_host_timeline_maxima():
    profiles = [_literal_stage_profile(0), _literal_stage_profile(1)]
    profiles[0]["host_timeline_ns"] = {
        "dispatch_synchronize": 100,
        "dispatch_completion_wait": 300,
        "total": 400,
    }
    profiles[1]["host_timeline_ns"] = {
        "dispatch_synchronize": 120,
        "dispatch_completion_wait": 250,
        "total": 370,
    }

    aggregated = _aggregate_stage_profiles("dispatch", profiles)

    assert aggregated["host_timeline_ns"] == {
        "dispatch_synchronize": 120,
        "dispatch_completion_wait": 300,
        "total": 400,
    }


def test_stage_profile_exposes_only_semantically_relevant_work_counts():
    counts = _literal_work_counts(3)

    aggregated = _aggregate_stage_profiles(
        "dispatch", [_literal_stage_profile(0)], rank_work_counts=[counts])

    stage = aggregated["per_rank"][0]["stages"][0]
    assert stage["stage_id"] == "D0"
    assert stage["work_counts"] == {"input_tokens": 3}


@pytest.mark.parametrize(
    ("generation", "operation", "match"),
    ((10, "dispatch", "generation"), (9, "combine", "operation")),
)
def test_stage_profile_rank_aggregation_rejects_mismatched_identity(
    generation, operation, match,
):
    profiles = [
        _literal_stage_profile(0),
        _literal_stage_profile(
            1, generation=generation, operation=operation),
    ]

    with pytest.raises(ValueError, match=match):
        _aggregate_stage_profiles("dispatch", profiles)


def test_stage_profile_rank_aggregation_reports_unavailable_reason_first():
    command_metrics = {
        "command_count": 3,
        "put_command_count": 7,
    }
    service = {"start": 100, "end": 200}
    profiles = [
        {
            "available": False,
            "reason": "stale_generation",
            "command_metrics": command_metrics,
            "service": service,
        },
        _literal_stage_profile(1),
    ]

    with pytest.raises(ValueError) as error:
        _aggregate_stage_profiles("dispatch", profiles)
    message = str(error.value)
    assert "stage profile unavailable on rank 0: stale_generation" in message
    assert repr(command_metrics) in message
    assert repr(service) in message


def _literal_timeline_operation(operation_id, raw_stages):
    stages = []
    for index, (raw_name, stage_id) in enumerate(raw_stages):
        start = 100 + index * 10
        stages.append({
            "name": raw_name,
            "stage_id": stage_id,
            "start": start,
            "end": start + 5,
            "span_cycles": 5,
        })
    return {
        "operation_id": operation_id,
        "logical_byte_components": {"scaleup": 100},
        "work_counts": _literal_work_counts(7),
        "per_rank": [{
            "rank": 0,
            "logical_byte_components": {"scaleup": 100},
            "work_counts": _literal_work_counts(7),
        }],
        "stage_profile": {
            "device_timeline_cycles": {
                "envelope_cycles": len(stages) * 10 - 5,
                "active_cycles": len(stages) * 5,
                "idle_cycles": (len(stages) - 1) * 5,
                "overlap_cycles": 0,
            },
            "host_timeline_ns": {"total": 2_000_000},
            "per_rank": [{
                "rank": 0,
                "host_timeline_ns": {"total": 2_000_000},
                "device_timeline_cycles": {
                    "start": 100,
                    "end": 100 + len(stages) * 10 - 5,
                    "envelope_cycles": len(stages) * 10 - 5,
                    "active_cycles": len(stages) * 5,
                    "idle_cycles": (len(stages) - 1) * 5,
                    "overlap_cycles": 0,
                },
                "stages": stages,
            }],
        },
    }


def _literal_timeline_report():
    dispatch_stages = (
        ("producer_control", "D0"),
        ("producer_group", "D1"),
        ("producer_prefix", "D2"),
        ("producer_record", "D3"),
        ("release_payload", "D4"),
        ("epilogue_acquire", "D5"),
        ("epilogue_expert_prefix", "D6"),
        ("epilogue_metadata", "D7"),
        ("epilogue_copy", "D8"),
        ("epilogue_complete", "F0"),
    )
    combine_stages = (
        ("producer_control", "C0"),
        ("producer_plan", "C1"),
        ("producer_record", "C2"),
        ("producer_local_copy", "C3"),
        ("release_payload", "C4"),
        ("epilogue_acquire", "C5"),
        ("epilogue_reduce", "C6"),
        ("epilogue_weights", "C7"),
        ("epilogue_complete", "F0"),
    )
    return {
        "schema_version": 3,
        "platform": "ascend",
        "git_commit": "abc123",
        "world_size": 1,
        "workload_fingerprint": "f" * 64,
        "execution_protocol": {
            "allow_multiple_reduction": 1,
            "stage_profile": 1,
        },
        "cases": [{
            "case_id": "representative",
            "status": "passed",
            "operations": [
                _literal_timeline_operation("combine", combine_stages),
                _literal_timeline_operation("dispatch", dispatch_stages),
            ],
        }],
    }


def test_operation_stage_semantics_include_virtual_c3_in_stable_order():
    assert tuple(
        semantic.stage_id
        for semantic in operation_stage_semantics("combine")
    ) == ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "F0")


def test_timeline_report_builds_deterministic_per_rank_stage_rows():
    timeline = build_timeline_report(_literal_timeline_report())

    assert timeline["timeline_schema_version"] == 1
    assert timeline["source"] == {
        "schema_version": 3,
        "platform": "ascend",
        "git_commit": "abc123",
        "world_size": 1,
        "workload_fingerprint": "f" * 64,
    }
    assert [row["stage_id"] for row in timeline["rows"]] == [
        "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "F0",
        "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "F0",
    ]
    assert timeline["rows"][0] == {
        "case_id": "representative",
        "operation_id": "dispatch",
        "rank": 0,
        "stage_id": "D0",
        "short_name": "control",
        "measurement_status": "measured",
        "raw_stages": ["producer_control"],
        "start_cycles": 100,
        "end_cycles": 105,
        "envelope_cycles": 5,
        "active_cycles": 5,
        "idle_cycles": 0,
        "overlap_cycles": 0,
        "operation_timeline_cycles": {
            "start": 100,
            "end": 195,
            "envelope_cycles": 95,
            "active_cycles": 50,
            "idle_cycles": 45,
            "overlap_cycles": 0,
        },
        "host_timeline_ns": {"total": 2_000_000},
        "work_counts": {"input_tokens": 7},
        "logical_byte_components": {"scaleup": 100},
        "ascend_functions": ["direct_dispatch_producer_control_vf"],
        "cuda_counterpart": "dispatch_impl prologue and notify-warps setup",
    }
    c3 = next(row for row in timeline["rows"] if row["stage_id"] == "C3")
    assert c3["measurement_status"] == "measured"
    assert c3["start_cycles"] == 130
    assert c3["end_cycles"] == 135
    assert c3["raw_stages"] == ["producer_local_copy"]


def test_timeline_markdown_keeps_cycles_and_formats_host_nanoseconds():
    markdown = render_timeline_markdown(_literal_timeline_report())

    assert "| Case | Operation | Rank | Stage |" in markdown
    assert "| representative | dispatch | 0 | D0 control |" in markdown
    assert "100-105 cycles" in markdown
    assert "2.000 ms" in markdown
    assert "direct_dispatch_producer_control_vf" in markdown
    assert "dispatch_impl prologue and notify-warps setup" in markdown
    assert "device ms" not in markdown


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda report: report["execution_protocol"].update(stage_profile=0),
         "stage_profile"),
        (lambda report: report["cases"][0]["operations"][0][
            "stage_profile"].pop("per_rank"), "rank profiles"),
        (lambda report: report["cases"][0]["operations"][0][
            "stage_profile"]["per_rank"][0]["stages"][0].update(
                stage_id="D0"), "stage ID"),
        (lambda report: report["cases"][0]["operations"][0][
            "stage_profile"]["per_rank"][0]["stages"][0].update(end=99),
         "interval"),
        (lambda report: report["cases"][0]["operations"][0].pop(
            "work_counts"), "work counts"),
    ),
)
def test_timeline_report_rejects_incomplete_or_inconsistent_profiles(
    mutate, match,
):
    report = _literal_timeline_report()
    mutate(report)

    with pytest.raises(ValueError, match=match):
        build_timeline_report(report)


def test_timeline_report_cli_emits_json(tmp_path):
    report_path = tmp_path / "profile.json"
    report_path.write_text(json.dumps(_literal_timeline_report()))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.ascend.benchmark.timeline_report",
            str(report_path),
            "--format",
            "json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"][0]["stage_id"] == "D0"


def test_fp8_empty_input_case_requests_exact_column_major_output():
    tree = ast.parse((ROOT / "tests/ascend/production/run_fp8_dispatch_combine.py").read_text())
    run = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    specs = next(node.value for node in ast.walk(run)
                 if isinstance(node, ast.Assign) and
                 any(isinstance(target, ast.Name) and target.id == "specs"
                     for target in node.targets))
    empty_case = next(value for key, value in zip(specs.keys, specs.values)
                      if isinstance(key, ast.Constant) and key.value == "empty-input")
    column_major = next((keyword.value.value for keyword in empty_case.keywords
                         if keyword.arg == "column_major_output"), False)
    assert column_major is True


def _report_fixture(
    platform,
    *,
    mean=2e-6,
    gbps=100.0,
    fingerprint="a" * 64,
    iterations=30,
    logical_bytes=200_000,
):
    return {
        "schema_version": 3,
        "formula_version": 1,
        "platform": platform,
        "world_size": 2,
        "workload_fingerprint": fingerprint,
        "execution_protocol": {
            "allow_multiple_reduction": 1,
            "stage_profile": 0,
        },
        "timing_protocol": {
            "timer": f"{platform}_event",
            "warmups": 30,
            "iterations": iterations,
            "rank_aggregation": "maximum_latency",
            "logical_byte_aggregation": "sum",
        },
        "cases": [{
            "case_id": (
                "ep-bf16-align1-bias0-hcopy0-prev0-async0-alloc0"
            ),
            "status": "passed",
            "operations": [{
                "operation_id": "dispatch",
                "formula_version": 1,
                "device_seconds": {
                    "mean": mean,
                    "p50": mean,
                    "p95": mean * 1.1,
                },
                "logical_bytes": {"scaleup": logical_bytes},
                "logical_gbps": gbps,
            }],
        }],
    }


def test_compare_reports_computes_latency_and_bandwidth_ratios():
    rows = compare_reports(
        _report_fixture("cuda", mean=2e-6, gbps=100.0),
        _report_fixture("ascend", mean=4e-6, gbps=50.0),
    )

    assert len(rows) == 1
    assert rows[0]["latency_ratio_ascend_over_cuda"] == 2.0
    assert rows[0]["bandwidth_ratio_ascend_over_cuda"] == 0.5


def test_compare_reports_rejects_workload_mismatch():
    with pytest.raises(ValueError, match="workload_fingerprint"):
        compare_reports(
            _report_fixture("cuda", fingerprint="a" * 64),
            _report_fixture("ascend", fingerprint="b" * 64),
        )


def test_compare_reports_rejects_timing_protocol_mismatch():
    with pytest.raises(ValueError, match="timing_protocol.iterations"):
        compare_reports(
            _report_fixture("cuda", iterations=30),
            _report_fixture("ascend", iterations=1),
        )


def test_compare_reports_rejects_execution_protocol_mismatch():
    cuda = _report_fixture("cuda")
    ascend = _report_fixture("ascend")
    ascend["execution_protocol"]["allow_multiple_reduction"] = 0

    with pytest.raises(ValueError, match="execution_protocol"):
        compare_reports(cuda, ascend)


def test_compare_reports_rejects_schema_v1_pair():
    cuda = _report_fixture("cuda")
    ascend = _report_fixture("ascend")
    cuda["schema_version"] = 1
    ascend["schema_version"] = 1

    with pytest.raises(ValueError, match="schema_version"):
        compare_reports(cuda, ascend)


def test_compare_reports_rejects_boolean_execution_protocol_value():
    cuda = _report_fixture("cuda")
    ascend = _report_fixture("ascend")
    cuda["execution_protocol"]["allow_multiple_reduction"] = True
    ascend["execution_protocol"]["allow_multiple_reduction"] = True

    with pytest.raises(
        ValueError,
        match="execution_protocol.allow_multiple_reduction",
    ):
        compare_reports(cuda, ascend)


def test_compare_reports_rejects_logical_byte_mismatch():
    with pytest.raises(ValueError, match="logical_bytes"):
        compare_reports(
            _report_fixture("cuda", logical_bytes=200_000),
            _report_fixture("ascend", logical_bytes=100_000),
        )


def test_compare_cli_table_displays_mean_p50_and_p95(tmp_path):
    cuda_path = tmp_path / "cuda.json"
    ascend_path = tmp_path / "ascend.json"
    cuda_path.write_text(json.dumps(_report_fixture("cuda")))
    ascend_path.write_text(json.dumps(_report_fixture("ascend")))

    result = subprocess.run(
        [sys.executable, str(COMPARE), str(cuda_path), str(ascend_path)],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )

    assert "CUDA mean/p50/p95 us" in result.stdout
    assert "Ascend mean/p50/p95 us" in result.stdout
