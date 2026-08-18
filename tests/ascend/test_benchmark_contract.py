import json
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
from tests.ascend.benchmark.runtime import AscendRuntime
from tests.ascend.benchmark.timing import logical_gbps, summarize_samples
from tests.ascend.benchmark.timing import NpuEventTimer
from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_manifest import EPModeCase, enumerate_ep_mode_cases


ROOT = Path(__file__).resolve().parents[2]
BENCH_EP = ROOT / "tests/ascend/benchmark/bench_ep.py"
RUNTIME = ROOT / "tests/ascend/benchmark/runtime.py"
COMPARE = ROOT / "tests/ascend/benchmark/compare.py"


def test_case_matrix_matches_upstream_order_and_size():
    cases = enumerate_ep_mode_cases()

    assert len(cases) == 144
    assert len({case.case_id for case in cases}) == 144
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
        ("performance", True): 12,
        ("performance", False): 12,
        ("functional", False): 120,
    }
    assert Counter(
        classification.reason
        for classification in classified
        if classification.suite == "functional"
    ) == {
        "event_chaining_deferred": 48,
        "async_overlap_deferred": 48,
        "comm_stream_allocation_deferred": 24,
    }


def test_supported_ascend_cases_are_the_sync_bf16_intersection():
    supported = [
        case
        for case in enumerate_ep_mode_cases()
        if classify_ascend_case(case).supported
    ]

    assert len(supported) == 12
    assert all(not case.use_fp8_dispatch for case in supported)
    assert all(not case.with_previous_event for case in supported)
    assert all(not case.async_with_compute_stream for case in supported)
    assert all(not case.allocate_on_comm_stream for case in supported)
    assert {
        (case.expert_alignment, case.num_bias, case.do_handle_copy)
        for case in supported
    } == {
        (alignment, num_bias, do_handle_copy)
        for alignment in (128, 1)
        for num_bias in (0, 1, 2)
        for do_handle_copy in (True, False)
    }


class _ContractTensor:
    shape = (1, 3)

    def data_ptr(self):
        return 1

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


class _ContractTorch:
    int32 = "int32"
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


def _run_correctness_contract_check(expanded_weights=None):
    tensor = _ContractTensor()
    expanded_weights = expanded_weights or tensor
    handle = SimpleNamespace(
        topk_idx=tensor,
        recv_src_metadata=tensor,
        psum_num_recv_tokens_per_expert=tensor,
    )
    event = SimpleNamespace(event=None)
    normal = (tensor, tensor, tensor, handle, event)
    expanded = (tensor, None, expanded_weights, handle, event)
    cached = (tensor, tensor, None, handle, event)
    cached_expanded = (tensor, None, expanded_weights, handle, event)
    combined = (tensor, tensor, event)
    case = EPModeCase(False, 1, False, 0, False, False, False)
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
        ((tensor, tensor, tensor, tensor, None), tensor, tensor),
        num_recv_tokens=1,
    )


def test_correctness_check_accepts_missing_cached_weights_like_upstream():
    _run_correctness_contract_check()


def test_correctness_check_preserves_all_expanded_weight_lanes():
    _run_correctness_contract_check(_ExpandedWeights())


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


def test_performance_report_contains_only_current_cases(tmp_path):
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
    )
    output = tmp_path / "benchmark.json"

    write_report_atomic(output, report)
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 1
    assert len(payload["cases"]) == 12
    assert payload["case_summary"] == {
        "total": 12,
        "pending": 12,
        "passed": 0,
        "failed": 0,
    }


def test_performance_report_rejects_deferred_case():
    deferred = next(
        case
        for case in enumerate_ep_mode_cases()
        if not classify_ascend_case(case).supported
    )

    with pytest.raises(ValueError, match="cannot add deferred case"):
        BenchmarkReport.empty_for_cases(
            platform="ascend",
            cases=(deferred,),
            classify=classify_ascend_case,
            workload_fingerprint="a" * 64,
            world_size=2,
        )


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
    )
    right = BenchmarkReport.empty_for_cases(
        platform="ascend",
        cases=cases,
        classify=classify_ascend_case,
        workload_fingerprint="b" * 64,
        world_size=2,
    )

    with pytest.raises(ValueError, match="workload_fingerprint"):
        validate_comparable(left, right)


def test_benchmark_parser_preserves_production_size_defaults():
    args = build_parser().parse_args([])

    assert args.num_tokens == 4096
    assert args.hidden == 7168
    assert args.num_topk == 6
    assert args.num_experts == 256
    assert args.warmups == 30
    assert args.iterations == 30
    assert args.allow_multiple_reduction == 1


def test_default_selection_contains_only_current_performance_cases():
    selected = _selected_case_ids(build_parser(), None)

    assert len(selected) == 12
    assert all("-bf16-" in case_id for case_id in selected)
    assert all("-prev0-async0-alloc0" in case_id for case_id in selected)


@pytest.mark.parametrize(
    ("suite", "expected_count"),
    (("all", 144), ("performance", 24), ("functional", 120)),
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
    assert all(
        case["reason"]
        for case in payload["cases"]
        if case["status"] == "deferred"
    )


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


@pytest.mark.parametrize(
    ("case_id", "suite", "reason"),
    (
        (
            "ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0",
            "performance",
            "fp8_runtime_deferred",
        ),
        (
            "ep-bf16-align128-bias0-hcopy1-prev1-async0-alloc1",
            "functional",
            "event_chaining_deferred",
        ),
    ),
)
def test_cli_rejects_noncurrent_performance_case_before_runtime_import(
    case_id, suite, reason,
):
    result = subprocess.run(
        [sys.executable, str(BENCH_EP), "--cases", case_id],
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 2
    assert suite in result.stderr
    assert reason in result.stderr
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
    assert "num_sms=1" in source
    assert "num_qps=0" in source
    assert source.index("buffer.destroy()") < source.index(
        "dist.destroy_process_group()")


def _report_fixture(platform, *, mean=2e-6, gbps=100.0,
                    fingerprint="a" * 64):
    return {
        "schema_version": 1,
        "formula_version": 1,
        "platform": platform,
        "world_size": 2,
        "workload_fingerprint": fingerprint,
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
