import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.ascend.benchmark.report import (
    BenchmarkReport,
    validate_comparable,
    write_report_atomic,
)
from tests.ascend.benchmark.compare import compare_reports
from tests.ascend.benchmark.bench_ep import build_parser
from tests.ascend.benchmark.timing import logical_gbps, summarize_samples
from tests.ascend.benchmark.timing import NpuEventTimer
from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_manifest import enumerate_ep_mode_cases


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


def test_ascend_capability_counts_are_exhaustive():
    capabilities = [
        classify_ascend_case(case) for case in enumerate_ep_mode_cases()
    ]
    counts = Counter(
        capability.reason
        for capability in capabilities
        if not capability.supported
    )

    assert sum(capability.supported for capability in capabilities) == 12
    assert counts == {
        "fp8_runtime_deferred": 72,
        "event_chaining_deferred": 24,
        "async_overlap_deferred": 24,
        "comm_stream_allocation_deferred": 12,
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


def test_report_contains_every_case_and_atomic_writer_round_trips(tmp_path):
    report = BenchmarkReport.empty_for_cases(
        platform="ascend",
        cases=enumerate_ep_mode_cases(),
        classify=classify_ascend_case,
        workload_fingerprint="a" * 64,
        world_size=2,
    )
    output = tmp_path / "benchmark.json"

    write_report_atomic(output, report)
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 1
    assert len(payload["cases"]) == 144
    assert sum(case["status"] == "unsupported" for case in payload["cases"]) == 132
    assert sum(case["status"] == "pending" for case in payload["cases"]) == 12


def test_comparison_rejects_incompatible_report_identity():
    cases = enumerate_ep_mode_cases()
    left = BenchmarkReport.empty_for_cases(
        platform="cuda",
        cases=cases,
        classify=lambda _case: type(
            "Supported", (), {"supported": True, "reason": ""})(),
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


def test_list_cases_is_host_only_and_exhaustive():
    result = subprocess.run(
        [
            sys.executable,
            str(BENCH_EP),
            "--list-cases",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    payload = json.loads(result.stdout)

    assert len(payload["cases"]) == 144
    assert payload["summary"] == {"supported": 12, "unsupported": 132}
    assert all(
        case["reason"]
        for case in payload["cases"]
        if case["status"] == "unsupported"
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
