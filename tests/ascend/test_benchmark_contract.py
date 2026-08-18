import json
from collections import Counter

import pytest

from tests.ascend.benchmark.report import (
    BenchmarkReport,
    validate_comparable,
    write_report_atomic,
)
from tests.ascend.benchmark.timing import logical_gbps, summarize_samples
from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_manifest import enumerate_ep_mode_cases


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
