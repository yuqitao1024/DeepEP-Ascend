from copy import deepcopy
from dataclasses import asdict
import math
import re
from pathlib import Path

import pytest

from tests.benchmark.profiles import BenchmarkProfile, PROFILES, profile_manifest
from tests.benchmark.report_markdown import (
    comparison_rows,
    identify_profile,
    operation_records,
    render_backend_markdown,
    render_comparison_markdown,
    validate_complete_report,
    write_text_atomic,
)
from tests.utils.ep_benchmark_manifest import enumerate_ep_mode_cases


OPERATIONS = (
    "dispatch",
    "expanded_dispatch",
    "cached_dispatch",
    "combine",
    "reduced_combine",
)


def complete_report(platform, profile_name, device_name):
    profile = PROFILES[profile_name]
    manifest = profile_manifest(profile)
    summary = {
        "minimum": 1e-6,
        "mean": 1e-6,
        "p50": 1e-6,
        "p95": 1e-6,
        "maximum": 1e-6,
    }
    operations = [
        {
            "operation_id": operation_id,
            "formula_version": 1,
            "device_seconds": dict(summary),
            "wall_seconds": dict(summary),
            "device_samples": [1e-6] * profile.iterations,
            "wall_samples": [1e-6] * profile.iterations,
            "logical_bytes": {"scaleup": 2000},
            "logical_gbps": 2.0,
            "per_rank": [],
        }
        for operation_id in OPERATIONS
    ]
    cases = [
        {
            "case_id": case.case_id,
            "mode": asdict(case),
            "status": "passed",
            "reason": "",
            "operations": deepcopy(operations),
        }
        for case in enumerate_ep_mode_cases()
    ]
    return {
        "schema_version": 1,
        "formula_version": 1,
        "generated_at": "2026-08-20T00:00:00+00:00",
        "git_commit": "a" * 40,
        "platform": platform,
        "device": {"name": device_name},
        "world_size": 8,
        "workload": asdict(manifest.spec),
        "workload_fingerprint": manifest.fingerprint,
        "timing_protocol": {
            "timer": "cuda_event" if platform == "cuda" else "npu_event",
            "warmups": profile.warmups,
            "iterations": profile.iterations,
            "rank_aggregation": "maximum_latency",
            "logical_byte_aggregation": "sum",
        },
        "case_summary": {
            "total": 144,
            "pending": 0,
            "passed": 144,
            "failed": 0,
        },
        "cases": cases,
        "failures": [],
    }


def test_profiles_are_fixed_eight_rank_workloads():
    canonical = PROFILES["canonical"]
    smoke = PROFILES["smoke"]
    assert (
        canonical.world_size, canonical.num_tokens, canonical.hidden,
        canonical.num_topk, canonical.num_experts, canonical.seed,
        canonical.warmups, canonical.iterations,
    ) == (8, 4096, 7168, 6, 256, 0, 30, 30)
    assert (
        smoke.world_size, smoke.num_tokens, smoke.hidden,
        smoke.num_topk, smoke.num_experts, smoke.seed,
        smoke.warmups, smoke.iterations,
    ) == (8, 16, 128, 2, 8, 0, 1, 1)
    assert profile_manifest(canonical).ranks[7].num_tokens == 4089
    assert profile_manifest(smoke).ranks[7].num_tokens == 9


def test_profiles_keep_fixed_routing_and_reduction_settings():
    canonical = PROFILES["canonical"]
    smoke = PROFILES["smoke"]
    assert (
        canonical.unbalanced_ratio,
        canonical.precise_unbalanced_ratio,
        canonical.masked_ratio,
        canonical.allow_multiple_reduction,
    ) == (1.0, False, 0.0, 1)
    assert (
        smoke.unbalanced_ratio,
        smoke.precise_unbalanced_ratio,
        smoke.masked_ratio,
        smoke.allow_multiple_reduction,
    ) == (1.0, False, 0.0, 1)


def test_complete_report_requires_144_cases_and_720_operations():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    validate_complete_report(
        report, platform="cuda", profile=PROFILES["smoke"], require_h800=True
    )
    assert len(operation_records(report)) == 720


def test_complete_report_rejects_fabricated_profile():
    fabricated = BenchmarkProfile(
        name="fabricated",
        world_size=4,
        num_tokens=16,
        hidden=128,
        num_topk=2,
        num_experts=8,
        seed=0,
        warmups=1,
        iterations=1,
    )
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    manifest = profile_manifest(fabricated)
    report["world_size"] = fabricated.world_size
    report["workload"] = asdict(manifest.spec)
    report["workload_fingerprint"] = manifest.fingerprint

    with pytest.raises(ValueError, match="profile"):
        validate_complete_report(
            report, platform="cuda", profile=fabricated, require_h800=True
        )


def test_complete_report_rejects_unknown_platform():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["platform"] = "other"
    report["timing_protocol"]["timer"] = "npu_event"

    with pytest.raises(ValueError, match="platform"):
        validate_complete_report(
            report, platform="other", profile=PROFILES["smoke"], require_h800=True
        )


def test_identify_profile_matches_exact_workload_and_timing_tuple():
    report = complete_report("ascend", "canonical", device_name="Ascend 910B")

    assert identify_profile(report) is PROFILES["canonical"]


def test_identify_profile_rejects_unknown_workload_and_timing_tuple():
    report = complete_report("ascend", "smoke", device_name="Ascend 910B")
    report["timing_protocol"]["iterations"] = 2

    with pytest.raises(
        ValueError, match="report does not match canonical or smoke profile"
    ):
        identify_profile(report)


def test_identify_profile_rejects_noncanonical_timer():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["timing_protocol"]["timer"] = "host_clock"

    with pytest.raises(
        ValueError, match="report does not match canonical or smoke profile"
    ):
        identify_profile(report)


def _set_wrong_platform(report):
    report["platform"] = "ascend"


def _set_wrong_world_size(report):
    report["world_size"] = 4


def _set_non_h800_device(report):
    report["device"]["name"] = "NVIDIA A100"


def _set_wrong_workload(report):
    report["workload"]["hidden"] = 129


def _set_wrong_fingerprint(report):
    report["workload_fingerprint"] = "b" * 64


def _set_wrong_warmups(report):
    report["timing_protocol"]["warmups"] = 2


def _set_wrong_iterations(report):
    report["timing_protocol"]["iterations"] = 2


def _set_wrong_aggregation(report):
    report["timing_protocol"]["rank_aggregation"] = "mean_latency"


def _set_wrong_case_order(report):
    report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]


def _set_duplicate_case(report):
    report["cases"][1]["case_id"] = report["cases"][0]["case_id"]


def _set_failed_case(report):
    report["cases"][0]["status"] = "failed"


def _set_failures(report):
    report["failures"] = [{"case_id": report["cases"][0]["case_id"]}]


def _set_wrong_operation_order(report):
    operations = report["cases"][0]["operations"]
    operations[0], operations[1] = operations[1], operations[0]


def _set_wrong_formula_version(report):
    report["cases"][0]["operations"][0]["formula_version"] = 2


def _set_zero_latency(report):
    report["cases"][0]["operations"][0]["device_seconds"]["mean"] = 0.0


def _set_nan_latency(report):
    report["cases"][0]["operations"][0]["device_seconds"]["p95"] = math.nan


def _set_zero_gbps(report):
    report["cases"][0]["operations"][0]["logical_gbps"] = 0.0


def _set_nan_gbps(report):
    report["cases"][0]["operations"][0]["logical_gbps"] = math.nan


def _set_wrong_sample_count(report):
    report["cases"][0]["operations"][0]["device_samples"] = []


@pytest.mark.parametrize(
    ("mutate", "field"),
    (
        (_set_wrong_platform, "platform"),
        (_set_wrong_world_size, "world_size"),
        (_set_non_h800_device, "device.name"),
        (_set_wrong_workload, "workload.hidden"),
        (_set_wrong_fingerprint, "workload_fingerprint"),
        (_set_wrong_warmups, "timing_protocol.warmups"),
        (_set_wrong_iterations, "timing_protocol.iterations"),
        (_set_wrong_aggregation, "timing_protocol.rank_aggregation"),
        (_set_wrong_case_order, "cases[0].case_id"),
        (_set_duplicate_case, "cases[1].case_id"),
        (_set_failed_case, "cases[0].status"),
        (_set_failures, "failures"),
        (_set_wrong_operation_order, "cases[0].operations[0].operation_id"),
        (_set_wrong_formula_version, "cases[0].operations[0].formula_version"),
        (_set_zero_latency, "cases[0].operations[0].device_seconds.mean"),
        (_set_nan_latency, "cases[0].operations[0].device_seconds.p95"),
        (_set_zero_gbps, "cases[0].operations[0].logical_gbps"),
        (_set_nan_gbps, "cases[0].operations[0].logical_gbps"),
        (_set_wrong_sample_count, "cases[0].operations[0].device_samples"),
    ),
)
def test_complete_report_rejects_each_invalid_field(mutate, field):
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    mutate(report)

    with pytest.raises(ValueError, match=re.escape(field)):
        validate_complete_report(
            report,
            platform="cuda",
            profile=PROFILES["smoke"],
            require_h800=True,
        )


def test_backend_markdown_renders_smoke_warning_and_canonical_detail_rows():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")

    markdown = render_backend_markdown(report, PROFILES["smoke"])

    assert "# EP Benchmark: CUDA / smoke" in markdown
    assert "NON-CANONICAL AUTOMATION VALIDATION" in markdown
    assert "| Case | Operation | Mean us | P50 us | P95 us | Logical GB/s |" in markdown
    assert markdown.count("\n| `ep-") == 720
    assert (
        "| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | dispatch | "
        "1.000 | 1.000 | 1.000 | 2.000 |"
    ) in markdown
    assert markdown.index("| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | dispatch") < markdown.index(
        "| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | cached_dispatch"
    )
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_backend_markdown_renders_canonical_provenance_workload_and_escaped_cells():
    report = complete_report("ascend", "canonical", device_name="Ascend|950\\rack\r\nnext")
    report["git_commit"] = "commit|abc\\def\r\nnext"

    markdown = render_backend_markdown(report, PROFILES["canonical"])

    assert "# EP Benchmark: Ascend / canonical" in markdown
    assert "NON-CANONICAL AUTOMATION VALIDATION" not in markdown
    assert "| Git commit | commit\\|abc\\\\def\\r\\nnext |" in markdown
    assert "| Device | Ascend\\|950\\\\rack\\r\\nnext |" in markdown
    assert "| World size | 8 |" in markdown
    assert "| Workload fingerprint | " + report["workload_fingerprint"] + " |" in markdown
    assert "| Tokens | 4096 |" in markdown
    assert "| Hidden | 7168 |" in markdown
    assert "| Top-k | 6 |" in markdown
    assert "| Experts | 256 |" in markdown


def _make_comparison_reports():
    cuda = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    ascend = complete_report("ascend", "smoke", device_name="Ascend 950")
    cuda_operation = cuda["cases"][0]["operations"][0]
    ascend_operation = ascend["cases"][0]["operations"][0]
    cuda_operation["device_seconds"].update(mean=1e-6, p50=2e-6, p95=3e-6)
    ascend_operation["device_seconds"].update(mean=2e-6, p50=4e-6, p95=6e-6)
    cuda_operation["logical_gbps"] = 4.0
    ascend_operation["logical_gbps"] = 2.0
    return cuda, ascend


def test_comparison_rows_and_markdown_render_ascend_over_cuda_ratios():
    cuda, ascend = _make_comparison_reports()

    rows = comparison_rows(cuda, ascend)
    markdown = render_comparison_markdown(cuda, ascend, PROFILES["smoke"])

    assert len(rows) == 720
    assert rows[0]["latency_ratio"] == pytest.approx(2.0)
    assert rows[0]["bandwidth_ratio"] == pytest.approx(0.5)
    assert "# EP Benchmark Comparison: H800 vs Ascend" in markdown
    assert "NON-CANONICAL AUTOMATION VALIDATION" in markdown
    assert markdown.count("\n| `ep-") == 720
    assert (
        "| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | dispatch | "
        "1.000 | 2.000 | 3.000 | 4.000 | 2.000 | 4.000 | 6.000 | 2.000 | "
        "2.000 | 0.500 |"
    ) in markdown


def _set_comparison_profile_mismatch(cuda, ascend):
    replacement = complete_report("ascend", "canonical", device_name="Ascend 950")
    ascend.clear()
    ascend.update(replacement)


def _set_comparison_schema_mismatch(cuda, ascend):
    ascend["schema_version"] = 2


def _set_comparison_formula_mismatch(cuda, ascend):
    ascend["formula_version"] = 2


def _set_comparison_fingerprint_mismatch(cuda, ascend):
    ascend["workload_fingerprint"] = "b" * 64


def _set_comparison_world_size_mismatch(cuda, ascend):
    ascend["world_size"] = 4


def _set_comparison_aggregation_mismatch(cuda, ascend):
    ascend["timing_protocol"]["rank_aggregation"] = "mean_latency"


def _set_comparison_case_sequence_mismatch(cuda, ascend):
    ascend["cases"][0], ascend["cases"][1] = ascend["cases"][1], ascend["cases"][0]


def _set_comparison_operation_sequence_mismatch(cuda, ascend):
    operations = ascend["cases"][0]["operations"]
    operations[0], operations[1] = operations[1], operations[0]


def _set_comparison_logical_bytes_mismatch(cuda, ascend):
    ascend["cases"][0]["operations"][0]["logical_bytes"] = {"scaleup": 2001}


@pytest.mark.parametrize(
    ("mutate", "field"),
    (
        (_set_comparison_profile_mismatch, "profile"),
        (_set_comparison_schema_mismatch, "schema_version"),
        (_set_comparison_formula_mismatch, "formula_version"),
        (_set_comparison_fingerprint_mismatch, "workload_fingerprint"),
        (_set_comparison_world_size_mismatch, "world_size"),
        (_set_comparison_aggregation_mismatch, "timing_protocol.rank_aggregation"),
        (_set_comparison_case_sequence_mismatch, "cases[0].case_id"),
        (_set_comparison_operation_sequence_mismatch, "cases[0].operations[0].operation_id"),
        (_set_comparison_logical_bytes_mismatch, "logical_bytes"),
    ),
)
def test_comparison_rejects_each_identity_mismatch_without_writing_output(
    tmp_path, mutate, field
):
    cuda, ascend = _make_comparison_reports()
    output = tmp_path / "comparison.md"
    mutate(cuda, ascend)

    with pytest.raises(ValueError, match=re.escape(field)):
        render_comparison_markdown(cuda, ascend, PROFILES["smoke"])

    assert not output.exists()


def test_write_text_atomic_replaces_destination_and_cleans_failed_temporary_file(
    tmp_path, monkeypatch
):
    output = tmp_path / "report.md"
    output.write_text("old\n", encoding="utf-8")
    write_text_atomic(output, "new\n")
    assert output.read_text(encoding="utf-8") == "new\n"

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("tests.benchmark.report_markdown.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_text_atomic(output, "uncommitted\n")

    assert output.read_text(encoding="utf-8") == "new\n"
    assert list(Path(tmp_path).glob(".report.md.*")) == []
