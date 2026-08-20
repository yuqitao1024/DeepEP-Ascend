from copy import deepcopy
from dataclasses import asdict
import math
import re

import pytest

from tests.benchmark.profiles import PROFILES, profile_manifest
from tests.benchmark.report_markdown import (
    identify_profile,
    operation_records,
    validate_complete_report,
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
