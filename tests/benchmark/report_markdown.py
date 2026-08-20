import math
from dataclasses import asdict
from typing import Any

from tests.benchmark.profiles import BenchmarkProfile, PROFILES, profile_manifest
from tests.utils.ep_benchmark_core import PERFORMANCE_OPERATIONS
from tests.utils.ep_benchmark_manifest import EPModeCase, enumerate_ep_mode_cases


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(field)


def _require_exact_dict(actual: Any, expected: dict[str, Any], field: str) -> None:
    if not isinstance(actual, dict):
        raise ValueError(field)
    for key, value in expected.items():
        _require_equal(actual.get(key), value, f"{field}.{key}")
    if actual.keys() != expected.keys():
        raise ValueError(field)


def _finite_positive(value: Any, field: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(field) from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(field)


def _sample_array(value: Any, iterations: int, field: str) -> None:
    if not isinstance(value, list) or len(value) != iterations:
        raise ValueError(field)
    for index, sample in enumerate(value):
        _finite_positive(sample, f"{field}[{index}]")


def _expected_timing(platform: str, profile: BenchmarkProfile) -> dict[str, Any]:
    return {
        "timer": "cuda_event" if platform == "cuda" else "npu_event",
        "warmups": profile.warmups,
        "iterations": profile.iterations,
        "rank_aggregation": "maximum_latency",
        "logical_byte_aggregation": "sum",
    }


def identify_profile(report: dict) -> BenchmarkProfile:
    platform = report.get("platform")
    if platform not in ("cuda", "ascend"):
        raise ValueError("report does not match canonical or smoke profile")
    matches = []
    for profile in PROFILES.values():
        manifest = profile_manifest(profile)
        if (
            report.get("world_size") == profile.world_size
            and report.get("workload") == asdict(manifest.spec)
            and report.get("workload_fingerprint") == manifest.fingerprint
            and report.get("timing_protocol") == _expected_timing(platform, profile)
        ):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError("report does not match canonical or smoke profile")
    return matches[0]


def _validate_operation(
    operation: dict,
    case_index: int,
    operation_index: int,
    profile: BenchmarkProfile,
) -> None:
    prefix = f"cases[{case_index}].operations[{operation_index}]"
    _require_equal(
        operation.get("operation_id"),
        PERFORMANCE_OPERATIONS[operation_index],
        f"{prefix}.operation_id",
    )
    _require_equal(operation.get("formula_version"), 1, f"{prefix}.formula_version")
    for clock in ("device_seconds", "wall_seconds"):
        summary = operation.get(clock)
        if not isinstance(summary, dict):
            raise ValueError(f"{prefix}.{clock}")
        for statistic in ("mean", "p50", "p95"):
            _finite_positive(summary.get(statistic), f"{prefix}.{clock}.{statistic}")
    _sample_array(
        operation.get("device_samples"), profile.iterations, f"{prefix}.device_samples"
    )
    _sample_array(
        operation.get("wall_samples"), profile.iterations, f"{prefix}.wall_samples"
    )
    logical_bytes = operation.get("logical_bytes")
    if not isinstance(logical_bytes, dict) or not logical_bytes:
        raise ValueError(f"{prefix}.logical_bytes")
    for category, count in logical_bytes.items():
        try:
            number = float(count)
        except (TypeError, ValueError):
            raise ValueError(f"{prefix}.logical_bytes.{category}") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{prefix}.logical_bytes.{category}")
    _finite_positive(operation.get("logical_gbps"), f"{prefix}.logical_gbps")


def validate_complete_report(
    report: dict,
    platform: str,
    profile: BenchmarkProfile,
    require_h800: bool = False,
) -> None:
    if (
        not isinstance(profile, BenchmarkProfile)
        or PROFILES.get(profile.name) is not profile
    ):
        raise ValueError("profile")
    if platform not in ("cuda", "ascend"):
        raise ValueError("platform")
    if not isinstance(report, dict):
        raise ValueError("report")
    _require_equal(report.get("schema_version"), 1, "schema_version")
    _require_equal(report.get("formula_version"), 1, "formula_version")
    _require_equal(report.get("platform"), platform, "platform")
    _require_equal(report.get("world_size"), profile.world_size, "world_size")
    manifest = profile_manifest(profile)
    _require_exact_dict(
        report.get("workload"), asdict(manifest.spec), "workload"
    )
    _require_equal(
        report.get("workload_fingerprint"),
        manifest.fingerprint,
        "workload_fingerprint",
    )
    _require_exact_dict(
        report.get("timing_protocol"),
        _expected_timing(platform, profile),
        "timing_protocol",
    )
    if require_h800:
        name = report.get("device", {}).get("name")
        if not isinstance(name, str) or "h800" not in name.lower():
            raise ValueError("device.name")

    expected_cases = enumerate_ep_mode_cases()
    _require_equal(
        report.get("case_summary"),
        {"total": 144, "pending": 0, "passed": 144, "failed": 0},
        "case_summary",
    )
    _require_equal(report.get("failures"), [], "failures")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("cases")
    for case_index, (case, expected_case) in enumerate(zip(cases, expected_cases)):
        prefix = f"cases[{case_index}]"
        if not isinstance(case, dict):
            raise ValueError(prefix)
        _require_equal(case.get("case_id"), expected_case.case_id, f"{prefix}.case_id")
        _require_equal(case.get("mode"), asdict(expected_case), f"{prefix}.mode")
        _require_equal(case.get("status"), "passed", f"{prefix}.status")
        _require_equal(case.get("reason"), "", f"{prefix}.reason")
        operations = case.get("operations")
        if not isinstance(operations, list) or len(operations) != len(PERFORMANCE_OPERATIONS):
            raise ValueError(f"{prefix}.operations")
        for operation_index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise ValueError(f"{prefix}.operations[{operation_index}]")
            _validate_operation(operation, case_index, operation_index, profile)


def operation_records(report: dict) -> tuple[tuple[EPModeCase, dict], ...]:
    cases = report.get("cases")
    expected_cases = enumerate_ep_mode_cases()
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("cases")
    records = []
    for case_index, (record, expected_case) in enumerate(zip(cases, expected_cases)):
        if not isinstance(record, dict):
            raise ValueError(f"cases[{case_index}]")
        _require_equal(
            record.get("case_id"), expected_case.case_id, f"cases[{case_index}].case_id"
        )
        operations = record.get("operations")
        if not isinstance(operations, list) or len(operations) != len(PERFORMANCE_OPERATIONS):
            raise ValueError(f"cases[{case_index}].operations")
        for operation_index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise ValueError(f"cases[{case_index}].operations[{operation_index}]")
            _require_equal(
                operation.get("operation_id"),
                PERFORMANCE_OPERATIONS[operation_index],
                f"cases[{case_index}].operations[{operation_index}].operation_id",
            )
            records.append((expected_case, operation))
    return tuple(records)
