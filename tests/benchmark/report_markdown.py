import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tests.ascend.benchmark.report import (
    SCHEMA_VERSION,
    validate_execution_protocol,
)
from tests.benchmark.profiles import (
    BenchmarkProfile,
    PROFILES,
    profile_cases,
    profile_manifest,
)
from tests.utils.ep_benchmark_core import PERFORMANCE_OPERATIONS
from tests.utils.ep_benchmark_manifest import EPModeCase


PROFILE_MISMATCH = "report does not match a known benchmark profile"


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
    if not isinstance(report, dict):
        raise ValueError(PROFILE_MISMATCH)
    _require_equal(report.get("schema_version"), SCHEMA_VERSION, "schema_version")
    platform = report.get("platform")
    if platform not in ("cuda", "ascend"):
        raise ValueError(PROFILE_MISMATCH)
    matches = []
    report_cases = report.get("cases")
    report_case_ids = (
        tuple(
            case.get("case_id") if isinstance(case, dict) else None
            for case in report_cases
        )
        if isinstance(report_cases, list)
        else None
    )
    device = report.get("device")
    for profile in PROFILES.values():
        manifest = profile_manifest(profile)
        if (
            report.get("world_size") == profile.world_size
            and report.get("workload") == asdict(manifest.spec)
            and report.get("workload_fingerprint") == manifest.fingerprint
            and report.get("timing_protocol") == _expected_timing(platform, profile)
            and report_case_ids
            == tuple(case.case_id for case in profile_cases(profile))
            and (
                platform != "ascend"
                or (
                    isinstance(device, dict)
                    and device.get("num_sms") == profile.ascend_num_sms
                    and device.get("num_qps") == 0
                )
            )
        ):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError(PROFILE_MISMATCH)
    profile = matches[0]
    validate_execution_protocol(
        report.get("execution_protocol"),
        expected_allow_multiple_reduction=profile.allow_multiple_reduction,
    )
    return profile


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
    _require_equal(
        report.get("schema_version"), SCHEMA_VERSION, "schema_version"
    )
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
    validate_execution_protocol(
        report.get("execution_protocol"),
        expected_allow_multiple_reduction=profile.allow_multiple_reduction,
    )
    _require_exact_dict(
        report.get("timing_protocol"),
        _expected_timing(platform, profile),
        "timing_protocol",
    )
    device = report.get("device")
    if not isinstance(device, dict):
        raise ValueError("device")
    if require_h800:
        name = device.get("name")
        if not isinstance(name, str) or "h800" not in name.lower():
            raise ValueError("device.name")
    if platform == "ascend":
        _require_equal(
            device.get("num_sms"), profile.ascend_num_sms, "device.num_sms"
        )
        _require_equal(device.get("num_qps"), 0, "device.num_qps")

    expected_cases = profile_cases(profile)
    expected_case_count = len(expected_cases)
    _require_equal(
        report.get("case_summary"),
        {
            "total": expected_case_count,
            "pending": 0,
            "passed": expected_case_count,
            "failed": 0,
        },
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
    expected_cases = profile_cases(identify_profile(report))
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


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace(
        "\r", "\\r"
    ).replace("\n", "\\n")


def _document(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _backend_title(platform: str) -> str:
    if platform == "cuda":
        return "CUDA"
    if platform == "ascend":
        return "Ascend"
    raise ValueError("platform")


def _workload_metadata_lines(
    report: dict, profile: BenchmarkProfile
) -> list[str]:
    workload = report["workload"]
    return [
        "## Workload",
        "| Field | Value |",
        "| --- | ---: |",
        f"| Tokens | {workload['num_tokens']} |",
        f"| Hidden | {workload['hidden']} |",
        f"| Top-k | {workload['num_topk']} |",
        f"| Experts | {workload['num_experts']} |",
        f"| Seed | {workload['seed']} |",
        f"| Warmups | {profile.warmups} |",
        f"| Iterations | {profile.iterations} |",
    ]


def _backend_metadata_lines(report: dict, profile: BenchmarkProfile) -> list[str]:
    return [
        "## Provenance",
        "| Field | Value |",
        "| --- | --- |",
        f"| Generated at | {_markdown_cell(report.get('generated_at', ''))} |",
        f"| Git commit | {_markdown_cell(report.get('git_commit', ''))} |",
        f"| Platform | {_backend_title(report['platform'])} |",
        f"| Device | {_markdown_cell(report.get('device', {}).get('name', ''))} |",
        f"| World size | {report['world_size']} |",
        f"| Workload fingerprint | {_markdown_cell(report['workload_fingerprint'])} |",
        "",
        *_workload_metadata_lines(report, profile),
    ]


def render_backend_markdown(report: dict, profile: BenchmarkProfile) -> str:
    platform = report.get("platform") if isinstance(report, dict) else None
    if platform not in ("cuda", "ascend"):
        raise ValueError("platform")
    validate_complete_report(
        report,
        platform=platform,
        profile=profile,
        require_h800=platform == "cuda",
    )
    lines = [f"# EP Benchmark: {_backend_title(platform)} / {profile.name}"]
    if profile.name == "smoke":
        lines.extend(("", "**NON-CANONICAL AUTOMATION VALIDATION**"))
    elif profile.name == "representative":
        lines.extend((
            "",
            "**DEEPEP V2 PRECHECK: 1 OF 144; NOT A FORMAL PERFORMANCE RESULT**",
        ))
    elif profile.name == "canonical":
        lines.extend((
            "",
            "**FORMAL PERFORMANCE MATRIX: ALL 144 CASES / 720 OPERATIONS**",
        ))
    lines.extend(("", *_backend_metadata_lines(report, profile), "", "## Detail"))
    lines.extend((
        "| Case | Operation | Mean us | P50 us | P95 us | Logical GB/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ))
    for case, operation in operation_records(report):
        device_seconds = operation["device_seconds"]
        lines.append(
            f"| `{_markdown_cell(case.case_id)}` | "
            f"{_markdown_cell(operation['operation_id'])} | "
            f"{float(device_seconds['mean']) * 1e6:.3f} | "
            f"{float(device_seconds['p50']) * 1e6:.3f} | "
            f"{float(device_seconds['p95']) * 1e6:.3f} | "
            f"{float(operation['logical_gbps']):.3f} |"
        )
    return _document(lines)


def _comparison_profile(cuda: dict, ascend: dict) -> BenchmarkProfile:
    try:
        cuda_profile = identify_profile(cuda)
    except ValueError:
        ascend_profile = identify_profile(ascend)
        validate_complete_report(
            cuda,
            platform="cuda",
            profile=ascend_profile,
            require_h800=True,
        )
        raise AssertionError("unreachable")
    try:
        ascend_profile = identify_profile(ascend)
    except ValueError:
        validate_complete_report(
            ascend,
            platform="ascend",
            profile=cuda_profile,
        )
        raise AssertionError("unreachable")
    if cuda_profile is not ascend_profile:
        raise ValueError("profile")
    validate_complete_report(
        cuda,
        platform="cuda",
        profile=cuda_profile,
        require_h800=True,
    )
    validate_complete_report(
        ascend,
        platform="ascend",
        profile=ascend_profile,
    )
    return cuda_profile


def _comparison_equal(cuda: Any, ascend: Any, field: str) -> None:
    if cuda != ascend:
        raise ValueError(field)


def comparison_rows(cuda: dict, ascend: dict) -> tuple[dict, ...]:
    _comparison_profile(cuda, ascend)
    for field in (
        "schema_version",
        "formula_version",
        "world_size",
        "workload_fingerprint",
        "execution_protocol",
    ):
        _comparison_equal(cuda.get(field), ascend.get(field), field)
    for field in (
        "warmups",
        "iterations",
        "rank_aggregation",
        "logical_byte_aggregation",
    ):
        _comparison_equal(
            cuda["timing_protocol"].get(field),
            ascend["timing_protocol"].get(field),
            f"timing_protocol.{field}",
        )

    rows = []
    for index, ((cuda_case, cuda_operation), (ascend_case, ascend_operation)) in enumerate(
        zip(operation_records(cuda), operation_records(ascend))
    ):
        prefix = f"records[{index}]"
        _comparison_equal(cuda_case.case_id, ascend_case.case_id, f"{prefix}.case_id")
        _comparison_equal(
            cuda_operation["operation_id"],
            ascend_operation["operation_id"],
            f"{prefix}.operation_id",
        )
        _comparison_equal(
            cuda_operation["formula_version"],
            ascend_operation["formula_version"],
            f"{prefix}.formula_version",
        )
        _comparison_equal(
            cuda_operation["logical_bytes"],
            ascend_operation["logical_bytes"],
            f"{prefix}.logical_bytes",
        )
        cuda_seconds = cuda_operation["device_seconds"]
        ascend_seconds = ascend_operation["device_seconds"]
        cuda_mean = float(cuda_seconds["mean"])
        ascend_mean = float(ascend_seconds["mean"])
        cuda_gbps = float(cuda_operation["logical_gbps"])
        ascend_gbps = float(ascend_operation["logical_gbps"])
        rows.append({
            "case_id": cuda_case.case_id,
            "operation_id": cuda_operation["operation_id"],
            "cuda_mean_seconds": cuda_mean,
            "cuda_p50_seconds": float(cuda_seconds["p50"]),
            "cuda_p95_seconds": float(cuda_seconds["p95"]),
            "cuda_logical_gbps": cuda_gbps,
            "ascend_mean_seconds": ascend_mean,
            "ascend_p50_seconds": float(ascend_seconds["p50"]),
            "ascend_p95_seconds": float(ascend_seconds["p95"]),
            "ascend_logical_gbps": ascend_gbps,
            "latency_ratio": ascend_mean / cuda_mean,
            "bandwidth_ratio": ascend_gbps / cuda_gbps,
        })
    return tuple(rows)


def render_comparison_markdown(
    cuda: dict, ascend: dict, profile: BenchmarkProfile
) -> str:
    rows = comparison_rows(cuda, ascend)
    if identify_profile(cuda) is not profile:
        raise ValueError("profile")
    lines = ["# EP Benchmark Comparison: H800 vs Ascend"]
    if profile.name == "smoke":
        lines.extend(("", "**NON-CANONICAL AUTOMATION VALIDATION**"))
    elif profile.name == "representative":
        lines.extend((
            "",
            "**DEEPEP V2 PRECHECK: 1 OF 144; NOT A FORMAL PERFORMANCE RESULT**",
        ))
    elif profile.name == "canonical":
        lines.extend((
            "",
            "**FORMAL PERFORMANCE MATRIX: ALL 144 CASES / 720 OPERATIONS**",
        ))
    lines.extend((
        "",
        "## Provenance",
        "| Field | H800 | Ascend |",
        "| --- | --- | --- |",
        f"| Git commit | {_markdown_cell(cuda.get('git_commit', ''))} | "
        f"{_markdown_cell(ascend.get('git_commit', ''))} |",
        f"| Device | {_markdown_cell(cuda.get('device', {}).get('name', ''))} | "
        f"{_markdown_cell(ascend.get('device', {}).get('name', ''))} |",
        f"| World size | {cuda['world_size']} | {ascend['world_size']} |",
        f"| Workload fingerprint | {_markdown_cell(cuda['workload_fingerprint'])} | "
        f"{_markdown_cell(ascend['workload_fingerprint'])} |",
        "",
        *_workload_metadata_lines(cuda, profile),
        "",
        "## Detail",
        "| Case | Operation | H800 Mean us | H800 P50 us | H800 P95 us | "
        "H800 Logical GB/s | Ascend Mean us | Ascend P50 us | Ascend P95 us | "
        "Ascend Logical GB/s | Latency Ascend/H800 | Bandwidth Ascend/H800 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for row in rows:
        lines.append(
            f"| `{_markdown_cell(row['case_id'])}` | "
            f"{_markdown_cell(row['operation_id'])} | "
            f"{row['cuda_mean_seconds'] * 1e6:.3f} | "
            f"{row['cuda_p50_seconds'] * 1e6:.3f} | "
            f"{row['cuda_p95_seconds'] * 1e6:.3f} | "
            f"{row['cuda_logical_gbps']:.3f} | "
            f"{row['ascend_mean_seconds'] * 1e6:.3f} | "
            f"{row['ascend_p50_seconds'] * 1e6:.3f} | "
            f"{row['ascend_p95_seconds'] * 1e6:.3f} | "
            f"{row['ascend_logical_gbps']:.3f} | "
            f"{row['latency_ratio']:.3f} | "
            f"{row['bandwidth_ratio']:.3f} |"
        )
    return _document(lines)


def write_text_atomic(path: Path, content: str) -> None:
    target = Path(path)
    temporary: Path | None = None
    committed = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent, text=True
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        committed = True
    finally:
        if temporary is not None and not committed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
