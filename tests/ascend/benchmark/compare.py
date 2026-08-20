import argparse
import json
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ascend.benchmark.report import (  # noqa: E402
    SCHEMA_VERSION,
    validate_execution_protocol,
)


def _validate_report_identity(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "benchmark reports are not comparable: schema_version"
        )
    validate_execution_protocol(report.get("execution_protocol"))


def _require_equal(
    cuda: dict[str, Any],
    ascend: dict[str, Any],
    fields: tuple[str, ...],
    namespace: str = "",
) -> None:
    mismatches = [field for field in fields if cuda.get(field) != ascend.get(field)]
    if mismatches:
        if namespace:
            mismatches = [f"{namespace}.{field}" for field in mismatches]
        raise ValueError(
            "benchmark reports are not comparable: " + ", ".join(mismatches)
        )


def _passed_operations(report: dict[str, Any]) -> dict[tuple[str, str], dict]:
    operations = {}
    for case in report.get("cases", ()):
        if case.get("status") != "passed":
            continue
        case_id = case.get("case_id")
        for operation in case.get("operations", ()):
            key = (case_id, operation.get("operation_id"))
            if key in operations:
                raise ValueError(
                    f"duplicate case_id/operation_id record: {key}"
                )
            operations[key] = operation
    return operations


def _positive(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for field in path:
        value = value[field]
    value = float(value)
    if value <= 0:
        raise ValueError("comparison metrics must be positive")
    return value


def compare_reports(
    cuda: dict[str, Any], ascend: dict[str, Any]
) -> list[dict[str, Any]]:
    _validate_report_identity(cuda)
    _validate_report_identity(ascend)
    _require_equal(
        cuda,
        ascend,
        (
            "schema_version",
            "formula_version",
            "world_size",
            "workload_fingerprint",
            "execution_protocol",
        ),
    )
    _require_equal(
        cuda.get("timing_protocol", {}),
        ascend.get("timing_protocol", {}),
        (
            "warmups",
            "iterations",
            "rank_aggregation",
            "logical_byte_aggregation",
        ),
        namespace="timing_protocol",
    )
    if cuda.get("platform") != "cuda" or ascend.get("platform") != "ascend":
        raise ValueError("reports must be ordered as CUDA then Ascend")

    cuda_operations = _passed_operations(cuda)
    ascend_operations = _passed_operations(ascend)
    if cuda_operations.keys() != ascend_operations.keys():
        raise ValueError(
            "benchmark reports are not comparable: case_id/operation_id"
        )

    rows = []
    for case_id, operation_id in sorted(cuda_operations):
        cuda_operation = cuda_operations[(case_id, operation_id)]
        ascend_operation = ascend_operations[(case_id, operation_id)]
        if cuda_operation.get("formula_version") != ascend_operation.get(
            "formula_version"
        ):
            raise ValueError(
                "benchmark reports are not comparable: formula_version"
            )
        if cuda_operation.get("logical_bytes") != ascend_operation.get(
            "logical_bytes"
        ):
            raise ValueError(
                "benchmark reports are not comparable: logical_bytes "
                f"for {case_id}/{operation_id}"
            )
        cuda_mean = _positive(cuda_operation, ("device_seconds", "mean"))
        ascend_mean = _positive(
            ascend_operation, ("device_seconds", "mean")
        )
        cuda_gbps = _positive(cuda_operation, ("logical_gbps",))
        ascend_gbps = _positive(ascend_operation, ("logical_gbps",))
        rows.append({
            "case_id": case_id,
            "operation_id": operation_id,
            "world_size": cuda["world_size"],
            "cuda_mean_seconds": cuda_mean,
            "cuda_p50_seconds": _positive(
                cuda_operation, ("device_seconds", "p50")
            ),
            "cuda_p95_seconds": _positive(
                cuda_operation, ("device_seconds", "p95")
            ),
            "ascend_mean_seconds": ascend_mean,
            "ascend_p50_seconds": _positive(
                ascend_operation, ("device_seconds", "p50")
            ),
            "ascend_p95_seconds": _positive(
                ascend_operation, ("device_seconds", "p95")
            ),
            "cuda_logical_gbps": cuda_gbps,
            "ascend_logical_gbps": ascend_gbps,
            "latency_ratio_ascend_over_cuda": ascend_mean / cuda_mean,
            "bandwidth_ratio_ascend_over_cuda": ascend_gbps / cuda_gbps,
        })
    return rows


def _unsupported_summary(report: dict[str, Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for case in report.get("cases", ()):
        if case.get("status") == "unsupported":
            reason = case.get("reason", "unspecified")
            summary[reason] = summary.get(reason, 0) + 1
    return dict(sorted(summary.items()))


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "| Case | Operation | Ranks | CUDA mean/p50/p95 us | "
        "Ascend mean/p50/p95 us | "
        "CUDA GB/s | Ascend GB/s | Latency A/C | Bandwidth A/C |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| `{row['case_id']}` | {row['operation_id']} | "
            f"{row['world_size']} | "
            f"{row['cuda_mean_seconds'] * 1e6:.3f}/"
            f"{row['cuda_p50_seconds'] * 1e6:.3f}/"
            f"{row['cuda_p95_seconds'] * 1e6:.3f} | "
            f"{row['ascend_mean_seconds'] * 1e6:.3f}/"
            f"{row['ascend_p50_seconds'] * 1e6:.3f}/"
            f"{row['ascend_p95_seconds'] * 1e6:.3f} | "
            f"{row['cuda_logical_gbps']:.3f} | "
            f"{row['ascend_logical_gbps']:.3f} | "
            f"{row['latency_ratio_ascend_over_cuda']:.3f} | "
            f"{row['bandwidth_ratio_ascend_over_cuda']:.3f} |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare matching CUDA and Ascend EP benchmark reports"
    )
    parser.add_argument("cuda_report")
    parser.add_argument("ascend_report")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cuda = json.loads(Path(args.cuda_report).read_text(encoding="utf-8"))
    ascend = json.loads(Path(args.ascend_report).read_text(encoding="utf-8"))
    rows = compare_reports(cuda, ascend)
    unsupported = {
        "cuda": _unsupported_summary(cuda),
        "ascend": _unsupported_summary(ascend),
    }
    if args.format == "json":
        print(json.dumps({"rows": rows, "unsupported": unsupported}, sort_keys=True))
    else:
        _print_table(rows)
        print("\nUnsupported summary:")
        print(json.dumps(unsupported, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
