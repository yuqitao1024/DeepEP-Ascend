import argparse
import json
from pathlib import Path
from typing import Any

from tests.ascend.benchmark.report import SCHEMA_VERSION
from tests.ascend.benchmark.timeline import (
    operation_stage_semantics,
    stage_semantic,
)
from tests.utils.ep_benchmark_core import PERFORMANCE_OPERATIONS


TIMELINE_SCHEMA_VERSION = 1


def _nonnegative_int_mapping(value: Any, label: str) -> dict[str, int]:
    if (
        not isinstance(value, dict)
        or not value
        or any(not isinstance(key, str) or not key for key in value)
        or any(type(item) is not int or item < 0 for item in value.values())
    ):
        raise ValueError(label)
    return value


def _timeline_cycles(value: Any) -> dict[str, int]:
    names = {
        "start", "end", "envelope_cycles", "active_cycles",
        "idle_cycles", "overlap_cycles",
    }
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("timeline operation interval")
    if any(type(item) is not int or item < 0 for item in value.values()):
        raise ValueError("timeline operation interval")
    if (
        value["start"] <= 0
        or value["end"] < value["start"]
        or value["envelope_cycles"] != value["end"] - value["start"]
        or value["active_cycles"] + value["idle_cycles"]
        != value["envelope_cycles"]
    ):
        raise ValueError("timeline operation interval")
    return value


def _stage_interval(stages: list[dict[str, Any]]) -> dict[str, int | None]:
    if not stages:
        return {
            "start_cycles": None,
            "end_cycles": None,
            "envelope_cycles": 0,
            "active_cycles": 0,
            "idle_cycles": 0,
            "overlap_cycles": 0,
        }
    intervals = sorted((stage["start"], stage["end"]) for stage in stages)
    summed_spans = sum(stage["span_cycles"] for stage in stages)
    start_cycles = intervals[0][0]
    end_cycles = max(end for _, end in intervals)
    merged_start, merged_end = intervals[0]
    active_cycles = 0
    for start, end in intervals[1:]:
        if start > merged_end:
            active_cycles += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)
    active_cycles += merged_end - merged_start
    envelope_cycles = end_cycles - start_cycles
    return {
        "start_cycles": start_cycles,
        "end_cycles": end_cycles,
        "envelope_cycles": envelope_cycles,
        "active_cycles": active_cycles,
        "idle_cycles": envelope_cycles - active_cycles,
        "overlap_cycles": summed_spans - active_cycles,
    }


def _validate_stage(
    operation_id: str,
    stage: Any,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(stage, dict):
        raise ValueError("timeline stage interval")
    raw_name = stage.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError("timeline stage name")
    semantic = stage_semantic(operation_id, raw_name)
    if stage.get("stage_id") != semantic.stage_id:
        raise ValueError(
            f"timeline stage ID mismatch for {operation_id}.{raw_name}"
        )
    start = stage.get("start")
    end = stage.get("end")
    span = stage.get("span_cycles")
    if (
        type(start) is not int
        or type(end) is not int
        or type(span) is not int
        or start <= 0
        or end < start
        or span != end - start
    ):
        raise ValueError(f"timeline stage interval for {operation_id}.{raw_name}")
    return semantic.stage_id, stage


def _operation_rows(
    case_id: str,
    operation: dict[str, Any],
    world_size: int,
) -> list[dict[str, Any]]:
    operation_id = operation.get("operation_id")
    if operation_id not in PERFORMANCE_OPERATIONS:
        raise ValueError(f"timeline operation ID: {operation_id!r}")
    _nonnegative_int_mapping(operation.get("work_counts"), "timeline work counts")
    _nonnegative_int_mapping(
        operation.get("logical_byte_components"),
        "timeline logical byte components",
    )
    operation_ranks = operation.get("per_rank")
    if not isinstance(operation_ranks, list) or not operation_ranks:
        raise ValueError("timeline operation rank records")
    operation_by_rank = {
        record.get("rank"): record
        for record in operation_ranks
        if isinstance(record, dict)
    }
    if set(operation_by_rank) != set(range(world_size)):
        raise ValueError("timeline operation rank records")

    profile = operation.get("stage_profile")
    if not isinstance(profile, dict):
        raise ValueError("timeline stage profile")
    rank_profiles = profile.get("per_rank")
    if not isinstance(rank_profiles, list) or not rank_profiles:
        raise ValueError("timeline rank profiles")
    profile_by_rank = {
        rank_profile.get("rank"): rank_profile
        for rank_profile in rank_profiles
        if isinstance(rank_profile, dict)
    }
    if set(profile_by_rank) != set(range(world_size)):
        raise ValueError("timeline rank profiles")

    rows = []
    for rank in range(world_size):
        operation_rank = operation_by_rank[rank]
        rank_work_counts = _nonnegative_int_mapping(
            operation_rank.get("work_counts"), "timeline work counts"
        )
        logical_components = _nonnegative_int_mapping(
            operation_rank.get("logical_byte_components"),
            "timeline logical byte components",
        )
        rank_profile = profile_by_rank[rank]
        host_timeline_ns = _nonnegative_int_mapping(
            rank_profile.get("host_timeline_ns"), "timeline host nanoseconds"
        )
        operation_timeline = _timeline_cycles(
            rank_profile.get("device_timeline_cycles")
        )
        raw_stages = rank_profile.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("timeline rank stages")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw_stage in raw_stages:
            stage_id, validated = _validate_stage(operation_id, raw_stage)
            grouped.setdefault(stage_id, []).append(validated)

        for semantic in operation_stage_semantics(operation_id):
            stage_segments = grouped.pop(semantic.stage_id, [])
            if semantic.independently_timed and not stage_segments:
                raise ValueError(
                    f"timeline missing stage ID {semantic.stage_id} for "
                    f"{operation_id} rank {rank}"
                )
            interval = _stage_interval(stage_segments)
            rows.append({
                "case_id": case_id,
                "operation_id": operation_id,
                "rank": rank,
                "stage_id": semantic.stage_id,
                "short_name": semantic.short_name,
                "measurement_status": (
                    "measured" if stage_segments
                    else "not_independently_timed"
                ),
                "raw_stages": [stage["name"] for stage in stage_segments],
                **interval,
                "operation_timeline_cycles": dict(operation_timeline),
                "host_timeline_ns": dict(host_timeline_ns),
                "work_counts": {
                    key: rank_work_counts[key]
                    for key in semantic.work_count_keys
                },
                "logical_byte_components": dict(logical_components),
                "ascend_functions": list(semantic.ascend_functions),
                "cuda_counterpart": semantic.cuda_counterpart,
            })
        if grouped:
            raise ValueError(
                f"timeline duplicate or unexpected stage IDs: {sorted(grouped)}"
            )
    return rows


def build_timeline_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("timeline schema_version")
    if report.get("platform") != "ascend":
        raise ValueError("timeline platform")
    protocol = report.get("execution_protocol")
    if not isinstance(protocol, dict) or protocol.get("stage_profile") != 1:
        raise ValueError("timeline execution_protocol.stage_profile")
    world_size = report.get("world_size")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("timeline world_size")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("timeline cases")

    rows = []
    order = {name: index for index, name in enumerate(PERFORMANCE_OPERATIONS)}
    for case in cases:
        if not isinstance(case, dict) or case.get("status") != "passed":
            raise ValueError("timeline case status")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("timeline case identity")
        operations = case.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("timeline operations")
        operation_ids = [operation.get("operation_id") for operation in operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("timeline duplicate operations")
        for operation in sorted(
            operations, key=lambda value: order.get(value.get("operation_id"), 99)
        ):
            rows.extend(_operation_rows(case_id, operation, world_size))

    return {
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "source": {
            key: report.get(key)
            for key in (
                "schema_version", "platform", "git_commit", "world_size",
                "workload_fingerprint",
            )
        },
        "rows": rows,
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _named_values(values: dict[str, int], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return ", ".join(
        f"{name}={value}{suffix}" for name, value in sorted(values.items())
    ) or "-"


def render_timeline_markdown(report: dict[str, Any]) -> str:
    timeline = (
        report if report.get("timeline_schema_version") == TIMELINE_SCHEMA_VERSION
        else build_timeline_report(report)
    )
    lines = [
        "# Ascend EPv2 P5 timeline",
        "",
        "Device timestamps remain in device cycles. Host intervals are shown "
        "in milliseconds.",
        "",
        (
            "| Case | Operation | Rank | Stage | Status | Device interval | "
            "Active / idle / overlap | Host | Work | Logical bytes | "
            "Ascend implementation | CUDA counterpart |"
        ),
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in timeline["rows"]:
        if row["start_cycles"] is None:
            device_interval = "not independently timed"
        else:
            device_interval = (
                f"{row['start_cycles']}-{row['end_cycles']} cycles"
            )
        stage_cycles = (
            f"{row['active_cycles']} / {row['idle_cycles']} / "
            f"{row['overlap_cycles']} cycles"
        )
        host = ", ".join(
            f"{name}={value / 1_000_000:.3f} ms"
            for name, value in sorted(row["host_timeline_ns"].items())
        )
        values = (
            row["case_id"],
            row["operation_id"],
            row["rank"],
            f"{row['stage_id']} {row['short_name']}",
            row["measurement_status"],
            device_interval,
            stage_cycles,
            host,
            _named_values(row["work_counts"]),
            _named_values(row["logical_byte_components"], "bytes"),
            ", ".join(row["ascend_functions"]),
            row["cuda_counterpart"],
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render an Ascend EPv2 stage profile timeline"
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.format == "json":
        print(json.dumps(build_timeline_report(report), indent=2, sort_keys=True))
    else:
        print(render_timeline_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
