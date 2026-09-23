"""Align multi-rank Ascend stage profiles on one host timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tests.ascend.benchmark.report import SCHEMA_VERSION
from tests.utils.ep_benchmark_core import PERFORMANCE_OPERATIONS


CRITICAL_PATH_SCHEMA_VERSION = 3
DISPATCH_OPERATIONS = frozenset({"dispatch", "expanded_dispatch", "cached_dispatch"})
DEFAULT_CYCLE_HZ = 1_000_000_000
ANCHORS = {
    **{
        operation: ("dispatch_synchronize_end_ns", "dispatch synchronize")
        for operation in DISPATCH_OPERATIONS
    },
    "combine": ("combine_completion_end_ns", "combine completion"),
    "reduced_combine": ("combine_completion_end_ns", "combine completion"),
}


def _int(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(label)
    return value


def _host_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("critical path host timeline")
    return {
        name: _int(item, f"critical path host timeline.{name}")
        for name, item in value.items()
        if isinstance(name, str) and name
    }


def _select_operation(case: dict[str, Any], operation_id: str | None) -> dict[str, Any]:
    operations = case.get("operations")
    if not isinstance(operations, list):
        raise ValueError("critical path operations")
    selected = [
        operation
        for operation in operations
        if operation_id is None or operation.get("operation_id") == operation_id
    ]
    if len(selected) != 1:
        raise ValueError("critical path operation must select exactly one operation")
    return selected[0]


def _rank_profiles(operation: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    profile = operation.get("stage_profile")
    ranks = profile.get("per_rank") if isinstance(profile, dict) else None
    if not isinstance(ranks, list):
        raise ValueError("critical path rank profiles")
    by_rank = {rank.get("rank"): rank for rank in ranks if isinstance(rank, dict)}
    if set(by_rank) != set(range(world_size)):
        raise ValueError("critical path rank profile ranks")
    return [by_rank[rank] for rank in range(world_size)]


def _stage_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    stages = profile.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("critical path stages")
    rows = []
    for stage in stages:
        name = stage.get("name")
        start = _int(stage.get("start"), f"critical path stage {name}.start", positive=True)
        end = _int(stage.get("end"), f"critical path stage {name}.end")
        if end < start:
            raise ValueError(f"critical path stage {name} interval")
        rows.append({"name": name, "start": start, "end": end, "span_cycles": end - start})
    return rows


def _peer_rows(profile: dict[str, Any], key: str, field: str) -> list[dict[str, int]]:
    records = profile.get(key)
    if not isinstance(records, list):
        return []
    rows = []
    for record in records:
        peer = _int(record.get("world_rank"), f"critical path {key}.world_rank")
        cycles = _int(record.get(field), f"critical path {key}.{field}")
        rows.append({"peer": peer, "cycles": cycles})
    return rows


def _diagnostic_cycles(
    profile: dict[str, Any], start_name: str, end_name: str, cycle_hz: int
) -> int | None:
    start = profile.get(start_name)
    end = profile.get(end_name)
    if (
        type(start) is not int or type(end) is not int or end < start or
        end - start > 100_000_000_000
    ):
        return None
    return _cycles_to_ns(end - start, cycle_hz)


def _cycles_to_ns(cycles: int, cycle_hz: int) -> int:
    return int(cycles * 1_000_000_000 // cycle_hz)


def _common_ns(rank: dict[str, Any], cycles: int) -> int:
    return rank["anchor_ns"] - _cycles_to_ns(
        rank["device_last_cycles"] - cycles, rank["cycle_hz"]
    )


def _critical_path_candidates(ranks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [{
        "component": "host_anchor_spread",
        "span_ns": max(rank["anchor_ns"] for rank in ranks)
        - min(rank["anchor_ns"] for rank in ranks),
    }]
    for rank in ranks:
        candidates.append({
            "component": f"rank{rank['rank']}:timeline_idle",
            "span_ns": rank["idle_ns"],
        })
        if "acquire_vf_ns" in rank:
            candidates.append({
                "component": f"rank{rank['rank']}:epilogue_acquire_vf",
                "span_ns": rank["acquire_vf_ns"],
            })
        if "validate_vf_ns" in rank:
            candidates.append({
                "component": f"rank{rank['rank']}:epilogue_validate_vf",
                "span_ns": rank["validate_vf_ns"],
            })
        stage = max(rank["stages"], key=lambda item: item["span_cycles"])
        candidates.append({
            "component": f"rank{rank['rank']}:{stage['name']}",
            "span_ns": _cycles_to_ns(stage["span_cycles"], rank["cycle_hz"]),
        })
    return sorted(candidates, key=lambda item: item["span_ns"], reverse=True)[:8]


def _diagnostic_intervals(
    ranks: list[dict[str, Any]], name: str
) -> dict[str, int]:
    values = [rank[name] for rank in ranks if name in rank]
    if not values:
        return {}
    # Diagnostics are per-NPU cycle differences. Reject accidental subtraction
    # of two unrelated timestamps (for example, a race from an old probe).
    if any(value < 0 or value > 100_000_000_000 for value in values):
        return {}
    return {
        "min_ns": min(values),
        "max_ns": max(values),
        "spread_ns": max(values) - min(values),
    }


def build_critical_path_report(
    report: dict[str, Any],
    operation_id: str | None = None,
    *,
    cycle_hz: int = DEFAULT_CYCLE_HZ,
) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("platform") != "ascend":
        raise ValueError("critical path source report")
    protocol = report.get("execution_protocol")
    if not isinstance(protocol, dict) or protocol.get("stage_profile") != 1:
        raise ValueError("critical path requires --profile-stages")
    cycle_hz = _int(cycle_hz, "critical path cycle_hz", positive=True)
    world_size = _int(report.get("world_size"), "critical path world_size", positive=True)
    case = report.get("cases", [])[0]
    if case.get("status") != "passed":
        raise ValueError("critical path case status")
    operation = _select_operation(case, operation_id)
    operation_id = operation.get("operation_id")
    anchor_name, anchor_label = ANCHORS[operation_id]
    summary = operation.get("device_seconds")
    if not isinstance(summary, dict) or type(summary.get("mean")) not in (int, float):
        raise ValueError("critical path device timing summary")
    summary_mean_ms = float(summary["mean"]) * 1000.0

    ranks = []
    for rank, profile in enumerate(_rank_profiles(operation, world_size)):
        host = _host_mapping(profile.get("host_timeline_ns"))
        if anchor_name not in host:
            raise ValueError(f"critical path missing host anchor {anchor_name}")
        stages = _stage_rows(profile)
        acquire_vf_ns = _diagnostic_cycles(
            profile,
            "acquire_vf_start_cycles", "acquire_vf_end_cycles", cycle_hz)
        acquire_wait_ns = _diagnostic_cycles(
            profile,
            "acquire_wait_start_cycles", "acquire_wait_end_cycles", cycle_hz)
        validate_vf_ns = _diagnostic_cycles(
            profile,
            "validate_vf_start_cycles", "validate_vf_end_cycles", cycle_hz)
        diagnostics = {
            name: value
            for name, value in (
                ("acquire_vf_ns", acquire_vf_ns),
                ("acquire_wait_ns", acquire_wait_ns),
                ("validate_vf_ns", validate_vf_ns),
            )
            if value is not None
        }
        ranks.append({
            "rank": rank,
            "anchor_name": anchor_label,
            "anchor_ns": host[anchor_name],
            "cycle_hz": cycle_hz,
            "device_first_cycles": min(stage["start"] for stage in stages),
            "device_last_cycles": max(stage["end"] for stage in stages),
            "stages": stages,
            **diagnostics,
            "acquire_peer_diagnostics": _peer_rows(
                profile, "acquire_peer_diagnostics", "first_ready_cycles"
            ),
            "release_peer_publish_diagnostics": _peer_rows(
                profile, "release_peer_publish_diagnostics", "publish_cycles"
            ),
        })

    first_anchor = min(rank["anchor_ns"] for rank in ranks)
    for rank in ranks:
        rank["anchor_offset_ns"] = rank["anchor_ns"] - first_anchor
        rank["timeline"] = [{
            "name": stage["name"],
            "common_start_ns": _common_ns(rank, stage["start"]),
            "common_end_ns": _common_ns(rank, stage["end"]),
            "span_ns": _cycles_to_ns(stage["span_cycles"], rank["cycle_hz"]),
        } for stage in rank["stages"]]
        intervals = sorted(
            (item["common_start_ns"], item["common_end_ns"])
            for item in rank["timeline"]
        )
        active = 0
        merged_start, merged_end = intervals[0]
        for start, end in intervals[1:]:
            if start > merged_end:
                active += merged_end - merged_start
                merged_start, merged_end = start, end
            else:
                merged_end = max(merged_end, end)
        active += merged_end - merged_start
        envelope = merged_end - min(start for start, _ in intervals)
        rank["active_ns"] = active
        rank["idle_ns"] = envelope - active

    intervals = [item for rank in ranks for item in rank["timeline"]]
    common_start = min(item["common_start_ns"] for item in intervals)
    common_end = max(item["common_end_ns"] for item in intervals)
    for rank in ranks:
        for item in rank["timeline"]:
            item["relative_start_ns"] = item["common_start_ns"] - common_start
            item["relative_end_ns"] = item["common_end_ns"] - common_start
    latest_rank = max(
        ranks,
        key=lambda rank: (
            max(item["common_end_ns"] for item in rank["timeline"]), rank["rank"]
        ),
    )
    return {
        "critical_path_schema_version": CRITICAL_PATH_SCHEMA_VERSION,
        "source": {
            key: report.get(key)
            for key in ("schema_version", "platform", "git_commit", "world_size")
        },
        "operation_id": operation_id,
        "anchor": {
            "name": anchor_label,
            "spread_ns": max(rank["anchor_ns"] for rank in ranks) - first_anchor,
            "cycle_hz": cycle_hz,
        },
        "common_timeline": {
            "start_ns": common_start,
            "end_ns": common_end,
            "span_ns": common_end - common_start,
        },
        "latest_rank": latest_rank["rank"],
        "critical_path_candidates": _critical_path_candidates(ranks),
        "diagnostic_intervals": {
            name: summary
            for name in (
                "acquire_vf_ns", "acquire_wait_ns", "validate_vf_ns",
            )
            if (summary := _diagnostic_intervals(ranks, name))
        },
        "ranks": ranks,
        "operation_summary": {
            "mean_ms": summary_mean_ms,
            "active_ns": max(rank["active_ns"] for rank in ranks),
            "idle_ns": max(rank["idle_ns"] for rank in ranks),
        },
}


def _format_ns(value: int) -> str:
    return f"{value / 1_000_000:.3f} ms"


def _format_cycles(value: int) -> str:
    return f"{value / 1_000_000:.3f} M cycles"


def _peer_summary(rows: list[dict[str, int]]) -> str:
    if not rows:
        return "-"
    return ", ".join(f"peer{row['peer']}={_format_cycles(row['cycles'])}" for row in rows[:8])


def render_critical_path_markdown(critical_path: dict[str, Any]) -> str:
    anchor = critical_path["anchor"]
    timeline = critical_path["common_timeline"]
    summary = critical_path["operation_summary"]
    lines = [
        "# Ascend EP critical path profile",
        "",
        f"Operation: {critical_path['operation_id']}",
        "",
        f"Host anchor: {anchor['name']}; spread: {_format_ns(anchor['spread_ns'])}; "
        f"cycle rate: {anchor['cycle_hz']} Hz",
        "",
        f"Common timeline span: {_format_ns(timeline['span_ns'])}; "
        f"latest rank: {critical_path['latest_rank']}",
        "",
        f"Benchmark mean: {summary['mean_ms']:.3f} ms; "
        f"max measured stage activity: {_format_ns(summary['active_ns'])}; "
        f"max unattributed gap: {_format_ns(summary['idle_ns'])}",
        "",
        "Common-timeline values are host-anchored nanoseconds. Absolute device "
        "cycle counters are never compared across NPUs.",
        "",
        "## Critical-path candidates",
        "",
        "| Component | Span |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {item['component']} | {_format_ns(item['span_ns'])} |"
        for item in critical_path["critical_path_candidates"]
    )
    lines.extend([
        "## Epilogue diagnostics",
        "",
    ])
    diagnostics = critical_path.get("diagnostic_intervals", {})
    if diagnostics:
        lines.extend([
            "| Diagnostic | Min | Max | Rank spread |",
            "| --- | ---: | ---: | ---: |",
        ])
        labels = {
            "acquire_vf_ns": "acquire VF",
            "acquire_wait_ns": "acquire wait",
            "validate_vf_ns": "validate VF",
        }
        lines.extend(
            f"| {labels[name]} | {_format_ns(item['min_ns'])} | "
            f"{_format_ns(item['max_ns'])} | {_format_ns(item['spread_ns'])} |"
            for name, item in diagnostics.items() if name in labels
        )
    else:
        lines.append("No acquire/validate VF diagnostics in this report.")
    lines.extend([
        "",
        "## Rank summary",
        "",
        "| Rank | Anchor offset | Device envelope | Major stages | Epilogue diagnostics | Latest acquire peers | Release publish |",
        "| ---: | ---: | ---: | --- | --- | --- | --- |",
    ])
    for rank in critical_path["ranks"]:
        major = sorted(rank["timeline"], key=lambda item: item["span_ns"], reverse=True)[:5]
        envelope = rank["device_last_cycles"] - rank["device_first_cycles"]
        diagnostics = " ".join(
            f"{label}={_format_ns(rank[name])}"
            for label, name in (
                ("acquire VF", "acquire_vf_ns"),
                ("acquire wait", "acquire_wait_ns"),
                ("validate VF", "validate_vf_ns"),
            ) if name in rank
        )
        lines.append(
            f"| {rank['rank']} | {_format_ns(rank['anchor_offset_ns'])} | "
            f"{_format_cycles(envelope)} | "
            + ", ".join(f"{stage['name']}={_format_ns(stage['span_ns'])}" for stage in major)
            + " | "
            + (diagnostics if diagnostics else "-")
            + " | "
            + f"{_peer_summary(rank['acquire_peer_diagnostics'])} | "
            f"{_peer_summary(rank['release_peer_publish_diagnostics'])} |"
        )
    lines.extend([
        "",
        "## Common timeline",
        "",
        "| Rank | Stage | Start | End | Span |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    rows = [
        (rank["rank"], stage)
        for rank in critical_path["ranks"]
        for stage in rank["timeline"]
    ]
    rows.sort(key=lambda item: item[1]["common_start_ns"])
    lines.extend(
        f"| {rank} | {stage['name']} | {_format_ns(stage['relative_start_ns'])} | "
        f"{_format_ns(stage['relative_end_ns'])} | {_format_ns(stage['span_ns'])} |"
        for rank, stage in rows
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align Ascend EP stage profiles on a common host timeline"
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--operation", choices=PERFORMANCE_OPERATIONS, default="dispatch")
    parser.add_argument("--cycle-hz", type=int, default=DEFAULT_CYCLE_HZ)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    critical_path = build_critical_path_report(
        report, args.operation, cycle_hz=args.cycle_hz
    )
    if args.format == "json":
        print(json.dumps(critical_path, indent=2, sort_keys=True))
    else:
        print(render_critical_path_markdown(critical_path), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
