#!/usr/bin/env python3
"""Analyze DeepEP Ascend stage-profile benchmark reports.

The script intentionally uses only the Python standard library.  It accepts
the benchmark JSON written by tests/ascend/benchmark/bench_ep.py and can also
compare several reports in one invocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _ms(value: int | float | None) -> float | None:
    return None if value is None else float(value) / 1_000_000.0


def _first_case(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("report has no cases")
    for case in cases:
        if case.get("status") == "passed":
            return case
    raise ValueError("report has no passed case")


def _profiles(report: dict[str, Any], operation: str) -> tuple[str, list[dict[str, Any]]]:
    case = _first_case(report)
    for record in case.get("operations", []):
        if record.get("operation_id") != operation:
            continue
        profile = record.get("stage_profile")
        if not isinstance(profile, dict):
            raise ValueError(f"{operation}: stage_profile is missing")
        rows = profile.get("per_rank")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{operation}: stage_profile.per_rank is missing")
        return str(case.get("case_id", "unknown")), sorted(rows, key=lambda r: r.get("rank", 0))
    raise ValueError(f"operation not found: {operation}")


def _stage_rows(profile: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for stage in profile.get("stages", []):
        blocks = stage.get("blocks", [])
        starts = [b.get("start") for b in blocks if isinstance(b, dict)]
        ends = [b.get("end") for b in blocks if isinstance(b, dict)]
        valid = starts and len(starts) == len(ends) and all(
            isinstance(a, int) and isinstance(b, int) and b >= a
            for a, b in zip(starts, ends)
        )
        yield {
            "name": stage.get("name", "?"),
            "block_count": stage.get("block_count", len(blocks)),
            "span_cycles": max(ends) - min(starts) if valid else None,
            "work_counts": stage.get("work_counts", {}),
        }


def _acquire_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for rank_row in rows:
        rank = rank_row.get("rank")
        peers = rank_row.get("acquire_peer_diagnostics") or []
        result.append({
            "rank": rank,
            "peers": [
                {"source": peer.get("world_rank"), "first_ready_cycles": peer.get("first_ready_cycles")}
                for peer in peers
            ],
            "wait_start_cycles": rank_row.get("acquire_wait_start_cycles"),
            "wait_end_cycles": rank_row.get("acquire_wait_end_cycles"),
        })
    return result


def _publication_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for rank_row in rows:
        result.append({
            "rank": rank_row.get("rank"),
            "peers": [
                {"destination": peer.get("world_rank"), "publish_cycles": peer.get("publish_cycles")}
                for peer in (rank_row.get("release_peer_publish_diagnostics") or [])
            ],
        })
    return result


def analyze(report: dict[str, Any], operation: str, cycles_per_ns: float) -> dict[str, Any]:
    case_id, rows = _profiles(report, operation)
    entries = [
        row.get("host_timeline_ns", {}).get("dispatch_entry_ns")
        for row in rows
        if isinstance(row.get("host_timeline_ns"), dict)
    ]
    entries = [value for value in entries if isinstance(value, int)]
    entry_base = min(entries) if entries else None
    rank_summary = []
    for row in rows:
        host = row.get("host_timeline_ns", {})
        entry = host.get("dispatch_entry_ns") if isinstance(host, dict) else None
        normalized_ready = None
        wait_end = row.get("acquire_wait_end_cycles")
        if entry_base is not None and isinstance(entry, int) and isinstance(wait_end, int):
            normalized_ready = ((entry - entry_base) / 1_000_000 +
                                _ms(wait_end / cycles_per_ns))
        rank_summary.append({
            "rank": row.get("rank"),
            "entry_skew_ms": _ms(entry - entry_base) if entry_base is not None and isinstance(entry, int) else None,
            "acquire_wait_ms": _ms(wait_end),
            "normalized_ready_ms": normalized_ready,
            "device_envelope_cycles": row.get("device_timeline_cycles", {}).get("envelope_cycles"),
            "stages": list(_stage_rows(row)),
        })
    normalized = [r["normalized_ready_ms"] for r in rank_summary if r["normalized_ready_ms"] is not None]
    stage_candidates: dict[str, dict[str, Any]] = {}
    for row in rank_summary:
        for stage in row["stages"]:
            span = stage["span_cycles"]
            if not isinstance(span, int):
                continue
            candidate = stage_candidates.setdefault(
                stage["name"], {"max_span_cycles": 0, "rank": row["rank"], "block_count": stage["block_count"]})
            if span > candidate["max_span_cycles"]:
                candidate.update(max_span_cycles=span, rank=row["rank"], block_count=stage["block_count"])
    return {
        "case_id": case_id,
        "operation": operation,
        "world_size": len(rows),
        "entry_spread_ms": _ms(max(entries) - min(entries)) if entries else None,
        "normalized_ready_ms": {
            "min": min(normalized) if normalized else None,
            "max": max(normalized) if normalized else None,
            "spread": (max(normalized) - min(normalized)) if normalized else None,
        },
        "per_rank": rank_summary,
        # These are candidates, not a proven dependency chain. Stages may
        # overlap; the final critical path still needs stream/event evidence.
        "critical_path_candidates": sorted(
            [dict(name=name, **values) for name, values in stage_candidates.items()],
            key=lambda value: value["max_span_cycles"], reverse=True,
        ),
        "acquire_first_ready": _acquire_matrix(rows),
        "release_publication": _publication_matrix(rows),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Ascend profile analysis: {summary['case_id']}",
        "",
        f"Operation: `{summary['operation']}`; ranks: {summary['world_size']}",
        f"Host entry spread: {summary['entry_spread_ms']:.3f} ms" if summary["entry_spread_ms"] is not None else "Host entry spread: n/a",
        "",
        "| Rank | Entry skew (ms) | Acquire wait (ms) | Ready after entry normalization (ms) | Envelope (cycles) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_rank"]:
        values = [row["rank"], row["entry_skew_ms"], row["acquire_wait_ms"], row["normalized_ready_ms"], row["device_envelope_cycles"]]
        rendered = ["-" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value)) for value in values]
        lines.append("| " + " | ".join(rendered) + " |")
    lines += ["", "## Stage block counts", "", "| Rank | Stage | Blocks | Span (cycles) | Work counts |", "| ---: | --- | ---: | ---: | --- |"]
    for row in summary["per_rank"]:
        for stage in row["stages"]:
            lines.append(f"| {row['rank']} | {stage['name']} | {stage['block_count']} | {stage['span_cycles'] if stage['span_cycles'] is not None else '-'} | {json.dumps(stage['work_counts'], sort_keys=True)} |")
    lines += ["", "## Critical-path candidates", "", "Stage spans are candidates only; overlap and stream dependencies must be checked separately.", "", "| Stage | Max span (cycles) | Rank | Blocks |", "| --- | ---: | ---: | ---: |"]
    for candidate in summary["critical_path_candidates"]:
        lines.append(f"| {candidate['name']} | {candidate['max_span_cycles']} | {candidate['rank']} | {candidate['block_count']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--operation", default="dispatch")
    parser.add_argument(
        "--cycles-per-ns", type=float, default=1.0,
        help="device cycle to nanosecond conversion used for normalized ready time (default: 1)",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    outputs = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if args.cycles_per_ns <= 0:
            parser.error("--cycles-per-ns must be positive")
        result = analyze(report, args.operation, args.cycles_per_ns)
        result["source"] = str(path)
        outputs.append(result)
    if args.format == "json":
        print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2, sort_keys=True))
    else:
        print("\n".join(_markdown(result).rstrip() for result in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
