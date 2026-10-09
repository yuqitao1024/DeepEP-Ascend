#!/usr/bin/env python3
"""Align current DeepEP-Ascend reporting with the official EP benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


OFFICIAL_OPERATION_MAP = {
    "dispatch": "expanded_dispatch",
    "combine": "reduced_combine",
}

OFFICIAL_REFERENCE_GBPS = {
    "dispatch": {"minimum": 373.0, "mean": 374.0, "maximum": 375.0},
    "combine": {"minimum": 345.0, "mean": 346.0, "maximum": 347.0},
}


def _select_operation(report: dict[str, Any], operation_id: str) -> dict[str, Any]:
    if "operations" in report:
        case = report
    else:
        case = report.get("cases", [{}])[0]
    matches = [
        operation
        for operation in case.get("operations", [])
        if operation.get("operation_id") == operation_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {operation_id!r} operation, got {len(matches)}"
        )
    return matches[0]


def _require_stage_profile(report: dict[str, Any], operation_id: str) -> dict[str, Any]:
    profile = _select_operation(report, operation_id).get("stage_profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{operation_id}.stage_profile is missing")
    return profile


def _select_rank_msprof(report: dict[str, Any], operation: str, rank: int) -> dict[str, Any]:
    rows = report["ranks"][rank]
    matches = [row for row in rows if row.get("official_operation") == operation]
    if len(matches) != 1:
        raise ValueError(f"msprof rank {rank} has {len(matches)} {operation} rows")
    return matches[0]


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("statistics require at least one value")
    ordered = sorted(values)
    return {
        "minimum": min(values),
        "mean": statistics.mean(values),
        "maximum": max(values),
        "p50": ordered[(len(ordered) - 1) // 2],
    }


def _gbps(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError("time must be positive")
    return num_bytes / seconds / 1e9


def _stage_clock_calibration(
    stage_operation: dict[str, Any], profile: dict[str, Any]
) -> float:
    envelope_cycles = profile.get("device_timeline_cycles", {}).get(
        "envelope_cycles"
    )
    mean_seconds = stage_operation.get("device_seconds", {}).get("mean")
    if (
        not isinstance(envelope_cycles, int)
        or envelope_cycles <= 0
        or not isinstance(mean_seconds, (int, float))
        or mean_seconds <= 0
    ):
        raise ValueError("stage clock calibration inputs are unavailable")
    # The stage JSON contains cycle counts but no explicit clock frequency.
    # Calibrate it from the event-timed device envelope and the captured
    # envelope-cycle count. This is approximately 0.9-1.0 GHz on this device.
    return envelope_cycles / mean_seconds


def _stage_time_seconds(
    stage_operation: dict[str, Any], profile: dict[str, Any], cycles: int
) -> float:
    return cycles / _stage_clock_calibration(stage_operation, profile)


def _official_bytes(stage_operation: dict[str, Any]) -> int:
    ranks = stage_operation.get("per_rank", [])
    if ranks:
        byte_values = [
            rank.get("logical_bytes", {}).get("scaleup", 0)
            for rank in ranks
        ]
    else:
        # Compact stage summaries omit per-rank counters. Fall back to the
        # aggregated scaleup byte total divided by the report world size.
        total = stage_operation.get("logical_bytes", {}).get("scaleup", 0)
        world_size = stage_operation.get("world_size", 8)
        byte_values = [total // world_size]
    if any(value <= 0 for value in byte_values):
        raise ValueError("scaleup logical bytes are missing")

    # For this single-node EP8 workload, the official formula is equivalent to
    # received_records * logical bytes-per-token. Use the exact per-rank byte
    # count from the stage report to avoid rounding hidden/topk route effects.
    return int(statistics.mean(byte_values))


def _stage_mappings(
    stage_operation: dict[str, Any], profile: dict[str, Any]
) -> dict[str, dict[str, float]]:
    phases = profile.get("phase_cycles", {})
    pipeline = profile.get("pipeline_cycles", {})
    if not phases or not pipeline:
        raise ValueError("phase/pipeline cycles are missing")

    producer = phases.get("producer", 0)
    cq_wait = phases.get("cq_wait", 0)
    network = pipeline.get("network", 0)
    service_issue_drain = phases.get("service_submit", 0) + cq_wait
    mappings = {
        "stage_service_issue_drain": service_issue_drain,
        "stage_cq_wait": cq_wait,
        "stage_network": network,
        "stage_producer": producer,
        "stage_producer_plus_network": producer + network,
    }
    return {
        name: {
            "cycles": cycles,
            "seconds": _stage_time_seconds(stage_operation, profile, cycles),
        }
        for name, cycles in mappings.items()
    }


def build_report(msprof_path: Path, stage_path: Path) -> dict[str, Any]:
    msprof = json.loads(msprof_path.read_text(encoding="utf-8"))
    stage = json.loads(stage_path.read_text(encoding="utf-8"))

    stage_case_id = stage.get("case_id", stage.get("cases", [{}])[0].get("case_id"))
    if msprof.get("case_id") != stage_case_id:
        raise ValueError("msprof and stage reports use different case IDs")

    operations: list[dict[str, Any]] = []
    for official_name, current_name in OFFICIAL_OPERATION_MAP.items():
        stage_op = _select_operation(stage, current_name)
        stage_profile = _require_stage_profile(stage, current_name)
        num_bytes = _official_bytes(stage_op)

        rank_msprof = []
        for rank in range(len(msprof.get("ranks", []))):
            row = _select_rank_msprof(msprof, official_name, rank)
            rank_msprof.append({
                "rank": rank,
                "kernel_us": row["urma"]["dur_us"],
                "bytes": int(row["urma_bytes"]),
                "reported_gbps": float(row["urma_gbps"]),
                "msprof_gbps": _gbps(int(row["urma_bytes"]), row["urma"]["dur_us"] / 1e6),
            })

        mappings: dict[str, Any] = {
            "msprof_raw_outer_kernel_time": {
                "seconds_per_rank": [
                    row["kernel_us"] / 1e6 for row in rank_msprof
                ],
                "description": (
                    "Raw dispatch_kernel/combine_kernel wall time. This outer "
                    "boundary is retained for msprof diagnostics only; the "
                    "staged transport service executes across later kernels."
                ),
            }
        }
        for mapping_name, timing in _stage_mappings(stage_op, stage_profile).items():
            mappings[mapping_name] = {
                "seconds": timing["seconds"],
                "cycles": timing["cycles"],
                "description": (
                    "Single aggregated stage-profile observation; not 50-sample msprof statistics"
                ),
            }

        bandwidths: dict[str, Any] = {}
        for mapping_name, mapping in mappings.items():
            if "seconds_per_rank" in mapping:
                seconds = mapping["seconds_per_rank"]
            else:
                seconds = [mapping["seconds"]]
            bandwidths[mapping_name] = {
                "gbps": _statistics([
                    _gbps(num_bytes if len(seconds) == 1 else rank["bytes"], value)
                    for rank, value in zip(rank_msprof, seconds)
                ]),
                "time_seconds": _statistics(seconds),
            }

        operations.append({
            "official_operation": official_name,
            "current_operation": current_name,
            "bytes_per_rank": num_bytes,
            "msprof_rank_data": rank_msprof,
            "stage_cycles": stage_profile.get("pipeline_cycles", {}),
            "stage_phase_cycles": stage_profile.get("phase_cycles", {}),
            "stage_clock_calibration_hz": _stage_clock_calibration(
                stage_op, stage_profile
            ),
            "bandwidth_mappings": bandwidths,
            "official_reference_gbps": OFFICIAL_REFERENCE_GBPS[official_name],
        })

    return {
        "schema_version": 1,
        "alignment_scope": "reporting-only",
        "case_id": stage_case_id,
        "workload": msprof.get("workload"),
        "timing_protocol": msprof.get("timing_protocol"),
        "operations": operations,
        "comparability_caveats": {
            "kernel_boundary": (
                "The current implementation has no single msprof kernel that "
                "matches official URMA issue-and-drain. producer_release only "
                "appends transport commands; use "
                "stage_service_issue_drain for the preferred proxy."
            ),
            "set_barrier_in_prologue": (
                "Current implementation lacks the official "
                "deep_ep._C.set_barrier_in_prologue API; the external barrier "
                "could not be moved into the communication kernel prologue."
            ),
            "defer_epilogue": (
                "Current implementation lacks the official defer_epilogue API; "
                "epilogue work cannot be split into the official separate kernels."
            ),
            "fp8_scale_factor_layout": (
                "Current expanded FP8 path uses column-major scale factors while "
                "the official public reference uses row-major. Byte count is the "
                "same for hidden=7168, but access locality is not identical."
            ),
            "stage_profile_sample_count": (
                "Stage profile has one aggregated observation, unlike msprof's "
                "50 sampled iterations; its min/mean/max represent rank/mapping "
                "variation, not iteration variance."
            ),
            "stage_clock_calibration": (
                "Stage JSON has cycle counters but no explicit frequency. "
                "Seconds are calibrated as device_timeline_cycles.envelope_cycles "
                "divided by device_seconds.mean; this is an approximation, not "
                "an independent hardware timer measurement."
            ),
            "package_version": (
                "Ignored per comparison policy; CANN/Torch package versions are "
                "not treated as implementation differences."
            ),
        },
        "notes": [
            "Bandwidth uses the official per-rank received-token byte formula.",
            "stage_service_issue_drain maps service_submit + cq_wait and is the preferred proxy for official URMA issue-and-drain.",
            "stage_network maps publication + service_submit + cq_wait + barrier_wait.",
            "stage_producer_plus_network is an upper-bound communication window.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# DeepEP-Ascend official reporting alignment",
        "",
        f"Case: `{report['case_id']}`",
        "",
        "## Bandwidth mappings",
        "",
    ]
    for operation in report["operations"]:
        official = operation["official_operation"]
        reference = operation["official_reference_gbps"]["mean"]
        lines.append(f"### {official} (`{operation['current_operation']}`)")
        lines.append("")
        lines.append(f"Bytes per rank: {operation['bytes_per_rank']:,}")
        lines.append("")
        lines.append("| Mapping | Min GB/s | Mean GB/s | Max GB/s | vs official mean |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, values in operation["bandwidth_mappings"].items():
            bandwidth = values["gbps"]
            ratio = bandwidth["mean"] / reference
            lines.append(
                f"| `{name}` | {bandwidth['minimum']:.2f} | "
                f"{bandwidth['mean']:.2f} | {bandwidth['maximum']:.2f} | "
                f"{ratio:.2f}x |"
            )
        lines.append("")

    lines.extend(["## Comparability caveats", ""])
    for name, caveat in report["comparability_caveats"].items():
        lines.append(f"- **{name}:** {caveat}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msprof", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.msprof, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = args.markdown or args.output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
