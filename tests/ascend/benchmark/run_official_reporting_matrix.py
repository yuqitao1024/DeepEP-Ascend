#!/usr/bin/env python3
"""Run the official msprof/stage reporting pair for the complete EP case matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_manifest import (
    WorkloadSpec,
    build_manifest,
    enumerate_ep_mode_cases,
    write_manifest,
)


DEFAULT_CASES = tuple(
    case.case_id
    for case in enumerate_ep_mode_cases()
    if classify_ascend_case(case).supported
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every supported EP case with the official msprof semantics "
            "and generate aligned reporting artifacts"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the manifest, per-case artifacts, and matrix summary",
    )
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help=(
            "Comma-separated case IDs. Defaults to the complete supported "
            "case matrix (currently 144 cases)"
        ),
    )
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--precise-unbalanced-ratio", action="store_true")
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--num-sms", type=int, default=64)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="Number of local ranks for every child benchmark run",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--stage-warmups", type=int, default=1)
    parser.add_argument("--stage-iterations", type=int, default=1)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=1800,
        help="Per-case timeout for each child benchmark run",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="One-based case index to start from (inclusive)",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=0,
        help="One-based case index to stop at (inclusive); 0 means the end",
    )
    parser.add_argument(
        "--indices",
        default="",
        help=(
            "Exact one-based case indices, supporting lists and ranges "
            "(for example: 1,2,3 or 100-102,105). Takes precedence over "
            "--start and --stop"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load the existing matrix summary, skip passed cases, and merge "
            "new rows by one-based case index"
        ),
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the matrix at the first failed case",
    )
    return parser


def _run_child(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "command": command,
            "returncode": 124,
            "output": output + f"\n\nTimed out after {timeout} seconds\n",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selection(args: argparse.Namespace) -> tuple[str, ...]:
    cases_by_id = {case.case_id: case for case in enumerate_ep_mode_cases()}
    selected = tuple(
        case_id.strip() for case_id in args.cases.split(",") if case_id.strip()
    )
    if not selected:
        raise ValueError("at least one case ID is required")
    unknown = [case_id for case_id in selected if case_id not in cases_by_id]
    if unknown:
        raise ValueError("unknown case IDs: " + ", ".join(unknown))
    unsupported = [
        case_id
        for case_id in selected
        if not classify_ascend_case(cases_by_id[case_id]).supported
    ]
    if unsupported:
        raise ValueError("unsupported case IDs: " + ", ".join(unsupported))
    return selected


def _parse_indices(value: str, case_count: int) -> tuple[int, ...]:
    indices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            bounds = item.split("-")
            if len(bounds) != 2:
                raise ValueError(f"invalid index range: {item}")
            try:
                first, last = (int(bound.strip()) for bound in bounds)
            except ValueError as error:
                raise ValueError(f"invalid index range: {item}") from error
            if first < 1 or last < first:
                raise ValueError(f"invalid index range: {item}")
            indices.extend(range(first, last + 1))
        else:
            try:
                index = int(item)
            except ValueError as error:
                raise ValueError(f"invalid case index: {item}") from error
            if index < 1:
                raise ValueError(f"case index must be positive: {item}")
            indices.append(index)

    if not indices:
        raise ValueError("--indices must select at least one case")
    invalid = sorted(set(index for index in indices if index > case_count))
    if invalid:
        raise ValueError(
            f"case indices out of range (1-{case_count}): "
            + ", ".join(map(str, invalid))
        )
    return tuple(dict.fromkeys(indices))


def _case_prefix(index: int, case_id: str) -> str:
    return f"{index:03d}-{case_id}"


def _base_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--num-tokens", str(args.num_tokens),
        "--hidden", str(args.hidden),
        "--num-topk", str(args.num_topk),
        "--num-experts", str(args.num_experts),
        "--seed", str(args.seed),
        "--unbalanced-ratio", str(args.unbalanced_ratio),
        "--masked-ratio", str(args.masked_ratio),
        "--num-sms", str(args.num_sms),
        "--warmups", str(args.warmups),
        "--iterations", str(args.iterations),
    ]
    if args.precise_unbalanced_ratio:
        arguments.append("--precise-unbalanced-ratio")
    if args.skip_check:
        arguments.append("--skip-check")
    return arguments


def _summarize_alignment(alignment_path: Path) -> dict[str, Any]:
    payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "case_id": payload.get("case_id"),
        "bytes_per_rank": {},
        "preferred_mapping": {},
    }
    for operation in payload.get("operations", []):
        name = operation.get("official_operation")
        summary["bytes_per_rank"][name] = operation.get("bytes_per_rank")
        mapping = operation.get("bandwidth_mappings", {}).get(
            "stage_service_issue_drain", {}
        )
        summary["preferred_mapping"][name] = mapping.get("gbps")
    return summary


def _write_markdown(
    path: Path,
    workload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    passed = sum(row["status"] == "passed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    lines = [
        "# Official-reporting EP matrix",
        "",
        "This report reuses the single-case official msprof protocol and the "
        "stage-based reporting alignment for every selected case.",
        "",
        "## Workload",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in workload.items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Case summary",
        "",
        f"- Selected: {len(rows)}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        "",
        "## Preferred proxy bandwidth",
        "",
        "The preferred proxy is stage_service_issue_drain.",
        "",
        "| Index | Case | Status | Dispatch GB/s | Combine GB/s |",
        "|---:|---|---|---:|---:|",
    ])
    for row in rows:
        dispatch = row["summary"].get("preferred_mapping", {}).get("dispatch", {})
        combine = row["summary"].get("preferred_mapping", {}).get("combine", {})
        dispatch_mean = dispatch.get("mean") if isinstance(dispatch, dict) else None
        combine_mean = combine.get("mean") if isinstance(combine, dict) else None
        lines.append(
            f"| {row['index']} | `{row['case_id']}` | {row['status']} | "
            f"{dispatch_mean:.2f} | {combine_mean:.2f} |"
            if dispatch_mean is not None and combine_mean is not None
            else f"| {row['index']} | `{row['case_id']}` | {row['status']} | N/A | N/A |"
        )
    lines.extend([
        "",
        "## Failed cases",
        "",
    ])
    if failed:
        for row in rows:
            if row["status"] == "failed":
                lines.append(f"- `{row['case_id']}`: {row['reason']}")
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _artifact_paths(
    msprof_path: Path, stage_path: Path, alignment_path: Path, log_path: Path
) -> dict[str, str]:
    return {
        "msprof": str(msprof_path),
        "stage": str(stage_path),
        "alignment": str(alignment_path),
        "log": str(log_path),
    }


def _write_matrix_summary(
    output_dir: Path,
    selected_count: int,
    rows: list[dict[str, Any]],
    manifest: Any,
) -> None:
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_count": selected_count,
        "case_summary": {
            "total": len(rows),
            "passed": sum(row["status"] == "passed" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
        },
        "workload_fingerprint": manifest.fingerprint,
        "workload": manifest.to_dict()["spec"],
        "cases": rows,
    }
    _write_json_atomic(output_dir / "matrix-summary.json", summary)
    _write_markdown(output_dir / "matrix-summary.md", summary["workload"], rows)


def run_matrix(args: argparse.Namespace) -> int:
    all_cases = _selection(args)
    if args.indices:
        selected_indices = _parse_indices(args.indices, len(all_cases))
        selected = tuple(all_cases[index - 1] for index in selected_indices)
    else:
        if args.start < 1:
            raise ValueError("--start must be at least one")
        if args.stop and args.stop < args.start:
            raise ValueError("--stop must not be less than --start")
        if args.stop:
            selected_indices = tuple(range(args.start, args.stop + 1))
            selected = all_cases[args.start - 1:args.stop]
        else:
            selected_indices = tuple(
                range(args.start, len(all_cases) + 1)
            )
            selected = all_cases[args.start - 1:]

    repository = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir
    msprof_dir = output_dir / "msprof"
    stage_dir = output_dir / "stage"
    alignment_dir = output_dir / "alignment"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, msprof_dir, stage_dir, alignment_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torchrun_executable = shutil.which("torchrun")
    torchrun = (
        [torchrun_executable]
        if torchrun_executable is not None
        else [sys.executable, "-m", "torch.distributed.run"]
    )
    launcher_arguments = [
        "--standalone", "--nproc-per-node", str(args.nproc_per_node)
    ]

    manifest_path = output_dir / "workload.json"
    manifest = build_manifest(WorkloadSpec(
        world_size=args.nproc_per_node,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        seed=args.seed,
        unbalanced_ratio=args.unbalanced_ratio,
        precise_unbalanced_ratio=args.precise_unbalanced_ratio,
        masked_ratio=args.masked_ratio,
    ))
    write_manifest(manifest_path, manifest)

    summary_path = output_dir / "matrix-summary.json"
    rows: list[dict[str, Any]] = []
    selected_count = len(selected)
    if args.resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("workload_fingerprint") != manifest.fingerprint:
            raise ValueError(
                "cannot resume: workload fingerprint does not match the "
                "existing matrix summary"
            )
        rows = list(previous.get("cases", []))
        selected_count = len(all_cases)

    existing_by_index = {row["index"]: row for row in rows}
    rows = [
        row for row in rows
        if row["index"] not in selected_indices
    ]
    skipped = {
        index for index in selected_indices
        if existing_by_index.get(index, {}).get("status") == "passed"
    }
    for position, case_id in zip(selected_indices, selected, strict=True):
        if position in skipped:
            print(f"[{position}] skipping passed {case_id}", flush=True)
            continue
        prefix = _case_prefix(position, case_id)
        msprof_path = msprof_dir / f"{prefix}.json"
        stage_path = stage_dir / f"{prefix}.json"
        alignment_path = alignment_dir / f"{prefix}.json"
        alignment_markdown_path = alignment_dir / f"{prefix}.md"
        log_path = logs_dir / f"{prefix}.log"
        selected_position = sum(
            1
            for index in selected_indices
            if index not in skipped and index <= position
        )
        print(
            f"[{position}] running {case_id} ({selected_position}/"
            f"{len(selected)})",
            flush=True,
        )

        msprof_command = torchrun + launcher_arguments + [
            str(repository / "tests/ascend/benchmark/bench_ep_msprof.py"),
            "--case", case_id,
            "--output", str(msprof_path),
        ] + _base_arguments(args)
        msprof_result = _run_child(
            msprof_command, args.case_timeout_seconds
        )
        log_path.write_text(
            "$ " + " ".join(msprof_command) + "\n\n" + msprof_result["output"],
            encoding="utf-8",
        )
        if msprof_result["returncode"] != 0:
            rows.append({
                "index": position,
                "case_id": case_id,
                "status": "failed",
                "reason": (
                    f"msprof exited with {msprof_result['returncode']}; "
                    f"log: {log_path}"
                ),
                "summary": {},
                "artifacts": _artifact_paths(
                    msprof_path, stage_path, alignment_path, log_path
                ),
                "log": str(log_path),
            })
            if args.stop_on_failure:
                break
            continue

        stage_command = torchrun + launcher_arguments + [
            str(repository / "tests/ascend/benchmark/bench_ep.py"),
            "--cases", case_id,
            "--workload-manifest", str(manifest_path),
            "--profile-stages",
            "--warmups", str(args.stage_warmups),
            "--iterations", str(args.stage_iterations),
            "--num-sms", str(args.num_sms),
            "--output", str(stage_path),
        ]
        if args.skip_check:
            stage_command.append("--skip-check")
        stage_result = _run_child(stage_command, args.case_timeout_seconds)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                "$ " + " ".join(stage_command) + "\n\n"
                + stage_result["output"]
            )
        if stage_result["returncode"] != 0:
            rows.append({
                "index": position,
                "case_id": case_id,
                "status": "failed",
                "reason": (
                    f"stage exited with {stage_result['returncode']}; "
                    f"log: {log_path}"
                ),
                "summary": {},
                "artifacts": _artifact_paths(
                    msprof_path, stage_path, alignment_path, log_path
                ),
                "log": str(log_path),
            })
            if args.stop_on_failure:
                break
            continue

        align_command = [
            sys.executable,
            str(repository / "tests/ascend/benchmark/align_official_reporting.py"),
            "--msprof", str(msprof_path),
            "--stage", str(stage_path),
            "--output", str(alignment_path),
            "--markdown", str(alignment_markdown_path),
        ]
        align_result = _run_child(align_command, args.case_timeout_seconds)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                "$ " + " ".join(align_command) + "\n\n"
                + align_result["output"]
            )
        if align_result["returncode"] != 0:
            rows.append({
                "index": position,
                "case_id": case_id,
                "status": "failed",
                "reason": (
                    f"alignment exited with {align_result['returncode']}; "
                    f"log: {log_path}"
                ),
                "summary": {},
                "artifacts": _artifact_paths(
                    msprof_path, stage_path, alignment_path, log_path
                ),
                "log": str(log_path),
            })
            if args.stop_on_failure:
                break
            continue

        rows.append({
            "index": position,
            "case_id": case_id,
            "status": "passed",
            "reason": "",
            "summary": _summarize_alignment(alignment_path),
            "artifacts": _artifact_paths(
                msprof_path, stage_path, alignment_path, log_path
            ),
            "log": str(log_path),
        })

        _write_matrix_summary(output_dir, selected_count, rows, manifest)

    rows.sort(key=lambda row: row["index"])
    _write_matrix_summary(output_dir, selected_count, rows, manifest)
    final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        f"matrix completed: {final_summary['case_summary']['passed']} passed, "
        f"{final_summary['case_summary']['failed']} failed",
        flush=True,
    )
    return 1 if final_summary["case_summary"]["failed"] else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
