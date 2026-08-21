import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_manifest import enumerate_ep_mode_cases


ASCEND_MAX_DATA_BLOCKS = 72


def _data_blocks(value: str) -> int:
    blocks = int(value)
    if not 1 <= blocks <= ASCEND_MAX_DATA_BLOCKS:
        raise argparse.ArgumentTypeError(
            f"must be in [1, {ASCEND_MAX_DATA_BLOCKS}]")
    return blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the supported Ascend EPv2 kernel matrix",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--suite",
        choices=("all", "performance", "functional"),
        default="all",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", default="ascend-ep-benchmark.json")
    parser.add_argument("--workload-manifest")
    parser.add_argument("--dump-manifest")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--precise-unbalanced-ratio", action="store_true")
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument(
        "--allow-multiple-reduction",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--num-sms", type=_data_blocks, default=72)
    parser.add_argument("--cases")
    parser.add_argument("--skip-check", action="store_true")
    return parser


def _case_records() -> list[dict]:
    records = []
    for case in enumerate_ep_mode_cases():
        capability = classify_ascend_case(case)
        records.append({
            "case_id": case.case_id,
            "mode": asdict(case),
            "suite": capability.suite,
            "status": "supported" if capability.supported else "deferred",
            "reason": capability.reason,
        })
    return records


def list_cases(output_format: str, suite: str) -> None:
    cases = [
        case
        for case in _case_records()
        if suite == "all" or case["suite"] == suite
    ]
    summary = {
        "total": len(cases),
        "supported": sum(case["status"] == "supported" for case in cases),
        "deferred": sum(case["status"] == "deferred" for case in cases),
    }
    if output_format == "json":
        print(json.dumps({"summary": summary, "cases": cases}, sort_keys=True))
        return

    print("| # | Case ID | Suite | Ascend status | Reason |")
    print("| ---: | --- | --- | --- | --- |")
    for index, case in enumerate(cases, start=1):
        print(
            f"| {index} | `{case['case_id']}` | {case['suite']} | "
            f"{case['status']} | {case['reason']} |"
        )
    print(
        f"\nSummary: {summary['total']} total, "
        f"{summary['supported']} supported, {summary['deferred']} deferred"
    )


def _selected_case_ids(parser: argparse.ArgumentParser, value: str | None):
    cases_by_id = {
        case.case_id: case for case in enumerate_ep_mode_cases()
    }
    current_ids = tuple(
        case_id
        for case_id, case in cases_by_id.items()
        if classify_ascend_case(case).supported
    )
    if value is None:
        return current_ids
    selected = tuple(case_id.strip() for case_id in value.split(",") if case_id.strip())
    unknown = tuple(case_id for case_id in selected if case_id not in cases_by_id)
    if unknown:
        parser.error("unknown case IDs: " + ", ".join(unknown))
    if not selected:
        parser.error("at least one case ID is required")
    for case_id in selected:
        capability = classify_ascend_case(cases_by_id[case_id])
        if not capability.supported:
            parser.error(
                f"cannot benchmark {capability.suite} case {case_id}: "
                f"{capability.reason}"
            )
    return selected


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    selected_case_ids = _selected_case_ids(parser, args.cases)
    if args.list_cases:
        list_cases(args.format, args.suite)
        return 0
    if args.suite != "all":
        parser.error("--suite is only valid with --list-cases")

    from tests.ascend.benchmark.runtime import run_benchmark

    return run_benchmark(args, selected_case_ids)


if __name__ == "__main__":
    raise SystemExit(main())
