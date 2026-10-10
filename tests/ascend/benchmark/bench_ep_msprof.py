import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_ep.utils.testing_msprof import bench_msprof
from tests.ascend.benchmark.bench_ep import _configure_process
from tests.ascend.benchmark.runtime import AscendRuntime, _resolve_manifest
from tests.utils.ep_benchmark_manifest import enumerate_ep_mode_cases


OFFICIAL_OPERATIONS = {
    "dispatch": "expanded_dispatch",
    "combine": "reduced_combine",
}
# The staged transport implementation has no single msprof kernel that covers
# the official URMA issue-and-drain boundary. producer_release only appends
# transport commands; service_submit/cq_wait execute later in the outer
# dispatch/combine and barrier kernels. Keep the communication kernel for raw
# msprof collection, but never report it as the official-equivalent transport
# service time.
KERNEL_MATCHES = {
    "expanded_dispatch": (
        "dispatch_kernel",
        "dispatch_epilogue_complete_kernel",
    ),
    "reduced_combine": (
        "combine_kernel",
        "combine_epilogue_complete_kernel",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the official Ascend EP workload with DeepSeek bench_msprof semantics",
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
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--case", default="ep-fp8-align128-bias0-hcopy0-prev0-async1-alloc0"
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def kernel_record(profile) -> dict[str, Any]:
    return {
        "dur_ns": profile.dur_ns,
        "dur_us": profile.dur_us,
        "aic_total_cycles": profile.aic_total_cycles,
        "aiv_total_cycles": profile.aiv_total_cycles,
        "aic_mad": profile.aic_mad,
        "aic_scalar": profile.aic_scalar,
        "aic_mte1": profile.aic_mte1,
        "aic_mte2": profile.aic_mte2,
        "aic_mte3": profile.aic_mte3,
        "aic_fixpipe": profile.aic_fixpipe,
        "aiv_vec": profile.aiv_vec,
        "aiv_scalar": profile.aiv_scalar,
        "aiv_mte2": profile.aiv_mte2,
        "aiv_mte3": profile.aiv_mte3,
    }


def select_profile(profiles: dict[str, Any], substring: str):
    matches = [name for name in profiles if substring in name]
    if len(matches) != 1:
        names = "\n".join(sorted(profiles)) or "(none)"
        raise RuntimeError(
            f"expected one kernel matching {substring!r}, got {len(matches)}:\n{names}"
        )
    return matches[0], profiles[matches[0]]


def main() -> int:
    import torch
    import torch.distributed as dist
    import torch_npu

    import deep_ep
    from deep_ep.utils.envs import get_ascend_aiv_count, init_seed

    del torch_npu
    parser = build_parser()
    args = parser.parse_args()
    args.allow_multiple_reduction = 1
    args.profile_stages = 0
    args.profile_launch_skew = False
    args.rank_launch_deadline_us = 0
    args.workload_manifest = ""
    args.dump_manifest = ""

    local_rank = int(os.environ["LOCAL_RANK"])
    _configure_process(local_rank, torch)
    torch.npu.set_device(local_rank)
    device_aiv_count = get_ascend_aiv_count(local_rank)
    if args.num_sms > device_aiv_count:
        raise ValueError(
            f"--num-sms={args.num_sms} exceeds device AIV count {device_aiv_count}"
        )

    dist.init_process_group(
        backend="hccl", timeout=timedelta(minutes=10)
    )
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("npu", local_rank)
    runtime = None
    try:
        cases = {case.case_id: case for case in enumerate_ep_mode_cases()}
        if args.case not in cases:
            raise ValueError(f"unknown case: {args.case}")
        case = cases[args.case]
        manifest = _resolve_manifest(args, world_size)
        init_seed(manifest.spec.seed)
        runtime = AscendRuntime(
            torch,
            dist,
            deep_ep,
            group,
            device,
            args,
            manifest,
            num_sms=args.num_sms,
            num_qps=0,
        )
        runtime.synchronized_step(
            runtime.construct_buffer, "buffer construction"
        )
        prepared = runtime.synchronized_step(
            lambda: runtime._prepare_case(case), f"{case.case_id}: preparation"
        )

        # Official test_ep.py keeps a 1<<26-int dirty buffer and zeros it
        # between the communication launch and current_stream_wait, ensuring a
        # cold-L2 epilogue during both profiling passes.
        l2_dirty_buffer = torch.empty(
            1 << 26, dtype=torch.int, device=device
        )
        local_results = []
        for official_name, operation_id in OFFICIAL_OPERATIONS.items():
            prepare = prepared.prepare_launches.get(operation_id)
            if prepare is not None:
                runtime.synchronized_step(
                    prepare,
                    f"{case.case_id}: {operation_id}: profile preparation",
                )
            def operation_once():
                result = prepared.launches[operation_id]()
                l2_dirty_buffer.zero_()
                if (case.async_with_compute_stream
                        and len(result) > 0
                        and hasattr(result[-1], "current_stream_wait")):
                    result[-1].current_stream_wait()
                return result

            profiles = bench_msprof(
                fn=operation_once,
                num_warmups=args.warmups,
                num_tests=args.iterations,
                flush_l2=True,
                barrier_comm_profiling=True,
                barrier=runtime.buffer.barrier,
                return_all_kernels=True,
            )
            urma_name, urma_profile = select_profile(
                profiles, KERNEL_MATCHES[operation_id][0]
            )
            epilogue_name, epilogue_profile = select_profile(
                profiles, KERNEL_MATCHES[operation_id][1]
            )
            urma_bytes = int(prepared.traffic[operation_id]["scaleup"])
            local_results.append({
                "official_operation": official_name,
                "current_operation": operation_id,
                "urma_kernel": urma_name,
                "epilogue_kernel": epilogue_name,
                "urma": kernel_record(urma_profile),
                "epilogue": kernel_record(epilogue_profile),
                "urma_bytes": urma_bytes,
                "urma_gbps": urma_profile.gbps(urma_bytes),
                "work_counts": prepared.work_counts[operation_id],
                "logical_bytes": prepared.traffic[operation_id],
                "all_kernels": {
                    name: kernel_record(profile)
                    for name, profile in sorted(profiles.items())
                },
            })

        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local_results, group=group)
        if rank == 0:
            report = {
                "platform": "ascend",
                "workload": asdict(manifest.spec),
                "case_id": case.case_id,
                "device": {
                    "name": torch.npu.get_device_name(local_rank),
                    "local_rank": local_rank,
                    "num_sms": args.num_sms,
                    "num_qps": 0,
                },
                "timing_protocol": {
                    "timer": "torch_npu_profiler_ffts",
                    "profiler": "official bench_msprof semantics",
                    "warmups": args.warmups,
                    "iterations": args.iterations,
                    "flush_l2": True,
                    "device_busy_before_barrier": True,
                    "barrier": "ElasticBuffer.barrier",
                    "barrier_location": "external-before-operation",
                    "internal_barrier_prologue_toggle": False,
                    "reason": "current ElasticBuffer lacks official set_barrier_in_prologue API",
                    "cold_l2_epilogue_zero": "1<<26 int32 elements after communication launch",
                    "bandwidth_bytes": "per-rank scaleup logical bytes",
                    "timing": (
                        "outer communication kernel for raw msprof collection; "
                        "the staged transport has no single official-equivalent "
                        "URMA issue-and-drain kernel"
                    ),
                },
                "cann_note": "CANN 9.3.0; official reference used CANN 9.2.0",
                "ranks": gathered,
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"msprof benchmark wrote {output}", flush=True)
        dist.barrier(group=group)
    finally:
        if runtime is not None:
            runtime.destroy()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
