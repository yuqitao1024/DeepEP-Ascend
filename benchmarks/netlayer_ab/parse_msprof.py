#!/usr/bin/env python3
"""Summarize netlayer_ab kernel durations from msprof output."""

import argparse
import csv
import glob
import math
import pathlib
import statistics

KERNEL_MARKER = "netlayer_ab_kernel"


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def collect_rank_durations(profile_root):
    ranks = {}
    pattern = profile_root / "**" / "op_summary_*.csv"
    for path in sorted(glob.glob(str(pattern), recursive=True)):
        durations = []
        device_id = None
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if KERNEL_MARKER in row["Op Name"]:
                    duration = float(row["Task Duration(us)"].strip())
                    if math.isfinite(duration) and duration > 0:
                        durations.append(duration / 1000.0)
                        device_id = row["Device_id"]
        if durations:
            if device_id in ranks:
                raise ValueError(f"duplicate profile for device {device_id}")
            ranks[device_id] = durations
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_root", type=pathlib.Path)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--peer-count", type=int, default=7)
    parser.add_argument("--drain-mode", choices=("per-peer", "post-all"), default="per-peer")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    ranks = collect_rank_durations(args.profile_root)
    if not ranks:
        raise SystemExit(f"no {KERNEL_MARKER} records under {args.profile_root}")
    means = {rank: statistics.mean(values) for rank, values in ranks.items()}
    bandwidths = {
        rank: args.peer_count * args.payload_bytes / (mean_ms / 1000.0) / 1e9
        for rank, mean_ms in means.items()
    }
    duration_values = list(means.values())
    bandwidth_values = list(bandwidths.values())
    counts = ",".join(f"{rank}:{len(values)}" for rank, values in sorted(ranks.items()))
    line = (
        f"profile_root={args.profile_root} "
        f"drain_mode={args.drain_mode} "
        f"payload_bytes={args.payload_bytes} peer_count={args.peer_count} "
        f"rank_count={len(ranks)} "
        f"kernel_invocation_counts={counts} "
        f"kernel_slowest_rank_ms={max(duration_values):.6f} "
        f"kernel_mean_rank_ms={statistics.mean(duration_values):.6f} "
        f"kernel_gbps_mean={statistics.mean(bandwidth_values):.6f} "
        f"kernel_gbps_min={min(bandwidth_values):.6f} "
        f"kernel_gbps_max={max(bandwidth_values):.6f} "
        f"kernel_gbps_p50={percentile(bandwidth_values, 0.50):.6f} "
        f"kernel_gbps_p95={percentile(bandwidth_values, 0.95):.6f}"
    )
    print(line)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
