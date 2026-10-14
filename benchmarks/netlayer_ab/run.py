#!/usr/bin/env python3
"""Run the netlayer AB benchmark with only Python's standard library."""

import argparse
import pathlib
import random
import subprocess
import sys
import tempfile
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=None)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--payload-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout-cycles", type=int, default=100_000_000)
    parser.add_argument("--drain-mode", choices=("per-peer", "post-all"), default="per-peer")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def parse_rank_result(path: pathlib.Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    required = (
        "wall_mean_ms", "wall_min_ms", "wall_max_ms", "wall_send_gibps",
        "peer_count",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{path}: missing result fields: {', '.join(missing)}")
    return values


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main():
    args = parse_args()
    drain_mode = {"per-peer": 0, "post-all": 1}[args.drain_mode]
    script_dir = pathlib.Path(__file__).resolve().parent
    binary = pathlib.Path(args.binary or script_dir / "build" / "netlayer_ab")
    if not binary.is_file():
        raise SystemExit(f"benchmark binary not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="netlayer-ab-") as directory:
        root_info = pathlib.Path(directory) / "root.bin"
        bootstrap = subprocess.run(
            [str(binary), "--make-root-info", str(root_info)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout,
        )
        if bootstrap.returncode != 0:
            # Some CANN builds reject HcclGetRootInfo without a live device
            # context while the batch scheduler holds all devices.  A random
            # token is valid input to HcclCommInitRootInfo; it is only a
            # bootstrap rendezvous token and does not select a channel or rank.
            root_info.unlink(missing_ok=True)
            root_info.write_bytes(random.SystemRandom().randbytes(4108))
        if not root_info.is_file() or root_info.stat().st_size != 4108:
            print(bootstrap.stdout, end="")
            print(bootstrap.stderr, end="", file=sys.stderr)
            raise SystemExit("failed to generate HCCL root info")

        result_files = [
            pathlib.Path(directory) / f"rank-{rank}.txt"
            for rank in range(args.world_size)
        ]
        processes = []
        for rank in range(args.world_size):
            command = [
                str(binary),
                "--rank", str(rank),
                "--world-size", str(args.world_size),
                "--layer-index", str(args.layer_index),
                "--payload-bytes", str(args.payload_bytes),
                "--chunk-bytes", str(args.chunk_bytes),
                "--iterations", str(args.iterations),
                "--warmup", str(args.warmup),
                "--timeout-cycles", str(args.timeout_cycles),
                "--drain-mode", str(drain_mode),
                "--root-info", str(root_info),
                *(["--publish-root-info"] if rank == 0 else []),
                "--result", str(result_files[rank]),
            ]
            processes.append(subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ))

        failed = False
        timed_out = False
        for rank, process in enumerate(processes):
            try:
                stdout, stderr = process.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                print(f"rank {rank} timed out; terminating all ranks", file=sys.stderr)
                for candidate in processes:
                    if candidate.poll() is None:
                        candidate.terminate()
                stdout, stderr = process.communicate()
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)
            if process.returncode != 0:
                print(f"rank {rank} exited with {process.returncode}", file=sys.stderr)
                failed = True
        if failed:
            raise SystemExit(1)
        if timed_out:
            raise SystemExit(124)

        ranks = [parse_rank_result(path) for path in result_files]
        wall_means = [float(rank["wall_mean_ms"]) for rank in ranks]
        peer_counts = [int(rank["peer_count"]) for rank in ranks]

        slowest_wall = max(wall_means)
        aggregate_wall_gibps = sum(float(rank["wall_send_gibps"]) for rank in ranks)
        slowest_rank_wall_gibps = (
            min(peer_counts) * args.payload_bytes
            / (slowest_wall / 1000.0) / (1024 ** 3)
        )

        line = (
            f"layer={args.layer_index} world_size={args.world_size} "
            f"payload_bytes={args.payload_bytes} chunk_bytes={args.chunk_bytes} "
            f"drain_mode={args.drain_mode} "
            f"iterations={args.iterations} warmup={args.warmup} "
            f"wall_slowest_mean_ms={slowest_wall:.6f} "
            f"wall_mean_rank_ms={statistics.mean(wall_means):.6f} "
            f"wall_send_sum_gibps={aggregate_wall_gibps:.6f} "
            f"wall_slowest_rank_gibps={slowest_rank_wall_gibps:.6f} "
            f"peer_counts={','.join(map(str, peer_counts))} "
            f"min_peer_count={min(peer_counts)}"
        )
        print(line)
        if args.output:
            pathlib.Path(args.output).write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
