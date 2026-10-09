#!/usr/bin/env python3
"""Run the netlayer AB benchmark with only Python's standard library."""

import argparse
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile


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
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
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

        means = []
        bandwidths = []
        peer_counts = []
        for rank in range(args.world_size):
            values = result_files[rank].read_text().splitlines()
            mean, _minimum, _maximum, bandwidth = map(float, values[:4])
            means.append(mean)
            bandwidths.append(bandwidth)
            peer_counts.append(int(values[4]))

        slowest = max(means)
        aggregate_gibps = sum(bandwidths)
        slowest_rank_gibps = (
            (args.world_size - 1) * args.payload_bytes
            / (slowest / 1000.0) / (1024 ** 3)
        )
        line = (
            f"layer={args.layer_index} world_size={args.world_size} "
            f"payload_bytes={args.payload_bytes} chunk_bytes={args.chunk_bytes} "
            f"slowest_mean_ms={slowest:.6f} "
            f"mean_rank_ms={statistics.mean(means):.6f} "
            f"send_sum_gibps={aggregate_gibps:.6f} "
            f"slowest_rank_gibps={slowest_rank_gibps:.6f} "
            f"peer_counts={','.join(map(str, peer_counts))} "
            f"min_peer_count={min(peer_counts)}"
        )
        print(line)
        if args.output:
            pathlib.Path(args.output).write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
