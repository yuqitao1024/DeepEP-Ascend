#!/usr/bin/env python3
"""Compare two netlayer benchmark result lines using only the stdlib."""

import argparse
import pathlib


def parse_result(path: pathlib.Path) -> dict:
    fields = {}
    for item in path.read_text(encoding="utf-8").split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    required = ("payload_bytes", "chunk_bytes", "world_size",
                "slowest_mean_ms", "mean_rank_ms", "send_sum_gibps",
                "slowest_rank_gibps")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"{path}: missing fields {', '.join(missing)}")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer0", required=True)
    parser.add_argument("--layer1", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    layer0 = parse_result(pathlib.Path(args.layer0))
    layer1 = parse_result(pathlib.Path(args.layer1))
    if (layer0["payload_bytes"], layer0["chunk_bytes"], layer0["world_size"]) != (
       layer1["payload_bytes"], layer1["chunk_bytes"], layer1["world_size"]):
        raise SystemExit("payload/chunk/world_size must match")

    lines = [
        f"world_size={layer0['world_size']} payload_bytes={layer0['payload_bytes']} "
        f"chunk_bytes={layer0['chunk_bytes']}",
        f"layer0: slowest_mean_ms={float(layer0['slowest_mean_ms']):.6f} "
        f"mean_rank_ms={float(layer0['mean_rank_ms']):.6f} "
        f"send_sum_gibps={float(layer0['send_sum_gibps']):.6f} "
        f"slowest_rank_gibps={float(layer0['slowest_rank_gibps']):.6f}",
        f"layer1: slowest_mean_ms={float(layer1['slowest_mean_ms']):.6f} "
        f"mean_rank_ms={float(layer1['mean_rank_ms']):.6f} "
        f"send_sum_gibps={float(layer1['send_sum_gibps']):.6f} "
        f"slowest_rank_gibps={float(layer1['slowest_rank_gibps']):.6f}",
    ]
    for metric in ("slowest_mean_ms", "mean_rank_ms"):
        speedup = float(layer0[metric]) / float(layer1[metric])
        lines.append(f"{metric}: layer1_vs_layer0_speedup={speedup:.6f}x")
    for metric in ("send_sum_gibps", "slowest_rank_gibps"):
        ratio = float(layer1[metric]) / float(layer0[metric])
        lines.append(f"{metric}: layer1_vs_layer0_ratio={ratio:.6f}x")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        pathlib.Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
