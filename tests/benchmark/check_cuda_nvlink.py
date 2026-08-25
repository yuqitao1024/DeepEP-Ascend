import argparse
from dataclasses import dataclass
import re
import subprocess
import sys


@dataclass(frozen=True)
class TopologyCheck:
    gpu_labels: tuple[str, ...]
    error: str | None


def validate_nvlink_clique(output, expected_gpus):
    tokenized_lines = [line.split() for line in output.splitlines()]
    header_labels = max(
        (
            tuple(token for token in tokens if re.fullmatch(r"GPU\d+", token))
            for tokens in tokenized_lines
        ),
        key=len,
        default=(),
    )
    if len(header_labels) != expected_gpus:
        return TopologyCheck(
            header_labels,
            f"expected {expected_gpus} GPUs in topology matrix, "
            f"found {len(header_labels)}",
        )

    rows = {
        tokens[0]: tuple(tokens[1:1 + len(header_labels)])
        for tokens in tokenized_lines
        if tokens and tokens[0] in header_labels
    }
    for source_index, source in enumerate(header_labels):
        connections = rows.get(source)
        if connections is None or len(connections) != len(header_labels):
            return TopologyCheck(
                header_labels,
                f"topology matrix row is missing or incomplete: {source}",
            )
        for target_index, target in enumerate(header_labels):
            if source_index == target_index:
                continue
            connection = connections[target_index]
            if connection != "NV8":
                return TopologyCheck(
                    header_labels,
                    f"{source} -> {target} uses {connection}, expected NV8",
                )

    return TopologyCheck(header_labels, None)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify a single-node CUDA NVLink clique",
    )
    parser.add_argument("--expected-gpus", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        completed = subprocess.run(
            ("nvidia-smi", "topo", "-m"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"ERROR: cannot query CUDA topology: {error}", file=sys.stderr)
        return 1

    print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    result = validate_nvlink_clique(completed.stdout, args.expected_gpus)
    if result.error is not None:
        print(f"ERROR: {result.error}", file=sys.stderr)
        return 1

    print(
        "CUDA_TOPOLOGY=SINGLE_NODE_NVLINK_CLIQUE "
        f"GPUS={len(result.gpu_labels)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
