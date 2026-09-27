"""Stress ordinary-to-cached receive-window reuse with distinguishable weights.

Run through torch.distributed.run with the normal bench_ep shape arguments.
All five operation oracles remain enabled; no diagnostic barrier is inserted.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ascend.benchmark import bench_ep, runtime


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reuse-repetitions", type=int, default=20)
    parser.add_argument("--reuse-mixed-streams", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.reuse_repetitions < 1:
        parser.error("--reuse-repetitions must be positive")
    if "--skip-check" in remaining:
        parser.error("receive-window reuse regression requires correctness checks")
    sys.argv = [sys.argv[0], *remaining]
    original_launch = runtime.AscendRuntime._launch
    original_prepare = runtime.AscendRuntime._prepare_case

    def launch(self, case, operation, arguments):
        if operation == "dispatch" and "handle" in arguments and not arguments.get("do_expand", False):
            arguments = dict(arguments)
            arguments["topk_weights"] = self.torch.full(
                (self.manifest.ranks[self.rank].num_tokens, self.manifest.spec.num_topk),
                0.375, dtype=self.torch.float32, device=self.device)
        return original_launch(self, case, operation, arguments)

    def prepare(self, case):
        if options.reuse_mixed_streams and self.rank % 2:
            case = replace(case, async_with_compute_stream=True,
                           allocate_on_comm_stream=True)
        for _ in range(options.reuse_repetitions):
            prepared = original_prepare(self, case)
        print(f"DISPATCH_RECEIVE_REUSE_PASSED rank={self.rank} repetitions={options.reuse_repetitions}", flush=True)
        return prepared

    runtime.AscendRuntime._launch = launch
    runtime.AscendRuntime._prepare_case = prepare
    try:
        return bench_ep.main()
    finally:
        runtime.AscendRuntime._launch = original_launch
        runtime.AscendRuntime._prepare_case = original_prepare


if __name__ == "__main__":
    raise SystemExit(main())
