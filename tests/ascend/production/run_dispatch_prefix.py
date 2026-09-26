"""Compare the production producer-prefix kernel with a serial CPU oracle.

Build core_ops/dispatch_prefix_adapter.cpp as a host C++17 shared library.
Run this script through the device queue; it uses synthetic local grouping
outputs, without transport, and supports one process per device via torchrun.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    import deep_ep._C as extension

    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(device)
    library = ctypes.CDLL(extension.__file__)
    entry = ctypes.cast(library.deep_ep_ascend_launch_dispatch_producer_prefix,
                        ctypes.c_void_p)
    adapter = ctypes.CDLL(args.adapter)
    launch = adapter.probe_dispatch_prefix
    launch.restype = ctypes.c_int
    launch.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
                       + [ctypes.c_uint64] * 2 + [ctypes.c_uint32] * 2
                       + [ctypes.c_void_p])
    passed = []
    guard = 0x5a5a5a5a5a5a5a5a

    def run(name, tiles=2048, world=8, threads=512, faults=(),
            capacity=100000, early=False, prior_status=0, prior_error=False):
        rank = device % world
        counts = ((torch.arange(tiles * world) * 7 + 3) % 5).reshape(tiles, world)
        totals = counts.sum(dim=0)
        prefixes = counts.cumsum(dim=0) - counts
        error_offset = 512 + tiles * world + 32
        workspace = torch.full((error_offset + max(tiles, threads) + 32,),
                               guard, dtype=torch.int64)
        workspace[0] = prior_status
        workspace[32:32 + world] = -1
        workspace[64:64 + world] = -1
        workspace[96:96 + world] = 0
        if prior_error:
            workspace[96:96 + world] = (123 << 32) | 6
        workspace.view(torch.int32)[256:256 + world * 4] = torch.arange(world * 4)
        workspace[512:512 + tiles * world] = counts.flatten()
        workspace[error_offset:error_offset + tiles] = 0
        for tile, error in faults:
            workspace[error_offset + tile] = error
        expected = workspace.clone()
        staging = torch.full((32 + world * 16 + 16,), guard, dtype=torch.int64)
        expected_staging = staging.clone()
        if not prior_status:
            expected[512:512 + tiles * world] = prefixes.flatten()
            expected[32:32 + world] = totals
            expected[64:64 + world] = 0
            first = next((workspace[error_offset + tile].item() for tile in range(tiles)
                          if workspace[error_offset + tile].item()), 0)
            if first:
                expected[96] = first
            for dest in range(world):
                if totals[dest] > capacity:
                    expected[96 + dest] = ((rank + 1) << 32) | 5
                if early:
                    begin = 32 + dest * 16
                    expected_staging[begin:begin + 6] = torch.tensor(
                        [7, totals[dest].item(), *range(dest * 4, dest * 4 + 4)])
        workspace, staging = workspace.npu(), staging.npu()
        torch.npu.synchronize()
        result = launch(entry, workspace.data_ptr(), staging.data_ptr(), rank, world,
                        tiles, capacity, threads, early, torch.npu.current_stream().npu_stream)
        assert result == 0, (name, result)
        torch.npu.synchronize()
        observed = workspace.cpu()
        # The front of tile_errors becomes prefix scratch only on the parallel
        # path. All other output words, padding and guards must match exactly.
        if not prior_status and threads % world == 0 and tiles >= threads:
            expected[error_offset:error_offset + threads] = observed[
                error_offset:error_offset + threads]
        assert torch.equal(observed, expected), (
            name, "workspace", torch.nonzero(observed != expected).flatten().tolist()[:16])
        assert torch.equal(staging.cpu(), expected_staging), (name, "route staging")
        passed.append(name)

    for world in (1, 3, 8, 32):
        for threads in (32, 128, 512):
            for tiles in (0, 1, threads - 1, threads, 2049):
                run(f"clean-w{world}-t{threads}-n{tiles}", world=world,
                    threads=threads, tiles=tiles)
    for threads in (32, 128, 512):
        for tile in (0, 1, 31, 32, threads - 1, threads, 2047):
            run(f"error-t{threads}-i{tile}", threads=threads,
                faults=((tile, (19 << 32) | 7), (2047, (2 << 32) | 1)))
        run(f"same-lane-{threads}", threads=threads,
            faults=((1, (12 << 32) | 7), (1 + threads, (2 << 32) | 1)))
        run(f"overflow-{threads}", threads=threads, capacity=4095,
            faults=((1, (19 << 32) | 7),), prior_error=True)
    run("status-preserved", prior_status=(3 << 32) | 6)
    run("error-preserved", prior_error=True)
    run("all-errors", faults=tuple((i, ((i + 1) << 32) | 7) for i in range(2048)))
    run("fallback-error", tiles=17, faults=((9, (19 << 32) | 7),))
    run("uneven-ranks-error", world=3, faults=((33, (19 << 32) | 7),))
    run("early-route", early=True)
    run("early-route-fallback", tiles=17, early=True)
    Path(f"{args.output}.rank{device}.json").write_text(json.dumps({
        "device": device, "passed": passed, "kernel": "production producer prefix",
    }, indent=2) + "\n")
    print(f"DISPATCH_PREFIX_PASSED rank={device} cases={len(passed)}", flush=True)


if __name__ == "__main__":
    main()
