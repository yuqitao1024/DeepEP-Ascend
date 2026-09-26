"""Compare the production epilogue parallel-prefix kernel with a CPU oracle."""

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
    entry = ctypes.cast(
        library.deep_ep_ascend_launch_direct_dispatch_epilogue_parallel_prefix,
        ctypes.c_void_p)
    adapter = ctypes.CDLL(args.adapter)
    launch = adapter.probe_dispatch_parallel_prefix
    launch.restype = ctypes.c_int
    launch.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
                       + [ctypes.c_uint64] * 4 + [ctypes.c_uint32] * 2
                       + [ctypes.c_void_p] + [ctypes.c_uint64] * 3)
    passed = []
    guard64 = 0x5a5a5a5a5a5a5a5a
    guard32 = 0x5a5a5a5a
    overflow_error = 5
    status_offset = 0
    rank_counts_offset = 256
    tile_counts_offset = 1024

    def aligned(count, alignment):
        return (count + alignment - 1) // alignment * alignment

    def run(name, world=8, experts=32, tiles=129, threads=512,
            alignment=8, capacity=1 << 30, prior_status=0):
        rank = device % world
        local = experts // world
        if local == 0 or experts % world:
            raise ValueError("the production kernel requires divided experts")
        if threads % 32:
            raise ValueError("SIMT warps require 32-thread multiples")

        counts = (torch.arange(tiles * local, dtype=torch.int64) * 7 + 3) % 5
        counts = counts.reshape(tiles, local)
        total = counts.sum(dim=0)
        prefixes = counts.cumsum(dim=0) - counts
        source_counts = (torch.arange(world, dtype=torch.int64) + 1) * 17
        rank_prefixes = source_counts.cumsum(dim=0)

        tile_words = tile_counts_offset // 8 + max(tiles * local, 1) + 16
        workspace = torch.full((tile_words,), guard64, dtype=torch.int64)
        workspace[0] = prior_status
        workspace[rank_counts_offset // 8:
                  rank_counts_offset // 8 + world] = source_counts
        workspace[tile_counts_offset // 8:
                  tile_counts_offset // 8 + tiles * local] = counts.flatten()
        expected_workspace = workspace.clone()
        if not prior_status:
            expected_workspace[tile_counts_offset // 8:
                               tile_counts_offset // 8 + tiles * local] = \
                prefixes.flatten()

        rank_output = torch.full((world + 16,), guard32, dtype=torch.int32)
        expert_output = torch.full(
            (experts + 17,), guard32, dtype=torch.int32)
        unaligned_output = torch.full(
            (experts + 16,), guard32, dtype=torch.int32)
        expected_rank = rank_output.clone()
        expected_expert = expert_output.clone()
        expected_unaligned = unaligned_output.clone()
        if not prior_status:
            expected_rank[:world] = rank_prefixes.to(torch.int32)
            expected_unaligned[rank * local:(rank + 1) * local] = \
                total.to(torch.int32)
            overflowed = False
            derived = 0
            for expert in range(experts):
                actual = total[expert - rank * local].item() if \
                    rank * local <= expert < (rank + 1) * local else 0
                if overflowed:
                    break
                expected_expert[expert] = derived
                if not (rank * local <= expert < (rank + 1) * local):
                    expected_unaligned[expert] = 0
                derived += aligned(actual, alignment)
                if derived > capacity:
                    overflowed = True
                    expected_expert[experts] = derived
                    break
            if overflowed:
                expected_workspace[0] = \
                    ((rank + 1) << 32) | overflow_error
            else:
                expected_expert[experts] = derived

        workspace = workspace.npu()
        rank_output = rank_output.npu()
        expert_output = expert_output.npu()
        unaligned_output = unaligned_output.npu()
        torch.npu.synchronize()
        result = launch(
            entry, workspace.data_ptr(), rank_output.data_ptr(),
            expert_output.data_ptr(), unaligned_output.data_ptr(),
            rank, world, experts, alignment, capacity, tiles, 1, threads,
            torch.npu.current_stream().npu_stream,
            status_offset, rank_counts_offset, tile_counts_offset)
        assert result == 0, (name, result)
        torch.npu.synchronize()
        observed_workspace = workspace.cpu()
        observed_rank = rank_output.cpu()
        observed_expert = expert_output.cpu()
        observed_unaligned = unaligned_output.cpu()

        assert torch.equal(observed_workspace, expected_workspace), (
            name, "workspace",
            torch.nonzero(observed_workspace != expected_workspace)
            .flatten().tolist()[:16])
        assert torch.equal(observed_rank, expected_rank), (
            name, "rank prefix")
        assert torch.equal(observed_unaligned, expected_unaligned), (
            name, "unaligned")
        overflowed = not prior_status and \
            expected_workspace[0].item() != 0
        if prior_status or overflowed:
            expected_status = \
                prior_status if prior_status else \
                (((rank + 1) << 32) | overflow_error)
            assert observed_workspace[0].item() == expected_status, (
                name, "overflow output must stop")
        else:
            assert torch.equal(observed_expert, expected_expert), (
                name, "expert prefix",
                torch.nonzero(observed_expert != expected_expert)
                .flatten().tolist()[:16])
        passed.append(name)

    for world in (1, 3, 8, 32):
        for threads in (32, 128, 512):
            for tiles in (0, 1, 31, 32, 33, 128, 129, 6250):
                run(f"w{world}-t{threads}-n{tiles}", world=world,
                    experts=world * 4, threads=threads, tiles=tiles)
    for alignment in (1, 8, 128):
        run(f"alignment-{alignment}", alignment=alignment, world=8,
            experts=32, tiles=129)
    for experts in (8, 32, 128, 256):
        run(f"experts-{experts}", experts=experts, world=8, tiles=129)
    run("capacity-overflow", world=8, experts=32, tiles=129,
        alignment=128, capacity=1023)
    run("status-preserved", prior_status=(3 << 32) | 6)
    Path(f"{args.output}.rank{device}.json").write_text(json.dumps({
        "device": device,
        "passed": passed,
        "kernel": "production dispatch epilogue parallel prefix",
    }, indent=2) + "\n")
    print(
        f"DISPATCH_PARALLEL_PREFIX_PASSED rank={device} "
        f"cases={len(passed)}", flush=True)


if __name__ == "__main__":
    main()
