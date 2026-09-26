"""Compare the production epilogue expert-count kernel with a CPU oracle."""

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
        library.deep_ep_ascend_launch_direct_dispatch_epilogue_count_experts,
        ctypes.c_void_p)
    adapter = ctypes.CDLL(args.adapter)
    launch = adapter.probe_dispatch_count
    launch.restype = ctypes.c_int
    launch.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
                       + [ctypes.c_uint64] * 4 + [ctypes.c_uint32] * 2
                       + [ctypes.c_void_p])
    passed = []

    def run(name, capacity=137, topk=8, world=8, experts=32,
            threads=512, blocks=1, partial=True):
        rank = device % world
        local = experts // world
        if local == 0:
            raise ValueError("the production kernel requires local experts")

        # source_counts is indexed by source rank, not by receive slot.
        if partial:
            source_counts = torch.minimum(
                torch.arange(world).remainder(5 * world + 3),
                torch.arange(world).remainder(5) + 1)
        else:
            source_counts = torch.full((world,), capacity)

        records = torch.zeros((world * capacity, topk), dtype=torch.int64)
        if records.numel():
            records[:] = torch.arange(world * capacity).remainder(
                experts).unsqueeze(1)
        tile_count = (world * capacity + 127) // 128
        expected = torch.zeros((tile_count, local), dtype=torch.int64)
        for source in range(world):
            for slot in range(int(source_counts[source].item())):
                if slot >= capacity:
                    break
                record = source * capacity + slot
                for expert in records[record].tolist():
                    if rank * local <= expert < (rank + 1) * local:
                        expected[record // 128, expert - rank * local] += 1

        output_offset = 1024  # 8192 bytes / sizeof(int64)
        workspace = torch.full(
            (output_offset + expected.numel() + 16,),
            0x5a5a5a5a5a5a5a5a,
            dtype=torch.int64)
        workspace[0] = 0
        workspace[32:32 + world] = source_counts
        records, workspace = records.npu(), workspace.npu()
        torch.npu.synchronize()
        result = launch(entry, records.data_ptr(), workspace.data_ptr(), rank, world,
                        experts, topk, capacity, topk * 8, blocks, threads,
                        torch.npu.current_stream().npu_stream)
        assert result == 0, (name, result)
        torch.npu.synchronize()
        observed = workspace[output_offset:output_offset + expected.numel()].cpu()
        assert torch.equal(observed, expected.flatten()), (
            name, "counts", "observed", observed[:32].tolist(),
            "expected", expected.flatten()[:32].tolist(),
            torch.nonzero(observed != expected.flatten()).flatten().tolist()[:16])
        assert workspace[output_offset + expected.numel():].cpu().equal(
            torch.full((16,), 0x5a5a5a5a5a5a5a5a, dtype=torch.int64)), (
            name, "output overrun")
        passed.append(name)

    for world in (1, 3, 8, 32):
        for threads in (32, 128, 512):
            for capacity in (0, 1, 17, 128, 129, 8193):
                run(f"w{world}-t{threads}-c{capacity}", world=world,
                    threads=threads, capacity=capacity)
    for topk in (1, 2, 8, 32):
        run(f"topk-{topk}", topk=topk, capacity=517)
    for experts in (8, 32, 128, 256):
        run(f"experts-{experts}", experts=experts, capacity=517)
    run("full-counts", partial=False, capacity=257)
    run("multi-block", blocks=2, capacity=1025)
    Path(f"{args.output}.rank{device}.json").write_text(json.dumps({
        "device": device, "passed": passed,
        "kernel": "production dispatch expert count",
    }, indent=2) + "\n")
    print(f"DISPATCH_COUNT_PASSED rank={device} cases={len(passed)}", flush=True)


if __name__ == "__main__":
    main()
