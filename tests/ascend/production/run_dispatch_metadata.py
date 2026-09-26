"""Compare the production epilogue metadata kernel with a CPU oracle."""

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
        library.deep_ep_ascend_launch_direct_dispatch_epilogue_metadata,
        ctypes.c_void_p)
    adapter = ctypes.CDLL(args.adapter)
    launch = adapter.probe_dispatch_metadata
    launch.restype = ctypes.c_int
    launch.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
                       + [ctypes.c_uint32]
                       + [ctypes.c_uint64] * 6 + [ctypes.c_uint32] * 2
                       + [ctypes.c_void_p] + [ctypes.c_uint64] * 3)
    passed = []
    guard = 0x5a
    status_offset = 0
    rank_counts_offset = 256
    rank_values_offset = 512
    topk_offset = 0
    metadata_offset = 64

    def run(name, world=8, experts=32, topk=8, capacity=137,
            threads=512, blocks=1, cached=False, prior_status=0):
        rank = device % world
        local = experts // world
        if local == 0 or experts % world:
            raise ValueError("the production kernel requires divided experts")

        source_counts = torch.minimum(
            torch.arange(world).remainder(5 * world + 3),
            torch.arange(world).remainder(5) + 1)
        source_counts = torch.minimum(
            source_counts, torch.full_like(source_counts, capacity))
        source_bases = torch.zeros(world, dtype=torch.int64)
        derived = 0
        for source in range(world):
            source_bases[source] = derived
            derived += int(source_counts[source].item())

        stride = topk * 8 + 64
        records = torch.full((max(world * capacity * stride, 8),), guard,
                             dtype=torch.uint8)
        records_view = records.view(torch.int64)
        records_int32 = records.view(torch.int32)
        for source in range(world):
            for slot in range(int(source_counts[source].item())):
                begin = (source * capacity + slot) * stride
                topk_begin = (begin + topk_offset) // 8
                records_view[topk_begin:topk_begin + topk] = torch.arange(
                    topk).remainder(experts)
                metadata_begin = (begin + metadata_offset) // 4
                records_int32[
                    metadata_begin:metadata_begin + 2] = torch.tensor(
                        [slot, rank], dtype=torch.int32)

        metadata_stride = 2 + topk
        output_count = max(int(source_counts.sum().item()), 1)
        metadata = torch.full(
            (output_count * metadata_stride + 16,), 0x5a5a5a5a,
            dtype=torch.int32)
        recv_topk = torch.full(
            (output_count * topk + 16,), -12345, dtype=torch.int64)
        expected_metadata = metadata.clone()
        expected_recv_topk = recv_topk.clone()
        if not prior_status:
            for source in range(world):
                for slot in range(int(source_counts[source].item())):
                    compact = int(source_bases[source].item()) + slot
                    output = expected_metadata[
                        compact * metadata_stride:
                        (compact + 1) * metadata_stride]
                    if not cached:
                        output[0] = source * capacity + slot
                        output[1] = source * topk + rank
                        output[2:] = -1
                    begin = (source * capacity + slot) * stride
                    topk_begin = (begin + topk_offset) // 8
                    for lane in range(topk):
                        expert = int(records_view[topk_begin + lane].item())
                        local_expert = (
                            expert - rank * local
                            if rank * local <= expert < (rank + 1) * local
                            else -1)
                        expected_recv_topk[compact * topk + lane] = local_expert

        workspace = torch.full((1024 + world * 4,), guard, dtype=torch.int64)
        workspace[0] = prior_status
        workspace[rank_counts_offset // 8:
                  rank_counts_offset // 8 + world] = source_counts
        workspace[rank_values_offset // 8:
                  rank_values_offset // 8 + world] = source_bases
        expected_workspace = workspace.clone()

        records, workspace = records.npu(), workspace.npu()
        metadata, recv_topk = metadata.npu(), recv_topk.npu()
        torch.npu.synchronize()
        result = launch(
            entry, records.data_ptr(), workspace.data_ptr(),
            recv_topk.data_ptr(), metadata.data_ptr(), rank, world,
            1 if cached else 0, experts, topk, capacity, stride,
            topk_offset, metadata_offset, blocks, threads,
            torch.npu.current_stream().npu_stream,
            status_offset, rank_counts_offset, rank_values_offset)
        assert result == 0, (name, result)
        torch.npu.synchronize()
        assert torch.equal(workspace.cpu(), expected_workspace), (
            name, "workspace")
        assert torch.equal(recv_topk.cpu(), expected_recv_topk), (
            name, "recv topk")
        assert torch.equal(metadata.cpu(), expected_metadata), (
            name, "metadata")
        passed.append(name)

    for world in (1, 3, 8, 32):
        for threads in (32, 128, 512):
            for capacity in (0, 1, 17, 128, 129, 8193):
                run(f"w{world}-t{threads}-c{capacity}", world=world,
                    experts=world * 4, threads=threads, capacity=capacity)
    for topk in (1, 2, 8, 32):
        run(f"topk-{topk}", topk=topk, capacity=517)
    for experts in (8, 32, 128, 256):
        run(f"experts-{experts}", experts=experts, capacity=517)
    run("multi-block", blocks=2, capacity=1025)
    run("cached", cached=True, capacity=517)
    run("status-preserved", prior_status=(3 << 32) | 6)
    Path(f"{args.output}.rank{device}.json").write_text(json.dumps({
        "device": device,
        "passed": passed,
        "kernel": "production dispatch epilogue metadata",
    }, indent=2) + "\n")
    print(
        f"DISPATCH_METADATA_PASSED rank={device} cases={len(passed)}",
        flush=True)


if __name__ == "__main__":
    main()
