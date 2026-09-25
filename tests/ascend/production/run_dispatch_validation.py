"""Exercise the production validation kernel, including deterministic errors.

Build core_ops/dispatch_validate_adapter.cpp with a host C++17 compiler as a
shared library, then pass --adapter to this runner under the selected CANN
environment. No transport operations are submitted: received peer records are
synthetic local device memory, while the launcher/kernel are from deep_ep._C.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True, help="Output prefix; one JSON per device")
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401
    import deep_ep._C as extension

    device_id = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(device_id)
    library = ctypes.CDLL(extension.__file__)
    entry = ctypes.cast(library.deep_ep_ascend_launch_direct_dispatch_epilogue_validate_records,
                        ctypes.c_void_p)
    adapter = ctypes.CDLL(args.adapter)
    launch = adapter.probe_dispatch_validate
    launch.restype = ctypes.c_int
    launch.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                       + [ctypes.c_uint64] * 3 + [ctypes.c_uint32] * 3
                       + [ctypes.c_void_p])
    passed = []
    cached, expanded = 1 << 0, 1 << 1

    def run(name, capacity=137, topk=8, world=8, blocks=2, threads=512,
            mode=0, faults=(), empty=False, partial=False, prior_status=0):
        rank = device_id % world
        stride = ((topk * 8 + 8 + 31) // 32) * 32
        count = world * capacity
        counts = [0 if empty else max(0, capacity - source % 3) if partial else capacity
                  for source in range(world)]
        bases, total = [], 0
        for value in counts:
            bases.append(total)
            total += value
        records = torch.zeros((count, stride), dtype=torch.uint8)
        topk_view = records.view(torch.int64)[:, :topk]
        record_meta = records.view(torch.int32)[:, topk * 2:topk * 2 + 2]
        topk_view.fill_(-1)
        topk_view[:, 0] = rank * 4
        if count:
            record_meta[:, 0] = torch.arange(count) % capacity
        metadata = torch.full((total, 2 + topk), -1, dtype=torch.int32)
        for source in range(world):
            begin, size = bases[source], counts[source]
            metadata[begin:begin + size, 0] = source * capacity + torch.arange(size)
            metadata[begin:begin + size, 1] = source * topk
            metadata[begin:begin + size, 2] = 0
        for record, kind in faults:
            source, slot = divmod(record, capacity)
            if kind == "topk":
                topk_view[record, -1] = world * 4
            elif kind == "master":
                record_meta[record, 1] = -1
            elif kind == "identity":
                metadata[bases[source] + slot, 0] = -1
            elif kind == "nonlocal-slot":
                metadata[bases[source] + slot, -1] = 0
            else:
                raise ValueError(kind)

        # Independent serial oracle: preserve the first failing record in each
        # tile, then the first failing tile, including the packed diagnostic.
        tiles = (count + 127) // 128
        expected = [0] * tiles
        for record in range(count):
            tile = record // 128
            source, slot = divmod(record, capacity)
            if slot >= counts[source] or expected[tile]:
                continue
            routes = topk_view[record].tolist()
            origin, master = record_meta[record].tolist()
            local = [rank * 4 <= expert < (rank + 1) * 4 for expert in routes]
            error, peer = 0, rank
            for lane, expert in enumerate(routes):
                if expert < -1 or expert >= world * 4:
                    error, peer = 1, 600000 + record
                    break
                if mode & cached and mode & expanded and not local[lane]:
                    if metadata[bases[source] + slot, 2 + lane].item() != -1:
                        error = 4
                        break
            if not error and (not any(local) or not 0 <= origin < capacity
                              or not 0 <= master < topk or not local[master]):
                error = 4
                reason = (100000 if not any(local) else 200000 if not 0 <= origin < capacity
                          else 300000 if not 0 <= master < topk else 400000)
                peer = (reason + record * 65536 + ((routes[0] & 255) << 8)
                        + (routes[-1] & 255)) & 0xffffffff
            if not error and mode & cached:
                reusable = metadata[bases[source] + slot]
                if (reusable[0].item() != source * capacity + origin
                        or reusable[1].item() != source * topk + master):
                    error = 4
            if error:
                expected[tile] = (((peer + 1) & 0xffffffff) << 32) | error
        first = next((tile for tile, value in enumerate(expected) if value), None)
        candidate = -1 if first is None else ((first + 1) << 32) | (expected[first] & 0xffffffff)
        status = prior_status or (0 if first is None else expected[first])
        workspace = torch.full((16384,), 0x5a, dtype=torch.uint8)
        words = workspace.view(torch.int64)
        words[0] = prior_status
        words[4:4 + world] = torch.tensor(counts, dtype=torch.int64)
        words[16:16 + world] = torch.tensor(bases, dtype=torch.int64)
        words[1024] = -1
        workspace.view(torch.int32)[2050] = 0
        records, metadata, workspace = (tensor.npu() for tensor in (records, metadata, workspace))
        torch.npu.synchronize()
        result = launch(entry, records.data_ptr(), workspace.data_ptr(), metadata.data_ptr(),
                        rank, world, capacity, topk, stride, mode, blocks, threads,
                        torch.npu.current_stream().npu_stream)
        assert result == 0, (name, result)
        torch.npu.synchronize()
        observed = workspace.cpu()
        words = observed.view(torch.int64)
        assert words[0].item() == status, (name, "status", words[0].item(), status)
        assert words[32:32 + tiles].tolist() == expected, (name, "tile errors")
        assert words[1024].item() == candidate, (name, "candidate")
        assert observed.view(torch.int32)[2050].item() == blocks * threads, (name, "completion")
        assert words[32 + tiles].item() == 0x5a5a5a5a5a5a5a5a, (name, "tile overrun")
        passed.append(name)

    for world in (1, 8):
        for capacity in (0, 1, 17, 129, 137):
            run(f"clean-w{world}-c{capacity}", world=world, capacity=capacity, partial=True)
    for threads in (32, 128, 512):
        run(f"multi-tile-{threads}", capacity=8193, threads=threads)
        run(f"earliest-record-{threads}", threads=threads,
            faults=((3, "master"), (4, "topk"), (128, "topk")))
        run(f"earliest-tile-{threads}", threads=threads,
            faults=((127, "topk"), (128, "master"), (1024, "topk")))
        run(f"same-thread-{threads}", threads=threads,
            faults=((1, "topk"), (2, "master")))
    run("last-record", faults=((1095, "topk"),))
    run("empty-route", empty=True, faults=((0, "topk"),))
    run("preexisting-error", prior_status=(1 << 32) | 6, faults=((4, "topk"),))
    for mode in (cached, cached | expanded):
        run(f"cached-clean-{mode}", mode=mode, partial=True)
        run(f"cached-identity-{mode}", mode=mode, faults=((3, "identity"), (4, "topk")))
    run("cached-nonlocal-slot", mode=cached | expanded,
        faults=((3, "nonlocal-slot"), (4, "topk")))
    for topk in (1, 2, 32):
        run(f"topk-{topk}", topk=topk)
    Path(f"{args.output}.rank{device_id}.json").write_text(json.dumps({
        "device": device_id, "passed": passed, "kernel": "production receive validation",
    }, indent=2) + "\n")
    print(f"DISPATCH_VALIDATION_PASSED rank={device_id} cases={len(passed)}", flush=True)


if __name__ == "__main__":
    main()
