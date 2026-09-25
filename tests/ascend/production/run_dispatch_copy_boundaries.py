"""Device regression for Dispatch payload copies; top-k weights are omitted."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


# (name, tile bytes, token capacity, hidden elements, FP8, async, empty routes)
CASES = (
    ("one-tile", 8192, 1, 17, False, False, False),
    ("empty-route", 8192, 17, 4097, False, False, True),
    *((f"tail-{tile}", tile, 17, 4865, False, False, False)
      for tile in (512, 1024, 2048, 4096, 8192)),
    ("fp8-tail-async", 8192, 17, 129, True, True, False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sms", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    import deep_ep
    from tests.utils.ep_benchmark_core import build_dispatch_arguments
    from tests.utils.ep_benchmark_manifest import EPModeCase

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", local_rank)
    completed = []
    try:
        for name, tile, capacity, hidden, fp8, asynchronous, empty in CASES:
            os.environ["DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES"] = str(tile)
            mode = EPModeCase(True, 128 if fp8 else 1, fp8, 0,
                              asynchronous, asynchronous, asynchronous)
            inputs, routes = [], []
            for source in range(world):
                tokens = capacity if source % 2 == 0 else capacity - 1
                values = (torch.arange(tokens * hidden).reshape(tokens, hidden)
                          + source * 97) % 251 - 125
                payload = values.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)
                if fp8:
                    scales = torch.full((tokens, (hidden + 127) // 128),
                                        float(source + 1), dtype=torch.float32)
                    payload = (payload, scales)
                inputs.append(payload)
                routes.append(torch.tensor([
                    [-1 if empty or (token + lane) % 3 == 0 else
                     ((source + token + lane) % world) * 2 + lane
                     for lane in range(2)] for token in range(tokens)
                ], dtype=torch.int64).reshape(tokens, 2))
            cpu_x = inputs[rank]
            x = tuple(t.to(device) for t in cpu_x) if fp8 else cpu_x.to(device)
            indices = routes[rank].to(device)
            buffer = deep_ep.ElasticBuffer(
                dist.group.WORLD, num_bytes=8 * 1024 * 1024,
                allow_hybrid_mode=False, explicitly_destroy=True,
                num_gpu_timeout_secs=30, num_cpu_timeout_secs=30)
            try:
                arguments = build_dispatch_arguments(
                    mode, x, indices, None, capacity, world * 2, args.num_sms, 0)

                def validate(result, expanded, zero_padding):
                    recv, _, weights, handle, _ = result
                    assert weights is None
                    count = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
                    metadata = handle.recv_src_metadata[:count].cpu()
                    expected_ids = [s * capacity + t
                                    for s in range(world) for t, row in enumerate(routes[s])
                                    if bool(((row >= rank * 2) & (row < rank * 2 + 2)).any())]
                    assert sorted(metadata[:, 0].tolist()) == expected_ids
                    received = (tuple(t.cpu() for t in recv) if fp8 else (recv.cpu(),))
                    destinations, payloads, scales = [], [], []
                    for record, row in enumerate(metadata.tolist()):
                        source, token = divmod(row[0], capacity)
                        source_x = inputs[source]
                        lanes = range(2) if expanded else (None,)
                        for lane in lanes:
                            destination = row[2 + lane] if expanded else record
                            if expanded:
                                expert = int(routes[source][token, lane])
                                valid = rank * 2 <= expert < rank * 2 + 2
                                assert (destination >= 0) == valid
                                if not valid:
                                    continue
                            destinations.append(destination)
                            payloads.append((source_x[0] if fp8 else source_x)[token])
                            if fp8:
                                scales.append(source_x[1][token])
                    assert len(set(destinations)) == len(destinations)
                    if destinations:
                        expected = torch.stack(payloads)
                        actual = received[0]
                        if fp8:
                            actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
                        torch.testing.assert_close(actual[destinations], expected, rtol=0, atol=0)
                        if fp8:
                            torch.testing.assert_close(received[1][destinations],
                                                       torch.stack(scales), rtol=0, atol=0)
                    if zero_padding:
                        padding = torch.ones(received[0].shape[0], dtype=torch.bool)
                        padding[destinations] = False
                        for tensor in received:
                            assert bool((tensor.contiguous().view(torch.uint8)[padding] == 0).all())

                def run(launch_args, expanded=False, zero_padding=False):
                    if asynchronous:
                        launch_args = dict(launch_args, previous_event=buffer.capture())
                    result = buffer.dispatch(**launch_args)
                    if asynchronous:
                        result[-1].current_stream_wait()
                    error = None
                    try:
                        validate(result, expanded, zero_padding)
                    except Exception as caught:
                        error = caught
                    failed = torch.tensor([int(error is not None)], device=device)
                    dist.all_reduce(failed)
                    if int(failed.item()):
                        raise error if error is not None else AssertionError("peer payload check failed")
                    return result[3]

                handle = run(arguments.normal)
                run(arguments.cached(handle))
                expanded_handle = run(arguments.expanded, expanded=True)
                run(arguments.cached_expanded(expanded_handle), expanded=True, zero_padding=True)
                completed.append(name)
                if rank == 0:
                    print(f"DISPATCH_COPY_CASE_PASSED {name}", flush=True)
            finally:
                buffer.destroy()
        if rank == 0:
            Path(args.output).write_text(json.dumps({
                "world_size": world, "passed": completed,
                "scope": "hidden payload, FP8 scales, routing metadata and padding; no top-k weights",
            }, indent=2) + "\n")
            print("DISPATCH_COPY_BOUNDARIES_PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
