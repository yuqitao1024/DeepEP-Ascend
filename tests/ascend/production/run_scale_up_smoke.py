import argparse
import json
import os
from datetime import timedelta


HIDDEN = 4


def _contract():
    return {
        "minimum_world_size": 2,
        "rank_limit": None,
        "cases": ["barrier", "bf16-all-to-all-round-trip"],
        "num_experts": "world_size",
        "num_tokens_per_rank": "world_size",
        "system_under_test": [
            "ElasticBuffer.barrier",
            "ElasticBuffer.dispatch",
            "ElasticBuffer.combine",
        ],
    }


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _run_runtime():
    import torch
    import torch.distributed as dist
    import torch_npu

    import deep_ep

    del torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size < 2:
        raise RuntimeError(
            f"scale-up smoke requires at least two ranks, got {world_size}")
    device = torch.device("npu", local_rank)
    buffer = None

    def synchronized_step(operation, label):
        local_error = None
        result = None
        try:
            result = operation()
        except BaseException as error:
            local_error = error
        failed = torch.tensor(
            [int(local_error is not None)], dtype=torch.int32, device=device)
        dist.all_reduce(failed, group=group)
        if int(failed.item()) != 0:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"{label} failed on a peer rank")
        return result

    try:
        buffer = synchronized_step(
            lambda: deep_ep.ElasticBuffer(
                group,
                num_bytes=2 * 1024 * 1024,
                num_gpu_timeout_secs=5,
                allow_hybrid_mode=False,
                explicitly_destroy=True,
            ),
            "buffer construction",
        )
        synchronized_step(
            lambda: buffer.barrier(with_cpu_sync=True, sequential=True),
            "barrier",
        )

        def materialize():
            values = torch.arange(
                world_size * HIDDEN, dtype=torch.float32, device=device)
            x = (values.reshape(world_size, HIDDEN) + rank * 100).to(
                torch.bfloat16)
            routes = torch.arange(
                world_size, dtype=torch.int64, device=device).reshape(
                    world_size, 1)
            return x, routes

        x, routes = synchronized_step(materialize, "input materialization")
        dispatch_result = synchronized_step(
            lambda: buffer.dispatch(
                x,
                topk_idx=routes,
                num_experts=world_size,
                num_max_tokens_per_rank=world_size,
                expert_alignment=1,
                num_sms=1,
                num_qps=0,
                do_handle_copy=True,
                do_cpu_sync=True,
            ),
            "dispatch",
        )

        def validate_dispatch():
            recv_x, recv_topk_idx, recv_weights, handle, event = \
                dispatch_result
            _check(
                tuple(recv_x.shape) == (world_size, HIDDEN),
                f"dispatch shape mismatch: {tuple(recv_x.shape)}")
            _check(
                recv_x.dtype == torch.bfloat16 and
                recv_x.device.type == "npu" and
                recv_x.device.index == local_rank,
                "dispatch output is not local BF16 NPU storage")
            _check(
                tuple(recv_topk_idx.shape) == (world_size, 1),
                "dispatch top-k shape mismatch")
            _check(recv_weights is None, "dispatch returned unexpected weights")
            _check(handle.num_experts == world_size, "handle expert count mismatch")
            _check(
                handle.num_max_tokens_per_rank == world_size,
                "handle token capacity mismatch")
            _check(
                event.event is None and event.extra_tensors is None and
                event.hook_after_wait is None,
                "synchronous dispatch returned deferred state")
            return recv_x, handle

        recv_x, handle = synchronized_step(
            validate_dispatch, "dispatch validation")
        combine_result = synchronized_step(
            lambda: buffer.combine(
                recv_x, handle, num_sms=1, num_qps=0),
            "combine",
        )

        def validate_combine():
            combined_x, combined_weights, event = combine_result
            _check(combined_weights is None, "combine returned unexpected weights")
            _check(
                event.event is None and event.extra_tensors is None and
                event.hook_after_wait is None,
                "synchronous combine returned deferred state")
            torch.testing.assert_close(
                combined_x.detach().cpu(), x.detach().cpu(), rtol=0, atol=0)

        synchronized_step(validate_combine, "combine validation")
        if rank == 0:
            print(
                f"Ascend {world_size}-rank scale-up smoke passed",
                flush=True,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", action="store_true")
    args = parser.parse_args()
    if args.contract:
        print(json.dumps(_contract(), sort_keys=True))
        return 0
    _run_runtime()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
