import argparse
import json
import os
import time
from datetime import timedelta


WORLD_SIZE = 4
SCALE_UP_SIZE = 2
HIDDEN = 4
BARRIER_GENERATIONS = 100


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _validate_environment(environ):
    expected = {
        "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
        "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
    }
    mismatches = {
        name: environ.get(name) for name, value in expected.items()
        if environ.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"logical-single-host requires exact Ascend topology environment: "
            f"expected {expected}, observed {mismatches}")


def _validate_world(world_size):
    if world_size != WORLD_SIZE:
        raise RuntimeError(
            f"logical-single-host requires exactly {WORLD_SIZE} ranks, "
            f"got {world_size}")


def _rank_mapping(rank):
    return rank // SCALE_UP_SIZE, rank % SCALE_UP_SIZE


def _route_matrix(rank):
    return {
        "self": rank,
        "local": rank ^ 1,
        "rail": rank ^ 2,
        "diagonal": rank ^ 3,
    }


def _routes_for_rank(rank):
    routes = _route_matrix(rank)
    return [
        routes["self"], routes["local"], routes["rail"],
        routes["diagonal"],
    ]


def _payloads_for_rank(rank):
    return [
        [rank * 64 + token * 8 + column for column in range(HIDDEN)]
        for token in range(WORLD_SIZE)
    ]


def _expected_receive(gathered_payloads, gathered_routes, destination_rank):
    received = []
    for source_rank in range(WORLD_SIZE):
        payloads = gathered_payloads[source_rank]
        routes = gathered_routes[source_rank]
        if len(payloads) != WORLD_SIZE or len(routes) != WORLD_SIZE:
            raise AssertionError(
                f"rank {source_rank} did not publish four payload routes")
        for token, route in enumerate(routes):
            if route == destination_rank:
                received.append(payloads[token])
    if len(received) != WORLD_SIZE:
        raise AssertionError(
            f"rank {destination_rank} expected {WORLD_SIZE} records, "
            f"got {len(received)}")
    return received


def _synchronized_step(torch, dist, group, device, operation, label):
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


def _destroy_twice(buffer):
    buffer.destroy()
    buffer.destroy()


def _contract():
    gathered_payloads = [_payloads_for_rank(rank) for rank in range(WORLD_SIZE)]
    gathered_routes = [_routes_for_rank(rank) for rank in range(WORLD_SIZE)]
    return {
        "barrier_generations": BARRIER_GENERATIONS,
        "cases": ["barrier", "bf16-all-to-all", "combine-round-trip"],
        "environment": {
            "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
            "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
        },
        "evidence": "logical-single-host",
        "expected_domain": [2, 2],
        "expected_world_size": WORLD_SIZE,
        "rank_mapping": [list(_rank_mapping(rank))
                         for rank in range(WORLD_SIZE)],
        "route_matrix": {
            str(rank): _route_matrix(rank) for rank in range(WORLD_SIZE)
        },
        "rank0_expected_receive": _expected_receive(
            gathered_payloads, gathered_routes, 0),
        "expected_local_topk": [[0] for _ in range(WORLD_SIZE)],
        "system_under_test": [
            "ElasticBuffer.barrier",
            "ElasticBuffer.dispatch",
            "ElasticBuffer.combine",
        ],
    }


def _run_runtime():
    _validate_environment(os.environ)

    import torch
    import torch.distributed as dist
    import torch_npu

    import deep_ep
    import deep_ep._C as extension

    try:
        from api_surface import assert_no_testing_diagnostic_surface
    except ModuleNotFoundError:
        from tests.ascend.production.api_surface import \
            assert_no_testing_diagnostic_surface

    del torch_npu
    assert_no_testing_diagnostic_surface(extension)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("npu", local_rank)
    buffer = None

    def synchronized_step(operation, label):
        return _synchronized_step(
            torch, dist, group, device, operation, label)

    try:
        _validate_world(world_size)
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

        def validate_topology():
            _check(buffer.get_logical_domain_size() == (2, 2),
                   "logical domain is not 2x2")
            _check(
                (buffer.scaleout_rank_idx, buffer.scaleup_rank_idx) ==
                    _rank_mapping(rank),
                f"rank {rank} does not use row-major 2x2 mapping")

        synchronized_step(validate_topology, "topology validation")

        def run_barriers():
            started = time.monotonic()
            for _ in range(BARRIER_GENERATIONS):
                buffer.barrier(with_cpu_sync=False, sequential=True)
            elapsed = time.monotonic() - started
            _check(elapsed < 240,
                   f"100 barrier generations exceeded bound: {elapsed:.3f}s")

        synchronized_step(run_barriers, "bounded barrier generations")

        payloads = _payloads_for_rank(rank)
        routes = _routes_for_rank(rank)
        gathered = [None] * WORLD_SIZE
        synchronized_step(
            lambda: dist.all_gather_object(
                gathered, {"payloads": payloads, "routes": routes},
                group=group),
            "reference gather",
        )
        gathered_payloads = [entry["payloads"] for entry in gathered]
        gathered_routes = [entry["routes"] for entry in gathered]
        expected_receive = _expected_receive(
            gathered_payloads, gathered_routes, rank)

        x = torch.tensor(payloads, dtype=torch.bfloat16, device=device)
        topk_idx = torch.tensor(
            routes, dtype=torch.int64, device=device).reshape(WORLD_SIZE, 1)
        dispatch_result = synchronized_step(
            lambda: buffer.dispatch(
                x,
                topk_idx=topk_idx,
                num_experts=WORLD_SIZE,
                num_max_tokens_per_rank=WORLD_SIZE,
                expert_alignment=1,
                num_sms=1,
                num_qps=0,
                do_handle_copy=True,
                do_cpu_sync=True,
            ),
            "BF16 all-to-all dispatch",
        )

        def validate_dispatch():
            recv_x, recv_topk_idx, recv_weights, handle, event = dispatch_result
            _check(tuple(recv_x.shape) == (WORLD_SIZE, HIDDEN),
                   f"dispatch shape mismatch: {tuple(recv_x.shape)}")
            _check(recv_x.dtype == torch.bfloat16,
                   "dispatch output is not BF16")
            _check(recv_x.device.type == "npu" and
                   recv_x.device.index == local_rank,
                   "dispatch output is not on the selected local NPU")
            expected_x = torch.tensor(
                expected_receive, dtype=torch.bfloat16, device=device)
            torch.testing.assert_close(recv_x, expected_x, rtol=0, atol=0)
            expected_topk = torch.zeros(
                (WORLD_SIZE, 1), dtype=torch.int64, device=device)
            torch.testing.assert_close(
                recv_topk_idx, expected_topk, rtol=0, atol=0)
            _check(recv_weights is None,
                   "dispatch returned unexpected routing weights")
            _check(event.event is None and event.extra_tensors is None and
                   event.hook_after_wait is None,
                   "synchronous dispatch returned deferred state")
            return recv_x, handle

        recv_x, handle = synchronized_step(
            validate_dispatch, "dispatch validation")
        combine_result = synchronized_step(
            lambda: buffer.combine(recv_x, handle, num_sms=1, num_qps=0),
            "combine round trip",
        )

        def validate_combine():
            combined_x, combined_weights, event = combine_result
            torch.testing.assert_close(combined_x, x, rtol=0, atol=0)
            _check(combined_weights is None,
                   "combine returned unexpected routing weights")
            _check(event.event is None and event.extra_tensors is None and
                   event.hook_after_wait is None,
                   "synchronous combine returned deferred state")

        synchronized_step(validate_combine, "combine validation")
        if rank == 0:
            print(
                "PASS logical-single-host four-rank logical scale-out smoke",
                flush=True)
    finally:
        if buffer is not None:
            _destroy_twice(buffer)
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
