import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta


WORLD_SIZE = 4
SCALE_UP_SIZE = 2
HIDDEN = 4
BARRIER_GENERATIONS = 100
STEP_TIMEOUT_SECONDS = 30
BARRIER_TIMEOUT_SECONDS = 240
CASE_NAMES = (
    "symmetric-all-to-all",
    "asymmetric-routing",
    "empty-source-ranks",
    "cached-handle-reuse",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    payloads: tuple
    routes: tuple
    expected_receives: tuple
    cached_payloads: object = None
    cached_expected_receives: object = None


def _case_specs():
    return {
        "symmetric-all-to-all": CaseSpec(
            "symmetric-all-to-all",
            (
                ((0, 1, 2, 3), (8, 9, 10, 11),
                 (16, 17, 18, 19), (24, 25, 26, 27)),
                ((64, 65, 66, 67), (72, 73, 74, 75),
                 (80, 81, 82, 83), (88, 89, 90, 91)),
                ((128, 129, 130, 131), (136, 137, 138, 139),
                 (144, 145, 146, 147), (152, 153, 154, 155)),
                ((192, 193, 194, 195), (200, 201, 202, 203),
                 (208, 209, 210, 211), (216, 217, 218, 219)),
            ),
            (
                (0, 1, 2, 3),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 1, 0),
            ),
            (
                ((0, 1, 2, 3), (72, 73, 74, 75),
                 (144, 145, 146, 147), (216, 217, 218, 219)),
                ((8, 9, 10, 11), (64, 65, 66, 67),
                 (152, 153, 154, 155), (208, 209, 210, 211)),
                ((16, 17, 18, 19), (88, 89, 90, 91),
                 (128, 129, 130, 131), (200, 201, 202, 203)),
                ((24, 25, 26, 27), (80, 81, 82, 83),
                 (136, 137, 138, 139), (192, 193, 194, 195)),
            ),
        ),
        "asymmetric-routing": CaseSpec(
            "asymmetric-routing",
            (
                ((30, 31, 32, 33), (34, 35, 36, 37),
                 (38, 39, 40, 41)),
                ((60, 61, 62, 63),),
                ((90, 91, 92, 93), (94, 95, 96, 97),
                 (98, 99, 100, 101), (102, 103, 104, 105)),
                ((120, 121, 122, 123), (124, 125, 126, 127)),
            ),
            (
                (0, 0, 3),
                (2,),
                (1, 1, 1, 3),
                (0, 2),
            ),
            (
                ((30, 31, 32, 33), (34, 35, 36, 37),
                 (120, 121, 122, 123)),
                ((90, 91, 92, 93), (94, 95, 96, 97),
                 (98, 99, 100, 101)),
                ((60, 61, 62, 63), (124, 125, 126, 127)),
                ((38, 39, 40, 41), (102, 103, 104, 105)),
            ),
        ),
        "empty-source-ranks": CaseSpec(
            "empty-source-ranks",
            (
                ((140, 141, 142, 143), (144, 145, 146, 147)),
                (),
                ((160, 161, 162, 163), (164, 165, 166, 167)),
                (),
            ),
            ((0, 2), (), (1, 3), ()),
            (
                ((140, 141, 142, 143),),
                ((160, 161, 162, 163),),
                ((144, 145, 146, 147),),
                ((164, 165, 166, 167),),
            ),
        ),
        "cached-handle-reuse": CaseSpec(
            "cached-handle-reuse",
            (
                ((10, 11, 12, 13), (14, 15, 16, 17)),
                ((50, 51, 52, 53), (54, 55, 56, 57)),
                ((90, 91, 92, 93), (94, 95, 96, 97)),
                ((130, 131, 132, 133), (134, 135, 136, 137)),
            ),
            ((0, 3), (1, 2), (2, 1), (3, 0)),
            (
                ((10, 11, 12, 13), (134, 135, 136, 137)),
                ((50, 51, 52, 53), (94, 95, 96, 97)),
                ((54, 55, 56, 57), (90, 91, 92, 93)),
                ((14, 15, 16, 17), (130, 131, 132, 133)),
            ),
            (
                ((18, 19, 20, 21), (22, 23, 24, 25)),
                ((58, 59, 60, 61), (62, 63, 64, 65)),
                ((98, 99, 100, 101), (102, 103, 104, 105)),
                ((138, 139, 140, 141), (142, 143, 144, 145)),
            ),
            (
                ((18, 19, 20, 21), (142, 143, 144, 145)),
                ((58, 59, 60, 61), (102, 103, 104, 105)),
                ((62, 63, 64, 65), (98, 99, 100, 101)),
                ((22, 23, 24, 25), (138, 139, 140, 141)),
            ),
        ),
    }


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
            "logical-single-host requires exact Ascend topology environment: "
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


def _literal_receive(payloads, routes):
    receives = [[] for _ in range(WORLD_SIZE)]
    for source_rank in range(WORLD_SIZE):
        _check(len(payloads[source_rank]) == len(routes[source_rank]),
               f"rank {source_rank} payload/route counts differ")
        for payload, destination in zip(
                payloads[source_rank], routes[source_rank], strict=True):
            _check(len(payload) == HIDDEN,
                   f"rank {source_rank} payload width is not {HIDDEN}")
            _check(0 <= destination < WORLD_SIZE,
                   f"rank {source_rank} has invalid route {destination}")
            receives[destination].append(payload)
    return tuple(tuple(receive) for receive in receives)


def _validate_case_specs(specs):
    _check(tuple(specs) == CASE_NAMES,
           "logical smoke case registry is incomplete")
    for spec in specs.values():
        _check(len(spec.payloads) == WORLD_SIZE and
               len(spec.routes) == WORLD_SIZE and
               len(spec.expected_receives) == WORLD_SIZE,
               f"{spec.name} does not define all ranks")
        _check(_literal_receive(spec.payloads, spec.routes) ==
               spec.expected_receives,
               f"{spec.name} literal expected receives are inconsistent")
        has_cached_payloads = spec.cached_payloads is not None
        has_cached_expected = spec.cached_expected_receives is not None
        _check(has_cached_payloads == has_cached_expected,
               f"{spec.name} cached literals are incomplete")
        if not has_cached_payloads:
            continue
        _check(tuple(map(len, spec.cached_payloads)) ==
               tuple(map(len, spec.payloads)),
               f"{spec.name} cached payload layout changed")
        _check(_literal_receive(spec.cached_payloads, spec.routes) ==
               spec.cached_expected_receives,
               f"{spec.name} cached expected receives are inconsistent")


def _failure_record(rank, label, error, elapsed):
    return {
        "elapsed_seconds": round(elapsed, 3),
        "error_type": type(error).__name__,
        "label": label,
        "message": str(error),
        "rank": rank,
    }


def _synchronized_step(dist, group, operation, label,
                       timeout_seconds=STEP_TIMEOUT_SECONDS):
    _check(timeout_seconds > 0, "step timeout must be positive")
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    started = time.monotonic()
    local_error = None
    result = None
    try:
        result = operation()
    except BaseException as error:
        local_error = error
    elapsed = time.monotonic() - started
    if elapsed > timeout_seconds:
        detail = (
            f"elapsed {elapsed:.3f}s exceeded {timeout_seconds:.3f}s")
        if local_error is not None:
            detail += (
                f" after {type(local_error).__name__}: {local_error}")
        local_error = TimeoutError(detail)

    local_failure = None if local_error is None else _failure_record(
        rank, label, local_error, elapsed)
    gathered_failures = [None] * world_size
    try:
        dist.all_gather_object(
            gathered_failures, local_failure, group=group)
    except BaseException as aggregation_error:
        detail = (
            f"{label} failure aggregation failed on rank {rank}: "
            f"{type(aggregation_error).__name__}: {aggregation_error}")
        if local_error is not None:
            detail += (
                f"; local error was {type(local_error).__name__}: "
                f"{local_error}")
        raise RuntimeError(detail) from (
            local_error if local_error is not None else aggregation_error)

    failures = [failure for failure in gathered_failures
                if failure is not None]
    if failures:
        raise RuntimeError(
            f"{label} failed collectively: "
            f"{json.dumps(failures, sort_keys=True)}") from local_error
    return result


def _destroy_twice(buffer):
    failures = []
    for attempt in range(1, 3):
        try:
            buffer.destroy()
        except BaseException as error:
            failures.append({
                "attempt": attempt,
                "error_type": type(error).__name__,
                "message": str(error),
            })
    if failures:
        raise RuntimeError(
            f"buffer destroy failures: {json.dumps(failures, sort_keys=True)}")


def _cleanup_runtime(buffer, synchronized_step, destroy_process_group):
    failures = []
    if buffer is not None:
        try:
            synchronized_step(
                lambda: _destroy_twice(buffer), "buffer teardown")
        except BaseException as error:
            failures.append({
                "error_type": type(error).__name__,
                "message": str(error),
                "stage": "buffer",
            })
    try:
        destroy_process_group()
    except BaseException as error:
        failures.append({
            "error_type": type(error).__name__,
            "message": str(error),
            "stage": "process_group",
        })
    if failures:
        raise RuntimeError(
            f"teardown failures: {json.dumps(failures, sort_keys=True)}")


def _contract():
    specs = _case_specs()
    _validate_case_specs(specs)
    return {
        "barrier_generations": BARRIER_GENERATIONS,
        "barrier_timeout_seconds": BARRIER_TIMEOUT_SECONDS,
        "case_names": list(CASE_NAMES),
        "contract_checks": [
            "literal-independent-fixtures",
            "literal-expected-receives",
            "bounded-step-elapsed",
            "collective-rank-error-aggregation",
            "cached-handle-identity",
            "dispatch-output-validation",
            "combine-round-trip-validation",
            "aggregate-teardown",
        ],
        "environment": {
            "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
            "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
        },
        "evidence": "logical-single-host",
        "expected_logical_domain": [2, 2],
        "expected_physical_domain": [1, 4],
        "expected_world_size": WORLD_SIZE,
        "rank_mapping": [list(_rank_mapping(rank))
                         for rank in range(WORLD_SIZE)],
        "route_matrix": {
            str(rank): _route_matrix(rank) for rank in range(WORLD_SIZE)
        },
        "step_timeout_seconds": STEP_TIMEOUT_SECONDS,
        "system_under_test": [
            "ElasticBuffer.barrier",
            "ElasticBuffer.dispatch",
            "ElasticBuffer.combine",
        ],
    }


class LogicalScaleOutSmoke:
    def __init__(self, torch, dist, deep_ep, group, device, local_rank):
        self.torch = torch
        self.dist = dist
        self.deep_ep = deep_ep
        self.group = group
        self.device = device
        self.local_rank = local_rank
        self.rank = dist.get_rank(group)
        self.specs = _case_specs()
        _validate_case_specs(self.specs)
        self.buffer = None

    def _step(self, operation, label, timeout_seconds=STEP_TIMEOUT_SECONDS):
        return _synchronized_step(
            self.dist, self.group, operation, label,
            timeout_seconds=timeout_seconds)

    def make_buffer(self):
        self.buffer = self._step(
            lambda: self.deep_ep.ElasticBuffer(
                self.group,
                num_bytes=2 * 1024 * 1024,
                num_gpu_timeout_secs=5,
                allow_hybrid_mode=False,
                explicitly_destroy=True,
            ),
            "buffer construction",
        )

        def validate_topology():
            _check(self.buffer.get_logical_domain_size() == (2, 2),
                   "logical domain is not 2x2")
            _check(self.buffer.get_physical_domain_size() == (1, 4),
                   "logical simulation was exposed as a physical domain")
            _check(
                (self.buffer.scaleout_rank_idx,
                 self.buffer.scaleup_rank_idx) == _rank_mapping(self.rank),
                f"rank {self.rank} does not use row-major 2x2 mapping")

        self._step(validate_topology, "topology validation")

    def _materialize(self, payloads, routes):
        local_payloads = payloads[self.rank]
        local_routes = routes[self.rank]
        x = self.torch.tensor(
            local_payloads, dtype=self.torch.bfloat16,
            device=self.device).reshape(len(local_payloads), HIDDEN).contiguous()
        topk_idx = self.torch.tensor(
            local_routes, dtype=self.torch.int64,
            device=self.device).reshape(len(local_routes), 1).contiguous()
        return x, topk_idx

    def _dispatch(self, x, topk_idx, handle):
        if handle is not None:
            return self.buffer.dispatch(
                x, handle=handle, num_sms=1, num_qps=0)
        return self.buffer.dispatch(
            x,
            topk_idx=topk_idx,
            num_experts=WORLD_SIZE,
            num_max_tokens_per_rank=WORLD_SIZE,
            expert_alignment=1,
            num_sms=1,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        )

    def _check_local_tensor(self, tensor, label):
        _check(tensor.device.type == "npu" and
               tensor.device.index == self.local_rank,
               f"{label} is not on the selected local NPU")

    @staticmethod
    def _check_event(event, label):
        _check(event.event is None and event.extra_tensors is None and
               event.hook_after_wait is None,
               f"synchronous {label} returned deferred state")

    def _validate_dispatch(self, result, expected_receive, provided_handle):
        recv_x, recv_topk_idx, recv_weights, handle, event = result
        expected_rows = len(expected_receive)
        _check(tuple(recv_x.shape) == (expected_rows, HIDDEN),
               f"dispatch shape mismatch: {tuple(recv_x.shape)}")
        _check(recv_x.dtype == self.torch.bfloat16,
               "dispatch output is not BF16")
        self._check_local_tensor(recv_x, "dispatch output")
        expected_x = self.torch.tensor(
            expected_receive, dtype=self.torch.bfloat16,
            device=self.device).reshape(expected_rows, HIDDEN)
        self.torch.testing.assert_close(recv_x, expected_x, rtol=0, atol=0)
        _check(recv_topk_idx is not None and
               tuple(recv_topk_idx.shape) == (expected_rows, 1),
               "dispatch top-k shape mismatch")
        expected_topk = self.torch.zeros(
            (expected_rows, 1), dtype=self.torch.int64,
            device=self.device)
        self.torch.testing.assert_close(
            recv_topk_idx, expected_topk, rtol=0, atol=0)
        _check(recv_weights is None,
               "dispatch returned unexpected routing weights")
        _check(handle is not None, "dispatch did not return a handle")
        if provided_handle is not None:
            _check(handle is provided_handle,
                   "cached dispatch replaced its public handle")
        self._check_event(event, "dispatch")
        return recv_x, handle

    def _validate_combine(self, result, expected_x):
        combined_x, combined_weights, event = result
        _check(tuple(combined_x.shape) == tuple(expected_x.shape),
               f"combine shape mismatch: {tuple(combined_x.shape)}")
        _check(combined_x.dtype == self.torch.bfloat16,
               "combine output is not BF16")
        self._check_local_tensor(combined_x, "combine output")
        self.torch.testing.assert_close(
            combined_x, expected_x, rtol=0, atol=0)
        _check(combined_weights is None,
               "combine returned unexpected routing weights")
        self._check_event(event, "combine")
        return combined_x

    def _round_trip(self, spec, payloads, expected_receives, variant,
                    handle=None):
        label = f"{spec.name} {variant}"
        x, topk_idx = self._step(
            lambda: self._materialize(payloads, spec.routes),
            f"{label}: materialize")
        dispatch_result = self._step(
            lambda: self._dispatch(x, topk_idx, handle),
            f"{label}: dispatch")
        recv_x, returned_handle = self._step(
            lambda: self._validate_dispatch(
                dispatch_result, expected_receives[self.rank], handle),
            f"{label}: validate dispatch")
        combine_result = self._step(
            lambda: self.buffer.combine(
                recv_x, returned_handle, num_sms=1, num_qps=0),
            f"{label}: combine")
        combined_x = self._step(
            lambda: self._validate_combine(combine_result, x),
            f"{label}: validate combine")
        return {
            "combined_x": combined_x,
            "handle": returned_handle,
            "recv_x": recv_x,
        }

    def _run_barriers(self):
        def run_generations():
            for _ in range(BARRIER_GENERATIONS):
                self.buffer.barrier(
                    with_cpu_sync=False, sequential=True)

        self._step(
            run_generations, "100 barrier generations",
            timeout_seconds=BARRIER_TIMEOUT_SECONDS)

    def _run_cached_case(self, spec):
        first = self._round_trip(
            spec, spec.payloads, spec.expected_receives, "fresh")
        second = self._round_trip(
            spec, spec.cached_payloads, spec.cached_expected_receives,
            "cached", handle=first["handle"])
        _check(second["handle"] is first["handle"],
               "cached round trip did not reuse the same handle")
        _check(not self.torch.equal(first["recv_x"], second["recv_x"]),
               "cached dispatch ignored changed payloads")
        _check(not self.torch.equal(
                   first["combined_x"], second["combined_x"]),
               "cached combine ignored changed payloads")

    def run(self):
        self._run_barriers()
        for name in CASE_NAMES:
            spec = self.specs[name]
            if spec.cached_payloads is not None:
                self._run_cached_case(spec)
            else:
                self._round_trip(
                    spec, spec.payloads, spec.expected_receives, "fresh")


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
    group = None
    smoke = None
    rank = None
    runtime_error = None
    cleanup_error = None
    try:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(minutes=5))
        group = dist.group.WORLD
        rank = dist.get_rank(group)
        _validate_world(dist.get_world_size(group))
        device = torch.device("npu", local_rank)
        smoke = LogicalScaleOutSmoke(
            torch, dist, deep_ep, group, device, local_rank)
        smoke.make_buffer()
        smoke.run()
    except BaseException as error:
        runtime_error = error
    finally:
        if group is not None:
            try:
                _cleanup_runtime(
                    None if smoke is None else smoke.buffer,
                    lambda operation, label: _synchronized_step(
                        dist, group, operation, label),
                    dist.destroy_process_group)
            except BaseException as error:
                cleanup_error = error

    if runtime_error is not None:
        if cleanup_error is not None:
            raise RuntimeError(
                f"runtime failed: {runtime_error}; cleanup failed: "
                f"{cleanup_error}") from runtime_error
        raise runtime_error
    if cleanup_error is not None:
        raise cleanup_error
    if rank == 0:
        print(
            "PASS logical-single-host four-rank logical scale-out smoke",
            flush=True)


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
