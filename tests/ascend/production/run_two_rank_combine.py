import argparse
import ast
import json
import os
import re
import struct
import time
from dataclasses import dataclass
from datetime import timedelta


WORLD_SIZE = 2
HIDDEN = 4
NUM_EXPERTS = 4
NUM_TOPK = 2
CAPACITY = 4
BF16_TOLERANCE = 1 / 128
PADDING_SENTINEL = -64.0
ORDER_VARIANT = "order-sensitive"
ORDER_BOUNDARY = float(1 << 24)

CASE_NAMES = (
    "normal",
    "expanded-multiple-reduction",
    "expanded-single-reduction",
    "weights",
    "zero-bias",
    "one-bias",
    "two-bias",
    "duplicate-same-rank-experts",
    "negative-one-route",
    "empty-input",
    "asymmetric-routing",
    "aligned-padding",
    "aligned-near-capacity",
    "cached-dispatch-changed-outputs",
    "sequential-100-generations",
    "cross-buffer-handle",
    "malformed-handle",
    "bounded-peer-diagnostics",
    "repeated-teardown",
)

REGULAR_CASES = CASE_NAMES[:13]
SPECIAL_CASES = CASE_NAMES[13:]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    payloads: tuple
    routes: tuple
    weights: object = None
    bias_count: int = 0
    do_expand: bool = False
    do_zero_padding: bool = False
    expert_alignment: int = 1
    allow_multiple_reduction: bool = True
    num_topk: int = NUM_TOPK
    require_padding: bool = False
    transform_variant: object = 0


def _payloads(counts, offset):
    return tuple(
        tuple(
            tuple(offset + rank * 40 + token * HIDDEN + column
                  for column in range(HIDDEN))
            for token in range(count))
        for rank, count in enumerate(counts))


def _case_specs():
    normal_routes = (
        ((0, 2), (1, -1)),
        ((2, 0), (3, -1)),
    )
    return {
        "normal": CaseSpec(
            "normal", _payloads((2, 2), 1), normal_routes),
        "expanded-multiple-reduction": CaseSpec(
            "expanded-multiple-reduction", _payloads((2, 2), 3),
            (((0, 1), (2, 0)), ((3, 2), (1, 3))),
            do_expand=True),
        "expanded-single-reduction": CaseSpec(
            "expanded-single-reduction", _payloads((1, 1), 5),
            (((0, 2, 3),), ((0, 2, 3),)),
            do_expand=True, allow_multiple_reduction=False, num_topk=3,
            transform_variant=ORDER_VARIANT),
        "weights": CaseSpec(
            "weights", _payloads((2, 2), 7),
            (((0, 2), (1, -1)), ((2, -1), (3, 0))),
            (((0.125, 0.25), (0.375, 0.875)),
             ((0.5, 0.625), (0.75, 1.0)))),
        "zero-bias": CaseSpec(
            "zero-bias", _payloads((2, 2), 9), normal_routes),
        "one-bias": CaseSpec(
            "one-bias", _payloads((2, 2), 11), normal_routes,
            bias_count=1),
        "two-bias": CaseSpec(
            "two-bias", _payloads((2, 2), 13), normal_routes,
            bias_count=2),
        "duplicate-same-rank-experts": CaseSpec(
            "duplicate-same-rank-experts", _payloads((2, 2), 15),
            (((0, 1), (2, 3)), ((1, 0), (3, 2)))),
        "negative-one-route": CaseSpec(
            "negative-one-route", _payloads((2, 2), 17),
            (((-1, -1), (2, -1)), ((-1, 0), (3, -1)))),
        "empty-input": CaseSpec(
            "empty-input", ((), ()), ((), ())),
        "asymmetric-routing": CaseSpec(
            "asymmetric-routing", _payloads((3, 1), 19),
            (((2, -1), (0, 3), (-1, -1)), ((1, -1),))),
        "aligned-padding": CaseSpec(
            "aligned-padding", _payloads((2, 1), 21),
            (((0, 2), (0, -1)), ((1, 3),)),
            do_expand=True, do_zero_padding=True, expert_alignment=4,
            require_padding=True),
        "aligned-near-capacity": CaseSpec(
            "aligned-near-capacity", _payloads((4, 4), 23),
            (((0, 2), (0, 3), (1, 2), (1, 3)),
             ((0, 2), (1, 2), (0, 3), (1, 3))),
            do_expand=True, do_zero_padding=True, expert_alignment=4),
        "cached-dispatch-changed-outputs": CaseSpec(
            "cached-dispatch-changed-outputs", _payloads((2, 2), 25),
            normal_routes,
            (((0.125, 0.25), (0.375, 0.5)),
             ((0.625, 0.75), (0.875, 1.0)))),
        "cross-buffer-handle": CaseSpec(
            "cross-buffer-handle", _payloads((2, 2), 27), normal_routes),
        "malformed-handle": CaseSpec(
            "malformed-handle", _payloads((2, 2), 29), normal_routes),
        "bounded-peer-diagnostics": CaseSpec(
            "bounded-peer-diagnostics", _payloads((1, 1), 31),
            (((NUM_EXPERTS, -1),), ((NUM_EXPERTS, -1),))),
        "repeated-teardown": CaseSpec(
            "repeated-teardown", _payloads((1, 1), 33),
            (((0, -1),), ((2, -1),))),
    }


def _synthetic_transform(origin_rank, origin_token, lane, variant=0):
    if variant == ORDER_VARIANT:
        order_columns = (
            (ORDER_BOUNDARY, 0.5),
            (0.5, -ORDER_BOUNDARY),
            (-ORDER_BOUNDARY, ORDER_BOUNDARY),
        )
        first, second = order_columns[lane]
        identity = 1 + origin_rank * 16 + origin_token * 4 + lane * 2
        return [first, second, float(identity), float(identity + 1)]
    base = 1 + origin_rank * 16 + origin_token * 4 + lane * 2 + variant
    return [float(base + column) for column in range(HIDDEN)]


def _float32_add(left, right):
    return struct.unpack("<f", struct.pack("<f", left + right))[0]


def _integer(value):
    return int(value.item()) if hasattr(value, "item") else int(value)


def _reference_values(routes, origin_rank, variant=0,
                      contributor_order=None, lane_order=None):
    local_experts = NUM_EXPERTS // WORLD_SIZE
    contributor_order = tuple(range(WORLD_SIZE)) if contributor_order is None \
        else tuple(contributor_order)
    result = []
    for origin_token, token_routes in enumerate(routes):
        lane_order_for_token = tuple(range(len(token_routes))) \
            if lane_order is None else tuple(lane_order)
        values = [0.0] * HIDDEN
        for contributor_rank in contributor_order:
            for lane in lane_order_for_token:
                expert = _integer(token_routes[lane])
                if expert < 0 or expert // local_experts != contributor_rank:
                    continue
                transform = _synthetic_transform(
                    origin_rank, origin_token, lane, variant)
                for column in range(HIDDEN):
                    values[column] = _float32_add(
                        values[column], transform[column])
        result.append(values)
    return result


def _masked_weight_values(routes, weights):
    return [
        [float(weight) if _integer(expert) >= 0 else 0.0
         for expert, weight in zip(token_routes, token_weights)]
        for token_routes, token_weights in zip(routes, weights)
    ]


def _apply_biases_once(values, biases):
    result = [list(row) for row in values]
    for bias in biases:
        for row, bias_row in zip(result, bias):
            for column in range(HIDDEN):
                row[column] = _float32_add(row[column], bias_row[column])
    return result


def _expanded_reference_rows(num_rows, metadata, gathered_routes,
                             contributor_rank, variant):
    rows = [[PADDING_SENTINEL] * HIDDEN for _ in range(num_rows)]
    mapped = set()
    for metadata_row in metadata:
        encoded_source = _integer(metadata_row[0])
        source_rank, source_token = divmod(encoded_source, CAPACITY)
        num_topk = len(metadata_row) - 2
        for lane in range(num_topk):
            destination = _integer(metadata_row[2 + lane])
            if destination < 0:
                continue
            expert = _integer(gathered_routes[source_rank][source_token][lane])
            local_experts = NUM_EXPERTS // WORLD_SIZE
            if expert // local_experts != contributor_rank:
                raise AssertionError("expanded metadata names a nonlocal lane")
            if destination in mapped or not 0 <= destination < num_rows:
                raise AssertionError("expanded metadata names an invalid row")
            mapped.add(destination)
            rows[destination] = _synthetic_transform(
                source_rank, source_token, lane, variant)
    return {
        "rows": rows,
        "mapped_rows": sorted(mapped),
        "padding_rows": [row for row in range(num_rows) if row not in mapped],
    }


def _reference_matrix(routes, origin_rank, variant=0):
    return {
        "shape": [len(routes), HIDDEN],
        "values": _reference_values(routes, origin_rank, variant),
    }


def _reference_fixture():
    routes = (
        ((0, 2), (1, -1)),
        ((2, 3),),
    )
    return {
        "rank0": _reference_values(routes[0], 0),
        "rank1": _reference_values(routes[1], 1),
    }


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _contract_tree():
    with open(__file__, encoding="utf-8") as source_file:
        return ast.parse(source_file.read(), filename=__file__)


def _method_map(tree, class_name):
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {
        node.name: node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _calls(node):
    return [
        _attribute_name(child.func) for child in ast.walk(node)
        if isinstance(child, ast.Call)
    ]


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _is_local_npu_tensor(tensor, device_index):
    device = getattr(tensor, "device", None)
    return device is not None and device.type == "npu" and \
        device.index == device_index


def _validate_expansion_mode(actual, expected):
    _check(actual == expected,
           "dispatch handle expansion mode does not match the case")


class CleanupFailures(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__("; ".join(str(failure) for failure in self.failures))


class FailureSynchronizationError(RuntimeError):
    pass


def _cleanup_runtime(matrix, destroy_process_group):
    failures = []
    if matrix is not None:
        try:
            matrix.destroy()
        except CleanupFailures as error:
            failures.extend(error.failures)
        except BaseException as error:
            failures.append(error)
    try:
        destroy_process_group()
    except BaseException as error:
        failures.append(error)
    if failures:
        raise CleanupFailures(failures)


def _behavior_fixtures():
    order_spec = _case_specs()["expanded-single-reduction"]
    _check(order_spec.do_expand and
           not order_spec.allow_multiple_reduction and
           order_spec.num_topk == 3 and
           order_spec.transform_variant == ORDER_VARIANT,
           "the public order-sensitive case is not registered")
    order_routes = order_spec.routes[0]
    canonical = _reference_values(
        order_routes, 0, ORDER_VARIANT)[0]
    lane_reversed = _reference_values(
        order_routes, 0, ORDER_VARIANT, lane_order=(2, 1, 0))[0]
    rank_reversed = _reference_values(
        order_routes, 0, ORDER_VARIANT, contributor_order=(1, 0))[0]

    weight_routes = ((0, -1),)
    activation_ignores_weight = _reference_values(weight_routes, 0)[0]
    restored_weights = _masked_weight_values(
        weight_routes, ((0.25, 0.75),))[0]
    bias_base = _reference_values(((0, 2),), 0)
    biases = (
        ((2.0, 2.0, 2.0, 2.0),),
        ((3.0, 3.0, 3.0, 3.0),),
    )
    bias_once = _apply_biases_once(bias_base, biases)[0]

    expanded = _expanded_reference_rows(
        4, ((0, 0, 2, -1),), (((0, 2),), ()), 0, 0)

    class FakeDevice:
        def __init__(self, device_type, index):
            self.type = device_type
            self.index = index

    class FakePlacedTensor:
        def __init__(self, device_type, index):
            self.device = FakeDevice(device_type, index)

    class FakeFlag:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def tensor(values, dtype, device):
            del dtype, device
            return FakeFlag(values[0])

    class FakeDist:
        def __init__(self, peer_failure=False, sync_failure=False):
            self.peer_failure = peer_failure
            self.sync_failure = sync_failure
            self.reductions = 0

        def all_reduce(self, flag, group):
            del group
            self.reductions += 1
            if self.sync_failure:
                raise RuntimeError("all_reduce failed")
            if self.peer_failure:
                flag.value = 1

    local_matrix = object.__new__(CombineMatrix)
    local_matrix.torch = FakeTorch()
    local_matrix.dist = FakeDist()
    local_matrix.group = object()
    local_matrix.device = FakeDevice("npu", 0)
    local_matrix.collectives_usable = True
    try:
        local_matrix._run_step(
            lambda: (_ for _ in ()).throw(ValueError("local failure")),
            "local-fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("local synchronized failure was accepted")

    peer_matrix = object.__new__(CombineMatrix)
    peer_matrix.torch = FakeTorch()
    peer_matrix.dist = FakeDist(peer_failure=True)
    peer_matrix.group = object()
    peer_matrix.device = FakeDevice("npu", 0)
    peer_matrix.collectives_usable = True
    peer_result_rejected = False
    try:
        peer_matrix._run_step(lambda: "must not escape", "peer-fixture")
    except RuntimeError:
        peer_result_rejected = True

    sync_matrix = object.__new__(CombineMatrix)
    sync_matrix.torch = FakeTorch()
    sync_matrix.dist = FakeDist(sync_failure=True)
    sync_matrix.group = object()
    sync_matrix.device = FakeDevice("npu", 0)
    sync_matrix.collectives_usable = True
    blocked_operation_calls = 0
    try:
        sync_matrix._run_step(lambda: None, "sync-fixture")
    except FailureSynchronizationError:
        pass
    try:
        def blocked_operation():
            nonlocal blocked_operation_calls
            blocked_operation_calls += 1

        sync_matrix._run_step(blocked_operation, "blocked-fixture")
    except FailureSynchronizationError:
        pass

    cleanup_calls = []

    class FakeBuffer:
        def __init__(self, name, fails):
            self.name = name
            self.fails = fails

        def destroy(self):
            cleanup_calls.append(self.name)
            if self.fails:
                raise RuntimeError(f"{self.name} failed")

    cleanup_matrix = object.__new__(CombineMatrix)
    cleanup_matrix.live_buffers = [
        FakeBuffer("buffer-c", True),
        FakeBuffer("buffer-b", False),
        FakeBuffer("buffer-a", True),
    ]
    cleanup_matrix.buffer = None
    cleanup_matrix.buffer_mode = None

    def destroy_group():
        cleanup_calls.append("process-group")
        raise RuntimeError("process-group failed")

    cleanup_failures = []
    try:
        _cleanup_runtime(cleanup_matrix, destroy_group)
    except CleanupFailures as error:
        cleanup_failures = [str(failure) for failure in error.failures]

    multiplied_activation = [
        _float32_add(0.0, value * 0.25)
        for value in activation_ignores_weight]
    bias_per_contributor = _apply_biases_once(
        _apply_biases_once(bias_base, biases), biases)[0]
    sentinel_read = [
        _float32_add(value, PADDING_SENTINEL)
        for value in activation_ignores_weight]
    handle_mode_rejected = False
    try:
        _validate_expansion_mode(False, True)
    except AssertionError:
        handle_mode_rejected = True

    def mutation_rejected(expected, mutant):
        try:
            _check(mutant == expected, "oracle mutation survived")
        except AssertionError:
            return True
        return False

    return {
        "cleanup": {
            "calls": cleanup_calls,
            "failures": cleanup_failures,
        },
        "expanded": {
            "mapped_rows": expanded["mapped_rows"],
            "padding_rows": expanded["padding_rows"],
            "row_2": expanded["rows"][2],
        },
        "mutations_rejected": {
            "bias_per_contributor": mutation_rejected(
                bias_once, bias_per_contributor),
            "expanded_reads_sentinel": mutation_rejected(
                activation_ignores_weight, sentinel_read),
            "handle_expansion_mode": handle_mode_rejected,
            "lane_order": mutation_rejected(canonical, lane_reversed),
            "rank_order": mutation_rejected(canonical, rank_reversed),
            "weight_multiplies_activation": mutation_rejected(
                activation_ignores_weight, multiplied_activation),
        },
        "ordering": {
            "canonical": canonical,
            "lane_reversed": lane_reversed,
            "rank_reversed": rank_reversed,
        },
        "placement": {
            "cpu": _is_local_npu_tensor(FakePlacedTensor("cpu", None), 0),
            "local_npu": _is_local_npu_tensor(
                FakePlacedTensor("npu", 0), 0),
            "wrong_npu": _is_local_npu_tensor(
                FakePlacedTensor("npu", 1), 0),
        },
        "synchronization": {
            "local_failure_reductions": local_matrix.dist.reductions,
            "peer_failure_reductions": peer_matrix.dist.reductions,
            "peer_result_rejected": peer_result_rejected,
            "post_sync_failure_operation_blocked":
                blocked_operation_calls == 0,
            "sync_failure_reductions": sync_matrix.dist.reductions,
        },
        "weight_bias": {
            "activation_ignores_weight": activation_ignores_weight,
            "bias_once": bias_once,
            "restored_weights": restored_weights,
        },
    }


def _contract():
    specs = _case_specs()
    _check(set(specs) == set(CASE_NAMES) - {"sequential-100-generations"},
           "combine case specifications do not match the matrix")
    _check(_reference_fixture() == {
        "rank0": [[4.0, 6.0, 8.0, 10.0], [5.0, 6.0, 7.0, 8.0]],
        "rank1": [[36.0, 38.0, 40.0, 42.0]],
    }, "the pure route/transform reference fixture changed")

    tree = _contract_tree()
    methods = _method_map(tree, "CombineMatrix")
    dispatch_calls = _calls(methods["_dispatch"])
    combine_calls = _calls(methods["_combine"])
    _check("buffer.dispatch" in dispatch_calls,
           "the system under test does not call public Buffer.dispatch")
    _check("buffer.combine" in combine_calls,
           "the system under test does not call public Buffer.combine")
    for call in ("self._gather_routes", "self._expert_outputs",
                 "self._expected", "self._verify"):
        _check(call in _calls(methods["_round_trip"]),
               f"round-trip path is missing {call}")
    _check("self.dist.all_gather" in _calls(methods["_gather_routes"]),
           "original literal routes are not gathered")
    _check("self._replace_buffer" in _calls(methods["_ensure_buffer"]),
           "reduction mode changes do not recreate the buffer")
    _check("self.dist.all_reduce" in _calls(methods["_run_step"]),
           "sub-operations do not synchronize failures")
    _check(_calls(methods["_round_trip"]).count("self._run_step") >= 6,
           "round-trip phases do not synchronize failures")
    _check(_calls(methods["_run_bounded_peer_diagnostics"]).count(
               "self._run_step") >= 3,
           "bounded diagnostics phases do not synchronize failures")
    _check("_is_local_npu_tensor" in _calls(methods["_verify"]),
           "combine outputs do not enforce local NPU placement")
    _check("_validate_expansion_mode" in
           _calls(methods["_expert_outputs"]),
           "expert outputs trust the returned handle mode")

    boundary_calls = _calls(methods["_case_boundary"])
    _check(boundary_calls.count("self.dist.barrier") == 2,
           "each case must have exactly two boundary barriers")
    _check("self.dist.all_reduce" in boundary_calls,
           "rank failures are not aggregated")
    for name, method in methods.items():
        if name != "_case_boundary":
            _check("self.dist.barrier" not in _calls(method),
                   f"barrier outside a case boundary: {name}")

    runtime = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_runtime")
    runtime_calls = _calls(runtime)
    _check(runtime_calls.count("dist.init_process_group") == 1 and
           "dist.new_group" not in runtime_calls,
           "runtime must use exactly one HCCL process group")
    _check("_cleanup_runtime" in runtime_calls,
           "buffer/group teardown is incomplete")
    cleanup = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and
        node.name == "_cleanup_runtime")
    cleanup_calls = _calls(cleanup)
    _check("matrix.destroy" in cleanup_calls and
           "destroy_process_group" in cleanup_calls,
           "buffers must be destroyed before the process group")

    return {
        "case_names": list(CASE_NAMES),
        "contract_checks": [
            "public-dispatch-combine",
            "literal-route-reference",
            "synthetic-origin-transform",
            "expanded-metadata-writes",
            "bf16-tolerance",
            "exact-float32-weights",
            "case-boundary-barriers",
            "distributed-failure-aggregation",
            "buffer-before-group-teardown",
            "reduction-mode-buffer-recreation",
            "bounded-peer-diagnostics",
            "public-handle-mutations",
            "one-hccl-group",
            "sub-operation-failure-synchronization",
            "order-sensitive-public-case",
            "oracle-behavior-mutations",
            "local-npu-output-placement",
            "best-effort-cleanup",
        ],
        "behavior_fixtures": _behavior_fixtures(),
        "expected_world_size": WORLD_SIZE,
        "empty_reference_shape": _reference_matrix((), 0)["shape"],
        "float32_order_fixture": _float32_add(
            _float32_add(16777216.0, 1.0), -16777216.0),
        "reference": "gathered-original-routes",
        "reference_fixture": _reference_fixture(),
        "system_under_test": ["Buffer.dispatch", "Buffer.combine"],
    }


class CombineMatrix:
    def __init__(self, torch, dist, deep_ep, group, device):
        self.torch = torch
        self.dist = dist
        self.deep_ep = deep_ep
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.specs = _case_specs()
        self.buffer = None
        self.buffer_mode = None
        self.live_buffers = []
        self.collectives_usable = True

    def _run_step(self, operation, label):
        if not self.collectives_usable:
            raise FailureSynchronizationError(
                f"{label} cannot start after synchronization failure")
        local_error = None
        result = None
        try:
            result = operation()
        except BaseException as error:
            local_error = error
        failed = self.torch.tensor(
            [int(local_error is not None)], dtype=self.torch.int32,
            device=self.device)
        try:
            self.dist.all_reduce(failed, group=self.group)
        except BaseException as sync_error:
            self.collectives_usable = False
            if local_error is not None:
                raise FailureSynchronizationError(
                    f"{label} failed locally and failure synchronization "
                    f"failed: {sync_error}") from local_error
            raise FailureSynchronizationError(
                f"{label} failure synchronization failed: {sync_error}") \
                from sync_error
        if int(failed.item()) != 0:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"{label} failed on the peer rank")
        return result

    def _make_buffer(self, allow_multiple_reduction):
        buffer = self.deep_ep.ElasticBuffer(
            self.group,
            num_bytes=2 * 1024 * 1024,
            num_gpu_timeout_secs=5,
            deterministic=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=allow_multiple_reduction,
            explicitly_destroy=True,
        )
        self.live_buffers.append(buffer)
        return buffer

    def _destroy_buffer(self, buffer):
        if buffer in self.live_buffers:
            try:
                buffer.destroy()
            finally:
                self.live_buffers.remove(buffer)
                if self.buffer is buffer:
                    self.buffer = None
                    self.buffer_mode = None

    def _replace_buffer(self, allow_multiple_reduction):
        if self.buffer is not None:
            self._destroy_buffer(self.buffer)
        self.buffer = self._make_buffer(allow_multiple_reduction)
        self.buffer_mode = allow_multiple_reduction

    def _ensure_buffer(self, allow_multiple_reduction):
        if self.buffer is None or self.buffer_mode != allow_multiple_reduction:
            self._replace_buffer(allow_multiple_reduction)
        return self.buffer

    def _tensor(self, rows, columns, dtype):
        return self.torch.tensor(
            rows, dtype=dtype, device=self.device).reshape(
                len(rows), columns).contiguous()

    def _materialize(self, spec, payload_delta=0, weight_scale=1.0):
        payloads = tuple(
            tuple(float(value + payload_delta) for value in row)
            for row in spec.payloads[self.rank])
        x = self._tensor(payloads, HIDDEN, self.torch.bfloat16)
        routes = self._tensor(
            spec.routes[self.rank], spec.num_topk, self.torch.int64)
        weights = None
        if spec.weights is not None:
            scaled = tuple(
                tuple(float(value * weight_scale) for value in row)
                for row in spec.weights[self.rank])
            weights = self._tensor(
                scaled, spec.num_topk, self.torch.float32)
        return x, routes, weights

    def _gather_routes(self, routes):
        count = routes.shape[0]
        num_topk = routes.shape[1]
        _check(0 <= count <= CAPACITY, "route count exceeds capacity")
        packed = self.torch.full(
            (1 + CAPACITY * num_topk,), -1,
            dtype=self.torch.int64, device=self.device)
        packed[0] = count
        if count:
            packed[1:1 + count * num_topk] = routes.reshape(-1)
        gathered = [self.torch.empty_like(packed) for _ in range(WORLD_SIZE)]
        self.dist.all_gather(gathered, packed, group=self.group)
        result = []
        for source_rank, values in enumerate(gathered):
            host = values.cpu()
            source_count = int(host[0].item())
            _check(0 <= source_count <= CAPACITY,
                   f"rank {source_rank} gathered invalid route count")
            result.append(host[1:1 + source_count * num_topk].reshape(
                source_count, num_topk).clone())
        return result

    def _dispatch(self, buffer, spec, x, routes, weights, handle=None):
        if handle is not None:
            return buffer.dispatch(
                x, topk_weights=weights, handle=handle,
                num_sms=1, num_qps=0,
                do_expand=spec.do_expand,
                do_zero_padding=spec.do_zero_padding)
        return buffer.dispatch(
            x,
            topk_idx=routes,
            topk_weights=weights,
            num_experts=NUM_EXPERTS,
            num_max_tokens_per_rank=CAPACITY,
            expert_alignment=spec.expert_alignment,
            num_sms=1,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
            do_expand=spec.do_expand,
            do_zero_padding=spec.do_zero_padding,
        )

    def _expert_outputs(self, recv_x, handle, gathered_routes, spec, variant):
        _validate_expansion_mode(handle.do_expand, spec.do_expand)
        metadata = handle.recv_src_metadata.detach().cpu()
        local_experts = NUM_EXPERTS // WORLD_SIZE
        first_expert = self.rank * local_experts
        last_expert = first_expert + local_experts
        if not handle.do_expand:
            rows = []
            for metadata_row in metadata:
                encoded_source = int(metadata_row[0].item())
                source_rank, source_token = divmod(encoded_source, CAPACITY)
                route = gathered_routes[source_rank][source_token]
                value = [0.0] * HIDDEN
                for lane, expert_tensor in enumerate(route):
                    expert = int(expert_tensor.item())
                    if first_expert <= expert < last_expert:
                        lane_value = _synthetic_transform(
                            source_rank, source_token, lane, variant)
                        for column in range(HIDDEN):
                            value[column] = _float32_add(
                                value[column], lane_value[column])
                rows.append(value)
            return self._tensor(rows, HIDDEN, self.torch.bfloat16)

        expanded = _expanded_reference_rows(
            recv_x.shape[0], metadata, gathered_routes, self.rank, variant)
        _check(not spec.require_padding or expanded["padding_rows"],
               f"{spec.name} did not produce an unmapped sentinel row")
        output = self._tensor(
            expanded["rows"], HIDDEN, self.torch.bfloat16)
        host = output.cpu()
        for row in expanded["padding_rows"]:
            _check(self.torch.equal(
                host[row],
                self.torch.full(
                    (HIDDEN,), PADDING_SENTINEL,
                    dtype=self.torch.bfloat16)),
                "expanded padding sentinel was overwritten")
        return output

    def _biases(self, spec, count):
        biases = []
        for bias_index in range(spec.bias_count):
            rows = [
                [float((bias_index + 1) * 2 + self.rank + column)
                 for column in range(HIDDEN)]
                for _ in range(count)]
            biases.append(self._tensor(rows, HIDDEN, self.torch.bfloat16))
        if not biases:
            return None
        return biases[0] if len(biases) == 1 else tuple(biases)

    def _expected(self, gathered_routes, variant, weights, biases):
        reference = _reference_matrix(
            gathered_routes[self.rank], self.rank, variant)
        values = reference["values"]
        if biases is not None:
            bias_tensors = (biases,) if isinstance(
                biases, self.torch.Tensor) else biases
            bias_values = tuple(
                bias.detach().cpu().float().tolist()
                for bias in bias_tensors)
            values = _apply_biases_once(values, bias_values)
        expected_x = self.torch.tensor(
            values, dtype=self.torch.float32).reshape(
                reference["shape"])
        expected_x = expected_x.to(self.torch.bfloat16)

        expected_weights = None
        if weights is not None:
            local_routes = gathered_routes[self.rank]
            masked = _masked_weight_values(
                local_routes, weights.detach().cpu().tolist())
            expected_weights = self.torch.tensor(
                masked, dtype=self.torch.float32).reshape(
                    local_routes.shape)
        return expected_x, expected_weights

    def _combine(self, buffer, expert_x, handle, expert_weights, biases):
        return buffer.combine(
            expert_x, handle, topk_weights=expert_weights,
            bias=biases, num_sms=1, num_qps=0)

    def _verify(self, result, expected_x, expected_weights):
        _check(isinstance(result, tuple) and len(result) == 3,
               "combine did not return the public three-field tuple")
        combined_x, combined_weights, event = result
        _check(_is_local_npu_tensor(combined_x, self.device.index),
               "combine activation is not on the local NPU")
        _check(combined_x.dtype == self.torch.bfloat16,
               "combine activation dtype is not BF16")
        _check(tuple(combined_x.shape) == tuple(expected_x.shape),
               "combine activation shape mismatch")
        self.torch.testing.assert_close(
            combined_x.detach().cpu().float(), expected_x.float(),
            rtol=BF16_TOLERANCE, atol=BF16_TOLERANCE)
        if expected_weights is None:
            _check(combined_weights is None,
                   "combine unexpectedly returned weights")
        else:
            _check(combined_weights is not None and
                   combined_weights.dtype == self.torch.float32,
                   "combine weights are missing or not float32")
            _check(_is_local_npu_tensor(
                combined_weights, self.device.index),
                "combine weights are not on the local NPU")
            _check(self.torch.equal(
                combined_weights.detach().cpu(), expected_weights),
                "combine did not restore the exact original weights")
        _check(event.event is None and event.extra_tensors is None and
               event.hook_after_wait is None,
               "synchronous combine returned deferred event state")
        return combined_x, combined_weights

    def _round_trip(self, spec, variant=None, handle=None, payload_delta=0,
                    weight_scale=1.0, buffer=None):
        variant = spec.transform_variant if variant is None else variant

        def prepare():
            selected_buffer = self._ensure_buffer(
                spec.allow_multiple_reduction) if buffer is None else buffer
            tensors = self._materialize(
                spec, payload_delta=payload_delta,
                weight_scale=weight_scale)
            return (selected_buffer,) + tensors

        selected_buffer, x, routes, weights = self._run_step(
            prepare, f"{spec.name}: prepare")
        gathered_routes = self._run_step(
            lambda: self._gather_routes(routes),
            f"{spec.name}: gather routes")
        dispatch_result = self._run_step(
            lambda: self._dispatch(
                selected_buffer, spec, x, routes, weights, handle=handle),
            f"{spec.name}: dispatch")
        recv_x, _, recv_weights, returned_handle, dispatch_event = \
            dispatch_result

        def prepare_combine():
            _check(dispatch_event.event is None,
                   "synchronous dispatch returned an event")
            _validate_expansion_mode(
                returned_handle.do_expand, spec.do_expand)
            if handle is not None:
                _check(returned_handle is handle,
                       "cached dispatch replaced its public handle")
            expert_x = self._expert_outputs(
                recv_x, returned_handle, gathered_routes, spec, variant)
            biases = self._biases(spec, routes.shape[0])
            expected = self._expected(
                gathered_routes, variant, weights, biases)
            return expert_x, biases, expected

        expert_x, biases, expected = self._run_step(
            prepare_combine, f"{spec.name}: prepare combine")
        result = self._run_step(
            lambda: self._combine(
                selected_buffer, expert_x, returned_handle,
                recv_weights, biases),
            f"{spec.name}: combine")
        combined_x, combined_weights = self._run_step(
            lambda: self._verify(result, *expected),
            f"{spec.name}: verify")
        return {
            "combined_x": combined_x,
            "combined_weights": combined_weights,
            "handle": returned_handle,
            "recv_x": recv_x,
            "expert_x": expert_x,
        }

    def _run_cached(self):
        spec = self.specs["cached-dispatch-changed-outputs"]
        first = self._round_trip(spec, variant=0)
        second = self._round_trip(
            spec, variant=8, handle=first["handle"],
            payload_delta=64, weight_scale=-1.0)
        _check(not self.torch.equal(
            first["recv_x"].cpu(), second["recv_x"].cpu()),
            "cached dispatch ignored changed input payloads")
        _check(not self.torch.equal(
            first["combined_x"].cpu(), second["combined_x"].cpu()),
            "cached round trip ignored changed expert outputs")
        _check(not self.torch.equal(
            first["combined_weights"].cpu(),
            second["combined_weights"].cpu()),
            "cached round trip ignored changed weights")

    def _run_generations(self):
        patterns = (
            (((0, 2),), ((1, 3),)),
            (((0, -1), (2, 3)), ((-1, -1),)),
            ((), ((0, 1), (3, -1))),
            (((-1, -1),), ((2, -1),)),
        )
        for generation in range(100):
            routes = patterns[generation % len(patterns)]
            spec = CaseSpec(
                "sequential-100-generations",
                _payloads((len(routes[0]), len(routes[1])), 40 + generation),
                routes)
            self._round_trip(spec, variant=generation % 16)

    def _expect_rejection(self, operation, expected):
        started = time.monotonic()
        try:
            operation()
        except RuntimeError as error:
            _check(expected in str(error),
                   f"rejection omitted {expected!r}: {error}")
            _check(time.monotonic() - started < 30,
                   "invalid handle path exceeded 30 seconds")
        else:
            raise AssertionError(f"invalid path was accepted ({expected})")

    def _run_cross_buffer(self):
        spec = self.specs["cross-buffer-handle"]
        source, x, routes, weights = self._run_step(
            lambda: (self._ensure_buffer(True),) + self._materialize(spec),
            "cross-buffer: prepare")
        gathered_routes = self._run_step(
            lambda: self._gather_routes(routes),
            "cross-buffer: gather routes")
        recv_x, _, recv_weights, handle, _ = self._run_step(
            lambda: self._dispatch(source, spec, x, routes, weights),
            "cross-buffer: dispatch")
        expert_x = self._run_step(
            lambda: self._expert_outputs(
                recv_x, handle, gathered_routes, spec, 0),
            "cross-buffer: expert outputs")
        target = self._run_step(
            lambda: self._make_buffer(True),
            "cross-buffer: target construction")
        try:
            self._run_step(lambda: self._expect_rejection(
                lambda: self._combine(
                    target, expert_x, handle, recv_weights, None),
                "dispatch handle"), "cross-buffer: reject foreign handle")
            self._round_trip(spec, variant=2, buffer=target)
        finally:
            if target in self.live_buffers:
                if self.collectives_usable:
                    self._run_step(
                        lambda: self._destroy_buffer(target),
                        "cross-buffer: target teardown")
                else:
                    self._destroy_buffer(target)

    def _run_malformed_handle(self):
        spec = self.specs["malformed-handle"]
        buffer, x, routes, weights = self._run_step(
            lambda: (self._ensure_buffer(True),) + self._materialize(spec),
            "malformed-handle: prepare")
        gathered_routes = self._run_step(
            lambda: self._gather_routes(routes),
            "malformed-handle: gather routes")
        recv_x, _, recv_weights, handle, _ = self._run_step(
            lambda: self._dispatch(buffer, spec, x, routes, weights),
            "malformed-handle: dispatch")
        expert_x = self._run_step(
            lambda: self._expert_outputs(
                recv_x, handle, gathered_routes, spec, 0),
            "malformed-handle: expert outputs")

        def mutate_handle():
            original = handle.token_metadata_at_forward
            malformed = original.clone()
            malformed[0] = malformed[0] ^ 1
            handle.token_metadata_at_forward = malformed
            return original

        original_descriptor = self._run_step(
            mutate_handle, "malformed-handle: mutate")
        try:
            self._run_step(lambda: self._expect_rejection(
                lambda: self._combine(
                    buffer, expert_x, handle, recv_weights, None),
                "dispatch handle"), "malformed-handle: reject")
        finally:
            handle.token_metadata_at_forward = original_descriptor
        self._run_step(
            lambda: None, "malformed-handle: restored")
        expected = self._run_step(
            lambda: self._expected(gathered_routes, 0, weights, None),
            "malformed-handle: expected")
        result = self._run_step(
            lambda: self._combine(
                buffer, expert_x, handle, recv_weights, None),
            "malformed-handle: retry combine")
        self._run_step(
            lambda: self._verify(result, *expected),
            "malformed-handle: verify retry")

    def _run_bounded_peer_diagnostics(self):
        spec = self.specs["bounded-peer-diagnostics"]
        buffer, x, routes, _ = self._run_step(
            lambda: (self._ensure_buffer(True),) + self._materialize(spec),
            "bounded-peer-diagnostics: prepare")
        gathered_routes = self._run_step(
            lambda: self._gather_routes(routes),
            "bounded-peer-diagnostics: gather routes")

        def expect_diagnostic():
            _check(all(int(rank_routes[0, 0].item()) == NUM_EXPERTS
                       for rank_routes in gathered_routes),
                   "invalid peer route changed during literal gather")
            started = time.monotonic()
            try:
                self._dispatch(buffer, spec, x, routes, None)
            except RuntimeError as error:
                message = str(error)
                for field in ("command_index", "opcode", "peer", "channel"):
                    _check(re.search(rf"{field}=[0-9]+", message) is not None,
                           f"peer diagnostic omitted numeric {field}: "
                           f"{message}")
                _check(f"dispatch failed on rank {self.rank}" in message,
                       f"peer diagnostic omitted local rank: {message}")
                _check(re.search(
                           r"generation=[1-9][0-9]*", message) is not None,
                       f"peer diagnostic has no positive generation: "
                       f"{message}")
                _check(time.monotonic() - started < 30,
                       "invalid peer path exceeded 30 seconds")
            else:
                raise AssertionError("invalid peer route was accepted")

        self._run_step(
            expect_diagnostic, "bounded-peer-diagnostics: invalid dispatch")

    def _run_repeated_teardown(self):
        spec = self.specs["repeated-teardown"]
        if self.buffer is not None:
            self._run_step(
                lambda: self._destroy_buffer(self.buffer),
                "repeated-teardown: initial teardown")
        for generation in range(3):
            temporary = self._run_step(
                lambda: self._make_buffer(True),
                f"repeated-teardown {generation}: construction")
            try:
                self._round_trip(
                    spec, variant=generation, buffer=temporary)
            finally:
                if temporary in self.live_buffers:
                    if self.collectives_usable:
                        self._run_step(
                            lambda: self._destroy_buffer(temporary),
                            f"repeated-teardown {generation}: teardown")
                    else:
                        self._destroy_buffer(temporary)

    def _run_named_case(self, name):
        if name in REGULAR_CASES:
            self._round_trip(self.specs[name])
        elif name == "cached-dispatch-changed-outputs":
            self._run_cached()
        elif name == "sequential-100-generations":
            self._run_generations()
        elif name == "cross-buffer-handle":
            self._run_cross_buffer()
        elif name == "malformed-handle":
            self._run_malformed_handle()
        elif name == "bounded-peer-diagnostics":
            self._run_bounded_peer_diagnostics()
        elif name == "repeated-teardown":
            self._run_repeated_teardown()
        else:
            raise AssertionError(f"unimplemented combine matrix case {name}")

    def _case_boundary(self, name):
        self.dist.barrier(self.group)
        local_error = None
        try:
            self._run_named_case(name)
        except BaseException as error:
            local_error = error
        if not self.collectives_usable:
            if local_error is not None:
                raise local_error
            raise FailureSynchronizationError(
                f"{name} cannot complete after synchronization failure")
        failed = self.torch.tensor(
            [int(local_error is not None)], dtype=self.torch.int32,
            device=self.device)
        self.dist.all_reduce(failed, group=self.group)
        self.dist.barrier(self.group)
        if int(failed.item()) != 0:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"{name} failed on the peer rank")
        if self.rank == 0:
            print(f"PASS {name}", flush=True)

    def run(self, selected_cases):
        for name in selected_cases:
            self._case_boundary(name)

    def destroy(self):
        failures = []
        for buffer in tuple(reversed(self.live_buffers)):
            try:
                self._destroy_buffer(buffer)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise CleanupFailures(failures)


def _parse_cases(value):
    if value is None:
        return CASE_NAMES
    selected = tuple(
        name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in selected if name not in CASE_NAMES]
    if unknown:
        raise ValueError(f"unknown combine matrix cases: {unknown}")
    if not selected:
        raise ValueError("at least one combine matrix case is required")
    if "bounded-peer-diagnostics" in selected and selected[-2:] != (
            "bounded-peer-diagnostics", "repeated-teardown"):
        raise ValueError(
            "bounded-peer-diagnostics must be followed by repeated-teardown")
    return selected


def _run_runtime(selected_cases):
    import torch
    import torch.distributed as dist
    import torch_npu

    import deep_ep

    del torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(
            f"combine matrix requires exactly two ranks, got {world_size}")
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    matrix = None
    runtime_error = None
    try:
        device = torch.device("npu", local_rank)
        matrix = CombineMatrix(torch, dist, deep_ep, group, device)
        matrix.run(selected_cases)
        if dist.get_rank(group) == 0:
            print(
                f"Phase 2G two-rank combine matrix passed "
                f"({len(selected_cases)} cases)", flush=True)
    except BaseException as error:
        runtime_error = error
    finally:
        cleanup_error = None
        try:
            _cleanup_runtime(matrix, dist.destroy_process_group)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--cases")
    args = parser.parse_args()
    if args.contract:
        print(json.dumps(_contract(), sort_keys=True))
        return 0
    selected_cases = _parse_cases(args.cases)
    _run_runtime(selected_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
