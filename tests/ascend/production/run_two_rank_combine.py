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
            "expanded-single-reduction", _payloads((2, 2), 5),
            (((0, 1), (2, -1)), ((3, 2), (0, 3))),
            do_expand=True, allow_multiple_reduction=False),
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
            do_expand=True, do_zero_padding=True, expert_alignment=4),
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
    base = 1 + origin_rank * 16 + origin_token * 4 + lane * 2 + variant
    return [float(base + column) for column in range(HIDDEN)]


def _float32_add(left, right):
    return struct.unpack("<f", struct.pack("<f", left + right))[0]


def _reference_values(routes, origin_rank, variant=0):
    local_experts = NUM_EXPERTS // WORLD_SIZE
    result = []
    for origin_token, token_routes in enumerate(routes):
        values = [0.0] * HIDDEN
        for contributor_rank in range(WORLD_SIZE):
            for lane, expert in enumerate(token_routes):
                if expert < 0 or expert // local_experts != contributor_rank:
                    continue
                transform = _synthetic_transform(
                    origin_rank, origin_token, lane, variant)
                for column in range(HIDDEN):
                    values[column] = _float32_add(
                        values[column], transform[column])
        result.append(values)
    return result


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
    _check("matrix.destroy" in runtime_calls and
           "dist.destroy_process_group" in runtime_calls,
           "buffer/group teardown is incomplete")
    runtime_try = next(
        node for node in runtime.body if isinstance(node, ast.Try))
    final_calls = [
        _attribute_name(node.value.func)
        for node in runtime_try.finalbody
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    guarded_calls = [
        _attribute_name(child.func)
        for node in runtime_try.finalbody if isinstance(node, ast.If)
        for child in ast.walk(node) if isinstance(child, ast.Call)]
    _check("matrix.destroy" in guarded_calls and
           final_calls == ["dist.destroy_process_group"],
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
        ],
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
            buffer.destroy()
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
            spec.routes[self.rank], NUM_TOPK, self.torch.int64)
        weights = None
        if spec.weights is not None:
            scaled = tuple(
                tuple(float(value * weight_scale) for value in row)
                for row in spec.weights[self.rank])
            weights = self._tensor(scaled, NUM_TOPK, self.torch.float32)
        return x, routes, weights

    def _gather_routes(self, routes):
        count = routes.shape[0]
        _check(0 <= count <= CAPACITY, "route count exceeds capacity")
        packed = self.torch.full(
            (1 + CAPACITY * NUM_TOPK,), -1,
            dtype=self.torch.int64, device=self.device)
        packed[0] = count
        if count:
            packed[1:1 + count * NUM_TOPK] = routes.reshape(-1)
        gathered = [self.torch.empty_like(packed) for _ in range(WORLD_SIZE)]
        self.dist.all_gather(gathered, packed, group=self.group)
        result = []
        for source_rank, values in enumerate(gathered):
            host = values.cpu()
            source_count = int(host[0].item())
            _check(0 <= source_count <= CAPACITY,
                   f"rank {source_rank} gathered invalid route count")
            result.append(host[1:1 + source_count * NUM_TOPK].reshape(
                source_count, NUM_TOPK).clone())
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

    def _expert_outputs(self, recv_x, handle, gathered_routes, variant):
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
                            value[column] += lane_value[column]
                rows.append(value)
            return self._tensor(rows, HIDDEN, self.torch.bfloat16)

        rows = [[PADDING_SENTINEL] * HIDDEN for _ in range(recv_x.shape[0])]
        written = set()
        for metadata_row in metadata:
            encoded_source = int(metadata_row[0].item())
            source_rank, source_token = divmod(encoded_source, CAPACITY)
            for lane in range(NUM_TOPK):
                destination = int(metadata_row[2 + lane].item())
                if destination < 0:
                    continue
                _check(destination not in written,
                       "expanded metadata names one row more than once")
                written.add(destination)
                rows[destination] = _synthetic_transform(
                    source_rank, source_token, lane, variant)
        output = self._tensor(rows, HIDDEN, self.torch.bfloat16)
        host = output.cpu()
        for row in range(len(rows)):
            if row not in written:
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
        expected_x = self.torch.tensor(
            reference["values"], dtype=self.torch.float32).reshape(
                reference["shape"])
        if biases is not None:
            bias_values = (biases,) if isinstance(
                biases, self.torch.Tensor) else biases
            for bias in bias_values:
                expected_x += bias.detach().cpu().float()
        expected_x = expected_x.to(self.torch.bfloat16)

        expected_weights = None
        if weights is not None:
            expected_weights = self.torch.zeros_like(weights, device="cpu")
            local_routes = gathered_routes[self.rank]
            for token in range(local_routes.shape[0]):
                for lane in range(NUM_TOPK):
                    if int(local_routes[token, lane].item()) >= 0:
                        expected_weights[token, lane] = weights[
                            token, lane].detach().cpu()
        return expected_x, expected_weights

    def _combine(self, buffer, expert_x, handle, expert_weights, biases):
        return buffer.combine(
            expert_x, handle, topk_weights=expert_weights,
            bias=biases, num_sms=1, num_qps=0)

    def _verify(self, result, expected_x, expected_weights):
        _check(isinstance(result, tuple) and len(result) == 3,
               "combine did not return the public three-field tuple")
        combined_x, combined_weights, event = result
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
            _check(self.torch.equal(
                combined_weights.detach().cpu(), expected_weights),
                "combine did not restore the exact original weights")
        _check(event.event is None and event.extra_tensors is None and
               event.hook_after_wait is None,
               "synchronous combine returned deferred event state")
        return combined_x, combined_weights

    def _round_trip(self, spec, variant=0, handle=None, payload_delta=0,
                    weight_scale=1.0, buffer=None):
        buffer = self._ensure_buffer(
            spec.allow_multiple_reduction) if buffer is None else buffer
        x, routes, weights = self._materialize(
            spec, payload_delta=payload_delta, weight_scale=weight_scale)
        gathered_routes = self._gather_routes(routes)
        dispatch_result = self._dispatch(
            buffer, spec, x, routes, weights, handle=handle)
        recv_x, _, recv_weights, returned_handle, dispatch_event = \
            dispatch_result
        _check(dispatch_event.event is None,
               "synchronous dispatch returned an event")
        if handle is not None:
            _check(returned_handle is handle,
                   "cached dispatch replaced its public handle")
        expert_x = self._expert_outputs(
            recv_x, returned_handle, gathered_routes, variant)
        biases = self._biases(spec, routes.shape[0])
        expected_x, expected_weights = self._expected(
            gathered_routes, variant, weights, biases)
        result = self._combine(
            buffer, expert_x, returned_handle, recv_weights, biases)
        combined_x, combined_weights = self._verify(
            result, expected_x, expected_weights)
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
        source = self._ensure_buffer(True)
        x, routes, weights = self._materialize(spec)
        gathered_routes = self._gather_routes(routes)
        recv_x, _, recv_weights, handle, _ = self._dispatch(
            source, spec, x, routes, weights)
        expert_x = self._expert_outputs(recv_x, handle, gathered_routes, 0)
        target = self._make_buffer(True)
        try:
            self._expect_rejection(
                lambda: self._combine(
                    target, expert_x, handle, recv_weights, None),
                "dispatch handle")
            self._round_trip(spec, variant=2, buffer=target)
        finally:
            self._destroy_buffer(target)

    def _run_malformed_handle(self):
        spec = self.specs["malformed-handle"]
        buffer = self._ensure_buffer(True)
        x, routes, weights = self._materialize(spec)
        gathered_routes = self._gather_routes(routes)
        recv_x, _, recv_weights, handle, _ = self._dispatch(
            buffer, spec, x, routes, weights)
        expert_x = self._expert_outputs(recv_x, handle, gathered_routes, 0)
        original_descriptor = handle.token_metadata_at_forward
        malformed_descriptor = original_descriptor.clone()
        malformed_descriptor[0] = malformed_descriptor[0] ^ 1
        handle.token_metadata_at_forward = malformed_descriptor
        try:
            self._expect_rejection(
                lambda: self._combine(
                    buffer, expert_x, handle, recv_weights, None),
                "dispatch handle")
        finally:
            handle.token_metadata_at_forward = original_descriptor
        expected_x, expected_weights = self._expected(
            gathered_routes, 0, weights, None)
        self._verify(
            self._combine(buffer, expert_x, handle, recv_weights, None),
            expected_x, expected_weights)

    def _run_bounded_peer_diagnostics(self):
        spec = self.specs["bounded-peer-diagnostics"]
        buffer = self._ensure_buffer(True)
        x, routes, _ = self._materialize(spec)
        gathered_routes = self._gather_routes(routes)
        _check(all(int(rank_routes[0, 0].item()) == NUM_EXPERTS
                   for rank_routes in gathered_routes),
               "invalid peer route changed during literal gather")
        started = time.monotonic()
        try:
            self._dispatch(buffer, spec, x, routes, None)
        except RuntimeError as error:
            message = str(error)
            for marker in (
                    f"dispatch failed on rank {self.rank}",
                    "command_index=", "opcode=", "peer=", "channel=",
                    "generation="):
                _check(marker in message,
                       f"peer diagnostic omitted {marker!r}: {message}")
            _check(re.search(r"generation=[1-9][0-9]*", message) is not None,
                   f"peer diagnostic has no positive generation: {message}")
            _check(time.monotonic() - started < 30,
                   "invalid peer path exceeded 30 seconds")
        else:
            raise AssertionError("invalid peer route was accepted")

    def _run_repeated_teardown(self):
        spec = self.specs["repeated-teardown"]
        if self.buffer is not None:
            self._destroy_buffer(self.buffer)
        for generation in range(3):
            temporary = self._make_buffer(True)
            try:
                self._round_trip(
                    spec, variant=generation, buffer=temporary)
            finally:
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
        for buffer in tuple(reversed(self.live_buffers)):
            self._destroy_buffer(buffer)


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
    try:
        device = torch.device("npu", local_rank)
        matrix = CombineMatrix(torch, dist, deep_ep, group, device)
        matrix.run(selected_cases)
        if dist.get_rank(group) == 0:
            print(
                f"Phase 2G two-rank combine matrix passed "
                f"({len(selected_cases)} cases)", flush=True)
    finally:
        if matrix is not None:
            matrix.destroy()
        dist.destroy_process_group()


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
