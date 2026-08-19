import argparse
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta


WORLD_SIZE = 2
HIDDEN = 4
NUM_EXPERTS = 4
NUM_TOPK = 2
CAPACITY = 4

CASE_NAMES = (
    "asymmetric-routing",
    "empty-input",
    "negative-one-route",
    "duplicate-destination-rank",
    "multiple-experts",
    "optional-weights",
    "expanded",
    "aligned-zero-padding",
    "aligned-near-capacity",
    "cached-reuse",
    "cached-aligned-near-capacity",
    "sequential-100-generations",
    "round-trip-smoke",
    "invalid-expert-diagnostics",
)

REGULAR_CASES = CASE_NAMES[:9]
SPECIAL_CASES = CASE_NAMES[9:]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    payloads: tuple
    routes: tuple
    weights: object = None
    num_topk: int = NUM_TOPK
    expert_alignment: int = 1
    do_expand: bool = False
    do_zero_padding: bool = False


def _case_specs():
    return {
        "asymmetric-routing": CaseSpec(
            "asymmetric-routing",
            (
                ((10, 11, 12, 13), (20, 21, 22, 23), (30, 31, 32, 33)),
                ((110, 111, 112, 113),),
            ),
            (
                ((2, -1), (0, 3), (-1, -1)),
                ((1, -1),),
            ),
        ),
        "empty-input": CaseSpec(
            "empty-input", ((), ()), ((), ())),
        "negative-one-route": CaseSpec(
            "negative-one-route",
            (
                ((14, 15, 16, 17), (24, 25, 26, 27)),
                ((114, 115, 116, 117), (124, 125, 126, 127)),
            ),
            (
                ((-1, -1), (2, -1)),
                ((-1, 0), (3, -1)),
            ),
        ),
        "duplicate-destination-rank": CaseSpec(
            "duplicate-destination-rank",
            (
                ((15, 16, 17, 18), (25, 26, 27, 28)),
                ((115, 116, 117, 118), (125, 126, 127, 128)),
            ),
            (
                ((0, 1), (2, 3)),
                ((0, 1), (2, 3)),
            ),
        ),
        "multiple-experts": CaseSpec(
            "multiple-experts",
            (
                ((16, 17, 18, 19), (26, 27, 28, 29)),
                ((116, 117, 118, 119), (126, 127, 128, 129)),
            ),
            (
                ((0, 2), (1, 3)),
                ((3, 1), (2, 0)),
            ),
        ),
        "optional-weights": CaseSpec(
            "optional-weights",
            (
                ((17, 18, 19, 20), (27, 28, 29, 30)),
                ((117, 118, 119, 120),),
            ),
            (
                ((2, 0), (1, -1)),
                ((3, 1),),
            ),
            (
                ((0.125, 0.25), (0.375, 0.5)),
                ((0.625, 0.75),),
            ),
        ),
        "expanded": CaseSpec(
            "expanded",
            (
                ((18, 19, 20, 21), (28, 29, 30, 31)),
                ((118, 119, 120, 121), (128, 129, 130, 131)),
            ),
            (
                ((0, 1), (2, 0)),
                ((3, 2), (1, 3)),
            ),
            (
                ((0.1, 0.2), (0.3, 0.4)),
                ((0.5, 0.6), (0.7, 0.8)),
            ),
            do_expand=True,
        ),
        "aligned-zero-padding": CaseSpec(
            "aligned-zero-padding",
            (
                ((19, 20, 21, 22), (29, 30, 31, 32)),
                ((119, 120, 121, 122),),
            ),
            (
                ((0, 2), (0, -1)),
                ((1, 3),),
            ),
            (
                ((0.15, 0.25), (0.35, 0.45)),
                ((0.55, 0.65),),
            ),
            expert_alignment=4,
            do_expand=True,
            do_zero_padding=True,
        ),
        "aligned-near-capacity": CaseSpec(
            "aligned-near-capacity",
            (
                ((21, 22, 23, 24), (31, 32, 33, 34),
                 (41, 42, 43, 44), (51, 52, 53, 54)),
                ((121, 122, 123, 124), (131, 132, 133, 134),
                 (141, 142, 143, 144), (151, 152, 153, 154)),
            ),
            (
                ((0,), (0,), (0,), (0,)),
                ((0,), (1,), (2,), (3,)),
            ),
            (
                ((0.11,), (0.21,), (0.31,), (0.41,)),
                ((0.51,), (0.61,), (0.71,), (0.81,)),
            ),
            num_topk=1,
            expert_alignment=4,
            do_expand=True,
            do_zero_padding=True,
        ),
        "cached-reuse": CaseSpec(
            "cached-reuse",
            (
                ((40, 41, 42, 43), (50, 51, 52, 53)),
                ((140, 141, 142, 143), (150, 151, 152, 153)),
            ),
            (
                ((2, 0), (3, -1)),
                ((1, 2), (0, -1)),
            ),
            (
                ((0.11, 0.21), (0.31, 0.41)),
                ((0.51, 0.61), (0.71, 0.81)),
            ),
        ),
        "cached-aligned-near-capacity": CaseSpec(
            "cached-aligned-near-capacity",
            (
                ((61, 62, 63, 64), (71, 72, 73, 74),
                 (81, 82, 83, 84), (91, 92, 93, 94)),
                ((161, 162, 163, 164), (171, 172, 173, 174),
                 (181, 182, 183, 184), (191, 192, 193, 194)),
            ),
            (
                ((0,), (0,), (0,), (0,)),
                ((0,), (1,), (2,), (3,)),
            ),
            (
                ((-0.11,), (-0.21,), (-0.31,), (-0.41,)),
                ((-0.51,), (-0.61,), (-0.71,), (-0.81,)),
            ),
            num_topk=1,
            expert_alignment=4,
            do_expand=True,
            do_zero_padding=True,
        ),
        "invalid-expert-diagnostics": CaseSpec(
            "invalid-expert-diagnostics",
            (
                ((60, 61, 62, 63),),
                ((160, 161, 162, 163),),
            ),
            (
                ((4, -1),),
                ((4, -1),),
            ),
        ),
    }


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _contract_methods():
    with open(__file__, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=__file__)
    matrix = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DispatchMatrix")
    return {
        node.name: node for node in matrix.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _method_calls(method):
    return [
        _attribute_name(node.func) for node in ast.walk(method)
        if isinstance(node, ast.Call)
    ]


def _contract():
    specs = _case_specs()
    required_specs = set(REGULAR_CASES) | {
        "cached-reuse", "cached-aligned-near-capacity",
        "invalid-expert-diagnostics"}
    _check(set(specs) == required_specs,
           "dispatch case specifications do not match the matrix")
    _check(SPECIAL_CASES == (
        "cached-reuse", "cached-aligned-near-capacity",
        "sequential-100-generations", "round-trip-smoke",
        "invalid-expert-diagnostics"),
        "dispatch special-case registry is incomplete")
    methods = _contract_methods()
    gather_calls = _method_calls(methods["_gather_literals"])
    dispatch_calls = _method_calls(methods["_dispatch_spec"])
    cached_calls = _method_calls(methods["_run_cached"])
    boundary_calls = _method_calls(methods["_case_boundary"])
    run_calls = _method_calls(methods["run"])
    _check(gather_calls.count("self.dist.all_gather") >= 2,
           "literal gather coverage is missing")
    for call in ("self._gather_literals", "self._reference",
                 "self.buffer.dispatch", "self._verify_result"):
        _check(call in dispatch_calls,
               f"dispatch/reference path is missing {call}")
    for call in ("self._dispatch_spec", "self._gather_literals",
                 "self._reference", "self.buffer.dispatch"):
        _check(call in cached_calls, f"cached path is missing {call}")
    cached_dispatches = [
        node for node in ast.walk(methods["_run_cached"])
        if isinstance(node, ast.Call) and
        _attribute_name(node.func) == "self.buffer.dispatch"]
    _check(any(any(keyword.arg == "handle" for keyword in call.keywords)
               for call in cached_dispatches),
           "cached dispatch does not reuse a handle")
    _check(boundary_calls.count("self.dist.barrier") >= 2,
           "rank case barriers are missing")
    _check("self.dist.all_reduce" in boundary_calls,
           "distributed failure reduction is missing")
    _check("self.buffer.destroy" in run_calls and any(
               isinstance(node, ast.Try) and node.finalbody
               for node in ast.walk(methods["run"])),
           "buffer cleanup is not protected by finally")
    return {
        "case_names": list(CASE_NAMES),
        "contract_checks": [
            "literal-gather",
            "independent-reference",
            "cached-handle-reuse",
            "rank-barriers",
            "distributed-failure-reduction",
            "buffer-cleanup",
        ],
        "dispatch_surface": "Buffer.dispatch",
        "expected_world_size": WORLD_SIZE,
        "reference": "gathered-literal-inputs",
    }


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


class DispatchMatrix:
    def __init__(self, torch, dist, deep_ep, group, device):
        self.torch = torch
        self.dist = dist
        self.deep_ep = deep_ep
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.specs = _case_specs()
        self.buffer = None
        self.combine_payload = None

    def make_buffer(self):
        Buffer = self.deep_ep.ElasticBuffer
        self.buffer = Buffer(
            self.group,
            num_bytes=2 * 1024 * 1024,
            num_gpu_timeout_secs=5,
            deterministic=False,
            allow_hybrid_mode=False,
            explicitly_destroy=True,
        )
        _check(self.buffer.get_logical_domain_size() == (1, 2),
               "unexpected logical domain")
        _check(self.buffer.get_physical_domain_size() == (1, 2),
               "unexpected physical domain")
        _check((self.buffer.scaleout_rank_idx,
                self.buffer.scaleup_rank_idx) == (0, self.rank),
               "unexpected rank mapping")

    def _tensor(self, rows, columns, dtype):
        return self.torch.tensor(
            rows, dtype=dtype, device=self.device).reshape(
                len(rows), columns).contiguous()

    def _materialize(self, spec):
        x = self._tensor(spec.payloads[self.rank], HIDDEN,
                         self.torch.bfloat16)
        routes = self._tensor(spec.routes[self.rank], spec.num_topk,
                              self.torch.int64)
        weights = None
        if spec.weights is not None:
            weights = self._tensor(spec.weights[self.rank], spec.num_topk,
                                   self.torch.float32)
        return x, routes, weights

    def _gather_literals(self, x, routes, weights):
        count = x.shape[0]
        _check(0 <= count <= CAPACITY, "literal token count exceeds capacity")
        num_topk = routes.shape[1]
        metadata = self.torch.full(
            (2 + CAPACITY * num_topk,), -1,
            dtype=self.torch.int64, device=self.device)
        metadata[0] = count
        metadata[1] = int(weights is not None)
        if count:
            metadata[2:2 + count * num_topk] = routes.reshape(-1)

        padded_x = self.torch.zeros(
            (CAPACITY, HIDDEN), dtype=self.torch.bfloat16,
            device=self.device)
        if count:
            padded_x[:count].copy_(x)

        gathered_metadata = [self.torch.empty_like(metadata)
                             for _ in range(WORLD_SIZE)]
        gathered_x = [self.torch.empty_like(padded_x)
                      for _ in range(WORLD_SIZE)]
        self.dist.all_gather(gathered_metadata, metadata, group=self.group)
        self.dist.all_gather(gathered_x, padded_x, group=self.group)

        gathered_weights = None
        if weights is not None:
            padded_weights = self.torch.zeros(
                (CAPACITY, num_topk), dtype=self.torch.float32,
                device=self.device)
            if count:
                padded_weights[:count].copy_(weights)
            gathered_weights = [self.torch.empty_like(padded_weights)
                                for _ in range(WORLD_SIZE)]
            self.dist.all_gather(
                gathered_weights, padded_weights, group=self.group)

        result = {"counts": [], "x": [], "routes": [], "weights": None}
        if weights is not None:
            result["weights"] = []
        for source_rank in range(WORLD_SIZE):
            host_metadata = gathered_metadata[source_rank].cpu()
            source_count = int(host_metadata[0].item())
            _check(0 <= source_count <= CAPACITY,
                   f"rank {source_rank} gathered invalid token count")
            _check(int(host_metadata[1].item()) == int(weights is not None),
                   f"rank {source_rank} disagrees on optional weights")
            result["counts"].append(source_count)
            result["x"].append(
                gathered_x[source_rank][:source_count].cpu().clone())
            route_values = host_metadata[
                2:2 + source_count * num_topk].reshape(
                    source_count, num_topk).clone()
            result["routes"].append(route_values)
            if weights is not None:
                result["weights"].append(
                    gathered_weights[source_rank][
                        :source_count].cpu().clone())
        return result

    @staticmethod
    def _aligned(value, alignment):
        return ((value + alignment - 1) // alignment) * alignment

    def _reference(self, gathered, spec):
        local_experts = NUM_EXPERTS // WORLD_SIZE
        first_local_expert = self.rank * local_experts
        last_local_expert = first_local_expert + local_experts

        incoming = []
        source_counts = []
        for source_rank in range(WORLD_SIZE):
            source_count = 0
            for token in range(gathered["counts"][source_rank]):
                route = gathered["routes"][source_rank][token]
                destinations = []
                for lane in range(spec.num_topk):
                    expert = int(route[lane].item())
                    _check(expert == -1 or 0 <= expert < NUM_EXPERTS,
                           f"reference received invalid expert {expert}")
                    if expert >= 0:
                        destination = expert // local_experts
                        if destination not in destinations:
                            destinations.append(destination)
                if self.rank not in destinations:
                    continue
                master_lane = next(
                    lane for lane in range(spec.num_topk)
                    if first_local_expert <= int(route[lane].item()) <
                    last_local_expert)
                incoming.append({
                    "source_rank": source_rank,
                    "source_token": token,
                    "master_lane": master_lane,
                    "x": gathered["x"][source_rank][token],
                    "route": route,
                    "weights": None if gathered["weights"] is None else
                        gathered["weights"][source_rank][token],
                })
                source_count += 1
            source_counts.append(source_count)

        rank_prefix = []
        running = 0
        for count in source_counts:
            running += count
            rank_prefix.append(running)

        expert_counts = [0] * NUM_EXPERTS
        for record in incoming:
            for expert_tensor in record["route"]:
                expert = int(expert_tensor.item())
                if first_local_expert <= expert < last_local_expert:
                    expert_counts[expert] += 1

        expanded_starts = []
        public_expert_prefix = []
        running = 0
        local_actual_counts = expert_counts[
            first_local_expert:last_local_expert]
        local_padded_counts = []
        for actual_count in local_actual_counts:
            expanded_starts.append(running)
            padded_count = self._aligned(
                actual_count, spec.expert_alignment)
            local_padded_counts.append(padded_count)
            public_expert_prefix.append(
                running + actual_count if spec.do_expand else
                running + padded_count)
            running += padded_count

        local_routes = gathered["routes"][self.rank]
        destination_slots = self.torch.full(
            local_routes.shape, -1, dtype=self.torch.int32)
        next_slots = [0] * WORLD_SIZE
        for token in range(gathered["counts"][self.rank]):
            route = local_routes[token]
            destinations = []
            for lane in range(spec.num_topk):
                expert = int(route[lane].item())
                if expert >= 0:
                    destination = expert // local_experts
                    if destination not in destinations:
                        destinations.append(destination)
            for destination in destinations:
                slot = next_slots[destination]
                next_slots[destination] += 1
                for lane in range(spec.num_topk):
                    expert = int(route[lane].item())
                    if expert >= 0 and expert // local_experts == destination:
                        destination_slots[token, lane] = \
                            self.rank * CAPACITY + slot

        source_metadata = self.torch.full(
            (len(incoming), spec.num_topk + 2), -1,
            dtype=self.torch.int32)
        for record_index, record in enumerate(incoming):
            source_metadata[record_index, 0] = \
                record["source_rank"] * CAPACITY + record["source_token"]
            source_metadata[record_index, 1] = \
                record["source_rank"] * spec.num_topk + record["master_lane"]

        if not spec.do_expand:
            recv_x = self.torch.stack(
                [record["x"] for record in incoming]) if incoming else \
                self.torch.empty((0, HIDDEN), dtype=self.torch.bfloat16)
            if incoming:
                recv_topk_idx = self.torch.stack(
                    [record["route"] for record in incoming])
                local_mask = (recv_topk_idx >= first_local_expert) & \
                    (recv_topk_idx < last_local_expert)
                recv_topk_idx = self.torch.where(
                    local_mask, recv_topk_idx - first_local_expert,
                    self.torch.full_like(recv_topk_idx, -1))
            else:
                recv_topk_idx = self.torch.empty(
                    (0, spec.num_topk), dtype=self.torch.int64)
            recv_weights = None
            if gathered["weights"] is not None:
                recv_weights = self.torch.stack(
                    [record["weights"] for record in incoming]) \
                    if incoming else self.torch.empty(
                        (0, spec.num_topk), dtype=self.torch.float32)
        else:
            recv_x = self.torch.zeros(
                (running, HIDDEN), dtype=self.torch.bfloat16)
            recv_topk_idx = None
            recv_weights = None if gathered["weights"] is None else \
                self.torch.zeros((running,), dtype=self.torch.float32)
            occurrences = [0] * NUM_EXPERTS
            for record_index, record in enumerate(incoming):
                for lane in range(spec.num_topk):
                    expert = int(record["route"][lane].item())
                    if not first_local_expert <= expert < last_local_expert:
                        continue
                    destination = expanded_starts[
                        expert - first_local_expert] + occurrences[expert]
                    occurrences[expert] += 1
                    source_metadata[record_index, 2 + lane] = destination
                    recv_x[destination].copy_(record["x"])
                    if recv_weights is not None:
                        recv_weights[destination] = record["weights"][lane]
            if not spec.do_zero_padding:
                _check(spec.expert_alignment == 1,
                       "non-zero-padded reference must not expose padding")

        return {
            "recv_x": recv_x,
            "recv_topk_idx": recv_topk_idx,
            "recv_weights": recv_weights,
            "rank_prefix": self.torch.tensor(
                rank_prefix, dtype=self.torch.int32),
            "expert_prefix": self.torch.tensor(
                public_expert_prefix, dtype=self.torch.int32),
            "unaligned": self.torch.tensor(
                local_actual_counts, dtype=self.torch.int32),
            "per_expert": local_padded_counts,
            "destination_slots": destination_slots,
            "source_metadata": source_metadata,
            "num_recv_tokens": len(incoming),
            "num_expanded_tokens": running,
            "local_routes": local_routes,
        }

    def _assert_tensor(self, actual, expected, label):
        _check(actual is not None, f"{label}: tensor is None")
        _check(actual.device.type == "npu", f"{label}: not an NPU tensor")
        _check(actual.device.index == self.device.index,
               f"{label}: wrong NPU {actual.device}")
        _check(actual.dtype == expected.dtype,
               f"{label}: dtype {actual.dtype} != {expected.dtype}")
        _check(tuple(actual.shape) == tuple(expected.shape),
               f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        observed = actual.detach().cpu()
        _check(self.torch.equal(observed, expected),
               f"{label}: observed {observed.tolist()} != {expected.tolist()}")

    def _verify_handle(self, handle, spec, reference, local_routes):
        expected_fields = {
            "do_expand", "num_experts", "expert_alignment",
            "num_max_tokens_per_rank", "num_sms", "topk_idx",
            "psum_num_recv_tokens_per_scaleup_rank",
            "psum_num_recv_tokens_per_expert",
            "num_unaligned_recv_tokens_per_expert",
            "num_recv_tokens_per_expert_list", "recv_src_metadata",
            "dst_buffer_slot_idx", "token_metadata_at_forward",
            "channel_linked_list", "num_recv_tokens",
            "num_expanded_tokens", "cached_recv_src_metadata_before_sort",
            "_ascend_owner", "_ascend_generation",
            "_ascend_descriptor_fingerprint",
        }
        _check(set(vars(handle)) == expected_fields,
               f"EPHandle fields differ: {sorted(vars(handle))}")
        _check(handle._ascend_owner is self.buffer,
               "handle Ascend owner differs")
        _check(handle._ascend_generation > 0 and
               handle._ascend_generation ==
               self.buffer._ascend_handle_generation,
               "handle Ascend generation differs")
        _check(handle.do_expand is spec.do_expand, "handle.do_expand")
        _check(handle.num_experts == NUM_EXPERTS, "handle.num_experts")
        _check(handle.expert_alignment == spec.expert_alignment,
               "handle.expert_alignment")
        _check(handle.num_max_tokens_per_rank == CAPACITY,
               "handle.num_max_tokens_per_rank")
        _check(handle.num_sms == 1, "handle.num_sms")
        _check(handle.num_recv_tokens == reference["num_recv_tokens"],
               "handle.num_recv_tokens")
        _check(handle.num_expanded_tokens == reference["num_expanded_tokens"],
               "handle.num_expanded_tokens")
        _check(handle.num_recv_tokens_per_expert_list ==
               reference["per_expert"],
               "handle.num_recv_tokens_per_expert_list")
        _check(handle.channel_linked_list is None,
               "handle.channel_linked_list")
        _check(handle.cached_recv_src_metadata_before_sort is None,
               "handle.cached_recv_src_metadata_before_sort")

        self._assert_tensor(
            handle.topk_idx, reference["local_routes"], "handle.topk_idx")
        _check(handle.topk_idx is not local_routes,
               "handle.topk_idx was not copied")
        if handle.topk_idx.numel() > 0:
            _check(handle.topk_idx.data_ptr() != local_routes.data_ptr(),
                   "handle.topk_idx shares nonempty storage")
        self._assert_tensor(
            handle.psum_num_recv_tokens_per_scaleup_rank,
            reference["rank_prefix"], "handle.rank_prefix")
        self._assert_tensor(
            handle.psum_num_recv_tokens_per_expert,
            reference["expert_prefix"], "handle.expert_prefix")
        self._assert_tensor(
            handle.num_unaligned_recv_tokens_per_expert,
            reference["unaligned"], "handle.unaligned")
        self._assert_tensor(
            handle.recv_src_metadata, reference["source_metadata"],
            "handle.recv_src_metadata")
        self._assert_tensor(
            handle.dst_buffer_slot_idx, reference["destination_slots"],
            "handle.dst_buffer_slot_idx")
        descriptor = handle.token_metadata_at_forward
        _check(descriptor is not None and descriptor.device.type == "npu" and
               descriptor.device.index == self.device.index and
               descriptor.dtype == self.torch.uint8 and descriptor.dim() == 1 and
               descriptor.numel() > 0 and descriptor.is_contiguous(),
               "handle dispatch attestation is not an opaque NPU byte tensor")
        observed_fingerprint = tuple(
            int(value) for value in descriptor.detach().cpu().reshape(-1).tolist())
        _check(handle._ascend_descriptor_fingerprint == observed_fingerprint,
               "handle dispatch attestation fingerprint differs")

    def _verify_result(self, spec, local_routes, reference, result,
                       expected_handle=None):
        _check(isinstance(result, tuple) and len(result) == 5,
               "dispatch did not return the public five-field tuple")
        recv_x, recv_topk_idx, recv_weights, handle, event = result
        self._assert_tensor(recv_x, reference["recv_x"], "recv_x")
        if reference["recv_topk_idx"] is None:
            _check(recv_topk_idx is None, "expanded recv_topk_idx is not None")
        else:
            self._assert_tensor(
                recv_topk_idx, reference["recv_topk_idx"], "recv_topk_idx")
        if reference["recv_weights"] is None:
            _check(recv_weights is None, "recv_topk_weights is not None")
        else:
            self._assert_tensor(
                recv_weights, reference["recv_weights"],
                "recv_topk_weights")
        _check(isinstance(handle, self.deep_ep.EPHandle),
               "dispatch returned the wrong handle type")
        if expected_handle is not None:
            _check(handle is expected_handle,
                   "cached dispatch replaced its EPHandle")
        self._verify_handle(handle, spec, reference, local_routes)
        _check(event.event is None, "synchronous dispatch returned an event")
        _check(event.extra_tensors is None, "dispatch event retained tensors")
        _check(event.hook_after_wait is None,
               "dispatch event retained a deferred hook")
        return recv_x, recv_weights, handle

    def _dispatch_spec(self, spec):
        x, routes, weights = self._materialize(spec)
        gathered = self._gather_literals(x, routes, weights)
        reference = self._reference(gathered, spec)
        result = self.buffer.dispatch(
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
        recv_x, _, handle = self._verify_result(
            spec, routes, reference, result)
        self.combine_payload = (recv_x, handle)
        return result, reference

    def _run_cached(self, case_name="cached-reuse"):
        spec = self.specs[case_name]
        first_result, _ = self._dispatch_spec(spec)
        first_x, _, first_weights, handle, _ = first_result
        changed_payloads = tuple(
            tuple(tuple(value + 300 for value in row) for row in rank_rows)
            for rank_rows in spec.payloads)
        changed_weights = None if spec.weights is None else tuple(
            tuple(tuple(-value for value in row) for row in rank_rows)
            for rank_rows in spec.weights)
        changed_spec = CaseSpec(
            spec.name, changed_payloads, spec.routes, changed_weights,
            num_topk=spec.num_topk,
            expert_alignment=spec.expert_alignment,
            do_expand=spec.do_expand,
            do_zero_padding=spec.do_zero_padding,
        )
        x, routes, weights = self._materialize(changed_spec)
        gathered = self._gather_literals(x, routes, weights)
        reference = self._reference(gathered, changed_spec)
        cached_result = self.buffer.dispatch(
            x,
            topk_weights=weights,
            handle=handle,
            num_sms=1,
            num_qps=0,
            do_expand=spec.do_expand,
            do_zero_padding=spec.do_zero_padding,
        )
        recv_x, recv_weights, cached_handle = self._verify_result(
            changed_spec, routes, reference, cached_result,
            expected_handle=handle)
        _check(not self.torch.equal(first_x.cpu(), recv_x.cpu()),
               "cached dispatch ignored the changed BF16 payload")
        _check(not self.torch.equal(first_weights.cpu(), recv_weights.cpu()),
               "cached dispatch ignored the changed weights")
        self.combine_payload = (recv_x, cached_handle)

    def _run_generations(self):
        for generation in range(100):
            base = generation + 1
            pattern = generation % 4
            if pattern == 0:
                payloads = (
                    ((base, 10, 11, 12),),
                    ((base + 100, 20, 21, 22),),
                )
                routes = (((2, -1),), ((0, -1),))
            elif pattern == 1:
                payloads = (
                    ((base, 30, 31, 32), (base + 1, 40, 41, 42)),
                    ((base + 100, 50, 51, 52),),
                )
                routes = (((0, -1), (2, 3)), ((1, 2),))
            elif pattern == 2:
                payloads = (
                    (),
                    ((base + 100, 60, 61, 62),
                     (base + 101, 70, 71, 72)),
                )
                routes = ((), ((0, 1), (3, -1)))
            else:
                payloads = (
                    ((base, 80, 81, 82),),
                    ((base + 100, 90, 91, 92),),
                )
                routes = (((-1, -1),), ((-1, -1),))
            spec = CaseSpec(
                "sequential-100-generations",
                payloads,
                routes,
            )
            self._dispatch_spec(spec)

    def _run_round_trip_smoke(self):
        spec = CaseSpec(
            "round-trip-smoke",
            (
                ((71, 72, 73, 74), (81, 82, 83, 84)),
                ((171, 172, 173, 174), (181, 182, 183, 184)),
            ),
            (
                ((0, -1), (2, -1)),
                ((1, -1), (3, -1)),
            ),
        )
        result, _ = self._dispatch_spec(spec)
        recv_x, _, _, handle, _ = result
        combined_x, combined_weights, event = self.buffer.combine(
            recv_x, handle, num_sms=1, num_qps=0)
        expected = self._tensor(
            spec.payloads[self.rank], HIDDEN, self.torch.bfloat16).cpu()
        self._assert_tensor(combined_x, expected, "round-trip combined_x")
        _check(combined_weights is None,
               "weightless round trip returned weights")
        _check(event.event is None and event.extra_tensors is None and
               event.hook_after_wait is None,
               "synchronous round trip returned deferred event state")

    def _run_invalid_diagnostic(self):
        spec = self.specs["invalid-expert-diagnostics"]
        x, routes, weights = self._materialize(spec)
        gathered = self._gather_literals(x, routes, weights)
        for source_routes in gathered["routes"]:
            _check(int(source_routes[0, 0].item()) == NUM_EXPERTS,
                   "invalid expert literal changed during gather")
        started = time.monotonic()
        try:
            self.buffer.dispatch(
                x,
                topk_idx=routes,
                num_experts=NUM_EXPERTS,
                num_max_tokens_per_rank=CAPACITY,
                expert_alignment=1,
                num_sms=1,
                num_qps=0,
                do_cpu_sync=True,
            )
        except RuntimeError as error:
            elapsed = time.monotonic() - started
            message = str(error)
            for expected in (
                    f"dispatch failed on rank {self.rank}",
                    "backend error 65537",
                    "error=invalid_protocol",
                    "command_index=0",
                    "opcode=0",
                    "channel=0",
                    "backend_status=65537"):
                _check(expected in message,
                       f"invalid expert diagnostic omitted {expected!r}: "
                       f"{message}")
            _check(re.search(
                       rf"peer={self.rank}(?:\D|$)", message) is not None,
                   f"invalid expert diagnostic peer mismatch: {message}")
            _check(re.search(r"generation=[1-9][0-9]*", message) is not None,
                   f"invalid expert diagnostic omitted generation: {message}")
            _check(elapsed < 30,
                   f"invalid expert diagnostic was not bounded: {elapsed:.3f}s")
        else:
            raise AssertionError("invalid expert route was accepted")

    def _run_named_case(self, name):
        if name in REGULAR_CASES:
            self._dispatch_spec(self.specs[name])
        elif name == "cached-reuse":
            self._run_cached(name)
        elif name == "cached-aligned-near-capacity":
            self._run_cached(name)
        elif name == "sequential-100-generations":
            self._run_generations()
        elif name == "round-trip-smoke":
            self._run_round_trip_smoke()
        elif name == "invalid-expert-diagnostics":
            self._run_invalid_diagnostic()
        else:
            raise AssertionError(f"unimplemented matrix case {name}")

    def _case_boundary(self, name):
        self.dist.barrier(self.group)
        local_error = None
        try:
            self._run_named_case(name)
        except BaseException as error:
            local_error = error
        self.dist.barrier(self.group)

        failed = self.torch.tensor(
            [int(local_error is not None)], dtype=self.torch.int32,
            device=self.device)
        self.dist.all_reduce(failed, group=self.group)
        if int(failed.item()) != 0:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"{name} failed on the peer rank")
        if self.rank == 0:
            print(f"PASS {name}", flush=True)

    def run(self, selected_cases):
        try:
            self.make_buffer()
            for name in selected_cases:
                self._case_boundary(name)
        finally:
            if self.buffer is not None:
                self.buffer.destroy()


def _parse_cases(value):
    if value is None:
        return CASE_NAMES
    selected = tuple(name.strip() for name in value.split(",")
                     if name.strip())
    unknown = [name for name in selected if name not in CASE_NAMES]
    if unknown:
        raise ValueError(f"unknown dispatch matrix cases: {unknown}")
    if not selected:
        raise ValueError("at least one dispatch matrix case is required")
    if "invalid-expert-diagnostics" in selected and \
            selected[-1] != "invalid-expert-diagnostics":
        raise ValueError("invalid-expert-diagnostics must run last")
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
            f"dispatch matrix requires two ranks, got {world_size}")
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    try:
        device = torch.device("npu", local_rank)
        matrix = DispatchMatrix(torch, dist, deep_ep, group, device)
        matrix.run(selected_cases)
        if dist.get_rank(group) == 0:
            print(
                f"Phase 2F two-rank dispatch matrix passed "
                f"({len(selected_cases)} cases)", flush=True)
    finally:
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
