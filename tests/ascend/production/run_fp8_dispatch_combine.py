import argparse
import json
import os
from dataclasses import dataclass
from datetime import timedelta


SUPPORTED_WORLD_SIZES = (2, 4, 8)
CAPACITY = 4
HIDDEN = 512
NUM_TOPK = 2
SF_PACKS = 4

CASE_NAMES = (
    "normal-fp32",
    "packed-int32",
    "expanded",
    "zero-padded",
    "cached-representation-change",
    "weighted",
    "empty-input",
    "asymmetric-routing",
    "negative-one-route",
    "bf16-combine",
    "sequential-100-generations",
    "malformed-inputs",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    packed_sf: bool = False
    weighted: bool = False
    expanded: bool = False
    zero_padding: bool = False
    expert_alignment: int = 1
    column_major_output: bool = False


def _contract():
    return {
        "case_names": list(CASE_NAMES),
        "supported_world_sizes": list(SUPPORTED_WORLD_SIZES),
        "system_under_test": ["Buffer.dispatch", "Buffer.combine"],
        "reference": "gathered-fp8-bytes-and-sf-packs",
        "contract_checks": [
            "public-dispatch-combine",
            "independent-all-gather-reference",
            "exact-payload-bytes",
            "exact-scale-factor-packs",
            "exact-column-major-scale-stride",
            "fp32-and-packed-int32-scales",
            "row-and-column-major-scale-output",
            "cached-bf16-to-fp8-and-fp8-to-bf16",
            "independent-fp8-dequantization",
            "one-rank-malformed-recovery",
            "distributed-failure-aggregation",
            "finally-protected-teardown",
        ],
    }


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


class FP8RuntimeMatrix:
    def __init__(self, torch, dist, deep_ep, group, device):
        self.torch = torch
        self.dist = dist
        self.deep_ep = deep_ep
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.num_experts = self.world_size * 2
        self.buffer = None

    def make_buffer(self):
        self.buffer = self.deep_ep.ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=CAPACITY,
            hidden=HIDDEN,
            num_topk=NUM_TOPK,
            # BF16 sizing covers both cached representation transitions.
            use_fp8_dispatch=False,
            deterministic=False,
            allow_hybrid_mode=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=30,
        )
        _check(self.buffer.get_logical_domain_size() ==
               (1, self.world_size), "unexpected logical topology")

    def destroy(self):
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None

    def _token_count(self, name):
        if name == "empty-input":
            return 0
        if name == "asymmetric-routing":
            return 3 if self.rank == 0 else 1
        if name == "sequential-100-generations":
            return 1
        return 2

    def _routes(self, count, name):
        rows = []
        for token in range(count):
            first_rank = (self.rank + token + 1) % self.world_size
            second_rank = (self.rank + token + 2) % self.world_size
            first = first_rank * 2 + token % 2
            second = second_rank * 2 + (token + 1) % 2
            if name == "negative-one-route":
                first = -1 if token == 0 else first
                second = -1
            rows.append((first, second))
        return self.torch.tensor(
            rows, dtype=self.torch.int64, device=self.device).reshape(
                count, NUM_TOPK).contiguous()

    def _materialize(self, spec, salt=0, fp8=True):
        count = self._token_count(spec.name)
        values = self.torch.arange(
            count * HIDDEN, dtype=self.torch.float32,
            device=self.device).reshape(count, HIDDEN)
        values = values.remainder(31) - 15 + self.rank * 2 + salt
        x = values.to(
            self.torch.float8_e4m3fn if fp8 else self.torch.bfloat16).contiguous()
        routes = self._routes(count, spec.name)
        weights = None
        if spec.weighted:
            weights = self.torch.arange(
                count * NUM_TOPK, dtype=self.torch.float32,
                device=self.device).reshape(count, NUM_TOPK)
            weights = (weights + 1 + self.rank) / 16
        if not fp8:
            return x, None, routes, weights
        if spec.packed_sf:
            scale_values = self.torch.arange(
                count * SF_PACKS, dtype=self.torch.int32,
                device=self.device).reshape(SF_PACKS, count)
            sf = (scale_values + 0x3F404142 +
                  self.rank * 0x10101 + salt).transpose(0, 1)
        else:
            scale_values = self.torch.arange(
                count * SF_PACKS, dtype=self.torch.float32,
                device=self.device).reshape(count, SF_PACKS)
            sf = (scale_values + 1 + self.rank + salt) / 32
        return x, sf, routes, weights

    def _all_gather(self, x, sf, routes, weights):
        torch = self.torch
        count = x.shape[0]
        metadata = torch.tensor(
            [count, int(weights is not None)], dtype=torch.int32,
            device=self.device)
        fp8 = sf is not None
        payload = torch.zeros(
            (CAPACITY, HIDDEN), dtype=torch.uint8 if fp8 else x.dtype,
            device=self.device)
        sf_bits = (torch.zeros(
            (CAPACITY, SF_PACKS), dtype=torch.int32, device=self.device)
            if fp8 else None)
        route_rows = torch.full(
            (CAPACITY, NUM_TOPK), -1, dtype=torch.int64,
            device=self.device)
        weight_rows = torch.zeros(
            (CAPACITY, NUM_TOPK), dtype=torch.float32, device=self.device)
        if count:
            payload[:count].copy_(x.view(torch.uint8) if fp8 else x)
            if fp8:
                sf_bits[:count].copy_(sf.view(torch.int32))
            route_rows[:count].copy_(routes)
            if weights is not None:
                weight_rows[:count].copy_(weights)

        def gather(tensor):
            outputs = [torch.empty_like(tensor)
                       for _ in range(self.world_size)]
            self.dist.all_gather(outputs, tensor, group=self.group)
            return [output.cpu() for output in outputs]

        gathered_metadata = gather(metadata)
        gathered_payload = gather(payload)
        gathered_sf = gather(sf_bits) if fp8 else None
        gathered_routes = gather(route_rows)
        gathered_weights = gather(weight_rows)
        result = []
        for source_rank in range(self.world_size):
            source_count = int(gathered_metadata[source_rank][0].item())
            _check(0 <= source_count <= CAPACITY,
                   "gathered token count exceeds capacity")
            _check(int(gathered_metadata[source_rank][1].item()) ==
                   int(weights is not None), "weight presence differs by rank")
            result.append({
                "count": source_count,
                "payload": gathered_payload[source_rank][:source_count],
                "sf": (gathered_sf[source_rank][:source_count]
                       if fp8 else None),
                "routes": gathered_routes[source_rank][:source_count],
                "weights": (gathered_weights[source_rank][:source_count]
                            if weights is not None else None),
            })
        return result, fp8

    @staticmethod
    def _align(value, alignment):
        return ((value + alignment - 1) // alignment) * alignment

    def _reference(self, gathered, spec, fp8):
        torch = self.torch
        first_expert = self.rank * 2
        last_expert = first_expert + 2
        incoming = []
        for source_rank, source in enumerate(gathered):
            for token in range(source["count"]):
                route = source["routes"][token]
                if any(first_expert <= int(expert.item()) < last_expert
                       for expert in route):
                    incoming.append((source_rank, token, source))

        if not spec.expanded:
            payload = torch.stack([
                source["payload"][token]
                for _, token, source in incoming
            ]) if incoming else torch.empty(
                (0, HIDDEN), dtype=gathered[0]["payload"].dtype)
            sf = (torch.stack([
                source["sf"][token]
                for _, token, source in incoming
            ]) if incoming else torch.empty((0, SF_PACKS), dtype=torch.int32)) \
                if fp8 else None
            localized = []
            gathered_weight_rows = []
            for _, token, source in incoming:
                localized.append([
                    int(expert.item()) - first_expert
                    if first_expert <= int(expert.item()) < last_expert
                    else -1 for expert in source["routes"][token]
                ])
                if source["weights"] is not None:
                    gathered_weight_rows.append(source["weights"][token])
            topk = torch.tensor(localized, dtype=torch.int64).reshape(
                len(incoming), NUM_TOPK)
            weights = torch.stack(gathered_weight_rows) \
                if gathered_weight_rows else None
            return {"payload": payload, "sf": sf, "topk": topk,
                    "weights": weights, "rows": incoming, "fp8": fp8}

        expert_counts = [0, 0]
        for _, token, source in incoming:
            for expert in source["routes"][token]:
                value = int(expert.item())
                if first_expert <= value < last_expert:
                    expert_counts[value - first_expert] += 1
        starts = [0]
        starts.append(self._align(expert_counts[0], spec.expert_alignment))
        total = starts[1] + self._align(
            expert_counts[1], spec.expert_alignment)
        payload = torch.zeros(
            (total, HIDDEN), dtype=gathered[0]["payload"].dtype)
        sf = torch.zeros((total, SF_PACKS), dtype=torch.int32) if fp8 else None
        weights = torch.zeros((total,), dtype=torch.float32) \
            if spec.weighted else None
        occurrences = [0, 0]
        for _, token, source in incoming:
            for lane, expert in enumerate(source["routes"][token]):
                value = int(expert.item())
                if not first_expert <= value < last_expert:
                    continue
                local_expert = value - first_expert
                destination = starts[local_expert] + occurrences[local_expert]
                occurrences[local_expert] += 1
                payload[destination].copy_(source["payload"][token])
                if fp8:
                    sf[destination].copy_(source["sf"][token])
                if weights is not None:
                    weights[destination] = source["weights"][token, lane]
        return {"payload": payload, "sf": sf, "topk": None,
                "weights": weights, "rows": incoming, "fp8": fp8}

    def _assert_exact(self, actual, expected, label):
        _check(tuple(actual.shape) == tuple(expected.shape),
               f"{label} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        observed = actual.detach().cpu()
        _check(self.torch.equal(observed, expected),
               f"{label} differs: {observed.tolist()} != {expected.tolist()}")

    def _dequantize_fp8(self, payload, sf, packed):
        torch = self.torch
        values = payload.view(torch.float8_e4m3fn).to(torch.float32)
        if packed:
            shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int64)
            exponents = (sf.to(torch.int64).unsqueeze(-1) >> shifts) & 0xff
            factors = torch.pow(
                torch.tensor(2.0, dtype=torch.float32),
                exponents.to(torch.float32) - 127).reshape(sf.shape[0], -1)
        else:
            factors = sf.view(torch.float32)
        groups = torch.arange(HIDDEN, dtype=torch.int64) // 128
        return values * factors[:, groups]

    def _verify_dispatch(self, result, reference, spec, expected_handle=None):
        recv, recv_topk, recv_weights, handle, event = result
        if reference["fp8"]:
            _check(isinstance(recv, tuple) and len(recv) == 2,
                   "FP8 dispatch did not return payload and scales")
            recv_x, recv_sf = recv
            self._assert_exact(
                recv_x.view(self.torch.uint8), reference["payload"],
                "FP8 payload bytes")
            self._assert_exact(
                recv_sf.view(self.torch.int32), reference["sf"],
                "scale factor packs")
            _check(recv_x.dtype == self.torch.float8_e4m3fn,
                   f"unexpected FP8 payload dtype {recv_x.dtype}")
            expected_sf_dtype = (
                self.torch.int32 if spec.packed_sf else self.torch.float32)
            _check(recv_sf.dtype == expected_sf_dtype,
                   f"unexpected scale factor dtype {recv_sf.dtype}")
            expected_values = self._dequantize_fp8(
                reference["payload"], reference["sf"], spec.packed_sf)
            actual_values = self._dequantize_fp8(
                recv_x.detach().cpu().view(self.torch.uint8),
                recv_sf.detach().cpu(), spec.packed_sf)
            _check(self.torch.allclose(actual_values, expected_values,
                                       rtol=0, atol=0),
                   "independent FP8 dequantization differs")
            if spec.column_major_output:
                _check(recv_sf.stride() ==
                       (1, self._align(recv_sf.shape[0], 4)),
                       f"invalid column-major SF stride {recv_sf.stride()}")
            else:
                _check(recv_sf.stride() == (SF_PACKS, 1),
                       f"invalid row-major SF stride {recv_sf.stride()}")
        else:
            recv_x = recv
            self._assert_exact(recv_x, reference["payload"], "BF16 payload")
        if reference["topk"] is None:
            _check(recv_topk is None, "expanded dispatch returned top-k")
        else:
            self._assert_exact(recv_topk, reference["topk"], "top-k")
        if reference["weights"] is None:
            _check(recv_weights is None, "unexpected received weights")
        else:
            self._assert_exact(
                recv_weights, reference["weights"], "top-k weights")
        if expected_handle is not None:
            _check(handle is expected_handle, "cached dispatch replaced handle")
        _check(event.event is None, "synchronous dispatch returned event")
        return recv, handle

    def _dispatch(self, spec, salt=0, handle=None, fp8=True):
        x, sf, routes, weights = self._materialize(spec, salt, fp8)
        gathered, gathered_fp8 = self._all_gather(x, sf, routes, weights)
        reference = self._reference(gathered, spec, gathered_fp8)
        result = self.buffer.dispatch(
            (x, sf) if fp8 else x,
            topk_idx=None if handle is not None else routes,
            topk_weights=weights,
            handle=handle,
            num_experts=None if handle is not None else self.num_experts,
            num_max_tokens_per_rank=None if handle is not None else CAPACITY,
            expert_alignment=(None if handle is not None
                              else spec.expert_alignment),
            num_sms=1,
            num_qps=0,
            do_cpu_sync=None if handle is not None else True,
            do_expand=spec.expanded,
            do_zero_padding=spec.zero_padding,
            use_tma_aligned_col_major_sf=(spec.column_major_output and fp8),
        )
        recv, returned_handle = self._verify_dispatch(
            result, reference, spec, expected_handle=handle)
        return recv, returned_handle, gathered, reference

    def _run_cached(self):
        bf16 = CaseSpec("cached-bf16")
        _, bf16_handle, _, _ = self._dispatch(bf16, fp8=False)
        fp8 = CaseSpec("cached-fp8", packed_sf=True)
        fp8_recv, _, _, _ = self._dispatch(
            fp8, salt=7, handle=bf16_handle, fp8=True)
        _check(isinstance(fp8_recv, tuple) and
               fp8_recv[1].dtype == self.torch.int32,
               "cached BF16-to-FP8 dispatch lost packed scale factors")
        _, fp8_handle, _, _ = self._dispatch(fp8, salt=11, fp8=True)
        bf16_recv, _, _, _ = self._dispatch(
            bf16, salt=13, handle=fp8_handle, fp8=False)
        _check(not isinstance(bf16_recv, tuple),
               "cached FP8-to-BF16 dispatch returned scale factors")

    def _run_combine(self):
        spec = CaseSpec("normal-fp32")
        _, handle, gathered, reference = self._dispatch(spec)
        rows = len(reference["rows"])
        combine_x = self.torch.empty(
            (rows, HIDDEN), dtype=self.torch.bfloat16, device=self.device)
        for row, (source_rank, source_token, _) in enumerate(reference["rows"]):
            combine_x[row].fill_(
                source_rank * 100 + source_token * 10 + self.rank + 1)
        combined, combined_weights, event = self.buffer.combine(
            combine_x, handle, num_sms=1, num_qps=0)
        expected = self.torch.zeros(
            (gathered[self.rank]["count"], HIDDEN),
            dtype=self.torch.bfloat16)
        for token in range(gathered[self.rank]["count"]):
            destinations = set()
            for expert in gathered[self.rank]["routes"][token]:
                value = int(expert.item())
                if value >= 0:
                    destinations.add(value // 2)
            value = sum(self.rank * 100 + token * 10 + destination + 1
                        for destination in destinations)
            expected[token].fill_(value)
        self._assert_exact(combined, expected, "BF16 combine")
        _check(combined_weights is None and event.event is None,
               "BF16 combine returned unexpected optional output")

    def _run_generations(self):
        spec = CaseSpec("sequential-100-generations")
        for generation in range(100):
            self._dispatch(spec, salt=generation % 13)

    def _run_malformed(self):
        spec = CaseSpec("normal-fp32")
        x, sf, routes, _ = self._materialize(spec)
        invalid = (x, sf.to(self.torch.bfloat16)) if self.rank == 0 else (x, sf)
        try:
            self.buffer.dispatch(
                invalid, topk_idx=routes, num_experts=self.num_experts,
                num_max_tokens_per_rank=CAPACITY, num_sms=1, num_qps=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("one-rank malformed FP8 input was accepted")
        self._dispatch(spec, salt=3)

    def _case_boundary(self, name, operation):
        self.dist.barrier(group=self.group)
        local_error = None
        try:
            operation()
        except BaseException as error:
            local_error = error
        failed = self.torch.tensor(
            [int(local_error is not None)], dtype=self.torch.int32,
            device=self.device)
        self.dist.all_reduce(failed, group=self.group)
        self.dist.barrier(group=self.group)
        if int(failed.item()) != 0:
            if local_error is not None:
                raise RuntimeError(f"{name}: {local_error}") from local_error
            raise RuntimeError(f"{name} failed on a peer rank")

    def run(self, selected):
        self.make_buffer()
        specs = {
            "normal-fp32": CaseSpec("normal-fp32"),
            "packed-int32": CaseSpec("packed-int32", packed_sf=True,
                                     column_major_output=True),
            "expanded": CaseSpec("expanded", expanded=True),
            "zero-padded": CaseSpec(
                "zero-padded", expanded=True, zero_padding=True,
                expert_alignment=4, column_major_output=True),
            "weighted": CaseSpec("weighted", weighted=True),
            "empty-input": CaseSpec(
                "empty-input", column_major_output=True),
            "asymmetric-routing": CaseSpec("asymmetric-routing"),
            "negative-one-route": CaseSpec("negative-one-route"),
        }
        try:
            for name in selected:
                if name in specs:
                    operation = lambda spec=specs[name]: self._dispatch(spec)
                elif name == "cached-representation-change":
                    operation = self._run_cached
                elif name == "bf16-combine":
                    operation = self._run_combine
                elif name == "sequential-100-generations":
                    operation = self._run_generations
                elif name == "malformed-inputs":
                    operation = self._run_malformed
                else:
                    raise ValueError(f"unknown FP8 runtime case: {name}")
                self._case_boundary(name, operation)
        finally:
            self.buffer.destroy()
            self.buffer = None


def _parse_cases(value):
    if value is None:
        return CASE_NAMES
    selected = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(set(selected) - set(CASE_NAMES))
    if unknown:
        raise ValueError(f"unknown FP8 runtime cases: {unknown}")
    if not selected:
        raise ValueError("at least one FP8 runtime case is required")
    return selected


def _run_runtime(selected):
    import torch
    import torch.distributed as dist
    import torch_npu

    import deep_ep

    del torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise RuntimeError(
            f"FP8 runtime matrix requires 2, 4, or 8 ranks, got {world_size}")
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    matrix = None
    try:
        matrix = FP8RuntimeMatrix(
            torch, dist, deep_ep, group, torch.device("npu", local_rank))
        matrix.run(selected)
        if dist.get_rank(group) == 0:
            print(
                f"Phase 3F {world_size}-rank FP8 runtime matrix passed "
                f"({len(selected)} cases)", flush=True)
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
    _run_runtime(_parse_cases(args.cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
