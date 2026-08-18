import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass


WORLD_SIZE = 4
SCALE_UP_SIZE = 2
INVALID_SLOT = None


@dataclass(frozen=True)
class RouteRecord:
    origin_world_rank: int
    origin_source_row: int
    ingress_world_rank: int
    destination_world_rank: int
    destination_local_expert: int
    ingress_slot: object
    forwarded_slot: object
    generation: int
    topology_epoch: int = 1
    ingress_complete: bool = True
    forward_complete: bool = True


def _coordinates(rank):
    return rank // SCALE_UP_SIZE, rank % SCALE_UP_SIZE


def classify_route(origin, destination):
    origin_domain, origin_rail = _coordinates(origin)
    destination_domain, destination_rail = _coordinates(destination)
    if origin == destination:
        return "local", destination
    if origin_domain == destination_domain:
        return "scale-up", destination
    if origin_rail == destination_rail:
        return "scale-out", destination
    return "diagonal", destination_domain * SCALE_UP_SIZE + origin_rail


def reference_dispatch(payloads, destinations, generation):
    """Independent row-major 2x2 model; imports no backend route helper."""
    if len(payloads) != WORLD_SIZE or len(destinations) != WORLD_SIZE:
        raise ValueError("reference requires exactly four ranks")
    receives = [[] for _ in range(WORLD_SIZE)]
    records = [[] for _ in range(WORLD_SIZE)]
    ingress_counts = {}
    forwarded_counts = {}
    for origin in range(WORLD_SIZE):
        if len(payloads[origin]) != len(destinations[origin]):
            raise ValueError(f"rank {origin} payload and route counts differ")
        for source_row, (payload, destination) in enumerate(zip(
                payloads[origin], destinations[origin], strict=True)):
            if not 0 <= destination < WORLD_SIZE:
                raise ValueError(f"invalid destination {destination}")
            route_kind, ingress = classify_route(origin, destination)
            ingress_slot = INVALID_SLOT
            forwarded_slot = INVALID_SLOT
            if route_kind == "diagonal":
                ingress_key = (origin, ingress)
                forwarded_key = (origin, destination)
                ingress_slot = ingress_counts.get(ingress_key, 0)
                forwarded_slot = forwarded_counts.get(forwarded_key, 0)
                ingress_counts[ingress_key] = ingress_slot + 1
                forwarded_counts[forwarded_key] = forwarded_slot + 1
            receives[destination].append(payload)
            records[destination].append(RouteRecord(
                origin, source_row, ingress, destination, 0,
                ingress_slot, forwarded_slot, generation))
    return tuple(map(tuple, receives)), tuple(map(tuple, records))


def reference_cached_dispatch(payloads, destinations, prior_records, generation):
    receives, records = reference_dispatch(payloads, destinations, generation)
    prior_geometry = tuple(tuple(
        (row.origin_world_rank, row.origin_source_row,
         row.ingress_world_rank, row.destination_world_rank,
         row.destination_local_expert, row.ingress_slot, row.forwarded_slot)
        for row in rank_rows) for rank_rows in prior_records)
    next_geometry = tuple(tuple(
        (row.origin_world_rank, row.origin_source_row,
         row.ingress_world_rank, row.destination_world_rank,
         row.destination_local_expert, row.ingress_slot, row.forwarded_slot)
        for row in rank_rows) for rank_rows in records)
    if prior_geometry != next_geometry:
        raise ValueError("cached route geometry changed")
    return receives, records


def reference_combine(records, contributions, source_rows,
                      allow_multiple_reduction):
    outputs = [[0.0 for _ in range(source_rows[rank])]
               for rank in range(WORLD_SIZE)]
    for destination in range(WORLD_SIZE):
        if len(records[destination]) != len(contributions[destination]):
            raise ValueError("record and contribution counts differ")
        for record, values in zip(records[destination],
                                  contributions[destination], strict=True):
            values = values if allow_multiple_reduction else values[:1]
            outputs[record.origin_world_rank][record.origin_source_row] += sum(values)
    return tuple(tuple(rank) for rank in outputs)


def reference_reverse_schedule(records, contributions):
    """Return diagonal payload order after each recorded reverse-route stage."""
    reverse_forward = []
    for destination in range(WORLD_SIZE):
        if len(records[destination]) != len(contributions[destination]):
            raise ValueError("record and contribution counts differ")
        for record, value in zip(records[destination],
                                 contributions[destination], strict=True):
            route_kind, ingress = classify_route(
                record.origin_world_rank, destination)
            if route_kind != "diagonal":
                continue
            if ingress != record.ingress_world_rank:
                raise ValueError("record ingress does not match topology")
            reverse_forward.append((
                ingress, record.forwarded_slot, record, value))
    reverse_forward.sort(key=lambda item: (item[0], item[1]))

    reverse_return = sorted(
        reverse_forward,
        key=lambda item: (
            item[2].origin_world_rank, item[2].ingress_slot))
    return (
        tuple((ingress, forwarded_slot, value)
              for ingress, forwarded_slot, _, value in reverse_forward),
        tuple((record.origin_world_rank, record.ingress_slot, value)
              for _, _, record, value in reverse_return),
    )


def _contract():
    return {
        "allow_multiple_reduction": [False, True],
        "case_names": [
            "local-scale-up-scale-out-diagonal",
            "asymmetric-empty-ranks",
            "repeated-generations",
            "cached-dispatch",
            "reverse-combine",
        ],
        "environment": {
            "DEEP_EP_ASCEND_LOGICAL_SIMULATION": "1",
            "DEEP_EP_ASCEND_SCALE_UP_SIZE": "2",
        },
        "evidence": "logical-single-host",
        "handle_layout": [
            "DispatchHandleDescriptor",
            "HybridRouteRecord[route_record_count]",
        ],
        "rank_mapping": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "route_schedule_from_rank_0": {
            "0": {"ingress": 0, "stages": []},
            "1": {"ingress": 1, "stages": ["scale-up"]},
            "2": {"ingress": 2, "stages": ["scale-out"]},
            "3": {"ingress": 2, "stages": ["scale-out", "scale-up"]},
        },
        "world_size": WORLD_SIZE,
    }


def _run_runtime():
    expected_environment = _contract()["environment"]
    observed = {name: os.environ.get(name) for name in expected_environment}
    if observed != expected_environment:
        raise RuntimeError(
            f"hybrid smoke requires {expected_environment}, got {observed}")

    production_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(production_dir))
    import run_logical_scale_out_smoke as base
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep

    del torch_npu

    class HybridSmoke(base.LogicalScaleOutSmoke):
        def __init__(self, *args, allow_multiple_reduction, **kwargs):
            super().__init__(*args, **kwargs)
            self.allow_multiple_reduction = allow_multiple_reduction

        def make_buffer(self):
            self.buffer = self._step(
                lambda: self.deep_ep.ElasticBuffer(
                    self.group,
                    num_bytes=16 * 1024 * 1024,
                    num_gpu_timeout_secs=5,
                    allow_hybrid_mode=True,
                    allow_multiple_reduction=self.allow_multiple_reduction,
                    explicitly_destroy=True),
                "hybrid buffer construction")
            self._step(
                lambda: base._check(
                    self.buffer.get_logical_domain_size() == (2, 2),
                    "logical domain is not 2x2"),
                "hybrid topology validation")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    group = dist.group.WORLD
    if dist.get_world_size(group) != WORLD_SIZE:
        raise RuntimeError("hybrid smoke requires exactly four ranks")
    try:
        for multiple in (False, True):
            smoke = HybridSmoke(
                torch, dist, deep_ep, group, torch.device("npu", local_rank),
                local_rank, allow_multiple_reduction=multiple)
            smoke.make_buffer()
            try:
                smoke.run()
            finally:
                smoke.buffer.destroy()
    finally:
        dist.destroy_process_group()
    if int(os.environ["RANK"]) == 0:
        print("PASS logical-single-host device-only hybrid smoke", flush=True)


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
