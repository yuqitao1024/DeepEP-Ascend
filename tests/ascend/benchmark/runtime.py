import os
import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from tests.ascend.benchmark.report import (
    FORMULA_VERSION,
    BenchmarkReport,
    write_report_atomic,
)
from tests.ascend.benchmark.timing import (
    NpuEventTimer,
    logical_gbps,
    summarize_samples,
)
from tests.ascend.benchmark.timeline import stage_semantic
from tests.ascend.benchmark.workloads import classify_ascend_case
from tests.utils.ep_benchmark_core import (
    PERFORMANCE_OPERATIONS,
    TensorBytes,
    build_dispatch_arguments,
    calculate_combine_traffic,
    calculate_dispatch_traffic,
    count_unique_destinations,
    expanded_dispatch_copy_bytes,
)
from tests.utils.ep_benchmark_manifest import (
    BenchmarkManifest,
    EPModeCase,
    WorkloadSpec,
    build_manifest,
    enumerate_ep_mode_cases,
    load_manifest,
    write_manifest,
)


BF16_TOLERANCE = 1 / 128
NUM_SMS = 72
NUM_QPS = 0
WORK_COUNT_KEYS = frozenset({
    "input_tokens",
    "valid_routes",
    "received_records",
    "expanded_slots",
    "input_rows",
    "output_tokens",
    "hidden_elements",
    "topk_elements",
})


def _benchmark_timeout_seconds(environment: Any = os.environ) -> int:
    raw_value = environment.get(
        "DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS", "300")
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS must be a positive integer"
        ) from error
    if timeout_seconds <= 0:
        raise ValueError(
            "DEEP_EP_ASCEND_BENCHMARK_TIMEOUT_SECS must be a positive integer"
        )
    return timeout_seconds


class TorchNpuEventBackend:
    def __init__(self, torch_module: Any):
        self.torch = torch_module

    def synchronize(self) -> None:
        self.torch.npu.synchronize()

    def new_event(self, _name: str) -> Any:
        return self.torch.npu.Event(enable_timing=True)


@dataclass
class PreparedCase:
    case: EPModeCase
    x: Any
    topk_idx: Any
    topk_weights: Any
    bias: Any
    launches: dict[str, Callable[[], Any]]
    prepare_launches: dict[str, Callable[[], Any]]
    traffic: dict[str, dict[str, int]]
    work_counts: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        if set(self.work_counts) != set(PERFORMANCE_OPERATIONS):
            raise ValueError("work counts operation set")
        for operation_id, counts in self.work_counts.items():
            if set(counts) != WORK_COUNT_KEYS:
                raise ValueError(f"work counts keys for {operation_id}")
            if any(type(value) is not int or value < 0
                   for value in counts.values()):
                raise ValueError(f"work counts values for {operation_id}")


def _tensor_bytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


def _payload_rows(value: Any) -> int:
    payload = value[0] if isinstance(value, tuple) else value
    return payload.shape[0]


def _early_route_plan_observability(
    manifest: BenchmarkManifest,
    *,
    rank: int,
    enabled: bool,
) -> dict[str, Any]:
    spec = manifest.spec
    if not 0 <= rank < spec.world_size:
        raise ValueError("early route plan rank")
    local_experts = spec.num_experts // spec.world_size
    rank_counts = [0] * spec.world_size
    expert_counts = [
        [0] * local_experts for _ in range(spec.world_size)
    ]
    for routes in manifest.ranks[rank].topk_idx:
        destinations = set()
        for expert in routes:
            if expert < 0:
                continue
            destination = expert // local_experts
            local_expert = expert % local_experts
            destinations.add(destination)
            expert_counts[destination][local_expert] += 1
        for destination in destinations:
            rank_counts[destination] += 1

    maximum_experts = 256
    expert_capacity = (
        maximum_experts + spec.world_size - 1
    ) // spec.world_size
    slot_bytes = (
        16 + expert_capacity * 8 + 31
    ) // 32 * 32
    remote_count = spec.world_size - 1
    return {
        "enabled": enabled,
        "signal_index": 2,
        "slot_bytes": slot_bytes,
        "window_bytes": slot_bytes * spec.world_size,
        "published_rank_counts": rank_counts,
        "published_expert_counts": expert_counts,
        "commands": {
            "remote_put": remote_count if enabled else 0,
            "flush": 2 if enabled and remote_count else 0,
            "remote_signal": remote_count if enabled else 0,
        },
        "kernel_checks": {
            "route_payload_count_parity": enabled,
            "route_payload_generation_parity": enabled,
        },
    }


def _summary_dict(samples: list[float]) -> dict[str, float]:
    return summarize_samples(samples).to_dict()


def _total_logical_bytes(traffic: dict[str, int]) -> int:
    return sum(traffic.values())


def _build_operation_work_counts(
    spec: WorkloadSpec,
    *,
    valid_routes: int,
    normal_input_rows: int,
    normal_received_records: int,
    expanded_slots: int,
    combine_input_rows: int,
    reduced_input_rows: int,
) -> dict[str, dict[str, int]]:
    values = (
        spec.num_tokens,
        spec.hidden,
        spec.num_topk,
        valid_routes,
        normal_input_rows,
        normal_received_records,
        expanded_slots,
        combine_input_rows,
        reduced_input_rows,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("operation work count inputs")

    dispatch = {
        "input_tokens": spec.num_tokens,
        "valid_routes": valid_routes,
        "received_records": normal_received_records,
        "expanded_slots": 0,
        "input_rows": normal_input_rows,
        "output_tokens": normal_received_records,
        "hidden_elements": normal_input_rows * spec.hidden,
        "topk_elements": normal_input_rows * spec.num_topk,
    }
    combine = {
        "input_tokens": spec.num_tokens,
        "valid_routes": valid_routes,
        "received_records": normal_received_records,
        "expanded_slots": 0,
        "input_rows": combine_input_rows,
        "output_tokens": spec.num_tokens,
        "hidden_elements": combine_input_rows * spec.hidden,
        "topk_elements": spec.num_tokens * spec.num_topk,
    }
    return {
        "dispatch": dispatch,
        "expanded_dispatch": dispatch | {
            "expanded_slots": expanded_slots,
            "output_tokens": expanded_slots,
        },
        "cached_dispatch": dict(dispatch),
        "combine": combine,
        "reduced_combine": combine | {
            "expanded_slots": expanded_slots,
            "input_rows": reduced_input_rows,
            "hidden_elements": reduced_input_rows * spec.hidden,
        },
    }


def _derive_host_envelope_samples(
    wall_samples: list[float],
    device_samples: list[float],
) -> list[float]:
    if not wall_samples or len(wall_samples) != len(device_samples):
        raise ValueError("host envelope requires paired timing samples")
    overhead = []
    for wall, device in zip(wall_samples, device_samples):
        if (
            not math.isfinite(wall)
            or not math.isfinite(device)
            or wall <= 0.0
            or device <= 0.0
            or wall < device
        ):
            raise ValueError("host envelope timing samples are invalid")
        overhead.append(wall - device)
    return overhead


def _derive_stage_timeline(stages: list[dict[str, Any]]) -> dict[str, int]:
    if not stages:
        return {
            "start": 0,
            "end": 0,
            "envelope_cycles": 0,
            "active_cycles": 0,
            "idle_cycles": 0,
            "overlap_cycles": 0,
        }
    intervals = []
    summed_spans = 0
    for stage in stages:
        start = stage.get("start")
        end = stage.get("end")
        span = stage.get("span_cycles")
        if (
            type(start) is not int
            or type(end) is not int
            or type(span) is not int
            or start <= 0
            or end < start
            or span != end - start
        ):
            raise ValueError("stage timeline interval")
        intervals.append((start, end))
        summed_spans += span

    intervals.sort()
    timeline_start = intervals[0][0]
    timeline_end = intervals[0][1]
    active_cycles = 0
    merged_start, merged_end = intervals[0]
    for start, end in intervals[1:]:
        timeline_end = max(timeline_end, end)
        if start > merged_end:
            active_cycles += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)
    active_cycles += merged_end - merged_start
    envelope_cycles = timeline_end - timeline_start
    return {
        "start": timeline_start,
        "end": timeline_end,
        "envelope_cycles": envelope_cycles,
        "active_cycles": active_cycles,
        "idle_cycles": envelope_cycles - active_cycles,
        "overlap_cycles": summed_spans - active_cycles,
    }


def _configure_stage_profile_environment(enabled: bool) -> None:
    name = "DEEP_EP_ASCEND_PROFILE_STAGES"
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def _aggregate_stage_profiles(
    operation_id: str,
    rank_profiles: list[dict[str, Any]],
    rank_work_counts: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    if not rank_profiles:
        raise ValueError("stage profile requires at least one rank")
    if rank_work_counts is not None and len(rank_work_counts) != len(
        rank_profiles
    ):
        raise ValueError("stage profile work count rank mismatch")
    expected_operation = (
        "dispatch" if operation_id in {
            "dispatch", "expanded_dispatch", "cached_dispatch"
        } else "combine"
    )
    for rank, profile in enumerate(rank_profiles):
        if profile.get("available") is not True:
            reason = profile.get("reason", "unknown")
            details = {
                key: profile[key]
                for key in (
                    "stage", "block", "start", "end", "command_metrics",
                    "service",
                )
                if key in profile
            }
            suffix = f": {details!r}" if details else ""
            raise ValueError(
                f"stage profile unavailable on rank {rank}: {reason}"
                f"{suffix}")
    generations = {profile.get("generation") for profile in rank_profiles}
    operations = {profile.get("operation") for profile in rank_profiles}
    if len(generations) != 1:
        raise ValueError(
            f"stage profile generation mismatch for {operation_id}: "
            f"{sorted(generations, key=repr)!r}")
    if operations != {expected_operation}:
        raise ValueError(
            f"stage profile operation mismatch for {operation_id}: "
            f"expected {expected_operation!r}, got "
            f"{sorted(operations, key=repr)!r}")

    generation = next(iter(generations))
    phase_names = (
        "producer", "publication", "service_submit", "cq_wait",
        "barrier_wait",
        "consumer_wait", "consumer_compute", "epilogue",
    )
    service_cycle_names = (
        "cycles", "wait_cycles", "payload_command_cycles",
        "control_command_cycles", "flush_command_cycles",
        "barrier_command_cycles", "barrier_poll_cycles",
    )
    per_rank = []
    stage_spans: dict[str, int] = {}
    host_timeline_ns: dict[str, int] | None = None
    timeline_cycle_names = (
        "envelope_cycles", "active_cycles", "idle_cycles", "overlap_cycles",
    )
    device_timeline_cycles = {name: 0 for name in timeline_cycle_names}
    phase_cycles = {name: 0 for name in phase_names}
    service_cycles = {name: 0 for name in service_cycle_names}
    for rank, profile in enumerate(rank_profiles):
        if profile.get("completion_generation") != generation:
            raise ValueError("stage profile completion generation mismatch")
        rank_stages = []
        for stage in profile.get("stages", ()):
            blocks = stage.get("blocks", ())
            if len(blocks) != stage.get("block_count"):
                raise ValueError("stage profile block count mismatch")
            starts = [block.get("start", 0) for block in blocks]
            ends = [block.get("end", 0) for block in blocks]
            if (
                not starts
                or any(type(value) is not int or value <= 0 for value in starts)
                or any(type(value) is not int for value in ends)
                or any(end < start for start, end in zip(starts, ends))
            ):
                raise ValueError("stage profile block cycles")
            span = max(ends) - min(starts)
            stage_name = stage.get("name")
            if not isinstance(stage_name, str) or not stage_name:
                raise ValueError("stage profile stage name")
            stage_spans[stage_name] = max(
                stage_spans.get(stage_name, 0), span)
            semantic = (
                {
                    "stage_id": "FULL",
                    "short_name": "full instrumentation fallback",
                    "ascend_functions": (),
                    "cuda_counterpart": "",
                    "work_count_keys": (),
                    "independently_timed": True,
                }
                if stage_name == "full"
                else asdict(stage_semantic(operation_id, stage_name))
            )
            rank_stages.append(dict(
                stage,
                **semantic,
                **({
                    "work_counts": {
                        key: rank_work_counts[rank][key]
                        for key in semantic["work_count_keys"]
                    },
                } if rank_work_counts is not None else {}),
                start=min(starts),
                end=max(ends),
                span_cycles=span,
            ))

        rank_timeline = _derive_stage_timeline(rank_stages)
        for name in timeline_cycle_names:
            device_timeline_cycles[name] = max(
                device_timeline_cycles[name], rank_timeline[name])

        rank_phases = profile.get("phase_cycles")
        if not isinstance(rank_phases, dict) or set(rank_phases) != set(
            phase_names
        ):
            raise ValueError("stage profile phase cycles")
        for name in phase_names:
            value = rank_phases[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"stage profile phase cycles.{name}")
        full_stages = [
            stage for stage in rank_stages
            if stage.get("id") == 0
        ]
        if full_stages:
            if (len(rank_stages) != 1 or
                    full_stages[0].get("name") != "full"):
                raise ValueError("stage profile full stage mask")
            rank_phases = {name: 0 for name in phase_names}
            rank_phases["producer"] = full_stages[0]["span_cycles"]
        for name in phase_names:
            phase_cycles[name] = max(phase_cycles[name], rank_phases[name])
        rank_service = profile.get("service")
        if not isinstance(rank_service, dict):
            raise ValueError("stage profile service cycles")
        for name in service_cycle_names:
            value = rank_service.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"stage profile service cycles.{name}")
            service_cycles[name] = max(service_cycles[name], value)
        rank_host_timeline = profile.get("host_timeline_ns")
        if rank_host_timeline is not None:
            if not isinstance(rank_host_timeline, dict) or not rank_host_timeline:
                raise ValueError("stage profile host timeline")
            if host_timeline_ns is None:
                host_timeline_ns = {name: 0 for name in rank_host_timeline}
            if set(rank_host_timeline) != set(host_timeline_ns):
                raise ValueError("stage profile host timeline fields")
            for name, value in rank_host_timeline.items():
                if type(value) is not int or value < 0:
                    raise ValueError(f"stage profile host timeline.{name}")
                host_timeline_ns[name] = max(host_timeline_ns[name], value)
        elif any(
            candidate.get("host_timeline_ns") is not None
            for candidate in rank_profiles
        ):
            raise ValueError("stage profile host timeline missing from one rank")
        per_rank.append(dict(
            profile, rank=rank, stages=rank_stages,
            device_timeline_cycles=rank_timeline,
        ))

    producer = phase_cycles["producer"]
    network = sum(
        phase_cycles[name]
        for name in (
            "publication", "service_submit", "cq_wait", "barrier_wait"
        )
    )
    consumer = sum(
        phase_cycles[name]
        for name in ("consumer_wait", "consumer_compute", "epilogue")
    )
    total = producer + network + consumer
    ceiling = total / max(producer, network, consumer) if total else 1.0
    result = {
        "operation": expected_operation,
        "generation": generation,
        "stage_spans_cycles": stage_spans,
        "device_timeline_cycles": device_timeline_cycles,
        "phase_cycles": phase_cycles,
        "service_cycles": service_cycles,
        "pipeline_cycles": {
            "producer": producer,
            "network": network,
            "consumer": consumer,
        },
        "optimistic_speedup_ceiling": ceiling,
        "per_rank": per_rank,
    }
    if host_timeline_ns is not None:
        result["host_timeline_ns"] = host_timeline_ns
    return result


def _aggregate_rank_operations(
    rank_operations: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    operations = []
    for operation_index, operation_id in enumerate(PERFORMANCE_OPERATIONS):
        rank_records = [records[operation_index] for records in rank_operations]
        device_samples = [
            max(record["device_samples"][sample_index] for record in rank_records)
            for sample_index in range(len(rank_records[0]["device_samples"]))
        ]
        wall_samples = [
            max(record["wall_samples"][sample_index] for record in rank_records)
            for sample_index in range(len(rank_records[0]["wall_samples"]))
        ]
        logical_bytes = {
            key: sum(record["logical_bytes"].get(key, 0) for record in rank_records)
            for key in sorted({
                key
                for record in rank_records
                for key in record["logical_bytes"]
            })
        }
        work_counts = {
            key: max(record["work_counts"][key] for record in rank_records)
            for key in sorted(WORK_COUNT_KEYS)
        }
        mean_seconds = summarize_samples(device_samples).mean
        operations.append({
            "operation_id": operation_id,
            "formula_version": FORMULA_VERSION,
            "device_seconds": _summary_dict(device_samples),
            "wall_seconds": _summary_dict(wall_samples),
            "device_samples": device_samples,
            "wall_samples": wall_samples,
            "logical_bytes": logical_bytes,
            "logical_byte_components": dict(logical_bytes),
            "work_counts": work_counts,
            "logical_gbps": logical_gbps(
                _total_logical_bytes(logical_bytes), mean_seconds
            ),
            "per_rank": [
                {
                    "rank": rank,
                    "device_seconds": _summary_dict(record["device_samples"]),
                    "wall_seconds": _summary_dict(record["wall_samples"]),
                    "logical_bytes": record["logical_bytes"],
                    "logical_byte_components": record[
                        "logical_byte_components"
                    ],
                    "work_counts": record["work_counts"],
                }
                for rank, record in enumerate(rank_records)
            ],
        })
        profiles = [record.get("stage_profile") for record in rank_records]
        if any(profile is not None for profile in profiles):
            if any(profile is None for profile in profiles):
                raise ValueError("stage profile missing from one or more ranks")
            operations[-1]["stage_profile"] = _aggregate_stage_profiles(
                operation_id,
                profiles,
                rank_work_counts=[
                    record["work_counts"] for record in rank_records
                ],
            )
            host_envelope_samples = _derive_host_envelope_samples(
                wall_samples, device_samples)
            operations[-1]["stage_profile"]["host_envelope_seconds"] = (
                _summary_dict(host_envelope_samples)
            )
            operations[-1]["stage_profile"]["host_envelope_samples"] = (
                host_envelope_samples
            )
        route_plans = [record.get("route_plan") for record in rank_records]
        if any(route_plan is not None for route_plan in route_plans):
            if any(route_plan is None for route_plan in route_plans):
                raise ValueError("route plan observability missing from one rank")
            operations[-1]["route_plan"] = {
                "enabled": all(
                    route_plan["enabled"] for route_plan in route_plans
                ),
                "signal_index": route_plans[0]["signal_index"],
                "slot_bytes": route_plans[0]["slot_bytes"],
                "window_bytes": route_plans[0]["window_bytes"],
                "per_rank": [
                    dict(route_plan, rank=rank)
                    for rank, route_plan in enumerate(route_plans)
                ],
            }
    return operations


class AscendRuntime:
    def __init__(
        self,
        torch_module: Any,
        dist_module: Any,
        deep_ep_module: Any,
        group: Any,
        device: Any,
        args: Any,
        manifest: BenchmarkManifest,
        num_sms: int = NUM_SMS,
        num_qps: int = NUM_QPS,
    ):
        self.torch = torch_module
        self.dist = dist_module
        self.deep_ep = deep_ep_module
        self.group = group
        self.device = device
        self.args = args
        self.manifest = manifest
        self.rank = dist_module.get_rank(group)
        self.world_size = dist_module.get_world_size(group)
        self.num_sms = num_sms
        self.num_qps = num_qps
        self.buffer = None
        self.timer = NpuEventTimer(TorchNpuEventBackend(torch_module))

    def synchronized_step(self, operation: Callable[[], Any], label: str) -> Any:
        local_error = None
        result = None
        try:
            result = operation()
        except BaseException as error:
            local_error = error
        failed = self.torch.tensor(
            [int(local_error is not None)],
            dtype=self.torch.int32,
            device=self.device,
        )
        self.dist.all_reduce(failed, group=self.group)
        if int(failed.item()) != 0:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"{label} failed on a peer rank")
        return result

    def _launch(
        self,
        case: EPModeCase,
        operation: str,
        arguments: dict[str, Any],
    ) -> Any:
        launch_arguments = dict(arguments)
        if case.with_previous_event:
            launch_arguments["previous_event"] = self.buffer.capture()
        result = getattr(self.buffer, operation)(**launch_arguments)
        if case.async_with_compute_stream:
            result[-1].current_stream_wait()
        return result

    def construct_buffer(self) -> None:
        spec = self.manifest.spec
        Buffer = self.deep_ep.ElasticBuffer
        bf16_bytes = Buffer.get_buffer_size_hint(
            self.group, spec.num_tokens, spec.hidden, spec.num_topk,
            use_fp8_dispatch=False, allow_hybrid_mode=False,
            allow_multiple_reduction=bool(self.args.allow_multiple_reduction),
        )
        fp8_bytes = Buffer.get_buffer_size_hint(
            self.group, spec.num_tokens, spec.hidden, spec.num_topk,
            use_fp8_dispatch=True, allow_hybrid_mode=False,
            allow_multiple_reduction=bool(self.args.allow_multiple_reduction),
        )
        self.buffer = self.deep_ep.ElasticBuffer(
            self.group,
            num_bytes=max(bf16_bytes, fp8_bytes),
            deterministic=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=bool(
                self.args.allow_multiple_reduction
            ),
            prefer_overlap_with_compute=False,
            num_allocated_qps=0,
            explicitly_destroy=True,
            num_gpu_timeout_secs=_benchmark_timeout_seconds(),
            num_cpu_timeout_secs=_benchmark_timeout_seconds(),
        )

    def destroy(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None

    def _materialize(self, case: EPModeCase) -> tuple[Any, Any, Any, Any]:
        rank_workload = self.manifest.ranks[self.rank]
        spec = self.manifest.spec
        bf16_x = self.torch.randn(
            (rank_workload.num_tokens, spec.hidden),
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        x = bf16_x
        if case.use_fp8_dispatch:
            scale_group_elements = 128
            padded_hidden = (
                (spec.hidden + scale_group_elements - 1)
                // scale_group_elements
                * scale_group_elements
            )
            quantization_input = bf16_x.float()
            if padded_hidden != spec.hidden:
                quantization_input = self.torch.nn.functional.pad(
                    quantization_input, (0, padded_hidden - spec.hidden))
            grouped = quantization_input.reshape(
                rank_workload.num_tokens, -1, scale_group_elements)
            scales = grouped.abs().amax(dim=2).clamp(min=1e-4) / 448.0
            payload = (grouped / scales.unsqueeze(2)).to(
                self.torch.float8_e4m3fn).reshape(
                    rank_workload.num_tokens, padded_hidden
                ).narrow(1, 0, spec.hidden).contiguous()
            x = (payload, scales.contiguous())
        topk_idx = self.torch.tensor(
            rank_workload.topk_idx,
            dtype=self.torch.int64,
            device=self.device,
        ).reshape(rank_workload.num_tokens, spec.num_topk).contiguous()
        topk_weights = self.torch.tensor(
            rank_workload.topk_weights,
            dtype=self.torch.float32,
            device=self.device,
        ).reshape(rank_workload.num_tokens, spec.num_topk).contiguous()
        bias = None
        if case.num_bias == 1:
            bias = self.torch.randn(
                bf16_x.shape, dtype=self.torch.bfloat16, device=self.device
            )
        elif case.num_bias == 2:
            bias = tuple(
                self.torch.randn(
                    bf16_x.shape, dtype=self.torch.bfloat16, device=self.device
                )
                for _ in range(2)
            )
        return x, topk_idx, topk_weights, bias

    def _reference(self, x: Any, topk_idx: Any, topk_weights: Any, bias: Any):
        from deep_ep.utils.refs import combine as ref_combine
        from deep_ep.utils.refs import dispatch as ref_dispatch
        from deep_ep.utils.refs import generate_pre_combine_data

        spec = self.manifest.spec
        reference_x = ((x[0].view(self.torch.uint8), x[1])
                       if isinstance(x, tuple) else x)
        ref_dispatched = ref_dispatch(
            reference_x,
            topk_idx,
            topk_weights,
            spec.num_tokens,
            spec.num_experts,
        )
        ref_y = generate_pre_combine_data(
            self.rank * spec.num_tokens
            + self.torch.arange(
                topk_idx.shape[0], dtype=self.torch.int64, device=self.device
            ),
            spec.num_tokens,
            spec.num_topk,
            spec.hidden,
        )
        ref_y[topk_idx == -1] = 0
        if self.args.allow_multiple_reduction:
            combine_recipe = (True, False)
            reduced_recipe = (True, False)
        else:
            combine_recipe = (True, False)
            reduced_recipe = (False, False)
        logical = self.buffer.get_logical_domain_size()
        ref_combined = ref_combine(
            ref_y,
            topk_idx,
            *logical,
            spec.num_experts,
            bias,
            *combine_recipe,
        )
        ref_reduced = ref_combine(
            ref_y,
            topk_idx,
            *logical,
            spec.num_experts,
            bias,
            *reduced_recipe,
        )
        return ref_dispatched, ref_combined, ref_reduced

    def _prepare_case(self, case: EPModeCase) -> PreparedCase:
        from deep_ep.utils.refs import generate_pre_combine_data
        from deep_ep.utils.refs import ordered_accumulate

        spec = self.manifest.spec
        x, topk_idx, topk_weights, bias = self._materialize(case)
        reference = None
        if not self.args.skip_check:
            reference = self._reference(x, topk_idx, topk_weights, bias)

        dispatch_arguments = build_dispatch_arguments(
            case=case,
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_tokens_per_rank=spec.num_tokens,
            num_experts=spec.num_experts,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
        )
        normal = self._launch(case, "dispatch", dispatch_arguments.normal)
        recv_x, recv_topk_idx, recv_weights, handle, _ = normal
        cached = self._launch(
            case, "dispatch", dispatch_arguments.cached(handle))
        handle = cached[3]

        num_recv_tokens = int(
            handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        )
        src_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
        local_y = generate_pre_combine_data(
            src_global_idx, spec.num_tokens, spec.num_topk, spec.hidden
        )
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
        recv_payload = recv_x[0] if case.use_fp8_dispatch else recv_x
        input_for_combine = self.torch.empty(
            recv_payload.shape, dtype=self.torch.bfloat16, device=self.device)
        input_for_combine[:num_recv_tokens] = ordered_accumulate(local_y)

        combine_args = {
            "x": input_for_combine,
            "topk_weights": recv_weights,
            "bias": bias,
            "handle": handle,
            "num_sms": self.num_sms,
            "num_qps": self.num_qps,
            "async_with_compute_stream": case.async_with_compute_stream,
            "allocate_on_comm_stream": case.allocate_on_comm_stream,
        }
        combined = self._launch(case, "combine", combine_args)

        expanded = self._launch(case, "dispatch", dispatch_arguments.expanded)
        expanded_x, _, expanded_weights, expanded_handle, _ = expanded
        cached_expanded = self._launch(
            case,
            "dispatch",
            dispatch_arguments.cached_expanded(expanded_handle),
        )
        expanded_handle = cached_expanded[3]
        num_expanded_tokens = int(
            expanded_handle.psum_num_recv_tokens_per_expert[-1].item()
        )
        expanded_src_global_idx = expanded_handle.recv_src_metadata[
            :num_recv_tokens, 0
        ]
        expanded_local_y = generate_pre_combine_data(
            expanded_src_global_idx,
            spec.num_tokens,
            spec.num_topk,
            spec.hidden,
        )
        input_for_reduced = self.torch.empty(
            ((expanded_x[0] if case.use_fp8_dispatch else expanded_x).shape[0] + 1,
             spec.hidden),
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        expanded_slots = expanded_handle.recv_src_metadata[
            :num_recv_tokens, 2:
        ].flatten()
        input_for_reduced[expanded_slots] = expanded_local_y.reshape(
            -1, spec.hidden
        )
        input_for_reduced = input_for_reduced[:-1]
        reduced_args = {
            "x": input_for_reduced,
            "bias": bias,
            "handle": expanded_handle,
            "num_sms": self.num_sms,
            "num_qps": self.num_qps,
            "async_with_compute_stream": case.async_with_compute_stream,
            "allocate_on_comm_stream": case.allocate_on_comm_stream,
        }
        if self.args.allow_multiple_reduction:
            reduced_args["topk_weights"] = expanded_weights
        reduced = self._launch(case, "combine", reduced_args)

        if reference is not None:
            self._check_case(
                case,
                topk_idx,
                topk_weights,
                normal,
                expanded,
                cached,
                cached_expanded,
                combined,
                reduced,
                reference,
                num_recv_tokens,
            )

        timing_handles = {
            "cached_dispatch": handle,
            "combine": handle,
            "reduced_combine": expanded_handle,
        }

        def refresh_handle(operation_id: str, arguments: dict[str, Any]) -> None:
            timing_handles[operation_id] = self._launch(
                case, "dispatch", arguments)[3]

        launches = {
            "dispatch": lambda: self._launch(
                case, "dispatch", dispatch_arguments.normal),
            "expanded_dispatch": lambda: self._launch(
                case, "dispatch", dispatch_arguments.expanded),
            "cached_dispatch": lambda: self._launch(
                case,
                "dispatch",
                dispatch_arguments.cached(timing_handles["cached_dispatch"]),
            ),
            "combine": lambda: self._launch(
                case,
                "combine",
                combine_args | {"handle": timing_handles["combine"]},
            ),
            "reduced_combine": lambda: self._launch(
                case,
                "combine",
                reduced_args | {
                    "handle": timing_handles["reduced_combine"]},
            ),
        }
        prepare_launches = {
            "cached_dispatch": lambda: refresh_handle(
                "cached_dispatch", dispatch_arguments.normal),
            "combine": lambda: refresh_handle(
                "combine", dispatch_arguments.normal),
            "reduced_combine": lambda: refresh_handle(
                "reduced_combine", dispatch_arguments.expanded),
        }
        traffic = self._traffic(
            topk_idx,
            recv_x,
            recv_topk_idx,
            recv_weights,
            handle,
            expanded_handle,
            combined,
            num_recv_tokens,
            num_expanded_tokens,
            bias,
        )
        routes = self.manifest.ranks[self.rank].topk_idx
        valid_routes = sum(
            route >= 0 for row in routes for route in row
        )
        work_counts = _build_operation_work_counts(
            spec,
            valid_routes=valid_routes,
            normal_input_rows=_payload_rows(x),
            normal_received_records=num_recv_tokens,
            expanded_slots=num_expanded_tokens,
            combine_input_rows=input_for_combine.shape[0],
            reduced_input_rows=input_for_reduced.shape[0],
        )
        return PreparedCase(
            case=case,
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            bias=bias,
            launches=launches,
            prepare_launches=prepare_launches,
            traffic=traffic,
            work_counts=work_counts,
        )

    def _check_case(
        self,
        case: EPModeCase,
        topk_idx: Any,
        topk_weights: Any,
        normal: Any,
        expanded: Any,
        cached: Any,
        cached_expanded: Any,
        combined: Any,
        reduced: Any,
        reference: Any,
        num_recv_tokens: int,
    ) -> None:
        ref_dispatch, ref_combined, ref_reduced = reference
        ref_x, ref_idx, ref_weights, ref_src_idx, _ = ref_dispatch
        recv_x, recv_idx, recv_weights, handle, event = normal
        expanded_x, expanded_idx, expanded_weights, expanded_handle, _ = expanded
        cached_x, cached_idx, _, cached_handle, _ = cached
        cached_expanded_x, _, cached_expanded_weights, _, _ = cached_expanded

        assert (event.event is not None) == case.async_with_compute_stream
        assert expanded_idx is None
        assert (topk_idx.data_ptr() != handle.topk_idx.data_ptr()) == (
            case.do_handle_copy
        )
        assert handle.topk_idx.data_ptr() == cached_handle.topk_idx.data_ptr()
        assert (topk_idx.data_ptr() != cached_handle.topk_idx.data_ptr()) == (
            case.do_handle_copy
        )
        def assert_payload_equal(actual, expected):
            if case.use_fp8_dispatch:
                assert self.torch.equal(
                    actual[0].view(self.torch.uint8),
                    expected[0].view(self.torch.uint8))
                assert self.torch.equal(actual[1], expected[1])
            else:
                self.torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        def index_payload(value, index):
            return ((value[0].view(self.torch.uint8)[index], value[1][index])
                    if case.use_fp8_dispatch else value[index])

        assert_payload_equal(recv_x, cached_x)
        assert self.torch.equal(recv_idx, cached_idx)

        order = self.torch.argsort(handle.recv_src_metadata[:, 0])
        assert self.torch.equal(handle.recv_src_metadata[order, 0], ref_src_idx)
        assert_payload_equal(index_payload(recv_x, order), ref_x)
        assert self.torch.equal(recv_idx[order], ref_idx)
        ref_mask = ref_idx < 0
        assert self.torch.equal(
            recv_weights[order].masked_fill(ref_mask, 0),
            ref_weights.masked_fill(ref_mask, 0),
        )

        expanded_indices = expanded_handle.recv_src_metadata[:, 2:]
        expanded_mask = expanded_indices >= 0
        safe_indices = expanded_indices.masked_fill(~expanded_mask, 0)
        first_valid = expanded_mask.to(self.torch.int32).argmax(dim=1)
        row_indices = self.torch.arange(
            expanded_indices.shape[0], device=self.device
        )
        if case.use_fp8_dispatch:
            folded_x = (
                expanded_x[0].view(self.torch.uint8)[safe_indices][
                    row_indices, first_valid],
                expanded_x[1][safe_indices][row_indices, first_valid],
            )
        else:
            folded_x = expanded_x[safe_indices][row_indices, first_valid]
        folded_weights = expanded_weights[safe_indices]
        expanded_order = self.torch.argsort(
            expanded_handle.recv_src_metadata[:, 0]
        )
        assert_payload_equal(index_payload(folded_x, expanded_order), ref_x)
        assert self.torch.equal(
            folded_weights[expanded_order].masked_fill(ref_mask, 0),
            ref_weights.masked_fill(ref_mask, 0),
        )
        valid_slots = expanded_indices[expanded_mask]
        assert_payload_equal(
            index_payload(expanded_x, valid_slots),
            index_payload(cached_expanded_x, valid_slots))
        assert self.torch.equal(
            expanded_weights[valid_slots], cached_expanded_weights[valid_slots]
        )
        local_experts = self.manifest.spec.num_experts // self.world_size
        for expert in range(local_experts):
            start = int(expanded_handle.psum_num_recv_tokens_per_expert[expert].item())
            end = ((start + case.expert_alignment - 1) // case.expert_alignment) * case.expert_alignment
            if case.use_fp8_dispatch:
                assert bool((cached_expanded_x[0].view(self.torch.uint8)[
                    start:end] == 0).all().item())
                assert bool((cached_expanded_x[1][start:end] == 0).all().item())
            else:
                assert bool((cached_expanded_x[start:end] == 0).all().item())
            assert bool((cached_expanded_weights[start:end] == 0).all().item())

        combined_x, combined_weights, _ = combined
        reduced_x, reduced_weights, _ = reduced
        self.torch.testing.assert_close(
            combined_x.float(),
            ref_combined.float(),
            rtol=BF16_TOLERANCE,
            atol=BF16_TOLERANCE,
        )
        self.torch.testing.assert_close(
            reduced_x.float(),
            ref_reduced.float(),
            rtol=BF16_TOLERANCE,
            atol=BF16_TOLERANCE,
        )
        assert self.torch.equal(combined_weights, topk_weights)
        if self.args.allow_multiple_reduction:
            assert self.torch.equal(reduced_weights, topk_weights)
        assert (recv_x[0] if case.use_fp8_dispatch else recv_x).shape[0] == \
            num_recv_tokens

    def _traffic(
        self,
        topk_idx: Any,
        recv_x: Any,
        recv_topk_idx: Any,
        recv_weights: Any,
        handle: Any,
        expanded_handle: Any,
        combined: Any,
        num_recv_tokens: int,
        num_expanded_tokens: int,
        bias: Any,
    ) -> dict[str, dict[str, int]]:
        spec = self.manifest.spec
        num_scaleout_ranks, num_scaleup_ranks = (
            self.buffer.get_logical_domain_size()
        )
        routes = self.manifest.ranks[self.rank].topk_idx
        experts_per_scaleout = spec.num_experts // num_scaleout_ranks
        num_scaleout_send_tokens = (
            count_unique_destinations(routes, divisor=experts_per_scaleout)
            if num_scaleout_ranks > 1
            else 0
        )
        recv_sf = recv_x[1] if isinstance(recv_x, tuple) else None
        recv_x = recv_x[0] if isinstance(recv_x, tuple) else recv_x
        payload_row_bytes = recv_x.shape[1] * recv_x.element_size()
        if recv_sf is not None:
            payload_row_bytes += recv_sf.shape[1] * recv_sf.element_size()
        dispatch_tensors = (
            TensorBytes(recv_x.shape[0], payload_row_bytes),
            TensorBytes(
                recv_topk_idx.shape[0],
                recv_topk_idx.shape[1] * recv_topk_idx.element_size(),
            ),
            TensorBytes(
                recv_weights.shape[0],
                recv_weights.shape[1] * recv_weights.element_size(),
            ),
        )
        dispatch = calculate_dispatch_traffic(
            dispatch_tensors,
            num_recv_tokens=num_recv_tokens,
            num_scaleup_recv_tokens=num_recv_tokens,
            num_scaleout_send_tokens=num_scaleout_send_tokens,
        )
        dispatch_bytes = {
            "scaleout": dispatch.scaleout_bytes,
            "scaleup": dispatch.scaleup_bytes,
            "copy": dispatch.copy_bytes,
        }
        metadata_row_bytes = (
            expanded_handle.recv_src_metadata.shape[1]
            * expanded_handle.recv_src_metadata.element_size()
        )
        expanded_bytes = dict(dispatch_bytes)
        expanded_bytes["copy"] = expanded_dispatch_copy_bytes(
            num_recv_tokens,
            num_expanded_tokens,
            dispatch.bytes_per_token,
            metadata_row_bytes,
        )
        cached_bytes = dict(dispatch_bytes)
        cached_bytes["copy"] = 2 * dispatch.scaleup_bytes

        experts_per_rank = spec.num_experts // self.world_size
        if self.args.allow_multiple_reduction:
            combine_counts = (
                num_scaleout_send_tokens,
                num_recv_tokens,
                count_unique_destinations(routes, divisor=experts_per_rank),
            )
            reduced_counts = combine_counts
        else:
            rank_destinations = count_unique_destinations(
                routes, divisor=experts_per_rank
            )
            combine_counts = (
                0 if num_scaleout_ranks == 1 else rank_destinations,
                num_recv_tokens,
                rank_destinations,
            )
            reduced_counts = (
                0
                if num_scaleout_ranks == 1
                else count_unique_destinations(routes),
                int((recv_topk_idx[:num_recv_tokens] != -1).sum().item()),
                count_unique_destinations(routes),
            )
        combine_row_bytes = spec.hidden * 2
        if recv_weights is not None:
            combine_row_bytes += recv_weights.shape[1] * recv_weights.element_size()
        bias_bytes = 0
        if bias is not None:
            bias_tensors = bias if isinstance(bias, tuple) else (bias,)
            bias_bytes = sum(_tensor_bytes(value) for value in bias_tensors)
        combined_x, combined_weights, _ = combined
        reduction_write_bytes = _tensor_bytes(combined_x)
        if combined_weights is not None:
            reduction_write_bytes += _tensor_bytes(combined_weights)

        def combine_bytes(counts: tuple[int, int, int]) -> dict[str, int]:
            value = calculate_combine_traffic(
                *counts,
                bytes_per_token=combine_row_bytes,
                bias_bytes=bias_bytes,
                reduction_write_bytes=reduction_write_bytes,
            )
            return {
                "scaleout": value.scaleout_bytes,
                "scaleup": value.scaleup_bytes,
                "reduction": value.reduction_bytes,
            }

        return {
            "dispatch": dispatch_bytes,
            "expanded_dispatch": expanded_bytes,
            "cached_dispatch": cached_bytes,
            "combine": combine_bytes(combine_counts),
            "reduced_combine": combine_bytes(reduced_counts),
        }

    def run_case(self, case: EPModeCase) -> list[dict[str, Any]]:
        prepared = self.synchronized_step(
            lambda: self._prepare_case(case), f"{case.case_id}: preparation"
        )
        records = []
        for operation_id in PERFORMANCE_OPERATIONS:
            prepare = prepared.prepare_launches.get(operation_id)
            if prepare is not None:
                self.synchronized_step(
                    prepare, f"{case.case_id}: {operation_id}: preparation"
                )
            operation = prepared.launches[operation_id]
            for _ in range(self.args.warmups):
                self.buffer.barrier(with_cpu_sync=True, sequential=True)
                operation()
            device_samples = []
            wall_samples = []
            for _ in range(self.args.iterations):
                self.buffer.barrier(with_cpu_sync=True, sequential=True)
                sample = self.timer.measure(operation)
                device_samples.append(sample.device_seconds)
                wall_samples.append(sample.wall_seconds)
            record = {
                "operation_id": operation_id,
                "device_samples": device_samples,
                "wall_samples": wall_samples,
                "logical_bytes": prepared.traffic[operation_id],
                "logical_byte_components": prepared.traffic[operation_id],
                "work_counts": prepared.work_counts[operation_id],
            }
            if operation_id == "dispatch" and hasattr(self, "manifest"):
                selector_enabled = (
                    os.environ.get(
                        "DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN"
                    ) == "1"
                    and getattr(case, "use_fp8_dispatch", False)
                    and not getattr(case, "with_previous_event", False)
                    and not getattr(case, "async_with_compute_stream", False)
                    and not getattr(case, "allocate_on_comm_stream", False)
                    and self.num_sms > 1
                    and self.manifest.spec.num_experts <= 256
                    and self.manifest.spec.num_topk <= 8
                    and self.world_size <= 8
                )
                record["route_plan"] = _early_route_plan_observability(
                    self.manifest,
                    rank=self.rank,
                    enabled=selector_enabled,
                )
            if getattr(self.args, "profile_stages", False):
                def capture_profile():
                    self.buffer.barrier(with_cpu_sync=True, sequential=True)
                    self.buffer.reset_stage_profile()
                    operation()
                    self.torch.npu.synchronize()
                    return self.buffer.get_stage_profile()

                record["stage_profile"] = self.synchronized_step(
                    capture_profile,
                    f"{case.case_id}: {operation_id}: stage profile",
                )
            records.append(record)
        return records


def run_supported_matrix(
    runtime: AscendRuntime,
    selected_case_ids: tuple[str, ...],
    report: BenchmarkReport,
    classify=classify_ascend_case,
) -> None:
    cases_by_id = {case.case_id: case for case in enumerate_ep_mode_cases()}
    records_by_id = {record["case_id"]: record for record in report.cases}
    for case_id in selected_case_ids:
        case = cases_by_id[case_id]
        record = records_by_id[case_id]
        capability = classify(case)
        if not capability.supported:
            raise ValueError(
                f"cannot benchmark {capability.suite} case {case_id}: "
                f"{capability.reason}"
            )
        local_operations = runtime.run_case(case)
        gathered: list[Any] = [None] * runtime.world_size
        runtime.dist.all_gather_object(
            gathered, local_operations, group=runtime.group
        )
        if runtime.rank == 0:
            record["status"] = "passed"
            record["reason"] = ""
            record["operations"] = _aggregate_rank_operations(gathered)


def _resolve_manifest(args: Any, world_size: int) -> BenchmarkManifest:
    if args.workload_manifest:
        manifest = load_manifest(args.workload_manifest)
        if manifest.spec.world_size != world_size:
            raise ValueError(
                "manifest world_size does not match the process group"
            )
        return manifest
    return build_manifest(WorkloadSpec(
        world_size=world_size,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        seed=args.seed,
        unbalanced_ratio=args.unbalanced_ratio,
        precise_unbalanced_ratio=args.precise_unbalanced_ratio,
        masked_ratio=args.masked_ratio,
    ))


def run_benchmark(args: Any, selected_case_ids: tuple[str, ...]) -> int:
    import torch
    import torch.distributed as dist
    import torch_npu

    import deep_ep
    from deep_ep.utils.envs import init_seed

    del torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("npu", local_rank)
    runtime = None
    runtime_error = None
    try:
        manifest = _resolve_manifest(args, world_size)
        init_seed(manifest.spec.seed)
        if args.dump_manifest and rank == 0:
            write_manifest(Path(args.dump_manifest), manifest)
        report = BenchmarkReport.empty_for_cases(
            platform="ascend",
            cases=(
                case
                for case in enumerate_ep_mode_cases()
                if case.case_id in selected_case_ids
            ),
            classify=classify_ascend_case,
            workload_fingerprint=manifest.fingerprint,
            world_size=world_size,
            allow_multiple_reduction=args.allow_multiple_reduction,
            stage_profile=int(args.profile_stages),
        )
        report.workload = asdict(manifest.spec)
        report.timing_protocol = {
            "timer": "npu_event",
            "warmups": args.warmups,
            "iterations": args.iterations,
            "rank_aggregation": "maximum_latency",
            "logical_byte_aggregation": "sum",
        }
        report.device = {
            "name": torch.npu.get_device_name(local_rank),
            "local_rank": local_rank,
            "num_sms": args.num_sms,
            "num_qps": NUM_QPS,
        }
        runtime = AscendRuntime(
            torch, dist, deep_ep, group, device, args, manifest,
            num_sms=args.num_sms, num_qps=0,
        )
        _configure_stage_profile_environment(args.profile_stages)
        runtime.synchronized_step(
            runtime.construct_buffer, "buffer construction"
        )
        run_supported_matrix(runtime, selected_case_ids, report)
        if rank == 0:
            write_report_atomic(args.output, report)
            print(
                f"Ascend EP benchmark wrote {args.output} "
                f"({sum(case['status'] == 'passed' for case in report.cases)} "
                "cases passed)",
                flush=True,
            )
    except BaseException as error:
        runtime_error = error
    finally:
        cleanup_error = None
        try:
            if runtime is not None:
                runtime.destroy()
            dist.destroy_process_group()
        except BaseException as error:
            cleanup_error = error
        if runtime_error is not None:
            if cleanup_error is not None:
                raise RuntimeError(
                    f"runtime failed: {runtime_error}; cleanup failed: "
                    f"{cleanup_error}"
                ) from runtime_error
            raise runtime_error
        if cleanup_error is not None:
            raise cleanup_error
    return 0
