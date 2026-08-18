import os
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
NUM_SMS = 1
NUM_QPS = 0


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
    traffic: dict[str, dict[str, int]]


def _tensor_bytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


def _summary_dict(samples: list[float]) -> dict[str, float]:
    return summarize_samples(samples).to_dict()


def _total_logical_bytes(traffic: dict[str, int]) -> int:
    return sum(traffic.values())


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
        mean_seconds = summarize_samples(device_samples).mean
        operations.append({
            "operation_id": operation_id,
            "formula_version": FORMULA_VERSION,
            "device_seconds": _summary_dict(device_samples),
            "wall_seconds": _summary_dict(wall_samples),
            "device_samples": device_samples,
            "wall_samples": wall_samples,
            "logical_bytes": logical_bytes,
            "logical_gbps": logical_gbps(
                _total_logical_bytes(logical_bytes), mean_seconds
            ),
            "per_rank": [
                {
                    "rank": rank,
                    "device_seconds": _summary_dict(record["device_samples"]),
                    "wall_seconds": _summary_dict(record["wall_samples"]),
                    "logical_bytes": record["logical_bytes"],
                }
                for rank, record in enumerate(rank_records)
            ],
        })
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

    def construct_buffer(self) -> None:
        spec = self.manifest.spec
        self.buffer = self.deep_ep.ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=spec.num_tokens,
            hidden=spec.hidden,
            num_topk=spec.num_topk,
            use_fp8_dispatch=False,
            deterministic=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=bool(
                self.args.allow_multiple_reduction
            ),
            prefer_overlap_with_compute=False,
            num_allocated_qps=0,
            explicitly_destroy=True,
            num_gpu_timeout_secs=300,
            num_cpu_timeout_secs=300,
        )

    def destroy(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None

    def _materialize(self, case: EPModeCase) -> tuple[Any, Any, Any, Any]:
        rank_workload = self.manifest.ranks[self.rank]
        spec = self.manifest.spec
        x = self.torch.randn(
            (rank_workload.num_tokens, spec.hidden),
            dtype=self.torch.bfloat16,
            device=self.device,
        )
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
                x.shape, dtype=self.torch.bfloat16, device=self.device
            )
        elif case.num_bias == 2:
            bias = tuple(
                self.torch.randn(
                    x.shape, dtype=self.torch.bfloat16, device=self.device
                )
                for _ in range(2)
            )
        return x, topk_idx, topk_weights, bias

    def _reference(self, x: Any, topk_idx: Any, topk_weights: Any, bias: Any):
        from deep_ep.utils.refs import combine as ref_combine
        from deep_ep.utils.refs import dispatch as ref_dispatch
        from deep_ep.utils.refs import generate_pre_combine_data

        spec = self.manifest.spec
        ref_dispatched = ref_dispatch(
            x,
            topk_idx,
            topk_weights,
            spec.num_tokens,
            spec.num_experts,
        )
        ref_y = generate_pre_combine_data(
            self.rank * spec.num_tokens
            + self.torch.arange(
                x.shape[0], dtype=self.torch.int64, device=self.device
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
            num_sms=1,
            num_qps=0,
        )
        normal = self.buffer.dispatch(**dispatch_arguments.normal)
        expanded = self.buffer.dispatch(**dispatch_arguments.expanded)
        recv_x, recv_topk_idx, recv_weights, handle, _ = normal
        expanded_x, _, expanded_weights, expanded_handle, _ = expanded
        cached = self.buffer.dispatch(**dispatch_arguments.cached(handle))
        cached_expanded = self.buffer.dispatch(
            **dispatch_arguments.cached_expanded(expanded_handle)
        )

        num_recv_tokens = int(
            handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        )
        num_expanded_tokens = int(
            expanded_handle.psum_num_recv_tokens_per_expert[-1].item()
        )
        src_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
        local_y = generate_pre_combine_data(
            src_global_idx, spec.num_tokens, spec.num_topk, spec.hidden
        )
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
        input_for_combine = self.torch.empty_like(recv_x)
        input_for_combine[:num_recv_tokens] = ordered_accumulate(local_y)

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
            (expanded_x.shape[0] + 1, spec.hidden),
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

        combine_args = {
            "x": input_for_combine,
            "topk_weights": recv_weights,
            "bias": bias,
            "handle": handle,
            "num_sms": 1,
            "num_qps": 0,
        }
        reduced_args = {
            "x": input_for_reduced,
            "bias": bias,
            "handle": expanded_handle,
            "num_sms": 1,
            "num_qps": 0,
        }
        if self.args.allow_multiple_reduction:
            reduced_args["topk_weights"] = expanded_weights
        combined = self.buffer.combine(**combine_args)
        reduced = self.buffer.combine(**reduced_args)

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

        launches = {
            "dispatch": lambda: self.buffer.dispatch(
                **dispatch_arguments.normal
            ),
            "expanded_dispatch": lambda: self.buffer.dispatch(
                **dispatch_arguments.expanded
            ),
            "cached_dispatch": lambda: self.buffer.dispatch(
                **dispatch_arguments.cached(handle)
            ),
            "combine": lambda: self.buffer.combine(**combine_args),
            "reduced_combine": lambda: self.buffer.combine(**reduced_args),
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
        return PreparedCase(
            case=case,
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            bias=bias,
            launches=launches,
            traffic=traffic,
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
        cached_x, cached_idx, cached_weights, cached_handle, _ = cached
        cached_expanded_x, _, cached_expanded_weights, _, _ = cached_expanded

        assert event.event is None
        assert expanded_idx is None
        assert (topk_idx.data_ptr() != handle.topk_idx.data_ptr()) == (
            case.do_handle_copy
        )
        assert handle.topk_idx.data_ptr() == cached_handle.topk_idx.data_ptr()
        assert (topk_idx.data_ptr() != cached_handle.topk_idx.data_ptr()) == (
            case.do_handle_copy
        )
        self.torch.testing.assert_close(
            recv_x, cached_x, rtol=0, atol=0
        )
        assert self.torch.equal(recv_idx, cached_idx)
        assert self.torch.equal(recv_weights, cached_weights)

        order = self.torch.argsort(handle.recv_src_metadata[:, 0])
        assert self.torch.equal(handle.recv_src_metadata[order, 0], ref_src_idx)
        self.torch.testing.assert_close(recv_x[order], ref_x, rtol=0, atol=0)
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
        folded_x = expanded_x[safe_indices][row_indices, first_valid]
        folded_weights = expanded_weights[safe_indices][row_indices, first_valid]
        expanded_order = self.torch.argsort(
            expanded_handle.recv_src_metadata[:, 0]
        )
        self.torch.testing.assert_close(
            folded_x[expanded_order], ref_x, rtol=0, atol=0
        )
        assert self.torch.equal(
            folded_weights[expanded_order].masked_fill(ref_mask, 0),
            ref_weights.masked_fill(ref_mask, 0),
        )
        valid_slots = expanded_indices[expanded_mask]
        self.torch.testing.assert_close(
            expanded_x[valid_slots], cached_expanded_x[valid_slots], rtol=0, atol=0
        )
        assert self.torch.equal(
            expanded_weights[valid_slots], cached_expanded_weights[valid_slots]
        )
        local_experts = self.manifest.spec.num_experts // self.world_size
        for expert in range(local_experts):
            start = int(expanded_handle.psum_num_recv_tokens_per_expert[expert].item())
            end = ((start + case.expert_alignment - 1) // case.expert_alignment) * case.expert_alignment
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
        assert recv_x.shape[0] == num_recv_tokens

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
        dispatch_tensors = (
            TensorBytes(recv_x.shape[0], recv_x.shape[1] * recv_x.element_size()),
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
        combine_row_bytes = recv_x.shape[1] * recv_x.element_size()
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
            records.append({
                "operation_id": operation_id,
                "device_samples": device_samples,
                "wall_samples": wall_samples,
                "logical_bytes": prepared.traffic[operation_id],
            })
        return records


def run_supported_matrix(
    runtime: AscendRuntime,
    selected_case_ids: tuple[str, ...],
    report: BenchmarkReport,
) -> None:
    selected = set(selected_case_ids)
    cases_by_id = {case.case_id: case for case in enumerate_ep_mode_cases()}
    records_by_id = {record["case_id"]: record for record in report.cases}
    for case_id, case in cases_by_id.items():
        record = records_by_id[case_id]
        capability = classify_ascend_case(case)
        if not capability.supported:
            continue
        if case_id not in selected:
            record["status"] = "skipped"
            record["reason"] = "not_selected"
            continue
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
            cases=enumerate_ep_mode_cases(),
            classify=classify_ascend_case,
            workload_fingerprint=manifest.fingerprint,
            world_size=world_size,
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
        }
        runtime = AscendRuntime(
            torch, dist, deep_ep, group, device, args, manifest
        )
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
