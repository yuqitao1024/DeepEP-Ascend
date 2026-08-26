from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class StageSemantic:
    stage_id: str
    short_name: str
    ascend_functions: tuple[str, ...]
    cuda_counterpart: str
    work_count_keys: tuple[str, ...]
    independently_timed: bool = True


_DISPATCH_STAGES = MappingProxyType({
    "producer_control": StageSemantic(
        "D0", "control",
        ("direct_dispatch_producer_control_vf",),
        "dispatch_impl prologue and notify-warps setup",
        ("input_tokens",),
    ),
    "producer_group": StageSemantic(
        "D1", "grouping",
        ("direct_dispatch_producer_group_vf",),
        "notify warps in dispatch_impl",
        ("input_tokens", "valid_routes"),
    ),
    "producer_prefix": StageSemantic(
        "D2", "prefix/slot",
        ("direct_dispatch_producer_prefix_vf",),
        "warp prefix and atomic destination-slot assignment in dispatch_impl",
        ("valid_routes",),
    ),
    "producer_record": StageSemantic(
        "D3", "record packing",
        (
            "direct_dispatch_producer_record_vf",
            "direct_dispatch_producer_vector_payload_impl",
        ),
        (
            "dispatch warps, TMA load/store, scale copy, and metadata writes "
            "in dispatch_impl"
        ),
        ("valid_routes", "hidden_elements", "topk_elements"),
    ),
    "release_payload": StageSemantic(
        "D4", "publication",
        ("direct_dispatch_producer_release_vf",),
        "Gin put, put-value, flush, and GPU barrier in dispatch_impl",
        ("valid_routes", "hidden_elements", "topk_elements"),
    ),
    "release_control": StageSemantic(
        "D4", "publication",
        ("direct_dispatch_producer_release_vf",),
        "Gin put, put-value, flush, and GPU barrier in dispatch_impl",
        ("valid_routes", "hidden_elements", "topk_elements"),
    ),
    "release_barrier": StageSemantic(
        "D4", "publication",
        ("direct_dispatch_producer_release_vf",),
        "Gin put, put-value, flush, and GPU barrier in dispatch_impl",
        ("valid_routes", "hidden_elements", "topk_elements"),
    ),
    "epilogue_acquire": StageSemantic(
        "D5", "acquire/count",
        (
            "direct_dispatch_epilogue_acquire_vf",
            "direct_dispatch_epilogue_validate_records_vf",
            "direct_dispatch_epilogue_count_experts_vf",
        ),
        (
            "notify results, acquire, validation, and token traversal in "
            "dispatch_copy_epilogue_impl"
        ),
        ("received_records",),
    ),
    "epilogue_validate": StageSemantic(
        "D5", "acquire/count",
        (
            "direct_dispatch_epilogue_acquire_vf",
            "direct_dispatch_epilogue_validate_records_vf",
            "direct_dispatch_epilogue_count_experts_vf",
        ),
        (
            "notify results, acquire, validation, and token traversal in "
            "dispatch_copy_epilogue_impl"
        ),
        ("received_records",),
    ),
    "epilogue_validate_reduce": StageSemantic(
        "D5", "acquire/count",
        (
            "direct_dispatch_epilogue_acquire_vf",
            "direct_dispatch_epilogue_validate_records_vf",
            "direct_dispatch_epilogue_count_experts_vf",
        ),
        (
            "notify results, acquire, validation, and token traversal in "
            "dispatch_copy_epilogue_impl"
        ),
        ("received_records",),
    ),
    "epilogue_expert_count": StageSemantic(
        "D5", "acquire/count",
        (
            "direct_dispatch_epilogue_acquire_vf",
            "direct_dispatch_epilogue_validate_records_vf",
            "direct_dispatch_epilogue_count_experts_vf",
        ),
        (
            "notify results, acquire, validation, and token traversal in "
            "dispatch_copy_epilogue_impl"
        ),
        ("received_records",),
    ),
    "epilogue_expert_prefix": StageSemantic(
        "D6", "expert prefix",
        ("direct_dispatch_epilogue_prefix_vf",),
        "warp prefix and exchange in dispatch_copy_epilogue_impl",
        ("received_records",),
    ),
    "epilogue_metadata": StageSemantic(
        "D7", "metadata/destination",
        (
            "direct_dispatch_epilogue_metadata_vf",
            "direct_dispatch_epilogue_assign_destinations_vf",
        ),
        (
            "destination assignment and metadata traversal in "
            "dispatch_copy_epilogue_impl"
        ),
        ("received_records", "output_tokens", "topk_elements"),
    ),
    "epilogue_copy": StageSemantic(
        "D8", "output copy",
        (
            "direct_dispatch_epilogue_copy_outputs_vf",
            "direct_dispatch_epilogue_vector_payload_impl",
        ),
        "per-warp TMA load/store in dispatch_copy_epilogue_impl",
        ("output_tokens", "hidden_elements", "topk_elements"),
    ),
    "epilogue_complete": StageSemantic(
        "F0", "completion",
        ("direct_dispatch_epilogue_complete_vf",),
        "final store, barrier, or programmatic launch completion",
        (),
    ),
})


_COMBINE_STAGES = MappingProxyType({
    "producer_control": StageSemantic(
        "C0", "control",
        ("direct_combine_producer_control_vf",),
        "combine_impl prologue and workspace setup",
        ("input_rows",),
    ),
    "producer_plan": StageSemantic(
        "C1", "plan/prefix",
        (
            "direct_combine_producer_plan_vf",
            "direct_combine_producer_plan_prefix_vf",
        ),
        "contributor metadata traversal and prefix in combine_impl",
        ("input_rows", "valid_routes"),
    ),
    "producer_plan_prefix": StageSemantic(
        "C1", "plan/prefix",
        (
            "direct_combine_producer_plan_vf",
            "direct_combine_producer_plan_prefix_vf",
        ),
        "contributor metadata traversal and prefix in combine_impl",
        ("input_rows", "valid_routes"),
    ),
    "producer_record": StageSemantic(
        "C2", "record",
        (
            "direct_combine_producer_record_vf",
            "direct_combine_producer_vector_payload_impl",
        ),
        "TMA copy or warp-cooperative local reduction in combine_impl",
        ("input_rows", "hidden_elements", "topk_elements"),
    ),
    "producer_local_copy": StageSemantic(
        "C3", "local staging",
        ("direct_combine_producer_local_copy_vf",),
        "local-copy and send-buffer preparation in combine_impl",
        ("input_rows", "hidden_elements", "topk_elements"),
        independently_timed=False,
    ),
    "release_payload": StageSemantic(
        "C4", "publication",
        ("direct_combine_producer_release_vf",),
        "Gin put and final GPU barrier in combine_impl",
        ("input_rows", "hidden_elements", "topk_elements"),
    ),
    "release_control": StageSemantic(
        "C4", "publication",
        ("direct_combine_producer_release_vf",),
        "Gin put and final GPU barrier in combine_impl",
        ("input_rows", "hidden_elements", "topk_elements"),
    ),
    "release_barrier": StageSemantic(
        "C4", "publication",
        ("direct_combine_producer_release_vf",),
        "Gin put and final GPU barrier in combine_impl",
        ("input_rows", "hidden_elements", "topk_elements"),
    ),
    "epilogue_acquire": StageSemantic(
        "C5", "contributor slots",
        (
            "direct_combine_epilogue_acquire_vf",
            "direct_combine_epilogue_validate_vf",
            "direct_combine_epilogue_reduce_errors_vf",
            "direct_combine_epilogue_prepare_vector_slots_vf",
        ),
        "warp ballot, exchange, and compute_topk_slots",
        ("received_records", "output_tokens", "topk_elements"),
    ),
    "epilogue_validate": StageSemantic(
        "C5", "contributor slots",
        (
            "direct_combine_epilogue_acquire_vf",
            "direct_combine_epilogue_validate_vf",
            "direct_combine_epilogue_reduce_errors_vf",
            "direct_combine_epilogue_prepare_vector_slots_vf",
        ),
        "warp ballot, exchange, and compute_topk_slots",
        ("received_records", "output_tokens", "topk_elements"),
    ),
    "epilogue_validate_reduce": StageSemantic(
        "C5", "contributor slots",
        (
            "direct_combine_epilogue_acquire_vf",
            "direct_combine_epilogue_validate_vf",
            "direct_combine_epilogue_reduce_errors_vf",
            "direct_combine_epilogue_prepare_vector_slots_vf",
        ),
        "warp ballot, exchange, and compute_topk_slots",
        ("received_records", "output_tokens", "topk_elements"),
    ),
    "epilogue_reduce": StageSemantic(
        "C6", "reduction/copy",
        (
            "direct_combine_epilogue_reduce_vf",
            "direct_combine_epilogue_vector_reduce_impl",
        ),
        "combine_reduce in combine_reduce_epilogue_impl",
        ("output_tokens", "hidden_elements"),
    ),
    "epilogue_weights": StageSemantic(
        "C7", "weights",
        (
            "direct_combine_epilogue_weights_vf",
            "direct_combine_epilogue_vector_tail_vf",
        ),
        "lane-level top-k weight copy in combine_reduce_epilogue_impl",
        ("output_tokens", "topk_elements"),
    ),
    "epilogue_complete": StageSemantic(
        "F0", "completion",
        ("direct_combine_epilogue_complete_vf",),
        "final store, barrier, or programmatic launch completion",
        (),
    ),
})


_DISPATCH_OPERATIONS = frozenset({
    "dispatch", "expanded_dispatch", "cached_dispatch",
})
_COMBINE_OPERATIONS = frozenset({"combine", "reduced_combine"})


def stage_semantic(operation_id: str, raw_name: str) -> StageSemantic:
    if operation_id in _DISPATCH_OPERATIONS:
        catalog = _DISPATCH_STAGES
    elif operation_id in _COMBINE_OPERATIONS:
        catalog = _COMBINE_STAGES
    else:
        raise ValueError(f"stage semantic operation: {operation_id!r}")
    try:
        return catalog[raw_name]
    except KeyError as error:
        raise ValueError(
            f"stage semantic missing for {operation_id}.{raw_name}"
        ) from error


def operation_stage_semantics(operation_id: str) -> tuple[StageSemantic, ...]:
    if operation_id in _DISPATCH_OPERATIONS:
        catalog = _DISPATCH_STAGES
    elif operation_id in _COMBINE_OPERATIONS:
        catalog = _COMBINE_STAGES
    else:
        raise ValueError(f"stage semantic operation: {operation_id!r}")

    semantics = []
    seen = set()
    for semantic in catalog.values():
        if semantic.stage_id not in seen:
            semantics.append(semantic)
            seen.add(semantic.stage_id)
    return tuple(semantics)
