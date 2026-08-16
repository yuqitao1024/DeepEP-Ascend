#include "runtime.hpp"

namespace deep_ep::ascend::elastic {

namespace {

constexpr transport::TransportCapabilities kDispatchTransportCapabilities =
    transport::capability_bit(
        transport::TransportCapability::kSymmetricWindow) |
    transport::capability_bit(
        transport::TransportCapability::kDevicePut) |
    transport::capability_bit(
        transport::TransportCapability::kDevicePutValue) |
    transport::capability_bit(
        transport::TransportCapability::kRemoteSignal) |
    transport::capability_bit(
        transport::TransportCapability::kSystemMemoryOrdering) |
    transport::capability_bit(
        transport::TransportCapability::kDeviceBarrier) |
    transport::capability_bit(
        transport::TransportCapability::kScaleUpTeam);

CoreRuntimeStatus invalid(const char* message) {
    return {CoreRuntimeStatusCode::kInvalidArgument, 0, message};
}

CoreRuntimeStatus launch_failure(int backend_code, const char* message) {
    return {CoreRuntimeStatusCode::kLaunchFailure, backend_code, message};
}

bool has_deferred_mode(CoreModeFlags flags) {
    constexpr CoreModeFlags deferred =
        mode_bit(CoreMode::kAsyncEvent) |
        mode_bit(CoreMode::kCpuSync) |
        mode_bit(CoreMode::kHybrid) |
        mode_bit(CoreMode::kPipeline) |
        mode_bit(CoreMode::kEngram);
    return (flags & deferred) != 0;
}

bool same_topology(const CoreTopology& lhs, const CoreTopology& rhs) {
    return lhs.world_rank == rhs.world_rank &&
           lhs.world_size == rhs.world_size &&
           lhs.scale_up_rank == rhs.scale_up_rank &&
           lhs.scale_up_size == rhs.scale_up_size &&
           lhs.scale_out_rank == rhs.scale_out_rank &&
           lhs.scale_out_size == rhs.scale_out_size;
}

bool same_token_layout(const TokenLayout& lhs, const TokenLayout& rhs) {
    return lhs.hidden_offset == rhs.hidden_offset &&
           lhs.hidden_bytes == rhs.hidden_bytes &&
           lhs.scale_factor_offset == rhs.scale_factor_offset &&
           lhs.scale_factor_bytes == rhs.scale_factor_bytes &&
           lhs.topk_index_offset == rhs.topk_index_offset &&
           lhs.topk_index_bytes == rhs.topk_index_bytes &&
           lhs.topk_weight_offset == rhs.topk_weight_offset &&
           lhs.topk_weight_bytes == rhs.topk_weight_bytes &&
           lhs.source_metadata_offset == rhs.source_metadata_offset &&
           lhs.source_metadata_bytes == rhs.source_metadata_bytes &&
           lhs.stride_bytes == rhs.stride_bytes;
}

bool same_workspace_layout(
    const WorkspaceLayout& lhs, const WorkspaceLayout& rhs) {
    return lhs.barrier_offset == rhs.barrier_offset &&
           lhs.barrier_bytes == rhs.barrier_bytes &&
           lhs.block_count_offset == rhs.block_count_offset &&
           lhs.block_count_bytes == rhs.block_count_bytes &&
           lhs.reduced_count_offset == rhs.reduced_count_offset &&
           lhs.reduced_count_bytes == rhs.reduced_count_bytes &&
           lhs.prefix_sum_offset == rhs.prefix_sum_offset &&
           lhs.prefix_sum_bytes == rhs.prefix_sum_bytes &&
           lhs.slot_offset == rhs.slot_offset &&
           lhs.slot_bytes == rhs.slot_bytes &&
           lhs.source_metadata_offset == rhs.source_metadata_offset &&
           lhs.source_metadata_bytes == rhs.source_metadata_bytes &&
           lhs.scratch_offset == rhs.scratch_offset &&
           lhs.scratch_bytes == rhs.scratch_bytes &&
           lhs.total_bytes == rhs.total_bytes;
}

bool same_symmetric_window_layout(
    const SymmetricWindowLayout& lhs,
    const SymmetricWindowLayout& rhs) {
    return lhs.abi_version == rhs.abi_version &&
           lhs.struct_size == rhs.struct_size &&
           lhs.control_offset == rhs.control_offset &&
           lhs.control_bytes == rhs.control_bytes &&
           lhs.dispatch_offset == rhs.dispatch_offset &&
           lhs.dispatch_record_bytes == rhs.dispatch_record_bytes &&
           lhs.dispatch_source_shard_bytes ==
               rhs.dispatch_source_shard_bytes &&
           lhs.dispatch_source_shard_count ==
               rhs.dispatch_source_shard_count &&
           lhs.dispatch_bytes == rhs.dispatch_bytes &&
           lhs.combine_offset == rhs.combine_offset &&
           lhs.combine_record_bytes == rhs.combine_record_bytes &&
           lhs.combine_contributor_shard_bytes ==
               rhs.combine_contributor_shard_bytes &&
           lhs.combine_contributor_shard_count ==
               rhs.combine_contributor_shard_count &&
           lhs.combine_bytes == rhs.combine_bytes &&
           lhs.reserve_offset == rhs.reserve_offset &&
           lhs.reserve_bytes == rhs.reserve_bytes &&
           lhs.total_bytes == rhs.total_bytes &&
           lhs.dispatch_control_offset == rhs.dispatch_control_offset &&
           lhs.dispatch_control_bytes == rhs.dispatch_control_bytes &&
           lhs.dispatch_receive_offset == rhs.dispatch_receive_offset &&
           lhs.dispatch_receive_shard_bytes ==
               rhs.dispatch_receive_shard_bytes &&
           lhs.dispatch_receive_shard_count ==
               rhs.dispatch_receive_shard_count &&
           lhs.dispatch_receive_bytes == rhs.dispatch_receive_bytes &&
           lhs.dispatch_staging_offset == rhs.dispatch_staging_offset &&
           lhs.dispatch_staging_shard_bytes ==
               rhs.dispatch_staging_shard_bytes &&
           lhs.dispatch_staging_shard_count ==
               rhs.dispatch_staging_shard_count &&
           lhs.dispatch_staging_bytes == rhs.dispatch_staging_bytes;
}

bool context_topology_matches(
    const transport::TransportTopology& context,
    const CoreTopology& tiling) {
    return context.world_rank == tiling.world_rank &&
           context.world_size == tiling.world_size &&
           context.scale_up_rank == tiling.scale_up_rank &&
           context.scale_up_size == tiling.scale_up_size &&
           context.scale_out_rank == tiling.scale_out_rank &&
           context.scale_out_size == tiling.scale_out_size;
}

CoreRuntimeStatus validate_operation_topology(const CoreTiling& tiling) {
    if (tiling.operation == OperationKind::kBarrier) {
        if (!is_two_rank_scale_up_topology(tiling.topology))
            return {CoreRuntimeStatusCode::kUnsupportedTopology, 0,
                    "barrier requires exactly two scale-up ranks"};
        return {};
    }
    if (tiling.operation == OperationKind::kDispatch &&
        (is_single_rank_topology(tiling.topology) ||
         is_two_rank_scale_up_topology(tiling.topology)))
        return {};
    if (!is_single_rank_topology(tiling.topology))
        return {CoreRuntimeStatusCode::kUnsupportedTopology, 0,
                "dispatch and combine multi-rank launch is deferred"};
    return {};
}

CoreRuntimeStatus validate_transport_context(const CoreTiling& tiling) {
    const auto& context = tiling.transport_context;
    if (context.abi_version != transport::kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(transport::DeviceTransportContext) ||
        !context_topology_matches(context.topology, tiling.topology))
        return invalid("transport context does not match tiling topology");
    if (tiling.operation != OperationKind::kBarrier &&
        !(tiling.operation == OperationKind::kDispatch &&
          is_two_rank_scale_up_topology(tiling.topology)))
        return {};
    const auto required = tiling.operation == OperationKind::kDispatch ?
        kDispatchTransportCapabilities : kBarrierTransportCapabilities;
    if ((context.capabilities & required) != required)
        return invalid("transport context lacks required capabilities");
    if (context.local_window_base == 0 || context.peer_address_table == 0 ||
        context.channel_table == 0 || context.backend_context == 0)
        return invalid("operation requires an exported transport context");
    return {};
}

bool is_aligned(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) %
               kAscendElasticAlignment ==
           0;
}

CoreRuntimeStatus validate_tiling_descriptor(const CoreTiling& tiling) {
    if (tiling.abi_version != kCoreTilingAbiVersion ||
        tiling.struct_size != sizeof(CoreTiling))
        return invalid("invalid core tiling ABI");
    if (!detail::valid_topology(tiling.topology))
        return invalid("invalid topology");
    const auto topology_status = validate_operation_topology(tiling);
    if (!topology_status.ok())
        return topology_status;
    if (tiling.element_kind == ElementKind::kFloat8E4M3)
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "FP8 runtime execution is deferred"};
    if (tiling.element_kind != ElementKind::kBFloat16)
        return invalid("invalid element kind");
    if (has_deferred_mode(tiling.mode_flags))
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "requested mode is deferred until a real transport exists"};

    constexpr CoreModeFlags known_modes =
        mode_bit(CoreMode::kCached) | mode_bit(CoreMode::kExpanded) |
        mode_bit(CoreMode::kZeroPadding) |
        mode_bit(CoreMode::kAllowMultipleReduction) |
        mode_bit(CoreMode::kAsyncEvent) | mode_bit(CoreMode::kCpuSync) |
        mode_bit(CoreMode::kHybrid) | mode_bit(CoreMode::kPipeline) |
        mode_bit(CoreMode::kEngram);
    if ((tiling.mode_flags & ~known_modes) != 0)
        return invalid("unknown core mode flag");

    CoreModeFlags operation_modes = 0;
    switch (tiling.operation) {
        case OperationKind::kBarrier:
            break;
        case OperationKind::kDispatch:
            operation_modes = mode_bit(CoreMode::kCached) |
                              mode_bit(CoreMode::kExpanded) |
                              mode_bit(CoreMode::kZeroPadding);
            break;
        case OperationKind::kCombine:
            operation_modes = mode_bit(CoreMode::kExpanded) |
                              mode_bit(CoreMode::kAllowMultipleReduction);
            break;
        default:
            return invalid("invalid operation kind");
    }
    if ((tiling.mode_flags & ~operation_modes) != 0)
        return invalid("mode is invalid for operation");

    CoreTilingInput input{};
    input.operation = tiling.operation;
    input.element_kind = tiling.element_kind;
    input.mode_flags = tiling.mode_flags;
    input.num_tokens = tiling.num_tokens;
    input.hidden = tiling.hidden;
    input.num_experts = tiling.num_experts;
    input.num_topk = tiling.num_topk;
    input.expert_alignment = tiling.expert_alignment;
    input.num_max_tokens_per_rank = tiling.num_max_tokens_per_rank;
    input.num_scale_factor_packs = tiling.num_scale_factor_packs;
    input.scale_factor_pack_bytes = tiling.scale_factor_pack_bytes;
    input.has_reusable_slots = has_mode(tiling.mode_flags, CoreMode::kCached);
    input.topology = tiling.topology;
    CoreTiling expected{};
    if (!build_core_tiling(input, &expected).ok())
        return invalid("invalid core tiling fields");
    if (tiling.launch.num_blocks != expected.launch.num_blocks ||
        tiling.launch.num_threads != expected.launch.num_threads ||
        tiling.launch.dynamic_ub_bytes != expected.launch.dynamic_ub_bytes ||
        !same_topology(tiling.topology, expected.topology) ||
        !same_token_layout(tiling.token_layout, expected.token_layout) ||
        !same_workspace_layout(
            tiling.workspace_layout, expected.workspace_layout) ||
        !same_symmetric_window_layout(
            tiling.symmetric_window_layout,
            expected.symmetric_window_layout) ||
        tiling.dispatch_output_capacity !=
            expected.dispatch_output_capacity ||
        tiling.communication_buffer_bytes !=
            expected.communication_buffer_bytes ||
        tiling.workspace_bytes != expected.workspace_bytes)
        return invalid("core tiling descriptor is inconsistent");
    return validate_transport_context(tiling);
}

}  // namespace

CoreLaunchStorage required_core_launch_storage(const CoreTiling& tiling) {
    return {tiling.communication_buffer_bytes, tiling.workspace_bytes};
}

CoreRuntimeStatus validate_internal_launch(
    const CoreTiling& tiling, const CoreLaunchStorage& storage) {
    const auto descriptor_status = validate_tiling_descriptor(tiling);
    if (!descriptor_status.ok())
        return descriptor_status;
    if (storage.workspace_bytes < tiling.workspace_bytes ||
        storage.communication_buffer_bytes <
            tiling.communication_buffer_bytes)
        return {CoreRuntimeStatusCode::kInsufficientStorage, 0,
                "provided storage is smaller than the tiling requirement"};
    return {};
}

CoreRuntimeStatus launch_internal_barrier(
    const BarrierArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream) {
    const auto status = validate_internal_launch(tiling, storage);
    if (!status.ok())
        return status;
    if (tiling.operation != OperationKind::kBarrier)
        return invalid("barrier launch requires barrier tiling");
    if (stream == nullptr)
        return invalid("barrier stream must not be null");
    if (arguments.generation == 0)
        return invalid("barrier generation must not be zero");
    if (arguments.timeout_cycles == 0)
        return invalid("barrier timeout must not be zero");
    if (arguments.workspace == nullptr)
        return invalid("barrier workspace must not be null");
    if (!is_aligned(arguments.workspace))
        return invalid("barrier workspace is misaligned");
    const int result = deep_ep_ascend_launch_barrier(
        arguments, tiling, stream);
    return result == 0 ? CoreRuntimeStatus{} :
        launch_failure(result, "barrier kernel launch failed");
}

CoreRuntimeStatus launch_internal_dispatch(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream) {
    const auto status = validate_internal_launch(tiling, storage);
    if (!status.ok())
        return status;
    if (tiling.operation != OperationKind::kDispatch)
        return invalid("dispatch launch requires dispatch tiling");
    if (arguments.generation == 0)
        return invalid("dispatch generation must not be zero");
    if (arguments.timeout_cycles == 0)
        return invalid("dispatch timeout must not be zero");
    if ((tiling.num_tokens != 0 &&
         (arguments.x == nullptr || arguments.topk_indices == nullptr ||
          arguments.destination_slots == nullptr)) ||
        arguments.communication_buffer == nullptr ||
        arguments.workspace == nullptr || arguments.recv_x == nullptr ||
        arguments.recv_topk_indices == nullptr ||
        arguments.prefix_per_rank == nullptr ||
        arguments.prefix_per_expert == nullptr ||
        arguments.unaligned_per_expert == nullptr ||
        arguments.source_metadata == nullptr)
        return invalid("dispatch required argument is null");
    if (!is_aligned(arguments.communication_buffer) ||
        !is_aligned(arguments.workspace))
        return invalid("dispatch storage is misaligned");
    const int result = deep_ep_ascend_launch_dispatch(arguments, tiling, stream);
    return result == 0 ? CoreRuntimeStatus{} :
        launch_failure(result, "dispatch kernel launch failed");
}

CoreRuntimeStatus launch_internal_combine(
    const CombineArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream) {
    const auto status = validate_internal_launch(tiling, storage);
    if (!status.ok())
        return status;
    if (tiling.operation != OperationKind::kCombine)
        return invalid("combine launch requires combine tiling");
    if (arguments.x == nullptr || arguments.source_metadata == nullptr ||
        arguments.combined_topk_indices == nullptr ||
        arguments.prefix_per_rank == nullptr ||
        arguments.communication_buffer == nullptr ||
        arguments.workspace == nullptr || arguments.combined_x == nullptr)
        return invalid("combine required argument is null");
    if (!is_aligned(arguments.communication_buffer) ||
        !is_aligned(arguments.workspace))
        return invalid("combine storage is misaligned");
    int result = deep_ep_ascend_launch_combine(arguments, tiling, stream);
    if (result != 0)
        return launch_failure(result, "combine kernel launch failed");
    result = deep_ep_ascend_launch_combine_epilogue(
        arguments, tiling, stream);
    return result == 0 ? CoreRuntimeStatus{} :
        launch_failure(result, "combine epilogue launch failed");
}

}  // namespace deep_ep::ascend::elastic
