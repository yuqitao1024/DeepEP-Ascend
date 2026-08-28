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
        transport::TransportCapability::kAsyncCompletion) |
    transport::capability_bit(
        transport::TransportCapability::kSystemMemoryOrdering) |
    transport::capability_bit(
        transport::TransportCapability::kDeviceBarrier) |
    transport::capability_bit(
        transport::TransportCapability::kScaleUpTeam);

constexpr transport::TransportCapabilities kCombineTransportCapabilities =
    kDispatchTransportCapabilities;

CoreRuntimeStatus invalid(const char* message) {
    return {CoreRuntimeStatusCode::kInvalidArgument, 0, message};
}

CoreRuntimeStatus launch_failure(int backend_code, const char* message) {
    return {CoreRuntimeStatusCode::kLaunchFailure, backend_code, message};
}

bool has_deferred_mode(CoreModeFlags flags) {
    constexpr CoreModeFlags deferred = mode_bit(CoreMode::kEngram);
    return (flags & deferred) != 0;
}

bool same_topology(const CoreTopology& lhs, const CoreTopology& rhs) {
    return lhs.world_rank == rhs.world_rank &&
           lhs.world_size == rhs.world_size &&
           lhs.scale_up_rank == rhs.scale_up_rank &&
           lhs.scale_up_size == rhs.scale_up_size &&
           lhs.scale_out_rank == rhs.scale_out_rank &&
           lhs.scale_out_size == rhs.scale_out_size &&
           lhs.kind == rhs.kind && lhs.epoch == rhs.epoch;
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
           lhs.scratch_status_offset == rhs.scratch_status_offset &&
           lhs.scratch_local_count_offset == rhs.scratch_local_count_offset &&
           lhs.scratch_rank_counts_offset == rhs.scratch_rank_counts_offset &&
           lhs.scratch_rank_values_offset == rhs.scratch_rank_values_offset &&
           lhs.scratch_outbound_ingress_counts_offset ==
               rhs.scratch_outbound_ingress_counts_offset &&
           lhs.scratch_outbound_ingress_count ==
               rhs.scratch_outbound_ingress_count &&
           lhs.scratch_rank_indices_offset == rhs.scratch_rank_indices_offset &&
           lhs.scratch_rank_flags_offset == rhs.scratch_rank_flags_offset &&
           lhs.scratch_rank_count == rhs.scratch_rank_count &&
           lhs.dispatch_error_offset == rhs.dispatch_error_offset &&
           lhs.dispatch_error_count == rhs.dispatch_error_count &&
           lhs.dispatch_group_owner_offset ==
               rhs.dispatch_group_owner_offset &&
           lhs.dispatch_group_owner_bytes == rhs.dispatch_group_owner_bytes &&
           lhs.dispatch_group_tile_offset ==
               rhs.dispatch_group_tile_offset &&
           lhs.dispatch_group_tile_bytes == rhs.dispatch_group_tile_bytes &&
           lhs.dispatch_group_tile_count == rhs.dispatch_group_tile_count &&
           lhs.dispatch_group_error_offset ==
               rhs.dispatch_group_error_offset &&
           lhs.dispatch_group_error_bytes == rhs.dispatch_group_error_bytes &&
           lhs.dispatch_rank_bitmap_offset ==
               rhs.dispatch_rank_bitmap_offset &&
           lhs.dispatch_rank_bitmap_bytes == rhs.dispatch_rank_bitmap_bytes &&
           lhs.dispatch_expert_bitmap_offset ==
               rhs.dispatch_expert_bitmap_offset &&
           lhs.dispatch_expert_bitmap_bytes ==
               rhs.dispatch_expert_bitmap_bytes &&
           lhs.dispatch_receive_tile_error_offset ==
               rhs.dispatch_receive_tile_error_offset &&
           lhs.dispatch_receive_tile_error_bytes ==
               rhs.dispatch_receive_tile_error_bytes &&
           lhs.dispatch_receive_tile_count ==
               rhs.dispatch_receive_tile_count &&
           lhs.dispatch_expert_tile_count_offset ==
               rhs.dispatch_expert_tile_count_offset &&
           lhs.dispatch_expert_tile_count_bytes ==
               rhs.dispatch_expert_tile_count_bytes &&
           lhs.dispatch_expert_tile_count ==
               rhs.dispatch_expert_tile_count &&
           lhs.dispatch_pipeline_offset == rhs.dispatch_pipeline_offset &&
           lhs.dispatch_pipeline_bytes == rhs.dispatch_pipeline_bytes &&
           lhs.combine_record_slots_offset ==
               rhs.combine_record_slots_offset &&
           lhs.combine_record_slots_bytes ==
               rhs.combine_record_slots_bytes &&
           lhs.combine_producer_tile_rank_count_offset ==
               rhs.combine_producer_tile_rank_count_offset &&
           lhs.combine_producer_tile_rank_count_bytes ==
               rhs.combine_producer_tile_rank_count_bytes &&
           lhs.combine_producer_tile_error_offset ==
               rhs.combine_producer_tile_error_offset &&
           lhs.combine_producer_tile_error_bytes ==
               rhs.combine_producer_tile_error_bytes &&
           lhs.combine_producer_tile_count ==
               rhs.combine_producer_tile_count &&
           lhs.combine_receive_tile_error_offset ==
               rhs.combine_receive_tile_error_offset &&
           lhs.combine_receive_tile_error_bytes ==
               rhs.combine_receive_tile_error_bytes &&
           lhs.combine_receive_tile_count ==
               rhs.combine_receive_tile_count &&
           lhs.combine_receive_record_index_offset ==
               rhs.combine_receive_record_index_offset &&
           lhs.combine_receive_record_index_bytes ==
               rhs.combine_receive_record_index_bytes &&
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
           lhs.dispatch_staging_bytes == rhs.dispatch_staging_bytes &&
           lhs.combine_control_offset == rhs.combine_control_offset &&
           lhs.combine_control_bytes == rhs.combine_control_bytes &&
           lhs.combine_receive_offset == rhs.combine_receive_offset &&
           lhs.combine_receive_shard_bytes ==
               rhs.combine_receive_shard_bytes &&
           lhs.combine_receive_shard_count ==
               rhs.combine_receive_shard_count &&
           lhs.combine_receive_bytes == rhs.combine_receive_bytes &&
           lhs.combine_staging_offset == rhs.combine_staging_offset &&
           lhs.combine_staging_shard_bytes ==
               rhs.combine_staging_shard_bytes &&
           lhs.combine_staging_shard_count ==
               rhs.combine_staging_shard_count &&
           lhs.combine_staging_bytes == rhs.combine_staging_bytes &&
           lhs.combine_weight_offset == rhs.combine_weight_offset &&
           lhs.barrier_generation_offset == rhs.barrier_generation_offset &&
           lhs.barrier_generation_bytes == rhs.barrier_generation_bytes &&
           lhs.barrier_generation_count == rhs.barrier_generation_count &&
           lhs.barrier_completion_offset == rhs.barrier_completion_offset &&
           lhs.barrier_completion_bytes == rhs.barrier_completion_bytes &&
           lhs.barrier_completion_count == rhs.barrier_completion_count &&
           lhs.hybrid_route_record_offset == rhs.hybrid_route_record_offset &&
           lhs.hybrid_route_record_bytes == rhs.hybrid_route_record_bytes &&
           lhs.hybrid_route_record_count == rhs.hybrid_route_record_count &&
           lhs.hybrid_dispatch_ingress_control_offset ==
               rhs.hybrid_dispatch_ingress_control_offset &&
           lhs.hybrid_dispatch_ingress_control_bytes ==
               rhs.hybrid_dispatch_ingress_control_bytes &&
           lhs.hybrid_dispatch_ingress_control_count ==
               rhs.hybrid_dispatch_ingress_control_count &&
           lhs.hybrid_dispatch_ingress_shard_offset ==
               rhs.hybrid_dispatch_ingress_shard_offset &&
           lhs.hybrid_dispatch_ingress_shard_bytes ==
               rhs.hybrid_dispatch_ingress_shard_bytes &&
           lhs.hybrid_dispatch_ingress_shard_count ==
               rhs.hybrid_dispatch_ingress_shard_count &&
           lhs.hybrid_dispatch_ingress_bytes ==
               rhs.hybrid_dispatch_ingress_bytes &&
           lhs.hybrid_dispatch_forward_control_offset ==
               rhs.hybrid_dispatch_forward_control_offset &&
           lhs.hybrid_dispatch_forward_control_bytes ==
               rhs.hybrid_dispatch_forward_control_bytes &&
           lhs.hybrid_dispatch_forward_control_count ==
               rhs.hybrid_dispatch_forward_control_count &&
           lhs.hybrid_dispatch_forward_shard_offset ==
               rhs.hybrid_dispatch_forward_shard_offset &&
           lhs.hybrid_dispatch_forward_shard_bytes ==
               rhs.hybrid_dispatch_forward_shard_bytes &&
           lhs.hybrid_dispatch_forward_shard_count ==
               rhs.hybrid_dispatch_forward_shard_count &&
           lhs.hybrid_dispatch_forward_bytes ==
               rhs.hybrid_dispatch_forward_bytes &&
           lhs.hybrid_combine_reverse_forward_control_offset ==
               rhs.hybrid_combine_reverse_forward_control_offset &&
           lhs.hybrid_combine_reverse_forward_control_bytes ==
               rhs.hybrid_combine_reverse_forward_control_bytes &&
           lhs.hybrid_combine_reverse_forward_control_count ==
               rhs.hybrid_combine_reverse_forward_control_count &&
           lhs.hybrid_combine_reverse_forward_shard_offset ==
               rhs.hybrid_combine_reverse_forward_shard_offset &&
           lhs.hybrid_combine_reverse_forward_shard_bytes ==
               rhs.hybrid_combine_reverse_forward_shard_bytes &&
           lhs.hybrid_combine_reverse_forward_shard_count ==
               rhs.hybrid_combine_reverse_forward_shard_count &&
           lhs.hybrid_combine_reverse_forward_bytes ==
               rhs.hybrid_combine_reverse_forward_bytes &&
           lhs.hybrid_combine_return_control_offset ==
               rhs.hybrid_combine_return_control_offset &&
           lhs.hybrid_combine_return_control_bytes ==
               rhs.hybrid_combine_return_control_bytes &&
           lhs.hybrid_combine_return_control_count ==
               rhs.hybrid_combine_return_control_count &&
           lhs.hybrid_combine_return_shard_offset ==
               rhs.hybrid_combine_return_shard_offset &&
           lhs.hybrid_combine_return_shard_bytes ==
               rhs.hybrid_combine_return_shard_bytes &&
           lhs.hybrid_combine_return_shard_count ==
               rhs.hybrid_combine_return_shard_count &&
           lhs.hybrid_combine_return_bytes ==
               rhs.hybrid_combine_return_bytes &&
           lhs.hybrid_dispatch_ingress_staging_offset ==
               rhs.hybrid_dispatch_ingress_staging_offset &&
           lhs.hybrid_dispatch_ingress_staging_shard_bytes ==
               rhs.hybrid_dispatch_ingress_staging_shard_bytes &&
           lhs.hybrid_dispatch_ingress_staging_shard_count ==
               rhs.hybrid_dispatch_ingress_staging_shard_count &&
           lhs.hybrid_dispatch_ingress_staging_bytes ==
               rhs.hybrid_dispatch_ingress_staging_bytes;
}

bool context_topology_matches(
    const transport::TransportTopology& context,
    const CoreTopology& tiling) {
    return transport::valid_transport_topology(context) &&
           context.world_rank == tiling.world_rank &&
           context.world_size == tiling.world_size &&
           context.scale_up_rank == tiling.scale_up_rank &&
           context.scale_up_size == tiling.scale_up_size &&
           context.scale_out_rank == tiling.scale_out_rank &&
           context.scale_out_size == tiling.scale_out_size &&
           context.kind == tiling.kind && context.epoch == tiling.epoch;
}

CoreRuntimeStatus validate_operation_topology(const CoreTiling& tiling) {
    const bool two_dimensional =
        tiling.topology.kind ==
            transport::TransportTopologyKind::kLogicalSimulation ||
        tiling.topology.kind ==
            transport::TransportTopologyKind::kPhysical2D;
    if (tiling.operation == OperationKind::kBarrier) {
        if (!is_scale_up_topology(tiling.topology) && !two_dimensional)
            return {CoreRuntimeStatusCode::kUnsupportedTopology, 0,
                    "barrier requires a supported multi-rank topology"};
        return {};
    }
    if (tiling.operation == OperationKind::kDispatch &&
        (is_single_rank_topology(tiling.topology) ||
         is_scale_up_topology(tiling.topology) || two_dimensional))
        return {};
    if (tiling.operation == OperationKind::kCombine &&
        (is_scale_up_topology(tiling.topology) || two_dimensional))
        return {};
    if (tiling.operation == OperationKind::kCombine)
        return {CoreRuntimeStatusCode::kUnsupportedTopology, 0,
                "combine requires a supported multi-rank topology"};
    return {CoreRuntimeStatusCode::kUnsupportedTopology, 0,
            "unsupported operation topology"};
}

CoreRuntimeStatus validate_transport_context(const CoreTiling& tiling) {
    const auto& context = tiling.transport_context;
    if (context.abi_version != transport::kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(transport::DeviceTransportContext) ||
        !context_topology_matches(context.topology, tiling.topology))
        return invalid("transport context does not match tiling topology");
    if (is_single_rank_topology(tiling.topology))
        return {};
    auto required = tiling.operation == OperationKind::kDispatch ?
        kDispatchTransportCapabilities :
        (tiling.operation == OperationKind::kCombine ?
             kCombineTransportCapabilities : kBarrierTransportCapabilities);
    if (tiling.topology.kind == transport::TransportTopologyKind::kPhysical2D)
        required |= transport::capability_bit(
            transport::TransportCapability::kScaleOutTeam);
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
    if (tiling.element_kind != ElementKind::kBFloat16 &&
        tiling.element_kind != ElementKind::kFloat8E4M3)
        return invalid("invalid element kind");
    if (tiling.element_kind == ElementKind::kFloat8E4M3 &&
        tiling.operation != OperationKind::kDispatch)
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "FP8 is supported only for dispatch"};
    const auto topology_status = validate_operation_topology(tiling);
    if (!topology_status.ok())
        return topology_status;
    if (has_deferred_mode(tiling.mode_flags))
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "requested mode is deferred until a real transport exists"};
    const bool cpu_sync = has_mode(tiling.mode_flags, CoreMode::kCpuSync);
    const bool async_event = has_mode(tiling.mode_flags, CoreMode::kAsyncEvent);
    const bool pipeline = has_mode(tiling.mode_flags, CoreMode::kPipeline);
    if (async_event && !cpu_sync)
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "async dispatch requires the CPU-count split"};
    if ((cpu_sync || async_event) &&
        (tiling.operation != OperationKind::kDispatch ||
         (tiling.element_kind != ElementKind::kBFloat16 &&
          tiling.element_kind != ElementKind::kFloat8E4M3) ||
         has_mode(tiling.mode_flags, CoreMode::kCached) ||
         has_mode(tiling.mode_flags, CoreMode::kHybrid) ||
         tiling.topology.scale_out_size != 1))
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "split dispatch supports uncached BF16 or FP8 pure scale-up only"};
    if (pipeline &&
        (tiling.operation != OperationKind::kDispatch ||
         has_mode(tiling.mode_flags, CoreMode::kCached) ||
         has_mode(tiling.mode_flags, CoreMode::kHybrid) ||
         (!cpu_sync && tiling.element_kind != ElementKind::kFloat8E4M3) ||
         async_event || tiling.topology.world_size < 2 ||
         tiling.topology.scale_out_size != 1))
        return {CoreRuntimeStatusCode::kUnsupportedMode, 0,
                "pipeline dispatch supports uncached CPU-split or FP8 "
                "device-prefix scale-up only"};

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
                              mode_bit(CoreMode::kZeroPadding) |
                              mode_bit(CoreMode::kAllowMultipleReduction) |
                              mode_bit(CoreMode::kAsyncEvent) |
                              mode_bit(CoreMode::kCpuSync) |
                              mode_bit(CoreMode::kHybrid) |
                              mode_bit(CoreMode::kPipeline);
            break;
        case OperationKind::kCombine:
            operation_modes = mode_bit(CoreMode::kExpanded) |
                              mode_bit(CoreMode::kAllowMultipleReduction) |
                              mode_bit(CoreMode::kHybrid);
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
    input.data_num_blocks = tiling.data_launch.num_blocks;
    input.has_reusable_slots = has_mode(tiling.mode_flags, CoreMode::kCached);
    input.topology = tiling.topology;
    CoreTiling expected{};
    if (!build_core_tiling(input, &expected).ok())
        return invalid("invalid core tiling fields");
    if (tiling.launch.num_blocks != expected.launch.num_blocks ||
        tiling.launch.num_threads != expected.launch.num_threads ||
        tiling.launch.dynamic_ub_bytes != expected.launch.dynamic_ub_bytes ||
        tiling.control_launch.num_blocks !=
            expected.control_launch.num_blocks ||
        tiling.control_launch.num_threads !=
            expected.control_launch.num_threads ||
        tiling.control_launch.dynamic_ub_bytes !=
            expected.control_launch.dynamic_ub_bytes ||
        tiling.data_launch.num_blocks != expected.data_launch.num_blocks ||
        tiling.data_launch.num_threads != expected.data_launch.num_threads ||
        tiling.data_launch.dynamic_ub_bytes !=
            expected.data_launch.dynamic_ub_bytes ||
        !same_topology(tiling.topology, expected.topology) ||
        !same_token_layout(tiling.token_layout, expected.token_layout) ||
        !same_workspace_layout(
            tiling.workspace_layout, expected.workspace_layout) ||
        !same_symmetric_window_layout(
            tiling.symmetric_window_layout,
            expected.symmetric_window_layout) ||
        tiling.dispatch_output_capacity !=
            expected.dispatch_output_capacity ||
        tiling.hybrid_route_capacity != expected.hybrid_route_capacity ||
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

CoreRuntimeStatus launch_internal_dispatch_impl(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream,
    void* communication_stream) {
    const auto status = validate_internal_launch(tiling, storage);
    if (!status.ok())
        return status;
    if (tiling.operation != OperationKind::kDispatch)
        return invalid("dispatch launch requires dispatch tiling");
    if (arguments.generation == 0)
        return invalid("dispatch generation must not be zero");
    if (arguments.timeout_cycles == 0)
        return invalid("dispatch timeout must not be zero");
    const bool count_only = has_mode(
        tiling.mode_flags, CoreMode::kCpuSync);
    if (tiling.num_tokens != 0 && arguments.x == nullptr)
        return invalid("dispatch x is null for non-empty input");
    if (tiling.num_tokens != 0 && arguments.topk_indices == nullptr)
        return invalid("dispatch top-k indices are null for non-empty input");
    if (tiling.num_tokens != 0 && arguments.destination_slots == nullptr)
        return invalid("dispatch destination slots are null for non-empty input");
    if (arguments.communication_buffer == nullptr)
        return invalid("dispatch communication buffer is null");
    if (arguments.workspace == nullptr)
        return invalid("dispatch workspace is null");
    if (!count_only && arguments.num_output_tokens != 0 &&
        arguments.recv_x == nullptr)
        return invalid("dispatch output is null for non-empty output");
    if (!count_only && arguments.num_recv_tokens != 0 &&
        arguments.recv_topk_indices == nullptr)
        return invalid("dispatch received top-k indices are null");
    if (!count_only && arguments.num_recv_tokens != 0 &&
        arguments.source_metadata == nullptr)
        return invalid("dispatch source metadata is null");
    if (arguments.prefix_per_rank == nullptr)
        return invalid("dispatch rank prefix is null");
    if (arguments.prefix_per_expert == nullptr)
        return invalid("dispatch expert prefix is null");
    if (arguments.unaligned_per_expert == nullptr)
        return invalid("dispatch unaligned counts are null");
    if (has_mode(tiling.mode_flags, CoreMode::kHybrid) &&
        arguments.route_records == nullptr)
        return invalid("dispatch hybrid route records are null");
    if (has_mode(tiling.mode_flags, CoreMode::kHybrid) &&
        !has_mode(tiling.mode_flags, CoreMode::kCached) &&
        arguments.route_record_capacity < tiling.hybrid_route_capacity)
        return invalid("dispatch hybrid route capacity is insufficient");
    const bool fp8 = tiling.element_kind == ElementKind::kFloat8E4M3;
    if ((fp8 &&
         ((tiling.num_tokens != 0 && arguments.scale_factors == nullptr) ||
          arguments.scale_factor_token_stride == 0 ||
          arguments.scale_factor_pack_stride == 0 ||
          (!count_only &&
           (arguments.recv_scale_factors == nullptr ||
            arguments.recv_scale_factor_token_stride == 0 ||
            arguments.recv_scale_factor_pack_stride == 0)))) ||
        (!fp8 && (arguments.scale_factors != nullptr ||
                  arguments.recv_scale_factors != nullptr ||
                  arguments.scale_factor_token_stride != 0 ||
                  arguments.scale_factor_pack_stride != 0 ||
                  arguments.recv_scale_factor_token_stride != 0 ||
                  arguments.recv_scale_factor_pack_stride != 0)))
        return invalid("dispatch scale factors do not match element kind");
    if (!is_aligned(arguments.communication_buffer) ||
        !is_aligned(arguments.workspace))
        return invalid("dispatch storage is misaligned");
    const int result = communication_stream == nullptr ?
        deep_ep_ascend_launch_dispatch(arguments, tiling, stream) :
        deep_ep_ascend_launch_dispatch_pipeline(
            arguments, tiling, stream, communication_stream);
    return result == 0 ? CoreRuntimeStatus{} :
        launch_failure(result, "dispatch kernel launch failed");
}

CoreRuntimeStatus launch_internal_dispatch(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream) {
    if (has_mode(tiling.mode_flags, CoreMode::kPipeline))
        return invalid("pipeline dispatch requires a communication stream");
    return launch_internal_dispatch_impl(
        arguments, tiling, storage, stream, nullptr);
}

CoreRuntimeStatus launch_internal_dispatch_pipeline(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* producer_stream,
    void* communication_stream) {
    if (!has_mode(tiling.mode_flags, CoreMode::kPipeline) ||
        producer_stream == nullptr || communication_stream == nullptr ||
        producer_stream == communication_stream ||
        (arguments.pipeline_chunk_slots == 0 &&
         arguments.pipeline_chunk_tiles == 0) ||
        (arguments.pipeline_chunk_slots != 0 &&
         arguments.pipeline_chunk_tiles != 0))
        return invalid("invalid pipeline dispatch streams or chunk size");
    return launch_internal_dispatch_impl(
        arguments, tiling, storage, producer_stream, communication_stream);
}

CoreRuntimeStatus launch_internal_dispatch_epilogue(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream) {
    const auto status = validate_internal_launch(tiling, storage);
    if (!status.ok())
        return status;
    if (tiling.operation != OperationKind::kDispatch ||
        !has_mode(tiling.mode_flags, CoreMode::kCpuSync))
        return invalid("dispatch epilogue requires CPU-count split tiling");
    if (stream == nullptr)
        return invalid("dispatch epilogue stream must not be null");
    if (arguments.generation == 0 || arguments.timeout_cycles == 0)
        return invalid("dispatch epilogue requires generation and timeout");
    const auto maximum_recv_tokens = tiling.num_max_tokens_per_rank *
        static_cast<std::uint64_t>(tiling.topology.world_size);
    if (arguments.num_recv_tokens > maximum_recv_tokens ||
        arguments.num_output_tokens > tiling.dispatch_output_capacity)
        return invalid("dispatch epilogue output count exceeds capacity");
    if (arguments.communication_buffer == nullptr ||
        arguments.workspace == nullptr ||
        (arguments.num_output_tokens != 0 && arguments.recv_x == nullptr) ||
        (arguments.num_recv_tokens != 0 &&
         (arguments.recv_topk_indices == nullptr ||
          arguments.source_metadata == nullptr)) ||
        arguments.prefix_per_rank == nullptr ||
        arguments.prefix_per_expert == nullptr ||
        arguments.unaligned_per_expert == nullptr)
        return invalid("dispatch epilogue required argument is null");
    const bool fp8 = tiling.element_kind == ElementKind::kFloat8E4M3;
    if ((fp8 && arguments.num_output_tokens != 0 &&
         (arguments.recv_scale_factors == nullptr ||
          arguments.recv_scale_factor_token_stride == 0 ||
          arguments.recv_scale_factor_pack_stride == 0)) ||
        (!fp8 && (arguments.scale_factors != nullptr ||
                  arguments.recv_scale_factors != nullptr ||
                  arguments.scale_factor_token_stride != 0 ||
                  arguments.scale_factor_pack_stride != 0 ||
                  arguments.recv_scale_factor_token_stride != 0 ||
                  arguments.recv_scale_factor_pack_stride != 0)))
        return invalid("dispatch epilogue scale factors do not match element kind");
    const int result = deep_ep_ascend_launch_dispatch_epilogue(
        arguments, tiling, stream);
    return result == 0 ? CoreRuntimeStatus{} :
        launch_failure(result, "dispatch epilogue kernel launch failed");
}

CoreRuntimeStatus launch_internal_combine(
    const CombineArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream) {
    if (tiling.operation == OperationKind::kCombine &&
        tiling.num_tokens > tiling.num_max_tokens_per_rank)
        return invalid("combine token count exceeds shard capacity");
    const auto status = validate_internal_launch(tiling, storage);
    if (!status.ok())
        return status;
    if (tiling.operation != OperationKind::kCombine)
        return invalid("combine launch requires combine tiling");
    if (arguments.generation == 0)
        return invalid("combine generation must not be zero");
    if (arguments.timeout_cycles == 0)
        return invalid("combine timeout must not be zero");
    if (arguments.local_window_base == 0 ||
        arguments.local_window_base !=
            tiling.transport_context.local_window_base)
        return invalid("combine local window does not match transport");
    const bool expanded = has_mode(tiling.mode_flags, CoreMode::kExpanded);
    const bool allow_multiple_reduction = has_mode(
        tiling.mode_flags, CoreMode::kAllowMultipleReduction);
    if ((!expanded && arguments.num_source_rows != arguments.num_input_rows) ||
        (expanded && !allow_multiple_reduction &&
         arguments.topk_weights != nullptr))
        return invalid(expanded ?
            "expanded combine weights require allow_multiple_reduction" :
            "combine source and input row counts must match");
    if (tiling.num_max_tokens_per_rank >
        static_cast<std::uint64_t>(0x7fffffff) /
            static_cast<std::uint64_t>(tiling.topology.world_size))
        return invalid("combine shard capacity is not encodable");
    const std::uint64_t maximum_source_rows =
        tiling.num_max_tokens_per_rank *
        static_cast<std::uint64_t>(tiling.topology.world_size);
    if (arguments.num_source_rows > maximum_source_rows)
        return invalid("combine source row count exceeds fixed shards");
    const std::uint64_t maximum_input_rows = expanded ?
        tiling.dispatch_output_capacity : maximum_source_rows;
    if (arguments.num_input_rows > maximum_input_rows)
        return invalid("combine input row count exceeds fixed shards");
    if ((arguments.num_source_rows != 0 &&
         (arguments.x == nullptr || arguments.source_metadata == nullptr)) ||
        (tiling.num_tokens != 0 &&
         (arguments.combined_topk_indices == nullptr ||
          arguments.combined_x == nullptr)) ||
        arguments.prefix_per_rank == nullptr ||
        arguments.communication_buffer == nullptr ||
        arguments.workspace == nullptr ||
        (has_mode(tiling.mode_flags, CoreMode::kHybrid) &&
         arguments.route_record_count != 0 &&
         arguments.route_records == nullptr))
        return invalid("combine required argument is null");
    if ((arguments.topk_weights == nullptr) !=
        (arguments.combined_topk_weights == nullptr))
        return invalid("combine weights require both input and output");
    if (!is_aligned(arguments.communication_buffer) ||
        !is_aligned(arguments.workspace))
        return invalid("combine storage is misaligned");
    const int result = deep_ep_ascend_launch_combine(arguments, tiling, stream);
    return result == 0 ? CoreRuntimeStatus{} :
        launch_failure(result, "combine kernel launch failed");
}

}  // namespace deep_ep::ascend::elastic
