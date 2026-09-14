#pragma once

#include "dispatch_vf_launchers.hpp"

inline constexpr DirectReleaseSegment host_dispatch_release_segment(
    DirectDispatchStage stage, bool profile_enabled) noexcept {
    if (stage == DirectDispatchStage::kFull)
        return DirectReleaseSegment::kAll;
    if (stage == DirectDispatchStage::kProducerRelease)
        return profile_enabled ? DirectReleaseSegment::kPayload :
                                 DirectReleaseSegment::kAll;
    if (profile_enabled &&
        stage == DirectDispatchStage::kProducerReleaseControl)
        return DirectReleaseSegment::kControl;
    if (profile_enabled &&
        stage == DirectDispatchStage::kProducerReleaseBarrier)
        return DirectReleaseSegment::kBarrier;
    return DirectReleaseSegment::kNone;
}

inline int launch_direct_dispatch_epilogue_acquire_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_acquire(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.prefix_per_rank,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        tiling.mode_flags,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_experts,
        tiling.expert_alignment,
        tiling.num_max_tokens_per_rank,
        tiling.dispatch_output_capacity,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_control_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_count,
        tiling.symmetric_window_layout.dispatch_receive_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_local_count_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.dispatch_route_source_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.workspace_layout.scratch_rank_count,
        tiling.workspace_layout.dispatch_error_offset,
        tiling.workspace_layout.dispatch_error_count,
        tiling.workspace_layout.dispatch_rank_bitmap_bytes,
        tiling.workspace_layout.dispatch_expert_bitmap_bytes,
        tiling.workspace_layout.dispatch_pipeline_offset,
        tiling.workspace_layout.dispatch_pipeline_bytes,
        tiling.token_layout.stride_bytes,
        (arguments.early_route_plan),
        (arguments.pipeline_source_chunk),
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_validate_records_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_validate_records(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.source_metadata,
        tiling.transport_context.local_window_base,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.mode_flags,
        arguments.generation,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.workspace_layout.dispatch_receive_tile_error_offset,
        tiling.workspace_layout.dispatch_receive_tile_count,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.source_metadata_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_reduce_errors_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_reduce_errors(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        static_cast<std::uint32_t>( DispatchProtocolStage::kEpilogue),
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_receive_tile_error_offset,
        0,
        tiling.workspace_layout.dispatch_receive_tile_count,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_count_experts_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_count_experts(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.local_window_base,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.dispatch_expert_tile_count_offset,
        tiling.workspace_layout.dispatch_expert_tile_count,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.topk_index_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_parallel_prefix_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_parallel_prefix(
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.prefix_per_rank,
        arguments.prefix_per_expert,
        arguments.unaligned_per_expert,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        tiling.num_experts,
        tiling.expert_alignment,
        tiling.dispatch_output_capacity,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.dispatch_expert_tile_count_offset,
        tiling.workspace_layout.dispatch_expert_tile_count,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_prefix_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_prefix(
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.prefix_per_rank,
        arguments.prefix_per_expert,
        arguments.unaligned_per_expert,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        tiling.mode_flags,
        arguments.generation,
        tiling.num_experts,
        tiling.expert_alignment,
        tiling.dispatch_output_capacity,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.dispatch_expert_tile_count_offset,
        tiling.workspace_layout.dispatch_expert_tile_count,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_metadata_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_metadata(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.recv_topk_indices,
        arguments.source_metadata,
        tiling.transport_context.local_window_base,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.mode_flags,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout .dispatch_receive_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.source_metadata_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_assign_destinations_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_assign_destinations(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.prefix_per_expert,
        arguments.unaligned_per_expert,
        arguments.source_metadata,
        tiling.transport_context.local_window_base,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.mode_flags,
        arguments.generation,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.dispatch_output_capacity,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout .dispatch_receive_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.workspace_layout.dispatch_expert_tile_count_offset,
        tiling.workspace_layout.dispatch_expert_tile_count,
        tiling.workspace_layout.dispatch_error_offset,
        tiling.workspace_layout.dispatch_expert_bitmap_offset,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.topk_index_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_reduce_errors_variant_1(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_reduce_errors(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        static_cast<std::uint32_t>( DispatchProtocolStage::kEpilogue),
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_error_offset,
        static_cast<std::uint64_t>( tiling.transport_context.topology.world_size),
        tiling.num_experts / static_cast<std::uint64_t>( tiling.transport_context.topology.world_size),
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_clear_padding_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_clear_padding(
        static_cast<std::uint8_t*>(arguments.workspace),
        static_cast<std::uint8_t*>(arguments.recv_x),
        static_cast<std::uint8_t*>(arguments.recv_scale_factors),
        arguments.recv_topk_weights,
        arguments.prefix_per_expert,
        tiling.mode_flags,
        tiling.num_experts,
        tiling.token_layout.hidden_bytes,
        tiling.token_layout.scale_factor_bytes,
        arguments.recv_scale_factor_token_stride,
        arguments.recv_scale_factor_pack_stride,
        tiling.workspace_layout.scratch_status_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_epilogue_copy_outputs_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_epilogue_copy_outputs(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        static_cast<std::uint8_t*>(arguments.recv_x),
        static_cast<std::uint8_t*>(arguments.recv_scale_factors),
        arguments.recv_topk_weights,
        arguments.source_metadata,
        tiling.transport_context.local_window_base,
        tiling.transport_context.topology.world_size,
        tiling.mode_flags,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout .dispatch_receive_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.hidden_offset,
        tiling.token_layout.hidden_bytes,
        stage == DirectDispatchStage::kEpilogueCopy ?
            tiling.token_layout.hidden_bytes -
                tiling.token_layout.hidden_bytes % 32 : 0,
        tiling.token_layout.scale_factor_offset,
        tiling.token_layout.scale_factor_bytes,
        arguments.recv_scale_factor_token_stride,
        arguments.recv_scale_factor_pack_stride,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.topk_weight_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_dispatch_producer_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_dispatch_producer(
        static_cast<const std::uint8_t*>(arguments.x),
        static_cast<const std::uint8_t*>(arguments.scale_factors),
        arguments.topk_indices,
        arguments.topk_weights,
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.destination_slots,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>(tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        tiling.mode_flags,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_tokens,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_control_bytes,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_count,
        tiling.symmetric_window_layout.dispatch_receive_bytes,
        tiling.symmetric_window_layout.dispatch_staging_offset,
        tiling.symmetric_window_layout.dispatch_staging_shard_bytes,
        tiling.symmetric_window_layout.dispatch_staging_shard_count,
        tiling.symmetric_window_layout.dispatch_staging_bytes,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_control_offset,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_control_bytes,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_shard_offset,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_shard_bytes,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_shard_count,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_bytes,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_staging_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_ingress_staging_shard_bytes,
        tiling.symmetric_window_layout .hybrid_dispatch_ingress_staging_shard_count,
        tiling.symmetric_window_layout.hybrid_dispatch_ingress_staging_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_local_count_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.workspace_layout.scratch_outbound_ingress_counts_offset,
        tiling.workspace_layout.scratch_outbound_ingress_count,
        tiling.workspace_layout.scratch_rank_indices_offset,
        tiling.workspace_layout.scratch_rank_flags_offset,
        tiling.workspace_layout.scratch_rank_count,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.hidden_offset,
        tiling.token_layout.hidden_bytes,
        tiling.token_layout.scale_factor_offset,
        tiling.token_layout.scale_factor_bytes,
        arguments.scale_factor_token_stride,
        arguments.scale_factor_pack_stride,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.topk_weight_offset,
        tiling.token_layout.source_metadata_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_producer_control_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_producer_control(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        tiling.num_experts,
        tiling.num_max_tokens_per_rank,
        tiling.dispatch_output_capacity,
        tiling.symmetric_window_layout.dispatch_control_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_count,
        tiling.symmetric_window_layout.dispatch_receive_bytes,
        tiling.symmetric_window_layout.dispatch_staging_shard_bytes,
        tiling.symmetric_window_layout.dispatch_staging_shard_count,
        tiling.symmetric_window_layout.dispatch_staging_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_local_count_offset,
        tiling.workspace_layout.scratch_rank_count,
        tiling.workspace_layout.dispatch_error_offset,
        tiling.workspace_layout.dispatch_error_count,
        tiling.workspace_layout.dispatch_rank_bitmap_bytes,
        tiling.workspace_layout.dispatch_expert_bitmap_bytes,
        tiling.token_layout.stride_bytes,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_producer_expert_count_init_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_producer_expert_count_init(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_group_expert_count_offset,
        tiling.workspace_layout.dispatch_group_expert_count,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_producer_plan_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_producer_plan(
        arguments.topk_indices,
        arguments.destination_slots,
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.mode_flags,
        arguments.generation,
        tiling.num_tokens,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        tiling.workspace_layout.dispatch_error_offset,
        tiling.workspace_layout.dispatch_rank_bitmap_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_reduce_errors_variant_2(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_reduce_errors(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        static_cast<std::uint32_t>( DispatchProtocolStage::kProducer),
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_error_offset,
        0,
        static_cast<std::uint64_t>( tiling.transport_context.topology.world_size),
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_reduce_errors_variant_3(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_reduce_errors(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        static_cast<std::uint32_t>( DispatchProtocolStage::kProducer),
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_error_offset,
        0,
        static_cast<std::uint64_t>( tiling.transport_context.topology.world_size),
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_publish_route_plan_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_publish_route_plan(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.symmetric_window_layout.dispatch_route_plan_offset,
        tiling.symmetric_window_layout .dispatch_route_plan_slot_bytes,
        tiling.symmetric_window_layout.dispatch_staging_offset,
        tiling.symmetric_window_layout.dispatch_staging_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_acquire_route_plan_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_acquire_route_plan(
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.prefix_per_rank,
        arguments.prefix_per_expert,
        arguments.unaligned_per_expert,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_experts,
        tiling.expert_alignment,
        tiling.num_max_tokens_per_rank,
        tiling.dispatch_output_capacity,
        tiling.symmetric_window_layout.dispatch_route_plan_offset,
        tiling.symmetric_window_layout .dispatch_route_plan_slot_bytes,
        tiling.symmetric_window_layout .dispatch_route_plan_expert_capacity,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_route_source_counts_offset,
        tiling.workspace_layout.scratch_rank_values_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_producer_record_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_producer_record(
        static_cast<const std::uint8_t*>(arguments.x),
        static_cast<const std::uint8_t*>(arguments.scale_factors),
        arguments.topk_indices,
        arguments.topk_weights,
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.destination_slots,
        tiling.transport_context.local_window_base,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.mode_flags,
        (stage == DirectDispatchStage::kProducerRecord) && (tiling.num_topk <= kTopkSubgroupWidth && tiling.transport_context.topology.world_size <= static_cast<int>(kTopkSubgroupWidth)) ? 1U : 0U,
        tiling.num_tokens,
        tiling.num_experts,
        tiling.num_topk,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_staging_offset,
        tiling.symmetric_window_layout.dispatch_staging_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.dispatch_group_owner_offset,
        tiling.workspace_layout.dispatch_group_tile_offset,
        tiling.workspace_layout.dispatch_group_tile_count,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.hidden_offset,
        tiling.token_layout.hidden_bytes,
        (stage == DirectDispatchStage::kProducerRecord) && (tiling.num_topk <= kTopkSubgroupWidth && tiling.transport_context.topology.world_size <= static_cast<int>(kTopkSubgroupWidth)) ? (arguments.token_fanout != 0 ? tiling.token_layout.hidden_bytes - tiling.token_layout.hidden_bytes % kDispatchTokenFanoutAlignmentBytes : tiling.token_layout.hidden_bytes - tiling.token_layout.hidden_bytes % kDispatchProducerVectorTileBytes) : 0,
        arguments.pipeline_chunk_begin,
        arguments.pipeline_chunk_end,
        arguments.pipeline_source_chunk,
        tiling.token_layout.scale_factor_offset,
        tiling.token_layout.scale_factor_bytes,
        arguments.scale_factor_token_stride,
        arguments.scale_factor_pack_stride,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.topk_weight_offset,
        tiling.token_layout.source_metadata_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_producer_release_variant_1(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_producer_release(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_staging_offset,
        tiling.symmetric_window_layout.dispatch_staging_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_local_count_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.dispatch_pipeline_offset,
        tiling.workspace_layout.dispatch_pipeline_bytes,
        tiling.workspace_layout.dispatch_group_tile_offset,
        tiling.workspace_layout.dispatch_group_tile_count,
        arguments.pipeline_chunk_begin,
        arguments.pipeline_chunk_end,
        arguments.pipeline_chunk_index,
        arguments.pipeline_final_chunk,
        arguments.pipeline_source_chunk,
        tiling.token_layout.stride_bytes,
        static_cast<std::uint32_t>(
            host_dispatch_release_segment(stage, profile_enabled)),
        stream, launch, tiling.launch.num_threads);
}

inline int launch_direct_dispatch_pipeline_wait_variant_1(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_direct_dispatch_pipeline_wait(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        tiling.workspace_layout.dispatch_pipeline_offset,
        arguments.pipeline_chunk_index,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_hybrid_dispatch_forward_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_hybrid_dispatch_forward(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_experts,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout .hybrid_dispatch_ingress_control_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_ingress_shard_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_ingress_shard_bytes,
        tiling.symmetric_window_layout .hybrid_dispatch_forward_control_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_forward_shard_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_forward_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.source_metadata_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_hybrid_dispatch_prepare_epilogue_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_hybrid_dispatch_prepare_epilogue(
        static_cast<std::uint8_t*>(arguments.workspace),
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>( tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_max_tokens_per_rank,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout .hybrid_dispatch_forward_control_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_forward_shard_offset,
        tiling.symmetric_window_layout .hybrid_dispatch_forward_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.token_layout.stride_bytes,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_dispatch_epilogue_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_vf_dispatch_epilogue(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        static_cast<std::uint8_t*>(arguments.recv_x),
        static_cast<std::uint8_t*>(arguments.recv_scale_factors),
        arguments.recv_topk_indices,
        arguments.recv_topk_weights,
        arguments.prefix_per_rank,
        arguments.prefix_per_expert,
        arguments.unaligned_per_expert,
        arguments.source_metadata,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>(tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        tiling.mode_flags,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_tokens,
        tiling.num_experts,
        tiling.num_topk,
        tiling.expert_alignment,
        tiling.num_max_tokens_per_rank,
        tiling.dispatch_output_capacity,
        tiling.symmetric_window_layout.control_offset,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_control_bytes,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_count,
        tiling.symmetric_window_layout.dispatch_receive_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_local_count_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_count,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.hidden_offset,
        tiling.token_layout.hidden_bytes,
        tiling.token_layout.scale_factor_offset,
        tiling.token_layout.scale_factor_bytes,
        arguments.recv_scale_factor_token_stride,
        arguments.recv_scale_factor_pack_stride,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.topk_weight_offset,
        tiling.token_layout.source_metadata_offset,
        has_mode(tiling.mode_flags, CoreMode::kCpuSync) ? 0U : 1U,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_hybrid_dispatch_record_routes_variant_0(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_hybrid_dispatch_record_routes(
        static_cast<std::uint8_t*>(arguments.workspace),
        arguments.route_records,
        tiling.transport_context.local_window_base,
        tiling.transport_context.backend_context,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.epoch,
        arguments.generation,
        tiling.num_experts,
        tiling.num_max_tokens_per_rank,
        arguments.route_record_capacity,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.source_metadata_offset,
        stream, launch, tiling.launch.num_threads);
}

inline int launch_dispatch_epilogue_variant_1(
    DispatchArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectDispatchStage stage,
    std::uint32_t copy_outputs, bool profile_enabled) {
    (void)stage;
    (void)copy_outputs;
    (void)profile_enabled;
    return deep_ep_ascend_launch_vf_dispatch_epilogue(
        static_cast<std::uint8_t*>(arguments.communication_buffer),
        static_cast<std::uint8_t*>(arguments.workspace),
        static_cast<std::uint8_t*>(arguments.recv_x),
        static_cast<std::uint8_t*>(arguments.recv_scale_factors),
        arguments.recv_topk_indices,
        arguments.recv_topk_weights,
        arguments.prefix_per_rank,
        arguments.prefix_per_expert,
        arguments.unaligned_per_expert,
        arguments.source_metadata,
        tiling.transport_context.abi_version,
        tiling.transport_context.struct_size,
        tiling.transport_context.capabilities,
        tiling.transport_context.local_window_base,
        tiling.transport_context.channel_table,
        tiling.transport_context.peer_address_table,
        tiling.transport_context.topology.abi_version,
        tiling.transport_context.topology.struct_size,
        tiling.transport_context.topology.world_rank,
        tiling.transport_context.topology.world_size,
        tiling.transport_context.topology.scale_up_rank,
        tiling.transport_context.topology.scale_up_size,
        tiling.transport_context.topology.scale_out_rank,
        tiling.transport_context.topology.scale_out_size,
        static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct),
        static_cast<std::uint32_t>(tiling.transport_context.topology.kind),
        tiling.transport_context.topology.epoch,
        tiling.transport_context.backend_context,
        tiling.mode_flags,
        arguments.generation,
        arguments.timeout_cycles,
        tiling.num_tokens,
        tiling.num_experts,
        tiling.num_topk,
        tiling.expert_alignment,
        tiling.num_max_tokens_per_rank,
        tiling.dispatch_output_capacity,
        tiling.symmetric_window_layout.control_offset,
        tiling.symmetric_window_layout.dispatch_control_offset,
        tiling.symmetric_window_layout.dispatch_control_bytes,
        tiling.symmetric_window_layout.dispatch_receive_offset,
        tiling.symmetric_window_layout.dispatch_receive_shard_bytes,
        tiling.symmetric_window_layout.dispatch_receive_shard_count,
        tiling.symmetric_window_layout.dispatch_receive_bytes,
        tiling.workspace_layout.scratch_status_offset,
        tiling.workspace_layout.scratch_local_count_offset,
        tiling.workspace_layout.scratch_rank_counts_offset,
        tiling.workspace_layout.scratch_rank_count,
        tiling.token_layout.stride_bytes,
        tiling.token_layout.hidden_offset,
        tiling.token_layout.hidden_bytes,
        tiling.token_layout.scale_factor_offset,
        tiling.token_layout.scale_factor_bytes,
        arguments.recv_scale_factor_token_stride,
        arguments.recv_scale_factor_pack_stride,
        tiling.token_layout.topk_index_offset,
        tiling.token_layout.topk_weight_offset,
        tiling.token_layout.source_metadata_offset,
        1U,
        stream, launch, tiling.launch.num_threads);
}
