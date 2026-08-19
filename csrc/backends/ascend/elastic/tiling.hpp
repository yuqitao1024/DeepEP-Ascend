#pragma once

#include <cstdint>

#include "layout.hpp"
#include "../transport/types.hpp"

namespace deep_ep::ascend::elastic {

enum class OperationKind : std::uint8_t { kBarrier, kDispatch, kCombine };
enum class ElementKind : std::uint8_t { kBFloat16, kFloat8E4M3 };
inline constexpr transport::TransportCapabilities
    kBarrierTransportCapabilities =
        transport::capability_bit(
            transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(
            transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(
            transport::TransportCapability::kScaleUpTeam);

struct CoreLaunchShape {
    std::uint32_t num_blocks = 1;
    std::uint32_t num_threads = 512;
    std::uint32_t dynamic_ub_bytes = 0;
};

struct CoreTilingInput {
    OperationKind operation = OperationKind::kBarrier;
    ElementKind element_kind = ElementKind::kBFloat16;
    CoreModeFlags mode_flags = 0;
    std::uint64_t num_tokens = 0;
    std::uint64_t hidden = 0;
    std::uint64_t num_experts = 0;
    std::uint64_t num_topk = 0;
    std::uint64_t expert_alignment = 1;
    std::uint64_t num_max_tokens_per_rank = 0;
    std::uint64_t num_scale_factor_packs = 0;
    std::uint64_t scale_factor_pack_bytes = 0;
    bool has_reusable_slots = false;
    CoreTopology topology{};
};

inline constexpr std::uint32_t kCoreTilingAbiVersion = 11;

struct CoreTiling {
    std::uint32_t abi_version = kCoreTilingAbiVersion;
    std::uint32_t struct_size = 0;
    OperationKind operation = OperationKind::kBarrier;
    ElementKind element_kind = ElementKind::kBFloat16;
    CoreModeFlags mode_flags = 0;
    std::uint64_t num_tokens = 0;
    std::uint64_t hidden = 0;
    std::uint64_t num_experts = 0;
    std::uint64_t num_topk = 0;
    std::uint64_t expert_alignment = 1;
    std::uint64_t num_max_tokens_per_rank = 0;
    std::uint64_t dispatch_output_capacity = 0;
    std::uint64_t hybrid_route_capacity = 0;
    std::uint64_t num_scale_factor_packs = 0;
    std::uint64_t scale_factor_pack_bytes = 0;
    CoreTopology topology{};
    CoreLaunchShape launch{};
    TokenLayout token_layout{};
    WorkspaceLayout workspace_layout{};
    SymmetricWindowLayout symmetric_window_layout{};
    std::uint64_t communication_buffer_bytes = 0;
    std::uint64_t workspace_bytes = 0;
    transport::DeviceTransportContext transport_context{};
};

enum class TilingStatusCode : std::uint8_t {
    kSuccess,
    kInvalidArgument,
    kInvalidMode,
    kUnsupportedTopology,
    kOverflow,
};

struct TilingStatus {
    TilingStatusCode code = TilingStatusCode::kSuccess;
    const char* message = "";

    constexpr bool ok() const { return code == TilingStatusCode::kSuccess; }

    static constexpr TilingStatus invalid(const char* message) {
        return {TilingStatusCode::kInvalidArgument, message};
    }
    static constexpr TilingStatus invalid_mode(const char* message) {
        return {TilingStatusCode::kInvalidMode, message};
    }
    static constexpr TilingStatus overflow(const char* message) {
        return {TilingStatusCode::kOverflow, message};
    }
};

namespace detail {

inline bool valid_topology(const CoreTopology& topology) {
    const bool dimensions_valid =
        topology.world_size > 0 && topology.scale_up_size > 0 &&
        topology.scale_out_size > 0 && topology.world_rank >= 0 &&
        topology.world_rank < topology.world_size &&
        topology.scale_up_rank >= 0 &&
        topology.scale_up_rank < topology.scale_up_size &&
        topology.scale_out_rank >= 0 &&
        topology.scale_out_rank < topology.scale_out_size &&
        topology.epoch > 0 &&
        static_cast<std::int64_t>(topology.scale_up_size) *
                topology.scale_out_size ==
            topology.world_size &&
        topology.scale_up_rank ==
            topology.world_rank % topology.scale_up_size &&
        topology.scale_out_rank ==
            topology.world_rank / topology.scale_up_size;
    if (!dimensions_valid)
        return false;
    if (topology.kind == transport::TransportTopologyKind::kFlatScaleUp)
        return topology.scale_up_size == topology.world_size &&
               topology.scale_out_size == 1;
    return (topology.kind == transport::TransportTopologyKind::kPhysical2D ||
            topology.kind ==
                transport::TransportTopologyKind::kLogicalSimulation) &&
           topology.scale_out_size >= 2;
}

inline bool build_token_layout(
    const CoreTilingInput& input, std::uint64_t element_bytes,
    TokenLayout* output) {
    TokenLayout layout{};
    LayoutBuilder builder;
    if (!checked_multiply(input.hidden, element_bytes, &layout.hidden_bytes) ||
        !builder.append(layout.hidden_bytes, &layout.hidden_offset) ||
        !checked_multiply(input.num_scale_factor_packs,
                          input.scale_factor_pack_bytes,
                          &layout.scale_factor_bytes) ||
        !builder.append(layout.scale_factor_bytes,
                        &layout.scale_factor_offset) ||
        !checked_multiply(input.num_topk, sizeof(std::int64_t),
                          &layout.topk_index_bytes) ||
        !builder.append(layout.topk_index_bytes, &layout.topk_index_offset) ||
        !checked_multiply(input.num_topk, sizeof(float),
                          &layout.topk_weight_bytes) ||
        !builder.append(layout.topk_weight_bytes,
                        &layout.topk_weight_offset))
        return false;
    std::uint64_t metadata_fields = 0;
    if (!checked_add(input.num_topk, 2, &metadata_fields) ||
        !checked_multiply(metadata_fields, sizeof(std::int32_t),
                          &layout.source_metadata_bytes) ||
        !builder.append(layout.source_metadata_bytes,
                        &layout.source_metadata_offset) ||
        !builder.finish(&layout.stride_bytes))
        return false;
    *output = layout;
    return true;
}

inline bool build_workspace_layout(
    const CoreTilingInput& input, WorkspaceLayout* output) {
    WorkspaceLayout layout{};
    LayoutBuilder builder;
    layout.barrier_bytes = 2 * sizeof(std::uint64_t);
    std::uint64_t rank_expert_fields = 0;
    std::uint64_t prefix_fields = 0;
    if (!checked_add(static_cast<std::uint64_t>(input.topology.world_size),
                     input.num_experts, &rank_expert_fields) ||
        !checked_add(rank_expert_fields, 1, &prefix_fields) ||
        !checked_multiply(rank_expert_fields, sizeof(std::int64_t),
                          &layout.block_count_bytes))
        return false;
    layout.reduced_count_bytes = layout.block_count_bytes;
    if (!checked_multiply(prefix_fields, sizeof(std::int32_t),
                          &layout.prefix_sum_bytes) ||
        !checked_multiply(input.num_tokens, input.num_topk,
                          &layout.slot_bytes) ||
        !checked_multiply(layout.slot_bytes, sizeof(std::int32_t),
                          &layout.slot_bytes))
        return false;
    std::uint64_t metadata_fields = 0;
    if (!checked_add(input.num_topk, 2, &metadata_fields) ||
        !checked_multiply(input.num_tokens, metadata_fields,
                          &layout.source_metadata_bytes) ||
        !checked_multiply(layout.source_metadata_bytes,
                          sizeof(std::int32_t),
                          &layout.source_metadata_bytes))
        return false;
    layout.scratch_rank_count =
        static_cast<std::uint64_t>(input.topology.world_size);
    layout.scratch_outbound_ingress_count =
        has_mode(input.mode_flags, CoreMode::kHybrid) ?
            layout.scratch_rank_count : 0;
    std::uint64_t rank_u64_bytes = 0;
    std::uint64_t outbound_ingress_count_bytes = 0;
    std::uint64_t rank_i32_bytes = 0;
    std::uint64_t scratch_cursor = 0;
    if (!checked_multiply(
            layout.scratch_rank_count, sizeof(std::uint64_t),
            &rank_u64_bytes) ||
        !checked_multiply(
            layout.scratch_rank_count, sizeof(std::int32_t),
            &rank_i32_bytes) ||
        !checked_multiply(
            layout.scratch_outbound_ingress_count, sizeof(std::uint64_t),
            &outbound_ingress_count_bytes) ||
        !checked_add(scratch_cursor, sizeof(std::uint64_t),
                     &scratch_cursor))
        return false;
    layout.scratch_local_count_offset = scratch_cursor;
    if (!checked_add(scratch_cursor, sizeof(std::uint64_t),
                     &scratch_cursor))
        return false;
    layout.scratch_rank_counts_offset = scratch_cursor;
    if (!checked_add(scratch_cursor, rank_u64_bytes, &scratch_cursor))
        return false;
    layout.scratch_rank_values_offset = scratch_cursor;
    if (!checked_add(scratch_cursor, rank_u64_bytes, &scratch_cursor))
        return false;
    if (layout.scratch_outbound_ingress_count != 0) {
        layout.scratch_outbound_ingress_counts_offset = scratch_cursor;
        if (!checked_add(
                scratch_cursor, outbound_ingress_count_bytes,
                &scratch_cursor))
            return false;
    }
    layout.scratch_rank_indices_offset = scratch_cursor;
    if (!checked_add(scratch_cursor, rank_i32_bytes, &scratch_cursor))
        return false;
    layout.scratch_rank_flags_offset = scratch_cursor;
    if (!checked_add(
            scratch_cursor, layout.scratch_rank_count, &scratch_cursor) ||
        !checked_align(
            scratch_cursor, kAscendElasticAlignment,
            &layout.scratch_bytes))
        return false;
    if (!builder.append(layout.barrier_bytes, &layout.barrier_offset) ||
        !builder.append(layout.block_count_bytes, &layout.block_count_offset) ||
        !builder.append(layout.reduced_count_bytes,
                        &layout.reduced_count_offset) ||
        !builder.append(layout.prefix_sum_bytes, &layout.prefix_sum_offset) ||
        !builder.append(layout.slot_bytes, &layout.slot_offset) ||
        !builder.append(layout.source_metadata_bytes,
                        &layout.source_metadata_offset) ||
        !builder.append(layout.scratch_bytes, &layout.scratch_offset) ||
        !builder.finish(&layout.total_bytes))
        return false;
    layout.scratch_status_offset = layout.scratch_offset;
    if (!checked_add(
            layout.scratch_offset, layout.scratch_local_count_offset,
            &layout.scratch_local_count_offset) ||
        !checked_add(
            layout.scratch_offset, layout.scratch_rank_counts_offset,
            &layout.scratch_rank_counts_offset) ||
        !checked_add(
            layout.scratch_offset, layout.scratch_rank_values_offset,
            &layout.scratch_rank_values_offset) ||
        (layout.scratch_outbound_ingress_count != 0 &&
         !checked_add(
             layout.scratch_offset,
             layout.scratch_outbound_ingress_counts_offset,
             &layout.scratch_outbound_ingress_counts_offset)) ||
        !checked_add(
            layout.scratch_offset, layout.scratch_rank_indices_offset,
            &layout.scratch_rank_indices_offset) ||
        !checked_add(
            layout.scratch_offset, layout.scratch_rank_flags_offset,
            &layout.scratch_rank_flags_offset))
        return false;
    *output = layout;
    return true;
}

inline bool build_dispatch_output_capacity(
    const CoreTilingInput& input, std::uint64_t* output) {
    if (output == nullptr)
        return false;
    *output = 0;
    if (input.operation != OperationKind::kDispatch &&
        input.operation != OperationKind::kCombine)
        return true;

    std::uint64_t raw_lane_count = 0;
    const std::uint64_t padding_per_expert = input.expert_alignment - 1;
    std::uint64_t total_padding = 0;
    std::uint64_t padded_lane_count = 0;
    std::uint64_t adjusted_lane_count = 0;
    const auto local_experts = input.num_experts /
        static_cast<std::uint64_t>(input.topology.world_size);
    if (!checked_multiply(
            input.num_max_tokens_per_rank,
            static_cast<std::uint64_t>(input.topology.world_size),
            &raw_lane_count) ||
        !checked_multiply(raw_lane_count, input.num_topk, &raw_lane_count) ||
        !checked_multiply(
            padding_per_expert, local_experts, &total_padding) ||
        !checked_add(raw_lane_count, total_padding, &padded_lane_count) ||
        !checked_add(
            padded_lane_count, padding_per_expert, &adjusted_lane_count))
        return false;
    *output = (adjusted_lane_count / input.expert_alignment) *
        input.expert_alignment;
    return true;
}

}  // namespace detail

inline TilingStatus build_core_tiling(
    const CoreTilingInput& input, CoreTiling* output) {
    if (output == nullptr)
        return TilingStatus::invalid("output must not be null");
    *output = {};
    if (!detail::valid_topology(input.topology))
        return TilingStatus::invalid("invalid topology");
    const bool requires_token_shape =
        input.operation != OperationKind::kBarrier;
    if (requires_token_shape &&
        (input.hidden == 0 || input.num_experts == 0 ||
         input.num_topk == 0 || input.expert_alignment == 0 ||
         input.num_max_tokens_per_rank == 0 ||
         input.num_topk > input.num_experts))
        return TilingStatus::invalid("invalid shape");
    if (input.operation == OperationKind::kDispatch &&
        input.num_tokens > input.num_max_tokens_per_rank)
        return TilingStatus::invalid(
            "dispatch token count exceeds shard capacity");
    if (input.operation == OperationKind::kCombine &&
        input.num_tokens > input.num_max_tokens_per_rank)
        return TilingStatus::invalid(
            "combine token count exceeds shard capacity");
    if (input.num_experts %
            static_cast<std::uint64_t>(input.topology.world_size) != 0)
        return TilingStatus::invalid("experts must divide ranks");
    if (has_mode(input.mode_flags, CoreMode::kZeroPadding) &&
        !has_mode(input.mode_flags, CoreMode::kExpanded))
        return TilingStatus::invalid_mode(
            "zero padding requires expanded mode");
    if (has_mode(input.mode_flags, CoreMode::kCached) &&
        !input.has_reusable_slots)
        return TilingStatus::invalid_mode(
            "cached mode requires reusable slots");
    if (requires_token_shape &&
        input.element_kind == ElementKind::kFloat8E4M3 &&
        (input.num_scale_factor_packs == 0 ||
         input.scale_factor_pack_bytes == 0))
        return TilingStatus::invalid(
            "FP8 dispatch requires scale factor packs");

    const std::uint64_t element_bytes =
        input.element_kind == ElementKind::kBFloat16 ? 2 : 1;
    CoreTiling tiling{};
    if (!detail::build_token_layout(
            input, element_bytes, &tiling.token_layout) ||
        !detail::build_workspace_layout(input, &tiling.workspace_layout) ||
        !detail::build_dispatch_output_capacity(
            input, &tiling.dispatch_output_capacity))
        return TilingStatus::overflow("layout size overflow");
    if (has_mode(input.mode_flags, CoreMode::kHybrid) &&
        !checked_multiply(
            input.num_max_tokens_per_rank,
            static_cast<std::uint64_t>(input.topology.world_size),
            &tiling.hybrid_route_capacity))
        return TilingStatus::overflow("hybrid route capacity overflow");

    if (!is_single_rank_topology(input.topology)) {
        SymmetricWindowInput window_input{};
        window_input.world_size =
            static_cast<std::uint32_t>(input.topology.world_size);
        window_input.num_max_tokens_per_rank = input.num_max_tokens_per_rank;
        window_input.hidden = input.hidden;
        window_input.num_topk = input.num_topk;
        window_input.element_bytes = element_bytes;
        window_input.expanded =
            has_mode(input.mode_flags, CoreMode::kExpanded);
        window_input.allow_multiple_reduction = has_mode(
            input.mode_flags, CoreMode::kAllowMultipleReduction);
        window_input.hybrid = has_mode(input.mode_flags, CoreMode::kHybrid);
        window_input.hybrid_route_capacity = tiling.hybrid_route_capacity;
        const auto layout_status = build_symmetric_window_layout(
            window_input, &tiling.symmetric_window_layout);
        if (!layout_status.ok())
            return layout_status.code == LayoutStatusCode::kOverflow ?
                TilingStatus::overflow(layout_status.message) :
                TilingStatus::invalid(layout_status.message);
        tiling.communication_buffer_bytes =
            tiling.symmetric_window_layout.total_bytes;
    } else {
        std::uint64_t token_capacity = input.num_max_tokens_per_rank;
        if (has_mode(input.mode_flags, CoreMode::kExpanded) &&
            !checked_multiply(token_capacity, input.num_topk, &token_capacity))
            return TilingStatus::overflow("token capacity overflow");
        if (!checked_multiply(token_capacity, tiling.token_layout.stride_bytes,
                              &tiling.communication_buffer_bytes))
            return TilingStatus::overflow("communication buffer overflow");
    }

    tiling.abi_version = kCoreTilingAbiVersion;
    tiling.struct_size = sizeof(CoreTiling);
    tiling.operation = input.operation;
    tiling.element_kind = input.element_kind;
    tiling.mode_flags = input.mode_flags;
    tiling.num_tokens = input.num_tokens;
    tiling.hidden = input.hidden;
    tiling.num_experts = input.num_experts;
    tiling.num_topk = input.num_topk;
    tiling.expert_alignment = input.expert_alignment;
    tiling.num_max_tokens_per_rank = input.num_max_tokens_per_rank;
    tiling.num_scale_factor_packs = input.num_scale_factor_packs;
    tiling.scale_factor_pack_bytes = input.scale_factor_pack_bytes;
    tiling.topology = input.topology;
    tiling.workspace_bytes = tiling.workspace_layout.total_bytes;
    tiling.transport_context = transport::make_device_transport_context();
    tiling.transport_context.topology.world_rank = input.topology.world_rank;
    tiling.transport_context.topology.world_size = input.topology.world_size;
    tiling.transport_context.topology.scale_up_rank =
        input.topology.scale_up_rank;
    tiling.transport_context.topology.scale_up_size =
        input.topology.scale_up_size;
    tiling.transport_context.topology.scale_out_rank =
        input.topology.scale_out_rank;
    tiling.transport_context.topology.scale_out_size =
        input.topology.scale_out_size;
    tiling.transport_context.topology.struct_size =
        sizeof(transport::TransportTopology);
    tiling.transport_context.topology.kind = input.topology.kind;
    tiling.transport_context.topology.epoch = input.topology.epoch;
    *output = tiling;
    return {};
}

inline TilingStatus validate_single_rank(const CoreTiling& tiling) {
    if (!is_single_rank_topology(tiling.topology))
        return {TilingStatusCode::kUnsupportedTopology,
                "internal launch supports only one rank"};
    return {};
}

}  // namespace deep_ep::ascend::elastic
