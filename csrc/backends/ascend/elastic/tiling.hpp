#pragma once

#include <cstdint>

#include "layout.hpp"
#include "../transport/types.hpp"

namespace deep_ep::ascend::elastic {

enum class OperationKind : std::uint8_t { kBarrier, kDispatch, kCombine };
enum class ElementKind : std::uint8_t { kBFloat16, kFloat8E4M3 };
enum class CoreMode : std::uint8_t {
    kCached,
    kExpanded,
    kZeroPadding,
    kAllowMultipleReduction,
    kAsyncEvent,
    kCpuSync,
    kHybrid,
    kPipeline,
    kEngram,
};

using CoreModeFlags = std::uint32_t;

constexpr CoreModeFlags mode_bit(CoreMode mode) {
    return CoreModeFlags{1} << static_cast<std::uint8_t>(mode);
}

constexpr bool has_mode(CoreModeFlags flags, CoreMode mode) {
    return (flags & mode_bit(mode)) != 0;
}

struct CoreTopology {
    int world_rank = 0;
    int world_size = 1;
    int scale_up_rank = 0;
    int scale_up_size = 1;
    int scale_out_rank = 0;
    int scale_out_size = 1;
};

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

inline constexpr std::uint32_t kCoreTilingAbiVersion = 1;

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
    std::uint64_t num_scale_factor_packs = 0;
    std::uint64_t scale_factor_pack_bytes = 0;
    CoreTopology topology{};
    CoreLaunchShape launch{};
    TokenLayout token_layout{};
    WorkspaceLayout workspace_layout{};
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
    return topology.world_size > 0 && topology.scale_up_size > 0 &&
           topology.scale_out_size > 0 && topology.world_rank >= 0 &&
           topology.world_rank < topology.world_size &&
           topology.scale_up_rank >= 0 &&
           topology.scale_up_rank < topology.scale_up_size &&
           topology.scale_out_rank >= 0 &&
           topology.scale_out_rank < topology.scale_out_size;
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
    layout.scratch_bytes = kAscendElasticAlignment;
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
    *output = layout;
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
        !detail::build_workspace_layout(input, &tiling.workspace_layout))
        return TilingStatus::overflow("layout size overflow");

    std::uint64_t token_capacity = input.num_max_tokens_per_rank;
    if (has_mode(input.mode_flags, CoreMode::kExpanded) &&
        !checked_multiply(token_capacity, input.num_topk, &token_capacity))
        return TilingStatus::overflow("token capacity overflow");
    if (!checked_multiply(token_capacity, tiling.token_layout.stride_bytes,
                          &tiling.communication_buffer_bytes))
        return TilingStatus::overflow("communication buffer overflow");

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
    *output = tiling;
    return {};
}

inline TilingStatus validate_single_rank(const CoreTiling& tiling) {
    if (tiling.topology.world_size != 1 ||
        tiling.topology.scale_up_size != 1 ||
        tiling.topology.scale_out_size != 1)
        return {TilingStatusCode::kUnsupportedTopology,
                "internal launch supports only one rank"};
    return {};
}

}  // namespace deep_ep::ascend::elastic
