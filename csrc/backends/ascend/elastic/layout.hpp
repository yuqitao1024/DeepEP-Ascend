#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kAscendElasticAlignment = 32;
inline constexpr std::uint64_t kPublicElasticBufferAlignment = 2ULL << 20U;
inline constexpr std::uint32_t kSymmetricWindowAbiVersion = 4;
inline constexpr std::uint64_t kCombineControlSlotBytes =
    2 * sizeof(std::uint64_t);
inline constexpr std::uint64_t kCombineRecordHeaderBytes =
    2 * sizeof(std::uint32_t) + 4 * sizeof(std::int32_t);

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

constexpr bool is_single_rank_topology(const CoreTopology& topology) {
    return topology.world_rank == 0 && topology.world_size == 1 &&
           topology.scale_up_rank == 0 && topology.scale_up_size == 1 &&
           topology.scale_out_rank == 0 && topology.scale_out_size == 1;
}

constexpr bool is_scale_up_topology(const CoreTopology& topology) {
    return topology.world_size >= 2 && topology.world_rank >= 0 &&
           topology.world_rank < topology.world_size &&
           topology.scale_up_size == topology.world_size &&
           topology.scale_up_rank == topology.world_rank &&
           topology.scale_out_rank == 0 && topology.scale_out_size == 1;
}

constexpr bool checked_add(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* result) {
    if (result == nullptr ||
        rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
        return false;
    *result = lhs + rhs;
    return true;
}

constexpr bool checked_multiply(
    std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* result) {
    if (result == nullptr ||
        (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs))
        return false;
    *result = lhs * rhs;
    return true;
}

constexpr bool checked_align(
    std::uint64_t value, std::uint64_t alignment, std::uint64_t* result) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        return false;
    const auto mask = alignment - 1;
    std::uint64_t adjusted = 0;
    if (!checked_add(value, mask, &adjusted))
        return false;
    *result = adjusted & ~mask;
    return true;
}

constexpr bool checked_rank_slot_offset(
    std::uint64_t region_offset, std::uint64_t region_count, int rank,
    std::uint64_t* result) {
    if (result == nullptr || rank < 0 ||
        static_cast<std::uint64_t>(rank) >= region_count)
        return false;
    std::uint64_t rank_bytes = 0;
    return checked_multiply(
               static_cast<std::uint64_t>(rank), sizeof(std::uint64_t),
               &rank_bytes) &&
           checked_add(region_offset, rank_bytes, result);
}

struct TokenLayout {
    std::uint64_t hidden_offset = 0;
    std::uint64_t hidden_bytes = 0;
    std::uint64_t scale_factor_offset = 0;
    std::uint64_t scale_factor_bytes = 0;
    std::uint64_t topk_index_offset = 0;
    std::uint64_t topk_index_bytes = 0;
    std::uint64_t topk_weight_offset = 0;
    std::uint64_t topk_weight_bytes = 0;
    std::uint64_t source_metadata_offset = 0;
    std::uint64_t source_metadata_bytes = 0;
    std::uint64_t stride_bytes = 0;
};

struct WorkspaceLayout {
    std::uint64_t barrier_offset = 0;
    std::uint64_t barrier_bytes = 0;
    std::uint64_t block_count_offset = 0;
    std::uint64_t block_count_bytes = 0;
    std::uint64_t reduced_count_offset = 0;
    std::uint64_t reduced_count_bytes = 0;
    std::uint64_t prefix_sum_offset = 0;
    std::uint64_t prefix_sum_bytes = 0;
    std::uint64_t slot_offset = 0;
    std::uint64_t slot_bytes = 0;
    std::uint64_t source_metadata_offset = 0;
    std::uint64_t source_metadata_bytes = 0;
    std::uint64_t scratch_offset = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t scratch_status_offset = 0;
    std::uint64_t scratch_local_count_offset = 0;
    std::uint64_t scratch_rank_counts_offset = 0;
    std::uint64_t scratch_rank_values_offset = 0;
    std::uint64_t scratch_rank_indices_offset = 0;
    std::uint64_t scratch_rank_flags_offset = 0;
    std::uint64_t scratch_rank_count = 0;
    std::uint64_t total_bytes = 0;
};

struct alignas(32) SymmetricControlHeader {
    std::uint64_t dispatch_generation = 0;
    std::uint64_t combine_generation = 0;
    std::uint64_t reserved[2]{};
};

static_assert(sizeof(SymmetricControlHeader) == 32);

struct alignas(16) DispatchControlSlot {
    std::uint64_t generation = 0;
    std::uint64_t count = 0;
};

static_assert(sizeof(DispatchControlSlot) == 16);

enum class LayoutStatusCode : std::uint8_t {
    kSuccess,
    kInvalidArgument,
    kOverflow,
};

struct LayoutStatus {
    LayoutStatusCode code = LayoutStatusCode::kSuccess;
    const char* message = "";

    constexpr bool ok() const { return code == LayoutStatusCode::kSuccess; }

    static constexpr LayoutStatus invalid(const char* message) {
        return {LayoutStatusCode::kInvalidArgument, message};
    }

    static constexpr LayoutStatus overflow(const char* message) {
        return {LayoutStatusCode::kOverflow, message};
    }
};

struct SymmetricWindowInput {
    std::uint32_t world_size = 2;
    std::uint64_t num_max_tokens_per_rank = 0;
    std::uint64_t hidden = 0;
    std::uint64_t num_topk = 0;
    std::uint64_t element_bytes = 2;
    bool expanded = false;
    bool allow_multiple_reduction = false;
};

struct SymmetricWindowLayout {
    std::uint32_t abi_version = kSymmetricWindowAbiVersion;
    std::uint32_t struct_size = 0;
    std::uint64_t control_offset = 0;
    std::uint64_t control_bytes = 0;
    std::uint64_t dispatch_offset = 0;
    std::uint64_t dispatch_record_bytes = 0;
    // Existing ABI field: describes the Phase 2F staging-shard geometry.
    std::uint64_t dispatch_source_shard_bytes = 0;
    std::uint64_t dispatch_source_shard_count = 0;
    std::uint64_t dispatch_bytes = 0;
    std::uint64_t combine_offset = 0;
    std::uint64_t combine_record_bytes = 0;
    std::uint64_t combine_contributor_shard_bytes = 0;
    std::uint64_t combine_contributor_shard_count = 0;
    std::uint64_t combine_bytes = 0;
    std::uint64_t reserve_offset = 0;
    std::uint64_t reserve_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t dispatch_control_offset = 0;
    std::uint64_t dispatch_control_bytes = 0;
    std::uint64_t dispatch_receive_offset = 0;
    std::uint64_t dispatch_receive_shard_bytes = 0;
    std::uint64_t dispatch_receive_shard_count = 0;
    std::uint64_t dispatch_receive_bytes = 0;
    std::uint64_t dispatch_staging_offset = 0;
    std::uint64_t dispatch_staging_shard_bytes = 0;
    std::uint64_t dispatch_staging_shard_count = 0;
    std::uint64_t dispatch_staging_bytes = 0;
    std::uint64_t combine_control_offset = 0;
    std::uint64_t combine_control_bytes = 0;
    std::uint64_t combine_receive_offset = 0;
    std::uint64_t combine_receive_shard_bytes = 0;
    std::uint64_t combine_receive_shard_count = 0;
    std::uint64_t combine_receive_bytes = 0;
    std::uint64_t combine_staging_offset = 0;
    std::uint64_t combine_staging_shard_bytes = 0;
    std::uint64_t combine_staging_shard_count = 0;
    std::uint64_t combine_staging_bytes = 0;
    std::uint64_t combine_weight_offset = 0;
    std::uint64_t barrier_generation_offset = 0;
    std::uint64_t barrier_generation_bytes = 0;
    std::uint64_t barrier_generation_count = 0;
    std::uint64_t barrier_completion_offset = 0;
    std::uint64_t barrier_completion_bytes = 0;
    std::uint64_t barrier_completion_count = 0;
};

class LayoutBuilder {
public:
    constexpr bool append(
        std::uint64_t bytes, std::uint64_t* offset,
        std::uint64_t alignment = kAscendElasticAlignment) {
        std::uint64_t aligned_cursor = 0;
        if (offset == nullptr || !checked_align(cursor_, alignment, &aligned_cursor))
            return false;
        std::uint64_t next = 0;
        if (!checked_add(aligned_cursor, bytes, &next))
            return false;
        *offset = aligned_cursor;
        cursor_ = next;
        return true;
    }

    constexpr bool finish(
        std::uint64_t* total_bytes,
        std::uint64_t alignment = kAscendElasticAlignment) const {
        return checked_align(cursor_, alignment, total_bytes);
    }

private:
    std::uint64_t cursor_ = 0;
};

inline LayoutStatus build_symmetric_window_layout(
    const SymmetricWindowInput& input, SymmetricWindowLayout* output) {
    if (output == nullptr)
        return LayoutStatus::invalid("output must not be null");
    *output = {};
    if (input.world_size < 2)
        return LayoutStatus::invalid(
            "production layout requires at least two ranks");

    const bool barrier_only = input.num_max_tokens_per_rank == 0 &&
                              input.hidden == 0 && input.num_topk == 0;
    if (!barrier_only &&
        (input.num_max_tokens_per_rank == 0 || input.hidden == 0 ||
         input.element_bytes != 2))
        return LayoutStatus::invalid("invalid BF16 production layout shape");

    SymmetricWindowLayout layout{};
    layout.struct_size = sizeof(SymmetricWindowLayout);
    layout.dispatch_source_shard_count = input.world_size;
    layout.combine_contributor_shard_count = input.world_size;
    layout.barrier_generation_count = input.world_size;
    layout.barrier_completion_count = input.world_size;

    if (!checked_multiply(input.world_size, sizeof(std::uint64_t),
                          &layout.barrier_generation_bytes) ||
        !checked_multiply(input.world_size, sizeof(std::uint64_t),
                          &layout.barrier_completion_bytes) ||
        !checked_multiply(input.world_size, sizeof(DispatchControlSlot),
                          &layout.dispatch_control_bytes))
        return LayoutStatus::overflow("control region size overflow");
    LayoutBuilder control;
    std::uint64_t ignored_control_header_offset = 0;
    if (!control.append(
            sizeof(SymmetricControlHeader),
            &ignored_control_header_offset) ||
        !control.append(
            layout.barrier_generation_bytes,
            &layout.barrier_generation_offset) ||
        !control.append(
            layout.barrier_completion_bytes,
            &layout.barrier_completion_offset) ||
        !control.append(
            layout.dispatch_control_bytes,
            &layout.dispatch_control_offset) ||
        !control.finish(&layout.control_bytes))
        return LayoutStatus::overflow("control region layout overflow");

    if (!barrier_only) {
        const std::uint64_t effective_topk =
            input.num_topk == 0 ? input.world_size : input.num_topk;
        std::uint64_t hidden_bytes = 0;
        std::uint64_t topk_index_bytes = 0;
        std::uint64_t topk_weight_bytes = 0;
        std::uint64_t metadata_fields = 0;
        std::uint64_t metadata_bytes = 0;
        if (!checked_multiply(input.hidden, input.element_bytes,
                              &hidden_bytes) ||
            !checked_multiply(effective_topk, sizeof(std::int64_t),
                              &topk_index_bytes) ||
            !checked_multiply(effective_topk, sizeof(float),
                              &topk_weight_bytes) ||
            !checked_add(effective_topk, 2, &metadata_fields) ||
            !checked_multiply(metadata_fields, sizeof(std::int32_t),
                              &metadata_bytes))
            return LayoutStatus::overflow("dispatch record size overflow");

        LayoutBuilder dispatch_record;
        std::uint64_t ignored = 0;
        if (!dispatch_record.append(hidden_bytes, &ignored) ||
            !dispatch_record.append(topk_index_bytes, &ignored) ||
            !dispatch_record.append(topk_weight_bytes, &ignored) ||
            !dispatch_record.append(metadata_bytes, &ignored) ||
            !dispatch_record.finish(&layout.dispatch_record_bytes))
            return LayoutStatus::overflow("dispatch record layout overflow");

        if (!checked_multiply(input.num_max_tokens_per_rank,
                              layout.dispatch_record_bytes,
                              &layout.dispatch_receive_shard_bytes) ||
            !checked_align(layout.dispatch_receive_shard_bytes,
                           kAscendElasticAlignment,
                           &layout.dispatch_receive_shard_bytes) ||
            !checked_multiply(layout.dispatch_receive_shard_bytes,
                              input.world_size,
                              &layout.dispatch_receive_bytes) ||
            !checked_multiply(input.num_max_tokens_per_rank,
                              layout.dispatch_record_bytes,
                              &layout.dispatch_staging_shard_bytes) ||
            !checked_align(layout.dispatch_staging_shard_bytes,
                           kAscendElasticAlignment,
                           &layout.dispatch_staging_shard_bytes) ||
            !checked_multiply(layout.dispatch_staging_shard_bytes,
                              input.world_size,
                              &layout.dispatch_staging_bytes) ||
            !checked_add(layout.dispatch_receive_bytes,
                         layout.dispatch_staging_bytes,
                         &layout.dispatch_bytes))
            return LayoutStatus::overflow("dispatch region size overflow");
        layout.dispatch_receive_shard_count = input.world_size;
        layout.dispatch_staging_shard_count = input.world_size;
        layout.dispatch_source_shard_bytes =
            layout.dispatch_staging_shard_bytes;

        LayoutBuilder combine_record;
        if (!combine_record.append(hidden_bytes, &ignored) ||
            !combine_record.append(topk_weight_bytes,
                                   &layout.combine_weight_offset) ||
            !combine_record.append(kCombineRecordHeaderBytes, &ignored) ||
            !combine_record.finish(&layout.combine_record_bytes))
            return LayoutStatus::overflow("combine record layout overflow");

        std::uint64_t combine_capacity = input.num_max_tokens_per_rank;
        if (input.expanded && !input.allow_multiple_reduction &&
            !checked_multiply(combine_capacity, effective_topk,
                              &combine_capacity))
            return LayoutStatus::overflow("combine record capacity overflow");
        if (!checked_multiply(input.world_size, kCombineControlSlotBytes,
                              &layout.combine_control_bytes) ||
            !checked_multiply(combine_capacity, layout.combine_record_bytes,
                              &layout.combine_receive_shard_bytes) ||
            !checked_align(layout.combine_receive_shard_bytes,
                           kAscendElasticAlignment,
                           &layout.combine_receive_shard_bytes) ||
            !checked_multiply(layout.combine_receive_shard_bytes,
                              input.world_size, &layout.combine_receive_bytes) ||
            !checked_multiply(combine_capacity, layout.combine_record_bytes,
                              &layout.combine_staging_shard_bytes) ||
            !checked_align(layout.combine_staging_shard_bytes,
                           kAscendElasticAlignment,
                           &layout.combine_staging_shard_bytes) ||
            !checked_multiply(layout.combine_staging_shard_bytes,
                              input.world_size, &layout.combine_staging_bytes) ||
            !checked_add(layout.combine_control_bytes,
                         layout.combine_receive_bytes,
                         &layout.combine_bytes) ||
            !checked_add(layout.combine_bytes, layout.combine_staging_bytes,
                         &layout.combine_bytes))
            return LayoutStatus::overflow("combine region size overflow");
        layout.combine_contributor_shard_bytes =
            layout.combine_receive_shard_bytes;
        layout.combine_receive_shard_count = input.world_size;
        layout.combine_staging_shard_count = input.world_size;
    }

    layout.reserve_bytes = kAscendElasticAlignment;
    LayoutBuilder window;
    if (!window.append(layout.control_bytes, &layout.control_offset) ||
        !window.append(layout.dispatch_bytes, &layout.dispatch_offset) ||
        !window.append(layout.combine_bytes, &layout.combine_offset) ||
        !window.append(layout.reserve_bytes, &layout.reserve_offset) ||
        !window.finish(&layout.total_bytes, kPublicElasticBufferAlignment))
        return LayoutStatus::overflow("symmetric window size overflow");
    if (!checked_add(layout.control_offset, layout.barrier_generation_offset,
                     &layout.barrier_generation_offset) ||
        !checked_add(layout.control_offset, layout.barrier_completion_offset,
                     &layout.barrier_completion_offset) ||
        !checked_add(layout.control_offset, layout.dispatch_control_offset,
                     &layout.dispatch_control_offset) ||
        !checked_add(layout.dispatch_offset, layout.dispatch_receive_bytes,
                     &layout.dispatch_staging_offset) ||
        !checked_add(layout.combine_offset, layout.combine_control_bytes,
                     &layout.combine_receive_offset) ||
        !checked_add(layout.combine_receive_offset,
                     layout.combine_receive_bytes,
                     &layout.combine_staging_offset))
        return LayoutStatus::overflow("symmetric window offset overflow");
    layout.dispatch_receive_offset = layout.dispatch_offset;
    layout.combine_control_offset = layout.combine_offset;

    *output = layout;
    return {};
}

}  // namespace deep_ep::ascend::elastic
