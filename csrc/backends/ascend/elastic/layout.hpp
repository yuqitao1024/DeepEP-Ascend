#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kAscendElasticAlignment = 32;
inline constexpr std::uint64_t kPublicElasticBufferAlignment = 2ULL << 20U;
inline constexpr std::uint32_t kSymmetricWindowAbiVersion = 1;

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
    std::uint64_t total_bytes = 0;
};

struct alignas(32) SymmetricControlHeader {
    std::uint64_t dispatch_generation = 0;
    std::uint64_t combine_generation = 0;
    std::uint64_t barrier_generation[2]{};
    std::uint64_t barrier_completion[2]{};
    std::uint64_t reserved[2]{};
};

static_assert(sizeof(SymmetricControlHeader) == 64);

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
};

struct SymmetricWindowLayout {
    std::uint32_t abi_version = kSymmetricWindowAbiVersion;
    std::uint32_t struct_size = 0;
    std::uint64_t control_offset = 0;
    std::uint64_t control_bytes = 0;
    std::uint64_t dispatch_offset = 0;
    std::uint64_t dispatch_record_bytes = 0;
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
    if (input.world_size != 2)
        return LayoutStatus::invalid("production layout requires two ranks");

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

    std::uint64_t count_record_bytes = 0;
    if (!checked_multiply(input.world_size, 4 * sizeof(std::uint64_t),
                          &count_record_bytes) ||
        !checked_add(sizeof(SymmetricControlHeader), count_record_bytes,
                     &layout.control_bytes))
        return LayoutStatus::overflow("control region size overflow");

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

        std::uint64_t dispatch_slots = 0;
        if (!checked_multiply(input.num_max_tokens_per_rank, effective_topk,
                              &dispatch_slots) ||
            !checked_multiply(dispatch_slots, layout.dispatch_record_bytes,
                              &layout.dispatch_source_shard_bytes) ||
            !checked_align(layout.dispatch_source_shard_bytes,
                           kAscendElasticAlignment,
                           &layout.dispatch_source_shard_bytes) ||
            !checked_multiply(layout.dispatch_source_shard_bytes,
                              input.world_size, &layout.dispatch_bytes))
            return LayoutStatus::overflow("dispatch region size overflow");

        LayoutBuilder combine_record;
        if (!combine_record.append(hidden_bytes, &ignored) ||
            !combine_record.append(topk_weight_bytes, &ignored) ||
            !combine_record.append(4 * sizeof(std::int32_t), &ignored) ||
            !combine_record.finish(&layout.combine_record_bytes) ||
            !checked_multiply(input.num_max_tokens_per_rank,
                              layout.combine_record_bytes,
                              &layout.combine_contributor_shard_bytes) ||
            !checked_align(layout.combine_contributor_shard_bytes,
                           kAscendElasticAlignment,
                           &layout.combine_contributor_shard_bytes) ||
            !checked_multiply(layout.combine_contributor_shard_bytes,
                              input.world_size, &layout.combine_bytes))
            return LayoutStatus::overflow("combine region size overflow");
    }

    layout.reserve_bytes = kAscendElasticAlignment;
    LayoutBuilder window;
    if (!window.append(layout.control_bytes, &layout.control_offset) ||
        !window.append(layout.dispatch_bytes, &layout.dispatch_offset) ||
        !window.append(layout.combine_bytes, &layout.combine_offset) ||
        !window.append(layout.reserve_bytes, &layout.reserve_offset) ||
        !window.finish(&layout.total_bytes, kPublicElasticBufferAlignment))
        return LayoutStatus::overflow("symmetric window size overflow");

    *output = layout;
    return {};
}

}  // namespace deep_ep::ascend::elastic
