#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kAscendElasticAlignment = 32;

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

    constexpr bool finish(std::uint64_t* total_bytes) const {
        return checked_align(cursor_, kAscendElasticAlignment, total_bytes);
    }

private:
    std::uint64_t cursor_ = 0;
};

}  // namespace deep_ep::ascend::elastic
