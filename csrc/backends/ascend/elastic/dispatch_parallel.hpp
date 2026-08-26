#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kDispatchBitmapWordBits = 64;
inline constexpr std::uint64_t kDispatchReceiveRecordsPerTile = 128;

#ifndef DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE
#define DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE inline constexpr
#define DEEP_EP_ASCEND_DISPATCH_PARALLEL_AICORE_CALLEE_LOCAL 1
#endif

struct DispatchConsumerCopyPlan {
    std::uint64_t vector_bytes = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t scalar_begin = 0;
    bool valid = false;
};

struct DispatchExpertPrefixWorkerPlan {
    std::uint32_t local_experts = 0;
    std::uint32_t active_threads = 0;
    bool valid = false;
};

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE DispatchExpertPrefixWorkerPlan
dispatch_expert_prefix_worker_plan(
    std::uint64_t num_experts, std::uint64_t world_size,
    std::uint64_t num_threads) noexcept {
    DispatchExpertPrefixWorkerPlan plan{};
    if (num_experts == 0 || world_size == 0 || num_threads == 0 ||
        num_experts % world_size != 0)
        return plan;
    const std::uint64_t local_experts = num_experts / world_size;
    if (local_experts == 0 || local_experts > num_threads ||
        local_experts > std::numeric_limits<std::uint32_t>::max())
        return plan;
    plan.local_experts = static_cast<std::uint32_t>(local_experts);
    plan.active_threads = plan.local_experts;
    plan.valid = true;
    return plan;
}

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE DispatchConsumerCopyPlan
dispatch_consumer_copy_plan(
    std::uint64_t hidden_bytes, std::uint64_t tile_bytes,
    std::uint64_t alignment_bytes) noexcept {
    DispatchConsumerCopyPlan plan{};
    if (tile_bytes == 0 || alignment_bytes == 0 ||
        (tile_bytes & (tile_bytes - 1)) != 0 ||
        tile_bytes % alignment_bytes != 0)
        return plan;
    plan.vector_bytes = hidden_bytes - hidden_bytes % alignment_bytes;
    plan.scalar_begin = plan.vector_bytes;
    plan.tile_count = plan.vector_bytes / tile_bytes +
        (plan.vector_bytes % tile_bytes != 0 ? 1 : 0);
    plan.valid = true;
    return plan;
}

constexpr bool direct_dispatch_cached_bitmap_owner(
    std::uint32_t block_index) noexcept {
    return block_index == 0;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline DispatchExpertPrefixWorkerPlan
dispatch_simt_expert_prefix_worker_plan(
    std::uint64_t num_experts, std::uint64_t world_size,
    std::uint64_t num_threads) noexcept {
    DispatchExpertPrefixWorkerPlan plan{};
    if (num_experts == 0 || world_size == 0 || num_threads == 0 ||
        num_experts % world_size != 0)
        return plan;
    const std::uint64_t local_experts = num_experts / world_size;
    if (local_experts == 0 || local_experts > num_threads ||
        local_experts > std::numeric_limits<std::uint32_t>::max())
        return plan;
    plan.local_experts = static_cast<std::uint32_t>(local_experts);
    plan.active_threads = plan.local_experts;
    plan.valid = true;
    return plan;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
direct_dispatch_simt_cached_bitmap_owner(
    std::uint32_t block_index) noexcept {
    return block_index == 0;
}
#endif

constexpr bool dispatch_receive_tile_count(
    std::uint64_t world_size, std::uint64_t shard_capacity,
    std::uint64_t* tile_count) noexcept {
    if (tile_count == nullptr ||
        (shard_capacity != 0 &&
         world_size > std::numeric_limits<std::uint64_t>::max() /
             shard_capacity))
        return false;
    const std::uint64_t records = world_size * shard_capacity;
    *tile_count = records / kDispatchReceiveRecordsPerTile +
        (records % kDispatchReceiveRecordsPerTile != 0 ? 1 : 0);
    return true;
}

constexpr bool dispatch_receive_logical_record(
    std::uint64_t source_rank, std::uint64_t source_slot,
    std::uint64_t world_size, std::uint64_t shard_capacity,
    std::uint64_t* logical_record) noexcept {
    if (logical_record == nullptr || shard_capacity == 0 ||
        source_rank >= world_size || source_slot >= shard_capacity ||
        source_rank >
            (std::numeric_limits<std::uint64_t>::max() - source_slot) /
                shard_capacity)
        return false;
    *logical_record = source_rank * shard_capacity + source_slot;
    return true;
}

constexpr bool dispatch_receive_record_coordinates(
    std::uint64_t logical_record, std::uint64_t world_size,
    std::uint64_t shard_capacity, std::uint64_t* source_rank,
    std::uint64_t* source_slot) noexcept {
    if (source_rank == nullptr || source_slot == nullptr ||
        shard_capacity == 0 ||
        logical_record / shard_capacity >= world_size)
        return false;
    *source_rank = logical_record / shard_capacity;
    *source_slot = logical_record % shard_capacity;
    return true;
}

constexpr bool dispatch_compact_record_coordinates(
    std::uint64_t compact_record, const std::uint64_t* source_bases,
    const std::uint64_t* source_counts, std::uint64_t world_size,
    std::uint64_t* source_rank, std::uint64_t* source_slot) noexcept {
    if (source_bases == nullptr || source_counts == nullptr ||
        source_rank == nullptr || source_slot == nullptr || world_size == 0)
        return false;
    for (std::uint64_t rank = 0; rank < world_size; ++rank) {
        const std::uint64_t base = source_bases[rank];
        if (compact_record >= base &&
            compact_record - base < source_counts[rank]) {
            *source_rank = rank;
            *source_slot = compact_record - base;
            return true;
        }
    }
    return false;
}

struct DispatchCountBridgeLayout {
    std::uint64_t rank_prefix_offset = 0;
    std::uint64_t kernel_expert_prefix_offset = 0;
    std::uint64_t kernel_unaligned_offset = 0;
    std::uint64_t kernel_elements = 0;
    std::uint64_t public_expert_prefix_offset = 0;
    std::uint64_t public_unaligned_offset = 0;
    std::uint64_t public_elements = 0;
    bool valid = false;
};

constexpr bool dispatch_count_bridge_align(
    std::uint64_t value, std::uint64_t alignment,
    std::uint64_t* output) noexcept {
    if (output == nullptr || alignment == 0 ||
        value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
        return false;
    *output = ((value + alignment - 1) / alignment) * alignment;
    return true;
}

constexpr DispatchCountBridgeLayout dispatch_count_bridge_layout(
    std::uint64_t world_size, std::uint64_t num_experts,
    std::uint64_t local_experts,
    std::uint64_t alignment_elements) noexcept {
    DispatchCountBridgeLayout layout{};
    if (world_size == 0 || num_experts == 0 || local_experts == 0 ||
        alignment_elements == 0 || local_experts > num_experts)
        return layout;
    layout.rank_prefix_offset = 0;
    if (!dispatch_count_bridge_align(
            world_size, alignment_elements,
            &layout.kernel_expert_prefix_offset) ||
        num_experts == std::numeric_limits<std::uint64_t>::max())
        return layout;
    const std::uint64_t expert_prefix_end =
        layout.kernel_expert_prefix_offset + num_experts + 1;
    if (expert_prefix_end < layout.kernel_expert_prefix_offset ||
        !dispatch_count_bridge_align(
            expert_prefix_end, alignment_elements,
            &layout.kernel_unaligned_offset) ||
        num_experts > std::numeric_limits<std::uint64_t>::max() -
                          layout.kernel_unaligned_offset)
        return layout;
    layout.kernel_elements = layout.kernel_unaligned_offset + num_experts;

    layout.public_expert_prefix_offset = 0;
    if (!dispatch_count_bridge_align(
            local_experts, alignment_elements,
            &layout.public_unaligned_offset) ||
        local_experts > std::numeric_limits<std::uint64_t>::max() -
                            layout.public_unaligned_offset)
        return layout;
    layout.public_elements = layout.public_unaligned_offset + local_experts;
    layout.valid = true;
    return layout;
}

constexpr bool dispatch_expert_tile_index(
    std::uint64_t tile, std::uint64_t local_expert,
    std::uint64_t tile_count, std::uint64_t local_experts,
    std::uint64_t* index) noexcept {
    if (index == nullptr || tile >= tile_count ||
        local_expert >= local_experts ||
        (local_experts != 0 &&
         tile > (std::numeric_limits<std::uint64_t>::max() - local_expert) /
             local_experts))
        return false;
    *index = tile * local_experts + local_expert;
    return true;
}

constexpr bool dispatch_expert_destination(
    std::uint64_t expert_prefix, std::uint64_t tile_prefix,
    std::uint64_t local_occurrence, std::uint64_t capacity,
    std::uint64_t* destination) noexcept {
    if (destination == nullptr ||
        tile_prefix >
            std::numeric_limits<std::uint64_t>::max() - expert_prefix ||
        local_occurrence >
            std::numeric_limits<std::uint64_t>::max() -
                expert_prefix - tile_prefix)
        return false;
    *destination = expert_prefix + tile_prefix + local_occurrence;
    return *destination < capacity;
}

constexpr bool dispatch_bitmap_words(
    std::uint64_t bits, std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    *words = bits / kDispatchBitmapWordBits +
        (bits % kDispatchBitmapWordBits != 0 ? 1 : 0);
    return true;
}

constexpr bool dispatch_owner_bitmap_words(
    std::uint64_t owners, std::uint64_t bits_per_owner,
    std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    std::uint64_t words_per_owner = 0;
    if (!dispatch_bitmap_words(bits_per_owner, &words_per_owner) ||
        (words_per_owner != 0 &&
         owners > std::numeric_limits<std::uint64_t>::max() /
             words_per_owner))
        return false;
    *words = owners * words_per_owner;
    return true;
}

constexpr bool dispatch_owner_bitmap_range(
    std::uint64_t owner, std::uint64_t owners,
    std::uint64_t bits_per_owner, std::uint64_t* base_word,
    std::uint64_t* word_count) noexcept {
    if (base_word == nullptr || word_count == nullptr || owner >= owners)
        return false;
    if (!dispatch_bitmap_words(bits_per_owner, word_count) ||
        (*word_count != 0 &&
         owner > std::numeric_limits<std::uint64_t>::max() /
             *word_count))
        return false;
    *base_word = owner * *word_count;
    return true;
}

constexpr bool dispatch_bitmap_location(
    std::uint64_t bit, std::uint64_t bit_count,
    std::uint64_t* word, std::uint64_t* mask) noexcept {
    if (word == nullptr || mask == nullptr || bit >= bit_count)
        return false;
    *word = bit / kDispatchBitmapWordBits;
    *mask = std::uint64_t{1} << (bit % kDispatchBitmapWordBits);
    return true;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_receive_record_coordinates(
    std::uint64_t logical_record, std::uint64_t world_size,
    std::uint64_t shard_capacity, std::uint64_t* source_rank,
    std::uint64_t* source_slot) noexcept {
    if (source_rank == nullptr || source_slot == nullptr ||
        shard_capacity == 0 ||
        logical_record / shard_capacity >= world_size)
        return false;
    *source_rank = logical_record / shard_capacity;
    *source_slot = logical_record % shard_capacity;
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_expert_tile_index(
    std::uint64_t tile, std::uint64_t local_expert,
    std::uint64_t tile_count, std::uint64_t local_experts,
    std::uint64_t* index) noexcept {
    if (index == nullptr || tile >= tile_count ||
        local_expert >= local_experts ||
        (local_experts != 0 &&
         tile > (std::numeric_limits<std::uint64_t>::max() - local_expert) /
             local_experts))
        return false;
    *index = tile * local_experts + local_expert;
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_bitmap_words(
    std::uint64_t bits, std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    *words = bits / kDispatchBitmapWordBits +
        (bits % kDispatchBitmapWordBits != 0 ? 1 : 0);
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_owner_bitmap_words(
    std::uint64_t owners, std::uint64_t bits_per_owner,
    std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    std::uint64_t words_per_owner = 0;
    if (!dispatch_simt_bitmap_words(bits_per_owner, &words_per_owner) ||
        (words_per_owner != 0 &&
         owners > std::numeric_limits<std::uint64_t>::max() /
             words_per_owner))
        return false;
    *words = owners * words_per_owner;
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_owner_bitmap_range(
    std::uint64_t owner, std::uint64_t owners,
    std::uint64_t bits_per_owner, std::uint64_t* base_word,
    std::uint64_t* word_count) noexcept {
    if (base_word == nullptr || word_count == nullptr || owner >= owners)
        return false;
    if (!dispatch_simt_bitmap_words(bits_per_owner, word_count) ||
        (*word_count != 0 &&
         owner > std::numeric_limits<std::uint64_t>::max() /
             *word_count))
        return false;
    *base_word = owner * *word_count;
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_bitmap_location(
    std::uint64_t bit, std::uint64_t bit_count,
    std::uint64_t* word, std::uint64_t* mask) noexcept {
    if (word == nullptr || mask == nullptr || bit >= bit_count)
        return false;
    *word = bit / kDispatchBitmapWordBits;
    *mask = std::uint64_t{1} << (bit % kDispatchBitmapWordBits);
    return true;
}

#endif

#if defined(DEEP_EP_ASCEND_DISPATCH_PARALLEL_AICORE_CALLEE_LOCAL)
#undef DEEP_EP_ASCEND_DISPATCH_PARALLEL_AICORE_CALLEE_LOCAL
#undef DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE
#endif

}  // namespace deep_ep::ascend::elastic
