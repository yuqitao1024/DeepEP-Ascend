#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kDispatchBitmapWordBits = 64;
inline constexpr std::uint64_t kDispatchReceiveRecordsPerTile = 128;

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

}  // namespace deep_ep::ascend::elastic
