#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kCombineRecordsPerTile = 128;

constexpr bool combine_tile_count(
    std::uint64_t record_count, std::uint64_t* tile_count) noexcept {
    if (tile_count == nullptr)
        return false;
    *tile_count = record_count / kCombineRecordsPerTile +
        (record_count % kCombineRecordsPerTile != 0 ? 1 : 0);
    return true;
}

constexpr bool combine_tile_rank_index(
    std::uint64_t tile, std::uint64_t rank, std::uint64_t tile_count,
    std::uint64_t world_size, std::uint64_t* index) noexcept {
    if (index == nullptr || tile >= tile_count || rank >= world_size ||
        (world_size != 0 &&
         tile > (std::numeric_limits<std::uint64_t>::max() - rank) /
             world_size))
        return false;
    *index = tile * world_size + rank;
    return true;
}

constexpr bool combine_receive_record_coordinates(
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

constexpr bool combine_record_slot_index(
    std::uint64_t token, std::uint64_t lane, std::uint64_t num_tokens,
    std::uint64_t num_topk, std::uint64_t* index) noexcept {
    if (index == nullptr || token >= num_tokens || lane >= num_topk ||
        (num_topk != 0 &&
         token > (std::numeric_limits<std::uint64_t>::max() - lane) /
             num_topk))
        return false;
    *index = token * num_topk + lane;
    return true;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
combine_simt_receive_record_coordinates(
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
combine_simt_record_slot_index(
    std::uint64_t token, std::uint64_t lane, std::uint64_t num_tokens,
    std::uint64_t num_topk, std::uint64_t* index) noexcept {
    if (index == nullptr || token >= num_tokens || lane >= num_topk ||
        (num_topk != 0 &&
         token > (std::numeric_limits<std::uint64_t>::max() - lane) /
             num_topk))
        return false;
    *index = token * num_topk + lane;
    return true;
}

#endif

}  // namespace deep_ep::ascend::elastic
