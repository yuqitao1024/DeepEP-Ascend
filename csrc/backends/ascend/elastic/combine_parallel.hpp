#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kCombineRecordsPerTile = 128;

struct CombineProducerPayloadCopyPlan {
    bool valid = false;
    std::uint64_t vector_elements = 0;
    std::uint64_t scalar_begin = 0;
};

enum class CombineExpandedVectorReduceConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct CombineExpandedVectorReduceConfig {
    bool enabled = false;
};

inline CombineExpandedVectorReduceConfigStatus
select_combine_expanded_vector_reduce_config(
    const char* value, bool direct, bool expanded,
    bool allow_multiple_reduction, bool hybrid, std::uint64_t num_topk,
    CombineExpandedVectorReduceConfig* output) noexcept {
    if (output == nullptr)
        return CombineExpandedVectorReduceConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr || (value[0] == '0' && value[1] == '\0'))
        return CombineExpandedVectorReduceConfigStatus::kDisabled;
    if (value[0] != '1' || value[1] != '\0')
        return CombineExpandedVectorReduceConfigStatus::kInvalid;
    if (!direct || !expanded || !allow_multiple_reduction || hybrid ||
        num_topk == 0 || num_topk > 32)
        return CombineExpandedVectorReduceConfigStatus::kDisabled;
    output->enabled = true;
    return CombineExpandedVectorReduceConfigStatus::kEnabled;
}

constexpr CombineProducerPayloadCopyPlan
combine_expanded_producer_payload_plan(
    std::uint64_t hidden_elements, std::uint64_t vector_tile_elements,
    std::uint64_t data_copy_alignment_elements, bool enabled) noexcept {
    if (vector_tile_elements == 0 || data_copy_alignment_elements == 0 ||
        vector_tile_elements % data_copy_alignment_elements != 0)
        return {};
    const std::uint64_t vector_elements = enabled ?
        hidden_elements - hidden_elements % vector_tile_elements : 0;
    return {true, vector_elements, vector_elements};
}

constexpr CombineProducerPayloadCopyPlan combine_producer_payload_copy_plan(
    std::uint64_t hidden_elements, std::uint64_t vector_tile_elements,
    std::uint64_t data_copy_alignment_elements, bool expanded) noexcept {
    if (vector_tile_elements == 0 || data_copy_alignment_elements == 0 ||
        vector_tile_elements % data_copy_alignment_elements != 0)
        return {};
    const std::uint64_t vector_elements =
        expanded || hidden_elements % data_copy_alignment_elements != 0 ? 0 :
        hidden_elements - hidden_elements % vector_tile_elements;
    return {true, vector_elements, vector_elements};
}

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE
__aicore__ inline CombineProducerPayloadCopyPlan
aicore_combine_expanded_producer_payload_plan(
    std::uint64_t hidden_elements, std::uint64_t vector_tile_elements,
    std::uint64_t data_copy_alignment_elements, bool enabled) noexcept {
    if (vector_tile_elements == 0 || data_copy_alignment_elements == 0 ||
        vector_tile_elements % data_copy_alignment_elements != 0)
        return {};
    const std::uint64_t vector_elements = enabled ?
        hidden_elements - hidden_elements % vector_tile_elements : 0;
    return {true, vector_elements, vector_elements};
}

__aicore__ inline CombineProducerPayloadCopyPlan
aicore_combine_producer_payload_copy_plan(
    std::uint64_t hidden_elements, std::uint64_t vector_tile_elements,
    std::uint64_t data_copy_alignment_elements, bool expanded) noexcept {
    if (vector_tile_elements == 0 || data_copy_alignment_elements == 0 ||
        vector_tile_elements % data_copy_alignment_elements != 0)
        return {};
    const std::uint64_t vector_elements =
        expanded || hidden_elements % data_copy_alignment_elements != 0 ? 0 :
        hidden_elements - hidden_elements % vector_tile_elements;
    return {true, vector_elements, vector_elements};
}
#endif

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline CombineProducerPayloadCopyPlan
simt_combine_expanded_producer_payload_plan(
    std::uint64_t hidden_elements, std::uint64_t vector_tile_elements,
    std::uint64_t data_copy_alignment_elements, bool enabled) noexcept {
    if (vector_tile_elements == 0 || data_copy_alignment_elements == 0 ||
        vector_tile_elements % data_copy_alignment_elements != 0)
        return {};
    const std::uint64_t vector_elements = enabled ?
        hidden_elements - hidden_elements % vector_tile_elements : 0;
    return {true, vector_elements, vector_elements};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline CombineProducerPayloadCopyPlan
simt_combine_producer_payload_copy_plan(
    std::uint64_t hidden_elements, std::uint64_t vector_tile_elements,
    std::uint64_t data_copy_alignment_elements, bool expanded) noexcept {
    if (vector_tile_elements == 0 || data_copy_alignment_elements == 0 ||
        vector_tile_elements % data_copy_alignment_elements != 0)
        return {};
    const std::uint64_t vector_elements =
        expanded || hidden_elements % data_copy_alignment_elements != 0 ? 0 :
        hidden_elements - hidden_elements % vector_tile_elements;
    return {true, vector_elements, vector_elements};
}
#endif

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

constexpr bool combine_rank_prefix_range(
    const std::int32_t* prefix_per_rank, std::uint64_t world_size,
    std::uint64_t num_rows, std::uint64_t rank, std::uint64_t* begin,
    std::uint64_t* end) noexcept {
    if (prefix_per_rank == nullptr || begin == nullptr || end == nullptr ||
        world_size == 0 || rank >= world_size)
        return false;
    std::uint64_t previous = 0;
    for (std::uint64_t current_rank = 0;
         current_rank < world_size; ++current_rank) {
        const std::int32_t encoded_end = prefix_per_rank[current_rank];
        if (encoded_end < 0 ||
            static_cast<std::uint64_t>(encoded_end) < previous ||
            static_cast<std::uint64_t>(encoded_end) > num_rows)
            return false;
        if (current_rank == rank) {
            *begin = previous;
            *end = static_cast<std::uint64_t>(encoded_end);
        }
        previous = static_cast<std::uint64_t>(encoded_end);
    }
    return previous == num_rows;
}

constexpr bool combine_destination_rank_for_row(
    std::uint64_t row, const std::int32_t* prefix_per_rank,
    std::uint64_t world_size, std::uint64_t num_rows,
    std::uint64_t* rank) noexcept {
    if (rank == nullptr || row >= num_rows || prefix_per_rank == nullptr ||
        world_size == 0)
        return false;
    std::uint64_t previous = 0;
    for (std::uint64_t current_rank = 0;
         current_rank < world_size; ++current_rank) {
        const std::int32_t encoded_end = prefix_per_rank[current_rank];
        if (encoded_end < 0 ||
            static_cast<std::uint64_t>(encoded_end) < previous ||
            static_cast<std::uint64_t>(encoded_end) > num_rows)
            return false;
        const auto current_end = static_cast<std::uint64_t>(encoded_end);
        if (row < current_end) {
            *rank = current_rank;
            return prefix_per_rank[world_size - 1] ==
                static_cast<std::int32_t>(num_rows);
        }
        previous = current_end;
    }
    return false;
}

constexpr bool combine_record_destination_slot(
    std::uint64_t tile_prefix, std::uint64_t local_occurrence,
    std::uint64_t capacity, std::uint64_t* slot) noexcept {
    if (slot == nullptr ||
        local_occurrence >
            std::numeric_limits<std::uint64_t>::max() - tile_prefix)
        return false;
    *slot = tile_prefix + local_occurrence;
    return *slot < capacity;
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
combine_simt_record_destination_slot(
    std::uint64_t tile_prefix, std::uint64_t local_occurrence,
    std::uint64_t capacity, std::uint64_t* slot) noexcept {
    if (slot == nullptr ||
        local_occurrence >
            std::numeric_limits<std::uint64_t>::max() - tile_prefix)
        return false;
    *slot = tile_prefix + local_occurrence;
    return *slot < capacity;
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
