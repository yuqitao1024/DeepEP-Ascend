#pragma once

#include <cstdint>

#include "dispatch_state.hpp"
#include "../transport/transport_commands.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE \
    __SIMT_DEVICE_FUNCTIONS_DECL__
#define DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL __gm__
#else
#define DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE
#if !defined(DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL)
#define DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL
#endif
#endif

namespace deep_ep::ascend::elastic {

inline constexpr std::uint32_t kCombineRecordAbiVersion = 1;

using CombineSequence = DispatchSequence;
using CombineAttempt = DispatchAttempt;

struct alignas(16) CombineControlSlot {
    std::uint64_t generation = 0;
    std::uint64_t count = 0;
};

static_assert(sizeof(CombineControlSlot) == kCombineControlSlotBytes);

struct CombineRecordHeader {
    std::uint32_t abi_version = kCombineRecordAbiVersion;
    std::uint32_t struct_size = sizeof(CombineRecordHeader);
    std::int32_t origin_token = -1;
    std::int32_t contributor_rank = -1;
    std::int32_t master_lane = -1;
    std::int32_t contribution_lane = -1;
};

static_assert(sizeof(CombineRecordHeader) == kCombineRecordHeaderBytes);

struct HybridCombineRouteMetadata {
    std::uint64_t ingress_slot = kInvalidHybridRouteSlot;
    std::uint64_t forwarded_slot = kInvalidHybridRouteSlot;
};

static_assert(
    sizeof(HybridCombineRouteMetadata) ==
    kHybridCombineRouteMetadataBytes);
static_assert(
    kDirectCombineRecordTrailerBytes >= sizeof(CombineRecordHeader));
static_assert(
    kHybridCombineRecordTrailerBytes >=
    sizeof(CombineRecordHeader) + sizeof(HybridCombineRouteMetadata));

struct CombineRecordTrailerLayout {
    std::uint64_t header_offset = 0;
    std::uint64_t route_metadata_offset = 0;
    bool has_route_metadata = false;
    bool valid = false;
};

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr
CombineRecordTrailerLayout combine_record_trailer_layout(
    std::uint64_t record_bytes, bool hybrid) noexcept {
    CombineRecordTrailerLayout result{};
    result.has_route_metadata = hybrid;
    const std::uint64_t minimum_bytes = hybrid ?
        sizeof(CombineRecordHeader) + sizeof(HybridCombineRouteMetadata) :
        sizeof(CombineRecordHeader);
    if (record_bytes < minimum_bytes)
        return result;
    const std::uint64_t reserved_trailer_bytes = hybrid ?
        kHybridCombineRecordTrailerBytes :
        kDirectCombineRecordTrailerBytes;
    result.header_offset = record_bytes >= reserved_trailer_bytes ?
        record_bytes - reserved_trailer_bytes : 0;
    if (result.header_offset + sizeof(CombineRecordHeader) > record_bytes)
        return result;
    if (hybrid) {
        result.route_metadata_offset =
            result.header_offset + sizeof(CombineRecordHeader);
        if (result.route_metadata_offset +
                sizeof(HybridCombineRouteMetadata) > record_bytes)
            return result;
    }
    result.valid = true;
    return result;
}

// A source-specific adapter exposes received records in this neutral form so
// origin reduction can be exercised on the host and run on the device.
struct CombineOriginRecordView {
    CombineRecordHeader header{};
    float activation = 0.0F;
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const float* routing_weights = nullptr;
};

struct CombineOriginRecordRange {
    const CombineOriginRecordView* records = nullptr;
    std::uint64_t count = 0;
};

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr bool
is_clean_combine_transport_completion(
    std::uint64_t expected_generation, std::uint64_t consumed_generation,
    std::uint32_t diagnostic_abi_version,
    std::uint64_t diagnostic_generation,
    transport::DeviceTransportError diagnostic_error) noexcept {
    return consumed_generation == expected_generation &&
           diagnostic_abi_version == transport::kTransportCommandAbiVersion &&
           diagnostic_generation == expected_generation &&
           diagnostic_error == transport::DeviceTransportError::kNone;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr std::uintptr_t
combine_receive_shard_address(
    std::uintptr_t local_window_base, std::uint64_t receive_offset,
    std::uint64_t contributor_rank,
    std::uint64_t receive_shard_bytes) noexcept {
    return local_window_base + receive_offset +
           contributor_rank * receive_shard_bytes;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr std::uintptr_t
combine_staging_shard_address(
    std::uintptr_t local_window_base, std::uint64_t staging_offset,
    std::uint64_t destination_rank,
    std::uint64_t staging_shard_bytes) noexcept {
    return local_window_base + staging_offset +
           destination_rank * staging_shard_bytes;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr bool
is_valid_combine_token_extent(
    std::uint64_t num_tokens, std::uint64_t shard_capacity) noexcept {
    return num_tokens <= shard_capacity;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr bool
is_valid_combine_origin_token(
    std::int32_t origin_token, std::uint64_t num_tokens,
    std::uint64_t shard_capacity) noexcept {
    return is_dispatch_local_index(origin_token, num_tokens) &&
           is_dispatch_local_index(origin_token, shard_capacity);
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr bool
is_valid_combine_source_identity(
    std::int32_t encoded_source, int destination_rank,
    int current_rank,
    std::uint64_t shard_capacity, std::uint64_t num_tokens) noexcept {
    const int origin_rank = decode_dispatch_source_rank(
        encoded_source, shard_capacity);
    const std::int32_t origin_token = decode_dispatch_local_index(
        encoded_source, shard_capacity);
    const std::uint64_t origin_token_bound =
        origin_rank == current_rank ? num_tokens : shard_capacity;
    return origin_rank == destination_rank &&
           is_valid_combine_origin_token(
               origin_token, origin_token_bound, shard_capacity);
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr bool
combine_expanded_input_row_is_valid(
    std::int32_t input_row, std::uint64_t num_input_rows) noexcept {
    return input_row == -1 ||
           (input_row >= 0 &&
            static_cast<std::uint64_t>(input_row) < num_input_rows);
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr bool
is_valid_combine_record_lanes(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const std::int64_t* routes,
    std::uint64_t num_topk, std::uint64_t num_experts,
    std::uint64_t num_local_experts, std::int32_t contributor_rank,
    std::int32_t master_lane, std::int32_t contribution_lane,
    bool expanded, bool allow_multiple_reduction) noexcept {
    if (routes == nullptr || num_local_experts == 0 || contributor_rank < 0 ||
        !is_dispatch_local_index(master_lane, num_topk) ||
        !is_dispatch_local_index(contribution_lane, num_topk))
        return false;

    const std::uint64_t owner =
        static_cast<std::uint64_t>(contributor_rank);
    std::int32_t canonical_master = -1;
    for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
        const std::int64_t expert = routes[lane];
        if (expert < 0 || static_cast<std::uint64_t>(expert) >= num_experts)
            continue;
        if (static_cast<std::uint64_t>(expert) / num_local_experts == owner) {
            canonical_master = static_cast<std::int32_t>(lane);
            break;
        }
    }
    if (master_lane != canonical_master)
        return false;

    const std::int64_t contribution_expert = routes[
        static_cast<std::uint64_t>(contribution_lane)];
    if (contribution_expert < 0 ||
        static_cast<std::uint64_t>(contribution_expert) >= num_experts ||
        static_cast<std::uint64_t>(contribution_expert) /
                num_local_experts != owner)
        return false;
    return expanded && !allow_multiple_reduction ? true :
        contribution_lane == master_lane;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr std::uint64_t
combine_expanded_record_count(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const std::int32_t* input_rows,
    std::uint64_t num_topk,
    bool allow_multiple_reduction) noexcept {
    std::uint64_t valid_lanes = 0;
    for (std::uint64_t lane = 0; lane < num_topk; ++lane)
        valid_lanes += input_rows[lane] == -1 ? 0 : 1;
    return allow_multiple_reduction ? (valid_lanes == 0 ? 0 : 1) :
        valid_lanes;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr std::int32_t
combine_expanded_record_lane(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const std::int32_t* input_rows,
    std::uint64_t num_topk,
    std::uint64_t record_index) noexcept {
    std::uint64_t seen = 0;
    for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
        if (input_rows[lane] == -1)
            continue;
        if (seen++ == record_index)
            return static_cast<std::int32_t>(lane);
    }
    return -1;
}

template <typename Value>
DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE inline float
combine_reduce_expanded_lanes(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const Value* input,
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const std::int32_t* input_rows,
    std::uint64_t num_topk, std::uint64_t num_input_rows,
    std::uint64_t hidden_elements, std::uint64_t hidden) noexcept {
    float value = 0.0F;
    for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
        const std::int32_t input_row = input_rows[lane];
        if (input_row == -1)
            continue;
        if (!combine_expanded_input_row_is_valid(input_row, num_input_rows))
            return 0.0F;
        value += static_cast<float>(input[
            static_cast<std::uint64_t>(input_row) * hidden_elements + hidden]);
    }
    return value;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE inline float
combine_routing_weight(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const float* weights,
    std::uint64_t index) noexcept {
    return weights[index];
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE inline float
combine_normal_record_routing_weight(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const float* weights,
    std::uint64_t source_row, std::int32_t lane,
    std::int32_t master_lane, std::uint64_t num_topk) noexcept {
    if (weights == nullptr ||
        !is_dispatch_local_index(lane, num_topk) ||
        !is_dispatch_local_index(master_lane, num_topk))
        return 0.0F;
    return combine_routing_weight(
        weights, source_row * num_topk + static_cast<std::uint64_t>(lane));
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE inline void
combine_fill_normal_record_routing_weights(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const float* weights,
    std::uint64_t source_row, std::int32_t master_lane,
    std::uint64_t num_topk,
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL std::uint8_t* record,
    std::uint64_t weight_offset) noexcept {
    if (record == nullptr)
        return;
    auto* record_weights =
        reinterpret_cast<DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL float*>(
            record + weight_offset);
    for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
        record_weights[lane] = combine_normal_record_routing_weight(
            weights, source_row, static_cast<std::int32_t>(lane),
            master_lane, num_topk);
    }
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE inline CombineRecordHeader
load_combine_record_header(
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const CombineRecordHeader* header)
    noexcept {
    CombineRecordHeader result{};
    result.abi_version = header->abi_version;
    result.struct_size = header->struct_size;
    result.origin_token = header->origin_token;
    result.contributor_rank = header->contributor_rank;
    result.master_lane = header->master_lane;
    result.contribution_lane = header->contribution_lane;
    return result;
}

DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE constexpr float
combine_apply_biases(float value, float bias_0, float bias_1) noexcept {
    return value + bias_0 + bias_1;
}

template <typename RecordSource>
DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE inline float
combine_reduce_origin_records(
    const RecordSource& source, std::uint64_t contributor_count,
    std::int32_t origin_token,
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL const std::int64_t*
        combined_topk_indices,
    std::uint64_t num_topk, std::uint64_t num_experts,
    std::uint64_t num_local_experts, bool expanded,
    bool allow_multiple_reduction, std::uint64_t hidden, float bias_0,
    float bias_1,
    DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL float* output_weights) noexcept {
    if (output_weights != nullptr) {
        for (std::uint64_t lane = 0; lane < num_topk; ++lane)
            output_weights[lane] = 0.0F;
    }

    float activation = 0.0F;
    if (num_local_experts == 0)
        return combine_apply_biases(activation, bias_0, bias_1);

    // Contributor rank and logical lane, rather than receive-slot layout,
    // define the FP32 accumulation order.
    for (std::uint64_t contributor_rank = 0;
         contributor_rank < contributor_count; ++contributor_rank) {
        for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
            const std::int64_t expert = combined_topk_indices[
                static_cast<std::uint64_t>(origin_token) * num_topk + lane];
            if (expert < 0 || static_cast<std::uint64_t>(expert) >= num_experts ||
                static_cast<std::uint64_t>(expert) / num_local_experts !=
                    contributor_rank)
                continue;
            for (std::uint64_t slot = 0;
                 slot < source.record_count(contributor_rank); ++slot) {
                const CombineOriginRecordView record = source.record_at(
                    contributor_rank, slot, hidden);
                if (record.header.origin_token != origin_token ||
                    record.header.contribution_lane !=
                        static_cast<std::int32_t>(lane))
                    continue;
                activation += record.activation;
                break;
            }
        }
    }

    if (output_weights != nullptr) {
        for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
            const std::int64_t expert = combined_topk_indices[
                static_cast<std::uint64_t>(origin_token) * num_topk + lane];
            if (expert < 0 ||
                static_cast<std::uint64_t>(expert) >= num_experts)
                continue;
            const std::uint64_t contributor_rank =
                static_cast<std::uint64_t>(expert) / num_local_experts;
            if (contributor_rank >= contributor_count)
                continue;
            for (std::uint64_t slot = 0;
                 slot < source.record_count(contributor_rank); ++slot) {
                const CombineOriginRecordView record = source.record_at(
                    contributor_rank, slot, hidden);
                if (record.header.origin_token != origin_token ||
                    (expanded && !allow_multiple_reduction &&
                     record.header.contribution_lane !=
                         static_cast<std::int32_t>(lane)))
                    continue;
                output_weights[lane] = record.routing_weights == nullptr ?
                    0.0F : record.routing_weights[lane];
                break;
            }
        }
    }
    return combine_apply_biases(activation, bias_0, bias_1);
}

}  // namespace deep_ep::ascend::elastic

#undef DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE
#undef DEEP_EP_ASCEND_COMBINE_STATE_GLOBAL
