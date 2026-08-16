#pragma once

#include <cstdint>

#include "dispatch_state.hpp"
#include "../transport/transport_commands.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE \
    __SIMT_DEVICE_FUNCTIONS_DECL__
#else
#define DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE
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

}  // namespace deep_ep::ascend::elastic

#undef DEEP_EP_ASCEND_COMBINE_STATE_SIMT_CALLEE
