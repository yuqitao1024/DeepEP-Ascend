#pragma once

#include <cstdint>

#include "dispatch_state.hpp"

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

}  // namespace deep_ep::ascend::elastic
