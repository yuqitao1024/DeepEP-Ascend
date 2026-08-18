#pragma once

#include <cstdint>

namespace deep_ep::ascend::transport::sync_layout {

inline constexpr std::uint32_t kLogicalSignalCount = 4;
inline constexpr std::uint32_t kLogicalBarrierCount = 2;
inline constexpr std::uint32_t kDispatchReleaseSignalIndex = 0;
inline constexpr std::uint32_t kCombineReleaseSignalIndex = 1;
inline constexpr std::uint32_t kScaleOutBarrierIndex = 0;
inline constexpr std::uint32_t kScaleUpBarrierIndex = 1;

// HCOMM world teams only accept barrier synchronization storage. The first
// groups back logical signals; the remaining group backs barrier session zero.
inline constexpr std::uint32_t kWorldTeamSignalCount = 0;
inline constexpr std::uint32_t kWorldTeamCounterCount = 0;
inline constexpr std::uint32_t kWorldTeamBarrierCount =
    kLogicalSignalCount + kLogicalBarrierCount;

inline constexpr bool has_required_world_team_layout(
    std::uint32_t signal_count, std::uint32_t counter_count,
    std::uint32_t barrier_count) {
    return signal_count == kWorldTeamSignalCount &&
           counter_count == kWorldTeamCounterCount &&
           barrier_count >= kWorldTeamBarrierCount;
}

inline constexpr std::uint64_t signal_offset(
    std::uint32_t member_count, std::uint32_t signal_index,
    std::uint32_t source_member) {
    return (static_cast<std::uint64_t>(signal_index) * member_count +
            source_member) * sizeof(std::uint64_t);
}

inline constexpr std::uint64_t barrier_offset(
    std::uint32_t member_count, std::uint32_t barrier_index,
    std::uint32_t source_member) {
    return ((static_cast<std::uint64_t>(kLogicalSignalCount) + barrier_index) *
            member_count + source_member) * sizeof(std::uint64_t);
}

}  // namespace deep_ep::ascend::transport::sync_layout
