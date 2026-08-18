#pragma once

#include <cstdint>

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_SYNC_LAYOUT_CALLEE \
    __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#else
#define DEEP_EP_ASCEND_SYNC_LAYOUT_CALLEE inline constexpr
#endif

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

enum class SignalAddressFailure : std::uint32_t {
    kNone = 0,
    kMissingTeam = 10,
    kInvalidSourceMember = 11,
    kInvalidSelfMember = 12,
    kInvalidSignalCount = 13,
    kInvalidCounterCount = 14,
    kInvalidBarrierCount = 15,
    kInvalidSignalIndex = 16,
    kMissingRemoteSyncMemories = 17,
    kMissingLocalSyncMemory = 18,
};

DEEP_EP_ASCEND_SYNC_LAYOUT_CALLEE SignalAddressFailure
classify_signal_address_layout(
    std::uint32_t member_count, std::uint32_t self_member,
    std::uint32_t signal_count, std::uint32_t counter_count,
    std::uint32_t barrier_count, std::uintptr_t remote_sync_memories,
    int source_member, std::uint32_t signal_index) {
    if (source_member < 0 ||
        static_cast<std::uint32_t>(source_member) >= member_count)
        return SignalAddressFailure::kInvalidSourceMember;
    if (self_member >= member_count)
        return SignalAddressFailure::kInvalidSelfMember;
    if (signal_count != kWorldTeamSignalCount)
        return SignalAddressFailure::kInvalidSignalCount;
    if (counter_count != kWorldTeamCounterCount)
        return SignalAddressFailure::kInvalidCounterCount;
    if (barrier_count < kWorldTeamBarrierCount)
        return SignalAddressFailure::kInvalidBarrierCount;
    if (signal_index >= kLogicalSignalCount)
        return SignalAddressFailure::kInvalidSignalIndex;
    if (remote_sync_memories == 0)
        return SignalAddressFailure::kMissingRemoteSyncMemories;
    return SignalAddressFailure::kNone;
}

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

#undef DEEP_EP_ASCEND_SYNC_LAYOUT_CALLEE
