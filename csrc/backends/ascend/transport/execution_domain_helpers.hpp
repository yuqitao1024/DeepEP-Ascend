#pragma once

#include "sync_layout.hpp"
#include "transport_commands.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE \
    __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#else
#define DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE inline constexpr
#endif

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE
#define DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE __aicore__ inline
#else
#define DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE inline constexpr
#endif

namespace deep_ep::ascend::transport::command {

// Keep these wrappers leaf-only: Bisheng cannot link device-domain calls to
// ordinary inline constexpr host helpers.
DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE std::uint64_t
aicore_profile_payload_bytes(
    TransportCommandOpcode opcode, std::uint64_t bytes) {
    switch (opcode) {
        case TransportCommandOpcode::kPut:
            return bytes;
        case TransportCommandOpcode::kPutValue64:
        case TransportCommandOpcode::kRemoteAdd64:
            return sizeof(std::uint64_t);
        default:
            return 0;
    }
}

DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE bool simt_barrier_team_enabled(
    const TransportTopology& topology, std::uint32_t team_mask,
    TransportTeam team) {
    if (team == TransportTeam::kScaleOut)
        return topology.scale_out_size > 1 &&
            (team_mask & kScaleOutTeamMask) != 0;
    if (team == TransportTeam::kScaleUp)
        return topology.scale_up_size > 1 &&
            (team_mask & kScaleUpTeamMask) != 0;
    return false;
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool
aicore_valid_staged_context_header(
    std::uint32_t abi_version, std::uint32_t struct_size,
    std::uint32_t cann_compatibility, std::uintptr_t command_queue) {
    return abi_version == kTransportCommandAbiVersion &&
           struct_size == sizeof(StagedTransportContext) &&
           cann_compatibility == kStagedTransportCannCompatibility &&
           command_queue != 0;
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool
aicore_valid_command_queue_header(
    std::uint32_t abi_version, std::uint32_t struct_size,
    std::uintptr_t commands, std::uint32_t capacity, std::uint32_t count,
    std::uintptr_t service_state, std::uintptr_t diagnostic) {
    return abi_version == kTransportCommandAbiVersion &&
           struct_size == sizeof(TransportCommandQueue) && commands != 0 &&
           count <= capacity && service_state != 0 && diagnostic != 0;
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool
aicore_valid_service_state_header(
    std::uint32_t abi_version, std::uint32_t struct_size) {
    return abi_version == kTransportCommandAbiVersion &&
           struct_size == sizeof(TransportServiceState);
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool
aicore_valid_diagnostic_header(std::uint32_t abi_version) {
    return abi_version == kTransportCommandAbiVersion;
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE std::uint64_t
aicore_mix_registration_cookie(std::uint64_t state, std::uint64_t value) {
    return state ^ (value + 0x9e3779b97f4a7c15ULL + (state << 6U) +
                    (state >> 2U));
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE std::uint64_t
aicore_registration_cookie(
    std::uintptr_t command_queue, std::uintptr_t commands,
    std::uintptr_t service_state, std::uintptr_t diagnostic,
    std::uint32_t capacity) {
    std::uint64_t state = 0x445045505452414eULL;
    state = aicore_mix_registration_cookie(state, command_queue);
    state = aicore_mix_registration_cookie(state, commands);
    state = aicore_mix_registration_cookie(state, service_state);
    state = aicore_mix_registration_cookie(state, diagnostic);
    state = aicore_mix_registration_cookie(state, capacity);
    return state | 1ULL;
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool
aicore_valid_registration_cookie(
    std::uint64_t cookie, std::uintptr_t command_queue,
    std::uintptr_t commands, std::uintptr_t service_state,
    std::uintptr_t diagnostic, std::uint32_t capacity) {
    return cookie != 0 && cookie == aicore_registration_cookie(
        command_queue, commands, service_state, diagnostic, capacity);
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool aicore_barrier_team_enabled(
    const TransportTopology& topology, std::uint32_t team_mask,
    TransportTeam team) {
    if (team == TransportTeam::kScaleOut)
        return topology.scale_out_size > 1 &&
            (team_mask & kScaleOutTeamMask) != 0;
    if (team == TransportTeam::kScaleUp)
        return topology.scale_up_size > 1 &&
            (team_mask & kScaleUpTeamMask) != 0;
    return false;
}

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE bool aicore_barrier_peer_in_team(
    const TransportTopology& topology, TransportTeam team, int world_peer) {
    if (world_peer < 0 || world_peer >= topology.world_size ||
        world_peer == topology.world_rank)
        return false;
    if (team == TransportTeam::kScaleOut)
        return world_peer % topology.scale_up_size == topology.scale_up_rank;
    if (team == TransportTeam::kScaleUp)
        return world_peer / topology.scale_up_size == topology.scale_out_rank;
    return false;
}

}  // namespace deep_ep::ascend::transport::command

namespace deep_ep::ascend::transport::sync_layout {

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE std::uint64_t aicore_barrier_offset(
    std::uint32_t member_count, std::uint32_t barrier_index,
    std::uint32_t source_member) {
    return ((static_cast<std::uint64_t>(kLogicalSignalCount) + barrier_index) *
            member_count + source_member) * sizeof(std::uint64_t);
}

}  // namespace deep_ep::ascend::transport::sync_layout

#undef DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE
#undef DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE
