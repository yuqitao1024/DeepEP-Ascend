#pragma once

#include "stage_profile.hpp"
#include "sync_layout.hpp"
#include "transport_commands.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE \
    __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#else
#define DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE inline constexpr
#endif

#if !defined(DEEP_EP_ASCEND_SIMT_GLOBAL)
#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_SIMT_GLOBAL __gm__
#else
#define DEEP_EP_ASCEND_SIMT_GLOBAL
#endif
#define DEEP_EP_ASCEND_HELPERS_OWN_SIMT_GLOBAL 1
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

DEEP_EP_ASCEND_AICORE_EXECUTION_INLINE TransportQueueDepthSnapshot
aicore_merge_queue_depth_snapshots(
    TransportQueueDepthSnapshot aggregate,
    TransportQueueDepthSnapshot observed) {
    if (observed.sq_depth > aggregate.sq_depth)
        aggregate.sq_depth = observed.sq_depth;
    if (observed.cq_depth > aggregate.cq_depth)
        aggregate.cq_depth = observed.cq_depth;
    return aggregate;
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

DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE bool simt_publish_request(
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request,
    std::uint32_t command_begin,
    std::uint32_t command_end, std::uint64_t queue_generation) {
    if (request == nullptr)
        return false;
    if (request->abi_version != kDeviceRequestAbiVersion) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidAbi;
        return false;
    }
    if (request->state == DeviceRequestState::kPending ||
        request->state < DeviceRequestState::kEmpty ||
        request->state > DeviceRequestState::kFailed ||
        command_end <= command_begin || queue_generation == 0) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidProtocol;
        return false;
    }
    request->command_begin = command_begin;
    request->command_end = command_end;
    request->queue_generation = queue_generation;
    request->consumed_target = command_end;
    request->terminal_error = DeviceTransportError::kNone;
    request->state = DeviceRequestState::kPending;
    return true;
}

DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE bool simt_observe_request(
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request,
    std::uint64_t queue_generation,
    std::uint64_t consumed_generation, std::uint32_t consumed_count,
    std::uint64_t diagnostic_generation,
    std::uint32_t diagnostic_command_index,
    DeviceTransportError diagnostic_error) {
    if (request == nullptr)
        return true;
    if (request->state == DeviceRequestState::kCompleted ||
        request->state == DeviceRequestState::kFailed)
        return true;
    if (request->abi_version != kDeviceRequestAbiVersion) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidAbi;
        return true;
    }
    if (request->state != DeviceRequestState::kPending ||
        queue_generation != request->queue_generation ||
        (consumed_generation != 0 &&
         consumed_generation != request->queue_generation)) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidProtocol;
        return true;
    }
    if (diagnostic_generation == request->queue_generation &&
        diagnostic_error != DeviceTransportError::kNone &&
        diagnostic_command_index >= request->command_begin &&
        diagnostic_command_index < request->command_end) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = diagnostic_error;
        return true;
    }
    if (consumed_count >= request->consumed_target) {
        request->state = DeviceRequestState::kCompleted;
        request->terminal_error = DeviceTransportError::kNone;
        return true;
    }
    if (consumed_generation == request->queue_generation) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kCompletionFailure;
        return true;
    }
    return false;
}

DEEP_EP_ASCEND_SIMT_EXECUTION_INLINE bool simt_timeout_request(
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) {
    if (request == nullptr)
        return true;
    if (request->state == DeviceRequestState::kCompleted ||
        request->state == DeviceRequestState::kFailed)
        return true;
    request->state = DeviceRequestState::kFailed;
    request->terminal_error = DeviceTransportError::kCompletionTimeout;
    return true;
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

#if defined(DEEP_EP_ASCEND_HELPERS_OWN_SIMT_GLOBAL)
#undef DEEP_EP_ASCEND_HELPERS_OWN_SIMT_GLOBAL
#undef DEEP_EP_ASCEND_SIMT_GLOBAL
#endif

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
