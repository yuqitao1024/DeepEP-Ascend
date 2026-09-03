#pragma once

#include <cstddef>
#include <cstdint>

#include "cann_compat.hpp"
#include "execution_domain_helpers.hpp"
#include "stage_profile.hpp"
#include "sync_layout.hpp"
#include "transport_commands.hpp"
#include "urma_wqe.hpp"

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE
#include "aicore_intrinsics.hpp"
#endif

namespace deep_ep::ascend::transport::service {

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE
#define DEEP_EP_ASCEND_POLL_INLINE __aicore__ inline
#else
#define DEEP_EP_ASCEND_POLL_INLINE inline constexpr
#endif

DEEP_EP_ASCEND_POLL_INLINE bool barrier_poll_timed_out(
    std::uint64_t start_cycles, std::uint64_t current_cycles,
    std::uint64_t timeout_cycles, std::uint64_t retry_count,
    std::uint64_t retry_limit) {
    if (timeout_cycles == 0)
        return retry_count >= retry_limit;
    const auto elapsed_cycles = current_cycles - start_cycles;
    return elapsed_cycles >= timeout_cycles;
}

#undef DEEP_EP_ASCEND_POLL_INLINE

namespace model {

struct State {
    std::uint32_t member_count = 1;
    std::uint32_t self_member = 0;
    TransportTopology topology{};
    std::uint64_t retry_limit = 1;
    bool completions_enabled = true;
    std::uint32_t consumed_count = 0;
    std::uint32_t executed_count = 0;
    std::uint32_t event_count = 0;
    TransportCommandOpcode events[64]{};
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t outstanding = 0;
    std::uint64_t barrier_generation = 0;
    std::uint64_t retry_count = 0;
    int completion_failure_world_peer = 0;
    std::uint32_t barrier_phase_count = 0;
    TransportTeam barrier_teams[64]{};
    int barrier_world_peers[64]{};
};

inline State make_state(
    std::uint32_t member_count, std::uint32_t self_member,
    std::uint64_t retry_limit) {
    State result;
    result.member_count = member_count;
    result.self_member = self_member;
    (void)build_transport_topology(
        static_cast<int>(self_member), static_cast<int>(member_count),
        static_cast<int>(member_count),
        TransportTopologyKind::kFlatScaleUp, 1, &result.topology);
    result.retry_limit = retry_limit;
    return result;
}

inline State make_state(
    const TransportTopology& topology, std::uint64_t retry_limit) {
    State result;
    result.member_count = static_cast<std::uint32_t>(topology.world_size);
    result.self_member = static_cast<std::uint32_t>(topology.world_rank);
    result.topology = topology;
    result.retry_limit = retry_limit;
    return result;
}

inline bool validate(
    const TransportCommand& command, const State& state,
    DeviceTransportError& error) {
    error = command::validate_for_dispatch(command, state.topology);
    return error == DeviceTransportError::kNone;
}

inline bool drain(
    State& state, DeviceTransportDiagnostic& diagnostic,
    std::uint32_t command_index, TransportCommandOpcode opcode) {
    if (state.outstanding == 0)
        return true;
    if (!state.completions_enabled) {
        state.retry_count = state.retry_limit;
        command::record_first_error(
            diagnostic, DeviceTransportError::kCompletionTimeout,
            command_index, opcode, state.completion_failure_world_peer, 0);
        return false;
    }
    state.completed = state.submitted;
    state.outstanding = 0;
    return true;
}

inline bool execute(
    const TransportCommand* commands, std::uint32_t count, State& state,
    DeviceTransportDiagnostic& diagnostic) {
    if (state.consumed_count > count) {
        command::record_first_error(
            diagnostic, DeviceTransportError::kInvalidQueue,
            state.consumed_count, TransportCommandOpcode::kNone, 0, 0);
        return false;
    }
    for (std::uint32_t index = state.consumed_count; index < count; ++index) {
        const auto& current = commands[index];
        DeviceTransportError error = DeviceTransportError::kNone;
        if (!validate(current, state, error)) {
            command::record_first_error(
                diagnostic, error, index, current.opcode, current.team,
                current.peer, current.world_peer, current.channel);
            return false;
        }

        if (current.opcode == TransportCommandOpcode::kFlush) {
            if (!drain(state, diagnostic, index, current.opcode))
                return false;
        } else if (current.opcode == TransportCommandOpcode::kBarrier) {
            for (const auto team : {
                     TransportTeam::kScaleOut, TransportTeam::kScaleUp}) {
                if (!command::barrier_team_enabled(
                        state.topology, current.options, team))
                    continue;
                for (int peer = 0; peer < state.topology.world_size; ++peer) {
                    if (!command::barrier_peer_in_team(
                            state.topology, team, peer))
                        continue;
                    if (state.barrier_phase_count < 64) {
                        state.barrier_teams[state.barrier_phase_count] = team;
                        state.barrier_world_peers[state.barrier_phase_count] =
                            peer;
                        ++state.barrier_phase_count;
                    }
                    ++state.submitted;
                    ++state.outstanding;
                }
                if (!drain(state, diagnostic, index, current.opcode))
                    return false;
            }
            ++state.barrier_generation;
        } else {
            ++state.submitted;
            ++state.outstanding;
        }

        if (state.event_count < 64)
            state.events[state.event_count++] = current.opcode;
        ++state.executed_count;
        state.consumed_count = index + 1;
    }
    return true;
}

}  // namespace model

inline std::uint64_t signal_offset(
    std::uint32_t member_count, std::uint32_t signal_index,
    std::uint32_t source_member) {
    return sync_layout::signal_offset(
        member_count, signal_index, source_member);
}

inline std::uint64_t fetch_result_offset(std::uint32_t peer) {
    return static_cast<std::uint64_t>(peer) * sizeof(std::uint64_t);
}

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE

namespace detail {

inline constexpr std::uint64_t kDefaultRetryLimit = 1000000;
inline constexpr std::uint32_t kServiceScratchBytes = 512;

__aicore__ inline bool valid_command_route(
    const DeviceTransportContext& context,
    __gm__ const TransportCommand* current) {
    if (current == nullptr)
        return false;
    const auto& topology = context.topology;
    if (topology.abi_version != kTransportTopologyAbiVersion ||
        topology.struct_size != sizeof(TransportTopology) ||
        topology.epoch == 0 || topology.world_size <= 0 ||
        topology.world_rank < 0 ||
        topology.world_rank >= topology.world_size ||
        topology.scale_up_size <= 0 || topology.scale_out_size <= 0 ||
        topology.scale_up_rank < 0 ||
        topology.scale_up_rank >= topology.scale_up_size ||
        topology.scale_out_rank < 0 ||
        topology.scale_out_rank >= topology.scale_out_size ||
        static_cast<std::int64_t>(topology.scale_up_size) *
                topology.scale_out_size != topology.world_size ||
        topology.scale_up_rank !=
            topology.world_rank % topology.scale_up_size ||
        topology.scale_out_rank !=
            topology.world_rank / topology.scale_up_size)
        return false;
    if (topology.kind == TransportTopologyKind::kFlatScaleUp) {
        if (topology.scale_up_size != topology.world_size ||
            topology.scale_out_size != 1)
            return false;
    } else if (topology.kind == TransportTopologyKind::kPhysical2D ||
               topology.kind == TransportTopologyKind::kLogicalSimulation) {
        if (topology.scale_out_size < 2)
            return false;
    } else {
        return false;
    }

    int expected = -1;
    switch (current->team) {
        case TransportTeam::kWorld:
            if (current->peer < 0 || current->peer >= topology.world_size)
                return false;
            expected = current->peer;
            break;
        case TransportTeam::kScaleUp:
            if (current->peer < 0 ||
                current->peer >= topology.scale_up_size)
                return false;
            expected = topology.scale_out_rank * topology.scale_up_size +
                current->peer;
            break;
        case TransportTeam::kScaleOut:
            if (current->peer < 0 ||
                current->peer >= topology.scale_out_size)
                return false;
            expected = current->peer * topology.scale_up_size +
                topology.scale_up_rank;
            break;
        default: return false;
    }
    return expected >= 0 && expected < topology.world_size &&
           expected == current->world_peer;
}

__aicore__ inline DeviceTransportError validate_command(
    const DeviceTransportContext& context,
    __gm__ const TransportCommand* current) {
    if (current == nullptr)
        return DeviceTransportError::kInvalidQueue;
    if (current->opcode == TransportCommandOpcode::kNone)
        return DeviceTransportError::kUnsupportedOperation;
    if (current->channel != 0)
        return DeviceTransportError::kInvalidChannel;

    if (current->opcode == TransportCommandOpcode::kPut ||
        current->opcode == TransportCommandOpcode::kPutValue64 ||
        current->opcode == TransportCommandOpcode::kRemoteAdd64 ||
        current->opcode == TransportCommandOpcode::kSignal) {
        if (!valid_command_route(context, current))
            return DeviceTransportError::kInvalidRank;
    }

    switch (current->opcode) {
        case TransportCommandOpcode::kPut:
            if (current->destination == kNullDeviceAddress ||
                current->source == kNullDeviceAddress || current->bytes == 0)
                return DeviceTransportError::kInvalidAddress;
            if (current->scope != CooperationScope::kParticipant ||
                current->segment != MemorySegment::kDevice ||
                current->options != kDefaultOptions ||
                current->action_kind != RemoteActionKind::kNone)
                return DeviceTransportError::kInvalidProtocol;
            return DeviceTransportError::kNone;
        case TransportCommandOpcode::kPutValue64:
            if (current->destination == kNullDeviceAddress)
                return DeviceTransportError::kInvalidAddress;
            if (current->value_bytes != sizeof(std::uint64_t) ||
                current->options != kDefaultOptions)
                return DeviceTransportError::kInvalidProtocol;
            return DeviceTransportError::kNone;
        case TransportCommandOpcode::kRemoteAdd64:
            if (current->destination == kNullDeviceAddress)
                return DeviceTransportError::kInvalidAddress;
            return current->options == kDefaultOptions ?
                DeviceTransportError::kNone :
                DeviceTransportError::kInvalidProtocol;
        case TransportCommandOpcode::kSignal:
            if ((current->action_kind != RemoteActionKind::kSignalAdd &&
                 current->action_kind !=
                     RemoteActionKind::kSignalIncrement &&
                 current->action_kind != RemoteActionKind::kSignalSet) ||
                current->options != kDefaultOptions)
                return DeviceTransportError::kInvalidProtocol;
            return DeviceTransportError::kNone;
        case TransportCommandOpcode::kFlush:
            if (current->options != kDefaultOptions ||
                current->team != TransportTeam::kWorld ||
                current->peer != 0 || current->world_peer != 0)
                return DeviceTransportError::kInvalidProtocol;
            if (current->scope != CooperationScope::kParticipant &&
                current->scope != CooperationScope::kWorkgroup &&
                current->scope != CooperationScope::kDevice)
                return DeviceTransportError::kInvalidProtocol;
            return DeviceTransportError::kNone;
        case TransportCommandOpcode::kBarrier:
            if (current->options == 0 ||
                (current->options & ~kWorldTeamMask) != 0 ||
                current->team != TransportTeam::kWorld ||
                current->peer != 0 || current->world_peer != 0)
                return DeviceTransportError::kInvalidProtocol;
            if (!command::aicore_barrier_team_enabled(
                    context.topology, current->options,
                    TransportTeam::kScaleOut) &&
                !command::aicore_barrier_team_enabled(
                    context.topology, current->options,
                    TransportTeam::kScaleUp))
                return DeviceTransportError::kInvalidProtocol;
            return DeviceTransportError::kNone;
        default: return DeviceTransportError::kUnsupportedOperation;
    }
}

__aicore__ inline __gm__ StagedTransportContext* staged_context(
    const DeviceTransportContext& context) {
    return reinterpret_cast<__gm__ StagedTransportContext*>(
        context.backend_context);
}

template <bool ProfileEnabled = true>
__aicore__ inline __gm__ TransportStageProfile* profile_buffer(
    const DeviceTransportContext& context) {
    if constexpr (!ProfileEnabled)
        return nullptr;
    auto* staged = staged_context(context);
    if (staged == nullptr)
        return nullptr;
    aicore::flush_cacheline(staged);
    if (staged->abi_version != kTransportCommandAbiVersion ||
        staged->struct_size != sizeof(StagedTransportContext) ||
        staged->cann_compatibility != kStagedTransportCannCompatibility ||
        staged->stage_profile == 0 ||
        staged->stage_profile_bytes != sizeof(TransportStageProfile))
        return nullptr;
    return reinterpret_cast<__gm__ TransportStageProfile*>(
        staged->stage_profile);
}

__aicore__ inline __gm__ TransportCommandQueue* command_queue(
    const DeviceTransportContext& context) {
    if (context.abi_version != kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(DeviceTransportContext) ||
        context.backend_context == 0)
        return nullptr;
    auto* staged = staged_context(context);
    if (staged == nullptr)
        return nullptr;
    aicore::flush_cacheline(staged);
    if (staged->abi_version != kTransportCommandAbiVersion ||
        staged->struct_size != sizeof(StagedTransportContext) ||
        staged->cann_compatibility != kStagedTransportCannCompatibility ||
        staged->command_queue == 0)
        return nullptr;
    return reinterpret_cast<__gm__ TransportCommandQueue*>(
        staged->command_queue);
}

__aicore__ inline __gm__ TransportServiceState* service_state(
    __gm__ TransportCommandQueue* queue) {
    return queue == nullptr ? nullptr :
        reinterpret_cast<__gm__ TransportServiceState*>(queue->service_state);
}

__aicore__ inline __gm__ DeviceTransportDiagnostic* diagnostic(
    __gm__ TransportCommandQueue* queue) {
    return queue == nullptr ? nullptr :
        reinterpret_cast<__gm__ DeviceTransportDiagnostic*>(queue->diagnostic);
}

__aicore__ inline bool valid_registered_queue(
    __gm__ StagedTransportContext* staged,
    __gm__ TransportCommandQueue* queue) {
    if (staged == nullptr || queue == nullptr)
        return false;
    aicore::flush_cacheline(staged);
    if (!command::aicore_valid_staged_context_header(
            staged->abi_version, staged->struct_size,
            staged->cann_compatibility, staged->command_queue) ||
        staged->command_queue != reinterpret_cast<std::uintptr_t>(queue))
        return false;
    aicore::flush_cacheline(queue);
    if (!command::aicore_valid_command_queue_header(
            queue->abi_version, queue->struct_size, queue->commands,
            queue->capacity, queue->count, queue->service_state,
            queue->diagnostic) ||
        !command::aicore_valid_registration_cookie(
            staged->reserved, staged->command_queue, queue->commands,
            queue->service_state, queue->diagnostic, queue->capacity))
        return false;
    auto* state = service_state(queue);
    auto* output = diagnostic(queue);
    aicore::flush_cacheline(state);
    aicore::flush_cacheline(output);
    return command::aicore_valid_service_state_header(
               state->abi_version, state->struct_size) &&
           command::aicore_valid_diagnostic_header(output->abi_version);
}

__aicore__ inline void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    TransportTeam transport_team, int peer, int world_peer,
    std::uint32_t channel, std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    auto* output = diagnostic(queue);
    if (output == nullptr)
        return;
    aicore::flush_cacheline(output);
    if (output->error != DeviceTransportError::kNone)
        return;
    output->command_index = command_index;
    output->opcode = opcode;
    output->peer = static_cast<std::uint32_t>(peer);
    output->world_peer = world_peer;
    output->team = transport_team;
    output->channel = channel;
    output->backend_status = backend_status;
    output->reserved = reserved;
    output->error = error;
    aicore::system_fence();
    aicore::flush_cacheline(output);
}

__aicore__ inline void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    std::uint32_t command_index, TransportCommandOpcode opcode, int world_peer,
    std::uint32_t channel, std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    record_error(
        queue, error, command_index, opcode, TransportTeam::kWorld,
        world_peer, world_peer, channel, backend_status, reserved);
}

__aicore__ inline __gm__ cann_abi::Team* team(
    const DeviceTransportContext& context) {
    return reinterpret_cast<__gm__ cann_abi::Team*>(context.channel_table);
}

__aicore__ inline __gm__ cann_abi::Window* window(
    const DeviceTransportContext& context) {
    return reinterpret_cast<__gm__ cann_abi::Window*>(
        context.peer_address_table);
}

__aicore__ inline __gm__ cann_abi::Channel* resolve_peer(
    __gm__ cann_abi::Team* transport_team, std::uint32_t peer,
    std::uint32_t channel) {
    if (transport_team == nullptr || peer >= transport_team->member_count ||
        transport_team->channel_counts == 0)
        return nullptr;
    auto* counts = reinterpret_cast<__gm__ std::uint32_t*>(
        transport_team->channel_counts);
    std::uint64_t index = channel;
    for (std::uint32_t member = 0; member < peer; ++member)
        index += counts[member];
    if (channel >= counts[peer] || transport_team->channels == 0)
        return nullptr;
    return reinterpret_cast<__gm__ cann_abi::Channel*>(
        transport_team->channels) + index;
}

__aicore__ inline __gm__ cann_abi::RegisteredBuffer* resolve_buffer(
    std::uint64_t table, std::uint32_t count, std::uint64_t address,
    std::uint64_t bytes) {
    auto* buffers = reinterpret_cast<__gm__ cann_abi::RegisteredBuffer*>(table);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto base = buffers[index].address;
        const auto size = buffers[index].bytes;
        if (address >= base && bytes <= size && address - base <= size - bytes)
            return buffers + index;
    }
    return nullptr;
}

struct ResolvedPeer {
    __gm__ cann_abi::Channel* channel = nullptr;
    __gm__ cann_abi::SqContext* sq = nullptr;
    __gm__ cann_abi::CqContext* cq = nullptr;
    std::uint32_t world_peer = 0;
};

__aicore__ inline DeviceTransportError channel_error(
    const DeviceTransportContext& context, std::uint32_t peer,
    std::uint32_t channel_index) {
    auto* resolved_channel = resolve_peer(team(context), peer, channel_index);
    if (resolved_channel == nullptr)
        return DeviceTransportError::kInvalidChannel;
    if (resolved_channel->protocol !=
        static_cast<std::int32_t>(cann_abi::kUbcCtpProtocol))
        return DeviceTransportError::kInvalidProtocol;
    if (resolved_channel->sq_count <= cann_abi::kDefaultQueueIndex ||
        resolved_channel->cq_count <= cann_abi::kDefaultQueueIndex ||
        resolved_channel->sq_contexts == 0 ||
        resolved_channel->cq_contexts == 0)
        return DeviceTransportError::kInvalidQueue;
    return DeviceTransportError::kNone;
}

__aicore__ inline cann_abi::SqContext snapshot_sq(
    __gm__ cann_abi::SqContext* source) {
    cann_abi::SqContext result{};
    result.entry_bytes = source->entry_bytes;
    result.depth = source->depth;
    result.transport_path_id = source->transport_path_id;
    for (std::uint32_t byte = 0; byte < sizeof(result.remote_eid); ++byte)
        result.remote_eid[byte] = source->remote_eid[byte];
    return result;
}

__aicore__ inline cann_abi::RegisteredBuffer snapshot_buffer(
    __gm__ cann_abi::RegisteredBuffer* source) {
    cann_abi::RegisteredBuffer result{};
    result.address = source->address;
    result.bytes = source->bytes;
    result.token_id = source->token_id;
    result.token_value = source->token_value;
    return result;
}

__aicore__ inline ResolvedPeer resolve_context(
    const DeviceTransportContext& context, std::uint32_t peer,
    std::uint32_t channel_index) {
    ResolvedPeer result;
    if (channel_error(context, peer, channel_index) !=
        DeviceTransportError::kNone)
        return result;
    result.channel = resolve_peer(team(context), peer, channel_index);
    result.sq = reinterpret_cast<__gm__ cann_abi::SqContext*>(
        result.channel->sq_contexts) + cann_abi::kDefaultQueueIndex;
    result.cq = reinterpret_cast<__gm__ cann_abi::CqContext*>(
        result.channel->cq_contexts) + cann_abi::kDefaultQueueIndex;
    result.world_peer = peer;
    return result;
}

template <bool ProfileEnabled = true>
__aicore__ inline void update_queue_profile(
    const DeviceTransportContext& context,
    __gm__ TransportStageProfile* profile) {
    if constexpr (!ProfileEnabled)
        return;
    if (profile == nullptr)
        return;
    auto* transport_team = team(context);
    if (transport_team == nullptr || transport_team->channel_counts == 0)
        return;
    auto* counts = reinterpret_cast<__gm__ std::uint32_t*>(
        transport_team->channel_counts);
    TransportQueueDepthSnapshot aggregate{};
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        for (std::uint32_t channel = 0; channel < counts[peer]; ++channel) {
            const auto resolved = resolve_context(context, peer, channel);
            if (resolved.sq == nullptr || resolved.cq == nullptr ||
                resolved.sq->head == 0 || resolved.sq->tail == 0 ||
                resolved.cq->tail == 0)
                continue;
            const auto head_value = aicore::load_device(
                reinterpret_cast<__gm__ std::uint64_t*>(resolved.sq->head));
            const auto submitted = urma::sq_request_count(head_value);
            const auto sq_completed = aicore::load_device(
                reinterpret_cast<__gm__ std::uint32_t*>(resolved.sq->tail));
            const auto cq_completed = aicore::load_device(
                reinterpret_cast<__gm__ std::uint32_t*>(resolved.cq->tail));
            aggregate = command::aicore_merge_queue_depth_snapshots(
                aggregate,
                TransportQueueDepthSnapshot{
                    submitted - sq_completed, submitted - cq_completed});
        }
    }
    profile->sq_depth = aggregate.sq_depth;
    profile->cq_depth = aggregate.cq_depth;
    if (aggregate.sq_depth > profile->sq_high_watermark)
        profile->sq_high_watermark = aggregate.sq_depth;
    if (aggregate.cq_depth > profile->cq_high_watermark)
        profile->cq_high_watermark = aggregate.cq_depth;
}

template <bool ProfileEnabled = true>
__aicore__ inline void finish_drain_profile(
    const DeviceTransportContext& context,
    __gm__ TransportStageProfile* profile, std::uint64_t wait_start) {
    if constexpr (!ProfileEnabled)
        return;
    if (profile == nullptr)
        return;
    profile->wait_cycles += static_cast<std::uint64_t>(
        AscendC::GetSystemCycle()) - wait_start;
    update_queue_profile<ProfileEnabled>(context, profile);
}

__aicore__ inline DeviceTransportError preflight_command_channels(
    const DeviceTransportContext& context,
    __gm__ const TransportCommand* current, int& failed_world_peer) {
    failed_world_peer = current == nullptr ? -1 : current->world_peer;
    if (current == nullptr)
        return DeviceTransportError::kInvalidQueue;
    const bool collective =
        current->opcode == TransportCommandOpcode::kFlush ||
        current->opcode == TransportCommandOpcode::kBarrier;
    if (collective && context.topology.world_size <= 1)
        return DeviceTransportError::kNone;
    auto* transport_team = team(context);
    if (transport_team == nullptr ||
        transport_team->member_count !=
            static_cast<std::uint32_t>(context.topology.world_size) ||
        transport_team->self_member !=
            static_cast<std::uint32_t>(context.topology.world_rank))
        return DeviceTransportError::kInvalidQueue;

    if (collective) {
        for (std::uint32_t peer = 0; peer < transport_team->member_count;
             ++peer) {
            if (peer == transport_team->self_member)
                continue;
            const auto error = channel_error(context, peer, current->channel);
            if (error != DeviceTransportError::kNone) {
                failed_world_peer = static_cast<int>(peer);
                return error;
            }
        }
        return DeviceTransportError::kNone;
    }

    if (current->opcode == TransportCommandOpcode::kPut ||
        current->opcode == TransportCommandOpcode::kPutValue64 ||
        current->opcode == TransportCommandOpcode::kRemoteAdd64 ||
        current->opcode == TransportCommandOpcode::kSignal)
        return channel_error(
            context, static_cast<std::uint32_t>(current->world_peer),
            current->channel);
    return DeviceTransportError::kNone;
}

template <bool ProfileEnabled = true>
__aicore__ inline bool drain_channel(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue, ResolvedPeer peer,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    std::uint64_t retry_limit,
    __gm__ TransportStageProfile* profile,
    const AscendC::LocalTensor<std::uint32_t>& service_scratch) {
    if (peer.channel == nullptr || peer.sq == nullptr || peer.cq == nullptr ||
        peer.sq->head == 0 || peer.sq->tail == 0 || peer.cq->base == 0 ||
        peer.cq->tail == 0 || peer.cq->doorbell == 0 || peer.cq->depth == 0 ||
        peer.cq->entry_bytes != sizeof(cann_abi::UrmaCqe)) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, command_index,
            opcode, static_cast<int>(peer.world_peer), 0);
        return false;
    }
    std::uint64_t wait_start = 0;
    if constexpr (ProfileEnabled) {
        if (profile != nullptr)
            wait_start = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
    }
    const auto head_value = aicore::load_device(
        reinterpret_cast<__gm__ std::uint64_t*>(peer.sq->head));
    const std::uint32_t expected = urma::sq_request_count(head_value);
    std::uint32_t tail = aicore::load_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.cq->tail));
    const auto cqe_scratch = service_scratch[32];
    while (tail != expected) {
        auto* cqe = reinterpret_cast<__gm__ std::uint32_t*>(
            peer.cq->base +
            static_cast<std::uint64_t>(tail % peer.cq->depth) *
                peer.cq->entry_bytes);
        AscendC::GlobalTensor<std::uint32_t> cqe_global;
        cqe_global.SetGlobalBuffer(cqe);
        std::uint64_t retry = 0;
        std::uint32_t word0 = 0;
        do {
            aicore::poll_nop();
            aicore::sync_event<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopy(cqe_scratch, cqe_global, std::uint32_t{16});
            aicore::sync_event<AscendC::HardEvent::MTE2_S>();
            word0 = cqe_scratch.GetValue(0);
            if (urma::cqe_owner_valid((word0 >> 2U) & 1U, tail,
                                      peer.cq->depth))
                break;
            ++retry;
        } while (retry < retry_limit);
        if (retry >= retry_limit) {
            auto* output = diagnostic(queue);
            if (output != nullptr) {
                aicore::flush_cacheline(output);
                if (output->error == DeviceTransportError::kNone) {
                    output->sq_head = urma::sq_position(head_value);
                    output->cq_head = expected;
                    output->cq_tail = tail;
                    output->backend_status = word0;
                    output->reserved = head_value;
                }
            }
            record_error(
                queue, DeviceTransportError::kCompletionTimeout,
                command_index, opcode, static_cast<int>(peer.world_peer), 0);
            finish_drain_profile<ProfileEnabled>(
                context, profile, wait_start);
            return false;
        }
        const auto substatus = (word0 >> 16U) & 0xffU;
        const auto status = (word0 >> 24U) & 0xffU;
        if (status != 0 || substatus != 0) {
            record_error(
                queue, DeviceTransportError::kCompletionFailure,
                command_index, opcode, static_cast<int>(peer.world_peer), 0,
                (status << 8U) | substatus);
            finish_drain_profile<ProfileEnabled>(
                context, profile, wait_start);
            return false;
        }
        ++tail;
    }
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.cq->tail), tail);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.cq->doorbell),
        tail & 0x00ffffffU);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.sq->tail), tail);
    finish_drain_profile<ProfileEnabled>(context, profile, wait_start);
    return true;
}

template <typename Request>
__aicore__ inline void copy_request(
    __gm__ std::uint8_t* queue_base, std::uint32_t depth,
    std::uint32_t entry_bytes, std::uint32_t head,
    const Request& request,
    const AscendC::LocalTensor<std::uint32_t>& wqe_scratch) {
    const auto* source = reinterpret_cast<const std::uint32_t*>(&request);
    constexpr std::uint32_t blocks = sizeof(Request) / 64;
    constexpr std::uint32_t words = sizeof(Request) / sizeof(std::uint32_t);
    for (std::uint32_t word = 0; word < words; ++word)
        wqe_scratch.SetValue(word, source[word]);
    aicore::sync_event<AscendC::HardEvent::S_MTE3>();
    for (std::uint32_t block = 0; block < blocks; ++block) {
        AscendC::GlobalTensor<std::uint32_t> sq_global;
        auto* destination = reinterpret_cast<__gm__ std::uint32_t*>(
            queue_base + static_cast<std::uint64_t>((head + block) % depth) *
                entry_bytes);
        sq_global.SetGlobalBuffer(destination);
        AscendC::DataCopy(sq_global, wqe_scratch[block * 16],
                          std::uint32_t{16});
    }
    aicore::sync_event<AscendC::HardEvent::MTE3_S>();
}

template <bool ProfileEnabled = true, typename Request>
__aicore__ inline bool post_request(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue, ResolvedPeer peer,
    Request request, std::uint32_t command_index,
    TransportCommandOpcode opcode, std::uint64_t retry_limit,
    __gm__ TransportStageProfile* profile,
    const AscendC::LocalTensor<std::uint32_t>& wqe_scratch) {
    constexpr std::uint32_t blocks = sizeof(Request) / 64;
    if (peer.channel == nullptr || peer.sq == nullptr || peer.cq == nullptr ||
        peer.sq->entry_bytes != 64 || peer.sq->depth == 0 ||
        peer.sq->head == 0 || peer.sq->tail == 0 ||
        peer.cq->entry_bytes != sizeof(cann_abi::UrmaCqe) ||
        peer.cq->depth == 0 || peer.cq->tail == 0) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, command_index,
            opcode, 0, 0);
        return false;
    }
    auto head_value = aicore::load_device(
        reinterpret_cast<__gm__ std::uint64_t*>(peer.sq->head));
    auto position = urma::sq_position(head_value);
    auto request_count = urma::sq_request_count(head_value);
    const auto completed = aicore::load_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.sq->tail));
    update_queue_profile<ProfileEnabled>(context, profile);
    if (request_count - completed + 1 >= peer.cq->depth) {
        if (!drain_channel<ProfileEnabled>(
                context, queue, peer, command_index, opcode, retry_limit,
                profile,
                wqe_scratch))
            return false;
        head_value = aicore::load_device(
            reinterpret_cast<__gm__ std::uint64_t*>(peer.sq->head));
        position = urma::sq_position(head_value);
        request_count = urma::sq_request_count(head_value);
    }
    request.sqe.word0 =
        (request.sqe.word0 & 0x7fff0000U) |
        (urma::sq_slot(position, peer.sq->depth) & 0xffffU) |
        (urma::owner_for(position, peer.sq->depth) << 31U);
    copy_request(
        reinterpret_cast<__gm__ std::uint8_t*>(peer.sq->base),
        peer.sq->depth, peer.sq->entry_bytes, position, request,
        wqe_scratch);
    position += blocks;
    ++request_count;
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint64_t*>(peer.sq->head),
        urma::pack_sq_head(position, request_count));
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.sq->doorbell),
        position);
    update_queue_profile<ProfileEnabled>(context, profile);
    return true;
}

__aicore__ inline bool resolve_remote_target(
    const DeviceTransportContext& context, std::uint32_t peer,
    std::uint64_t logical_address, std::uint64_t bytes,
    std::uint64_t& remote_address) {
    auto* transport_window = window(context);
    auto* transport_team = team(context);
    if (transport_window == nullptr || transport_team == nullptr ||
        transport_window->memories == 0 ||
        peer >= transport_team->member_count ||
        logical_address < context.local_window_base)
        return false;
    auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
        transport_window->memories);
    const auto offset = logical_address - context.local_window_base;
    const auto local_size = memories[transport_team->self_member].bytes;
    if (bytes > local_size || offset > local_size - bytes ||
        bytes > memories[peer].bytes || offset > memories[peer].bytes - bytes)
        return false;
    remote_address = memories[peer].address + offset;
    return true;
}

template <bool ProfileEnabled = true>
__aicore__ inline bool drain_all(
    const DeviceTransportContext& context, __gm__ TransportCommandQueue* queue,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    std::uint64_t retry_limit,
    __gm__ TransportStageProfile* profile,
    const AscendC::LocalTensor<std::uint32_t>& service_scratch) {
    if (context.topology.world_size <= 1)
        return true;
    auto* transport_team = team(context);
    if (transport_team == nullptr)
        return false;
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        auto resolved = resolve_context(context, peer, 0);
        if (!drain_channel<ProfileEnabled>(
                context, queue, resolved, command_index, opcode, retry_limit,
                profile,
                service_scratch))
            return false;
    }
    return true;
}

__aicore__ inline std::uint64_t default_fetch_result(
    const DeviceTransportContext& context, std::uint32_t peer) {
    auto* staged = staged_context(context);
    const auto offset =
        static_cast<std::uint64_t>(peer) * sizeof(std::uint64_t);
    if (staged != nullptr && staged->fetch_results != 0 &&
        offset + sizeof(std::uint64_t) <= staged->fetch_result_bytes)
        return staged->fetch_results + offset;
    auto* transport_team = team(context);
    return transport_team == nullptr ? 0 :
        transport_team->shadow_sync_memory.address + offset;
}

template <bool ProfileEnabled = true>
__aicore__ inline bool post_faa(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue, std::uint32_t peer_index,
    std::uint64_t remote_address, std::uint64_t fetch_address,
    std::uint64_t add_value, std::uint32_t command_index,
    TransportCommandOpcode opcode, std::uint64_t retry_limit,
    __gm__ TransportStageProfile* profile,
    const AscendC::LocalTensor<std::uint32_t>& wqe_scratch) {
    auto peer = resolve_context(context, peer_index, 0);
    if (peer.channel == nullptr || fetch_address == 0) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, command_index,
            opcode, peer_index, 0);
        return false;
    }
    auto* remote = resolve_buffer(
        peer.channel->remote_buffers, peer.channel->remote_buffer_count,
        remote_address, sizeof(std::uint64_t));
    if (remote == nullptr) {
        record_error(
            queue, DeviceTransportError::kInvalidAddress, command_index,
            opcode, peer_index, 0);
        return false;
    }
    const auto sq = snapshot_sq(peer.sq);
    const auto remote_buffer = snapshot_buffer(remote);
    const auto request = urma::make_faa64(
        sq, remote_buffer, 0, remote_address,
        fetch_address, add_value);
    return post_request<ProfileEnabled>(
        context, queue, peer, request, command_index, opcode, retry_limit,
        profile, wqe_scratch);
}

template <bool ProfileEnabled = true>
__aicore__ inline bool execute_signal(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current, std::uint32_t command_index,
    std::uint64_t retry_limit,
    __gm__ TransportStageProfile* profile,
    const AscendC::LocalTensor<std::uint32_t>& wqe_scratch) {
    auto* transport_team = team(context);
    if (transport_team == nullptr || current->world_peer < 0 ||
        (transport_team != nullptr &&
         static_cast<std::uint32_t>(current->world_peer) >=
             transport_team->member_count)) {
        record_error(
            queue, DeviceTransportError::kInvalidRank, command_index,
            current->opcode, current->team, current->peer,
            current->world_peer, current->channel, 1);
        return false;
    }
    std::uint64_t remote_address = 0;
    if (current->action_kind == RemoteActionKind::kSignalIncrement ||
        current->action_kind == RemoteActionKind::kSignalSet) {
        std::uint32_t failure_stage = 0;
        std::uint64_t failure_value = 0;
        if (transport_team->signal_count !=
                sync_layout::kWorldTeamSignalCount) {
            failure_stage = 2;
            failure_value = transport_team->signal_count;
        } else if (transport_team->counter_count !=
                       sync_layout::kWorldTeamCounterCount) {
            failure_stage = 3;
            failure_value = transport_team->counter_count;
        } else if (transport_team->barrier_count <
                       sync_layout::kWorldTeamBarrierCount) {
            failure_stage = 4;
            failure_value = transport_team->barrier_count;
        } else if (current->signal_index >=
                       sync_layout::kLogicalSignalCount) {
            failure_stage = 5;
            failure_value = current->signal_index;
        } else if (transport_team->remote_sync_memories == 0) {
            failure_stage = 6;
        }
        if (failure_stage != 0) {
            record_error(
                queue, DeviceTransportError::kInvalidProtocol,
                command_index, current->opcode, current->team, current->peer,
                current->world_peer, current->channel, failure_stage,
                failure_value);
            return false;
        }
        auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
            transport_team->remote_sync_memories);
        const auto offset =
            (static_cast<std::uint64_t>(current->signal_index) *
                 transport_team->member_count +
             transport_team->self_member) * sizeof(std::uint64_t);
        remote_address = memories[current->world_peer].address + offset;
    } else if (current->action_kind == RemoteActionKind::kSignalAdd) {
        const auto logical = context.local_window_base +
            current->symmetric_offset;
        if (!resolve_remote_target(
                context, static_cast<std::uint32_t>(current->world_peer), logical,
                sizeof(std::uint64_t), remote_address)) {
            record_error(
                queue, DeviceTransportError::kInvalidAddress, command_index,
                current->opcode, current->team, current->peer,
                current->world_peer, current->channel, 7, logical);
            return false;
        }
    } else {
        record_error(
            queue, DeviceTransportError::kUnsupportedOperation,
            command_index, current->opcode, current->team, current->peer,
            current->world_peer, current->channel, 8,
            static_cast<std::uint64_t>(current->action_kind));
        return false;
    }
    if (current->action_kind == RemoteActionKind::kSignalSet) {
        auto peer = resolve_context(
            context, static_cast<std::uint32_t>(current->world_peer), 0);
        auto* remote = peer.channel == nullptr ? nullptr : resolve_buffer(
            peer.channel->remote_buffers, peer.channel->remote_buffer_count,
            remote_address, sizeof(std::uint64_t));
        if (remote == nullptr) {
            record_error(
                queue, DeviceTransportError::kInvalidAddress, command_index,
                current->opcode, current->team, current->peer,
                current->world_peer, current->channel, 9, remote_address);
            return false;
        }
        const auto request = urma::make_inline_write64(
            snapshot_sq(peer.sq), snapshot_buffer(remote), 0,
            remote_address, current->value);
        return post_request<ProfileEnabled>(
            context, queue, peer, request, command_index, current->opcode,
            retry_limit, profile, wqe_scratch);
    }
    return post_faa<ProfileEnabled>(
        context, queue, static_cast<std::uint32_t>(current->world_peer),
        remote_address,
        default_fetch_result(
            context, static_cast<std::uint32_t>(current->world_peer)),
        current->value, command_index, current->opcode, retry_limit,
        profile, wqe_scratch);
}

template <bool ProfileEnabled = true>
__aicore__ inline bool execute_barrier(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current, std::uint32_t command_index,
    __gm__ TransportServiceState* state, std::uint64_t retry_limit,
    __gm__ TransportStageProfile* profile,
    const AscendC::LocalTensor<std::uint32_t>& wqe_scratch) {
    if (context.topology.world_size <= 1) {
        ++state->barrier_generation;
        return true;
    }
    auto* transport_team = team(context);
    if (transport_team == nullptr ||
        transport_team->signal_count !=
            sync_layout::kWorldTeamSignalCount ||
        transport_team->counter_count !=
            sync_layout::kWorldTeamCounterCount ||
        transport_team->barrier_count <
            sync_layout::kWorldTeamBarrierCount ||
        transport_team->remote_sync_memories == 0)
        return false;
    auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
        transport_team->remote_sync_memories);
    const auto generation = state->barrier_generation + 1;
    std::uint32_t phase_index = 0;
    for (const auto phase_team : {
             TransportTeam::kScaleOut, TransportTeam::kScaleUp}) {
        if (!command::aicore_barrier_team_enabled(
                context.topology, current->options, phase_team))
            continue;
        if (phase_index >= kTransportProfileBarrierPhaseCount)
            return false;
        const std::uint32_t barrier_index =
            phase_team == TransportTeam::kScaleOut ?
                sync_layout::kScaleOutBarrierIndex :
                sync_layout::kScaleUpBarrierIndex;
        std::uint64_t issue_start = 0;
        if constexpr (ProfileEnabled)
            issue_start = static_cast<std::uint64_t>(
                AscendC::GetSystemCycle());
        std::uint64_t peer_count = 0;
        for (std::uint32_t peer = 0;
             peer < transport_team->member_count; ++peer) {
            if (!command::aicore_barrier_peer_in_team(
                    context.topology, phase_team, static_cast<int>(peer)))
                continue;
            ++peer_count;
            auto* peer_profile = profile == nullptr ? nullptr :
                &profile->barrier_peers[phase_index][peer];
            if constexpr (ProfileEnabled) {
                if (peer_profile != nullptr) {
                    peer_profile->world_peer = peer;
                    peer_profile->valid = 1;
                    peer_profile->issue_start_cycles =
                        static_cast<std::uint64_t>(AscendC::GetSystemCycle());
                }
            }
            const auto offset = sync_layout::aicore_barrier_offset(
                transport_team->member_count, barrier_index,
                transport_team->self_member);
            const bool posted = post_faa<ProfileEnabled>(
                    context, queue, peer, memories[peer].address + offset,
                    transport_team->shadow_sync_memory.address + offset, 1,
                    command_index, current->opcode, retry_limit, profile,
                    wqe_scratch);
            if constexpr (ProfileEnabled) {
                if (peer_profile != nullptr)
                    peer_profile->issue_end_cycles =
                        static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            }
            if (!posted)
                return false;
        }
        if constexpr (ProfileEnabled) {
            if (profile != nullptr) {
                const auto issue_end = static_cast<std::uint64_t>(
                    AscendC::GetSystemCycle());
                profile->barrier_issue_cycles += issue_end - issue_start;
                profile->barrier_peer_count += peer_count;
            }
        }
        std::uint64_t drain_start = 0;
        if constexpr (ProfileEnabled)
            drain_start = static_cast<std::uint64_t>(
                AscendC::GetSystemCycle());
        for (std::uint32_t peer = 0;
             peer < transport_team->member_count; ++peer) {
            if (peer == transport_team->self_member)
                continue;
            const bool in_phase = command::aicore_barrier_peer_in_team(
                context.topology, phase_team, static_cast<int>(peer));
            auto* peer_profile = profile == nullptr ? nullptr :
                &profile->barrier_peers[phase_index][peer];
            if constexpr (ProfileEnabled) {
                if (in_phase && peer_profile != nullptr)
                    peer_profile->drain_start_cycles =
                        static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            }
            const auto resolved = resolve_context(context, peer, 0);
            const bool drained = drain_channel<ProfileEnabled>(
                context, queue, resolved, command_index, current->opcode,
                retry_limit, profile, wqe_scratch);
            if constexpr (ProfileEnabled) {
                if (in_phase && peer_profile != nullptr)
                    peer_profile->drain_end_cycles =
                        static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            }
            if (!drained)
                return false;
        }
        if constexpr (ProfileEnabled) {
            if (profile != nullptr) {
                const auto drain_end = static_cast<std::uint64_t>(
                    AscendC::GetSystemCycle());
                profile->barrier_drain_cycles += drain_end - drain_start;
            }
        }
        // Poll all outstanding peers in one loop.  The previous per-peer
        // loop accumulated independent arrival delays when an early peer was
        // slow, even though the later peers had already completed.
        std::uint64_t pending_peers = 0;
        for (std::uint32_t peer = 0;
             peer < transport_team->member_count; ++peer) {
            if (!command::aicore_barrier_peer_in_team(
                    context.topology, phase_team, static_cast<int>(peer)))
                continue;
            pending_peers |= std::uint64_t{1} << peer;
        }
        const auto start_cycles = static_cast<std::uint64_t>(
            AscendC::GetSystemCycle());
        std::uint64_t first_observation = 0;
        std::uint64_t retry = 0;
        std::uint32_t pending_clear_order = 0;
        while (pending_peers != 0) {
            aicore::poll_nop();
            std::uint64_t observed_pending = 0;
            for (std::uint32_t peer = 0;
                 peer < transport_team->member_count; ++peer) {
                if ((pending_peers & (std::uint64_t{1} << peer)) == 0)
                    continue;
                const auto offset = sync_layout::aicore_barrier_offset(
                    transport_team->member_count, barrier_index, peer);
                auto* signal = reinterpret_cast<__gm__ std::uint64_t*>(
                    memories[transport_team->self_member].address + offset);
                const auto observation_cycles = static_cast<std::uint64_t>(
                    AscendC::GetSystemCycle());
                if (first_observation == 0)
                    first_observation = observation_cycles;
                auto* peer_profile = profile == nullptr ? nullptr :
                    &profile->barrier_peers[phase_index][peer];
                if constexpr (ProfileEnabled) {
                    if (peer_profile != nullptr &&
                        peer_profile->first_observation_cycles == 0)
                        peer_profile->first_observation_cycles =
                            observation_cycles;
                }
                // Barrier counters are remote FAA results in GM. A direct
                // cache-bypassing scalar read avoids an MTE2/UB round trip on
                // every poll while retaining the same visibility guarantee.
                if (aicore::load_published(signal) < generation)
                    observed_pending |= std::uint64_t{1} << peer;
                else if constexpr (ProfileEnabled) {
                    if (peer_profile != nullptr &&
                        peer_profile->ready_cycles == 0) {
                        peer_profile->ready_cycles =
                            static_cast<std::uint64_t>(
                                AscendC::GetSystemCycle());
                        peer_profile->pending_clear_order =
                            ++pending_clear_order;
                    }
                }
            }
            pending_peers = observed_pending;
            if (pending_peers == 0)
                break;
            ++retry;
            const auto current_cycles = static_cast<std::uint64_t>(
                AscendC::GetSystemCycle());
            if (barrier_poll_timed_out(
                    start_cycles, current_cycles, current->timeout_cycles,
                    retry, retry_limit)) {
                std::uint32_t timeout_peer = 0;
                for (; timeout_peer < transport_team->member_count &&
                     timeout_peer < 64;
                     ++timeout_peer) {
                    if ((pending_peers &
                         (std::uint64_t{1} << timeout_peer)) != 0)
                        break;
                }
                const int logical_peer =
                    phase_team == TransportTeam::kScaleOut ?
                        static_cast<int>(timeout_peer) /
                            context.topology.scale_up_size :
                        static_cast<int>(timeout_peer) %
                            context.topology.scale_up_size;
                record_error(
                    queue, DeviceTransportError::kCompletionTimeout,
                    command_index, current->opcode, phase_team,
                    logical_peer, static_cast<int>(timeout_peer), 0);
                return false;
            }
        }
        if constexpr (ProfileEnabled) {
            if (profile != nullptr) {
                const auto poll_end = static_cast<std::uint64_t>(
                    AscendC::GetSystemCycle());
                profile->barrier_poll_elapsed_cycles +=
                    poll_end - start_cycles;
                profile->barrier_poll_iterations += retry;
                if (first_observation >= start_cycles)
                    profile->barrier_first_observation_cycles +=
                        first_observation - start_cycles;
            }
        }
        ++phase_index;
    }
    if constexpr (ProfileEnabled) {
        if (profile != nullptr) {
            const auto completion_start = static_cast<std::uint64_t>(
                AscendC::GetSystemCycle());
            state->barrier_generation = generation;
            const auto completion_end = static_cast<std::uint64_t>(
                AscendC::GetSystemCycle());
            profile->barrier_completion_cycles +=
                completion_end - completion_start;
            return true;
        }
    }
    state->barrier_generation = generation;
    return true;
}

}  // namespace detail

__aicore__ inline void begin_profile(
    const DeviceTransportContext& context, TransportProfileOperation operation,
    std::uint64_t generation) {
    auto* profile = detail::profile_buffer(context);
    if (profile == nullptr)
        return;
    profile->abi_version = kTransportStageProfileAbiVersion;
    profile->struct_size = sizeof(TransportStageProfile);
    profile->operation = operation;
    profile->flags = 0;
    profile->generation = generation;
    profile->completion_generation = 0;
    profile->valid_stage_mask = 0;
    profile->command_count = 0;
    profile->put_command_count = 0;
    profile->sq_depth = 0;
    profile->cq_depth = 0;
    profile->sq_high_watermark = 0;
    profile->cq_high_watermark = 0;
    profile->command_bytes = 0;
    profile->service_start_cycles = 0;
    profile->service_end_cycles = 0;
    profile->wait_cycles = 0;
    profile->payload_command_cycles = 0;
    profile->control_command_cycles = 0;
    profile->flush_command_cycles = 0;
    profile->barrier_command_cycles = 0;
    profile->barrier_poll_cycles = 0;
    profile->barrier_issue_cycles = 0;
    profile->barrier_drain_cycles = 0;
    profile->barrier_poll_iterations = 0;
    profile->barrier_peer_count = 0;
    profile->barrier_first_observation_cycles = 0;
    profile->barrier_completion_cycles = 0;
    profile->barrier_poll_elapsed_cycles = 0;
    aicore::system_fence();
    aicore::flush_stage_profile_header(profile);
}

__aicore__ inline void complete_profile(
    const DeviceTransportContext& context, std::uint64_t generation) {
    auto* profile = detail::profile_buffer(context);
    if (profile == nullptr || profile->generation != generation)
        return;
    profile->completion_generation = generation;
    aicore::system_fence();
    for (std::uint32_t phase = 0;
         phase < kTransportProfileBarrierPhaseCount; ++phase)
        for (std::uint32_t peer = 0;
             peer < kTransportProfileMaxBarrierPeers; ++peer)
            if (profile->barrier_peers[phase][peer].valid != 0)
                aicore::flush_cacheline(
                    &profile->barrier_peers[phase][peer]);
    aicore::flush_stage_profile_header(profile);
}

__aicore__ inline std::uint64_t device_stage_profile_pipeline_mask(
    TransportProfileOperation operation, bool release_ablation) {
    if (operation == TransportProfileOperation::kDispatch)
        return release_ablation ? kTransportDispatchReleaseAblationStageMask :
                                  kTransportDispatchPipelineStageMask;
    if (operation == TransportProfileOperation::kCombine)
        return release_ablation ? kTransportCombineReleaseAblationStageMask :
                                  kTransportCombinePipelineStageMask;
    return 0;
}

__aicore__ inline std::uint64_t stage_profile_completed_mask(
    TransportProfileOperation operation, std::uint32_t stage,
    bool release_ablation) {
    if (stage == 0)
        return kTransportStageProfileFullMask;
    if (operation == TransportProfileOperation::kDispatch && stage == 13)
        return device_stage_profile_pipeline_mask(operation, release_ablation);
    if (operation == TransportProfileOperation::kCombine && stage == 11)
        return device_stage_profile_pipeline_mask(operation, release_ablation);
    return 0;
}

template <bool ProfileEnabled = true>
__aicore__ inline void record_stage_start(
    const DeviceTransportContext& context,
    TransportProfileOperation operation, std::uint64_t generation,
    std::uint32_t stage, std::uint32_t block, std::uint32_t block_count) {
    if constexpr (!ProfileEnabled)
        return;
    auto* profile = detail::profile_buffer<ProfileEnabled>(context);
    if (profile == nullptr || stage >= kTransportProfileStageCount ||
        block >= kTransportProfileMaxBlocks)
        return;
    profile->operation = operation;
    profile->generation = generation;
    const bool release_ablation_stage =
        (operation == TransportProfileOperation::kDispatch && stage >= 14) ||
        (operation == TransportProfileOperation::kCombine && stage >= 12);
    if (block == 0 && release_ablation_stage) {
        profile->flags |= kTransportStageProfileReleaseAblation;
        aicore::flush_stage_profile_header(profile);
    }
    auto* cycles = &profile->stages[stage].blocks[block];
    cycles->start = record_transport_stage_start(
        cycles->start,
        static_cast<std::uint64_t>(AscendC::GetSystemCycle()));
    if (block == 0)
        profile->stages[stage].block_count = block_count;
}

template <bool ProfileEnabled = true>
__aicore__ inline void record_stage_end(
    const DeviceTransportContext& context, TransportProfileOperation operation,
    std::uint64_t generation,
    std::uint32_t stage, std::uint32_t block, bool complete_operation) {
    if constexpr (!ProfileEnabled)
        return;
    auto* profile = detail::profile_buffer<ProfileEnabled>(context);
    if (profile == nullptr || profile->generation != generation ||
        stage >= kTransportProfileStageCount ||
        block >= kTransportProfileMaxBlocks)
        return;
    auto* cycles = &profile->stages[stage].blocks[block];
    cycles->end = record_transport_stage_end(
        cycles->start, cycles->end,
        static_cast<std::uint64_t>(AscendC::GetSystemCycle()));
    aicore::system_fence();
    aicore::flush_cacheline(&profile->stages[stage].blocks[block]);
    if (block == 0)
        aicore::flush_cacheline(&profile->stages[stage]);
    if (block == 0 && complete_operation) {
        profile->valid_stage_mask = stage_profile_completed_mask(
            operation, stage,
            (profile->flags & kTransportStageProfileReleaseAblation) != 0);
        complete_profile(context, generation);
    }
}

__aicore__ inline void reset(
    const DeviceTransportContext& context, std::uint64_t generation) {
    auto* staged = detail::staged_context(context);
    auto* queue = detail::command_queue(context);
    if (!detail::valid_registered_queue(staged, queue))
        return;
    auto* state = detail::service_state(queue);
    auto* output = detail::diagnostic(queue);
    state->consumed_count = 0;
    state->active = 0;
    state->consumed_generation = 0;
    if (state->default_retry_limit == 0)
        state->default_retry_limit = detail::kDefaultRetryLimit;
    output->error = DeviceTransportError::kNone;
    output->command_index = 0;
    output->opcode = TransportCommandOpcode::kNone;
    output->peer = 0;
    output->channel = 0;
    output->sq_head = 0;
    output->cq_head = 0;
    output->cq_tail = 0;
    output->backend_status = 0;
    output->generation = generation;
    output->world_peer = 0;
    output->team = TransportTeam::kWorld;
    output->reserved0[0] = 0;
    output->reserved0[1] = 0;
    output->reserved0[2] = 0;
    output->reserved = 0;
    queue->count = 0;
    queue->generation = generation;
    aicore::system_fence();
    aicore::flush_cacheline(state);
    aicore::flush_cacheline(output);
    aicore::flush_cacheline(queue);
}

template <bool ProfileEnabled = true>
__aicore__ inline void execute_body(const DeviceTransportContext& context) {
    auto* staged = detail::staged_context(context);
    auto* queue = detail::command_queue(context);
    if (!detail::valid_registered_queue(staged, queue))
        return;
    auto* state = detail::service_state(queue);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratch_buffer;
    if (!pipe.InitBuffer(scratch_buffer, detail::kServiceScratchBytes)) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidQueue, 0,
            TransportCommandOpcode::kNone, 0, 0);
        return;
    }
    const auto wqe_scratch = scratch_buffer.Get<std::uint32_t>();
    const std::uint64_t retry_limit = state->default_retry_limit == 0 ?
        detail::kDefaultRetryLimit : state->default_retry_limit;
    aicore::flush_cacheline(queue);
    const std::uint32_t command_begin = state->consumed_count;
    if (command_begin > queue->count) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidQueue, command_begin,
            TransportCommandOpcode::kNone, 0, 0);
        return;
    }
    state->active = 1;
    state->consumed_generation = 0;
    aicore::system_fence();
    aicore::flush_cacheline(state);
    auto* commands = reinterpret_cast<__gm__ TransportCommand*>(
        queue->commands);
    const std::uint32_t count = queue->count;
    __gm__ TransportStageProfile* profile = nullptr;
    if constexpr (ProfileEnabled)
        profile = detail::profile_buffer<ProfileEnabled>(context);
    if constexpr (ProfileEnabled) {
        if (profile != nullptr) {
            profile->service_start_cycles =
                record_transport_stage_start(
                    profile->service_start_cycles,
                    static_cast<std::uint64_t>(AscendC::GetSystemCycle()));
        }
    }

    for (std::uint32_t index = command_begin; index < count; ++index) {
        auto* current = commands + index;
        aicore::flush_cacheline(current);
        if constexpr (ProfileEnabled) {
            if (profile != nullptr) {
                ++profile->command_count;
                profile->command_bytes +=
                    command::aicore_profile_payload_bytes(
                        current->opcode, current->bytes);
                if (current->opcode == TransportCommandOpcode::kPut)
                    ++profile->put_command_count;
            }
        }
        bool success = true;
        const auto command_error = detail::validate_command(context, current);
        if (command_error != DeviceTransportError::kNone) {
            detail::record_error(
                queue, command_error, index, current->opcode, current->team,
                current->peer, current->world_peer, current->channel);
            success = false;
        }

        int failed_world_peer = current->world_peer;
        const auto channel_error = success ?
            detail::preflight_command_channels(
                context, current, failed_world_peer) :
            DeviceTransportError::kNone;
        if (channel_error != DeviceTransportError::kNone) {
            const bool collective =
                current->opcode == TransportCommandOpcode::kFlush ||
                current->opcode == TransportCommandOpcode::kBarrier;
            detail::record_error(
                queue, channel_error, index, current->opcode,
                collective ? TransportTeam::kWorld : current->team,
                collective ? failed_world_peer : current->peer,
                failed_world_peer, current->channel);
            success = false;
        }

        if (!success) {
            // Validation is complete before any transport submission.
        } else if (current->opcode == TransportCommandOpcode::kFlush) {
            success = detail::drain_all<ProfileEnabled>(
                context, queue, index, current->opcode, retry_limit,
                profile, wqe_scratch);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidQueue, index,
                    current->opcode, current->team, current->peer,
                    current->world_peer, current->channel);
        } else if (current->opcode == TransportCommandOpcode::kBarrier) {
            success = detail::execute_barrier<ProfileEnabled>(
                context, queue, current, index, state, retry_limit,
                profile, wqe_scratch);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidQueue, index,
                    current->opcode, current->team, current->peer,
                    current->world_peer, current->channel);
        } else if (current->opcode == TransportCommandOpcode::kSignal) {
            success = detail::execute_signal<ProfileEnabled>(
                context, queue, current, index, retry_limit, profile,
                wqe_scratch);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidAddress, index,
                    current->opcode, current->team, current->peer,
                    current->world_peer, current->channel);
        } else {
            auto peer = detail::resolve_context(
                context, static_cast<std::uint32_t>(current->world_peer),
                current->channel);
            std::uint64_t remote_address = 0;
            const std::uint64_t bytes =
                current->opcode == TransportCommandOpcode::kPut ?
                    current->bytes : sizeof(std::uint64_t);
            if (peer.channel == nullptr ||
                !detail::resolve_remote_target(
                    context, static_cast<std::uint32_t>(current->world_peer),
                    current->destination, bytes, remote_address)) {
                detail::record_error(
                    queue, DeviceTransportError::kInvalidAddress, index,
                    current->opcode, current->team, current->peer,
                    current->world_peer, current->channel);
                success = false;
            } else if (current->opcode == TransportCommandOpcode::kPut) {
                auto* local = detail::resolve_buffer(
                    peer.channel->local_buffers,
                    peer.channel->local_buffer_count, current->source,
                    current->bytes);
                auto* remote = detail::resolve_buffer(
                    peer.channel->remote_buffers,
                    peer.channel->remote_buffer_count, remote_address,
                    current->bytes);
                if (local == nullptr || remote == nullptr ||
                    current->bytes > 0xffffffffULL) {
                    detail::record_error(
                        queue, DeviceTransportError::kInvalidAddress, index,
                        current->opcode, current->team, current->peer,
                        current->world_peer, current->channel);
                    success = false;
                } else {
                    const auto sq = detail::snapshot_sq(peer.sq);
                    const auto remote_buffer =
                        detail::snapshot_buffer(remote);
                    const auto request = urma::make_write(
                        sq, remote_buffer, 0,
                        remote_address, current->source,
                        static_cast<std::uint32_t>(current->bytes),
                        local->token_id);
                    success = detail::post_request<ProfileEnabled>(
                        context, queue, peer, request, index, current->opcode,
                        retry_limit, profile, wqe_scratch);
                }
            } else if (current->opcode ==
                       TransportCommandOpcode::kPutValue64) {
                auto* remote = detail::resolve_buffer(
                    peer.channel->remote_buffers,
                    peer.channel->remote_buffer_count, remote_address,
                    sizeof(std::uint64_t));
                if (remote == nullptr) {
                    detail::record_error(
                        queue, DeviceTransportError::kInvalidAddress, index,
                        current->opcode, current->team, current->peer,
                        current->world_peer, current->channel);
                    success = false;
                } else {
                    const auto sq = detail::snapshot_sq(peer.sq);
                    const auto remote_buffer =
                        detail::snapshot_buffer(remote);
                    const auto request = urma::make_inline_write64(
                        sq, remote_buffer, 0,
                        remote_address, current->value);
                    success = detail::post_request<ProfileEnabled>(
                        context, queue, peer, request, index, current->opcode,
                        retry_limit, profile, wqe_scratch);
                }
            } else if (current->opcode ==
                       TransportCommandOpcode::kRemoteAdd64) {
                success = detail::post_faa<ProfileEnabled>(
                    context, queue,
                    static_cast<std::uint32_t>(current->world_peer),
                    remote_address,
                    detail::default_fetch_result(
                        context,
                        static_cast<std::uint32_t>(current->world_peer)),
                    current->value, index, current->opcode, retry_limit,
                    profile, wqe_scratch);
            } else {
                detail::record_error(
                    queue, DeviceTransportError::kUnsupportedOperation,
                    index, current->opcode, current->team, current->peer,
                    current->world_peer, current->channel);
                success = false;
            }
        }
        if (!success)
            break;
        state->consumed_count = index + 1;
    }
    state->active = 0;
    state->consumed_generation = queue->generation;
    if constexpr (ProfileEnabled) {
        if (profile != nullptr) {
            profile->service_end_cycles =
                record_transport_stage_end(
                    profile->service_start_cycles,
                    profile->service_end_cycles,
                    static_cast<std::uint64_t>(AscendC::GetSystemCycle()));
            detail::update_queue_profile<ProfileEnabled>(context, profile);
            aicore::flush_stage_profile_header(profile);
        }
    }
    aicore::system_fence();
    aicore::flush_cacheline(state);
}

__aicore__ static __attribute__((noinline)) void execute_profiled(
    const DeviceTransportContext& context) {
    execute_body<true>(context);
}

template <bool ProfileEnabled = true>
__aicore__ inline void execute(const DeviceTransportContext& context) {
    if constexpr (ProfileEnabled)
        execute_profiled(context);
    else
        execute_body<false>(context);
}

#endif

}  // namespace deep_ep::ascend::transport::service
