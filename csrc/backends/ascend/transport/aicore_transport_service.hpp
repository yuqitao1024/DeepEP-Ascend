#pragma once

#include <cstddef>
#include <cstdint>

#include "cann_compat.hpp"
#include "transport_commands.hpp"
#include "urma_wqe.hpp"

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE
#include "aicore_intrinsics.hpp"
#endif

namespace deep_ep::ascend::transport::service {

namespace model {

struct State {
    std::uint32_t member_count = 1;
    std::uint32_t self_member = 0;
    std::uint64_t retry_limit = 1;
    bool completions_enabled = true;
    std::uint32_t executed_count = 0;
    std::uint32_t event_count = 0;
    TransportCommandOpcode events[64]{};
    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t outstanding = 0;
    std::uint64_t barrier_generation = 0;
    std::uint64_t retry_count = 0;
};

inline State make_state(
    std::uint32_t member_count, std::uint32_t self_member,
    std::uint64_t retry_limit) {
    State result;
    result.member_count = member_count;
    result.self_member = self_member;
    result.retry_limit = retry_limit;
    return result;
}

inline bool is_remote_operation(TransportCommandOpcode opcode) {
    return opcode == TransportCommandOpcode::kPut ||
           opcode == TransportCommandOpcode::kPutValue64 ||
           opcode == TransportCommandOpcode::kRemoteAdd64 ||
           opcode == TransportCommandOpcode::kSignal;
}

inline bool validate(
    const TransportCommand& command, const State& state,
    DeviceTransportError& error) {
    if (command.opcode == TransportCommandOpcode::kNone) {
        error = DeviceTransportError::kUnsupportedOperation;
        return false;
    }
    if (command.channel != 0) {
        error = DeviceTransportError::kInvalidChannel;
        return false;
    }
    if (is_remote_operation(command.opcode)) {
        if (command.team == TransportTeam::kScaleOut) {
            error = DeviceTransportError::kUnsupportedOperation;
            return false;
        }
        if (command.peer < 0 ||
            static_cast<std::uint32_t>(command.peer) >= state.member_count) {
            error = DeviceTransportError::kInvalidRank;
            return false;
        }
    }
    if ((command.opcode == TransportCommandOpcode::kPut ||
         command.opcode == TransportCommandOpcode::kPutValue64 ||
         command.opcode == TransportCommandOpcode::kRemoteAdd64) &&
        command.destination == kNullDeviceAddress) {
        error = DeviceTransportError::kInvalidAddress;
        return false;
    }
    if (command.opcode == TransportCommandOpcode::kPut &&
        (command.source == kNullDeviceAddress || command.bytes == 0)) {
        error = DeviceTransportError::kInvalidAddress;
        return false;
    }
    if (command.opcode == TransportCommandOpcode::kPutValue64 &&
        command.value_bytes != sizeof(std::uint64_t)) {
        error = DeviceTransportError::kUnsupportedOperation;
        return false;
    }
    if (command.opcode == TransportCommandOpcode::kSignal &&
        command.action_kind != RemoteActionKind::kSignalAdd &&
        command.action_kind != RemoteActionKind::kSignalIncrement) {
        error = DeviceTransportError::kUnsupportedOperation;
        return false;
    }
    return true;
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
            command_index, opcode, 0, 0);
        return false;
    }
    state.completed = state.submitted;
    state.outstanding = 0;
    return true;
}

inline bool execute(
    const TransportCommand* commands, std::uint32_t count, State& state,
    DeviceTransportDiagnostic& diagnostic) {
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& current = commands[index];
        DeviceTransportError error = DeviceTransportError::kNone;
        if (!validate(current, state, error)) {
            command::record_first_error(
                diagnostic, error, index, current.opcode, current.peer,
                current.channel);
            return false;
        }

        if (current.opcode == TransportCommandOpcode::kFlush) {
            if (!drain(state, diagnostic, index, current.opcode))
                return false;
        } else if (current.opcode == TransportCommandOpcode::kBarrier) {
            state.submitted += state.member_count > 0 ?
                state.member_count - 1 : 0;
            state.outstanding = state.submitted - state.completed;
            if (!drain(state, diagnostic, index, current.opcode))
                return false;
            ++state.barrier_generation;
        } else {
            ++state.submitted;
            ++state.outstanding;
        }

        if (state.event_count < 64)
            state.events[state.event_count++] = current.opcode;
        ++state.executed_count;
    }
    return true;
}

}  // namespace model

inline std::uint64_t signal_offset(
    std::uint32_t member_count, std::uint32_t signal_index,
    std::uint32_t source_member) {
    return (static_cast<std::uint64_t>(signal_index) * member_count +
            source_member) * sizeof(std::uint64_t);
}

inline std::uint64_t fetch_result_offset(std::uint32_t peer) {
    return static_cast<std::uint64_t>(peer) * sizeof(std::uint64_t);
}

#if defined(DEEP_EP_ASCEND_AICORE_URMA_SERVICE) && \
    DEEP_EP_ASCEND_AICORE_URMA_SERVICE

namespace detail {

inline constexpr std::uint64_t kDefaultRetryLimit = 1000000;

__aicore__ inline __gm__ StagedTransportContext* staged_context(
    const DeviceTransportContext& context) {
    return reinterpret_cast<__gm__ StagedTransportContext*>(
        context.backend_context);
}

__aicore__ inline __gm__ TransportCommandQueue* command_queue(
    const DeviceTransportContext& context) {
    auto* staged = staged_context(context);
    if (staged == nullptr)
        return nullptr;
    aicore::flush_cacheline(staged);
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

__aicore__ inline void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    std::uint32_t command_index, TransportCommandOpcode opcode, int peer,
    std::uint32_t channel, std::uint32_t backend_status = 0) {
    auto* output = diagnostic(queue);
    if (output == nullptr)
        return;
    aicore::flush_cacheline(output);
    if (output->error != DeviceTransportError::kNone)
        return;
    output->command_index = command_index;
    output->opcode = opcode;
    output->peer = static_cast<std::uint32_t>(peer);
    output->channel = channel;
    output->backend_status = backend_status;
    output->error = error;
    aicore::system_fence();
    aicore::flush_cacheline(output);
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
};

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
    result.channel = resolve_peer(team(context), peer, channel_index);
    if (result.channel == nullptr ||
        result.channel->protocol !=
            static_cast<std::int32_t>(cann_abi::kUbcCtpProtocol) ||
        result.channel->sq_count <= cann_abi::kDefaultQueueIndex ||
        result.channel->cq_count <= cann_abi::kDefaultQueueIndex)
        return ResolvedPeer{};
    result.sq = reinterpret_cast<__gm__ cann_abi::SqContext*>(
        result.channel->sq_contexts) + cann_abi::kDefaultQueueIndex;
    result.cq = reinterpret_cast<__gm__ cann_abi::CqContext*>(
        result.channel->cq_contexts) + cann_abi::kDefaultQueueIndex;
    return result;
}

__aicore__ inline bool drain_channel(
    __gm__ TransportCommandQueue* queue, ResolvedPeer peer,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    std::uint64_t retry_limit) {
    if (peer.channel == nullptr || peer.cq == nullptr || peer.cq->base == 0 ||
        peer.cq->doorbell == 0 || peer.cq->depth == 0 ||
        peer.cq->entry_bytes != sizeof(cann_abi::UrmaCqe)) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, command_index,
            opcode, 0, 0);
        return false;
    }
    while (peer.channel->cq_tail < peer.channel->cq_head) {
        const std::uint32_t tail = peer.channel->cq_tail;
        auto* cqe = reinterpret_cast<__gm__ cann_abi::UrmaCqe*>(
            peer.cq->base +
            static_cast<std::uint64_t>(tail % peer.cq->depth) *
                peer.cq->entry_bytes);
        std::uint64_t retry = 0;
        std::uint32_t word0 = 0;
        do {
            aicore::flush_cacheline(cqe);
            word0 = cqe->words[0];
            if (urma::cqe_owner_valid((word0 >> 2U) & 1U, tail,
                                      peer.cq->depth))
                break;
            ++retry;
        } while (retry < retry_limit);
        if (retry >= retry_limit) {
            record_error(
                queue, DeviceTransportError::kCompletionTimeout,
                command_index, opcode, 0, 0);
            return false;
        }
        const auto substatus = (word0 >> 16U) & 0xffU;
        const auto status = (word0 >> 24U) & 0xffU;
        if (status != 0 || substatus != 0) {
            record_error(
                queue, DeviceTransportError::kCompletionFailure,
                command_index, opcode, 0, 0,
                (status << 8U) | substatus);
            return false;
        }
        ++peer.channel->cq_tail;
    }
    peer.channel->sq_tail = peer.channel->sq_head;
    aicore::system_fence();
    aicore::flush_cacheline(peer.channel);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.cq->doorbell),
        peer.channel->cq_tail & 0x00ffffffU);
    return true;
}

template <typename Request>
__aicore__ inline void copy_request(
    __gm__ std::uint8_t* queue_base, std::uint32_t depth,
    std::uint32_t entry_bytes, std::uint32_t head,
    const Request& request) {
    const auto* source = reinterpret_cast<const std::uint64_t*>(&request);
    constexpr std::uint32_t blocks = sizeof(Request) / 64;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        auto* destination = reinterpret_cast<__gm__ std::uint64_t*>(
            queue_base + static_cast<std::uint64_t>((head + block) % depth) *
                entry_bytes);
        for (std::uint32_t word = 0; word < 8; ++word)
            destination[word] = source[block * 8 + word];
        aicore::flush_cacheline(destination);
    }
}

template <typename Request>
__aicore__ inline bool post_request(
    __gm__ TransportCommandQueue* queue, ResolvedPeer peer,
    const Request& request, std::uint32_t command_index,
    TransportCommandOpcode opcode, std::uint64_t retry_limit) {
    constexpr std::uint32_t blocks = sizeof(Request) / 64;
    if (peer.channel == nullptr || peer.sq == nullptr || peer.cq == nullptr ||
        peer.sq->entry_bytes != 64 || peer.sq->depth == 0 ||
        peer.cq->entry_bytes != sizeof(cann_abi::UrmaCqe) ||
        peer.cq->depth == 0) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, command_index,
            opcode, 0, 0);
        return false;
    }
    if (peer.channel->sq_head - peer.channel->sq_tail + blocks >
        peer.sq->depth) {
        if (!drain_channel(
                queue, peer, command_index, opcode, retry_limit))
            return false;
    }
    copy_request(
        reinterpret_cast<__gm__ std::uint8_t*>(peer.sq->base),
        peer.sq->depth, peer.sq->entry_bytes, peer.channel->sq_head, request);
    peer.channel->sq_head += blocks;
    ++peer.channel->cq_head;
    aicore::system_fence();
    aicore::flush_cacheline(peer.channel);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.sq->doorbell),
        peer.channel->sq_head);
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

__aicore__ inline bool drain_all(
    const DeviceTransportContext& context, __gm__ TransportCommandQueue* queue,
    std::uint32_t command_index, TransportCommandOpcode opcode,
    std::uint64_t retry_limit) {
    if (context.topology.world_size <= 1)
        return true;
    auto* transport_team = team(context);
    if (transport_team == nullptr)
        return false;
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        auto resolved = resolve_context(context, peer, 0);
        if (!drain_channel(queue, resolved, command_index, opcode, retry_limit))
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

__aicore__ inline bool post_faa(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue, std::uint32_t peer_index,
    std::uint64_t remote_address, std::uint64_t fetch_address,
    std::uint64_t add_value, std::uint32_t command_index,
    TransportCommandOpcode opcode, std::uint64_t retry_limit) {
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
    auto* local = resolve_buffer(
        peer.channel->local_buffers, peer.channel->local_buffer_count,
        fetch_address, sizeof(std::uint64_t));
    if (remote == nullptr || local == nullptr) {
        record_error(
            queue, DeviceTransportError::kInvalidAddress, command_index,
            opcode, peer_index, 0);
        return false;
    }
    const auto sq = snapshot_sq(peer.sq);
    const auto remote_buffer = snapshot_buffer(remote);
    const auto request = urma::make_faa64(
        sq, remote_buffer, peer.channel->sq_head, remote_address,
        fetch_address, add_value, local->token_id);
    return post_request(
        queue, peer, request, command_index, opcode, retry_limit);
}

__aicore__ inline bool execute_signal(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current, std::uint32_t command_index,
    std::uint64_t retry_limit) {
    auto* transport_team = team(context);
    if (transport_team == nullptr || current->peer < 0 ||
        static_cast<std::uint32_t>(current->peer) >=
            transport_team->member_count)
        return false;
    std::uint64_t remote_address = 0;
    if (current->action_kind == RemoteActionKind::kSignalIncrement) {
        if (current->signal_index >= transport_team->signal_count ||
            transport_team->remote_sync_memories == 0)
            return false;
        auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
            transport_team->remote_sync_memories);
        const auto offset =
            (static_cast<std::uint64_t>(current->signal_index) *
                 transport_team->member_count +
             transport_team->self_member) * sizeof(std::uint64_t);
        remote_address = memories[current->peer].address + offset;
    } else if (current->action_kind == RemoteActionKind::kSignalAdd) {
        const auto logical = context.local_window_base +
            current->symmetric_offset;
        if (!resolve_remote_target(
                context, static_cast<std::uint32_t>(current->peer), logical,
                sizeof(std::uint64_t), remote_address))
            return false;
    } else {
        return false;
    }
    return post_faa(
        context, queue, static_cast<std::uint32_t>(current->peer),
        remote_address,
        default_fetch_result(context, static_cast<std::uint32_t>(current->peer)),
        current->value, command_index, current->opcode, retry_limit);
}

__aicore__ inline bool execute_barrier(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current, std::uint32_t command_index,
    __gm__ TransportServiceState* state, std::uint64_t retry_limit) {
    if (context.topology.world_size <= 1) {
        ++state->barrier_generation;
        return true;
    }
    auto* transport_team = team(context);
    if (transport_team == nullptr || transport_team->barrier_count == 0 ||
        transport_team->remote_sync_memories == 0)
        return false;
    auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
        transport_team->remote_sync_memories);
    const std::uint64_t barrier_base =
        static_cast<std::uint64_t>(transport_team->signal_count +
                                   transport_team->counter_count) *
        transport_team->member_count;
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        const auto offset =
            (barrier_base + transport_team->self_member) *
            sizeof(std::uint64_t);
        const auto remote_address = memories[peer].address + offset;
        const auto fetch_address =
            transport_team->shadow_sync_memory.address + offset;
        if (!post_faa(
                context, queue, peer, remote_address, fetch_address, 1,
                command_index, current->opcode, retry_limit))
            return false;
    }
    if (!drain_all(
            context, queue, command_index, current->opcode, retry_limit))
        return false;

    const auto generation = state->barrier_generation + 1;
    const auto limit = current->timeout_cycles == 0 ?
        retry_limit : current->timeout_cycles;
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        const auto offset = (barrier_base + peer) * sizeof(std::uint64_t);
        auto* signal = reinterpret_cast<__gm__ std::uint64_t*>(
            memories[transport_team->self_member].address + offset);
        std::uint64_t retry = 0;
        while (retry < limit) {
            aicore::flush_cacheline(signal);
            if (*signal >= generation)
                break;
            ++retry;
        }
        if (retry >= limit) {
            record_error(
                queue, DeviceTransportError::kCompletionTimeout,
                command_index, current->opcode, peer, 0);
            return false;
        }
    }
    state->barrier_generation = generation;
    return true;
}

}  // namespace detail

__aicore__ inline void reset(
    const DeviceTransportContext& context, std::uint64_t generation) {
    auto* queue = detail::command_queue(context);
    if (queue == nullptr)
        return;
    queue->count = 0;
    queue->generation = generation;
    auto* state = detail::service_state(queue);
    if (state != nullptr) {
        state->abi_version = kTransportCommandAbiVersion;
        state->struct_size = sizeof(TransportServiceState);
        state->consumed_count = 0;
        state->active = 0;
        state->consumed_generation = 0;
        if (state->default_retry_limit == 0)
            state->default_retry_limit = detail::kDefaultRetryLimit;
        aicore::flush_cacheline(state);
    }
    auto* output = detail::diagnostic(queue);
    if (output != nullptr) {
        output->abi_version = kTransportCommandAbiVersion;
        output->error = DeviceTransportError::kNone;
        output->command_index = 0;
        output->opcode = TransportCommandOpcode::kNone;
        output->generation = generation;
        aicore::flush_cacheline(output);
    }
    aicore::system_fence();
    aicore::flush_cacheline(queue);
}

__aicore__ inline void execute(const DeviceTransportContext& context) {
    auto* queue = detail::command_queue(context);
    if (queue == nullptr)
        return;
    aicore::flush_cacheline(queue);
    auto* state = detail::service_state(queue);
    if (state == nullptr)
        return;
    const std::uint64_t retry_limit = state->default_retry_limit == 0 ?
        detail::kDefaultRetryLimit : state->default_retry_limit;
    state->active = 1;
    auto* commands = reinterpret_cast<__gm__ TransportCommand*>(
        queue->commands);
    const std::uint32_t count = queue->count;

    for (std::uint32_t index = 0; index < count; ++index) {
        auto* current = commands + index;
        aicore::flush_cacheline(current);
        bool success = true;
        if (current->opcode == TransportCommandOpcode::kFlush) {
            success = detail::drain_all(
                context, queue, index, current->opcode, retry_limit);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidQueue, index,
                    current->opcode, 0, current->channel);
        } else if (current->opcode == TransportCommandOpcode::kBarrier) {
            success = detail::execute_barrier(
                context, queue, current, index, state, retry_limit);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidQueue, index,
                    current->opcode, 0, current->channel);
        } else if (current->peer < 0 ||
                   current->peer >= context.topology.world_size) {
            detail::record_error(
                queue, DeviceTransportError::kInvalidRank, index,
                current->opcode, current->peer, current->channel);
            success = false;
        } else if (current->opcode == TransportCommandOpcode::kSignal) {
            success = detail::execute_signal(
                context, queue, current, index, retry_limit);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidAddress, index,
                    current->opcode, current->peer, current->channel);
        } else {
            auto peer = detail::resolve_context(
                context, static_cast<std::uint32_t>(current->peer),
                current->channel);
            std::uint64_t remote_address = 0;
            const std::uint64_t bytes =
                current->opcode == TransportCommandOpcode::kPut ?
                    current->bytes : sizeof(std::uint64_t);
            if (peer.channel == nullptr ||
                !detail::resolve_remote_target(
                    context, static_cast<std::uint32_t>(current->peer),
                    current->destination, bytes, remote_address)) {
                detail::record_error(
                    queue, DeviceTransportError::kInvalidAddress, index,
                    current->opcode, current->peer, current->channel);
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
                        current->opcode, current->peer, current->channel);
                    success = false;
                } else {
                    const auto sq = detail::snapshot_sq(peer.sq);
                    const auto remote_buffer =
                        detail::snapshot_buffer(remote);
                    const auto request = urma::make_write(
                        sq, remote_buffer, peer.channel->sq_head,
                        remote_address, current->source,
                        static_cast<std::uint32_t>(current->bytes),
                        local->token_id);
                    success = detail::post_request(
                        queue, peer, request, index, current->opcode,
                        retry_limit);
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
                        current->opcode, current->peer, current->channel);
                    success = false;
                } else {
                    const auto sq = detail::snapshot_sq(peer.sq);
                    const auto remote_buffer =
                        detail::snapshot_buffer(remote);
                    const auto request = urma::make_inline_write64(
                        sq, remote_buffer, peer.channel->sq_head,
                        remote_address, current->value);
                    success = detail::post_request(
                        queue, peer, request, index, current->opcode,
                        retry_limit);
                }
            } else if (current->opcode ==
                       TransportCommandOpcode::kRemoteAdd64) {
                success = detail::post_faa(
                    context, queue,
                    static_cast<std::uint32_t>(current->peer),
                    remote_address,
                    detail::default_fetch_result(
                        context, static_cast<std::uint32_t>(current->peer)),
                    current->value, index, current->opcode, retry_limit);
            } else {
                detail::record_error(
                    queue, DeviceTransportError::kUnsupportedOperation,
                    index, current->opcode, current->peer, current->channel);
                success = false;
            }
        }
        if (!success)
            break;
        state->consumed_count = index + 1;
    }
    state->active = 0;
    state->consumed_generation = queue->generation;
    aicore::system_fence();
    aicore::flush_cacheline(state);
}

#endif

}  // namespace deep_ep::ascend::transport::service
