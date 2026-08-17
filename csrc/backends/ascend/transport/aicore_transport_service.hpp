#pragma once

#include <cstddef>
#include <cstdint>

#include "cann_compat.hpp"
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
    result.topology.world_rank = static_cast<int>(self_member);
    result.topology.world_size = static_cast<int>(member_count);
    result.topology.scale_up_rank = static_cast<int>(self_member);
    result.topology.scale_up_size = static_cast<int>(member_count);
    result.topology.scale_out_rank = 0;
    result.topology.scale_out_size = 1;
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
        int expected_world_peer = -1;
        if (!command::checked_world_peer(
                state.topology, command.team, command.peer,
                &expected_world_peer) ||
            command.world_peer != expected_world_peer ||
            command.world_peer < 0 ||
            static_cast<std::uint32_t>(command.world_peer) >=
                state.member_count) {
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
                diagnostic, error, index, current.opcode, current.team,
                current.peer, current.world_peer, current.channel);
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
    if (topology.world_size <= 0 || topology.scale_up_size <= 0 ||
        topology.scale_out_size <= 0 || topology.world_rank < 0 ||
        topology.world_rank >= topology.world_size ||
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
    return expected == current->world_peer;
}

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
    (void)peer;
    output->command_index = command_index;
    output->opcode = opcode;
    output->channel = channel;
    output->backend_status = backend_status;
    output->error = error;
    aicore::system_fence();
    aicore::flush_cacheline(output);
}

__aicore__ inline void record_route(
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current) {
    auto* output = diagnostic(queue);
    if (output == nullptr || current == nullptr)
        return;
    output->peer = static_cast<std::uint32_t>(current->peer);
    output->world_peer = current->world_peer;
    output->team = current->team;
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
    std::uint64_t retry_limit,
    const AscendC::LocalTensor<std::uint32_t>& service_scratch) {
    if (peer.channel == nullptr || peer.sq == nullptr || peer.cq == nullptr ||
        peer.sq->head == 0 || peer.sq->tail == 0 || peer.cq->base == 0 ||
        peer.cq->tail == 0 || peer.cq->doorbell == 0 || peer.cq->depth == 0 ||
        peer.cq->entry_bytes != sizeof(cann_abi::UrmaCqe)) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, command_index,
            opcode, 0, 0);
        return false;
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
        ++tail;
    }
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.cq->tail), tail);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.cq->doorbell),
        tail & 0x00ffffffU);
    aicore::store_device(
        reinterpret_cast<__gm__ std::uint32_t*>(peer.sq->tail), tail);
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

template <typename Request>
__aicore__ inline bool post_request(
    __gm__ TransportCommandQueue* queue, ResolvedPeer peer,
    Request request, std::uint32_t command_index,
    TransportCommandOpcode opcode, std::uint64_t retry_limit,
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
    if (request_count - completed + 1 >= peer.cq->depth) {
        if (!drain_channel(
                queue, peer, command_index, opcode, retry_limit,
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
    std::uint64_t retry_limit,
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
        if (!drain_channel(
                queue, resolved, command_index, opcode, retry_limit,
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

__aicore__ inline bool post_faa(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue, std::uint32_t peer_index,
    std::uint64_t remote_address, std::uint64_t fetch_address,
    std::uint64_t add_value, std::uint32_t command_index,
    TransportCommandOpcode opcode, std::uint64_t retry_limit,
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
    return post_request(
        queue, peer, request, command_index, opcode, retry_limit,
        wqe_scratch);
}

__aicore__ inline bool execute_signal(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current, std::uint32_t command_index,
    std::uint64_t retry_limit,
    const AscendC::LocalTensor<std::uint32_t>& wqe_scratch) {
    auto* transport_team = team(context);
    if (transport_team == nullptr || current->world_peer < 0 ||
        static_cast<std::uint32_t>(current->world_peer) >=
            transport_team->member_count)
        return false;
    std::uint64_t remote_address = 0;
    if (current->action_kind == RemoteActionKind::kSignalIncrement) {
        if (transport_team->signal_count !=
                sync_layout::kWorldTeamSignalCount ||
            transport_team->counter_count !=
                sync_layout::kWorldTeamCounterCount ||
            transport_team->barrier_count <
                sync_layout::kWorldTeamBarrierCount ||
            current->signal_index >= sync_layout::kLogicalSignalCount ||
            transport_team->remote_sync_memories == 0)
            return false;
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
                sizeof(std::uint64_t), remote_address))
            return false;
    } else {
        return false;
    }
    return post_faa(
        context, queue, static_cast<std::uint32_t>(current->world_peer),
        remote_address,
        default_fetch_result(
            context, static_cast<std::uint32_t>(current->world_peer)),
        current->value, command_index, current->opcode, retry_limit,
        wqe_scratch);
}

__aicore__ inline bool execute_barrier(
    const DeviceTransportContext& context,
    __gm__ TransportCommandQueue* queue,
    __gm__ const TransportCommand* current, std::uint32_t command_index,
    __gm__ TransportServiceState* state, std::uint64_t retry_limit,
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
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        const auto offset =
            (static_cast<std::uint64_t>(sync_layout::kLogicalSignalCount) *
                 transport_team->member_count +
             transport_team->self_member) * sizeof(std::uint64_t);
        const auto remote_address = memories[peer].address + offset;
        const auto fetch_address =
            transport_team->shadow_sync_memory.address + offset;
        if (!post_faa(
                context, queue, peer, remote_address, fetch_address, 1,
                command_index, current->opcode, retry_limit, wqe_scratch))
            return false;
    }
    if (!drain_all(
            context, queue, command_index, current->opcode, retry_limit,
            wqe_scratch))
        return false;

    const auto generation = state->barrier_generation + 1;
    const auto start_cycles = static_cast<std::uint64_t>(
        AscendC::GetSystemCycle());
    for (std::uint32_t peer = 0; peer < transport_team->member_count; ++peer) {
        if (peer == transport_team->self_member)
            continue;
        const auto offset =
            (static_cast<std::uint64_t>(sync_layout::kLogicalSignalCount) *
                 transport_team->member_count +
             peer) * sizeof(std::uint64_t);
        auto* signal = reinterpret_cast<__gm__ std::uint64_t*>(
            memories[transport_team->self_member].address + offset);
        AscendC::GlobalTensor<std::uint64_t> signal_global;
        signal_global.SetGlobalBuffer(signal);
        const auto signal_scratch =
            wqe_scratch[48].ReinterpretCast<std::uint64_t>();
        const AscendC::DataCopyExtParams copy_params{
            1, sizeof(std::uint64_t), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<std::uint64_t> pad_params{
            false, 0, 0, 0};
        std::uint64_t retry = 0;
        std::uint64_t observed = 0;
        while (true) {
            aicore::sync_event<AscendC::HardEvent::S_MTE2>();
            AscendC::DataCopyPad(signal_scratch, signal_global, copy_params,
                                 pad_params);
            aicore::sync_event<AscendC::HardEvent::MTE2_S>();
            observed = signal_scratch.GetValue(0);
            if (observed >= generation)
                break;
            ++retry;
            const auto current_cycles = static_cast<std::uint64_t>(
                AscendC::GetSystemCycle());
            if (barrier_poll_timed_out(
                    start_cycles, current_cycles, current->timeout_cycles,
                    retry, retry_limit)) {
                record_error(
                    queue, DeviceTransportError::kCompletionTimeout,
                    command_index, current->opcode, peer, 0);
                return false;
            }
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
    state->active = 1;
    auto* commands = reinterpret_cast<__gm__ TransportCommand*>(
        queue->commands);
    const std::uint32_t count = queue->count;

    for (std::uint32_t index = 0; index < count; ++index) {
        auto* current = commands + index;
        aicore::flush_cacheline(current);
        detail::record_route(queue, current);
        bool success = true;
        if (current->opcode == TransportCommandOpcode::kFlush) {
            success = detail::drain_all(
                context, queue, index, current->opcode, retry_limit,
                wqe_scratch);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidQueue, index,
                    current->opcode, 0, current->channel);
        } else if (current->opcode == TransportCommandOpcode::kBarrier) {
            success = detail::execute_barrier(
                context, queue, current, index, state, retry_limit,
                wqe_scratch);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidQueue, index,
                    current->opcode, 0, current->channel);
        } else if (!detail::valid_command_route(context, current) ||
                   current->world_peer < 0 ||
                   current->world_peer >= context.topology.world_size) {
            detail::record_error(
                queue, DeviceTransportError::kInvalidRank, index,
                current->opcode, current->world_peer, current->channel);
            success = false;
        } else if (current->opcode == TransportCommandOpcode::kSignal) {
            success = detail::execute_signal(
                context, queue, current, index, retry_limit, wqe_scratch);
            if (!success)
                detail::record_error(
                    queue, DeviceTransportError::kInvalidAddress, index,
                    current->opcode, current->world_peer, current->channel);
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
                    current->opcode, current->world_peer, current->channel);
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
                        current->opcode, current->world_peer, current->channel);
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
                    success = detail::post_request(
                        queue, peer, request, index, current->opcode,
                        retry_limit, wqe_scratch);
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
                        current->opcode, current->world_peer, current->channel);
                    success = false;
                } else {
                    const auto sq = detail::snapshot_sq(peer.sq);
                    const auto remote_buffer =
                        detail::snapshot_buffer(remote);
                    const auto request = urma::make_inline_write64(
                        sq, remote_buffer, 0,
                        remote_address, current->value);
                    success = detail::post_request(
                        queue, peer, request, index, current->opcode,
                        retry_limit, wqe_scratch);
                }
            } else if (current->opcode ==
                       TransportCommandOpcode::kRemoteAdd64) {
                success = detail::post_faa(
                    context, queue,
                    static_cast<std::uint32_t>(current->world_peer),
                    remote_address,
                    detail::default_fetch_result(
                        context,
                        static_cast<std::uint32_t>(current->world_peer)),
                    current->value, index, current->opcode, retry_limit,
                    wqe_scratch);
            } else {
                detail::record_error(
                    queue, DeviceTransportError::kUnsupportedOperation,
                    index, current->opcode, current->world_peer,
                    current->channel);
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
