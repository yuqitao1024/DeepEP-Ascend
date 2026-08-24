#pragma once

#include "cann_compat.hpp"
#include "device_topology.hpp"
#include "simt_intrinsics.hpp"
#include "execution_domain_helpers.hpp"
#include "sync_layout.hpp"
#include "transport_commands.hpp"

namespace deep_ep::ascend::transport::device {

namespace detail {

inline constexpr std::uint64_t kDefaultRequestRetryLimit = 1000000;

DEEP_EP_ASCEND_SIMT_CALLEE int local_rank(
    const DeviceTransportContext& context, TransportTeam team) {
    switch (team) {
        case TransportTeam::kWorld: return context.topology.world_rank;
        case TransportTeam::kScaleUp: return context.topology.scale_up_rank;
        case TransportTeam::kScaleOut: return context.topology.scale_out_rank;
    }
    return -1;
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ TransportCommandQueue* command_queue(
    const DeviceTransportContext& context) {
    if (context.abi_version != kDeviceTransportAbiVersion ||
        context.struct_size != sizeof(DeviceTransportContext) ||
        context.backend_context == 0)
        return nullptr;

    auto* staged = reinterpret_cast<__gm__ StagedTransportContext*>(
        context.backend_context);
    if (simt::load_observed(&staged->abi_version) !=
            kTransportCommandAbiVersion ||
        simt::load_observed(&staged->struct_size) !=
            sizeof(StagedTransportContext) ||
        simt::load_observed(&staged->cann_compatibility) !=
            kStagedTransportCannCompatibility)
        return nullptr;

    const auto address = simt::load_observed(&staged->command_queue);
    return reinterpret_cast<__gm__ TransportCommandQueue*>(address);
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ DeviceTransportDiagnostic* diagnostic(
    __gm__ TransportCommandQueue* queue) {
    if (queue == nullptr)
        return nullptr;
    return reinterpret_cast<__gm__ DeviceTransportDiagnostic*>(
        simt::load_observed(&queue->diagnostic));
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ TransportServiceState* service_state(
    __gm__ TransportCommandQueue* queue) {
    if (queue == nullptr)
        return nullptr;
    return reinterpret_cast<__gm__ TransportServiceState*>(
        simt::load_observed(&queue->service_state));
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ std::uint64_t* signal_address(
    const DeviceTransportContext& context, int source_rank,
    std::uint32_t signal_index, std::uint32_t* failure_stage = nullptr,
    std::uint64_t* failure_value = nullptr) {
    if (failure_stage != nullptr)
        *failure_stage = 0;
    if (failure_value != nullptr)
        *failure_value = 0;
    if (context.channel_table == 0) {
        if (failure_stage != nullptr)
            *failure_stage = static_cast<std::uint32_t>(
                sync_layout::SignalAddressFailure::kMissingTeam);
        return nullptr;
    }
    auto* team = reinterpret_cast<__gm__ cann_abi::Team*>(
        context.channel_table);
    const auto members = simt::load_observed(&team->member_count);
    const auto self = simt::load_observed(&team->self_member);
    const auto signals = simt::load_observed(&team->signal_count);
    const auto counters = simt::load_observed(&team->counter_count);
    const auto barriers = simt::load_observed(&team->barrier_count);
    const auto memories_address =
        simt::load_observed(&team->remote_sync_memories);
    const auto failure = sync_layout::classify_signal_address_layout(
        members, self, signals, counters, barriers, memories_address,
        source_rank, signal_index);
    if (failure != sync_layout::SignalAddressFailure::kNone) {
        if (failure_stage != nullptr)
            *failure_stage = static_cast<std::uint32_t>(failure);
        if (failure_value != nullptr) {
            switch (failure) {
                case sync_layout::SignalAddressFailure::kInvalidSourceMember:
                    *failure_value = static_cast<std::uint64_t>(source_rank);
                    break;
                case sync_layout::SignalAddressFailure::kInvalidSelfMember:
                    *failure_value = self;
                    break;
                case sync_layout::SignalAddressFailure::kInvalidSignalCount:
                    *failure_value = signals;
                    break;
                case sync_layout::SignalAddressFailure::kInvalidCounterCount:
                    *failure_value = counters;
                    break;
                case sync_layout::SignalAddressFailure::kInvalidBarrierCount:
                    *failure_value = barriers;
                    break;
                case sync_layout::SignalAddressFailure::kInvalidSignalIndex:
                    *failure_value = signal_index;
                    break;
                case sync_layout::SignalAddressFailure::kMissingRemoteSyncMemories:
                    *failure_value = memories_address;
                    break;
                default: break;
            }
        }
        return nullptr;
    }
    auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
        memories_address);
    const auto base = simt::load_observed(&memories[self].address);
    if (base == 0) {
        if (failure_stage != nullptr)
            *failure_stage = static_cast<std::uint32_t>(
                sync_layout::SignalAddressFailure::kMissingLocalSyncMemory);
        if (failure_value != nullptr)
            *failure_value = self;
        return nullptr;
    }
    const auto offset =
        (static_cast<std::uint64_t>(signal_index) * members +
         static_cast<std::uint32_t>(source_rank)) * sizeof(std::uint64_t);
    return reinterpret_cast<__gm__ std::uint64_t*>(
        base + offset);
}

DEEP_EP_ASCEND_SIMT_CALLEE void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    TransportCommandOpcode opcode, TransportTeam team, int peer,
    int world_peer, DeviceChannel channel, std::uint32_t backend_status = 0,
    std::uint64_t reserved = 0) {
    auto* output = diagnostic(queue);
    if (output == nullptr ||
        simt::load_observed(reinterpret_cast<__gm__ std::uint32_t*>(
            &output->error)) !=
            static_cast<std::uint32_t>(DeviceTransportError::kNone))
        return;

    const std::uint32_t command_index = queue == nullptr ? 0 :
        simt::load_observed(&queue->count);
    simt::store_published(&output->command_index, command_index);
    simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(&output->opcode),
        static_cast<std::uint32_t>(opcode));
    simt::store_published(&output->peer, static_cast<std::uint32_t>(peer));
    simt::store_published(&output->world_peer, world_peer);
    simt::store_published(
        reinterpret_cast<__gm__ std::uint8_t*>(&output->team),
        static_cast<std::uint8_t>(team));
    simt::store_published(&output->channel, channel);
    if (backend_status != 0 || reserved != 0) {
        simt::store_published(&output->backend_status, backend_status);
        simt::store_published(&output->reserved, reserved);
    }
    simt::system_fence();
    simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(&output->error),
        static_cast<std::uint32_t>(error));
}

DEEP_EP_ASCEND_SIMT_CALLEE void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    TransportCommandOpcode opcode, int peer, DeviceChannel channel,
    std::uint32_t backend_status = 0, std::uint64_t reserved = 0) {
    record_error(
        queue, error, opcode, TransportTeam::kWorld, peer, peer, channel,
        backend_status, reserved);
}

DEEP_EP_ASCEND_SIMT_CALLEE bool validate_queue(
    __gm__ TransportCommandQueue* queue, TransportCommandOpcode opcode,
    int peer, DeviceChannel channel) {
    if (queue == nullptr)
        return false;
    if (simt::load_observed(&queue->abi_version) !=
            kTransportCommandAbiVersion ||
        simt::load_observed(&queue->struct_size) !=
            sizeof(TransportCommandQueue)) {
        record_error(
            queue, DeviceTransportError::kInvalidAbi, opcode, peer, channel);
        return false;
    }
    if (simt::load_observed(&queue->commands) == 0 ||
        simt::load_observed(&queue->diagnostic) == 0) {
        record_error(
            queue, DeviceTransportError::kInvalidQueue, opcode, peer, channel);
        return false;
    }
    return true;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool translate_peer(
    const DeviceTransportContext& context, __gm__ TransportCommandQueue* queue,
    TransportTeam team, int peer, DeviceChannel channel,
    TransportCommandOpcode opcode, int* world_peer) {
    if (channel != 0) {
        record_error(
            queue, DeviceTransportError::kInvalidChannel, opcode, team, peer,
            -1, channel);
        return false;
    }
    if (!checked_world_peer(
            context.topology, team, peer, world_peer)) {
        record_error(
            queue, DeviceTransportError::kInvalidRank, opcode, team, peer, -1,
            channel);
        return false;
    }
    return true;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool append(
    __gm__ TransportCommandQueue* queue,
    const TransportCommand& transport_command) {
    const auto count = simt::load_observed(&queue->count);
    const auto capacity = simt::load_observed(&queue->capacity);
    if (count >= capacity) {
        record_error(
            queue, DeviceTransportError::kCommandOverflow,
            transport_command.opcode, transport_command.team,
            transport_command.peer, transport_command.world_peer,
            transport_command.channel);
        return false;
    }

    const auto command_address = simt::load_observed(&queue->commands);
    auto* target = reinterpret_cast<__gm__ TransportCommand*>(
        command_address) + count;
    auto* target_words = reinterpret_cast<__gm__ std::uint64_t*>(target);
    const auto* source_words =
        reinterpret_cast<const std::uint64_t*>(&transport_command);
    for (std::uint32_t word = 0;
         word < sizeof(TransportCommand) / sizeof(std::uint64_t); ++word)
        simt::store_published(&target_words[word], source_words[word]);

    simt::system_fence();
    simt::store_published(&queue->count, count + 1);
    simt::system_fence();
    const auto generation = simt::load_observed(&queue->generation);
    simt::store_published(&queue->generation, generation);
    return true;
}

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ TransportCommandQueue* prepare(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int peer, TransportCommandOpcode opcode,
    int* world_peer) {
    auto* queue = command_queue(context);
    if (!validate_queue(queue, opcode, peer, channel) ||
        !translate_peer(
            context, queue, team, peer, channel, opcode, world_peer))
        return nullptr;
    return queue;
}

}  // namespace detail

DEEP_EP_ASCEND_SIMT_CALLEE bool is_peer_directly_accessible(
    const DeviceTransportContext& context, TransportTeam team, int rank) {
    return rank == detail::local_rank(context, team);
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t get_symmetric_offset(
    const DeviceTransportContext& context, DeviceAddress local_address) {
    if (local_address == kNullDeviceAddress || context.local_window_base == 0)
        return 0;
    return local_address - context.local_window_base;
}

DEEP_EP_ASCEND_SIMT_CALLEE DeviceAddress get_symmetric_pointer(
    const DeviceTransportContext& context, TransportTeam team, int rank,
    DeviceAddress local_address) {
    return is_peer_directly_accessible(context, team, rank) ?
        local_address : kNullDeviceAddress;
}

DEEP_EP_ASCEND_SIMT_CALLEE void put(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int destination_rank, DeviceAddress destination,
    DeviceAddress source, std::size_t bytes, CooperationScope scope,
    MemorySegment segment, DeviceOptions options,
    const RemoteAction& remote_action) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    auto* queue = detail::prepare(
        context, channel, team, destination_rank,
        TransportCommandOpcode::kPut, &world_peer);
    if (queue == nullptr)
        return;
    if (destination == kNullDeviceAddress || source == kNullDeviceAddress ||
        bytes == 0) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPut, team, destination_rank, world_peer,
            channel);
        return;
    }
    if (scope != CooperationScope::kParticipant ||
        segment != MemorySegment::kDevice || options != kDefaultOptions ||
        remote_action.kind != RemoteActionKind::kNone) {
        detail::record_error(
            queue, DeviceTransportError::kUnsupportedOperation,
            TransportCommandOpcode::kPut, team, destination_rank, world_peer,
            channel);
        return;
    }

    TransportCommand command{};
    command.opcode = TransportCommandOpcode::kPut;
    command.team = team;
    command.scope = scope;
    command.segment = segment;
    command.peer = destination_rank;
    command.world_peer = world_peer;
    command.channel = channel;
    command.options = options;
    command.source = source;
    command.destination = destination;
    command.bytes = bytes;
    detail::append(queue, command);
}

DEEP_EP_ASCEND_SIMT_CALLEE void get(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    DeviceAddress, DeviceAddress, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions) {}

DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int destination_rank, DeviceAddress destination,
    std::uint64_t value, std::uint32_t value_bytes, DeviceOptions options) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    auto* queue = detail::prepare(
        context, channel, team, destination_rank,
        TransportCommandOpcode::kPutValue64, &world_peer);
    if (queue == nullptr)
        return;
    if (destination == kNullDeviceAddress) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kPutValue64, team, destination_rank,
            world_peer, channel);
        return;
    }
    if (value_bytes != sizeof(std::uint64_t) ||
        options != kDefaultOptions) {
        detail::record_error(
            queue, DeviceTransportError::kUnsupportedOperation,
            TransportCommandOpcode::kPutValue64, team, destination_rank,
            world_peer, channel);
        return;
    }

    TransportCommand command{};
    command.opcode = TransportCommandOpcode::kPutValue64;
    command.team = team;
    command.peer = destination_rank;
    command.world_peer = world_peer;
    command.channel = channel;
    command.options = options;
    command.value_bytes = value_bytes;
    command.destination = destination;
    command.value = value;
    detail::append(queue, command);
}

DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int destination_rank, DeviceAddress destination,
    std::int64_t value) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    auto* queue = detail::prepare(
        context, channel, team, destination_rank,
        TransportCommandOpcode::kRemoteAdd64, &world_peer);
    if (queue == nullptr)
        return;
    if (destination == kNullDeviceAddress) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kRemoteAdd64, team, destination_rank,
            world_peer, channel);
        return;
    }

    simt::system_fence();
    TransportCommand command{};
    command.opcode = TransportCommandOpcode::kRemoteAdd64;
    command.team = team;
    command.peer = destination_rank;
    command.world_peer = world_peer;
    command.channel = channel;
    command.destination = destination;
    command.value = static_cast<std::uint64_t>(value);
    detail::append(queue, command);
}

DEEP_EP_ASCEND_SIMT_CALLEE void signal(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int destination_rank,
    const RemoteAction& remote_action) {
    if (threadIdx.x != 0)
        return;
    int world_peer = -1;
    auto* queue = detail::prepare(
        context, channel, team, destination_rank,
        TransportCommandOpcode::kSignal, &world_peer);
    if (queue == nullptr)
        return;
    if (remote_action.kind != RemoteActionKind::kSignalAdd &&
        remote_action.kind != RemoteActionKind::kSignalIncrement &&
        remote_action.kind != RemoteActionKind::kSignalSet) {
        detail::record_error(
            queue, DeviceTransportError::kUnsupportedOperation,
            TransportCommandOpcode::kSignal, team, destination_rank,
            world_peer, channel);
        return;
    }

    simt::system_fence();
    TransportCommand command{};
    command.opcode = TransportCommandOpcode::kSignal;
    command.team = team;
    command.action_kind = remote_action.kind;
    command.peer = destination_rank;
    command.world_peer = world_peer;
    command.channel = channel;
    command.signal_index = remote_action.signal_index;
    command.symmetric_offset = remote_action.symmetric_offset;
    command.value = remote_action.value;
    detail::append(queue, command);
}

DEEP_EP_ASCEND_SIMT_CALLEE SignalValue read_signal(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int source_rank, std::uint32_t signal_index) {
    int world_peer = -1;
    if (channel != 0 || !detail::checked_world_peer(
            context.topology, team, source_rank, &world_peer))
        return 0;
    auto* address = detail::signal_address(context, world_peer, signal_index);
    if (address == nullptr)
        return 0;
    const auto value = simt::load_observed(address);
    simt::system_fence();
    return value;
}

DEEP_EP_ASCEND_SIMT_CALLEE void wait_signal(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int source_rank, std::uint32_t signal_index,
    SignalValue target, std::uint64_t timeout_cycles) {
    auto* queue = detail::command_queue(context);
    int world_peer = -1;
    if (channel != 0) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidChannel,
            TransportCommandOpcode::kSignal, team, source_rank, world_peer,
            channel);
        return;
    }
    if (!detail::checked_world_peer(
            context.topology, team, source_rank, &world_peer)) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kSignal, team, source_rank, world_peer,
            channel);
        return;
    }
    std::uint32_t failure_stage = 0;
    std::uint64_t failure_value = 0;
    auto* address = detail::signal_address(
        context, world_peer, signal_index, &failure_stage, &failure_value);
    if (address == nullptr) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kSignal, team, source_rank, world_peer,
            channel, failure_stage, failure_value);
        return;
    }
    const auto limit = timeout_cycles == 0 ?
        std::uint64_t{1000000} : timeout_cycles;
    std::uint64_t retry = 0;
    while (retry < limit && simt::load_observed(address) < target)
        ++retry;
    if (retry >= limit) {
        detail::record_error(
            queue, DeviceTransportError::kCompletionTimeout,
            TransportCommandOpcode::kSignal, team, source_rank, world_peer,
            channel);
        return;
    }
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE void flush(
    const DeviceTransportContext& context, DeviceChannel channel,
    CooperationScope scope) {
    if (threadIdx.x != 0)
        return;
    auto* queue = detail::command_queue(context);
    if (!detail::validate_queue(
            queue, TransportCommandOpcode::kFlush, 0, channel))
        return;
    if (channel != 0) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidChannel,
            TransportCommandOpcode::kFlush, 0, channel);
        return;
    }

    TransportCommand command{};
    command.opcode = TransportCommandOpcode::kFlush;
    command.scope = scope;
    command.channel = channel;
    detail::append(queue, command);
}

DEEP_EP_ASCEND_SIMT_CALLEE void flush_async(
    const DeviceTransportContext& context, DeviceChannel channel,
    TransportTeam team, int peer_rank, CooperationScope scope,
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) {
    if (threadIdx.x != 0)
        return;
    auto* queue = detail::command_queue(context);
    if (!detail::validate_queue(
            queue, TransportCommandOpcode::kFlush, peer_rank, channel))
        return;
    if (request == nullptr) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kFlush, team, peer_rank, -1, channel);
        return;
    }
    if (request->abi_version != kDeviceRequestAbiVersion) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidAbi;
        detail::record_error(
            queue, DeviceTransportError::kInvalidAbi,
            TransportCommandOpcode::kFlush, team, peer_rank, -1, channel);
        return;
    }
    if (request->state == DeviceRequestState::kPending ||
        request->state < DeviceRequestState::kEmpty ||
        request->state > DeviceRequestState::kFailed) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidProtocol;
        detail::record_error(
            queue, DeviceTransportError::kInvalidProtocol,
            TransportCommandOpcode::kFlush, team, peer_rank, -1, channel);
        return;
    }
    if (channel != 0) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidChannel;
        detail::record_error(
            queue, DeviceTransportError::kInvalidChannel,
            TransportCommandOpcode::kFlush, team, peer_rank, -1, channel);
        return;
    }
    int world_peer = -1;
    if (!detail::checked_world_peer(
            context.topology, team, peer_rank, &world_peer)) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidRank;
        detail::record_error(
            queue, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kFlush, team, peer_rank, world_peer,
            channel);
        return;
    }
    auto* state = detail::service_state(queue);
    if (state == nullptr) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidQueue;
        detail::record_error(
            queue, DeviceTransportError::kInvalidQueue,
            TransportCommandOpcode::kFlush, team, peer_rank, world_peer,
            channel);
        return;
    }

    const auto command_begin = simt::load_observed(&queue->count);
    const auto queue_generation = simt::load_observed(&queue->generation);
    TransportCommand flush_command{};
    flush_command.opcode = TransportCommandOpcode::kFlush;
    flush_command.scope = scope;
    flush_command.channel = channel;
    simt::store_published(
        &state->consumed_generation, std::uint64_t{0});
    if (!detail::append(queue, flush_command)) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kCommandOverflow;
        return;
    }
    command::simt_publish_request(
        request, command_begin, command_begin + 1, queue_generation);
}

DEEP_EP_ASCEND_SIMT_CALLEE void wait(
    const DeviceTransportContext& context,
    DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) {
    if (threadIdx.x != 0 || request == nullptr)
        return;
    if (request->state == DeviceRequestState::kCompleted ||
        request->state == DeviceRequestState::kFailed)
        return;

    auto* queue = detail::command_queue(context);
    if (!detail::validate_queue(
            queue, TransportCommandOpcode::kFlush, 0, 0)) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidQueue;
        return;
    }
    auto* state = detail::service_state(queue);
    auto* diagnostic = detail::diagnostic(queue);
    if (state == nullptr || diagnostic == nullptr) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidQueue;
        return;
    }
    if (simt::load_observed(&state->abi_version) !=
            kTransportCommandAbiVersion ||
        simt::load_observed(&state->struct_size) !=
            sizeof(TransportServiceState) ||
        simt::load_observed(&diagnostic->abi_version) !=
            kTransportCommandAbiVersion) {
        request->state = DeviceRequestState::kFailed;
        request->terminal_error = DeviceTransportError::kInvalidAbi;
        return;
    }

    const auto configured_limit =
        simt::load_observed(&state->default_retry_limit);
    const auto retry_limit = configured_limit == 0 ?
        detail::kDefaultRequestRetryLimit : configured_limit;
    std::uint64_t retry = 0;
    while (retry < retry_limit) {
        const auto queue_generation =
            simt::load_observed(&queue->generation);
        const auto consumed_generation =
            simt::load_observed(&state->consumed_generation);
        const auto consumed_count =
            simt::load_observed(&state->consumed_count);
        const auto diagnostic_error = static_cast<DeviceTransportError>(
            simt::load_observed(
                reinterpret_cast<__gm__ std::uint32_t*>(
                    &diagnostic->error)));
        if (diagnostic_error != DeviceTransportError::kNone)
            simt::system_fence();
        const auto diagnostic_generation =
            simt::load_observed(&diagnostic->generation);
        const auto diagnostic_command_index =
            simt::load_observed(&diagnostic->command_index);
        if (command::simt_observe_request(
                request, queue_generation, consumed_generation,
                consumed_count, diagnostic_generation,
                diagnostic_command_index, diagnostic_error)) {
            simt::system_fence();
            return;
        }
        ++retry;
    }
    command::simt_timeout_request(request);
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t load_acquire(
    DeviceAddress address) {
    if (address == kNullDeviceAddress)
        return 0;
    const auto value = simt::load_observed(
        reinterpret_cast<const __gm__ std::uint64_t*>(address));
    simt::system_fence();
    return value;
}

DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
    DeviceAddress address, std::uint64_t value) {
    if (address == kNullDeviceAddress)
        return;
    simt::system_fence();
    simt::store_published(
        reinterpret_cast<__gm__ std::uint64_t*>(address), value);
}

DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() {
    simt::system_fence();
}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t consumed_generation(
    const DeviceTransportContext& context) {
    auto* queue = detail::command_queue(context);
    if (queue == nullptr)
        return 0;
    const auto state_address = simt::load_observed(&queue->service_state);
    if (state_address == 0)
        return 0;
    auto* state = reinterpret_cast<__gm__ TransportServiceState*>(
        state_address);
    return simt::load_observed(&state->consumed_generation);
}

DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
    const DeviceTransportContext& context, std::uint32_t team_mask,
    DeviceAddress, std::uint64_t timeout_cycles) {
    if (threadIdx.x != 0)
        return;
    auto* queue = detail::command_queue(context);
    if (!detail::validate_queue(
            queue, TransportCommandOpcode::kBarrier, 0, 0))
        return;
    if (team_mask == 0 || (team_mask & ~kWorldTeamMask) != 0 ||
        (!command::simt_barrier_team_enabled(
             context.topology, team_mask, TransportTeam::kScaleOut) &&
         !command::simt_barrier_team_enabled(
             context.topology, team_mask, TransportTeam::kScaleUp))) {
        detail::record_error(
            queue, DeviceTransportError::kUnsupportedOperation,
            TransportCommandOpcode::kBarrier, 0, 0);
        return;
    }

    simt::system_fence();
    TransportCommand command{};
    command.opcode = TransportCommandOpcode::kBarrier;
    command.options = team_mask;
    command.timeout_cycles = timeout_cycles;
    detail::append(queue, command);
}

}  // namespace deep_ep::ascend::transport::device
