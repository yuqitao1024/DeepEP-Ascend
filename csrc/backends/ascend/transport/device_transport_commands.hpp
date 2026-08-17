#pragma once

#include "cann_compat.hpp"
#include "simt_intrinsics.hpp"
#include "sync_layout.hpp"
#include "transport_commands.hpp"

namespace deep_ep::ascend::transport::device {

namespace detail {

DEEP_EP_ASCEND_SIMT_CALLEE int local_rank(
    const DeviceTransportContext& context, TransportTeam team) {
    switch (team) {
        case TransportTeam::kWorld: return context.topology.world_rank;
        case TransportTeam::kScaleUp: return context.topology.scale_up_rank;
        case TransportTeam::kScaleOut: return context.topology.scale_out_rank;
    }
    return -1;
}

DEEP_EP_ASCEND_SIMT_CALLEE bool checked_world_peer(
    const TransportTopology& topology, TransportTeam team, int peer,
    int* world_peer) {
    if (world_peer == nullptr || topology.world_size <= 0 ||
        topology.scale_up_size <= 0 || topology.scale_out_size <= 0 ||
        topology.world_rank < 0 ||
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

    int translated = -1;
    switch (team) {
        case TransportTeam::kWorld:
            if (peer < 0 || peer >= topology.world_size)
                return false;
            translated = peer;
            break;
        case TransportTeam::kScaleUp:
            if (peer < 0 || peer >= topology.scale_up_size)
                return false;
            translated = topology.scale_out_rank * topology.scale_up_size +
                peer;
            break;
        case TransportTeam::kScaleOut:
            if (peer < 0 || peer >= topology.scale_out_size)
                return false;
            translated = peer * topology.scale_up_size +
                topology.scale_up_rank;
            break;
        default: return false;
    }
    if (translated < 0 || translated >= topology.world_size)
        return false;
    *world_peer = translated;
    return true;
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

DEEP_EP_ASCEND_SIMT_CALLEE __gm__ std::uint64_t* signal_address(
    const DeviceTransportContext& context, int source_rank,
    std::uint32_t signal_index) {
    if (context.channel_table == 0 || source_rank < 0)
        return nullptr;
    auto* team = reinterpret_cast<__gm__ cann_abi::Team*>(
        context.channel_table);
    const auto members = simt::load_observed(&team->member_count);
    const auto self = simt::load_observed(&team->self_member);
    const auto signals = simt::load_observed(&team->signal_count);
    const auto counters = simt::load_observed(&team->counter_count);
    const auto barriers = simt::load_observed(&team->barrier_count);
    const auto memories_address =
        simt::load_observed(&team->remote_sync_memories);
    if (static_cast<std::uint32_t>(source_rank) >= members || self >= members ||
        signals != sync_layout::kWorldTeamSignalCount ||
        counters != sync_layout::kWorldTeamCounterCount ||
        barriers < sync_layout::kWorldTeamBarrierCount ||
        signal_index >= sync_layout::kLogicalSignalCount ||
        memories_address == 0)
        return nullptr;
    auto* memories = reinterpret_cast<__gm__ cann_abi::Memory*>(
        memories_address);
    const auto base = simt::load_observed(&memories[self].address);
    const auto offset =
        (static_cast<std::uint64_t>(signal_index) * members +
         static_cast<std::uint32_t>(source_rank)) * sizeof(std::uint64_t);
    return reinterpret_cast<__gm__ std::uint64_t*>(
        base + offset);
}

DEEP_EP_ASCEND_SIMT_CALLEE void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    TransportCommandOpcode opcode, TransportTeam team, int peer,
    int world_peer, DeviceChannel channel) {
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
    simt::system_fence();
    simt::store_published(
        reinterpret_cast<__gm__ std::uint32_t*>(&output->error),
        static_cast<std::uint32_t>(error));
}

DEEP_EP_ASCEND_SIMT_CALLEE void record_error(
    __gm__ TransportCommandQueue* queue, DeviceTransportError error,
    TransportCommandOpcode opcode, int peer, DeviceChannel channel) {
    record_error(
        queue, error, opcode, TransportTeam::kWorld, peer, peer, channel);
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
        remote_action.kind != RemoteActionKind::kSignalIncrement) {
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
    if (channel != 0 || !detail::checked_world_peer(
            context.topology, team, source_rank, &world_peer)) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidRank,
            TransportCommandOpcode::kSignal, team, source_rank, world_peer,
            channel);
        return;
    }
    auto* address = detail::signal_address(context, world_peer, signal_index);
    if (address == nullptr) {
        detail::record_error(
            queue, DeviceTransportError::kInvalidAddress,
            TransportCommandOpcode::kSignal, team, source_rank, world_peer,
            channel);
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
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    CooperationScope, DeviceRequest*) {}

DEEP_EP_ASCEND_SIMT_CALLEE void wait(
    const DeviceTransportContext&, DeviceRequest*) {}

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
    if (team_mask != 1) {
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
