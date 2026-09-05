#pragma once

#include <cstddef>

#include "../transport/device_topology.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE __SIMT_DEVICE_FUNCTIONS_DECL__
#else
#define DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE
#endif

namespace deep_ep::ascend::elastic::release_protocol {

struct ReleaseControlObservation {
    bool acquired = false;
    std::uint64_t generation = 0;
    std::uint64_t count = 0;
};

template <typename Transport>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE void put_staged_payload(
    Transport& facade, const transport::TeamPeer& route,
    transport::DeviceAddress destination, transport::DeviceAddress source,
    std::size_t bytes) {
    facade.system_fence();
    if (bytes == 0)
        return;
    facade.put(
        route.team, route.peer, destination, source, bytes,
        transport::CooperationScope::kParticipant,
        transport::MemorySegment::kDevice, transport::kDefaultOptions,
        transport::RemoteAction::none());
}

template <typename Transport>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE void put_staged_records_striped(
    Transport& facade, const transport::TeamPeer& route,
    transport::DeviceAddress destination, transport::DeviceAddress source,
    std::uint64_t record_count, std::uint64_t record_bytes) {
    facade.system_fence();
    if (record_count == 0 || record_bytes == 0)
        return;

    const std::uint32_t available_channels =
        facade.channel_count(route.team, route.peer);
    const std::uint32_t channel_count = available_channels >
            static_cast<std::uint32_t>(transport::kMaxTransportChannels) ?
        static_cast<std::uint32_t>(transport::kMaxTransportChannels) :
        available_channels;
    if (channel_count == 0)
        return;
    const std::uint64_t records_per_channel =
        record_count / channel_count;
    const std::uint64_t remainder = record_count % channel_count;
    for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
        const std::uint64_t channel_records = records_per_channel +
            (channel < remainder ? 1 : 0);
        if (channel_records == 0)
            continue;
        const std::uint64_t record_begin =
            static_cast<std::uint64_t>(channel) * records_per_channel +
            (channel < remainder ? channel : remainder);
        const std::uint64_t byte_offset = record_begin * record_bytes;
        facade.put_on_channel(
            channel, route.team, route.peer, destination + byte_offset,
            source + byte_offset, channel_records * record_bytes,
            transport::CooperationScope::kParticipant,
            transport::MemorySegment::kDevice, transport::kDefaultOptions,
            transport::RemoteAction::none());
    }
}

template <typename Transport>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE void flush_payload(
    Transport& facade) {
    facade.flush(transport::CooperationScope::kDevice);
}

template <typename Transport>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE void publish_control_and_release(
    Transport& facade, const transport::TeamPeer& route,
    transport::DeviceAddress count_address, std::uint64_t count,
    transport::DeviceAddress generation_address, std::uint64_t generation,
    std::uint32_t signal_index) {
    facade.put_value(
        route.team, route.peer, count_address, count, sizeof(std::uint64_t),
        transport::kDefaultOptions);
    facade.put_value(
        route.team, route.peer, generation_address, generation,
        sizeof(std::uint64_t), transport::kDefaultOptions);
    facade.signal(
        route.team, route.peer,
        transport::RemoteAction::signal_set(signal_index, generation));
}

template <typename Transport>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE void publish_local_control(
    Transport& facade, transport::DeviceAddress count_address,
    std::uint64_t count, transport::DeviceAddress generation_address,
    std::uint64_t generation) {
    facade.system_fence();
    facade.store_release(count_address, count);
    facade.store_release(generation_address, generation);
}

template <typename Transport>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE bool acquire_release(
    Transport& facade, const transport::TransportTopology& topology,
    int signal_sender_world_rank, std::uint32_t signal_index,
    std::uint64_t generation, std::uint64_t timeout_cycles) {
    transport::TeamPeer route{};
    if (!transport::device::detail::checked_device_team_peer_for_world_rank(
            topology, signal_sender_world_rank, &route))
        return false;
    facade.wait_signal(
        route.team, route.peer, signal_index, generation, timeout_cycles);
    return facade.read_signal(route.team, route.peer, signal_index) >=
        generation;
}

template <typename Transport, typename Boundary, typename ControlSlots>
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE ReleaseControlObservation
observe_release_control(
    Transport& facade, const transport::TransportTopology& topology,
    const Boundary& boundary, int local_world_rank,
    ControlSlots control_slots, std::uint32_t signal_index,
    std::uint64_t generation,
    std::uint64_t timeout_cycles) {
    ReleaseControlObservation observation{};
    if (boundary.control_slot_world_rank < 0 ||
        boundary.control_slot_world_rank >= topology.world_size ||
        control_slots == nullptr ||
        sizeof(*control_slots) < 2 * sizeof(std::uint64_t))
        return observation;
    if (boundary.remote_acquire_required &&
        boundary.control_slot_world_rank != local_world_rank &&
        !acquire_release(
            facade, topology, boundary.signal_sender_world_rank,
            signal_index, generation, timeout_cycles))
        return observation;
    const auto slot =
        reinterpret_cast<transport::DeviceAddress>(control_slots) +
        static_cast<std::uint64_t>(boundary.control_slot_world_rank) *
            sizeof(*control_slots);
    observation.generation = facade.load_acquire(slot);
    observation.count = facade.load_acquire(
        slot + sizeof(std::uint64_t));
    observation.acquired = true;
    return observation;
}

}  // namespace deep_ep::ascend::elastic::release_protocol

#undef DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE
