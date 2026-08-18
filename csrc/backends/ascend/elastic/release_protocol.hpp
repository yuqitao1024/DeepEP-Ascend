#pragma once

#include "../transport/device_topology.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE __SIMT_DEVICE_FUNCTIONS_DECL__
#else
#define DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE
#endif

namespace deep_ep::ascend::elastic::release_protocol {

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
DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE bool acquire_release(
    Transport& facade, const transport::TransportTopology& topology,
    int source_world_rank, std::uint32_t signal_index,
    std::uint64_t generation, std::uint64_t timeout_cycles) {
    transport::TeamPeer route{};
    if (!transport::device::detail::checked_device_team_peer_for_world_rank(
            topology, source_world_rank, &route))
        return false;
    facade.wait_signal(
        route.team, route.peer, signal_index, generation, timeout_cycles);
    return facade.read_signal(route.team, route.peer, signal_index) >=
        generation;
}

}  // namespace deep_ep::ascend::elastic::release_protocol

#undef DEEP_EP_ASCEND_RELEASE_PROTOCOL_CALLEE
