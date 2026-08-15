#pragma once

#include "device_transport.hpp"

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
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    DeviceAddress, DeviceAddress, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions, const RemoteAction&) {}

DEEP_EP_ASCEND_SIMT_CALLEE void get(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    DeviceAddress, DeviceAddress, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions) {}

DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    DeviceAddress, std::uint64_t, std::uint32_t, DeviceOptions) {}

DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    DeviceAddress, std::int64_t) {}

DEEP_EP_ASCEND_SIMT_CALLEE void signal(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    const RemoteAction&) {}

DEEP_EP_ASCEND_SIMT_CALLEE SignalValue read_signal(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    std::uint32_t) {
    return 0;
}

DEEP_EP_ASCEND_SIMT_CALLEE void wait_signal(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    std::uint32_t, SignalValue, std::uint64_t) {}

DEEP_EP_ASCEND_SIMT_CALLEE void flush(
    const DeviceTransportContext&, DeviceChannel, CooperationScope) {}

DEEP_EP_ASCEND_SIMT_CALLEE void flush_async(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    CooperationScope, DeviceRequest*) {}

DEEP_EP_ASCEND_SIMT_CALLEE void wait(
    const DeviceTransportContext&, DeviceRequest*) {}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t load_acquire(
    DeviceAddress) {
    return 0;
}

DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
    DeviceAddress, std::uint64_t) {}

DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() {}

DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t consumed_generation(
    const DeviceTransportContext&) {
    return 0;
}

DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
    const DeviceTransportContext&, std::uint32_t, DeviceAddress,
    std::uint64_t) {}

}  // namespace deep_ep::ascend::transport::device
