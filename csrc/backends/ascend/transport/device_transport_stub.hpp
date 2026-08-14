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
    const DeviceTransportContext& context, const void* local_pointer) {
    const auto pointer = reinterpret_cast<std::uintptr_t>(local_pointer);
    if (pointer == 0 || context.local_window_base == 0)
        return 0;
    return pointer - context.local_window_base;
}

DEEP_EP_ASCEND_SIMT_CALLEE void* get_symmetric_pointer(
    const DeviceTransportContext& context, TransportTeam team, int rank,
    void* local_pointer) {
    return is_peer_directly_accessible(context, team, rank) ? local_pointer : nullptr;
}

DEEP_EP_ASCEND_SIMT_CALLEE void put(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    void*, const void*, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions, const RemoteAction&) {}

DEEP_EP_ASCEND_SIMT_CALLEE void get(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    const void*, void*, std::size_t, CooperationScope, MemorySegment,
    DeviceOptions) {}

DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    void*, std::uint64_t, std::uint32_t, DeviceOptions) {}

DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
    const DeviceTransportContext&, DeviceChannel, TransportTeam, int,
    std::int64_t*, std::int64_t) {}

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
    const std::uint64_t*) {
    return 0;
}

DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
    std::uint64_t*, std::uint64_t) {}

DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() {}

DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
    const DeviceTransportContext&, std::uint32_t, void*, std::uint64_t) {}

}  // namespace deep_ep::ascend::transport::device
