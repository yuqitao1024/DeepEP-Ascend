#pragma once

#include "device_transport_stub.hpp"

namespace deep_ep::ascend::transport {

class DeviceTransportFacade {
public:
    DEEP_EP_ASCEND_SIMT_CALLEE DeviceTransportFacade(
        const DeviceTransportContext& context, device::DeviceChannel channel)
        : context_(context), channel_(channel) {}

    DEEP_EP_ASCEND_SIMT_CALLEE bool is_peer_directly_accessible(
        TransportTeam team, int rank) const {
        return device::is_peer_directly_accessible(context_, team, rank);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t get_symmetric_offset(
        const void* local_pointer) const {
        return device::get_symmetric_offset(context_, local_pointer);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void* get_symmetric_pointer(
        TransportTeam team, int rank, void* local_pointer) const {
        return device::get_symmetric_pointer(context_, team, rank, local_pointer);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void put(
        TransportTeam team, int destination_rank, void* destination,
        const void* source, std::size_t bytes, CooperationScope scope,
        MemorySegment segment, DeviceOptions options,
        const RemoteAction& remote_action) const {
        device::put(context_, channel_, team, destination_rank, destination,
                    source, bytes, scope, segment, options, remote_action);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void get(
        TransportTeam team, int source_rank, const void* source,
        void* destination, std::size_t bytes, CooperationScope scope,
        MemorySegment segment, DeviceOptions options) const {
        device::get(context_, channel_, team, source_rank, source, destination,
                    bytes, scope, segment, options);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
        TransportTeam team, int destination_rank, void* destination,
        std::uint64_t value, std::uint32_t value_bytes,
        DeviceOptions options) const {
        device::put_value(context_, channel_, team, destination_rank,
                          destination, value, value_bytes, options);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
        TransportTeam team, int destination_rank, std::int64_t* destination,
        std::int64_t value) const {
        device::remote_add_release(context_, channel_, team, destination_rank,
                                   destination, value);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void signal(
        TransportTeam team, int destination_rank,
        const RemoteAction& remote_action) const {
        device::signal(context_, channel_, team, destination_rank, remote_action);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE SignalValue read_signal(
        TransportTeam team, int source_rank, std::uint32_t signal_index) const {
        return device::read_signal(
            context_, channel_, team, source_rank, signal_index);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void wait_signal(
        TransportTeam team, int source_rank, std::uint32_t signal_index,
        SignalValue target, std::uint64_t timeout_cycles) const {
        device::wait_signal(context_, channel_, team, source_rank, signal_index,
                            target, timeout_cycles);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void flush(CooperationScope scope) const {
        device::flush(context_, channel_, scope);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void flush_async(
        TransportTeam team, int peer_rank, CooperationScope scope,
        DeviceRequest* request) const {
        device::flush_async(
            context_, channel_, team, peer_rank, scope, request);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void wait(DeviceRequest* request) const {
        device::wait(context_, request);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t load_acquire(
        const std::uint64_t* pointer) const {
        return device::load_acquire(pointer);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
        std::uint64_t* pointer, std::uint64_t value) const {
        device::store_release(pointer, value);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() const {
        device::system_fence();
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
        std::uint32_t team_mask, void* workspace,
        std::uint64_t timeout_cycles) const {
        device::device_barrier(
            context_, team_mask, workspace, timeout_cycles);
    }

private:
    const DeviceTransportContext& context_;
    device::DeviceChannel channel_;
};

}  // namespace deep_ep::ascend::transport
