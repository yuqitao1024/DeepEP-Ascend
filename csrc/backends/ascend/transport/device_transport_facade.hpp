#pragma once

#if defined(DEEP_EP_ASCEND_STAGED_URMA) && DEEP_EP_ASCEND_STAGED_URMA
#include "device_transport_commands.hpp"
#else
#include "device_transport_stub.hpp"
#endif

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
        DeviceAddress local_address) const {
        return device::get_symmetric_offset(context_, local_address);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE DeviceAddress get_symmetric_pointer(
        TransportTeam team, int rank, DeviceAddress local_address) const {
        return device::get_symmetric_pointer(context_, team, rank, local_address);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void put(
        TransportTeam team, int destination_rank, DeviceAddress destination,
        DeviceAddress source, std::size_t bytes, CooperationScope scope,
        MemorySegment segment, DeviceOptions options,
        const RemoteAction& remote_action) const {
        device::put(context_, channel_, team, destination_rank, destination,
                    source, bytes, scope, segment, options, remote_action);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void get(
        TransportTeam team, int source_rank, DeviceAddress source,
        DeviceAddress destination, std::size_t bytes, CooperationScope scope,
        MemorySegment segment, DeviceOptions options) const {
        device::get(context_, channel_, team, source_rank, source, destination,
                    bytes, scope, segment, options);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
        TransportTeam team, int destination_rank, DeviceAddress destination,
        std::uint64_t value, std::uint32_t value_bytes,
        DeviceOptions options) const {
        device::put_value(context_, channel_, team, destination_rank,
                          destination, value, value_bytes, options);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
        TransportTeam team, int destination_rank, DeviceAddress destination,
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
        DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) const {
        device::flush_async(
            context_, channel_, team, peer_rank, scope, request);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void wait(
        DEEP_EP_ASCEND_SIMT_GLOBAL DeviceRequest* request) const {
        device::wait(context_, request);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t load_acquire(
        DeviceAddress address) const {
        return device::load_acquire(address);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
        DeviceAddress address, std::uint64_t value) const {
        device::store_release(address, value);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() const {
        device::system_fence();
    }

    DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t consumed_generation() const {
        return device::consumed_generation(context_);
    }

    DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
        std::uint32_t team_mask, DeviceAddress workspace,
        std::uint64_t timeout_cycles) const {
        device::device_barrier(
            context_, team_mask, workspace, timeout_cycles);
    }

private:
    const DeviceTransportContext& context_;
    device::DeviceChannel channel_;
};

}  // namespace deep_ep::ascend::transport
