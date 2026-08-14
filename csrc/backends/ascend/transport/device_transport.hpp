#pragma once

#include <cstddef>
#include <cstdint>

#include "types.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_SIMT_CALLEE __simt_callee__ inline
#else
#define DEEP_EP_ASCEND_SIMT_CALLEE inline
#endif

namespace deep_ep::ascend::transport::device {

using DeviceChannel = std::uint32_t;

DEEP_EP_ASCEND_SIMT_CALLEE bool is_peer_directly_accessible(
    const DeviceTransportContext&, TransportTeam, int rank);
DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t get_symmetric_offset(
    const DeviceTransportContext&, const void* local_pointer);
DEEP_EP_ASCEND_SIMT_CALLEE void* get_symmetric_pointer(
    const DeviceTransportContext&, TransportTeam, int rank, void* local_pointer);

DEEP_EP_ASCEND_SIMT_CALLEE void put(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int destination_rank, void* destination, const void* source,
    std::size_t bytes, CooperationScope, MemorySegment, DeviceOptions,
    const RemoteAction&);
DEEP_EP_ASCEND_SIMT_CALLEE void get(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int source_rank, const void* source, void* destination,
    std::size_t bytes, CooperationScope, MemorySegment, DeviceOptions);
DEEP_EP_ASCEND_SIMT_CALLEE void put_value(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int destination_rank, void* destination, std::uint64_t value,
    std::uint32_t value_bytes, DeviceOptions);
DEEP_EP_ASCEND_SIMT_CALLEE void remote_add_release(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int destination_rank, std::int64_t* destination, std::int64_t value);
DEEP_EP_ASCEND_SIMT_CALLEE void signal(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int destination_rank, const RemoteAction&);
DEEP_EP_ASCEND_SIMT_CALLEE SignalValue read_signal(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int source_rank, std::uint32_t signal_index);
DEEP_EP_ASCEND_SIMT_CALLEE void wait_signal(
    const DeviceTransportContext&, DeviceChannel, TransportTeam,
    int source_rank, std::uint32_t signal_index, SignalValue target,
    std::uint64_t timeout_cycles);
DEEP_EP_ASCEND_SIMT_CALLEE void flush(
    const DeviceTransportContext&, DeviceChannel, CooperationScope);
DEEP_EP_ASCEND_SIMT_CALLEE void flush_async(
    const DeviceTransportContext&, DeviceChannel, int peer_rank,
    CooperationScope, DeviceRequest* request);
DEEP_EP_ASCEND_SIMT_CALLEE void wait(
    const DeviceTransportContext&, DeviceRequest* request);
DEEP_EP_ASCEND_SIMT_CALLEE std::uint64_t load_acquire(
    const std::uint64_t* pointer);
DEEP_EP_ASCEND_SIMT_CALLEE void store_release(
    std::uint64_t* pointer, std::uint64_t value);
DEEP_EP_ASCEND_SIMT_CALLEE void system_fence();
DEEP_EP_ASCEND_SIMT_CALLEE void device_barrier(
    const DeviceTransportContext&, std::uint32_t team_mask,
    void* workspace, std::uint64_t timeout_cycles);

}  // namespace deep_ep::ascend::transport::device
