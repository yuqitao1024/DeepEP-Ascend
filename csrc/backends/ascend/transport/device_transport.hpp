#pragma once

#include <cstddef>
#include <cstdint>

#include "types.hpp"

namespace deep_ep::ascend::transport::device {

using DeviceChannel = std::uint32_t;
using DeviceOptions = std::uint32_t;
inline constexpr DeviceOptions kDefaultOptions = 0;

bool is_peer_directly_accessible(
    const DeviceTransportContext&, TransportTeam, int rank);
std::uint64_t get_symmetric_offset(
    const DeviceTransportContext&, const void* local_pointer);
void* get_symmetric_pointer(
    const DeviceTransportContext&, TransportTeam, int rank, void* local_pointer);

void put(const DeviceTransportContext&, DeviceChannel, TransportTeam,
         int destination_rank, void* destination, const void* source,
         std::size_t bytes, CooperationScope, DeviceOptions);
void get(const DeviceTransportContext&, DeviceChannel, TransportTeam,
         int source_rank, const void* source, void* destination,
         std::size_t bytes, CooperationScope, DeviceOptions);
void put_value(const DeviceTransportContext&, DeviceChannel, TransportTeam,
               int destination_rank, void* destination,
               std::uint64_t value, std::uint32_t value_bytes, DeviceOptions);
void remote_add_release(const DeviceTransportContext&, DeviceChannel,
                        TransportTeam, int destination_rank,
                        std::int64_t* destination, std::int64_t value);
void signal(const DeviceTransportContext&, DeviceChannel, TransportTeam,
            int destination_rank, std::uint32_t signal_index,
            std::uint64_t value);
void flush(const DeviceTransportContext&, DeviceChannel, CooperationScope);
void flush_async(const DeviceTransportContext&, DeviceChannel, int peer_rank,
                 CooperationScope, DeviceRequest* request);
void wait(const DeviceTransportContext&, DeviceRequest* request);
std::uint64_t load_acquire(const std::uint64_t* pointer);
void store_release(std::uint64_t* pointer, std::uint64_t value);
void system_fence();
void device_barrier(const DeviceTransportContext&, std::uint32_t team_mask,
                    void* workspace, std::uint64_t timeout_cycles);

}  // namespace deep_ep::ascend::transport::device
