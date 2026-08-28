#pragma once

#include <cstdint>
#include <limits>

#include "host_transport.hpp"

namespace deep_ep::ascend::transport {

constexpr bool checked_scale_up_command_capacity(
    int world_size, std::uint32_t* capacity) noexcept {
    if (capacity == nullptr || world_size <= 0)
        return false;
    const auto peers = static_cast<std::uint64_t>(world_size - 1);
    // ProducerControl leaves the normal payload/control release batch in the
    // queue.  Early route publication adds one put and one signal per peer,
    // plus its two drains and post-publication barrier.
    constexpr std::uint64_t kCommandsPerPeer = 7;
    constexpr std::uint64_t kFixedCommands = 4;
    if (peers >
        (std::numeric_limits<std::uint32_t>::max() - kFixedCommands) /
            kCommandsPerPeer)
        return false;
    *capacity = static_cast<std::uint32_t>(
        peers * kCommandsPerPeer + kFixedCommands);
    return true;
}

struct CannHostApi {
    void* user_data = nullptr;

    int (*get_rank)(void*, std::int64_t, std::uint32_t*) = nullptr;
    int (*get_size)(void*, std::int64_t, std::uint32_t*) = nullptr;
    int (*create_world_team)(
        void*, std::int64_t, std::uint32_t, std::uint32_t,
        const std::uint32_t*, std::uint32_t, std::uint32_t,
        std::uintptr_t*) = nullptr;
    int (*register_window)(
        void*, std::int64_t, std::uintptr_t, void*, std::uint64_t,
        std::uintptr_t*) = nullptr;
    int (*create_channels)(
        void*, std::int64_t, std::uintptr_t, std::uint32_t) = nullptr;

    int (*allocate_device)(void*, std::uint64_t, void**) = nullptr;
    int (*zero_device)(void*, void*, std::uint64_t) = nullptr;
    int (*copy_to_device)(void*, void*, const void*, std::uint64_t) = nullptr;
    int (*copy_from_device)(void*, void*, const void*, std::uint64_t) = nullptr;
    int (*free_device)(void*, void*) = nullptr;

    int (*deregister_window)(
        void*, std::uintptr_t, std::uintptr_t) = nullptr;
    int (*destroy_team)(void*, std::uintptr_t) = nullptr;
    int (*host_barrier)(void*, std::int64_t) = nullptr;
};

TransportStatus query_cann_communicator_size(
    std::int64_t communicator_handle, std::uint32_t* world_size,
    const CannHostApi& api);

TransportStatus query_cann_communicator_size(
    std::int64_t communicator_handle, std::uint32_t* world_size);

TransportCreateResult make_cann_transport(
    const TransportConfig& config, const CannHostApi& api);

TransportCreateResult make_cann_transport(const TransportConfig& config);

}  // namespace deep_ep::ascend::transport
