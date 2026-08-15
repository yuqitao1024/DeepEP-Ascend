#pragma once

#include <cstdint>

#include "host_transport.hpp"

namespace deep_ep::ascend::transport {

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

TransportCreateResult make_cann_transport(
    const TransportConfig& config, const CannHostApi& api);

TransportCreateResult make_cann_transport(const TransportConfig& config);

}  // namespace deep_ep::ascend::transport
