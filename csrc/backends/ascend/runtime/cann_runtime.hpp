#pragma once

#include <cstdint>
#include <memory>

#include "../elastic/layout.hpp"
#include "../transport/cann_transport.hpp"

namespace deep_ep::ascend::runtime {

struct CannRuntimeApi {
    void* user_data = nullptr;
    int (*allocate_device)(void*, std::uint64_t, void**) = nullptr;
    int (*zero_device)(void*, void*, std::uint64_t) = nullptr;
    int (*free_device)(void*, void*) = nullptr;
    int (*current_device)(void*, int*) = nullptr;
    void* (*current_stream)(void*) = nullptr;
    int (*synchronize_stream)(void*, void*) = nullptr;
    int (*synchronize_device)(void*) = nullptr;
    int (*copy_from_host)(void*, void*, const void*, std::uint64_t) = nullptr;
    int (*copy_to_host)(void*, void*, const void*, std::uint64_t) = nullptr;
};

CannRuntimeApi make_cann_runtime_api();

struct CannRuntimeAllocation {
    void* owner = nullptr;
    void* aligned = nullptr;
    std::uint64_t owner_bytes = 0;
    std::uint64_t bytes = 0;
};

class CannRuntimeResources {
public:
    CannRuntimeResources() = default;
    ~CannRuntimeResources();

    CannRuntimeResources(const CannRuntimeResources&) = delete;
    CannRuntimeResources& operator=(const CannRuntimeResources&) = delete;

    transport::TransportStatus initialize(
        const transport::TransportConfig& config,
        std::uint64_t workspace_bytes);
    transport::TransportStatus initialize(
        const transport::TransportConfig& config,
        std::uint64_t workspace_bytes, const CannRuntimeApi& runtime_api,
        const transport::CannHostApi& host_api);
    transport::TransportStatus destroy();

    bool initialized() const noexcept { return initialized_; }
    void* window_base() const noexcept { return window_.aligned; }
    void* workspace() const noexcept { return workspace_.aligned; }
    std::uint64_t workspace_bytes() const noexcept { return workspace_.bytes; }
    transport::HostTransport* transport() const noexcept {
        return transport_.get();
    }
    const transport::DeviceTransportContext& device_context() const noexcept {
        return device_context_;
    }
    int owning_device() const noexcept { return owning_device_; }

    transport::TransportStatus current_device(int* device);
    transport::TransportStatus current_stream(void** stream);
    transport::TransportStatus synchronize_stream(void* stream);
    transport::TransportStatus synchronize_device();
    transport::TransportStatus copy_from_host(
        void* destination, const void* source, std::uint64_t bytes);
    transport::TransportStatus copy_to_host(
        void* destination, const void* source, std::uint64_t bytes);

private:
    transport::TransportStatus initialize_impl(
        const transport::TransportConfig& config,
        std::uint64_t workspace_bytes, const CannRuntimeApi& runtime_api,
        const transport::CannHostApi* host_api);
    transport::TransportStatus allocate(
        std::uint64_t bytes, std::uint64_t alignment,
        CannRuntimeAllocation* allocation, const char* operation);
    void free_allocation(
        CannRuntimeAllocation& allocation, const char* operation,
        transport::TransportStatus& first_error);

    CannRuntimeApi runtime_api_{};
    CannRuntimeAllocation window_{};
    CannRuntimeAllocation workspace_{};
    std::unique_ptr<transport::HostTransport> transport_;
    transport::DeviceTransportContext device_context_{};
    int owning_device_ = -1;
    bool owns_resources_ = false;
    bool initialized_ = false;
};

}  // namespace deep_ep::ascend::runtime
