#include "cann_runtime.hpp"

#include "mapped_memory.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#if __has_include(<acl/acl.h>) && \
    __has_include(<torch_npu/csrc/core/npu/NPUStream.h>)
#define DEEP_EP_ASCEND_HAS_CANN_RUNTIME 1
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#else
#define DEEP_EP_ASCEND_HAS_CANN_RUNTIME 0
#endif

namespace deep_ep::ascend::runtime {
namespace {

using transport::TransportStatus;

bool valid_api(const CannRuntimeApi& api) {
    return api.allocate_device != nullptr && api.zero_device != nullptr &&
           api.free_device != nullptr && api.current_device != nullptr &&
           api.current_stream != nullptr &&
           api.synchronize_stream != nullptr &&
           api.synchronize_device != nullptr && api.copy_from_host != nullptr &&
           api.copy_to_host != nullptr;
}

TransportStatus backend_failure(const char* operation, int backend_code) {
    return TransportStatus::runtime_failure(
        operation, backend_code, "CANN runtime call failed");
}

void retain_first(
    TransportStatus candidate, TransportStatus& first_error) {
    if (!candidate.ok() && first_error.ok())
        first_error = std::move(candidate);
}

#if DEEP_EP_ASCEND_HAS_CANN_RUNTIME

int runtime_allocate(void*, std::uint64_t bytes, void** pointer) {
    return aclrtMalloc(
        pointer, static_cast<std::size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST);
}

int runtime_zero(void*, void* pointer, std::uint64_t bytes) {
    return aclrtMemset(
        pointer, static_cast<std::size_t>(bytes), 0,
        static_cast<std::size_t>(bytes));
}

int runtime_free(void*, void* pointer) {
    return aclrtFree(pointer);
}

int runtime_current_device(void*, int* device) {
    return aclrtGetDevice(device);
}

void* runtime_current_stream(void*) {
    return c10_npu::getCurrentNPUStream().stream(false);
}

int runtime_synchronize_stream(void*, void* stream) {
    return aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
}

int runtime_synchronize_device(void*) {
    return c10_npu::npuSynchronizeDevice(true) ? 0 : -1;
}

int runtime_copy_from_host(
    void*, void* destination, const void* source, std::uint64_t bytes) {
    return aclrtMemcpy(
        destination, static_cast<std::size_t>(bytes), source,
        static_cast<std::size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE);
}

int runtime_copy_to_host(
    void*, void* destination, const void* source, std::uint64_t bytes) {
    return aclrtMemcpy(
        destination, static_cast<std::size_t>(bytes), source,
        static_cast<std::size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST);
}

#endif

}  // namespace

CannRuntimeApi make_cann_runtime_api() {
#if DEEP_EP_ASCEND_HAS_CANN_RUNTIME
    return {nullptr, runtime_allocate, runtime_zero, runtime_free,
            runtime_current_device, runtime_current_stream, runtime_synchronize_stream,
            runtime_synchronize_device, runtime_copy_from_host,
            runtime_copy_to_host};
#else
    return {};
#endif
}

CannRuntimeResources::~CannRuntimeResources() {
    (void)destroy();
}

TransportStatus CannRuntimeResources::initialize(
    const transport::TransportConfig& config,
    std::uint64_t workspace_bytes) {
    return initialize_impl(
        config, workspace_bytes, make_cann_runtime_api(), nullptr);
}

TransportStatus CannRuntimeResources::initialize(
    const transport::TransportConfig& config,
    std::uint64_t workspace_bytes, const CannRuntimeApi& runtime_api,
    const transport::CannHostApi& host_api) {
    return initialize_impl(
        config, workspace_bytes, runtime_api, &host_api);
}

TransportStatus CannRuntimeResources::initialize_impl(
    const transport::TransportConfig& config,
    std::uint64_t workspace_bytes, const CannRuntimeApi& runtime_api,
    const transport::CannHostApi* host_api) {
    if (owns_resources_)
        return TransportStatus::invalid(
            "initialize_runtime", "runtime resources are already initialized");
    if (!valid_api(runtime_api))
        return TransportStatus::runtime_failure(
            "initialize_runtime", 0,
            "CANN and Torch NPU runtime APIs are unavailable at build time");
    std::uint32_t command_capacity = 0;
    if (config.world_size < 2 || config.rank < 0 ||
        config.rank >= config.world_size ||
        config.communicator_handle == 0 || config.device_buffer_bytes <= 0 ||
        !config.cpu_communicator_empty ||
        (config.cpu_buffer_bytes != 0 && !mapped_cpu_memory_supported()) ||
        config.requested_channels != 1 || workspace_bytes == 0 ||
        !transport::checked_scale_up_command_capacity(
            config.world_size, &command_capacity))
        return TransportStatus::invalid(
            "initialize_runtime", "invalid scale-up production configuration");

    runtime_api_ = runtime_api;
    owns_resources_ = true;
    TransportStatus status = TransportStatus::success();
    int device = -1;
    int result = runtime_api_.current_device(runtime_api_.user_data, &device);
    if (result != 0 || device < 0) {
        status = result == 0 ? TransportStatus::runtime_failure(
            "current_device", 0, "Torch NPU returned an invalid current device") :
            backend_failure("current_device", result);
        (void)destroy();
        return status;
    }
    owning_device_ = device;
    status = allocate(
        static_cast<std::uint64_t>(config.device_buffer_bytes),
        elastic::kPublicElasticBufferAlignment, &window_, "allocate_window");
    if (!status.ok()) {
        (void)destroy();
        return status;
    }

    auto created = host_api == nullptr ?
        transport::make_cann_transport(config) :
        transport::make_cann_transport(config, *host_api);
    if (!created.status.ok()) {
        status = std::move(created.status);
        (void)destroy();
        return status;
    }
    transport_ = std::move(created.transport);

    status = transport_->register_symmetric_window(
        window_.aligned, config.device_buffer_bytes);
    if (!status.ok()) {
        (void)destroy();
        return status;
    }
    status = transport_->acquire_channels(
        1, transport::CooperationScope::kParticipant);
    if (!status.ok()) {
        (void)destroy();
        return status;
    }
    status = transport_->export_device_context(&device_context_);
    if (!status.ok()) {
        (void)destroy();
        return status;
    }
    status = allocate(
        workspace_bytes, elastic::kAscendElasticAlignment, &workspace_,
        "allocate_workspace");
    if (!status.ok()) {
        (void)destroy();
        return status;
    }
    void* initial_stream = runtime_api_.current_stream(runtime_api_.user_data);
    if (initial_stream == nullptr) {
        status = TransportStatus::runtime_failure(
            "current_stream", 0, "Torch NPU returned a null current stream");
        (void)destroy();
        return status;
    }
    initialized_ = true;
    return TransportStatus::success();
}

TransportStatus CannRuntimeResources::allocate(
    std::uint64_t bytes, std::uint64_t alignment,
    CannRuntimeAllocation* allocation, const char* operation) {
    if (allocation == nullptr || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0 ||
        bytes > std::numeric_limits<std::uint64_t>::max() - (alignment - 1))
        return TransportStatus::invalid(operation, "invalid allocation size");
    const auto owner_bytes = bytes + alignment - 1;
    void* owner = nullptr;
    int result = runtime_api_.allocate_device(
        runtime_api_.user_data, owner_bytes, &owner);
    if (result != 0)
        return backend_failure(operation, result);
    if (owner == nullptr)
        return TransportStatus::runtime_failure(
            operation, 0, "CANN returned a null device allocation");

    const auto address = reinterpret_cast<std::uintptr_t>(owner);
    const auto aligned_address = (address + alignment - 1) & ~(alignment - 1);
    void* aligned = reinterpret_cast<void*>(aligned_address);
    result = runtime_api_.zero_device(runtime_api_.user_data, aligned, bytes);
    if (result != 0) {
        (void)runtime_api_.free_device(runtime_api_.user_data, owner);
        return backend_failure(operation, result);
    }
    *allocation = {owner, aligned, owner_bytes, bytes};
    return TransportStatus::success();
}

void CannRuntimeResources::free_allocation(
    CannRuntimeAllocation& allocation, const char* operation,
    TransportStatus& first_error) {
    if (allocation.owner == nullptr)
        return;
    const int result = runtime_api_.free_device(
        runtime_api_.user_data, allocation.owner);
    if (result != 0) {
        retain_first(backend_failure(operation, result), first_error);
        return;
    }
    allocation = {};
}

TransportStatus CannRuntimeResources::destroy() {
    if (!owns_resources_)
        return TransportStatus::success();

    initialized_ = false;
    device_context_ = {};
    owning_device_ = -1;
    TransportStatus first_error = TransportStatus::success();
    free_allocation(workspace_, "free_workspace", first_error);
    if (transport_ != nullptr) {
        const auto status = transport_->destroy();
        retain_first(status, first_error);
        if (status.ok())
            transport_.reset();
    }
    if (transport_ == nullptr)
        free_allocation(window_, "free_window", first_error);
    if (first_error.ok()) {
        owns_resources_ = false;
        runtime_api_ = {};
    }
    return first_error;
}

TransportStatus CannRuntimeResources::current_stream(void** stream) {
    if (!initialized_ || stream == nullptr)
        return TransportStatus::invalid(
            "current_stream", "invalid runtime stream request");
    *stream = runtime_api_.current_stream(runtime_api_.user_data);
    return *stream != nullptr ? TransportStatus::success() :
        TransportStatus::runtime_failure(
            "current_stream", 0, "Torch NPU returned a null current stream");
}

TransportStatus CannRuntimeResources::current_device(int* device) {
    if (!initialized_ || device == nullptr)
        return TransportStatus::invalid(
            "current_device", "invalid runtime device request");
    const int result = runtime_api_.current_device(runtime_api_.user_data, device);
    return result == 0 && *device >= 0 ? TransportStatus::success() :
        result == 0 ? TransportStatus::runtime_failure(
            "current_device", 0, "Torch NPU returned an invalid current device") :
        backend_failure("current_device", result);
}

TransportStatus CannRuntimeResources::synchronize_stream(void* stream) {
    if (!initialized_ || stream == nullptr)
        return TransportStatus::invalid(
            "synchronize_stream", "invalid runtime stream request");
    const int result = runtime_api_.synchronize_stream(
        runtime_api_.user_data, stream);
    return result == 0 ? TransportStatus::success()
                       : backend_failure("synchronize_stream", result);
}

TransportStatus CannRuntimeResources::synchronize_device() {
    if (!initialized_)
        return TransportStatus::invalid(
            "synchronize_device", "runtime resources are not initialized");
    const int result = runtime_api_.synchronize_device(runtime_api_.user_data);
    return result == 0 ? TransportStatus::success()
                       : backend_failure("synchronize_device", result);
}

TransportStatus CannRuntimeResources::copy_from_host(
    void* destination, const void* source, std::uint64_t bytes) {
    if (!initialized_ || destination == nullptr || source == nullptr || bytes == 0)
        return TransportStatus::invalid(
            "copy_from_host", "invalid runtime copy request");
    const int result = runtime_api_.copy_from_host(
        runtime_api_.user_data, destination, source, bytes);
    return result == 0 ? TransportStatus::success()
                       : backend_failure("copy_from_host", result);
}

TransportStatus CannRuntimeResources::copy_to_host(
    void* destination, const void* source, std::uint64_t bytes) {
    if (!initialized_ || destination == nullptr || source == nullptr || bytes == 0)
        return TransportStatus::invalid(
            "copy_to_host", "invalid runtime copy request");
    const int result = runtime_api_.copy_to_host(
        runtime_api_.user_data, destination, source, bytes);
    return result == 0 ? TransportStatus::success()
                       : backend_failure("copy_to_host", result);
}

}  // namespace deep_ep::ascend::runtime
