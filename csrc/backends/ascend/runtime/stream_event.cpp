#include "stream_event.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>

#if __has_include(<torch_npu/csrc/core/npu/NPUStream.h>) && \
    __has_include(<acl/acl_rt.h>)
#define DEEP_EP_ASCEND_HAS_STREAM_EVENT_RUNTIME 1
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl_rt.h>
#else
#define DEEP_EP_ASCEND_HAS_STREAM_EVENT_RUNTIME 0
#endif

namespace deep_ep::ascend::runtime {
namespace {

using transport::TransportStatus;

constexpr int kEventTimeoutBackendCode = -1;

#if DEEP_EP_ASCEND_TESTING
constexpr int kTestingFaultBackendCode = -2;

bool testing_fault_enabled(const char* expected) {
    const char* value = std::getenv(
        "DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT");
    return value != nullptr && std::strcmp(value, expected) == 0;
}
#endif

bool valid_api(const StreamEventApi& api) {
    return api.current_device != nullptr && api.current_stream != nullptr &&
           api.pool_stream != nullptr && api.create_event != nullptr &&
           api.record_event != nullptr && api.query_event != nullptr &&
           api.wait_event != nullptr && api.synchronize_event != nullptr &&
           api.destroy_event != nullptr;
}

TransportStatus backend_failure(const char* operation, int backend_code) {
    return TransportStatus::runtime_failure(
        operation, backend_code, "CANN stream event call failed");
}

bool same_device(const StreamIdentity& stream, int device_index) {
    return stream.raw != nullptr && stream.device_index == device_index;
}

#if DEEP_EP_ASCEND_HAS_STREAM_EVENT_RUNTIME

StreamIdentity stream_identity(c10_npu::NPUStream stream) {
    return {stream.stream(false), static_cast<std::int64_t>(stream.id()),
            stream.device_index(), static_cast<int>(stream.device_type())};
}

int runtime_current_device(void*, int* device_index) {
    return aclrtGetDevice(device_index);
}

int runtime_current_stream(void*, StreamIdentity* stream) {
    if (stream == nullptr)
        return kEventTimeoutBackendCode;
    int device_index = -1;
    const int result = runtime_current_device(nullptr, &device_index);
    if (result != ACL_SUCCESS || device_index < 0)
        return result == ACL_SUCCESS ? kEventTimeoutBackendCode : result;
    *stream = stream_identity(c10_npu::getCurrentNPUStream(device_index));
    return stream->raw == nullptr ? kEventTimeoutBackendCode : ACL_SUCCESS;
}

int runtime_pool_stream(
    void*, int device_index, bool high_priority, StreamIdentity* stream) {
    if (stream == nullptr)
        return -1;
    *stream = stream_identity(
        c10_npu::getStreamFromPool(high_priority, device_index));
    return stream->raw == nullptr ? -1 : 0;
}

int runtime_create_event(void*, void** event) {
    return aclrtCreateEventWithFlag(
        reinterpret_cast<aclrtEvent*>(event), ACL_EVENT_SYNC);
}

int runtime_record_event(void*, void* event, void* stream) {
    return aclrtRecordEvent(
        static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream));
}

int runtime_query_event(void*, void* event, bool* completed) {
    if (completed == nullptr)
        return -1;
    aclrtEventRecordedStatus status{};
    const int result = aclrtQueryEventStatus(
        static_cast<aclrtEvent>(event), &status);
    if (result != ACL_SUCCESS)
        return result;
    if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        *completed = true;
        return ACL_SUCCESS;
    }
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
        *completed = false;
        return ACL_SUCCESS;
    }
    return -1;
}

int runtime_wait_event(void*, void* stream, void* event) {
    return aclrtStreamWaitEvent(
        static_cast<aclrtStream>(stream), static_cast<aclrtEvent>(event));
}

int runtime_synchronize_event(void* user_data, void* event,
                              std::uint64_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    while (true) {
        bool completed = false;
        const int result = runtime_query_event(user_data, event, &completed);
        if (result != ACL_SUCCESS || completed)
            return result;
        if (std::chrono::steady_clock::now() >= deadline)
            return kEventTimeoutBackendCode;
        std::this_thread::yield();
    }
}

int runtime_destroy_event(void*, void* event) {
    return aclrtDestroyEvent(static_cast<aclrtEvent>(event));
}

#endif

}  // namespace

StreamEventApi make_stream_event_api() {
#if DEEP_EP_ASCEND_HAS_STREAM_EVENT_RUNTIME
    return {nullptr, runtime_current_device, runtime_current_stream,
            runtime_pool_stream, runtime_create_event, runtime_record_event,
            runtime_query_event, runtime_wait_event, runtime_synchronize_event,
            runtime_destroy_event};
#else
    return {};
#endif
}

NativeEventState::~NativeEventState() {
    try {
        if (state_ == State::Recorded && !finish(0).ok())
            return;
        (void)destroy();
    } catch (...) {
    }
}

TransportStatus NativeEventState::record(StreamIdentity stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::Created)
        return TransportStatus::invalid(
            "record_event", "native event is not ready to record");
    if (!same_device(stream, device_index_))
        return TransportStatus::invalid(
            "record_event", "stream does not belong to the event device");
#if DEEP_EP_ASCEND_TESTING
    if (testing_fault_enabled("record_failure"))
        return backend_failure("record_event", kTestingFaultBackendCode);
#endif
    const int result = api_.record_event(api_.user_data, native_event_, stream.raw);
    if (result != 0)
        return backend_failure("record_event", result);
    state_ = State::Recorded;
    return TransportStatus::success();
}

TransportStatus NativeEventState::wait(
    StreamIdentity stream,
    std::shared_ptr<NativeEventWaitLease>* wait_lease) {
    if (wait_lease == nullptr || *wait_lease != nullptr)
        return TransportStatus::invalid(
            "wait_event", "native event wait lease is unavailable");
    auto candidate = std::shared_ptr<NativeEventWaitLease>(
        new NativeEventWaitLease(shared_from_this()));
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::Recorded && state_ != State::Completed)
        return TransportStatus::invalid(
            "wait_event", "native event has not been recorded");
    if (!same_device(stream, device_index_))
        return TransportStatus::invalid(
            "wait_event", "stream does not belong to the event device");
    const int result = api_.wait_event(api_.user_data, stream.raw, native_event_);
    if (result != 0)
        return backend_failure("wait_event", result);
    ++active_wait_leases_;
    *wait_lease = std::move(candidate);
    return TransportStatus::success();
}

TransportStatus NativeEventState::current_stream(StreamIdentity* stream) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream == nullptr || api_.current_stream == nullptr)
        return TransportStatus::invalid(
            "current_stream_wait", "invalid current stream request");
    const int result = api_.current_stream(api_.user_data, stream);
    if (result != 0)
        return backend_failure("current_stream_wait", result);
    if (!same_device(*stream, device_index_))
        return TransportStatus::invalid(
            "current_stream_wait",
            "current NPU device does not belong to the event device");
    return TransportStatus::success();
}

TransportStatus NativeEventState::finish(std::uint64_t timeout_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == State::Completed)
        return TransportStatus::success();
    if (state_ != State::Recorded)
        return TransportStatus::invalid(
            "synchronize_event", "native event has not been recorded");
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    while (true) {
        bool completed = false;
        bool query_runtime = true;
#if DEEP_EP_ASCEND_TESTING
        query_runtime = !testing_fault_enabled("query_not_ready");
#endif
        if (query_runtime) {
            const int result = api_.query_event(
                api_.user_data, native_event_, &completed);
            if (result != 0)
                return backend_failure("query_event", result);
        }
        if (completed) {
            state_ = State::Completed;
            return TransportStatus::success();
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return backend_failure(
                "synchronize_event", kEventTimeoutBackendCode);
        }
        std::this_thread::yield();
    }
}

TransportStatus NativeEventState::destroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == State::Destroyed)
        return TransportStatus::success();
    if (state_ == State::Recorded)
        return TransportStatus::invalid(
            "destroy_event", "native event is not complete");
    if (active_wait_leases_ != 0)
        return TransportStatus::invalid(
            "destroy_event", "native event has outstanding stream waits");
#if DEEP_EP_ASCEND_TESTING
    if (testing_fault_enabled("destroy_failure"))
        return backend_failure("destroy_event", kTestingFaultBackendCode);
#endif
    const int result = api_.destroy_event(api_.user_data, native_event_);
    if (result != 0)
        return backend_failure("destroy_event", result);
    native_event_ = nullptr;
    state_ = State::Destroyed;
    return TransportStatus::success();
}

int NativeEventState::device_index() const noexcept {
    return device_index_;
}

void NativeEventState::release_wait_lease() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_wait_leases_ != 0)
        --active_wait_leases_;
}

NativeEventWaitLease::~NativeEventWaitLease() {
    if (event_ != nullptr)
        event_->release_wait_lease();
}

NativeEventCreateResult create_native_event(
    StreamEventApi api, int device_index) {
    if (device_index < 0 || !valid_api(api)) {
        return {TransportStatus::invalid(
                    "create_native_event", "invalid stream event runtime API"),
                nullptr};
    }
    int current_device = -1;
    const int current_device_result = api.current_device(
        api.user_data, &current_device);
    if (current_device_result != 0)
        return {backend_failure("current_device", current_device_result), nullptr};
    if (current_device != device_index) {
        return {TransportStatus::invalid(
                    "current_device", "requested event device is not current"),
                nullptr};
    }
    void* native_event = nullptr;
    const int result = api.create_event(api.user_data, &native_event);
    if (result != 0)
        return {backend_failure("create_event", result), nullptr};
    if (native_event == nullptr) {
        return {TransportStatus::runtime_failure(
                    "create_event", 0, "CANN returned a null native event"),
                nullptr};
    }
    return {TransportStatus::success(), std::shared_ptr<NativeEventState>(
                new NativeEventState(api, native_event, device_index))};
}

}  // namespace deep_ep::ascend::runtime
