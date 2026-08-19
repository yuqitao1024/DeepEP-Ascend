#include "stream_event.hpp"

#include <chrono>
#include <thread>
#include <utility>

#if __has_include(<acl/acl_rt.h>) && \
    __has_include(<torch_npu/csrc/core/npu/NPUStream.h>)
#define DEEP_EP_ASCEND_HAS_STREAM_EVENT_RUNTIME 1
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#else
#define DEEP_EP_ASCEND_HAS_STREAM_EVENT_RUNTIME 0
#endif

namespace deep_ep::ascend::runtime {
namespace {

using transport::TransportStatus;

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
        return -1;
    *stream = stream_identity(c10_npu::getCurrentNPUStream());
    return stream->raw == nullptr ? -1 : 0;
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
            return -1;
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
        (void)destroy();
    } catch (...) {
    }
}

TransportStatus NativeEventState::record(StreamIdentity stream) {
    if (state_ != State::Created)
        return TransportStatus::invalid(
            "record_event", "native event is not ready to record");
    if (!same_device(stream, device_index_))
        return TransportStatus::invalid(
            "record_event", "stream does not belong to the event device");
    const int result = api_.record_event(api_.user_data, native_event_, stream.raw);
    if (result != 0)
        return backend_failure("record_event", result);
    state_ = State::Recorded;
    return TransportStatus::success();
}

TransportStatus NativeEventState::wait(StreamIdentity stream) const {
    if (state_ != State::Recorded && state_ != State::Completed)
        return TransportStatus::invalid(
            "wait_event", "native event has not been recorded");
    if (!same_device(stream, device_index_))
        return TransportStatus::invalid(
            "wait_event", "stream does not belong to the event device");
    const int result = api_.wait_event(api_.user_data, stream.raw, native_event_);
    return result == 0 ? TransportStatus::success() :
        backend_failure("wait_event", result);
}

TransportStatus NativeEventState::finish(std::uint64_t timeout_ms) {
    if (state_ == State::Completed)
        return TransportStatus::success();
    if (state_ != State::Recorded)
        return TransportStatus::invalid(
            "synchronize_event", "native event has not been recorded");
    const int result = api_.synchronize_event(
        api_.user_data, native_event_, timeout_ms);
    if (result != 0)
        return backend_failure("synchronize_event", result);
    state_ = State::Completed;
    return TransportStatus::success();
}

TransportStatus NativeEventState::destroy() {
    if (state_ == State::Destroyed)
        return TransportStatus::success();
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

NativeEventCreateResult create_native_event(
    StreamEventApi api, int device_index) {
    if (device_index < 0 || !valid_api(api)) {
        return {TransportStatus::invalid(
                    "create_native_event", "invalid stream event runtime API"),
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
