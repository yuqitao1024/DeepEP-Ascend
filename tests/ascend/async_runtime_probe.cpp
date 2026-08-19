#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "csrc/backends/ascend/runtime/stream_event.hpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;

namespace {

template <typename T, typename = void>
struct has_synchronize_stream : std::false_type {};

template <typename T>
struct has_synchronize_stream<
    T, std::void_t<decltype(std::declval<T>().synchronize_stream)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_synchronize_device : std::false_type {};

template <typename T>
struct has_synchronize_device<
    T, std::void_t<decltype(std::declval<T>().synchronize_device)>>
    : std::true_type {};

static_assert(!has_synchronize_stream<runtime::StreamEventApi>::value);
static_assert(!has_synchronize_device<runtime::StreamEventApi>::value);

constexpr int kCreateFailure = 11;
constexpr int kRecordFailure = 12;
constexpr int kWaitFailure = 13;
constexpr int kTimeoutFailure = 14;
constexpr int kDestroyFailure = 15;

struct FakeApi {
    int create_result = 0;
    int record_result = 0;
    int wait_result = 0;
    int synchronize_result = 0;
    int destroy_result = 0;
    bool destroy_throws = false;
    int create_calls = 0;
    int record_calls = 0;
    int wait_calls = 0;
    int synchronize_calls = 0;
    int destroy_calls = 0;
    std::uint64_t last_timeout_ms = 0;
    void* native_event = reinterpret_cast<void*>(0x1000);
};

int current_device(void*, int* device) {
    *device = 2;
    return 0;
}

int current_stream(void*, runtime::StreamIdentity* stream) {
    *stream = {reinterpret_cast<void*>(0x2000), 1, 2, 9};
    return 0;
}

int pool_stream(void*, int, bool, runtime::StreamIdentity* stream) {
    *stream = {reinterpret_cast<void*>(0x3000), 2, 2, 9};
    return 0;
}

int create_event(void* user_data, void** event) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->create_calls;
    *event = fake->create_result == 0 ? fake->native_event : nullptr;
    return fake->create_result;
}

int record_event(void* user_data, void*, void*) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->record_calls;
    return fake->record_result;
}

int query_event(void*, void*, bool* completed) {
    *completed = true;
    return 0;
}

int wait_event(void* user_data, void*, void*) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->wait_calls;
    return fake->wait_result;
}

int synchronize_event(void* user_data, void*, std::uint64_t timeout_ms) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->synchronize_calls;
    fake->last_timeout_ms = timeout_ms;
    return fake->synchronize_result;
}

int destroy_event(void* user_data, void*) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->destroy_calls;
    if (fake->destroy_throws)
        throw std::runtime_error("destroy callback failure");
    return fake->destroy_result;
}

runtime::StreamEventApi api_for(FakeApi* fake) {
    return {fake, current_device, current_stream, pool_stream, create_event,
            record_event, query_event, wait_event, synchronize_event,
            destroy_event};
}

runtime::StreamIdentity stream_for(int device_index) {
    return {reinterpret_cast<void*>(0x4000), 3, device_index, 9};
}

bool is_failure(const transport::TransportStatus& status,
                transport::TransportStatusCode code, const char* operation,
                int backend_code = 0) {
    return status.code == code && status.operation == operation &&
           status.backend_code == backend_code;
}

bool invalid_callback_tables_are_rejected() {
    FakeApi fake;
    const auto complete = api_for(&fake);
    auto missing = complete;
    const auto rejects = [](const runtime::StreamEventApi& candidate) {
        const auto created = runtime::create_native_event(candidate, 2);
        return is_failure(
            created.status, transport::TransportStatusCode::kInvalidArgument,
            "create_native_event") && created.event == nullptr;
    };
    const auto reset_and_reject = [&complete, &missing, &rejects](auto member) {
        missing = complete;
        missing.*member = nullptr;
        return rejects(missing);
    };
    return reset_and_reject(&runtime::StreamEventApi::current_device) &&
        reset_and_reject(&runtime::StreamEventApi::current_stream) &&
        reset_and_reject(&runtime::StreamEventApi::pool_stream) &&
        reset_and_reject(&runtime::StreamEventApi::create_event) &&
        reset_and_reject(&runtime::StreamEventApi::record_event) &&
        reset_and_reject(&runtime::StreamEventApi::query_event) &&
        reset_and_reject(&runtime::StreamEventApi::wait_event) &&
        reset_and_reject(&runtime::StreamEventApi::synchronize_event) &&
        reset_and_reject(&runtime::StreamEventApi::destroy_event);
}

bool default_api_is_complete_or_unavailable() {
    const auto api = runtime::make_stream_event_api();
    const bool available = api.current_device != nullptr;
    return (available && api.current_stream != nullptr &&
            api.pool_stream != nullptr && api.create_event != nullptr &&
            api.record_event != nullptr && api.query_event != nullptr &&
            api.wait_event != nullptr && api.synchronize_event != nullptr &&
            api.destroy_event != nullptr) ||
        (!available && api.current_stream == nullptr &&
         api.pool_stream == nullptr && api.create_event == nullptr &&
         api.record_event == nullptr && api.query_event == nullptr &&
         api.wait_event == nullptr && api.synchronize_event == nullptr &&
         api.destroy_event == nullptr);
}

bool create_failures_are_reported() {
    FakeApi fake;
    fake.create_result = kCreateFailure;
    auto created = runtime::create_native_event(api_for(&fake), 2);
    if (!is_failure(created.status, transport::TransportStatusCode::kRuntimeFailure,
                    "create_event", kCreateFailure) || created.event != nullptr)
        return false;

    fake.create_result = 0;
    fake.native_event = nullptr;
    created = runtime::create_native_event(api_for(&fake), 2);
    return is_failure(created.status, transport::TransportStatusCode::kRuntimeFailure,
                      "create_event") && created.event == nullptr;
}

bool record_and_wait_enforce_device_identity() {
    FakeApi fake;
    auto created = runtime::create_native_event(api_for(&fake), 2);
    if (!created.status.ok() || created.event->device_index() != 2)
        return false;

    fake.record_result = kRecordFailure;
    auto status = created.event->record(stream_for(2));
    if (!is_failure(status, transport::TransportStatusCode::kRuntimeFailure,
                    "record_event", kRecordFailure))
        return false;
    fake.record_result = 0;
    status = created.event->record(stream_for(3));
    if (!is_failure(status, transport::TransportStatusCode::kInvalidArgument,
                    "record_event") || fake.record_calls != 1)
        return false;
    if (!created.event->record(stream_for(2)).ok() || fake.record_calls != 2)
        return false;

    status = created.event->wait(stream_for(3));
    if (!is_failure(status, transport::TransportStatusCode::kInvalidArgument,
                    "wait_event") || fake.wait_calls != 0)
        return false;
    fake.wait_result = kWaitFailure;
    status = created.event->wait(stream_for(2));
    if (!is_failure(status, transport::TransportStatusCode::kRuntimeFailure,
                    "wait_event", kWaitFailure))
        return false;
    fake.wait_result = 0;
    return created.event->wait(stream_for(2)).ok() && fake.wait_calls == 2;
}

bool finish_is_bounded_and_idempotent() {
    FakeApi fake;
    auto created = runtime::create_native_event(api_for(&fake), 2);
    if (!created.event->record(stream_for(2)).ok())
        return false;
    fake.synchronize_result = kTimeoutFailure;
    auto status = created.event->finish(7);
    if (!is_failure(status, transport::TransportStatusCode::kRuntimeFailure,
                    "synchronize_event", kTimeoutFailure) ||
        fake.last_timeout_ms != 7)
        return false;
    fake.synchronize_result = 0;
    if (!created.event->finish(9).ok() || fake.last_timeout_ms != 9)
        return false;
    return created.event->finish(11).ok() && fake.synchronize_calls == 2;
}

bool destroy_failure_retains_native_ownership_for_retry() {
    FakeApi fake;
    auto created = runtime::create_native_event(api_for(&fake), 2);
    fake.destroy_result = kDestroyFailure;
    auto status = created.event->destroy();
    if (!is_failure(status, transport::TransportStatusCode::kRuntimeFailure,
                    "destroy_event", kDestroyFailure) || fake.destroy_calls != 1)
        return false;
    fake.destroy_result = 0;
    if (!created.event->destroy().ok() || fake.destroy_calls != 2)
        return false;
    return created.event->destroy().ok() && fake.destroy_calls == 2;
}

bool destructor_attempts_nonthrowing_cleanup() {
    FakeApi fake;
    fake.destroy_throws = true;
    {
        auto created = runtime::create_native_event(api_for(&fake), 2);
        if (!created.status.ok())
            return false;
    }
    return fake.destroy_calls == 1;
}

}  // namespace

int main() {
    const bool passed = invalid_callback_tables_are_rejected() &&
        default_api_is_complete_or_unavailable() &&
        create_failures_are_reported() && record_and_wait_enforce_device_identity() &&
        finish_is_bounded_and_idempotent() &&
        destroy_failure_retains_native_ownership_for_retry() &&
        destructor_attempts_nonthrowing_cleanup();
    if (!passed) {
        std::cerr << "async runtime probe failed\n";
        return 1;
    }
    return 0;
}
