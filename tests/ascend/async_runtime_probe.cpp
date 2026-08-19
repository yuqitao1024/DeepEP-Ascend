#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// The host probe supplies the smallest complete tensor value needed to verify
// strong retention without importing Torch into this standalone C++ test.
#define DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR 1
namespace torch {
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<int> lifetime)
        : lifetime_(std::move(lifetime)) {}

private:
    std::shared_ptr<int> lifetime_;
};
}  // namespace torch

#include "csrc/backends/ascend/runtime/stream_event.hpp"
#include "csrc/backends/ascend/elastic/async_state.cpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;
namespace elastic = deep_ep::ascend::elastic;

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
constexpr int kTimeoutFailure = -1;
constexpr int kDestroyFailure = 15;
constexpr int kCurrentDeviceFailure = 16;
constexpr int kQueryFailure = 17;

struct FakeApi {
    int current_device_result = 0;
    int current_device_index = 2;
    int create_result = 0;
    int record_result = 0;
    int query_result = 0;
    int wait_result = 0;
    int synchronize_result = 0;
    int destroy_result = 0;
    bool destroy_throws = false;
    bool query_completed = true;
    std::atomic<int> current_device_calls{0};
    std::atomic<int> create_calls{0};
    std::atomic<int> record_calls{0};
    std::atomic<int> query_calls{0};
    std::atomic<int> wait_calls{0};
    std::atomic<int> synchronize_calls{0};
    std::atomic<int> destroy_calls{0};
    std::uint64_t last_timeout_ms = 0;
    void* native_event = reinterpret_cast<void*>(0x1000);
    std::mutex query_mutex;
    std::condition_variable query_cv;
    bool block_query = false;
    bool query_entered = false;
    bool release_query = false;
};

int current_device(void* user_data, int* device) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->current_device_calls;
    *device = fake->current_device_index;
    return fake->current_device_result;
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

int query_event(void* user_data, void*, bool* completed) {
    auto* fake = static_cast<FakeApi*>(user_data);
    ++fake->query_calls;
    {
        std::unique_lock<std::mutex> lock(fake->query_mutex);
        fake->query_entered = true;
        fake->query_cv.notify_all();
        if (fake->block_query) {
            fake->query_cv.wait(lock, [&] { return fake->release_query; });
        }
    }
    *completed = fake->query_completed;
    return fake->query_result;
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

bool create_checks_the_current_device_before_allocating_an_event() {
    FakeApi fake;
    fake.current_device_result = kCurrentDeviceFailure;
    auto created = runtime::create_native_event(api_for(&fake), 2);
    if (!is_failure(created.status, transport::TransportStatusCode::kRuntimeFailure,
                    "current_device", kCurrentDeviceFailure) ||
        fake.current_device_calls != 1 || fake.create_calls != 0)
        return false;

    fake.current_device_result = 0;
    fake.current_device_index = 3;
    fake.current_device_calls = 0;
    created = runtime::create_native_event(api_for(&fake), 2);
    if (!is_failure(created.status, transport::TransportStatusCode::kInvalidArgument,
                    "current_device") || fake.current_device_calls != 1 ||
        fake.create_calls != 0)
        return false;

    fake.current_device_index = 2;
    fake.current_device_calls = 0;
    created = runtime::create_native_event(api_for(&fake), 2);
    if (!created.status.ok() || fake.current_device_calls != 1 ||
        fake.create_calls != 1)
        return false;
    return created.event->destroy().ok();
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

    fake.query_result = kQueryFailure;
    auto status = created.event->finish(7);
    if (!is_failure(status, transport::TransportStatusCode::kRuntimeFailure,
                    "query_event", kQueryFailure) || fake.query_calls != 1)
        return false;

    fake.query_result = 0;
    fake.query_completed = false;
    status = created.event->finish(0);
    if (!is_failure(status, transport::TransportStatusCode::kRuntimeFailure,
                    "synchronize_event", kTimeoutFailure) ||
        fake.query_calls != 2)
        return false;

    fake.query_completed = true;
    if (!created.event->finish(9).ok() || fake.query_calls != 3)
        return false;
    return created.event->finish(11).ok() && fake.query_calls == 3 &&
        fake.synchronize_calls == 0;
}

bool timed_out_recorded_event_can_complete_and_retry_destruction() {
    FakeApi fake;
    auto created = runtime::create_native_event(api_for(&fake), 2);
    if (!created.status.ok() || !created.event->record(stream_for(2)).ok())
        return false;

    fake.query_completed = false;
    const auto timeout = created.event->finish(0);
    if (!is_failure(timeout, transport::TransportStatusCode::kRuntimeFailure,
                    "synchronize_event", kTimeoutFailure) ||
        fake.destroy_calls != 0)
        return false;

    const auto premature_destroy = created.event->destroy();
    if (!is_failure(premature_destroy,
                    transport::TransportStatusCode::kInvalidArgument,
                    "destroy_event") || fake.destroy_calls != 0)
        return false;

    fake.query_completed = true;
    return created.event->finish(0).ok() && created.event->destroy().ok() &&
        fake.destroy_calls == 1;
}

bool timed_out_recorded_event_is_not_force_destroyed_on_last_release() {
    FakeApi fake;
    {
        auto created = runtime::create_native_event(api_for(&fake), 2);
        if (!created.status.ok() || !created.event->record(stream_for(2)).ok())
            return false;
        fake.query_completed = false;
        const auto timeout = created.event->finish(0);
        if (!is_failure(timeout, transport::TransportStatusCode::kRuntimeFailure,
                        "synchronize_event", kTimeoutFailure))
            return false;
    }
    return fake.destroy_calls == 0;
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

#define ASYNC_CHECK(expression) do { if (!(expression)) return __LINE__; } while (0)

constexpr std::uint64_t kCompletionOffset = 128;
constexpr std::uint64_t kScratchStatusOffset = 64;
constexpr int kDiagnosticReadFailure = 31;
constexpr int kCompletionReadFailure = 32;
constexpr int kResourceDestroyFailure = 33;

struct CompletionTrace {
    std::atomic<int> diagnostic_reads{0};
    std::atomic<int> completion_reads{0};
    std::atomic<int> destroy_calls{0};
    std::atomic<int> destructor_calls{0};
};

class FakeCompletionResources final : public elastic::AsyncCompletionResources {
public:
    explicit FakeCompletionResources(std::shared_ptr<CompletionTrace> trace)
        : trace_(std::move(trace)) {}

    ~FakeCompletionResources() override { ++trace_->destructor_calls; }

    transport::TransportStatus read_diagnostic(
        elastic::BufferOperationKind kind, std::uint64_t scratch_status_offset,
        transport::DeviceTransportDiagnostic* output) override {
        ++trace_->diagnostic_reads;
        if (kind != expected_kind ||
            scratch_status_offset != kScratchStatusOffset || output == nullptr)
            return transport::TransportStatus::invalid(
                "read_diagnostic", "unexpected completion recipe");
        if (!diagnostic_status.ok())
            return diagnostic_status;
        *output = diagnostic;
        return transport::TransportStatus::success();
    }

    transport::TransportStatus read_completion(
        elastic::BufferOperationKind kind, std::uint64_t completion_offset,
        std::uint64_t* output) override {
        ++trace_->completion_reads;
        if (kind != expected_kind || completion_offset != kCompletionOffset ||
            output == nullptr)
            return transport::TransportStatus::invalid(
                "read_completion", "unexpected completion recipe");
        if (!completion_status.ok())
            return completion_status;
        *output = completion;
        return transport::TransportStatus::success();
    }

    transport::TransportStatus destroy() override {
        ++trace_->destroy_calls;
        if (destroy_failures_remaining > 0) {
            --destroy_failures_remaining;
            return transport::TransportStatus::runtime_failure(
                "destroy_resources", kResourceDestroyFailure,
                "injected resource destroy failure");
        }
        return transport::TransportStatus::success();
    }

    elastic::BufferOperationKind expected_kind =
        elastic::BufferOperationKind::kDispatch;
    transport::DeviceTransportDiagnostic diagnostic{};
    std::uint64_t completion = 0;
    transport::TransportStatus diagnostic_status =
        transport::TransportStatus::success();
    transport::TransportStatus completion_status =
        transport::TransportStatus::success();
    int destroy_failures_remaining = 0;

private:
    std::shared_ptr<CompletionTrace> trace_;
};

bool same_status(const transport::TransportStatus& lhs,
                 const transport::TransportStatus& rhs) {
    return lhs.code == rhs.code && lhs.operation == rhs.operation &&
           lhs.backend_code == rhs.backend_code && lhs.message == rhs.message;
}

struct PendingFixture {
    explicit PendingFixture(
        std::uint64_t last_generation = 0,
        std::uint64_t destroy_timeout_ms = 5,
        std::shared_ptr<elastic::AsyncStateHostTestControl> control = nullptr)
        : trace(std::make_shared<CompletionTrace>()),
          resources(std::make_shared<FakeCompletionResources>(trace)),
          state(std::make_shared<elastic::AsyncBufferState>(
              resources, destroy_timeout_ms, last_generation,
              std::move(control))) {}

    elastic::PendingOperationCreateResult launch(
        elastic::BufferOperationKind kind =
            elastic::BufferOperationKind::kDispatch,
        std::vector<std::optional<torch::Tensor>> retained_tensors = {},
        std::vector<elastic::EventDependency> predecessors = {},
        std::weak_ptr<runtime::NativeEventState>* weak_event = nullptr) {
        auto lease = state->coordinator().reserve(kind);
        if (!lease.valid() || !lease.activate())
            return {transport::TransportStatus::invalid(
                        "launch_fixture", "operation activation failed"),
                    nullptr};
        resources->expected_kind = kind;
        resources->diagnostic = {};
        resources->diagnostic.generation = lease.generation();
        resources->completion = lease.generation();
        auto created = runtime::create_native_event(api_for(&event_api), 2);
        if (!created.status.ok())
            return {created.status, nullptr};
        auto status = created.event->record(stream_for(2));
        if (!status.ok())
            return {status, nullptr};
        if (weak_event != nullptr)
            *weak_event = created.event;
        return state->publish(
            std::move(lease), std::move(created.event),
            {kind, resources->diagnostic.generation, kCompletionOffset,
             kScratchStatusOffset},
            std::move(retained_tensors), std::move(predecessors));
    }

    FakeApi event_api;
    std::shared_ptr<CompletionTrace> trace;
    std::shared_ptr<FakeCompletionResources> resources;
    std::shared_ptr<elastic::AsyncBufferState> state;
};

int check_synchronous_success() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok() && launched.operation != nullptr);
    const auto status = fixture.state->finish_pending();
    ASYNC_CHECK(status.ok());
    ASYNC_CHECK(launched.operation->state() == elastic::PendingState::kSucceeded);
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 0);
    ASYNC_CHECK(fixture.trace->destroy_calls == 0);
    ASYNC_CHECK(fixture.state->destroy().ok());
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    return 0;
}

int check_asynchronous_publish_stays_pending() {
    PendingFixture fixture;
    auto launched = fixture.launch(elastic::BufferOperationKind::kCombine);
    ASYNC_CHECK(launched.status.ok() && launched.operation != nullptr);
    ASYNC_CHECK(launched.operation->state() == elastic::PendingState::kLaunched);
    ASYNC_CHECK(launched.operation->generation() == 1);
    ASYNC_CHECK(fixture.state->coordinator().state() ==
                elastic::CoordinatorState::kActive);
    ASYNC_CHECK(fixture.event_api.query_calls == 0);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 0);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(launched.operation->finish(20).ok());
    return 0;
}

int check_two_thread_finish_is_exactly_once() {
    auto control = std::make_shared<elastic::AsyncStateHostTestControl>();
    PendingFixture fixture(0, 5, control);
    fixture.event_api.block_query = true;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    transport::TransportStatus first;
    transport::TransportStatus second;
    std::thread first_waiter([&] { first = launched.operation->finish(500); });
    {
        std::unique_lock<std::mutex> lock(fixture.event_api.query_mutex);
        fixture.event_api.query_cv.wait(lock, [&] {
            return fixture.event_api.query_entered;
        });
    }
    std::thread second_waiter([&] { second = launched.operation->finish(500); });
    bool second_entered_finalization_wait = false;
    {
        std::unique_lock<std::mutex> lock(control->mutex);
        second_entered_finalization_wait = control->cv.wait_for(
            lock, std::chrono::seconds(1), [&] {
                return control->finalization_waiter_entries >= 1;
            });
    }
    {
        std::lock_guard<std::mutex> lock(fixture.event_api.query_mutex);
        fixture.event_api.release_query = true;
    }
    fixture.event_api.query_cv.notify_all();
    first_waiter.join();
    second_waiter.join();
    ASYNC_CHECK(second_entered_finalization_wait);
    ASYNC_CHECK(control->finalization_waiter_entries == 1);
    ASYNC_CHECK(first.ok() && same_status(first, second));
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    return 0;
}

int check_repeated_finish_returns_the_stored_result() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    const auto first = launched.operation->finish(20);
    fixture.resources->diagnostic.error =
        transport::DeviceTransportError::kCompletionFailure;
    fixture.resources->completion = 999;
    const auto second = launched.operation->finish(20);
    ASYNC_CHECK(first.ok() && same_status(first, second));
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    return 0;
}

int check_second_reservation_defers_poison_until_finish() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    auto busy = fixture.state->coordinator().reserve(
        elastic::BufferOperationKind::kCombine);
    ASYNC_CHECK(busy.status() == elastic::LeaseStatus::kBusy);
    ASYNC_CHECK(fixture.state->coordinator().state() ==
                elastic::CoordinatorState::kActive);
    ASYNC_CHECK(launched.operation->finish(20).ok());
    ASYNC_CHECK(fixture.state->coordinator().state() ==
                elastic::CoordinatorState::kPoisoned);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 0);
    return 0;
}

int check_event_timeout_is_stable_and_teardown_is_retryable() {
    PendingFixture fixture;
    fixture.event_api.query_completed = false;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    const auto first = launched.operation->finish(0);
    ASYNC_CHECK(is_failure(
        first, transport::TransportStatusCode::kRuntimeFailure,
        "synchronize_event", kTimeoutFailure));
    ASYNC_CHECK(launched.operation->state() == elastic::PendingState::kFailed);
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 0);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 0);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(first, launched.operation->finish(20)));
    ASYNC_CHECK(fixture.event_api.query_calls == 1);

    fixture.event_api.query_completed = true;
    const auto destroy_status = fixture.state->destroy();
    ASYNC_CHECK(same_status(first, destroy_status));
    ASYNC_CHECK(fixture.event_api.query_calls == 2);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    return 0;
}

int check_diagnostic_read_failure_abandons_without_completion_read() {
    PendingFixture fixture;
    fixture.resources->diagnostic_status =
        transport::TransportStatus::runtime_failure(
            "read_diagnostic", kDiagnosticReadFailure,
            "injected diagnostic read failure");
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    const auto status = launched.operation->finish(20);
    ASYNC_CHECK(is_failure(
        status, transport::TransportStatusCode::kRuntimeFailure,
        "read_diagnostic", kDiagnosticReadFailure));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(status, launched.operation->finish(20)));
    const auto terminal = fixture.state->terminal_failure();
    ASYNC_CHECK(terminal.has_value() && same_status(status, *terminal));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    return 0;
}

int check_diagnostic_abi_mismatch_abandons() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    fixture.resources->diagnostic.abi_version = 0;
    const auto status = launched.operation->finish(20);
    ASYNC_CHECK(!status.ok());
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(status, launched.operation->finish(20)));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    return 0;
}

int check_diagnostic_error_abandons() {
    PendingFixture fixture;
    auto launched = fixture.launch(elastic::BufferOperationKind::kBarrier);
    ASYNC_CHECK(launched.status.ok());
    fixture.resources->diagnostic.error =
        transport::DeviceTransportError::kCompletionTimeout;
    fixture.resources->diagnostic.command_index = 7;
    fixture.resources->diagnostic.opcode =
        transport::TransportCommandOpcode::kBarrier;
    fixture.resources->diagnostic.peer = 3;
    fixture.resources->diagnostic.world_peer = 5;
    fixture.resources->diagnostic.team = transport::TransportTeam::kScaleOut;
    fixture.resources->diagnostic.channel = 11;
    fixture.resources->diagnostic.backend_status = 37;
    fixture.resources->diagnostic.reserved = 0x1234;
    fixture.resources->diagnostic.generation = 41;
    const auto status = launched.operation->finish(20);
    ASYNC_CHECK(is_failure(
        status, transport::TransportStatusCode::kRuntimeFailure,
        "barrier", 37));
    for (const auto* expected : {
             "device diagnostic reported failure",
             "error=completion_timeout",
             "command_index=7",
             "opcode=6",
             "peer=3",
             "world_peer=5",
             "team=2",
             "channel=11",
             "backend_status=37",
             "reserved=4660",
             "generation=41"}) {
        ASYNC_CHECK(status.message.find(expected) != std::string::npos);
    }
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(status, launched.operation->finish(20)));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    return 0;
}

int check_diagnostic_generation_mismatch_abandons() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    fixture.resources->diagnostic.generation = 999;
    const auto status = launched.operation->finish(20);
    ASYNC_CHECK(!status.ok());
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(status, launched.operation->finish(20)));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    return 0;
}

int check_completion_read_failure_abandons() {
    PendingFixture fixture;
    fixture.resources->completion_status =
        transport::TransportStatus::runtime_failure(
            "read_completion", kCompletionReadFailure,
            "injected completion read failure");
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    const auto status = launched.operation->finish(20);
    ASYNC_CHECK(is_failure(
        status, transport::TransportStatusCode::kRuntimeFailure,
        "read_completion", kCompletionReadFailure));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(status, launched.operation->finish(20)));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    return 0;
}

int check_completion_mismatch_abandons() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    fixture.resources->completion = 999;
    const auto status = launched.operation->finish(20);
    ASYNC_CHECK(!status.ok());
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(status, launched.operation->finish(20)));
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    return 0;
}

int check_event_destroy_failure_is_stable_and_retryable() {
    PendingFixture fixture;
    fixture.event_api.destroy_result = kDestroyFailure;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    const auto first = launched.operation->finish(20);
    ASYNC_CHECK(is_failure(
        first, transport::TransportStatusCode::kRuntimeFailure,
        "destroy_event", kDestroyFailure));
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(same_status(first, launched.operation->finish(20)));

    fixture.event_api.destroy_result = 0;
    const auto destroy_status = fixture.state->destroy();
    ASYNC_CHECK(same_status(first, destroy_status));
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 2);
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    return 0;
}

int check_post_activation_publish_failure_retains_event_for_destroy() {
    PendingFixture fixture;
    auto lease = fixture.state->coordinator().reserve(
        elastic::BufferOperationKind::kDispatch);
    ASYNC_CHECK(lease.valid() && lease.activate());
    auto created = runtime::create_native_event(api_for(&fixture.event_api), 2);
    ASYNC_CHECK(created.status.ok());
    ASYNC_CHECK(created.event->record(stream_for(2)).ok());
    std::weak_ptr<runtime::NativeEventState> event = created.event;
    const auto published = fixture.state->publish(
        std::move(lease), std::move(created.event),
        {elastic::BufferOperationKind::kDispatch, 999, kCompletionOffset,
         kScratchStatusOffset},
        {}, {});
    ASYNC_CHECK(!published.status.ok() && published.operation == nullptr);
    ASYNC_CHECK(fixture.state->coordinator().state() ==
                elastic::CoordinatorState::kPoisoned);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(fixture.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(!event.expired());
    ASYNC_CHECK(fixture.event_api.query_calls == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 0);
    ASYNC_CHECK(same_status(published.status, *fixture.state->terminal_failure()));

    const auto destroy_status = fixture.state->destroy();
    ASYNC_CHECK(same_status(published.status, destroy_status));
    ASYNC_CHECK(event.expired());
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 0);
    ASYNC_CHECK(fixture.trace->completion_reads == 0);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    return 0;
}

int check_destroy_race_retains_publish_owners_until_event_is_safe() {
    auto trace = std::make_shared<CompletionTrace>();
    auto resources = std::make_shared<FakeCompletionResources>(trace);
    auto control = std::make_shared<elastic::AsyncStateHostTestControl>();
    auto state = std::make_shared<elastic::AsyncBufferState>(
        resources, 100, 0, control);
    FakeApi event_api;
    auto lease = state->coordinator().reserve(
        elastic::BufferOperationKind::kDispatch);
    ASYNC_CHECK(lease.valid() && lease.activate());
    resources->diagnostic.generation = lease.generation();
    resources->completion = lease.generation();
    auto created = runtime::create_native_event(api_for(&event_api), 2);
    ASYNC_CHECK(created.status.ok());
    ASYNC_CHECK(created.event->record(stream_for(2)).ok());
    std::weak_ptr<runtime::NativeEventState> event_lifetime = created.event;

    FakeApi predecessor_api;
    auto predecessor = runtime::create_native_event(api_for(&predecessor_api), 2);
    ASYNC_CHECK(predecessor.status.ok());
    ASYNC_CHECK(predecessor.event->record(stream_for(2)).ok());
    std::weak_ptr<runtime::NativeEventState> predecessor_lifetime =
        predecessor.event;
    auto tensor_lifetime = std::make_shared<int>(19);
    std::weak_ptr<int> retained_tensor_lifetime = tensor_lifetime;
    std::vector<std::optional<torch::Tensor>> tensors;
    tensors.emplace_back(torch::Tensor(tensor_lifetime));
    std::vector<elastic::EventDependency> predecessors;
    predecessors.push_back({predecessor.event, nullptr});

    {
        std::lock_guard<std::mutex> lock(control->mutex);
        control->pause_destroy_after_snapshot = true;
    }
    transport::TransportStatus destroy_status;
    std::thread destroyer([&] { destroy_status = state->destroy(); });
    bool destroy_paused = false;
    {
        std::unique_lock<std::mutex> lock(control->mutex);
        destroy_paused = control->cv.wait_for(
            lock, std::chrono::seconds(1), [&] {
                return control->destroy_paused_after_snapshot;
            });
    }

    const auto published = state->publish(
        std::move(lease), std::move(created.event),
        {elastic::BufferOperationKind::kDispatch, 1, kCompletionOffset,
         kScratchStatusOffset},
        std::move(tensors), std::move(predecessors));
    tensor_lifetime.reset();
    predecessor.event.reset();
    const bool owners_retained = !event_lifetime.expired() &&
        !retained_tensor_lifetime.expired() && !predecessor_lifetime.expired();
    {
        std::lock_guard<std::mutex> lock(control->mutex);
        control->resume_destroy = true;
    }
    control->cv.notify_all();
    destroyer.join();

    ASYNC_CHECK(destroy_paused);
    ASYNC_CHECK(!published.status.ok() && published.operation == nullptr);
    ASYNC_CHECK(owners_retained);
    ASYNC_CHECK(same_status(published.status, destroy_status));
    ASYNC_CHECK(event_lifetime.expired());
    ASYNC_CHECK(retained_tensor_lifetime.expired());
    ASYNC_CHECK(predecessor_lifetime.expired());
    ASYNC_CHECK(event_api.query_calls == 1);
    ASYNC_CHECK(event_api.destroy_calls == 1);
    ASYNC_CHECK(trace->diagnostic_reads == 0);
    ASYNC_CHECK(trace->completion_reads == 0);
    ASYNC_CHECK(trace->destroy_calls == 1);
    return 0;
}

int check_foreign_lease_is_rejected_and_retained_by_target_state() {
    PendingFixture source;
    PendingFixture target;
    auto lease = source.state->coordinator().reserve(
        elastic::BufferOperationKind::kDispatch);
    ASYNC_CHECK(lease.valid() && lease.activate());
    auto created = runtime::create_native_event(api_for(&source.event_api), 2);
    ASYNC_CHECK(created.status.ok());
    ASYNC_CHECK(created.event->record(stream_for(2)).ok());
    std::weak_ptr<runtime::NativeEventState> event_lifetime = created.event;

    const auto published = target.state->publish(
        std::move(lease), std::move(created.event),
        {elastic::BufferOperationKind::kDispatch, 1, kCompletionOffset,
         kScratchStatusOffset},
        {}, {});

    ASYNC_CHECK(!published.status.ok() && published.operation == nullptr);
    ASYNC_CHECK(source.state->coordinator().state() ==
                elastic::CoordinatorState::kPoisoned);
    ASYNC_CHECK(source.state->coordinator().completed_operation_count() == 0);
    ASYNC_CHECK(source.state->coordinator().abandoned_operation_count() == 1);
    ASYNC_CHECK(target.state->coordinator().state() ==
                elastic::CoordinatorState::kIdle);
    ASYNC_CHECK(!event_lifetime.expired());
    ASYNC_CHECK(same_status(
        published.status, *target.state->terminal_failure()));

    const auto destroy_status = target.state->destroy();
    ASYNC_CHECK(same_status(published.status, destroy_status));
    ASYNC_CHECK(event_lifetime.expired());
    ASYNC_CHECK(source.event_api.query_calls == 1);
    ASYNC_CHECK(source.event_api.destroy_calls == 1);
    ASYNC_CHECK(target.trace->diagnostic_reads == 0);
    ASYNC_CHECK(target.trace->completion_reads == 0);
    ASYNC_CHECK(target.trace->destroy_calls == 1);
    ASYNC_CHECK(source.state->destroy().ok());
    return 0;
}

int check_dropped_final_event_is_drained_by_destroy() {
    PendingFixture fixture;
    std::weak_ptr<runtime::NativeEventState> event;
    auto launched = fixture.launch(
        elastic::BufferOperationKind::kDispatch, {}, {}, &event);
    ASYNC_CHECK(launched.status.ok() && !event.expired());
    std::weak_ptr<elastic::PendingOperation> pending = launched.operation;
    launched.operation.reset();
    ASYNC_CHECK(!pending.expired());
    fixture.state.reset();
    ASYNC_CHECK(event.expired());
    ASYNC_CHECK(pending.expired());
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    return 0;
}

int check_destroy_drains_pending_before_resources() {
    PendingFixture fixture;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    ASYNC_CHECK(fixture.state->destroy().ok());
    ASYNC_CHECK(launched.operation->state() == elastic::PendingState::kSucceeded);
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    ASYNC_CHECK(fixture.state->coordinator().state() ==
                elastic::CoordinatorState::kDestroyed);
    return 0;
}

int check_failed_resource_destroy_retries_without_refinalizing() {
    PendingFixture fixture;
    fixture.resources->destroy_failures_remaining = 1;
    auto launched = fixture.launch();
    ASYNC_CHECK(launched.status.ok());
    const auto first = fixture.state->destroy();
    ASYNC_CHECK(is_failure(
        first, transport::TransportStatusCode::kRuntimeFailure,
        "destroy_resources", kResourceDestroyFailure));
    ASYNC_CHECK(fixture.trace->destroy_calls == 1);
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    ASYNC_CHECK(fixture.state->destroy().ok());
    ASYNC_CHECK(fixture.trace->destroy_calls == 2);
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.event_api.destroy_calls == 1);
    return 0;
}

int check_pending_outlives_buffer_wrapper_without_raw_dependency() {
    auto trace = std::make_shared<CompletionTrace>();
    auto resources = std::make_shared<FakeCompletionResources>(trace);
    std::weak_ptr<FakeCompletionResources> resources_lifetime = resources;
    auto state = std::make_shared<elastic::AsyncBufferState>(resources, 5, 0);
    FakeApi event_api;
    auto lease = state->coordinator().reserve(
        elastic::BufferOperationKind::kDispatch);
    ASYNC_CHECK(lease.valid() && lease.activate());
    resources->diagnostic.generation = lease.generation();
    resources->completion = lease.generation();
    auto created = runtime::create_native_event(api_for(&event_api), 2);
    ASYNC_CHECK(created.status.ok());
    ASYNC_CHECK(created.event->record(stream_for(2)).ok());
    auto launched = state->publish(
        std::move(lease), std::move(created.event),
        {elastic::BufferOperationKind::kDispatch, 1, kCompletionOffset,
         kScratchStatusOffset},
        {}, {});
    ASYNC_CHECK(launched.status.ok());
    resources.reset();
    state.reset();
    ASYNC_CHECK(!resources_lifetime.expired());
    ASYNC_CHECK(launched.operation->finish(20).ok());
    ASYNC_CHECK(event_api.query_calls == 1);
    ASYNC_CHECK(trace->diagnostic_reads == 1);
    ASYNC_CHECK(trace->completion_reads == 1);
    ASYNC_CHECK(event_api.destroy_calls == 1);
    launched.operation.reset();
    ASYNC_CHECK(resources_lifetime.expired());
    ASYNC_CHECK(trace->destructor_calls == 1);
    return 0;
}

int check_retained_tensors_and_predecessors_release_after_safe_teardown() {
    PendingFixture predecessor_fixture;
    auto predecessor = predecessor_fixture.launch();
    ASYNC_CHECK(predecessor.status.ok());
    ASYNC_CHECK(predecessor.operation->finish(20).ok());
    auto predecessor_operation = predecessor.operation;
    predecessor_fixture.state.reset();
    predecessor.operation.reset();
    std::weak_ptr<elastic::PendingOperation> predecessor_lifetime =
        predecessor_operation;

    FakeApi predecessor_event_api;
    auto predecessor_event = runtime::create_native_event(
        api_for(&predecessor_event_api), 2);
    ASYNC_CHECK(predecessor_event.status.ok());
    ASYNC_CHECK(predecessor_event.event->record(stream_for(2)).ok());
    std::weak_ptr<runtime::NativeEventState> event_lifetime =
        predecessor_event.event;
    auto tensor_lifetime = std::make_shared<int>(7);
    std::weak_ptr<int> retained_tensor_lifetime = tensor_lifetime;

    std::vector<std::optional<torch::Tensor>> tensors;
    tensors.emplace_back(torch::Tensor(tensor_lifetime));
    std::vector<elastic::EventDependency> dependencies;
    dependencies.push_back(
        {predecessor_event.event, predecessor_operation});
    PendingFixture fixture;
    auto launched = fixture.launch(
        elastic::BufferOperationKind::kCombine, std::move(tensors),
        std::move(dependencies));
    ASYNC_CHECK(launched.status.ok());
    tensor_lifetime.reset();
    predecessor_event.event.reset();
    predecessor_operation.reset();
    ASYNC_CHECK(!retained_tensor_lifetime.expired());
    ASYNC_CHECK(!event_lifetime.expired());
    ASYNC_CHECK(!predecessor_lifetime.expired());
    ASYNC_CHECK(launched.operation->finish(20).ok());
    ASYNC_CHECK(retained_tensor_lifetime.expired());
    ASYNC_CHECK(event_lifetime.expired());
    ASYNC_CHECK(predecessor_lifetime.expired());
    ASYNC_CHECK(predecessor_event_api.destroy_calls == 1);
    return 0;
}

int check_generation_exhaustion_never_publishes_a_second_operation() {
    PendingFixture fixture(std::numeric_limits<std::uint64_t>::max() - 1);
    auto last = fixture.launch();
    ASYNC_CHECK(last.status.ok());
    ASYNC_CHECK(last.operation->generation() ==
                std::numeric_limits<std::uint64_t>::max());
    ASYNC_CHECK(last.operation->finish(20).ok());
    auto exhausted = fixture.state->coordinator().reserve(
        elastic::BufferOperationKind::kCombine);
    ASYNC_CHECK(exhausted.valid());
    ASYNC_CHECK(!exhausted.activate());
    ASYNC_CHECK(exhausted.status() == elastic::LeaseStatus::kGenerationExhausted);
    ASYNC_CHECK(fixture.state->coordinator().state() ==
                elastic::CoordinatorState::kPoisoned);
    ASYNC_CHECK(fixture.event_api.query_calls == 1);
    ASYNC_CHECK(fixture.trace->diagnostic_reads == 1);
    ASYNC_CHECK(fixture.trace->completion_reads == 1);
    ASYNC_CHECK(fixture.state->coordinator().completed_operation_count() == 1);
    return 0;
}

int run_async_state_contract() {
    if (const int status = check_synchronous_success()) return status;
    if (const int status = check_asynchronous_publish_stays_pending()) return status;
    if (const int status = check_two_thread_finish_is_exactly_once()) return status;
    if (const int status = check_repeated_finish_returns_the_stored_result())
        return status;
    if (const int status = check_second_reservation_defers_poison_until_finish())
        return status;
    if (const int status = check_event_timeout_is_stable_and_teardown_is_retryable())
        return status;
    if (const int status = check_diagnostic_read_failure_abandons_without_completion_read())
        return status;
    if (const int status = check_diagnostic_abi_mismatch_abandons()) return status;
    if (const int status = check_diagnostic_error_abandons()) return status;
    if (const int status = check_diagnostic_generation_mismatch_abandons())
        return status;
    if (const int status = check_completion_read_failure_abandons()) return status;
    if (const int status = check_completion_mismatch_abandons()) return status;
    if (const int status = check_event_destroy_failure_is_stable_and_retryable())
        return status;
    if (const int status = check_post_activation_publish_failure_retains_event_for_destroy())
        return status;
    if (const int status = check_destroy_race_retains_publish_owners_until_event_is_safe())
        return status;
    if (const int status = check_foreign_lease_is_rejected_and_retained_by_target_state())
        return status;
    if (const int status = check_dropped_final_event_is_drained_by_destroy())
        return status;
    if (const int status = check_destroy_drains_pending_before_resources())
        return status;
    if (const int status = check_failed_resource_destroy_retries_without_refinalizing())
        return status;
    if (const int status = check_pending_outlives_buffer_wrapper_without_raw_dependency())
        return status;
    if (const int status = check_retained_tensors_and_predecessors_release_after_safe_teardown())
        return status;
    return check_generation_exhaustion_never_publishes_a_second_operation();
}

}  // namespace

int main() {
    const bool passed = invalid_callback_tables_are_rejected() &&
        default_api_is_complete_or_unavailable() &&
        create_failures_are_reported() && record_and_wait_enforce_device_identity() &&
        create_checks_the_current_device_before_allocating_an_event() &&
        finish_is_bounded_and_idempotent() &&
        timed_out_recorded_event_can_complete_and_retry_destruction() &&
        timed_out_recorded_event_is_not_force_destroyed_on_last_release() &&
        destroy_failure_retains_native_ownership_for_retry() &&
        destructor_attempts_nonthrowing_cleanup();
    if (!passed) {
        std::cerr << "async runtime probe failed\n";
        return 1;
    }
    const int async_state_status = run_async_state_contract();
    if (async_state_status != 0) {
        std::cerr << "async state probe failed at line "
                  << async_state_status << '\n';
    }
    return async_state_status;
}
