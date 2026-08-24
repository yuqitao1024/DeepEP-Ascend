#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "csrc/backends/ascend/elastic_buffer.hpp"

namespace elastic = deep_ep::ascend::elastic;
namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;

namespace {

int failures = 0;

struct Trace;
struct StreamToken {
    Trace* trace = nullptr;
    bool communication = false;
};

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": "                \
                      << #expression << '\n';                                 \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

struct Trace {
    std::mutex mutex;
    std::condition_variable cv;
    bool block_sync = false;
    bool sync_entered = false;
    bool release_sync = false;
    std::atomic<int> sync_failures{0};
    std::atomic<int> barrier_launches{0};
    std::atomic<int> combine_launches{0};
    std::atomic<int> current_device_calls{0};
    std::atomic<int> current_stream_calls{0};
    std::atomic<int> pool_stream_calls{0};
    std::atomic<int> destroy_event_calls{0};
    std::atomic<int> create_event_calls{0};
    std::atomic<int> record_event_calls{0};
    std::atomic<int> synchronize_stream_calls{0};
    std::atomic<int> runtime_copy_to_host_calls{0};
    std::atomic<int> transport_copy_from_device_calls{0};
    std::atomic<int> runtime_free_calls{0};
    std::atomic<int> host_free_calls{0};
    std::atomic<int> deregister_calls{0};
    std::atomic<int> destroy_team_calls{0};
    int fail_create_event_on = 0;
    int fail_record_event_on = 0;
    bool fail_barrier_diagnostic = false;
    bool current_stream_completion_record = false;
    bool event_ready = true;
    std::vector<void*> runtime_allocations;
    std::vector<void*> transport_allocations;
    void* diagnostic = nullptr;
    void* registered_window = nullptr;
    std::vector<std::string> order;
    StreamToken compute_stream;
    StreamToken comm_stream;
};

Trace& self(void* data) { return *static_cast<Trace*>(data); }

int runtime_allocate(void* data, std::uint64_t bytes, void** pointer) {
    auto& trace = self(data);
    *pointer = std::malloc(static_cast<std::size_t>(bytes));
    if (*pointer == nullptr)
        return 11;
    trace.runtime_allocations.push_back(*pointer);
    return 0;
}

int runtime_zero(void*, void* pointer, std::uint64_t bytes) {
    std::memset(pointer, 0, static_cast<std::size_t>(bytes));
    return 0;
}

int runtime_free(void* data, void* pointer) {
    ++self(data).runtime_free_calls;
    std::free(pointer);
    return 0;
}

int stream_current_device(void* data, int* device) {
    ++self(data).current_device_calls;
    *device = 0;
    return 0;
}

int stream_current_stream(void* data, runtime::StreamIdentity* stream) {
    auto& trace = self(data);
    ++trace.current_stream_calls;
    trace.order.emplace_back("capture compute dependency");
    *stream = {&trace.compute_stream, 7, 0, 20};
    return 0;
}

int stream_pool_stream(
    void* data, int device, bool high_priority,
    runtime::StreamIdentity* stream) {
    auto& trace = self(data);
    ++trace.pool_stream_calls;
    if (device != 0 || !high_priority)
        return 18;
    *stream = {&trace.comm_stream, 11, device, 20};
    return 0;
}

int stream_create_event(void* data, void** event) {
    auto& trace = self(data);
    const int call = ++trace.create_event_calls;
    if (call == trace.fail_create_event_on)
        return 22;
    *event = new int(1);
    trace.order.emplace_back("create event");
    return 0;
}
int stream_record_event(void* data, void*, void* stream) {
    const auto* token = static_cast<StreamToken*>(stream);
    auto& trace = self(data);
    const int call = ++trace.record_event_calls;
    trace.order.emplace_back(
        token->communication || trace.current_stream_completion_record ?
        "record completion event" : "record compute dependency");
    if (call == trace.fail_record_event_on)
        return 23;
    return 0;
}
int stream_query_event(void* data, void*, bool* complete) {
    auto& trace = self(data);
    {
        std::unique_lock<std::mutex> lock(trace.mutex);
        if (trace.block_sync) {
            trace.sync_entered = true;
            trace.cv.notify_all();
            trace.cv.wait(lock, [&] { return trace.release_sync; });
            trace.block_sync = false;
            trace.sync_entered = false;
            trace.release_sync = false;
        }
    }
    *complete = trace.event_ready;
    trace.order.emplace_back(trace.event_ready ?
        "finish completion event" : "completion event pending");
    return 0;
}
int stream_wait_event(void* data, void* stream, void*) {
    const auto* token = static_cast<StreamToken*>(stream);
    if (!token->communication)
        return 19;
    self(data).order.emplace_back("comm waits dependency/previous event");
    return 0;
}
int stream_synchronize_event(void*, void*, std::uint64_t) { return 0; }
int stream_destroy_event(void* data, void* event) {
    ++self(data).destroy_event_calls;
    delete static_cast<int*>(event);
    return 0;
}

int runtime_synchronize_stream(void* data, void*) {
    auto& trace = self(data);
    ++trace.synchronize_stream_calls;
    {
        std::unique_lock<std::mutex> lock(trace.mutex);
        if (trace.block_sync) {
            trace.sync_entered = true;
            trace.cv.notify_all();
            trace.cv.wait(lock, [&] { return trace.release_sync; });
            trace.block_sync = false;
            trace.sync_entered = false;
            trace.release_sync = false;
        }
    }
    int remaining = trace.sync_failures.load();
    while (remaining > 0 &&
           !trace.sync_failures.compare_exchange_weak(remaining, remaining - 1)) {}
    return remaining > 0 ? 12 : 0;
}

int runtime_synchronize_device(void*) { return 0; }

int runtime_copy_from_host(
    void*, void* destination, const void* source, std::uint64_t bytes) {
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return 0;
}

int runtime_copy_to_host(
    void* data, void* destination, const void* source, std::uint64_t bytes) {
    ++self(data).runtime_copy_to_host_calls;
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return 0;
}

runtime::CannRuntimeApi runtime_api(Trace& trace) {
    return {&trace, runtime_allocate, runtime_zero, runtime_free,
            runtime_synchronize_stream, runtime_synchronize_device,
            runtime_copy_from_host, runtime_copy_to_host};
}

runtime::StreamEventApi stream_api(Trace& trace) {
    return {&trace, stream_current_device, stream_current_stream,
            stream_pool_stream, stream_create_event, stream_record_event,
            stream_query_event, stream_wait_event, stream_synchronize_event,
            stream_destroy_event};
}

int get_rank(void*, std::int64_t, std::uint32_t* rank) {
    *rank = 0;
    return 0;
}

int get_size(void*, std::int64_t, std::uint32_t* size) {
    *size = 2;
    return 0;
}

int create_team(
    void* data, std::int64_t, std::uint32_t, std::uint32_t,
    const std::uint32_t*, std::uint32_t, std::uint32_t,
    std::uintptr_t* team) {
    *team = reinterpret_cast<std::uintptr_t>(data) ^ 0x1111U;
    return 0;
}

int register_window(
    void* data, std::int64_t, std::uintptr_t, void* base, std::uint64_t,
    std::uintptr_t* window) {
    auto& trace = self(data);
    trace.registered_window = base;
    *window = reinterpret_cast<std::uintptr_t>(data) ^ 0x2222U;
    return 0;
}

int create_channels(
    void*, std::int64_t, std::uintptr_t, std::uint32_t) {
    return 0;
}

int host_allocate(void* data, std::uint64_t bytes, void** pointer) {
    auto& trace = self(data);
    *pointer = std::malloc(static_cast<std::size_t>(bytes));
    if (*pointer == nullptr)
        return 13;
    trace.transport_allocations.push_back(*pointer);
    if (trace.transport_allocations.size() == 4)
        trace.diagnostic = *pointer;
    return 0;
}

int host_zero(void*, void* pointer, std::uint64_t bytes) {
    std::memset(pointer, 0, static_cast<std::size_t>(bytes));
    return 0;
}

int host_copy_to_device(
    void*, void* destination, const void* source, std::uint64_t bytes) {
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return 0;
}

int host_copy_from_device(
    void* data, void* destination, const void* source, std::uint64_t bytes) {
    ++self(data).transport_copy_from_device_calls;
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return 0;
}

int host_free(void* data, void* pointer) {
    ++self(data).host_free_calls;
    std::free(pointer);
    return 0;
}

int deregister_window(void* data, std::uintptr_t, std::uintptr_t) {
    ++self(data).deregister_calls;
    return 0;
}

int destroy_team(void* data, std::uintptr_t) {
    ++self(data).destroy_team_calls;
    return 0;
}

transport::CannHostApi host_api(Trace& trace) {
    return {&trace, get_rank, get_size, create_team, register_window,
            create_channels, host_allocate, host_zero, host_copy_to_device,
            host_copy_from_device, host_free, deregister_window, destroy_team};
}

std::unique_ptr<runtime::CannRuntimeResources> make_resources(Trace& trace) {
    trace.compute_stream = {&trace, false};
    trace.comm_stream = {&trace, true};
    auto resources = std::make_unique<runtime::CannRuntimeResources>();
    transport::TransportConfig config{};
    config.rank = 0;
    config.world_size = 2;
    config.communicator_handle =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(&trace));
    config.device_buffer_bytes = 2 * 1024 * 1024;
    config.requested_channels = 1;
    const auto status = resources->initialize(
        config, 4096, runtime_api(trace), host_api(trace), stream_api(trace));
    CHECK(trace.current_device_calls == 1);
    CHECK(trace.pool_stream_calls == 1);
    CHECK(trace.destroy_event_calls == 0);
    if (status.ok()) {
        CHECK(resources->comm_stream().raw == &trace.comm_stream);
        CHECK(resources->comm_stream().stream_id == 11);
        CHECK(resources->comm_stream().device_index == 0);
    }
    return status.ok() ? std::move(resources) : nullptr;
}

struct ResourceIdentity {
    void* window = nullptr;
    void* workspace = nullptr;
    void* transport_object = nullptr;
    void* queue = nullptr;
    void* diagnostic = nullptr;
};

ResourceIdentity identity(
    const runtime::CannRuntimeResources& resources, const Trace& trace) {
    return {resources.window_base(), resources.workspace(),
            resources.transport(), trace.transport_allocations.at(1),
            trace.diagnostic};
}

using Buffer = deep_ep::ascend::ElasticBuffer;
using Event = deep_ep::ascend::EventHandle;
using Tensor = torch::Tensor;

std::unique_ptr<Buffer> make_buffer(
    Trace& trace, ResourceIdentity* resource_identity,
    std::uint64_t completion_timeout_ms = 5000,
    std::shared_ptr<Buffer::TestingLifecycleControl> lifecycle_control = nullptr) {
    auto resources = make_resources(trace);
    if (resources == nullptr)
        return nullptr;
    *resource_identity = identity(*resources, trace);
    return Buffer::make_testing_buffer(
        0, std::move(resources), 2 * 1024 * 1024, 1,
        true, 7, 2, 0, false, completion_timeout_ms,
        std::move(lifecycle_control));
}

std::unique_ptr<Buffer> make_combine_buffer(
    Trace& trace, ResourceIdentity* resource_identity,
    std::uint64_t completion_timeout_ms = 5000) {
    auto resources = make_resources(trace);
    if (resources == nullptr)
        return nullptr;
    *resource_identity = identity(*resources, trace);
    return Buffer::make_testing_buffer(
        0, std::move(resources), 2 * 1024 * 1024, 1,
        true, 7, 2, 1, false, completion_timeout_ms);
}

struct CombineInputs {
    Tensor x = torch::empty(
        {2, 8}, torch::TensorOptions().dtype(torch::kBFloat16));
    Tensor source = torch::empty(
        {2, 4}, torch::TensorOptions().dtype(torch::kInt));
    Tensor indices = torch::empty(
        {1, 2}, torch::TensorOptions().dtype(torch::kLong));
    Tensor prefix = torch::empty(
        {2}, torch::TensorOptions().dtype(torch::kInt));
    Tensor descriptor = torch::empty(
        {static_cast<std::int64_t>(
            sizeof(elastic::DispatchHandleDescriptor))},
        torch::TensorOptions().dtype(torch::kByte));

    CombineInputs() {
        indices.data_ptr<std::int64_t>()[0] = 0;
        indices.data_ptr<std::int64_t>()[1] = 1;
        prefix.data_ptr<std::int32_t>()[0] = 1;
        prefix.data_ptr<std::int32_t>()[1] = 2;
        const std::array<std::int32_t, 8> metadata{
            0, 0, -1, -1,
            4, 3, -1, -1};
        std::memcpy(source.data_ptr(), metadata.data(), sizeof(metadata));
        const auto value = elastic::make_attested_dispatch_handle_descriptor(
            7, {0, 2, 0, 2, 0, 1}, 1, 1, 8, 2, 2, 4, 4, 0);
        std::memcpy(descriptor.data_ptr(), &value, sizeof(value));
    }
};

auto launch_async_combine(Buffer& buffer, CombineInputs& inputs) {
    const std::optional<Tensor> no_tensor;
    const std::optional<Event> no_event;
    return buffer.combine(
        inputs.x, no_tensor, no_tensor, no_tensor, inputs.source,
        inputs.indices, inputs.prefix, inputs.descriptor, no_tensor,
        2, 4, 1, 0, no_event, no_event, true, false, false);
}

bool error_contains(
    const std::function<void()>& operation, const char* expected) {
    try {
        operation();
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

std::array<void*, 5> pointers(const ResourceIdentity& identity) {
    return {identity.window, identity.workspace, identity.transport_object,
            identity.queue, identity.diagnostic};
}

std::array<int, 5> resource_callback_snapshot(const Trace& trace) {
    return {trace.current_device_calls.load(),
            trace.current_stream_calls.load(),
            trace.synchronize_stream_calls.load(),
            trace.runtime_copy_to_host_calls.load(),
            trace.transport_copy_from_device_calls.load()};
}

void check_two_live_buffer_resource_and_failure_isolation() {
    Trace first_trace;
    Trace second_trace;
    ResourceIdentity first_identity;
    ResourceIdentity second_identity;
    auto first = make_buffer(first_trace, &first_identity);
    auto second = make_buffer(second_trace, &second_identity);
    CHECK(first != nullptr);
    CHECK(second != nullptr);
    if (first == nullptr || second == nullptr)
        return;

    CHECK(first->testing_operation_generation() == 0);
    CHECK(error_contains(
        [&] { first->barrier(false, false, false); }, "sequential=True"));
    CHECK(first->testing_operation_generation() == 0);
    first->barrier(false, false, true);
    CHECK(first->testing_operation_generation() == 1);

    const auto first_pointers = pointers(first_identity);
    const auto second_pointers = pointers(second_identity);
    for (auto* pointer : first_pointers)
        CHECK(pointer != nullptr);
    for (auto* pointer : second_pointers)
        CHECK(pointer != nullptr);
    for (auto* first_pointer : first_pointers)
        for (auto* second_pointer : second_pointers)
            CHECK(first_pointer != second_pointer);
    CHECK(first_trace.registered_window == first_identity.window);
    CHECK(second_trace.registered_window == second_identity.window);

    first->barrier(false, false, true);
    second->barrier(false, false, true);
    first_trace.fail_barrier_diagnostic = true;
    CHECK(error_contains(
        [&] { first->barrier(false, false, true); },
        "device diagnostic reported failure"));
    CHECK(error_contains(
        [&] { first->barrier(false, false, true); }, "poisoned"));
    second->barrier(false, false, true);
    CHECK(first_trace.barrier_launches == 3);
    CHECK(second_trace.barrier_launches == 2);

    CHECK(error_contains(
        [&] { first->destroy(); }, "device diagnostic reported failure"));
    CHECK(first_trace.destroy_event_calls == 3);
    CHECK(first_trace.runtime_free_calls > 0);
    CHECK(first_trace.deregister_calls == 1);
    CHECK(second_trace.runtime_free_calls == 0);
    CHECK(second_trace.host_free_calls == 0);
    CHECK(second_trace.deregister_calls == 0);
    CHECK(second_trace.destroy_team_calls == 0);
    second->barrier(false, false, true);
    CHECK(second_trace.barrier_launches == 3);
    second->destroy();
    CHECK(second_trace.destroy_event_calls == 3);
}

void check_destroy_is_busy_while_real_operation_uses_resources() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;

    {
        std::lock_guard<std::mutex> lock(trace.mutex);
        trace.block_sync = true;
    }
    bool operation_ok = false;
    bool operation_done = false;
    std::thread operation([&] {
        try {
            buffer->barrier(false, false, true);
            operation_ok = true;
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(trace.mutex);
            operation_done = true;
        }
        trace.cv.notify_all();
    });
    bool sync_entered = false;
    {
        std::unique_lock<std::mutex> lock(trace.mutex);
        trace.cv.wait(lock, [&] { return trace.sync_entered || operation_done; });
        sync_entered = trace.sync_entered;
    }
    if (!sync_entered) operation.join();
    CHECK(sync_entered);
    if (!sync_entered)
        return;

    const auto runtime_free_before = trace.runtime_free_calls.load();
    const auto host_free_before = trace.host_free_calls.load();
    const auto deregister_before = trace.deregister_calls.load();
    const auto destroy_team_before = trace.destroy_team_calls.load();
    CHECK(error_contains([&] { buffer->destroy(); }, "destroy is busy"));
    CHECK(trace.runtime_free_calls == runtime_free_before);
    CHECK(trace.host_free_calls == host_free_before);
    CHECK(trace.deregister_calls == deregister_before);
    CHECK(trace.destroy_team_calls == destroy_team_before);

    {
        std::lock_guard<std::mutex> lock(trace.mutex);
        trace.release_sync = true;
    }
    trace.cv.notify_all();
    operation.join();
    CHECK(operation_ok);
    CHECK(trace.barrier_launches == 1);
    buffer->destroy();
    CHECK(trace.runtime_free_calls > runtime_free_before);
    CHECK(trace.host_free_calls > host_free_before);
    CHECK(trace.deregister_calls == deregister_before + 1);
    CHECK(trace.destroy_team_calls == destroy_team_before + 1);
}

void check_cross_operation_busy_and_deferred_poison() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;

    {
        std::lock_guard<std::mutex> lock(trace.mutex);
        trace.block_sync = true;
    }
    bool operation_ok = false;
    bool operation_done = false;
    std::thread operation([&] {
        try {
            buffer->barrier(false, false, true);
            operation_ok = true;
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(trace.mutex);
            operation_done = true;
        }
        trace.cv.notify_all();
    });
    bool sync_entered = false;
    {
        std::unique_lock<std::mutex> lock(trace.mutex);
        trace.cv.wait(lock, [&] { return trace.sync_entered || operation_done; });
        sync_entered = trace.sync_entered;
    }
    if (!sync_entered) operation.join();
    CHECK(sync_entered);
    if (!sync_entered)
        return;

    const auto blocked_callbacks = resource_callback_snapshot(trace);
    CHECK(error_contains(
        [&] { (void)buffer->get_physical_domain_size(); },
        "get_physical_domain_size is busy"));
    CHECK(resource_callback_snapshot(trace) == blocked_callbacks);

    {
        std::lock_guard<std::mutex> lock(trace.mutex);
        trace.release_sync = true;
    }
    trace.cv.notify_all();
    operation.join();
    CHECK(operation_ok);
    CHECK(trace.barrier_launches == 1);

    const auto completed_callbacks = resource_callback_snapshot(trace);
    CHECK(error_contains(
        [&] { (void)buffer->get_logical_domain_size(); }, "poisoned"));
    CHECK(resource_callback_snapshot(trace) == completed_callbacks);
    buffer->destroy();
    CHECK(trace.runtime_free_calls > 0);
    CHECK(trace.host_free_calls > 0);
    CHECK(trace.deregister_calls == 1);
    CHECK(trace.destroy_team_calls == 1);
}

void check_comm_stream_barrier_order() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;
    trace.order.clear();
    buffer->barrier(true, false, true);
    const std::vector<std::string> expected{
        "capture compute dependency",
        "create event",
        "record compute dependency",
        "comm waits dependency/previous event",
        "create event",
        "activate lease and launch barrier on comm",
        "record completion event",
        "finish completion event",
        "finish completion event",
    };
    CHECK(trace.order == expected);
    CHECK(trace.synchronize_stream_calls == 0);
    CHECK(trace.destroy_event_calls == 1);
    buffer->destroy();
    CHECK(trace.destroy_event_calls == 2);
}

void check_barrier_completion_create_failure_precedes_launch() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;
    trace.fail_create_event_on = 2;
    CHECK(error_contains(
        [&] { buffer->barrier(true, false, true); }, "backend error 22"));
    CHECK(trace.barrier_launches == 0);
    CHECK(buffer->testing_operation_generation() == 0);
    buffer->destroy();
    CHECK(trace.runtime_free_calls > 0);
}

void check_barrier_completion_record_failure_retains_launch() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;
    trace.fail_record_event_on = trace.record_event_calls.load() + 2;
    CHECK(error_contains(
        [&] { buffer->barrier(true, false, true); }, "backend error 23"));
    CHECK(trace.barrier_launches == 1);
    CHECK(error_contains([&] { buffer->destroy(); }, "synchronize_event"));
    CHECK(error_contains([&] { buffer->destroy(); }, "synchronize_event"));
    CHECK(!buffer->is_destroyed());
    CHECK(trace.runtime_free_calls == 0);
    CHECK(trace.host_free_calls == 0);
    buffer.reset();
    CHECK(trace.runtime_free_calls == 0);
    CHECK(trace.host_free_calls == 0);
}

void check_failed_operation_destroy_invalidates_public_access() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;
    trace.fail_barrier_diagnostic = true;
    CHECK(error_contains(
        [&] { buffer->barrier(true, false, true); },
        "device diagnostic reported failure"));
    CHECK(error_contains(
        [&] { buffer->destroy(); }, "device diagnostic reported failure"));
    CHECK(trace.runtime_free_calls > 0);
    CHECK(trace.host_free_calls > 0);
    CHECK(buffer->is_destroyed());
    CHECK(error_contains(
        [&] { (void)buffer->get_comm_stream(); }, "runtime is destroyed"));
}

void check_current_stream_barrier_has_bounded_event_timeout() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_buffer(trace, &resource_identity, 0);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;
    trace.order.clear();
    trace.current_stream_completion_record = true;
    trace.event_ready = false;
    CHECK(error_contains(
        [&] { buffer->barrier(false, false, true); },
        "synchronize_event failed"));
    const std::vector<std::string> expected{
        "capture compute dependency",
        "create event",
        "activate lease and launch barrier on current",
        "record completion event",
        "completion event pending",
    };
    CHECK(trace.order == expected);
    CHECK(trace.synchronize_stream_calls == 0);
}

void check_async_combine_record_failure_quarantines_ownership() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto buffer = make_combine_buffer(trace, &resource_identity, 1);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;
    CombineInputs inputs;
    trace.order.clear();
    trace.fail_record_event_on = trace.record_event_calls.load() + 3;
    CHECK(error_contains(
        [&] { (void)launch_async_combine(*buffer, inputs); },
        "record_event"));
    CHECK(trace.combine_launches == 1);
    CHECK(trace.synchronize_stream_calls == 0);
    CHECK(error_contains([&] { buffer->destroy(); }, "synchronize_event"));
    CHECK(!buffer->is_destroyed());
    CHECK(trace.runtime_free_calls == 0);
    CHECK(trace.host_free_calls == 0);
    buffer.reset();
    CHECK(trace.runtime_free_calls == 0);
    CHECK(trace.host_free_calls == 0);
}

void check_concurrent_destroy_serializes_wrapper_ownership() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto control = std::make_shared<Buffer::TestingLifecycleControl>();
    control->destroy_attempts_required = 2;
    auto buffer = make_buffer(trace, &resource_identity, 5000, control);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;

    std::atomic<int> destroy_successes{0};
    const auto destroy = [&] {
        try {
            buffer->destroy();
            ++destroy_successes;
        } catch (...) {
        }
    };
    std::thread first(destroy);
    std::thread second(destroy);
    first.join();
    second.join();

    CHECK(destroy_successes == 2);
    CHECK(control->destroy_attempts() == 2);
    CHECK(control->maximum_active_owners() == 1);
    CHECK(buffer->is_destroyed());
    CHECK(trace.deregister_calls == 1);
    CHECK(trace.destroy_team_calls == 1);
    buffer->destroy();
}

void check_comm_stream_access_serializes_with_destroy() {
    Trace trace;
    ResourceIdentity resource_identity;
    auto control = std::make_shared<Buffer::TestingLifecycleControl>();
    control->getter_waits_for_destroy_attempt = true;
    auto buffer = make_buffer(trace, &resource_identity, 5000, control);
    CHECK(buffer != nullptr);
    if (buffer == nullptr)
        return;

    std::atomic<bool> getter_ok{false};
    std::atomic<bool> destroy_ok{false};
    std::thread getter([&] {
        try {
            (void)buffer->get_comm_stream();
            getter_ok = true;
        } catch (...) {
        }
    });
    control->wait_for_active_owner();
    std::thread destroyer([&] {
        try {
            buffer->destroy();
            destroy_ok = true;
        } catch (...) {
        }
    });
    getter.join();
    destroyer.join();

    CHECK(getter_ok);
    CHECK(destroy_ok);
    CHECK(control->destroy_attempts() == 1);
    CHECK(control->maximum_active_owners() == 1);
    CHECK(buffer->is_destroyed());
    CHECK(trace.deregister_calls == 1);
    CHECK(trace.destroy_team_calls == 1);
}

int wrapper_lifecycle_stress_iterations() {
    const auto* value = std::getenv("DEEP_EP_ASCEND_WRAPPER_LIFECYCLE_STRESS");
    if (value == nullptr)
        return 1;
    const auto iterations = std::strtol(value, nullptr, 10);
    return iterations > 0 && iterations <= 1000 ?
        static_cast<int>(iterations) : 1;
}

}  // namespace

extern "C" int deep_ep_ascend_launch_barrier(
    elastic::BarrierArguments arguments, elastic::CoreTiling tiling,
    void* stream) {
    auto& trace = *static_cast<StreamToken*>(stream)->trace;
    ++trace.barrier_launches;
    trace.order.emplace_back(
        static_cast<StreamToken*>(stream)->communication ?
            "activate lease and launch barrier on comm" :
            "activate lease and launch barrier on current");
    transport::DeviceTransportDiagnostic diagnostic{};
    diagnostic.abi_version = transport::kTransportCommandAbiVersion;
    diagnostic.generation = arguments.generation;
    if (trace.fail_barrier_diagnostic) {
        diagnostic.error =
            transport::DeviceTransportError::kCompletionFailure;
        diagnostic.backend_status = 24;
    }
    std::memcpy(trace.diagnostic, &diagnostic, sizeof(diagnostic));
    std::uint64_t completion_offset = 0;
    if (!elastic::checked_rank_slot_offset(
            tiling.symmetric_window_layout.barrier_completion_offset,
            tiling.symmetric_window_layout.barrier_completion_count,
            tiling.topology.world_rank, &completion_offset))
        return 21;
    auto* completion = reinterpret_cast<std::uint64_t*>(
        tiling.transport_context.local_window_base + completion_offset);
    *completion = arguments.generation;
    return 0;
}

extern "C" int deep_ep_ascend_launch_dispatch(
    elastic::DispatchArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch_pipeline(
    elastic::DispatchArguments arguments, elastic::CoreTiling tiling,
    void* producer_stream, void*) {
    return deep_ep_ascend_launch_dispatch(arguments, tiling, producer_stream);
}
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    elastic::DispatchArguments, elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine(
    elastic::CombineArguments arguments, elastic::CoreTiling tiling,
    void* stream) {
    auto& trace = *static_cast<StreamToken*>(stream)->trace;
    ++trace.combine_launches;
    trace.order.emplace_back(
        static_cast<StreamToken*>(stream)->communication ?
            "activate lease and launch combine on comm" :
            "activate lease and launch combine on current");
    transport::DeviceTransportDiagnostic diagnostic{};
    diagnostic.abi_version = transport::kTransportCommandAbiVersion;
    diagnostic.generation = arguments.generation;
    std::memcpy(trace.diagnostic, &diagnostic, sizeof(diagnostic));
    auto* control = reinterpret_cast<elastic::SymmetricControlHeader*>(
        arguments.local_window_base +
        tiling.symmetric_window_layout.control_offset);
    control->combine_generation = arguments.generation;
    return 0;
}
extern "C" int deep_ep_ascend_launch_combine_epilogue(
    elastic::CombineArguments, elastic::CoreTiling, void*) { return 0; }

int main() {
    check_two_live_buffer_resource_and_failure_isolation();
    check_destroy_is_busy_while_real_operation_uses_resources();
    check_cross_operation_busy_and_deferred_poison();
    check_comm_stream_barrier_order();
    check_barrier_completion_create_failure_precedes_launch();
    check_barrier_completion_record_failure_retains_launch();
    check_failed_operation_destroy_invalidates_public_access();
    check_current_stream_barrier_has_bounded_event_timeout();
    check_async_combine_record_failure_quarantines_ownership();
    for (int iteration = 0;
         iteration < wrapper_lifecycle_stress_iterations(); ++iteration) {
        check_concurrent_destroy_serializes_wrapper_ownership();
        check_comm_stream_access_serializes_with_destroy();
    }
    return failures == 0 ? 0 : 1;
}
