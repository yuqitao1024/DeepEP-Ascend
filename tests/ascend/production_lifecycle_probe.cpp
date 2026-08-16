#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/backends/ascend/runtime/cann_runtime.hpp"

namespace runtime = deep_ep::ascend::runtime;
namespace transport = deep_ep::ascend::transport;

namespace {

int failures = 0;

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << ": "                \
                      << #expression << '\n';                                 \
            ++failures;                                                       \
        }                                                                     \
    } while (false)

struct Trace {
    std::vector<std::string> events;
    int runtime_fail_call = -1;
    int runtime_calls = 0;
    int host_fail_call = -1;
    int host_calls = 0;
    std::uintptr_t runtime_next = 0x100003;
    std::uintptr_t host_next = 0x800000;
    std::uintptr_t current_stream = 0x5151;
    std::uintptr_t synchronized_stream = 0;
    int deregister_failures_remaining = 0;
    int destroy_team_failures_remaining = 0;
    int runtime_free_fail_call = -1;
    int runtime_free_calls = 0;
    int runtime_copy_from_host_failures_remaining = 0;
    bool null_stream = false;

    bool runtime_fail() {
        return runtime_fail_call >= 0 && runtime_calls++ == runtime_fail_call;
    }

    bool host_fail() {
        return host_fail_call >= 0 && host_calls++ == host_fail_call;
    }

    std::size_t count(const std::string& event) const {
        return static_cast<std::size_t>(
            std::count(events.begin(), events.end(), event));
    }

    std::size_t first(const std::string& event) const {
        const auto iterator = std::find(events.begin(), events.end(), event);
        return iterator == events.end() ? events.size() :
            static_cast<std::size_t>(iterator - events.begin());
    }
};

Trace& self(void* data) { return *static_cast<Trace*>(data); }

int runtime_allocate(void* data, std::uint64_t, void** pointer) {
    auto& trace = self(data);
    trace.events.emplace_back("runtime_allocate");
    if (trace.runtime_fail()) return 61;
    *pointer = reinterpret_cast<void*>(trace.runtime_next);
    trace.runtime_next += 0x400003;
    return 0;
}

int runtime_zero(void* data, void*, std::uint64_t) {
    auto& trace = self(data);
    trace.events.emplace_back("runtime_zero");
    return trace.runtime_fail() ? 62 : 0;
}

int runtime_free(void* data, void*) {
    auto& trace = self(data);
    trace.events.emplace_back("runtime_free");
    const bool fail = trace.runtime_free_calls++ == trace.runtime_free_fail_call;
    if (fail)
        return 63;
    return 0;
}

void* runtime_current_stream(void* data) {
    auto& trace = self(data);
    trace.events.emplace_back("current_stream");
    return trace.null_stream ? nullptr :
        reinterpret_cast<void*>(trace.current_stream);
}

int runtime_synchronize_stream(void* data, void* stream) {
    auto& trace = self(data);
    trace.events.emplace_back("synchronize_stream");
    trace.synchronized_stream = reinterpret_cast<std::uintptr_t>(stream);
    return 0;
}

int runtime_synchronize_device(void* data) {
    self(data).events.emplace_back("synchronize_device");
    return 0;
}

int runtime_copy_from_host(void* data, void*, const void*, std::uint64_t) {
    auto& trace = self(data);
    trace.events.emplace_back("runtime_copy_from_host");
    if (trace.runtime_copy_from_host_failures_remaining > 0) {
        --trace.runtime_copy_from_host_failures_remaining;
        return 64;
    }
    return 0;
}

int runtime_copy_to_host(
    void* data, void*, const void*, std::uint64_t) {
    self(data).events.emplace_back("runtime_copy_to_host");
    return 0;
}

runtime::CannRuntimeApi runtime_api(Trace& trace) {
    return {&trace, runtime_allocate, runtime_zero, runtime_free,
            runtime_current_stream, runtime_synchronize_stream,
            runtime_synchronize_device, runtime_copy_from_host,
            runtime_copy_to_host};
}

int get_rank(void* data, std::int64_t, std::uint32_t* rank) {
    auto& trace = self(data);
    trace.events.emplace_back("get_rank");
    if (trace.host_fail()) return 71;
    *rank = 0;
    return 0;
}

int get_size(void* data, std::int64_t, std::uint32_t* size) {
    auto& trace = self(data);
    trace.events.emplace_back("get_size");
    if (trace.host_fail()) return 72;
    *size = 2;
    return 0;
}

int create_team(
    void* data, std::int64_t, std::uint32_t, std::uint32_t,
    const std::uint32_t*, std::uint32_t, std::uint32_t,
    std::uintptr_t* team) {
    auto& trace = self(data);
    trace.events.emplace_back("create_team");
    if (trace.host_fail()) return 73;
    *team = 0x200000;
    return 0;
}

int register_window(
    void* data, std::int64_t, std::uintptr_t, void* base,
    std::uint64_t bytes, std::uintptr_t* window) {
    auto& trace = self(data);
    trace.events.emplace_back("register_window");
    CHECK(reinterpret_cast<std::uintptr_t>(base) %
          deep_ep::ascend::elastic::kPublicElasticBufferAlignment == 0);
    CHECK(bytes == 2 * 1024 * 1024);
    if (trace.host_fail()) return 74;
    *window = 0x300000;
    return 0;
}

int create_channels(
    void* data, std::int64_t, std::uintptr_t, std::uint32_t count) {
    auto& trace = self(data);
    trace.events.emplace_back("create_channels");
    CHECK(count == 1);
    return trace.host_fail() ? 75 : 0;
}

int host_allocate(void* data, std::uint64_t, void** pointer) {
    auto& trace = self(data);
    trace.events.emplace_back("host_allocate");
    if (trace.host_fail()) return 76;
    *pointer = reinterpret_cast<void*>(trace.host_next);
    trace.host_next += 0x1000;
    return 0;
}

int host_zero(void* data, void*, std::uint64_t) {
    auto& trace = self(data);
    trace.events.emplace_back("host_zero");
    return trace.host_fail() ? 77 : 0;
}

int copy_to_device(void* data, void*, const void*, std::uint64_t) {
    auto& trace = self(data);
    trace.events.emplace_back("copy_to_device");
    return trace.host_fail() ? 78 : 0;
}

int copy_from_device(void* data, void*, const void*, std::uint64_t) {
    auto& trace = self(data);
    trace.events.emplace_back("copy_from_device");
    return trace.host_fail() ? 79 : 0;
}

int host_free(void* data, void*) {
    self(data).events.emplace_back("host_free");
    return 0;
}

int deregister_window(void* data, std::uintptr_t, std::uintptr_t) {
    auto& trace = self(data);
    trace.events.emplace_back("deregister_window");
    if (trace.deregister_failures_remaining > 0) {
        --trace.deregister_failures_remaining;
        return 80;
    }
    return 0;
}

int destroy_team(void* data, std::uintptr_t) {
    auto& trace = self(data);
    trace.events.emplace_back("destroy_team");
    if (trace.destroy_team_failures_remaining > 0) {
        --trace.destroy_team_failures_remaining;
        return 81;
    }
    return 0;
}

transport::CannHostApi host_api(Trace& trace) {
    return {&trace, get_rank, get_size, create_team, register_window,
            create_channels, host_allocate, host_zero, copy_to_device,
            copy_from_device, host_free, deregister_window, destroy_team};
}

transport::TransportConfig config() {
    transport::TransportConfig value{};
    value.rank = 0;
    value.world_size = 2;
    value.communicator_handle = 0x1234;
    value.device_buffer_bytes = 2 * 1024 * 1024;
    value.requested_channels = 1;
    return value;
}

void check_success_and_idempotent_cleanup() {
    Trace trace;
    runtime::CannRuntimeResources resources;
    auto status = resources.initialize(
        config(), 4096, runtime_api(trace), host_api(trace));
    CHECK(status.ok());
    CHECK(resources.initialized());
    CHECK(resources.window_base() != nullptr);
    CHECK(reinterpret_cast<std::uintptr_t>(resources.window_base()) %
          deep_ep::ascend::elastic::kPublicElasticBufferAlignment == 0);
    CHECK(resources.workspace() != nullptr);
    CHECK(resources.transport() != nullptr);
    CHECK(resources.device_context().topology.world_size == 2);
    CHECK(trace.first("runtime_allocate") < trace.first("create_team"));
    CHECK(trace.first("create_team") < trace.first("register_window"));
    CHECK(trace.first("register_window") < trace.first("create_channels"));
    CHECK(trace.first("create_channels") < trace.first("current_stream"));
    CHECK(trace.count("runtime_allocate") == 2);

    trace.current_stream = 0x6161;
    void* current_stream = nullptr;
    CHECK(resources.current_stream(&current_stream).ok());
    CHECK(current_stream == reinterpret_cast<void*>(0x6161));
    CHECK(resources.synchronize_stream(current_stream).ok());
    CHECK(trace.synchronized_stream == 0x6161);
    std::uint64_t source = 3;
    CHECK(resources.copy_from_host(resources.workspace(), &source, sizeof(source)).ok());
    CHECK(trace.count("runtime_copy_from_host") == 1);

    CHECK(resources.destroy().ok());
    const auto after_destroy = trace.events.size();
    CHECK(resources.destroy().ok());
    CHECK(trace.events.size() == after_destroy);
    CHECK(trace.count("runtime_free") == 2);
    CHECK(trace.first("deregister_window") < trace.first("destroy_team"));
    CHECK(trace.events.back() == "runtime_free");
}

void check_copy_failure_preserves_resources_for_retry() {
    Trace trace;
    runtime::CannRuntimeResources resources;
    CHECK(resources.initialize(
        config(), 4096, runtime_api(trace), host_api(trace)).ok());
    std::uint64_t source = 3;
    trace.runtime_copy_from_host_failures_remaining = 1;
    const auto first = resources.copy_from_host(
        resources.workspace(), &source, sizeof(source));
    CHECK(!first.ok());
    CHECK(first.operation == "copy_from_host");
    CHECK(first.backend_code == 64);
    CHECK(resources.initialized());
    CHECK(resources.workspace() != nullptr);
    CHECK(resources.copy_from_host(
        resources.workspace(), &source, sizeof(source)).ok());
    CHECK(resources.destroy().ok());
}

void check_invalid_copy_requests_do_not_call_backend() {
    Trace trace;
    runtime::CannRuntimeResources resources;
    CHECK(resources.initialize(
        config(), 4096, runtime_api(trace), host_api(trace)).ok());
    std::uint64_t value = 3;
    const auto before = trace.count("runtime_copy_from_host");
    CHECK(!resources.copy_from_host(nullptr, &value, sizeof(value)).ok());
    CHECK(!resources.copy_from_host(resources.workspace(), nullptr, sizeof(value)).ok());
    CHECK(!resources.copy_from_host(resources.workspace(), &value, 0).ok());
    CHECK(trace.count("runtime_copy_from_host") == before);
    CHECK(resources.destroy().ok());
    CHECK(!resources.copy_from_host(resources.workspace(), &value, sizeof(value)).ok());
    CHECK(trace.count("runtime_copy_from_host") == before);
}

void check_runtime_failures_cleanup() {
    for (int fail_call = 0; fail_call < 4; ++fail_call) {
        Trace trace;
        trace.runtime_fail_call = fail_call;
        runtime::CannRuntimeResources resources;
        const auto status = resources.initialize(
            config(), 4096, runtime_api(trace), host_api(trace));
        CHECK(!status.ok());
        CHECK(status.backend_code == 61 || status.backend_code == 62);
        CHECK(!resources.initialized());
        CHECK(trace.count("runtime_free") <= trace.count("runtime_allocate"));
        const auto after_failure = trace.events.size();
        CHECK(resources.destroy().ok());
        CHECK(trace.events.size() == after_failure);
    }
}

void check_host_failure_cleans_outer_allocation() {
    for (int fail_call = 0; fail_call < 18; ++fail_call) {
        Trace trace;
        trace.host_fail_call = fail_call;
        runtime::CannRuntimeResources resources;
        const auto status = resources.initialize(
            config(), 4096, runtime_api(trace), host_api(trace));
        if (status.ok()) {
            CHECK(resources.destroy().ok());
            continue;
        }
        CHECK(!resources.initialized());
        CHECK(trace.count("runtime_free") >= 1);
        CHECK(trace.count("runtime_free") == trace.count("runtime_allocate"));
    }
}

void check_null_stream_cleans_all_resources() {
    Trace trace;
    trace.null_stream = true;
    runtime::CannRuntimeResources resources;
    const auto status = resources.initialize(
        config(), 4096, runtime_api(trace), host_api(trace));
    CHECK(!status.ok());
    CHECK(status.operation == "current_stream");
    CHECK(!resources.initialized());
    CHECK(trace.count("runtime_free") == trace.count("runtime_allocate"));
    CHECK(trace.count("deregister_window") == 1);
    CHECK(trace.count("destroy_team") == 1);
}

void check_deregister_failure_preserves_outer_window() {
    Trace trace;
    runtime::CannRuntimeResources resources;
    CHECK(resources.initialize(
        config(), 4096, runtime_api(trace), host_api(trace)).ok());
    trace.deregister_failures_remaining = 1;

    const auto first = resources.destroy();
    CHECK(!first.ok());
    CHECK(first.operation == "unregister_symmetric_window");
    CHECK(!resources.initialized());
    CHECK(resources.window_base() != nullptr);
    CHECK(trace.count("runtime_free") == 1);
    CHECK(trace.count("deregister_window") == 1);
    CHECK(trace.count("destroy_team") == 0);

    CHECK(resources.destroy().ok());
    CHECK(resources.window_base() == nullptr);
    CHECK(trace.count("runtime_free") == 2);
    CHECK(trace.count("deregister_window") == 2);
    CHECK(trace.count("destroy_team") == 1);
}

void check_team_destroy_failure_preserves_outer_window() {
    Trace trace;
    runtime::CannRuntimeResources resources;
    CHECK(resources.initialize(
        config(), 4096, runtime_api(trace), host_api(trace)).ok());
    trace.destroy_team_failures_remaining = 1;

    const auto first = resources.destroy();
    CHECK(!first.ok());
    CHECK(first.operation == "destroy_team");
    CHECK(!resources.initialized());
    CHECK(resources.window_base() != nullptr);
    CHECK(trace.count("runtime_free") == 1);
    CHECK(trace.count("deregister_window") == 1);
    CHECK(trace.count("destroy_team") == 1);

    CHECK(resources.destroy().ok());
    CHECK(resources.window_base() == nullptr);
    CHECK(trace.count("runtime_free") == 2);
    CHECK(trace.count("deregister_window") == 1);
    CHECK(trace.count("destroy_team") == 2);
}

void check_runtime_free_failure_is_retryable() {
    for (int fail_call = 0; fail_call < 2; ++fail_call) {
        Trace trace;
        runtime::CannRuntimeResources resources;
        CHECK(resources.initialize(
            config(), 4096, runtime_api(trace), host_api(trace)).ok());
        trace.runtime_free_fail_call = fail_call;

        const auto first = resources.destroy();
        CHECK(!first.ok());
        CHECK(first.operation ==
              (fail_call == 0 ? "free_workspace" : "free_window"));
        CHECK(!resources.initialized());
        CHECK((resources.workspace() != nullptr) == (fail_call == 0));
        CHECK((resources.window_base() != nullptr) == (fail_call == 1));

        CHECK(resources.destroy().ok());
        CHECK(resources.workspace() == nullptr);
        CHECK(resources.window_base() == nullptr);
        CHECK(trace.count("runtime_free") == 3);
    }
}

}  // namespace

int main() {
    check_success_and_idempotent_cleanup();
    check_runtime_failures_cleanup();
    check_host_failure_cleans_outer_allocation();
    check_null_stream_cleans_all_resources();
    check_deregister_failure_preserves_outer_window();
    check_team_destroy_failure_preserves_outer_window();
    check_runtime_free_failure_is_retryable();
    check_copy_failure_preserves_resources_for_retry();
    check_invalid_copy_requests_do_not_call_backend();
    return failures == 0 ? 0 : 1;
}
