#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

#include "csrc/backends/ascend/transport/cann_transport.hpp"

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

enum class Event : std::uint32_t {
    kGetRank,
    kGetSize,
    kCreateTeam,
    kRegisterWindow,
    kCreateChannels,
    kAllocate,
    kZero,
    kCopy,
    kFree,
    kDeregisterWindow,
    kDestroyTeam,
};

struct FakeApi {
    Event events[128]{};
    std::uint32_t event_count = 0;
    std::uint32_t rank = 1;
    std::uint32_t size = 2;
    std::uintptr_t next_pointer = 0x100000;
    int fail_event = -1;
    int event_calls = 0;

    bool fail_now() {
        return fail_event >= 0 && event_calls++ == fail_event;
    }

    void record(Event event) {
        events[event_count++] = event;
    }

    std::uint32_t count(Event event) const {
        std::uint32_t result = 0;
        for (std::uint32_t index = 0; index < event_count; ++index)
            result += events[index] == event ? 1U : 0U;
        return result;
    }

    static FakeApi& self(void* data) {
        return *static_cast<FakeApi*>(data);
    }

    static int get_rank(void* data, std::int64_t, std::uint32_t* rank) {
        auto& fake = self(data);
        fake.record(Event::kGetRank);
        if (fake.fail_now()) return 71;
        *rank = fake.rank;
        return 0;
    }

    static int get_size(void* data, std::int64_t, std::uint32_t* size) {
        auto& fake = self(data);
        fake.record(Event::kGetSize);
        if (fake.fail_now()) return 72;
        *size = fake.size;
        return 0;
    }

    static int create_team(
        void* data, std::int64_t, std::uint32_t, std::uint32_t,
        std::uint32_t signal_count, std::uint32_t barrier_count,
        std::uintptr_t* team) {
        auto& fake = self(data);
        fake.record(Event::kCreateTeam);
        CHECK(signal_count == 4);
        CHECK(barrier_count == 1);
        if (fake.fail_now()) return 73;
        *team = 0x200000;
        return 0;
    }

    static int register_window(
        void* data, std::int64_t, std::uintptr_t, void*, std::uint64_t,
        std::uintptr_t* window) {
        auto& fake = self(data);
        fake.record(Event::kRegisterWindow);
        if (fake.fail_now()) return 74;
        *window = 0x300000;
        return 0;
    }

    static int create_channels(
        void* data, std::int64_t, std::uintptr_t, std::uint32_t count) {
        auto& fake = self(data);
        fake.record(Event::kCreateChannels);
        CHECK(count == 1);
        if (fake.fail_now()) return 75;
        return 0;
    }

    static int allocate(void* data, std::uint64_t, void** pointer) {
        auto& fake = self(data);
        fake.record(Event::kAllocate);
        if (fake.fail_now()) return 76;
        *pointer = reinterpret_cast<void*>(fake.next_pointer);
        fake.next_pointer += 0x1000;
        return 0;
    }

    static int zero(void* data, void*, std::uint64_t) {
        auto& fake = self(data);
        fake.record(Event::kZero);
        if (fake.fail_now()) return 77;
        return 0;
    }

    static int copy(void* data, void*, const void*, std::uint64_t) {
        auto& fake = self(data);
        fake.record(Event::kCopy);
        if (fake.fail_now()) return 78;
        return 0;
    }

    static int free(void* data, void*) {
        auto& fake = self(data);
        fake.record(Event::kFree);
        return 0;
    }

    static int deregister_window(
        void* data, std::uintptr_t, std::uintptr_t) {
        auto& fake = self(data);
        fake.record(Event::kDeregisterWindow);
        return 0;
    }

    static int destroy_team(void* data, std::uintptr_t) {
        auto& fake = self(data);
        fake.record(Event::kDestroyTeam);
        return 0;
    }

    transport::CannHostApi api() {
        return {
            this, get_rank, get_size, create_team, register_window,
            create_channels, allocate, zero, copy, free,
            deregister_window, destroy_team,
        };
    }
};

transport::TransportConfig valid_config() {
    transport::TransportConfig config;
    config.rank = 1;
    config.world_size = 2;
    config.communicator_handle = 0x1234;
    config.device_buffer_bytes = 4096;
    config.requested_channels = 1;
    return config;
}

void check_success_and_reverse_cleanup() {
    FakeApi fake;
    auto created = transport::make_cann_transport(valid_config(), fake.api());
    CHECK(created.status.ok());
    CHECK(created.transport != nullptr);
    CHECK(created.transport->capabilities() == transport::kNoCapabilities);

    alignas(64) std::uint8_t window[4096]{};
    CHECK(created.transport->register_symmetric_window(window, sizeof(window)).ok());
    CHECK(created.transport->acquire_channels(
        1, transport::CooperationScope::kParticipant).ok());

    transport::DeviceTransportContext context{};
    CHECK(created.transport->export_device_context(&context).ok());
    CHECK(context.local_window_base ==
          reinterpret_cast<std::uintptr_t>(window));
    CHECK(context.peer_address_table == 0x300000);
    CHECK(context.channel_table == 0x200000);
    CHECK(context.backend_context != 0);
    CHECK(context.capabilities == transport::kNoCapabilities);
    CHECK(context.topology.world_rank == 1);
    CHECK(context.topology.world_size == 2);
    CHECK(fake.count(Event::kAllocate) == 6);
    CHECK(fake.count(Event::kZero) == 6);
    CHECK(fake.count(Event::kCopy) == 5);

    CHECK(created.transport->destroy().ok());
    const auto after_first_destroy = fake.event_count;
    CHECK(created.transport->destroy().ok());
    CHECK(fake.event_count == after_first_destroy);
    CHECK(fake.count(Event::kFree) == 6);
    CHECK(fake.events[fake.event_count - 2] == Event::kDeregisterWindow);
    CHECK(fake.events[fake.event_count - 1] == Event::kDestroyTeam);
}

void check_partial_failure_cleans_up() {
    for (int fail_event = 0; fail_event < 23; ++fail_event) {
        FakeApi fake;
        fake.fail_event = fail_event;
        auto created = transport::make_cann_transport(
            valid_config(), fake.api());
        if (!created.status.ok()) {
            CHECK(created.transport == nullptr);
            const auto allocations = static_cast<std::uint32_t>(
                (fake.next_pointer - 0x100000) / 0x1000);
            CHECK(fake.count(Event::kFree) == allocations);
            if (fake.count(Event::kCreateTeam) != 0 && fail_event > 2)
                CHECK(fake.events[fake.event_count - 1] == Event::kDestroyTeam);
            continue;
        }

        alignas(64) std::uint8_t window[4096]{};
        const auto window_status = created.transport->register_symmetric_window(
            window, sizeof(window));
        if (window_status.ok())
            (void)created.transport->acquire_channels(
                1, transport::CooperationScope::kParticipant);
        (void)created.transport->destroy();
        CHECK(fake.count(Event::kFree) == 6);
        CHECK(fake.events[fake.event_count - 1] == Event::kDestroyTeam);
    }
}

}  // namespace

int main() {
    check_success_and_reverse_cleanup();
    check_partial_failure_cleans_up();
    return failures == 0 ? 0 : 1;
}
