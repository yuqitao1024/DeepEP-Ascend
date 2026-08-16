#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

#include "csrc/backends/ascend/transport/cann_transport.hpp"
#include "csrc/backends/ascend/transport/transport_commands.hpp"

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
    kCopyFromDevice,
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
    int deregister_failures_remaining = 0;
    int destroy_team_failures_remaining = 0;
    transport::StagedTransportContext staged{};
    std::uint32_t staged_copy_count = 0;

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

    std::uint32_t first(Event event) const {
        for (std::uint32_t index = 0; index < event_count; ++index) {
            if (events[index] == event)
                return index;
        }
        return event_count;
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
        void* data, std::int64_t, std::uint32_t rank, std::uint32_t size,
        const std::uint32_t* rank_ids, std::uint32_t signal_count,
        std::uint32_t barrier_count, std::uintptr_t* team) {
        auto& fake = self(data);
        fake.record(Event::kCreateTeam);
        CHECK(rank == 1);
        CHECK(size == 2);
        CHECK(rank_ids != nullptr);
        CHECK(rank_ids[0] == 0);
        CHECK(rank_ids[1] == 1);
        CHECK(signal_count == 0);
        CHECK(barrier_count == 5);
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

    static int copy(
        void* data, void*, const void* source, std::uint64_t bytes) {
        auto& fake = self(data);
        fake.record(Event::kCopy);
        if (fake.fail_now()) return 78;
        if (bytes == sizeof(transport::StagedTransportContext)) {
            const auto* candidate =
                static_cast<const transport::StagedTransportContext*>(source);
            if (candidate->struct_size ==
                    sizeof(transport::StagedTransportContext) &&
                candidate->cann_compatibility ==
                    transport::kStagedTransportCannCompatibility) {
                fake.staged = *candidate;
                ++fake.staged_copy_count;
            }
        }
        return 0;
    }

    static int copy_from_device(
        void* data, void* destination, const void*, std::uint64_t bytes) {
        auto& fake = self(data);
        fake.record(Event::kCopyFromDevice);
        if (fake.fail_now()) return 79;
        CHECK(bytes == sizeof(transport::DeviceTransportDiagnostic));
        auto* diagnostic =
            static_cast<transport::DeviceTransportDiagnostic*>(destination);
        diagnostic->error = transport::DeviceTransportError::kCompletionTimeout;
        diagnostic->generation = 17;
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
        if (fake.deregister_failures_remaining > 0) {
            --fake.deregister_failures_remaining;
            return 80;
        }
        return 0;
    }

    static int destroy_team(void* data, std::uintptr_t) {
        auto& fake = self(data);
        fake.record(Event::kDestroyTeam);
        if (fake.destroy_team_failures_remaining > 0) {
            --fake.destroy_team_failures_remaining;
            return 81;
        }
        return 0;
    }

    transport::CannHostApi api() {
        return {
            this, get_rank, get_size, create_team, register_window,
            create_channels, allocate, zero, copy, copy_from_device, free,
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
    constexpr auto kValidatedCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(
            transport::TransportCapability::kRemoteAtomicAddRelease) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(transport::TransportCapability::kScaleUpTeam);
    static_assert(kValidatedCapabilities == 0x775);

    FakeApi fake;
    auto created = transport::make_cann_transport(valid_config(), fake.api());
    CHECK(created.status.ok());
    CHECK(created.transport != nullptr);
    CHECK(created.transport->capabilities() == kValidatedCapabilities);

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
    CHECK(context.capabilities == kValidatedCapabilities);
    CHECK(!transport::has_capability(
        context.capabilities, transport::TransportCapability::kDirectPeerPointer));
    CHECK(!transport::has_capability(
        context.capabilities, transport::TransportCapability::kDeviceGet));
    CHECK(!transport::has_capability(
        context.capabilities, transport::TransportCapability::kAsyncCompletion));
    CHECK(!transport::has_capability(
        context.capabilities, transport::TransportCapability::kScaleOutTeam));
    CHECK(context.topology.world_rank == 1);
    CHECK(context.topology.world_size == 2);
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(created.transport->read_diagnostic(&diagnostic).ok());
    CHECK(diagnostic.error ==
          transport::DeviceTransportError::kCompletionTimeout);
    CHECK(diagnostic.generation == 17);
    CHECK(fake.count(Event::kCopyFromDevice) == 1);
    CHECK(fake.first(Event::kCreateTeam) < fake.first(Event::kRegisterWindow));
    CHECK(fake.first(Event::kRegisterWindow) < fake.first(Event::kCreateChannels));
    CHECK(fake.first(Event::kCreateChannels) < fake.first(Event::kAllocate));
    CHECK(fake.staged_copy_count == 1);
    CHECK(fake.staged.fetch_results == 0);
    CHECK(fake.staged.fetch_result_bytes == 0);
    CHECK(fake.count(Event::kAllocate) == 5);
    CHECK(fake.count(Event::kZero) == 5);
    CHECK(fake.count(Event::kCopy) == 3);

    CHECK(created.transport->destroy().ok());
    const auto after_first_destroy = fake.event_count;
    CHECK(created.transport->destroy().ok());
    CHECK(fake.event_count == after_first_destroy);
    CHECK(fake.count(Event::kFree) == 5);
    CHECK(fake.events[fake.event_count - 2] == Event::kDeregisterWindow);
    CHECK(fake.events[fake.event_count - 1] == Event::kDestroyTeam);
}

void check_partial_failure_cleans_up() {
    for (int fail_event = 0; fail_event < 18; ++fail_event) {
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
        const auto allocations = static_cast<std::uint32_t>(
            (fake.next_pointer - 0x100000) / 0x1000);
        CHECK(fake.count(Event::kFree) == allocations);
        CHECK(fake.events[fake.event_count - 1] == Event::kDestroyTeam);
    }
}

std::unique_ptr<transport::HostTransport> make_active_transport(FakeApi& fake) {
    auto created = transport::make_cann_transport(valid_config(), fake.api());
    CHECK(created.status.ok());
    CHECK(created.transport != nullptr);
    alignas(64) static std::uint8_t window[4096]{};
    CHECK(created.transport->register_symmetric_window(
        window, sizeof(window)).ok());
    CHECK(created.transport->acquire_channels(
        1, transport::CooperationScope::kParticipant).ok());
    return std::move(created.transport);
}

void check_deregister_failure_is_retryable() {
    FakeApi fake;
    auto active = make_active_transport(fake);
    fake.deregister_failures_remaining = 1;

    const auto first = active->destroy();
    CHECK(!first.ok());
    CHECK(first.operation == "unregister_symmetric_window");
    CHECK(fake.count(Event::kDeregisterWindow) == 1);
    CHECK(fake.count(Event::kDestroyTeam) == 0);

    CHECK(active->destroy().ok());
    CHECK(fake.count(Event::kDeregisterWindow) == 2);
    CHECK(fake.count(Event::kDestroyTeam) == 1);
}

void check_team_destroy_failure_is_retryable() {
    FakeApi fake;
    auto active = make_active_transport(fake);
    fake.destroy_team_failures_remaining = 1;

    const auto first = active->destroy();
    CHECK(!first.ok());
    CHECK(first.operation == "destroy_team");
    CHECK(fake.count(Event::kDeregisterWindow) == 1);
    CHECK(fake.count(Event::kDestroyTeam) == 1);

    CHECK(active->destroy().ok());
    CHECK(fake.count(Event::kDeregisterWindow) == 1);
    CHECK(fake.count(Event::kDestroyTeam) == 2);
}

}  // namespace

int main() {
    check_success_and_reverse_cleanup();
    check_partial_failure_cleans_up();
    check_deregister_failure_is_retryable();
    check_team_destroy_failure_is_retryable();
    return failures == 0 ? 0 : 1;
}
