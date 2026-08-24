#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

#include "csrc/backends/ascend/transport/cann_transport.hpp"
#include "csrc/backends/ascend/transport/sync_layout.hpp"
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
    transport::TransportCommandQueue queue{};
    std::uint32_t queue_copy_count = 0;
    transport::DeviceTransportDiagnostic diagnostic{};
    std::uint32_t diagnostic_copy_count = 0;
    transport::TransportStageProfile profile{};
    std::uint32_t profile_copy_count = 0;
    void* profile_device_pointer = nullptr;
    std::uint64_t allocation_bytes[8]{};
    std::uint32_t allocation_count = 0;

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
        CHECK(rank == fake.rank);
        CHECK(size == fake.size);
        CHECK(rank_ids != nullptr);
        for (std::uint32_t index = 0; index < size; ++index)
            CHECK(rank_ids[index] == index);
        CHECK(signal_count == 0);
        CHECK(barrier_count ==
              transport::sync_layout::kWorldTeamBarrierCount);
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

    static int allocate(void* data, std::uint64_t bytes, void** pointer) {
        auto& fake = self(data);
        fake.record(Event::kAllocate);
        if (fake.fail_now()) return 76;
        CHECK(fake.allocation_count < 8);
        fake.allocation_bytes[fake.allocation_count++] = bytes;
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
        void* data, void* destination, const void* source,
        std::uint64_t bytes) {
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
        if (destination == reinterpret_cast<void*>(0x101000) &&
            bytes == sizeof(transport::TransportCommandQueue)) {
            fake.queue =
                *static_cast<const transport::TransportCommandQueue*>(source);
            ++fake.queue_copy_count;
        }
        if (destination == reinterpret_cast<void*>(0x103000) &&
            bytes == sizeof(transport::DeviceTransportDiagnostic)) {
            fake.diagnostic = *static_cast<
                const transport::DeviceTransportDiagnostic*>(source);
            ++fake.diagnostic_copy_count;
        }
        if (bytes == sizeof(transport::TransportStageProfile)) {
            fake.profile =
                *static_cast<const transport::TransportStageProfile*>(source);
            fake.profile_device_pointer = destination;
            ++fake.profile_copy_count;
        }
        return 0;
    }

    static int copy_from_device(
        void* data, void* destination, const void* source,
        std::uint64_t bytes) {
        auto& fake = self(data);
        fake.record(Event::kCopyFromDevice);
        if (fake.fail_now()) return 79;
        if (bytes == sizeof(transport::DeviceTransportDiagnostic)) {
            auto* diagnostic =
                static_cast<transport::DeviceTransportDiagnostic*>(destination);
            diagnostic->error =
                transport::DeviceTransportError::kCompletionTimeout;
            diagnostic->generation = 17;
        } else {
            CHECK(bytes == sizeof(transport::TransportStageProfile));
            CHECK(const_cast<void*>(source) == fake.profile_device_pointer);
            auto* profile =
                static_cast<transport::TransportStageProfile*>(destination);
            *profile = fake.profile;
            profile->operation = transport::TransportProfileOperation::kCombine;
            profile->generation = 23;
            profile->completion_generation = 23;
        }
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

transport::TransportConfig valid_config(int rank = 1, int world_size = 2) {
    transport::TransportConfig config;
    config.rank = rank;
    config.world_size = world_size;
    config.communicator_handle = 0x1234;
    config.device_buffer_bytes = 4096;
    config.requested_channels = 1;
    return config;
}

void check_rank_sized_command_queue() {
    struct Fixture {
        std::uint32_t world_size;
        std::uint32_t command_capacity;
    };
    for (const auto fixture : {
             Fixture{2, 6}, Fixture{4, 16}, Fixture{8, 36}}) {
        FakeApi fake;
        fake.rank = fixture.world_size - 1;
        fake.size = fixture.world_size;
        auto created = transport::make_cann_transport(
            valid_config(
                static_cast<int>(fake.rank),
                static_cast<int>(fake.size)),
            fake.api());
        CHECK(created.status.ok());
        CHECK(created.transport != nullptr);
        alignas(64) std::uint8_t window[4096]{};
        CHECK(created.transport->register_symmetric_window(
            window, sizeof(window)).ok());
        CHECK(created.transport->acquire_channels(
            1, transport::CooperationScope::kParticipant).ok());
        transport::DeviceTransportContext context{};
        CHECK(created.transport->export_device_context(&context).ok());
        CHECK(!transport::has_capability(
            context.capabilities,
            transport::TransportCapability::kStageProfile));
        CHECK(fake.allocation_count == 5);
        CHECK(fake.allocation_bytes[0] ==
              static_cast<std::uint64_t>(fixture.command_capacity) *
                  sizeof(transport::TransportCommand));
        CHECK(fake.queue_copy_count == 1);
        CHECK(fake.queue.capacity == fixture.command_capacity);
        CHECK(fake.staged.stage_profile == 0);
        CHECK(fake.staged.stage_profile_bytes == 0);
        CHECK(created.transport->destroy().ok());
    }

    std::uint32_t capacity = 0;
    CHECK(transport::checked_scale_up_command_capacity(2, &capacity));
    CHECK(capacity == 6);
    CHECK(transport::checked_scale_up_command_capacity(4, &capacity));
    CHECK(capacity == 16);

    constexpr int kLargestRepresentableWorldSize = 858993459;
    CHECK(transport::checked_scale_up_command_capacity(
        kLargestRepresentableWorldSize, &capacity));
    CHECK(capacity == 4294967291U);
    capacity = 0x12345678U;
    CHECK(!transport::checked_scale_up_command_capacity(
        kLargestRepresentableWorldSize + 1, &capacity));
    CHECK(capacity == 0x12345678U);
    CHECK(!transport::checked_scale_up_command_capacity(
        std::numeric_limits<int>::max(), &capacity));

    transport::TransportCommand commands[6]{};
    transport::TransportServiceState service{};
    transport::DeviceTransportDiagnostic diagnostic{};
    CHECK(transport::checked_scale_up_command_capacity(2, &capacity));
    auto queue = transport::command::make_queue(
        commands, capacity, &service, &diagnostic);
    CHECK(transport::command::append(
        queue, transport::command::make_put(
            transport::TransportTeam::kScaleUp, 1, 1, 0, 0x1000, 0x2000,
            64, transport::CooperationScope::kParticipant,
            transport::MemorySegment::kDevice, transport::kDefaultOptions)));
    CHECK(transport::command::append(
        queue, transport::command::make_flush(
            0, transport::CooperationScope::kDevice)));
    CHECK(transport::command::append(
        queue, transport::command::make_put_value64(
            transport::TransportTeam::kScaleUp, 1, 1, 0, 0x3000, 3,
            transport::kDefaultOptions)));
    CHECK(transport::command::append(
        queue, transport::command::make_put_value64(
            transport::TransportTeam::kScaleUp, 1, 1, 0, 0x3008, 7,
            transport::kDefaultOptions)));
    CHECK(transport::command::append(
        queue, transport::command::make_signal(
            transport::TransportTeam::kScaleUp, 1, 1, 0,
            transport::RemoteAction::signal_set(
                transport::sync_layout::kDispatchReleaseSignalIndex, 7))));
    CHECK(transport::command::append(
        queue, transport::command::make_barrier(
            transport::kWorldTeamMask, 1000)));
    CHECK(queue.count == capacity);
}

void check_explicit_two_dimensional_topology() {
    FakeApi fake;
    fake.rank = 3;
    fake.size = 4;
    auto config = valid_config(3, 4);
    config.scale_up_size = 2;
    config.topology_kind =
        transport::TransportTopologyKind::kLogicalSimulation;
    config.topology_epoch = 17;
    config.allow_hybrid_mode = true;
    auto created = transport::make_cann_transport(config, fake.api());
    CHECK(created.status.ok());
    transport::TransportTopology topology{};
    CHECK(created.transport->query_topology(&topology).ok());
    CHECK(topology.kind ==
          transport::TransportTopologyKind::kLogicalSimulation);
    CHECK(topology.epoch == 17);
    CHECK(topology.world_rank == 3 && topology.world_size == 4);
    CHECK(topology.scale_up_rank == 1 && topology.scale_up_size == 2);
    CHECK(topology.scale_out_rank == 1 && topology.scale_out_size == 2);
    CHECK(!transport::has_capability(
        created.transport->capabilities(),
        transport::TransportCapability::kScaleOutTeam));
    CHECK(created.transport->destroy().ok());

    for (const auto scale_up_size : {0, 3, 4}) {
        FakeApi invalid_fake;
        invalid_fake.rank = 0;
        invalid_fake.size = 4;
        auto invalid = valid_config(0, 4);
        invalid.scale_up_size = scale_up_size;
        invalid.topology_kind =
            transport::TransportTopologyKind::kLogicalSimulation;
        auto rejected = transport::make_cann_transport(
            invalid, invalid_fake.api());
        CHECK(!rejected.status.ok());
        CHECK(rejected.transport == nullptr);
    }
}

void check_communicator_size_query() {
    FakeApi fake;
    fake.size = 3;
    std::uint32_t world_size = 0;
    auto status = transport::query_cann_communicator_size(
        0x1234, &world_size, fake.api());
    CHECK(status.ok());
    CHECK(world_size == 3);

    status = transport::query_cann_communicator_size(
        0, &world_size, fake.api());
    CHECK(!status.ok());
    CHECK(status.operation == "query_communicator_size");

    fake.fail_event = 0;
    fake.event_calls = 0;
    status = transport::query_cann_communicator_size(
        0x1234, &world_size, fake.api());
    CHECK(!status.ok());
    CHECK(status.backend_code == 72);
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
    auto config = valid_config();
    config.stage_profile_enabled = true;
    auto created = transport::make_cann_transport(config, fake.api());
    CHECK(created.status.ok());
    CHECK(created.transport != nullptr);
    CHECK(created.transport->capabilities() ==
          (kValidatedCapabilities |
           transport::capability_bit(
               transport::TransportCapability::kStageProfile)));

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
    CHECK(context.capabilities ==
          (kValidatedCapabilities |
           transport::capability_bit(
               transport::TransportCapability::kStageProfile)));
    CHECK(transport::has_capability(
        context.capabilities,
        transport::TransportCapability::kStageProfile));
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
    CHECK(fake.staged.stage_profile != 0);
    CHECK(fake.staged.stage_profile_bytes ==
          sizeof(transport::TransportStageProfile));
    CHECK(fake.staged.reserved == transport::command::registration_cookie(
        fake.staged.command_queue, fake.queue.commands,
        fake.queue.service_state, fake.queue.diagnostic,
        fake.queue.capacity));
    CHECK(fake.diagnostic_copy_count == 1);
    CHECK(fake.diagnostic.abi_version ==
          transport::kTransportCommandAbiVersion);
    CHECK(fake.diagnostic.error == transport::DeviceTransportError::kNone);
    CHECK(fake.count(Event::kAllocate) == 6);
    CHECK(fake.count(Event::kZero) == 6);
    CHECK(fake.count(Event::kCopy) == 4);

    CHECK(created.transport->reset_stage_profile().ok());
    CHECK(fake.profile_copy_count == 1);
    CHECK(fake.profile.abi_version ==
          transport::kTransportStageProfileAbiVersion);
    CHECK(fake.profile.operation == transport::TransportProfileOperation::kNone);
    transport::TransportStageProfile profile{};
    CHECK(created.transport->read_stage_profile(&profile).ok());
    CHECK(profile.operation == transport::TransportProfileOperation::kCombine);
    CHECK(profile.generation == 23);
    CHECK(profile.completion_generation == 23);
    CHECK(fake.count(Event::kCopyFromDevice) == 2);

    CHECK(created.transport->destroy().ok());
    const auto after_first_destroy = fake.event_count;
    CHECK(created.transport->destroy().ok());
    CHECK(fake.event_count == after_first_destroy);
    CHECK(fake.count(Event::kFree) == 6);
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
    check_communicator_size_query();
    check_rank_sized_command_queue();
    check_explicit_two_dimensional_topology();
    check_success_and_reverse_cleanup();
    check_partial_failure_cleans_up();
    check_deregister_failure_is_retryable();
    check_team_destroy_failure_is_retryable();
    return failures == 0 ? 0 : 1;
}
