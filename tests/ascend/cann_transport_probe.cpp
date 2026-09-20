#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "csrc/backends/ascend/transport/cann_transport.hpp"
#include "csrc/backends/ascend/transport/sync_layout.hpp"
#include "csrc/backends/ascend/transport/transport_commands.hpp"

namespace transport = deep_ep::ascend::transport;

namespace {

int failures = 0;

#define CHECK(expression)                                                            \
    do {                                                                             \
        if (!(expression)) {                                                         \
            std::cerr << __FILE__ << ':' << __LINE__ << ": " << #expression << '\n'; \
            ++failures;                                                              \
        }                                                                            \
    } while (false)

enum class Event : std::uint32_t {
    kGetRank,
    kGetSize,
    kRegisterMemory,
    kAcquireChannels,
    kGetRemoteMemory,
    kAllocate,
    kZero,
    kCopy,
    kCopyFromDevice,
    kFree,
    kDestroyChannels,
    kDeregisterMemory,
};

struct FakeApi {
    Event events[128]{};
    std::uint32_t event_count = 0;
    std::uint32_t rank = 1;
    std::uint32_t size = 2;
    std::uintptr_t next_pointer = 0x100000;
    std::uint64_t window_bytes = 4096;
    std::uint64_t sync_bytes = transport::sync_layout::sync_window_bytes(2);
    std::uint32_t command_capacity = 0;
    std::uint64_t command_bytes = transport::checked_scale_up_command_capacity(size, channel_count, &command_capacity)
        ? command_capacity * sizeof(transport::TransportCommand)
        : 0;
    int fail_event = -1;
    int event_calls = 0;
    int destroy_channel_failures_remaining = 0;
    int deregister_failures_remaining = 0;
    transport::StagedTransportContext staged{};
    std::uint32_t staged_copy_count = 0;
    transport::TransportCommandQueue queue{};
    std::uint32_t queue_copy_count = 0;
    transport::DeviceTransportDiagnostic diagnostic{};
    std::uint32_t diagnostic_copy_count = 0;
    transport::TransportStageProfile profile{};
    std::uint32_t profile_copy_count = 0;
    void* profile_device_pointer = nullptr;
    std::uint64_t allocation_bytes[16]{};
    std::uint32_t allocation_count = 0;
    std::uint32_t channel_count = 1;

    bool fail_now() { return fail_event >= 0 && event_calls++ == fail_event; }

    void record(Event event) { events[event_count++] = event; }

    std::uint32_t count(Event event) const {
        std::uint32_t result = 0;
        for (std::uint32_t index = 0; index < event_count; ++index)
            result += events[index] == event ? 1U : 0U;
        return result;
    }

    std::uint32_t first(Event event) const {
        for (std::uint32_t index = 0; index < event_count; ++index)
            if (events[index] == event)
                return index;
        return event_count;
    }

    static FakeApi& self(void* data) { return *static_cast<FakeApi*>(data); }

    static int get_rank(void* data, std::int64_t, std::uint32_t* rank) {
        auto& fake = self(data);
        fake.record(Event::kGetRank);
        if (fake.fail_now())
            return 71;
        *rank = fake.rank;
        return 0;
    }

    static int get_size(void* data, std::int64_t, std::uint32_t* size) {
        auto& fake = self(data);
        fake.record(Event::kGetSize);
        if (fake.fail_now())
            return 72;
        *size = fake.size;
        return 0;
    }

    static int register_memory(void* data, std::int64_t, const char* tag, void*, std::uint64_t bytes, std::uintptr_t* memory) {
        auto& fake = self(data);
        fake.record(Event::kRegisterMemory);
        CHECK(std::string(tag) == "DeepEPUbcBuffer" || std::string(tag) == "DeepEPUbcSync" || std::string(tag) == "DeepEPUbcCommands");
        CHECK(bytes ==
              (std::string(tag) == "DeepEPUbcBuffer"     ? fake.window_bytes
                   : std::string(tag) == "DeepEPUbcSync" ? fake.sync_bytes
                                                         : fake.command_bytes));
        if (fake.fail_now())
            return 73;
        *memory = std::string(tag) == "DeepEPUbcBuffer" ? 0x300000 : std::string(tag) == "DeepEPUbcSync" ? 0x300100 : 0x300200;
        return 0;
    }

    static int acquire_channels(void* data,
                                std::int64_t,
                                std::uint32_t rank,
                                std::uint32_t peer,
                                const std::uintptr_t* memories,
                                std::uint32_t memory_count,
                                std::uint32_t count,
                                std::uintptr_t* channels) {
        auto& fake = self(data);
        fake.record(Event::kAcquireChannels);
        CHECK(rank == fake.rank);
        CHECK(peer != rank);
        CHECK(memories != nullptr && memory_count == 3);
        CHECK(memories[0] == 0x300000 && memories[1] == 0x300100 && memories[2] == 0x300200);
        CHECK(count == fake.channel_count);
        if (fake.fail_now())
            return 74;
        for (std::uint32_t index = 0; index < count; ++index)
            channels[index] = 0x400000 + index;
        return 0;
    }

    static int get_remote_memory(void* data, std::int64_t, std::uintptr_t, const char* tag, std::uintptr_t* address, std::uint64_t* bytes) {
        auto& fake = self(data);
        fake.record(Event::kGetRemoteMemory);
        if (fake.fail_now())
            return 75;
        if (std::string(tag) == "DeepEPUbcBuffer") {
            *address = 0x500000;
            *bytes = fake.window_bytes;
        } else {
            *address = 0x600000;
            *bytes = fake.sync_bytes;
        }
        return 0;
    }

    static int allocate(void* data, std::uint64_t bytes, void** pointer) {
        auto& fake = self(data);
        fake.record(Event::kAllocate);
        if (fake.fail_now())
            return 76;
        CHECK(fake.allocation_count < 16);
        fake.allocation_bytes[fake.allocation_count++] = bytes;
        *pointer = reinterpret_cast<void*>(fake.next_pointer);
        fake.next_pointer += 0x1000;
        return 0;
    }

    static int zero(void* data, void*, std::uint64_t) {
        auto& fake = self(data);
        fake.record(Event::kZero);
        if (fake.fail_now())
            return 77;
        return 0;
    }

    static int copy(void* data, void* destination, const void* source, std::uint64_t bytes) {
        auto& fake = self(data);
        fake.record(Event::kCopy);
        if (fake.fail_now())
            return 78;
        if (bytes == sizeof(transport::StagedTransportContext)) {
            const auto* candidate = static_cast<const transport::StagedTransportContext*>(source);
            if (candidate->struct_size == sizeof(transport::StagedTransportContext) &&
                candidate->cann_compatibility == transport::kStagedTransportCannCompatibility) {
                fake.staged = *candidate;
                ++fake.staged_copy_count;
            }
        }
        if (bytes == sizeof(transport::DeviceChannelTable)) {
            const auto* table = static_cast<const transport::DeviceChannelTable*>(source);
            CHECK(table->member_count == fake.size);
            CHECK(table->self_member == fake.rank);
            CHECK(table->channel_count == fake.channel_count);
        }
        if (destination == reinterpret_cast<void*>(0x106000) && bytes == sizeof(transport::TransportCommandQueue)) {
            fake.queue = *static_cast<const transport::TransportCommandQueue*>(source);
            ++fake.queue_copy_count;
        }
        if (destination == reinterpret_cast<void*>(0x103000) && bytes == sizeof(transport::DeviceTransportDiagnostic)) {
            fake.diagnostic = *static_cast<const transport::DeviceTransportDiagnostic*>(source);
            ++fake.diagnostic_copy_count;
        }
        if (bytes == sizeof(transport::TransportStageProfile)) {
            fake.profile = *static_cast<const transport::TransportStageProfile*>(source);
            fake.profile_device_pointer = destination;
            ++fake.profile_copy_count;
        }
        return 0;
    }

    static int copy_from_device(void* data, void* destination, const void* source, std::uint64_t bytes) {
        auto& fake = self(data);
        fake.record(Event::kCopyFromDevice);
        if (fake.fail_now())
            return 79;
        if (bytes == sizeof(transport::DeviceTransportDiagnostic)) {
            auto* diagnostic = static_cast<transport::DeviceTransportDiagnostic*>(destination);
            diagnostic->error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic->generation = 17;
        } else {
            CHECK(bytes == sizeof(transport::TransportStageProfile));
            CHECK(const_cast<void*>(source) == fake.profile_device_pointer);
            auto* profile = static_cast<transport::TransportStageProfile*>(destination);
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

    static int destroy_channels(void* data, std::int64_t, const std::uintptr_t*, std::uint32_t) {
        auto& fake = self(data);
        fake.record(Event::kDestroyChannels);
        if (fake.destroy_channel_failures_remaining > 0) {
            --fake.destroy_channel_failures_remaining;
            return 81;
        }
        return 0;
    }

    static int deregister_memory(void* data, std::int64_t, const char* tag, std::uintptr_t) {
        auto& fake = self(data);
        fake.record(Event::kDeregisterMemory);
        CHECK(std::string(tag) == "DeepEPUbcBuffer" || std::string(tag) == "DeepEPUbcSync" || std::string(tag) == "DeepEPUbcCommands");
        if (fake.deregister_failures_remaining > 0) {
            --fake.deregister_failures_remaining;
            return 82;
        }
        return 0;
    }

    transport::CannHostApi api() {
        return {
            this,
            get_rank,
            get_size,
            register_memory,
            acquire_channels,
            get_remote_memory,
            allocate,
            zero,
            copy,
            copy_from_device,
            free,
            destroy_channels,
            deregister_memory,
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

void check_communicator_size_query() {
    FakeApi fake;
    std::uint32_t world_size = 0;
    CHECK(transport::query_cann_communicator_size(0x1234, &world_size, fake.api()).ok());
    CHECK(world_size == 2);

    fake.fail_event = 0;
    fake.event_calls = 0;
    const auto status = transport::query_cann_communicator_size(0x1234, &world_size, fake.api());
    CHECK(!status.ok());
    CHECK(status.backend_code == 72);
}

std::unique_ptr<transport::HostTransport> make_active_transport(FakeApi& fake, bool profile = false) {
    auto config = valid_config(static_cast<int>(fake.rank), static_cast<int>(fake.size));
    config.stage_profile_enabled = profile;
    auto created = transport::make_cann_transport(config, fake.api());
    CHECK(created.status.ok());
    CHECK(created.transport != nullptr);
    alignas(64) static std::uint8_t window[4096]{};
    CHECK(created.transport->register_symmetric_window(window, sizeof(window)).ok());
    CHECK(created.transport->acquire_channels(static_cast<int>(fake.channel_count), transport::CooperationScope::kParticipant).ok());
    return std::move(created.transport);
}

void check_success_and_context() {
    FakeApi fake;
    fake.channel_count = 1;
    auto transport = make_active_transport(fake, true);
    transport::DeviceTransportContext context{};
    CHECK(transport->export_device_context(&context).ok());
    CHECK(context.local_window_base != 0);
    CHECK(context.channel_table == context.peer_address_table);
    CHECK(context.channel_table != 0);
    CHECK(context.channel_count == 1);
    CHECK(context.backend_context != 0);
    CHECK(fake.count(Event::kRegisterMemory) == 3);
    CHECK(fake.count(Event::kAcquireChannels) == 1);
    CHECK(fake.count(Event::kGetRemoteMemory) == 2);
    CHECK(fake.staged_copy_count == 1);
    CHECK(fake.queue_copy_count == 1);
    CHECK(transport->destroy().ok());
}

void check_failure_cleanup() {
    for (int fail_event = 0; fail_event < 22; ++fail_event) {
        FakeApi fake;
        fake.fail_event = fail_event;
        auto created = transport::make_cann_transport(valid_config(), fake.api());
        if (!created.status.ok()) {
            CHECK(created.transport == nullptr);
            continue;
        }
        alignas(64) static std::uint8_t window[4096]{};
        if (created.transport->register_symmetric_window(window, sizeof(window)).ok())
            (void)created.transport->acquire_channels(1, transport::CooperationScope::kParticipant);
        (void)created.transport->destroy();
        const auto allocations = static_cast<std::uint32_t>((fake.next_pointer - 0x100000) / 0x1000);
        CHECK(fake.count(Event::kFree) <= allocations);
    }
}

void check_destroy_channel_failure_is_retryable() {
    FakeApi fake;
    auto transport = make_active_transport(fake);
    // CANN 9.3 only supports HcclChannelDestroy for CCU channels.  AIV
    // channels are communicator-owned, so teardown only releases staged
    // resources and stops retaining the handles.
    CHECK(transport->destroy().ok());
    CHECK(fake.count(Event::kDestroyChannels) == 0);
}

}  // namespace

int main() {
    check_communicator_size_query();
    check_success_and_context();
    check_failure_cleanup();
    check_destroy_channel_failure_is_retryable();
    return failures == 0 ? 0 : 1;
}
