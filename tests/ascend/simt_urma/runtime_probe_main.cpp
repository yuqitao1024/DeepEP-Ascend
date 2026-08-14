#include "runtime_probe.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include "acl/acl.h"

#include "csrc/backends/ascend/transport/cann_compat.hpp"
#include "csrc/backends/ascend/transport/cann_transport.hpp"

namespace transport = deep_ep::ascend::transport;
namespace probe = deep_ep::ascend::transport::runtime_probe;

namespace {

constexpr std::uint64_t kWindowBytes = 64 * 1024;
constexpr std::uint32_t kBatchOperations = 64;

void write_error(char* output, std::size_t capacity, const char* format, ...) {
    if (output == nullptr || capacity == 0)
        return;
    va_list arguments;
    va_start(arguments, format);
    (void)std::vsnprintf(output, capacity, format, arguments);
    va_end(arguments);
}

bool check_acl(
    aclError result, const char* operation, char* error,
    std::size_t error_capacity) {
    if (result == ACL_SUCCESS)
        return true;
    write_error(
        error, error_capacity, "%s failed with ACL status %d", operation,
        static_cast<int>(result));
    return false;
}

bool parse_case(const char* name, probe::RuntimeCase* runtime_case) {
    if (name == nullptr || runtime_case == nullptr)
        return false;
    struct Entry {
        const char* name;
        probe::RuntimeCase value;
    };
    const Entry entries[] = {
        {"put", probe::RuntimeCase::kPut},
        {"put-value64", probe::RuntimeCase::kPutValue64},
        {"faa64", probe::RuntimeCase::kFaa64},
        {"signal", probe::RuntimeCase::kSignal},
        {"flush", probe::RuntimeCase::kFlush},
        {"payload-signal-order", probe::RuntimeCase::kPayloadSignalOrder},
        {"barrier-repeat", probe::RuntimeCase::kBarrierRepeat},
        {"queue-wrap", probe::RuntimeCase::kQueueWrap},
        {"phase-boundary", probe::RuntimeCase::kPhaseBoundary},
    };
    for (const auto& entry : entries) {
        if (std::strcmp(name, entry.name) == 0) {
            *runtime_case = entry.value;
            return true;
        }
    }
    return false;
}

bool copy_from_device(
    void* destination, const void* source, std::size_t bytes,
    const char* operation, char* error, std::size_t error_capacity) {
    return check_acl(
        aclrtMemcpy(destination, bytes, source, bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST),
        operation, error, error_capacity);
}

std::uint32_t inspect_sq_depth(
    const transport::DeviceTransportContext& context, std::uint32_t peer,
    char* error, std::size_t error_capacity) {
    transport::cann_abi::Team team{};
    if (!copy_from_device(
            &team, reinterpret_cast<const void*>(context.channel_table),
            sizeof(team), "copy team", error, error_capacity))
        return 0;
    if (peer >= team.member_count || team.channel_counts == 0 ||
        team.channels == 0) {
        write_error(error, error_capacity, "invalid team channel table");
        return 0;
    }
    std::uint32_t counts[64]{};
    if (team.member_count > 64 ||
        !copy_from_device(
            counts, reinterpret_cast<const void*>(team.channel_counts),
            team.member_count * sizeof(std::uint32_t), "copy channel counts",
            error, error_capacity))
        return 0;
    if (counts[peer] == 0) {
        write_error(error, error_capacity, "peer has no channel");
        return 0;
    }
    std::uint32_t channel_index = 0;
    for (std::uint32_t member = 0; member < peer; ++member)
        channel_index += counts[member];
    transport::cann_abi::Channel channel{};
    const auto channel_address = team.channels +
        static_cast<std::uint64_t>(channel_index) * sizeof(channel);
    if (!copy_from_device(
            &channel, reinterpret_cast<const void*>(channel_address),
            sizeof(channel), "copy channel", error, error_capacity) ||
        channel.sq_contexts == 0 || channel.sq_count == 0) {
        if (error != nullptr && error[0] == '\0')
            write_error(error, error_capacity, "invalid SQ table");
        return 0;
    }
    transport::cann_abi::SqContext sq{};
    if (!copy_from_device(
            &sq, reinterpret_cast<const void*>(channel.sq_contexts),
            sizeof(sq), "copy SQ context", error, error_capacity))
        return 0;
    return sq.depth;
}

class RuntimeResources {
public:
    ~RuntimeResources() {
        if (transport_ != nullptr)
            (void)transport_->destroy();
        if (stream_ != nullptr)
            (void)aclrtDestroyStream(stream_);
        if (window_ != nullptr)
            (void)aclrtFree(window_);
    }

    bool initialize(
        std::int64_t communicator_handle, std::uint32_t rank,
        std::uint32_t world_size, char* error,
        std::size_t error_capacity) {
        if (!check_acl(
                aclrtMalloc(
                    &window_, kWindowBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                "allocate window", error, error_capacity) ||
            !check_acl(
                aclrtMemset(window_, kWindowBytes, 0, kWindowBytes),
                "zero window", error, error_capacity) ||
            !check_acl(
                aclrtCreateStream(&stream_), "create stream", error,
                error_capacity))
            return false;

        transport::TransportConfig config;
        config.rank = static_cast<int>(rank);
        config.world_size = static_cast<int>(world_size);
        config.communicator_handle = communicator_handle;
        config.device_buffer_bytes = kWindowBytes;
        config.requested_channels = 1;
        auto created = transport::make_cann_transport(config);
        if (!created.status.ok()) {
            write_error(
                error, error_capacity, "%s failed: backend=%d %s",
                created.status.operation.c_str(), created.status.backend_code,
                created.status.message.c_str());
            return false;
        }
        transport_ = std::move(created.transport);
        auto status = transport_->register_symmetric_window(
            window_, kWindowBytes);
        if (status.ok())
            status = transport_->acquire_channels(
                1, transport::CooperationScope::kParticipant);
        if (status.ok())
            status = transport_->export_device_context(&context_);
        if (!status.ok()) {
            write_error(
                error, error_capacity, "%s failed: backend=%d %s",
                status.operation.c_str(), status.backend_code,
                status.message.c_str());
            return false;
        }
        return true;
    }

    bool run(
        probe::RuntimeCase runtime_case, std::uint32_t rank,
        std::uint32_t world_size, std::uint64_t iterations,
        char* error, std::size_t error_capacity) {
        const std::uint32_t peer = (rank + 1) % world_size;
        std::uint64_t launch_count = std::max<std::uint64_t>(iterations, 1);
        std::uint32_t operations = 1;
        if (runtime_case == probe::RuntimeCase::kQueueWrap) {
            const auto depth = inspect_sq_depth(
                context_, peer, error, error_capacity);
            if (depth == 0)
                return false;
            launch_count = (static_cast<std::uint64_t>(depth) +
                            kBatchOperations - 1) /
                           kBatchOperations + 1;
            operations = kBatchOperations;
        }

        auto* device_state = static_cast<probe::RuntimeState*>(window_);
        for (std::uint64_t launch = 0; launch < launch_count; ++launch) {
            const std::uint64_t generation = launch + 1;
            probe::RuntimeState state;
            state.source =
                (static_cast<std::uint64_t>(rank + 1) << 32U) | generation;
            state.generation = generation;
            if (!check_acl(
                    aclrtMemcpy(
                        device_state, sizeof(state), &state, sizeof(state),
                        ACL_MEMCPY_HOST_TO_DEVICE),
                    "initialize runtime state", error, error_capacity))
                return false;
            const int launch_status = deep_ep_ascend_urma_launch_runtime_probe(
                device_state, context_, runtime_case, peer, generation,
                operations, stream_);
            if (launch_status != 0) {
                write_error(
                    error, error_capacity, "kernel launch failed with %d",
                    launch_status);
                return false;
            }
            if (!check_acl(
                    aclrtSynchronizeStream(stream_), "synchronize stream",
                    error, error_capacity) ||
                !copy_from_device(
                    &state, device_state, sizeof(state), "copy runtime state",
                    error, error_capacity))
                return false;
            if (state.success != 1 ||
                state.diagnostic.error != transport::DeviceTransportError::kNone) {
                write_error(
                    error, error_capacity,
                    "semantic failure: case=%u generation=%llu success=%u "
                    "diagnostic=%u opcode=%u peer=%u channel=%u backend=%u "
                    "observed=0x%llx",
                    static_cast<unsigned>(runtime_case),
                    static_cast<unsigned long long>(generation), state.success,
                    static_cast<unsigned>(state.diagnostic.error),
                    static_cast<unsigned>(state.diagnostic.opcode),
                    state.diagnostic.peer, state.diagnostic.channel,
                    state.diagnostic.backend_status,
                    static_cast<unsigned long long>(state.observed));
                return false;
            }
        }
        return true;
    }

private:
    void* window_ = nullptr;
    aclrtStream stream_ = nullptr;
    std::unique_ptr<transport::HostTransport> transport_;
    transport::DeviceTransportContext context_{};
};

}  // namespace

extern "C" int deep_ep_ascend_urma_run_case(
    std::int64_t communicator_handle, std::uint32_t rank,
    std::uint32_t world_size, const char* case_name,
    std::uint64_t iterations, char* error, std::size_t error_capacity) {
    if (error != nullptr && error_capacity != 0)
        error[0] = '\0';
    if (communicator_handle == 0 || world_size != 2 || rank >= world_size) {
        write_error(error, error_capacity, "invalid communicator topology");
        return 2;
    }

    const bool teardown = case_name != nullptr &&
        std::strcmp(case_name, "teardown") == 0;
    probe::RuntimeCase runtime_case{};
    if (!teardown && !parse_case(case_name, &runtime_case)) {
        write_error(error, error_capacity, "unknown case: %s",
                    case_name == nullptr ? "<null>" : case_name);
        return 2;
    }

    RuntimeResources resources;
    if (!resources.initialize(
            communicator_handle, rank, world_size, error, error_capacity))
        return 1;
    if (!teardown && !resources.run(
            runtime_case, rank, world_size, iterations, error,
            error_capacity))
        return 1;
    return 0;
}

extern "C" int deep_ep_ascend_urma_run_local_phase_boundary(
    char* error, std::size_t error_capacity) {
    if (error != nullptr && error_capacity != 0)
        error[0] = '\0';
    probe::RuntimeState* device_state = nullptr;
    aclrtStream stream = nullptr;
    bool success = check_acl(
        aclrtMalloc(
            reinterpret_cast<void**>(&device_state), sizeof(*device_state),
            ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate local smoke state", error, error_capacity);
    if (success) {
        success = check_acl(
            aclrtMemset(
                device_state, sizeof(*device_state), 0,
                sizeof(*device_state)),
            "zero local smoke state", error, error_capacity);
    }
    if (success) {
        success = check_acl(
            aclrtCreateStream(&stream), "create local smoke stream", error,
            error_capacity);
    }
    transport::DeviceTransportContext context =
        transport::make_device_transport_context();
    context.topology.world_size = 1;
    context.topology.scale_up_size = 1;
    context.topology.scale_out_size = 1;
    if (success) {
        const int launch_status = deep_ep_ascend_urma_launch_runtime_probe(
            device_state, context, probe::RuntimeCase::kPhaseBoundary, 0, 1,
            1, stream);
        if (launch_status != 0) {
            write_error(
                error, error_capacity, "local smoke launch failed with %d",
                launch_status);
            success = false;
        }
    }
    if (success) {
        success = check_acl(
            aclrtSynchronizeStream(stream), "synchronize local smoke", error,
            error_capacity);
    }
    probe::RuntimeState state;
    if (success) {
        success = copy_from_device(
            &state, device_state, sizeof(state), "copy local smoke state",
            error, error_capacity);
    }
    if (success && state.success != 1) {
        write_error(
            error, error_capacity,
            "local phase boundary failed: sequence=%u observed=0x%llx",
            state.phase_sequence,
            static_cast<unsigned long long>(state.observed));
        success = false;
    }
    if (stream != nullptr)
        (void)aclrtDestroyStream(stream);
    if (device_state != nullptr)
        (void)aclrtFree(device_state);
    return success ? 0 : 1;
}
