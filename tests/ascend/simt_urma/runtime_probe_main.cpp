#include "runtime_probe.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "acl/acl.h"

#include "csrc/backends/ascend/transport/cann_compat.hpp"
#include "csrc/backends/ascend/transport/cann_transport.hpp"
#include "csrc/backends/ascend/transport/topology_config.hpp"

namespace transport = deep_ep::ascend::transport;
namespace probe = deep_ep::ascend::transport::runtime_probe;

namespace {

constexpr std::uint64_t kWindowBytes = 64 * 1024;

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
        {"signal-set", probe::RuntimeCase::kSignalSet},
        {"flush", probe::RuntimeCase::kFlush},
        {"payload-signal-order", probe::RuntimeCase::kPayloadSignalOrder},
        {"barrier-repeat", probe::RuntimeCase::kBarrierRepeat},
        {"queue-wrap", probe::RuntimeCase::kQueueWrap},
        {"profile-mixed", probe::RuntimeCase::kProfileMixed},
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

void dump_transport_descriptors(
    const transport::DeviceTransportContext& context, std::uint32_t peer) {
    char error[256]{};
    transport::cann_abi::Team team{};
    transport::cann_abi::Window window{};
    if (!copy_from_device(
            &team, reinterpret_cast<const void*>(context.channel_table),
            sizeof(team), "copy debug team", error, sizeof(error)) ||
        !copy_from_device(
            &window, reinterpret_cast<const void*>(context.peer_address_table),
            sizeof(window), "copy debug window", error, sizeof(error))) {
        std::fprintf(stderr, "URMA-DESC error=%s\n", error);
        return;
    }
    std::fprintf(
        stderr,
        "URMA-DESC team members=%u self=%u signals=%u counters=%u "
        "barriers=%u sync_bytes=%llu remote_sync_count=%u "
        "shadow=[0x%llx,+%llu] window_count=%u\n",
        team.member_count, team.self_member, team.signal_count,
        team.counter_count, team.barrier_count,
        static_cast<unsigned long long>(team.sync_memory_bytes),
        team.remote_sync_memory_count,
        static_cast<unsigned long long>(team.shadow_sync_memory.address),
        static_cast<unsigned long long>(team.shadow_sync_memory.bytes),
        window.memory_count);

    transport::cann_abi::Memory memories[64]{};
    if (team.remote_sync_memory_count <= 64 &&
        team.remote_sync_memories != 0 &&
        copy_from_device(
            memories, reinterpret_cast<const void*>(team.remote_sync_memories),
            team.remote_sync_memory_count * sizeof(memories[0]),
            "copy debug sync memories", error, sizeof(error))) {
        for (std::uint32_t index = 0;
             index < team.remote_sync_memory_count; ++index) {
            std::fprintf(
                stderr, "URMA-DESC sync[%u]=[0x%llx,+%llu]\n", index,
                static_cast<unsigned long long>(memories[index].address),
                static_cast<unsigned long long>(memories[index].bytes));
        }
    }
    if (window.memory_count <= 64 && window.memories != 0 &&
        copy_from_device(
            memories, reinterpret_cast<const void*>(window.memories),
            window.memory_count * sizeof(memories[0]),
            "copy debug window memories", error, sizeof(error))) {
        for (std::uint32_t index = 0; index < window.memory_count; ++index) {
            std::fprintf(
                stderr, "URMA-DESC window[%u]=[0x%llx,+%llu]\n", index,
                static_cast<unsigned long long>(memories[index].address),
                static_cast<unsigned long long>(memories[index].bytes));
        }
    }

    std::uint32_t counts[64]{};
    if (team.member_count > 64 || team.channel_counts == 0 ||
        !copy_from_device(
            counts, reinterpret_cast<const void*>(team.channel_counts),
            team.member_count * sizeof(counts[0]), "copy debug channel counts",
            error, sizeof(error)))
        return;
    std::uint32_t channel_index = 0;
    for (std::uint32_t member = 0; member < peer; ++member)
        channel_index += counts[member];
    transport::cann_abi::Channel channel{};
    if (peer >= team.member_count || counts[peer] == 0 || team.channels == 0 ||
        !copy_from_device(
            &channel,
            reinterpret_cast<const void*>(
                team.channels + static_cast<std::uint64_t>(channel_index) *
                    sizeof(channel)),
            sizeof(channel), "copy debug channel", error, sizeof(error)))
        return;
    std::fprintf(
        stderr,
        "URMA-DESC channel peer=%u local_buffers=%u remote_buffers=%u\n",
        peer, channel.local_buffer_count, channel.remote_buffer_count);
    if (channel.sq_count != 0 && channel.cq_count != 0 &&
        channel.sq_contexts != 0 && channel.cq_contexts != 0) {
        transport::cann_abi::SqContext sq{};
        transport::cann_abi::CqContext cq{};
        if (copy_from_device(
                &sq, reinterpret_cast<const void*>(channel.sq_contexts),
                sizeof(sq), "copy debug SQ context", error, sizeof(error)) &&
            copy_from_device(
                &cq, reinterpret_cast<const void*>(channel.cq_contexts),
                sizeof(cq), "copy debug CQ context", error, sizeof(error))) {
            std::fprintf(
                stderr,
                "URMA-DESC sq base=0x%llx head_ptr=0x%llx "
                "tail_ptr=0x%llx db=0x%llx entry=%u depth=%u\n",
                static_cast<unsigned long long>(sq.base),
                static_cast<unsigned long long>(sq.head),
                static_cast<unsigned long long>(sq.tail),
                static_cast<unsigned long long>(sq.doorbell), sq.entry_bytes,
                sq.depth);
            std::fprintf(
                stderr,
                "URMA-DESC cq base=0x%llx tail_ptr=0x%llx db=0x%llx "
                "entry=%u depth=%u\n",
                static_cast<unsigned long long>(cq.base),
                static_cast<unsigned long long>(cq.tail),
                static_cast<unsigned long long>(cq.doorbell), cq.entry_bytes,
                cq.depth);
        }
    }
    transport::cann_abi::RegisteredBuffer buffers[64]{};
    if (channel.local_buffer_count <= 64 && channel.local_buffers != 0 &&
        copy_from_device(
            buffers, reinterpret_cast<const void*>(channel.local_buffers),
            channel.local_buffer_count * sizeof(buffers[0]),
            "copy debug local buffers", error, sizeof(error))) {
        for (std::uint32_t index = 0; index < channel.local_buffer_count;
             ++index) {
            std::fprintf(
                stderr,
                "URMA-DESC local[%u]=[0x%llx,+%llu] token=%u value=%u\n",
                index, static_cast<unsigned long long>(buffers[index].address),
                static_cast<unsigned long long>(buffers[index].bytes),
                buffers[index].token_id, buffers[index].token_value);
        }
    }
    if (channel.remote_buffer_count <= 64 && channel.remote_buffers != 0 &&
        copy_from_device(
            buffers, reinterpret_cast<const void*>(channel.remote_buffers),
            channel.remote_buffer_count * sizeof(buffers[0]),
            "copy debug remote buffers", error, sizeof(error))) {
        for (std::uint32_t index = 0; index < channel.remote_buffer_count;
             ++index) {
            std::fprintf(
                stderr,
                "URMA-DESC remote[%u]=[0x%llx,+%llu] token=%u value=%u\n",
                index, static_cast<unsigned long long>(buffers[index].address),
                static_cast<unsigned long long>(buffers[index].bytes),
                buffers[index].token_id, buffers[index].token_value);
        }
    }
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

std::uint32_t inspect_cq_depth(
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
        channel.cq_contexts == 0 || channel.cq_count == 0) {
        if (error != nullptr && error[0] == '\0')
            write_error(error, error_capacity, "invalid CQ table");
        return 0;
    }
    transport::cann_abi::CqContext cq{};
    if (!copy_from_device(
            &cq, reinterpret_cast<const void*>(channel.cq_contexts),
            sizeof(cq), "copy CQ context", error, error_capacity))
        return 0;
    return cq.depth;
}

std::uint32_t inspect_command_capacity(
    const transport::DeviceTransportContext& context, char* error,
    std::size_t error_capacity) {
    transport::StagedTransportContext staged{};
    if (context.backend_context == 0 ||
        !copy_from_device(
            &staged, reinterpret_cast<const void*>(context.backend_context),
            sizeof(staged), "copy staged context", error, error_capacity) ||
        staged.command_queue == 0) {
        if (error != nullptr && error[0] == '\0')
            write_error(error, error_capacity, "invalid command queue");
        return 0;
    }
    transport::TransportCommandQueue queue{};
    if (!copy_from_device(
            &queue, reinterpret_cast<const void*>(staged.command_queue),
            sizeof(queue), "copy command queue", error, error_capacity))
        return 0;
    return queue.capacity;
}

class RuntimeResources {
public:
    ~RuntimeResources() {
        (void)shutdown(nullptr, 0);
    }

    bool initialize(
        std::int64_t communicator_handle, std::uint32_t rank,
        std::uint32_t world_size, char* error,
        std::size_t error_capacity) {
        communicator_handle_ = communicator_handle;
        rank_ = rank;
        world_size_ = world_size;
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
        config.stage_profile_enabled = true;
        auto topology_status =
            transport::configure_transport_topology_from_environment(&config);
        if (!topology_status.ok()) {
            write_error(
                error, error_capacity, "%s failed: backend=%d %s",
                topology_status.operation.c_str(), topology_status.backend_code,
                topology_status.message.c_str());
            return false;
        }
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
        if (std::getenv("DEEP_EP_ASCEND_URMA_DUMP_DESCRIPTORS") != nullptr)
            dump_transport_descriptors(context_, (rank + 1) % world_size);
        return true;
    }

    bool run(
        probe::RuntimeCase runtime_case, std::uint32_t rank,
        std::uint32_t world_size, std::uint64_t iterations,
        char* error, std::size_t error_capacity) {
        const std::uint32_t peer = (rank + 1) % world_size;
        std::uint64_t launch_count = std::max<std::uint64_t>(iterations, 1);
        std::uint32_t operations = 1;
        std::uint32_t command_capacity = 0;
        std::uint32_t pressure_depth = 0;
        std::uint64_t pressure_remaining = 0;
        if (runtime_case == probe::RuntimeCase::kQueueWrap) {
            const auto depth = inspect_sq_depth(
                context_, peer, error, error_capacity);
            if (depth == 0)
                return false;
            command_capacity = inspect_command_capacity(
                context_, error, error_capacity);
            operations = probe::queue_wrap_batch_operations(command_capacity);
            if (operations == 0) {
                if (error != nullptr && error[0] == '\0')
                    write_error(
                        error, error_capacity,
                        "command queue capacity %u cannot hold queue-wrap "
                        "barriers and payload", command_capacity);
                return false;
            }
            launch_count = (static_cast<std::uint64_t>(depth) +
                            operations - 1) /
                           operations + 1;
        } else if (runtime_case == probe::RuntimeCase::kProfileMixed) {
            pressure_depth = inspect_cq_depth(
                context_, peer, error, error_capacity);
            command_capacity = inspect_command_capacity(
                context_, error, error_capacity);
            if (pressure_depth <= 1 ||
                command_capacity < probe::kMixedProfileCommandCount) {
                if (error != nullptr && error[0] == '\0')
                    write_error(
                        error, error_capacity,
                        "cannot force queue pressure: cq_depth=%u "
                        "command_capacity=%u",
                        pressure_depth, command_capacity);
                return false;
            }
            pressure_remaining = pressure_depth - 1;
            launch_count = (pressure_remaining + command_capacity - 1) /
                command_capacity + 1;
        }

        auto* device_state = static_cast<probe::RuntimeState*>(window_);
        for (std::uint64_t launch = 0; launch < launch_count; ++launch) {
            const std::uint64_t generation = launch + 1;
            const bool finalize_profile_pressure =
                runtime_case == probe::RuntimeCase::kProfileMixed &&
                pressure_remaining == 0;
            if (runtime_case == probe::RuntimeCase::kProfileMixed) {
                operations = finalize_profile_pressure ? 1 :
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        pressure_remaining, command_capacity));
            }
            probe::RuntimeState state;
            state.source =
                (static_cast<std::uint64_t>(rank + 1) << 32U) | generation;
            state.generation = generation;
            if (!probe::reset_synchronize_and_launch(
                    finalize_profile_pressure,
                    [&] {
                        return check_acl(
                            aclrtMemcpy(
                                device_state, sizeof(state), &state,
                                sizeof(state), ACL_MEMCPY_HOST_TO_DEVICE),
                            "initialize runtime state", error,
                            error_capacity);
                    },
                    [&] {
                        const auto status = transport_->host_barrier();
                        if (status.ok())
                            return true;
                        write_error(
                            error, error_capacity,
                            "final profile host barrier failed: operation=%s "
                            "backend=%d %s",
                            status.operation.c_str(), status.backend_code,
                            status.message.c_str());
                        return false;
                    },
                    [&] {
                        const int launch_status =
                            deep_ep_ascend_urma_launch_runtime_probe(
                                device_state, context_, runtime_case, peer,
                                generation, operations,
                                finalize_profile_pressure, stream_);
                        if (launch_status == 0)
                            return true;
                        write_error(
                            error, error_capacity,
                            "kernel launch failed with %d", launch_status);
                        return false;
                    }))
                return false;
            if (!check_acl(
                    aclrtSynchronizeStream(stream_), "synchronize stream",
                    error, error_capacity) ||
                !copy_from_device(
                    &state, device_state, sizeof(state), "copy runtime state",
                    error, error_capacity))
                return false;
            if (state.success != 1 ||
                state.diagnostic.error != transport::DeviceTransportError::kNone) {
                if (std::getenv(
                        "DEEP_EP_ASCEND_URMA_DUMP_DESCRIPTORS") != nullptr)
                    dump_transport_descriptors(context_, peer);
                write_error(
                    error, error_capacity,
                    "semantic failure: case=%u generation=%llu success=%u "
                    "diagnostic=%u opcode=%u team=%u logical_peer=%u "
                    "world_peer=%d channel=%u "
                    "sq_position=%u cq_expected=%u cq_tail=%u "
                    "cqe_word0=0x%08x raw_sq_head=0x%llx "
                    "observed=0x%llx",
                    static_cast<unsigned>(runtime_case),
                    static_cast<unsigned long long>(generation), state.success,
                    static_cast<unsigned>(state.diagnostic.error),
                    static_cast<unsigned>(state.diagnostic.opcode),
                    static_cast<unsigned>(state.diagnostic.team),
                    state.diagnostic.peer, state.diagnostic.world_peer,
                    state.diagnostic.channel,
                    state.diagnostic.sq_head, state.diagnostic.cq_head,
                    state.diagnostic.cq_tail,
                    state.diagnostic.backend_status,
                    static_cast<unsigned long long>(
                        state.diagnostic.reserved),
                    static_cast<unsigned long long>(state.observed));
                return false;
            }
            if (probe::runtime_case_records_transport_profile(runtime_case)) {
                transport::TransportStageProfile profile{};
                const auto profile_status =
                    transport_->read_stage_profile(&profile);
                if (!profile_status.ok() ||
                    profile.abi_version !=
                        transport::kTransportStageProfileAbiVersion ||
                    profile.struct_size !=
                        sizeof(transport::TransportStageProfile) ||
                    profile.generation != generation ||
                    profile.completion_generation != generation ||
                    profile.command_count == 0 ||
                    profile.service_start_cycles == 0 ||
                    profile.service_end_cycles <
                        profile.service_start_cycles) {
                    write_error(
                        error, error_capacity,
                        "profile failure: generation=%llu completion=%llu "
                        "commands=%u service=%llu..%llu",
                        static_cast<unsigned long long>(profile.generation),
                        static_cast<unsigned long long>(
                            profile.completion_generation),
                        profile.command_count,
                        static_cast<unsigned long long>(
                            profile.service_start_cycles),
                        static_cast<unsigned long long>(
                            profile.service_end_cycles));
                    return false;
                }
                if (finalize_profile_pressure &&
                    (profile.command_count !=
                         probe::kMixedProfileCommandCount ||
                     profile.put_command_count !=
                         probe::kMixedProfilePutCommandCount ||
                     profile.command_bytes !=
                         probe::kMixedProfilePayloadBytes ||
                     profile.sq_depth != 0 || profile.cq_depth != 0 ||
                     profile.sq_high_watermark != pressure_depth - 1 ||
                     profile.cq_high_watermark != pressure_depth - 1 ||
                     profile.wait_cycles == 0 ||
                     transport::transport_stage_profile_command_metrics_status(
                         profile, true) !=
                         transport::TransportStageProfileCommandMetricsStatus::
                             kValid)) {
                    write_error(
                        error, error_capacity,
                        "mixed profile failure: commands=%u puts=%u bytes=%llu "
                        "depth=%u/%u hwm=%u/%u expected_hwm=%u wait=%llu",
                        profile.command_count, profile.put_command_count,
                        static_cast<unsigned long long>(profile.command_bytes),
                        profile.sq_depth, profile.cq_depth,
                        profile.sq_high_watermark,
                        profile.cq_high_watermark, pressure_depth - 1,
                        static_cast<unsigned long long>(profile.wait_cycles));
                    return false;
                }
            }
            if (runtime_case == probe::RuntimeCase::kProfileMixed &&
                !finalize_profile_pressure)
                pressure_remaining -= operations;
        }
        return true;
    }

    bool matches(
        std::int64_t communicator_handle, std::uint32_t rank,
        std::uint32_t world_size) const {
        return communicator_handle_ == communicator_handle && rank_ == rank &&
            world_size_ == world_size;
    }

    bool shutdown(char* error, std::size_t error_capacity) {
        bool success = true;
        if (transport_ != nullptr) {
            const auto status = transport_->destroy();
            if (!status.ok()) {
                success = false;
                write_error(
                    error, error_capacity, "%s failed: backend=%d %s",
                    status.operation.c_str(), status.backend_code,
                    status.message.c_str());
            }
            transport_.reset();
        }
        if (stream_ != nullptr) {
            const aclrtStream stream = stream_;
            stream_ = nullptr;
            const aclError result = aclrtDestroyStream(stream);
            if (result != ACL_SUCCESS && success) {
                success = false;
                write_error(
                    error, error_capacity,
                    "destroy stream failed with ACL status %d",
                    static_cast<int>(result));
            }
        }
        if (window_ != nullptr) {
            void* window = window_;
            window_ = nullptr;
            const aclError result = aclrtFree(window);
            if (result != ACL_SUCCESS && success) {
                success = false;
                write_error(
                    error, error_capacity,
                    "free window failed with ACL status %d",
                    static_cast<int>(result));
            }
        }
        context_ = {};
        return success;
    }

private:
    void* window_ = nullptr;
    aclrtStream stream_ = nullptr;
    std::unique_ptr<transport::HostTransport> transport_;
    transport::DeviceTransportContext context_{};
    std::int64_t communicator_handle_ = 0;
    std::uint32_t rank_ = 0;
    std::uint32_t world_size_ = 0;
};

}  // namespace

extern "C" int deep_ep_ascend_urma_run_case(
    std::int64_t communicator_handle, std::uint32_t rank,
    std::uint32_t world_size, const char* case_name,
    std::uint64_t iterations, char* error, std::size_t error_capacity) {
    if (error != nullptr && error_capacity != 0)
        error[0] = '\0';
    if (communicator_handle == 0 || world_size < 2 || rank >= world_size) {
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

    static std::unique_ptr<RuntimeResources> resources;
    if (teardown) {
        if (resources != nullptr && !resources->matches(
                communicator_handle, rank, world_size)) {
            write_error(
                error, error_capacity,
                "persistent runtime resources do not match communicator topology");
            return 2;
        }
        const bool shutdown = resources == nullptr || resources->shutdown(
            error, error_capacity);
        resources.reset();
        return shutdown ? 0 : 1;
    }

    if (resources != nullptr && !resources->matches(
            communicator_handle, rank, world_size)) {
        write_error(
            error, error_capacity,
            "persistent runtime resources do not match communicator topology");
        return 2;
    }
    if (resources == nullptr) {
        auto candidate = std::make_unique<RuntimeResources>();
        if (!candidate->initialize(
                communicator_handle, rank, world_size, error, error_capacity))
            return 1;
        resources = std::move(candidate);
    }
    if (!resources->run(
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
            1, false, stream);
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
