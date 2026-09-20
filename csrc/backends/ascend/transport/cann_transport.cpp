#include "cann_transport.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sync_layout.hpp"
#include "transport_commands.hpp"

#if __has_include(<acl/acl.h>) && __has_include(<hccl/hccl_comm.h>) && \
    __has_include(<hccl/hccl_channel.h>) && \
    __has_include(<hccl/hccl_rank_graph.h>)
#define DEEP_EP_ASCEND_HAS_CANN_HOST_API 1
#include <acl/acl.h>
#include <hccl/hccl_channel.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_rank_graph.h>
#else
#define DEEP_EP_ASCEND_HAS_CANN_HOST_API 0
#endif

namespace deep_ep::ascend::transport {
namespace {

constexpr std::uint64_t kDefaultRetryLimit = 1ULL << 20U;
constexpr char kPayloadMemoryTag[] = "DeepEPUbcBuffer";
constexpr char kSyncMemoryTag[] = "DeepEPUbcSync";
constexpr char kCommandMemoryTag[] = "DeepEPUbcCommands";
constexpr TransportCapabilities kValidatedCapabilities = capability_bit(TransportCapability::kSymmetricWindow) |
    capability_bit(TransportCapability::kDevicePut) | capability_bit(TransportCapability::kDevicePutValue) |
    capability_bit(TransportCapability::kRemoteAtomicAddRelease) | capability_bit(TransportCapability::kRemoteSignal) |
    capability_bit(TransportCapability::kAsyncCompletion) | capability_bit(TransportCapability::kSystemMemoryOrdering) |
    capability_bit(TransportCapability::kDeviceBarrier) | capability_bit(TransportCapability::kScaleUpTeam);

TransportStatus backend_failure(const char* operation, int backend_code) {
    return TransportStatus::runtime_failure(operation, backend_code, "CANN host transport call failed");
}

TransportStatus runtime_failure(const char* operation, const char* message) {
    return TransportStatus::runtime_failure(operation, 0, message);
}

bool valid_api(const CannHostApi& api) {
    return api.get_rank != nullptr && api.get_size != nullptr && api.register_memory != nullptr && api.acquire_channels != nullptr &&
        api.get_remote_memory != nullptr && api.allocate_device != nullptr && api.zero_device != nullptr && api.copy_to_device != nullptr &&
        api.copy_from_device != nullptr && api.free_device != nullptr && api.destroy_channels != nullptr &&
        api.deregister_memory != nullptr;
}

class CannHostTransport final : public HostTransport {
public:
    CannHostTransport(TransportConfig config, CannHostApi api, std::uint32_t rank, std::uint32_t world_size, std::uint32_t command_capacity)
        : config_(std::move(config)), api_(api), rank_(rank), world_size_(world_size), command_capacity_(command_capacity) {}

    ~CannHostTransport() override { (void)destroy(); }

    TransportCapabilities capabilities() const noexcept override {
        auto capabilities = kValidatedCapabilities;
        if (config_.stage_profile_enabled)
            capabilities |= capability_bit(TransportCapability::kStageProfile);
#if DEEP_EP_ASCEND_TESTING
        if (config_.topology_kind == TransportTopologyKind::kPhysical2D)
            capabilities |= capability_bit(TransportCapability::kScaleOutTeam);
#endif
        return capabilities;
    }

    TransportStatus query_topology(TransportTopology* topology) override {
        if (topology == nullptr)
            return TransportStatus::invalid("query_topology", "topology must not be null");
        *topology = topology_value();
        return TransportStatus::success();
    }

    TransportStatus register_symmetric_window(void* base, std::int64_t bytes) override {
        if (teardown_started_)
            return TransportStatus::invalid("register_symmetric_window", "transport teardown has started");
        if (base == nullptr || bytes <= 0)
            return TransportStatus::invalid("register_symmetric_window", "base must not be null and bytes must be positive");
        if (bytes != config_.device_buffer_bytes)
            return TransportStatus::invalid("register_symmetric_window", "window size must match device_buffer_bytes");
        if (window_registered_)
            return TransportStatus::invalid("register_symmetric_window", "window is already registered");

        const int payload_result = api_.register_memory(
            api_.user_data, config_.communicator_handle, kPayloadMemoryTag, base, static_cast<std::uint64_t>(bytes), &payload_memory_);
        if (payload_result != 0)
            return backend_failure("register_symmetric_window", payload_result);
        if (payload_memory_ == 0)
            return runtime_failure("register_symmetric_window", "CANN returned a null payload memory handle");

        window_registered_ = true;
        local_window_base_ = pointer_value(base);
        window_bytes_ = static_cast<std::uint64_t>(bytes);
        return TransportStatus::success();
    }

    TransportStatus unregister_symmetric_window() override {
        if (!window_registered_)
            return TransportStatus::success();
        if (channels_created_) {
            const auto status = release_backend_channels();
            if (!status.ok())
                return status;
        }
        // The channel owns references to all three MRs.  Destroy it first,
        // even when this method is called explicitly rather than destroy().
        TransportStatus first_error = TransportStatus::success();
        release_backend_resources(first_error);
        if (sync_registered_ && !deregister_sync() && first_error.ok())
            first_error = backend_failure("unregister_symmetric_window", sync_deregister_result_);
        if (!first_error.ok())
            return first_error;
        if (payload_memory_ != 0) {
            const int result = api_.deregister_memory(api_.user_data, config_.communicator_handle, kPayloadMemoryTag, payload_memory_);
            if (result != 0)
                return backend_failure("unregister_symmetric_window", result);
            payload_memory_ = 0;
        }
        window_registered_ = false;
        local_window_base_ = 0;
        window_bytes_ = 0;
        return TransportStatus::success();
    }

    TransportStatus get_peer_base_pointer(TransportTeam, int, std::uintptr_t*) override {
        return TransportStatus::unsupported("get_peer_base_pointer", "direct peer pointers are not validated for staged URMA");
    }

    TransportStatus acquire_channels(int count, CooperationScope scope) override {
        if (teardown_started_)
            return TransportStatus::invalid("acquire_channels", "transport teardown has started");
        if (!window_registered_)
            return TransportStatus::invalid("acquire_channels", "register a window first");
        if (!valid_transport_channel_count(count) || scope != CooperationScope::kParticipant)
            return TransportStatus::invalid("acquire_channels", "requires 1-4 participant-scope channels");
        if (config_.requested_channels != 0 && count != config_.requested_channels)
            return TransportStatus::invalid("acquire_channels", "count must match requested_channels");
        if (channels_active_)
            return TransportStatus::success();

        auto status = allocate_sync_window();
        if (!status.ok())
            return status;
        status = allocate_command_memory();
        if (!status.ok())
            return status;
        status = acquire_backend_channels(count);
        if (!status.ok()) {
            TransportStatus cleanup = TransportStatus::success();
            (void)release_backend_channels();
            release_backend_resources(cleanup);
            return status;
        }
        status = initialize_resources();
        if (!status.ok()) {
            TransportStatus cleanup = TransportStatus::success();
            (void)release_backend_channels();
            release_backend_resources(cleanup);
            return status;
        }
        channels_active_ = true;
        return TransportStatus::success();
    }

    TransportStatus release_channels() override {
        channels_active_ = false;
        return TransportStatus::success();
    }

    TransportStatus export_device_context(DeviceTransportContext* context) override {
        if (context == nullptr)
            return TransportStatus::invalid("export_device_context", "context must not be null");
        if (teardown_started_ || !window_registered_ || !channels_active_)
            return TransportStatus::invalid("export_device_context", "window and channels must be active");
        *context = make_device_transport_context();
        context->capabilities = capabilities();
        context->topology = topology_value();
        context->local_window_base = local_window_base_;
        context->peer_address_table = pointer_value(channel_table_);
        context->channel_table = pointer_value(channel_table_);
        context->channel_count = static_cast<std::uint32_t>(config_.requested_channels);
        context->backend_context = pointer_value(staged_);
        return TransportStatus::success();
    }

    TransportStatus read_diagnostic(DeviceTransportDiagnostic* diagnostic) override {
        if (diagnostic == nullptr)
            return TransportStatus::invalid("read_diagnostic", "diagnostic must not be null");
        if (teardown_started_ || diagnostic_ == nullptr)
            return TransportStatus::invalid("read_diagnostic", "transport diagnostic is unavailable");
        const int result = api_.copy_from_device(api_.user_data, diagnostic, diagnostic_, sizeof(*diagnostic));
        return result == 0 ? TransportStatus::success() : backend_failure("read_diagnostic", result);
    }

    TransportStatus reset_stage_profile() override {
        if (teardown_started_ || stage_profile_ == nullptr)
            return TransportStatus::unsupported("reset_stage_profile", "transport stage profile is disabled");
        const TransportStageProfile profile{};
        return copy_to_device(stage_profile_, profile, "reset_stage_profile");
    }

    TransportStatus read_stage_profile(TransportStageProfile* profile) override {
        if (profile == nullptr)
            return TransportStatus::invalid("read_stage_profile", "profile must not be null");
        if (teardown_started_ || stage_profile_ == nullptr)
            return TransportStatus::unsupported("read_stage_profile", "transport stage profile is disabled");
        const int result = api_.copy_from_device(api_.user_data, profile, stage_profile_, sizeof(*profile));
        return result == 0 ? TransportStatus::success() : backend_failure("read_stage_profile", result);
    }

    TransportStatus host_barrier() override {
        if (api_.host_barrier == nullptr)
            return TransportStatus::unsupported("host_barrier", "host barrier callback is unavailable");
        const int result = api_.host_barrier(api_.user_data, config_.communicator_handle);
        return result == 0 ? TransportStatus::success() : backend_failure("host_barrier", result);
    }

    TransportStatus destroy() override {
        if (destroyed_)
            return TransportStatus::success();
        teardown_started_ = true;
        channels_active_ = false;

        TransportStatus first_error = TransportStatus::success();
        const auto channel_status = release_backend_channels();
        // Channels keep references to the payload, sync, and command MRs.  If
        // channel destruction fails, stop before deregistering any memory so a
        // later retry starts from the same ownership state.
        if (!channel_status.ok())
            return channel_status;
        release_backend_resources(first_error);
        if (sync_registered_ && !deregister_sync() && first_error.ok())
            first_error = backend_failure("unregister_sync_memory", sync_deregister_result_);
        // Do not retry the sync deregistration in the same destroy call: an
        // MR whose first deregistration failed may still be in use.
        if (!first_error.ok())
            return first_error;
        if (payload_memory_ != 0) {
            const int result = api_.deregister_memory(api_.user_data, config_.communicator_handle, kPayloadMemoryTag, payload_memory_);
            if (result != 0 && first_error.ok())
                first_error = backend_failure("unregister_symmetric_window", result);
            else if (result == 0)
                payload_memory_ = 0;
        }
        if (payload_memory_ == 0) {
            window_registered_ = false;
            local_window_base_ = 0;
            window_bytes_ = 0;
        }
        destroyed_ = payload_memory_ == 0;
        return first_error;
    }

private:
    constexpr std::uint64_t command_capacity_bytes() const {
        return static_cast<std::uint64_t>(command_capacity_) * sizeof(TransportCommand);
    }

    static std::uintptr_t pointer_value(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

    TransportTopology topology_value() const {
        TransportTopology topology;
        const auto status = build_configured_transport_topology(config_, &topology);
        if (!status.ok())
            return {};
        topology.scale_up_direct = false;
        return topology;
    }

    TransportStatus allocate_zero(std::uint64_t bytes, void** pointer, const char* operation) {
        const int allocate_result = api_.allocate_device(api_.user_data, bytes, pointer);
        if (allocate_result != 0)
            return backend_failure(operation, allocate_result);
        if (*pointer == nullptr)
            return runtime_failure(operation, "CANN returned a null device allocation");
        const int zero_result = api_.zero_device(api_.user_data, *pointer, bytes);
        return zero_result == 0 ? TransportStatus::success() : backend_failure(operation, zero_result);
    }

    template <typename Value>
    TransportStatus copy_to_device(void* destination, const Value& value, const char* operation) {
        const int result = api_.copy_to_device(api_.user_data, destination, &value, sizeof(Value));
        return result == 0 ? TransportStatus::success() : backend_failure(operation, result);
    }

    TransportStatus allocate_sync_window() {
        if (sync_registered_)
            return TransportStatus::success();
        sync_bytes_ = sync_layout::sync_window_bytes(world_size_);
        auto status = allocate_zero(sync_bytes_, &sync_memory_, "allocate_sync_memory");
        if (!status.ok())
            return status;
        const int result = api_.register_memory(
            api_.user_data, config_.communicator_handle, kSyncMemoryTag, sync_memory_, sync_bytes_, &sync_memory_handle_);
        if (result != 0) {
            TransportStatus first_error = TransportStatus::success();
            free_resource(sync_memory_, "free_sync_memory", first_error);
            return backend_failure("register_sync_memory", result);
        }
        sync_registered_ = true;
        return TransportStatus::success();
    }

    bool deregister_sync() {
        if (!sync_registered_)
            return true;
        const int result = api_.deregister_memory(api_.user_data, config_.communicator_handle, kSyncMemoryTag, sync_memory_handle_);
        sync_deregister_result_ = result;
        if (result == 0) {
            sync_registered_ = false;
            sync_memory_handle_ = 0;
            TransportStatus first_error = TransportStatus::success();
            free_resource(sync_memory_, "free_sync_memory", first_error);
            sync_bytes_ = 0;
        }
        return result == 0;
    }

    TransportStatus allocate_command_memory() {
        if (command_registered_)
            return TransportStatus::success();
        auto status = allocate_zero(command_capacity_bytes(), &commands_, "allocate_commands");
        if (!status.ok())
            return status;
        const int result = api_.register_memory(
            api_.user_data, config_.communicator_handle, kCommandMemoryTag, commands_, command_capacity_bytes(), &command_memory_handle_);
        if (result != 0) {
            TransportStatus first_error = TransportStatus::success();
            free_resource(commands_, "free_commands", first_error);
            return backend_failure("register_command_memory", result);
        }
        command_registered_ = true;
        return TransportStatus::success();
    }

    TransportStatus acquire_backend_channels(std::uint32_t count) {
        if (channels_created_)
            return TransportStatus::success();
        const auto peers = world_size_ - 1;
        channel_handles_.assign(static_cast<std::size_t>(peers) * count, 0);
        std::size_t offset = 0;
        for (std::uint32_t peer = 0; peer < world_size_; ++peer) {
            if (peer == rank_)
                continue;
            const std::uintptr_t memories[3] = {payload_memory_, sync_memory_handle_, command_memory_handle_};
            const int result = api_.acquire_channels(
                api_.user_data, config_.communicator_handle, rank_, peer, memories, 3, count, channel_handles_.data() + offset);
            if (result != 0)
                return backend_failure("acquire_channels", result);
            for (std::uint32_t channel = 0; channel < count; ++channel) {
                if (channel_handles_[offset + channel] == 0)
                    return runtime_failure("acquire_channels", "CANN returned a null channel handle");
            }
            offset += count;
        }
        channels_created_ = true;
        return build_device_tables(count);
    }

    TransportStatus release_backend_channels() {
        if (!channels_created_)
            return TransportStatus::success();
        // HcclChannelDestroy supports only CCU channels in CANN 9.3.  AIV
        // channels are owned by the HCCL communicator and must outlive all
        // user-created device-side resources.  The official URMA workspace
        // likewise clears its handles without calling HcclChannelDestroy.
        channel_handles_.clear();
        channels_created_ = false;
        channels_active_ = false;
        return TransportStatus::success();
    }

    TransportStatus build_device_tables(std::uint32_t count) {
        std::vector<std::uint64_t> channels(static_cast<std::size_t>(world_size_) * count, 0);
        std::vector<std::uint64_t> remote_bases(world_size_, 0);
        std::vector<std::uint64_t> remote_sync_bases(world_size_, 0);
        std::size_t offset = 0;
        for (std::uint32_t peer = 0; peer < world_size_; ++peer) {
            if (peer == rank_)
                continue;
            for (std::uint32_t channel = 0; channel < count; ++channel)
                channels[static_cast<std::size_t>(peer) * count + channel] = channel_handles_[offset + channel];
            std::uintptr_t address = 0;
            std::uint64_t bytes = 0;
            int result = api_.get_remote_memory(
                api_.user_data, config_.communicator_handle, channel_handles_[offset], kPayloadMemoryTag, &address, &bytes);
            if (result != 0)
                return backend_failure("get_payload_remote_memory", result);
            if (address == 0 || bytes != window_bytes_)
                return runtime_failure("get_payload_remote_memory", "remote payload memory does not match the local window");
            remote_bases[peer] = address;
            result = api_.get_remote_memory(
                api_.user_data, config_.communicator_handle, channel_handles_[offset], kSyncMemoryTag, &address, &bytes);
            if (result != 0)
                return backend_failure("get_sync_remote_memory", result);
            if (address == 0 || bytes != sync_bytes_)
                return runtime_failure("get_sync_remote_memory", "remote sync memory does not match the local layout");
            remote_sync_bases[peer] = address;
            offset += count;
        }

        auto status = allocate_zero(channels.size() * sizeof(std::uint64_t), &device_channels_, "allocate_device_channels");
        if (!status.ok())
            return status;
        status = allocate_zero(remote_bases.size() * sizeof(std::uint64_t), &device_remote_bases_, "allocate_remote_bases");
        if (!status.ok())
            return status;
        status = allocate_zero(remote_sync_bases.size() * sizeof(std::uint64_t), &device_remote_sync_bases_, "allocate_remote_sync_bases");
        if (!status.ok())
            return status;
        status = allocate_zero(sizeof(DeviceChannelTable), &channel_table_, "allocate_channel_table");
        if (!status.ok())
            return status;

        const int copy_channels =
            api_.copy_to_device(api_.user_data, device_channels_, channels.data(), channels.size() * sizeof(std::uint64_t));
        if (copy_channels != 0)
            return backend_failure("initialize_device_channels", copy_channels);
        const int copy_bases =
            api_.copy_to_device(api_.user_data, device_remote_bases_, remote_bases.data(), remote_bases.size() * sizeof(std::uint64_t));
        if (copy_bases != 0)
            return backend_failure("initialize_remote_bases", copy_bases);
        const int copy_sync = api_.copy_to_device(
            api_.user_data, device_remote_sync_bases_, remote_sync_bases.data(), remote_sync_bases.size() * sizeof(std::uint64_t));
        if (copy_sync != 0)
            return backend_failure("initialize_remote_sync_bases", copy_sync);

        DeviceChannelTable table;
        table.member_count = world_size_;
        table.self_member = rank_;
        table.channel_count = count;
        table.channels = pointer_value(device_channels_);
        table.remote_bases = pointer_value(device_remote_bases_);
        table.remote_sync_bases = pointer_value(device_remote_sync_bases_);
        table.local_sync_base = pointer_value(sync_memory_);
        table.window_bytes = window_bytes_;
        table.sync_bytes = sync_bytes_;
        return copy_to_device(channel_table_, table, "initialize_channel_table");
    }

    TransportStatus initialize_resources() {
        if (resources_initialized_)
            return TransportStatus::success();
        auto status = allocate_zero(sizeof(TransportCommandQueue), &queue_, "allocate_queue");
        if (!status.ok())
            return status;
        status = allocate_zero(sizeof(TransportServiceState), &service_, "allocate_service");
        if (!status.ok())
            return status;
        status = allocate_zero(sizeof(DeviceTransportDiagnostic), &diagnostic_, "allocate_diagnostic");
        if (!status.ok())
            return status;
        if (config_.stage_profile_enabled) {
            status = allocate_zero(sizeof(TransportStageProfile), &stage_profile_, "allocate_stage_profile");
            if (!status.ok())
                return status;
        }
        status = allocate_zero(sizeof(StagedTransportContext), &staged_, "allocate_context");
        if (!status.ok())
            return status;
        TransportServiceState service_state;
        service_state.default_retry_limit = kDefaultRetryLimit;
        status = copy_to_device(service_, service_state, "initialize_service");
        if (!status.ok())
            return status;
        DeviceTransportDiagnostic diagnostic_state;
        status = copy_to_device(diagnostic_, diagnostic_state, "initialize_diagnostic");
        if (!status.ok())
            return status;
        TransportCommandQueue queue_state;
        queue_state.capacity = command_capacity_;
        queue_state.commands = pointer_value(commands_);
        queue_state.service_state = pointer_value(service_);
        queue_state.diagnostic = pointer_value(diagnostic_);
        status = copy_to_device(queue_, queue_state, "initialize_queue");
        if (!status.ok())
            return status;
        status = publish_staged_context("initialize_context");
        if (status.ok())
            resources_initialized_ = true;
        return status;
    }

    TransportStatus publish_staged_context(const char* operation) {
        StagedTransportContext context;
        context.command_queue = pointer_value(queue_);
        context.stage_profile = pointer_value(stage_profile_);
        context.stage_profile_bytes = stage_profile_ == nullptr ? 0 : sizeof(TransportStageProfile);
        context.reserved = command::registration_cookie(
            context.command_queue, pointer_value(commands_), pointer_value(service_), pointer_value(diagnostic_), command_capacity_);
        return copy_to_device(staged_, context, operation);
    }

    void free_resource(void*& pointer, const char* operation, TransportStatus& first_error) {
        if (pointer == nullptr)
            return;
        const int result = api_.free_device(api_.user_data, pointer);
        if (result != 0 && first_error.ok())
            first_error = backend_failure(operation, result);
        pointer = nullptr;
    }

    void release_backend_resources(TransportStatus& first_error) {
        free_resource(channel_table_, "free_channel_table", first_error);
        free_resource(device_remote_sync_bases_, "free_remote_sync_bases", first_error);
        free_resource(device_remote_bases_, "free_remote_bases", first_error);
        free_resource(device_channels_, "free_device_channels", first_error);
        free_resource(staged_, "free_context", first_error);
        free_resource(stage_profile_, "free_stage_profile", first_error);
        free_resource(diagnostic_, "free_diagnostic", first_error);
        free_resource(service_, "free_service", first_error);
        free_resource(queue_, "free_queue", first_error);
        if (command_registered_) {
            const int result =
                api_.deregister_memory(api_.user_data, config_.communicator_handle, kCommandMemoryTag, command_memory_handle_);
            if (result != 0 && first_error.ok())
                first_error = backend_failure("unregister_command_memory", result);
            else if (result == 0) {
                command_registered_ = false;
                command_memory_handle_ = 0;
            }
        }
        free_resource(commands_, "free_commands", first_error);
        resources_initialized_ = false;
    }

    TransportConfig config_;
    CannHostApi api_;
    std::uint32_t rank_ = 0;
    std::uint32_t world_size_ = 0;
    std::uint32_t command_capacity_ = 0;
    std::uintptr_t payload_memory_ = 0;
    bool window_registered_ = false;
    std::uintptr_t local_window_base_ = 0;
    std::uint64_t window_bytes_ = 0;
    void* sync_memory_ = nullptr;
    std::uintptr_t sync_memory_handle_ = 0;
    std::uint64_t sync_bytes_ = 0;
    bool sync_registered_ = false;
    std::uintptr_t command_memory_handle_ = 0;
    bool command_registered_ = false;
    int sync_deregister_result_ = 0;
    std::vector<std::uintptr_t> channel_handles_;
    bool channels_created_ = false;
    bool channels_active_ = false;
    bool resources_initialized_ = false;
    bool teardown_started_ = false;
    bool destroyed_ = false;

    void* device_channels_ = nullptr;
    void* device_remote_bases_ = nullptr;
    void* device_remote_sync_bases_ = nullptr;
    void* channel_table_ = nullptr;
    void* commands_ = nullptr;
    void* queue_ = nullptr;
    void* service_ = nullptr;
    void* diagnostic_ = nullptr;
    void* stage_profile_ = nullptr;
    void* staged_ = nullptr;
};

#if DEEP_EP_ASCEND_HAS_CANN_HOST_API

HcclComm comm_handle(std::int64_t handle) {
    return reinterpret_cast<HcclComm>(static_cast<std::uintptr_t>(handle));
}

bool protocol_supported(CommProtocol protocol) {
    return protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP;
}

int cann_get_rank(void*, std::int64_t comm, std::uint32_t* rank) {
    return HcclGetRankId(comm_handle(comm), rank);
}

int cann_get_size(void*, std::int64_t comm, std::uint32_t* size) {
    return HcclGetRankSize(comm_handle(comm), size);
}

int cann_allocate(void*, std::uint64_t bytes, void** pointer) {
    return aclrtMalloc(pointer, static_cast<std::size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST);
}

int cann_zero(void*, void* pointer, std::uint64_t bytes) {
    return aclrtMemset(pointer, static_cast<std::size_t>(bytes), 0, static_cast<std::size_t>(bytes));
}

int cann_copy(void*, void* destination, const void* source, std::uint64_t bytes) {
    return aclrtMemcpy(destination, static_cast<std::size_t>(bytes), source, static_cast<std::size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE);
}

int cann_copy_from_device(void*, void* destination, const void* source, std::uint64_t bytes) {
    return aclrtMemcpy(destination, static_cast<std::size_t>(bytes), source, static_cast<std::size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST);
}

int cann_free(void*, void* pointer) {
    return aclrtFree(pointer);
}

int cann_register_memory(void*, std::int64_t comm, const char* tag, void* base, std::uint64_t bytes, std::uintptr_t* memory) {
    CommMem value{};
    value.type = COMM_MEM_TYPE_DEVICE;
    value.addr = base;
    value.size = bytes;
    HcclMemHandle handle = nullptr;
    const auto result = HcclCommMemReg(comm_handle(comm), tag, &value, &handle);
    *memory = reinterpret_cast<std::uintptr_t>(handle);
    return result;
}

int cann_acquire_channels(void*,
                          std::int64_t comm,
                          std::uint32_t rank,
                          std::uint32_t peer,
                          const std::uintptr_t* memories,
                          std::uint32_t memory_count,
                          std::uint32_t count,
                          std::uintptr_t* channels) {
    HcclMemHandle handles[3] = {};
    if (memories == nullptr || memory_count != 3)
        return HCCL_E_PARA;
    handles[0] = reinterpret_cast<HcclMemHandle>(memories[0]);
    handles[1] = reinterpret_cast<HcclMemHandle>(memories[1]);
    handles[2] = reinterpret_cast<HcclMemHandle>(memories[2]);
    uint32_t* layers = nullptr;
    uint32_t layer_count = 0;
    auto result = HcclRankGraphGetLayers(comm_handle(comm), &layers, &layer_count);
    if (result != HCCL_SUCCESS)
        return result;
    if (layers == nullptr || layer_count == 0)
        return HCCL_E_NOT_FOUND;

    for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        CommLink* links = nullptr;
        uint32_t link_count = 0;
        result = HcclRankGraphGetLinks(comm_handle(comm), layers[layer_index], rank, peer, &links, &link_count);
        if (result != HCCL_SUCCESS)
            continue;
        for (uint32_t link_index = 0; link_index < link_count; ++link_index) {
            const CommLink& link = links[link_index];
            if (!protocol_supported(link.linkAttr.linkProtocol))
                continue;
            std::vector<HcclChannelDesc> descriptions(count);
            result = HcclChannelDescInit(descriptions.data(), count);
            if (result != HCCL_SUCCESS)
                return result;
            for (auto& description : descriptions) {
                description.remoteRank = peer;
                description.channelProtocol = link.linkAttr.linkProtocol;
                description.localEndpoint = link.srcEndpointDesc;
                description.remoteEndpoint = link.dstEndpointDesc;
                description.notifyNum = 0;
                description.memHandles = handles;
                description.memHandleNum = 3;
            }
            return HcclChannelAcquire(comm_handle(comm), COMM_ENGINE_AIV, descriptions.data(), count, channels);
        }
    }
    return HCCL_E_NOT_FOUND;
}

int cann_get_remote_memory(
    void*, std::int64_t comm, std::uintptr_t channel, const char* tag, std::uintptr_t* address, std::uint64_t* bytes) {
    uint32_t count = 0;
    CommMem* memories = nullptr;
    char** tags = nullptr;
    const auto result = HcclChannelGetRemoteMems(comm_handle(comm), static_cast<ChannelHandle>(channel), &count, &memories, &tags);
    if (result != HCCL_SUCCESS)
        return result;
    for (uint32_t index = 0; index < count; ++index) {
        if (tags != nullptr && tags[index] != nullptr && std::string(tags[index]) == tag) {
            *address = reinterpret_cast<std::uintptr_t>(memories[index].addr);
            *bytes = memories[index].size;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

int cann_destroy_channels(void*, std::int64_t comm, const std::uintptr_t* channels, std::uint32_t count) {
    std::vector<ChannelHandle> handles(count);
    for (uint32_t index = 0; index < count; ++index)
        handles[index] = static_cast<ChannelHandle>(channels[index]);
    return HcclChannelDestroy(comm_handle(comm), handles.data(), count);
}

int cann_deregister_memory(void*, std::int64_t comm, const char* tag, std::uintptr_t memory) {
    // HcclCommDeregMem is not exported by the installed 9.3.0 package.
    // Memory tags are removed when the communicator is destroyed.  Returning
    // success here keeps the callback ABI complete without trying to call
    // an unavailable symbol.
    (void)comm;
    (void)tag;
    (void)memory;
    return HCCL_SUCCESS;
}

int cann_host_barrier(void*, std::int64_t comm) {
    return HcclBarrier(comm_handle(comm), nullptr);
}

CannHostApi default_api() {
    return {
        nullptr,
        cann_get_rank,
        cann_get_size,
        cann_register_memory,
        cann_acquire_channels,
        cann_get_remote_memory,
        cann_allocate,
        cann_zero,
        cann_copy,
        cann_copy_from_device,
        cann_free,
        cann_destroy_channels,
        cann_deregister_memory,
        cann_host_barrier,
    };
}

#else

CannHostApi default_api() {
    return {};
}

#endif

TransportStatus validate_config(const TransportConfig& config) {
    if (config.world_size <= 0)
        return TransportStatus::invalid("make_cann_transport", "world_size must be positive");
    if (config.rank < 0 || config.rank >= config.world_size)
        return TransportStatus::invalid("make_cann_transport", "rank must be in [0, world_size)");
    if (config.communicator_handle == 0)
        return TransportStatus::invalid("make_cann_transport", "communicator_handle must not be zero");
    if (!config.cpu_communicator_empty)
        return TransportStatus::invalid("make_cann_transport", "cpu communicator must be empty");
    if (config.device_buffer_bytes <= 0)
        return TransportStatus::invalid("make_cann_transport", "device_buffer_bytes must be positive");
    if (config.cpu_buffer_bytes != 0)
        return TransportStatus::invalid("make_cann_transport", "cpu_buffer_bytes must be zero because mapped CPU memory is unsupported");
    if (!valid_transport_channel_count(config.requested_channels))
        return TransportStatus::invalid("make_cann_transport", "requested_channels must be in [1, 4]");
    TransportTopology topology;
    auto topology_status = build_configured_transport_topology(config, &topology);
    if (!topology_status.ok()) {
        topology_status.operation = "make_cann_transport";
        return topology_status;
    }
    std::uint32_t command_capacity = 0;
    if (!checked_scale_up_command_capacity(config.world_size, config.requested_channels, &command_capacity))
        return TransportStatus::invalid("make_cann_transport", "rank count exceeds transport command capacity");
    return TransportStatus::success();
}

}  // namespace

TransportStatus query_cann_communicator_size(std::int64_t communicator_handle, std::uint32_t* world_size, const CannHostApi& api) {
    if (world_size == nullptr || api.get_size == nullptr)
        return TransportStatus::invalid("query_cann_communicator_size", "world_size and get_size callback are required");
    const int result = api.get_size(api.user_data, communicator_handle, world_size);
    return result == 0 ? TransportStatus::success() : backend_failure("query_cann_communicator_size", result);
}

TransportStatus query_cann_communicator_size(std::int64_t communicator_handle, std::uint32_t* world_size) {
    return query_cann_communicator_size(communicator_handle, world_size, default_api());
}

TransportCreateResult make_cann_transport(const TransportConfig& config, const CannHostApi& api) {
    if (const auto status = validate_config(config); !status.ok())
        return {status, nullptr};
    if (!valid_api(api))
        return {TransportStatus::runtime_failure("make_cann_transport", 0, "CannHostApi is incomplete"), nullptr};
    std::uint32_t rank = 0;
    int result = api.get_rank(api.user_data, config.communicator_handle, &rank);
    if (result != 0)
        return {backend_failure("make_cann_transport", result), nullptr};
    if (static_cast<int>(rank) != config.rank)
        return {TransportStatus::invalid("make_cann_transport", "communicator rank does not match configured rank"), nullptr};
    std::uint32_t world_size = 0;
    result = api.get_size(api.user_data, config.communicator_handle, &world_size);
    if (result != 0)
        return {backend_failure("make_cann_transport", result), nullptr};
    if (static_cast<int>(world_size) != config.world_size)
        return {TransportStatus::invalid("make_cann_transport", "communicator size does not match configured world size"), nullptr};
    std::uint32_t command_capacity = 0;
    (void)checked_scale_up_command_capacity(config.world_size, config.requested_channels, &command_capacity);
    return {TransportStatus::success(), std::make_unique<CannHostTransport>(config, api, rank, world_size, command_capacity)};
}

TransportCreateResult make_cann_transport(const TransportConfig& config) {
    return make_cann_transport(config, default_api());
}

}  // namespace deep_ep::ascend::transport
