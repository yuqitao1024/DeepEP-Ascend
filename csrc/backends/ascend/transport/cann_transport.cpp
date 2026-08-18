#include "cann_transport.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "transport_commands.hpp"
#include "sync_layout.hpp"

#if __has_include(<acl/acl.h>) && __has_include(<hccl/hccl_comm.h>) && \
    __has_include(<hccl/hccl_team.h>)
#define DEEP_EP_ASCEND_HAS_CANN_HOST_API 1
#include <acl/acl.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_team.h>
#else
#define DEEP_EP_ASCEND_HAS_CANN_HOST_API 0
#endif

namespace deep_ep::ascend::transport {
namespace {

constexpr std::uint64_t kDefaultRetryLimit = 1ULL << 20U;
constexpr TransportCapabilities kValidatedCapabilities =
    capability_bit(TransportCapability::kSymmetricWindow) |
    capability_bit(TransportCapability::kDevicePut) |
    capability_bit(TransportCapability::kDevicePutValue) |
    capability_bit(TransportCapability::kRemoteAtomicAddRelease) |
    capability_bit(TransportCapability::kRemoteSignal) |
    capability_bit(TransportCapability::kSystemMemoryOrdering) |
    capability_bit(TransportCapability::kDeviceBarrier) |
    capability_bit(TransportCapability::kScaleUpTeam);

TransportStatus backend_failure(
    const char* operation, int backend_code) {
    return TransportStatus::runtime_failure(
        operation, backend_code, "CANN host transport call failed");
}

bool valid_api(const CannHostApi& api) {
    return api.get_rank != nullptr && api.get_size != nullptr &&
        api.create_world_team != nullptr && api.register_window != nullptr &&
        api.create_channels != nullptr && api.allocate_device != nullptr &&
        api.zero_device != nullptr && api.copy_to_device != nullptr &&
        api.copy_from_device != nullptr && api.free_device != nullptr &&
        api.deregister_window != nullptr && api.destroy_team != nullptr;
}

class CannHostTransport final : public HostTransport {
public:
    CannHostTransport(
        TransportConfig config, CannHostApi api, std::uint32_t rank,
        std::uint32_t world_size, std::uint32_t command_capacity,
        std::uintptr_t team)
        : config_(std::move(config)), api_(api), rank_(rank),
          world_size_(world_size), command_capacity_(command_capacity),
          team_(team) {}

    ~CannHostTransport() override { (void)destroy(); }

    TransportStatus initialize_resources() {
        if (resources_initialized_)
            return TransportStatus::success();
        auto status = allocate_zero(
            command_capacity_bytes(), &commands_, "allocate_commands");
        if (!status.ok()) return status;
        status = allocate_zero(
            sizeof(TransportCommandQueue), &queue_, "allocate_queue");
        if (!status.ok()) return status;
        status = allocate_zero(
            sizeof(TransportServiceState), &service_, "allocate_service");
        if (!status.ok()) return status;
        status = allocate_zero(
            sizeof(DeviceTransportDiagnostic), &diagnostic_,
            "allocate_diagnostic");
        if (!status.ok()) return status;
        status = allocate_zero(
            sizeof(StagedTransportContext), &staged_, "allocate_context");
        if (!status.ok()) return status;
        TransportServiceState service_state;
        service_state.default_retry_limit = kDefaultRetryLimit;
        status = copy_to_device(
            service_, service_state, "initialize_service");
        if (!status.ok()) return status;
        DeviceTransportDiagnostic diagnostic_state;
        status = copy_to_device(
            diagnostic_, diagnostic_state, "initialize_diagnostic");
        if (!status.ok()) return status;

        TransportCommandQueue queue_state;
        queue_state.capacity = command_capacity_;
        queue_state.commands = pointer_value(commands_);
        queue_state.service_state = pointer_value(service_);
        queue_state.diagnostic = pointer_value(diagnostic_);
        status = copy_to_device(queue_, queue_state, "initialize_queue");
        if (!status.ok()) return status;

        status = publish_staged_context("initialize_context");
        if (status.ok())
            resources_initialized_ = true;
        return status;
    }

    TransportCapabilities capabilities() const noexcept override {
        auto capabilities = kValidatedCapabilities;
#if DEEP_EP_ASCEND_TESTING
        if (config_.topology_kind == TransportTopologyKind::kPhysical2D)
            capabilities |= capability_bit(
                TransportCapability::kScaleOutTeam);
#endif
        return capabilities;
    }

    TransportStatus query_topology(TransportTopology* topology) override {
        if (topology == nullptr)
            return TransportStatus::invalid(
                "query_topology", "topology must not be null");
        *topology = topology_value();
        return TransportStatus::success();
    }

    TransportStatus register_symmetric_window(
        void* base, std::int64_t bytes) override {
        if (teardown_started_)
            return TransportStatus::invalid(
                "register_symmetric_window", "transport teardown has started");
        if (base == nullptr || bytes <= 0)
            return TransportStatus::invalid(
                "register_symmetric_window",
                "base must not be null and bytes must be positive");
        if (bytes != config_.device_buffer_bytes)
            return TransportStatus::invalid(
                "register_symmetric_window",
                "window size must match device_buffer_bytes");
        if (window_ != 0)
            return TransportStatus::invalid(
                "register_symmetric_window", "window is already registered");

        std::uintptr_t window = 0;
        const int result = api_.register_window(
            api_.user_data, config_.communicator_handle, team_, base,
            static_cast<std::uint64_t>(bytes), &window);
        if (result != 0)
            return backend_failure("register_symmetric_window", result);
        if (window == 0)
            return TransportStatus::runtime_failure(
                "register_symmetric_window", 0,
                "CANN returned a null window handle");

        window_ = window;
        local_window_base_ = pointer_value(base);
        window_bytes_ = static_cast<std::uint64_t>(bytes);
        return resources_initialized_ ?
            publish_staged_context("publish_window") :
            TransportStatus::success();
    }

    TransportStatus unregister_symmetric_window() override {
        if (window_ == 0)
            return TransportStatus::success();
        const int result = api_.deregister_window(
            api_.user_data, team_, window_);
        if (result != 0)
            return backend_failure("unregister_symmetric_window", result);
        window_ = 0;
        local_window_base_ = 0;
        window_bytes_ = 0;
        channels_active_ = false;
        return resources_initialized_ ?
            publish_staged_context("clear_window") :
            TransportStatus::success();
    }

    TransportStatus get_peer_base_pointer(
        TransportTeam, int, std::uintptr_t*) override {
        return TransportStatus::unsupported(
            "get_peer_base_pointer",
            "direct peer pointers are not validated for staged URMA");
    }

    TransportStatus acquire_channels(
        int count, CooperationScope scope) override {
        if (teardown_started_)
            return TransportStatus::invalid(
                "acquire_channels", "transport teardown has started");
        if (window_ == 0)
            return TransportStatus::invalid(
                "acquire_channels", "register a window first");
        if (count != 1 || scope != CooperationScope::kParticipant)
            return TransportStatus::invalid(
                "acquire_channels",
                "Phase 2D requires one participant-scope channel");
        if (config_.requested_channels != 0 &&
            count != config_.requested_channels)
            return TransportStatus::invalid(
                "acquire_channels",
                "count must match requested_channels");
        if (channels_active_)
            return TransportStatus::success();
        if (!channels_created_) {
            const int result = api_.create_channels(
                api_.user_data, config_.communicator_handle, team_,
                static_cast<std::uint32_t>(count));
            if (result != 0)
                return backend_failure("acquire_channels", result);
            channels_created_ = true;
        }
        auto status = initialize_resources();
        if (!status.ok()) {
            TransportStatus cleanup = TransportStatus::success();
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

    TransportStatus export_device_context(
        DeviceTransportContext* context) override {
        if (context == nullptr)
            return TransportStatus::invalid(
                "export_device_context", "context must not be null");
        if (teardown_started_ || window_ == 0 || !channels_active_)
            return TransportStatus::invalid(
                "export_device_context",
                "window and channels must be active");
        *context = make_device_transport_context();
        context->capabilities = capabilities();
        context->topology = topology_value();
        context->local_window_base = local_window_base_;
        context->peer_address_table = window_;
        context->channel_table = team_;
        context->backend_context = pointer_value(staged_);
        return TransportStatus::success();
    }

    TransportStatus read_diagnostic(
        DeviceTransportDiagnostic* diagnostic) override {
        if (diagnostic == nullptr)
            return TransportStatus::invalid(
                "read_diagnostic", "diagnostic must not be null");
        if (teardown_started_ || diagnostic_ == nullptr)
            return TransportStatus::invalid(
                "read_diagnostic", "transport diagnostic is unavailable");
        const int result = api_.copy_from_device(
            api_.user_data, diagnostic, diagnostic_, sizeof(*diagnostic));
        return result == 0 ? TransportStatus::success()
                           : backend_failure("read_diagnostic", result);
    }

    TransportStatus host_barrier() override {
        if (api_.host_barrier == nullptr)
            return TransportStatus::unsupported(
                "host_barrier", "host barrier callback is unavailable");
        const int result = api_.host_barrier(
            api_.user_data, config_.communicator_handle);
        return result == 0 ? TransportStatus::success()
                           : backend_failure("host_barrier", result);
    }

    TransportStatus destroy() override {
        if (destroyed_)
            return TransportStatus::success();
        teardown_started_ = true;
        channels_active_ = false;

        TransportStatus first_error = TransportStatus::success();
        release_backend_resources(first_error);

        if (window_ != 0) {
            const int result = api_.deregister_window(
                api_.user_data, team_, window_);
            if (result != 0) {
                if (first_error.ok())
                    first_error = backend_failure(
                        "unregister_symmetric_window", result);
            } else {
                window_ = 0;
                local_window_base_ = 0;
                window_bytes_ = 0;
            }
        }
        if (window_ == 0 && team_ != 0) {
            const int result = api_.destroy_team(api_.user_data, team_);
            if (result != 0) {
                if (first_error.ok())
                    first_error = backend_failure("destroy_team", result);
            } else {
                team_ = 0;
            }
        }
        destroyed_ = window_ == 0 && team_ == 0;
        return first_error;
    }

private:
    constexpr std::uint64_t command_capacity_bytes() const {
        return static_cast<std::uint64_t>(command_capacity_) *
            sizeof(TransportCommand);
    }

    static std::uintptr_t pointer_value(const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer);
    }

    TransportTopology topology_value() const {
        TransportTopology topology;
        const auto status = build_configured_transport_topology(
            config_, &topology);
        if (!status.ok())
            return {};
        topology.scale_up_direct = false;
        return topology;
    }

    TransportStatus allocate_zero(
        std::uint64_t bytes, void** pointer, const char* operation) {
        const int allocate_result = api_.allocate_device(
            api_.user_data, bytes, pointer);
        if (allocate_result != 0)
            return backend_failure(operation, allocate_result);
        if (*pointer == nullptr)
            return TransportStatus::runtime_failure(
                operation, 0, "CANN returned a null device allocation");
        const int zero_result = api_.zero_device(
            api_.user_data, *pointer, bytes);
        return zero_result == 0 ? TransportStatus::success()
                                : backend_failure(operation, zero_result);
    }

    template <typename Value>
    TransportStatus copy_to_device(
        void* destination, const Value& value, const char* operation) {
        const int result = api_.copy_to_device(
            api_.user_data, destination, &value, sizeof(Value));
        return result == 0 ? TransportStatus::success()
                           : backend_failure(operation, result);
    }

    TransportStatus publish_staged_context(const char* operation) {
        StagedTransportContext context;
        context.command_queue = pointer_value(queue_);
        context.team = team_;
        context.window = window_;
        context.reserved = command::registration_cookie(
            context.command_queue, pointer_value(commands_),
            pointer_value(service_), pointer_value(diagnostic_),
            command_capacity_);
        return copy_to_device(staged_, context, operation);
    }

    void free_resource(
        void*& pointer, const char* operation, TransportStatus& first_error) {
        if (pointer == nullptr)
            return;
        const int result = api_.free_device(api_.user_data, pointer);
        if (result != 0 && first_error.ok())
            first_error = backend_failure(operation, result);
        pointer = nullptr;
    }

    void release_backend_resources(TransportStatus& first_error) {
        free_resource(staged_, "free_context", first_error);
        free_resource(diagnostic_, "free_diagnostic", first_error);
        free_resource(service_, "free_service", first_error);
        free_resource(queue_, "free_queue", first_error);
        free_resource(commands_, "free_commands", first_error);
        resources_initialized_ = false;
    }

    TransportConfig config_;
    CannHostApi api_;
    std::uint32_t rank_ = 0;
    std::uint32_t world_size_ = 0;
    std::uint32_t command_capacity_ = 0;
    std::uintptr_t team_ = 0;
    std::uintptr_t window_ = 0;
    std::uintptr_t local_window_base_ = 0;
    std::uint64_t window_bytes_ = 0;
    bool channels_created_ = false;
    bool channels_active_ = false;
    bool resources_initialized_ = false;
    bool teardown_started_ = false;
    bool destroyed_ = false;

    void* commands_ = nullptr;
    void* queue_ = nullptr;
    void* service_ = nullptr;
    void* diagnostic_ = nullptr;
    void* staged_ = nullptr;
};

#if DEEP_EP_ASCEND_HAS_CANN_HOST_API

HcclComm comm_handle(std::int64_t handle) {
    return reinterpret_cast<HcclComm>(static_cast<std::uintptr_t>(handle));
}

HcommTeamHandle team_handle(std::uintptr_t handle) {
    return reinterpret_cast<HcommTeamHandle>(handle);
}

HcommWindowHandle window_handle(std::uintptr_t handle) {
    return reinterpret_cast<HcommWindowHandle>(handle);
}

int cann_get_rank(void*, std::int64_t comm, std::uint32_t* rank) {
    return HcclGetRankId(comm_handle(comm), rank);
}

int cann_get_size(void*, std::int64_t comm, std::uint32_t* size) {
    return HcclGetRankSize(comm_handle(comm), size);
}

int cann_create_world_team(
    void*, std::int64_t comm, std::uint32_t rank, std::uint32_t size,
    const std::uint32_t* rank_ids, std::uint32_t signal_count,
    std::uint32_t barrier_count, std::uintptr_t* team) {
    HcclTeamCreateDesc desc;
    auto result = HcclTeamCreateDescInit(&desc);
    if (result != HCCL_SUCCESS)
        return result;
    desc.rankIds = rank_ids;
    desc.rankNum = size;
    desc.selfRankId = rank;
    desc.protocol = COMM_PROTOCOL_UBC_CTP;
    desc.requirement.signalCount = signal_count;
    desc.requirement.counterCount = 0;
    desc.requirement.barrierCount = barrier_count;
    HcommTeamHandle handle = nullptr;
    result = HcclWorldTeamCreate(comm_handle(comm), &desc, &handle);
    *team = reinterpret_cast<std::uintptr_t>(handle);
    return result;
}

int cann_register_window(
    void*, std::int64_t comm, std::uintptr_t team, void* base,
    std::uint64_t bytes, std::uintptr_t* window) {
    CommMem memory{COMM_MEM_TYPE_DEVICE, base, bytes};
    HcommWindowHandle handle = nullptr;
    const auto result = HcclTeamWindowRegister(
        comm_handle(comm), team_handle(team), &memory, &handle, 0);
    *window = reinterpret_cast<std::uintptr_t>(handle);
    return result;
}

int cann_create_channels(
    void*, std::int64_t comm, std::uintptr_t team, std::uint32_t count) {
    HcclTeamCreateChannelsDesc desc;
    auto result = HcclTeamCreateChannelsDescInit(&desc);
    if (result != HCCL_SUCCESS)
        return result;
    desc.engine = COMM_ENGINE_AIV;
    desc.notifyNum = 0;
    desc.protocol = COMM_PROTOCOL_UBC_CTP;
    desc.channelCnt = count;
    return HcclTeamChannelsCreate(
        comm_handle(comm), team_handle(team), &desc);
}

int cann_allocate(void*, std::uint64_t bytes, void** pointer) {
    return aclrtMalloc(
        pointer, static_cast<std::size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST);
}

int cann_zero(void*, void* pointer, std::uint64_t bytes) {
    return aclrtMemset(
        pointer, static_cast<std::size_t>(bytes), 0,
        static_cast<std::size_t>(bytes));
}

int cann_copy(void*, void* destination, const void* source,
              std::uint64_t bytes) {
    return aclrtMemcpy(
        destination, static_cast<std::size_t>(bytes), source,
        static_cast<std::size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE);
}

int cann_copy_from_device(void*, void* destination, const void* source,
                          std::uint64_t bytes) {
    return aclrtMemcpy(
        destination, static_cast<std::size_t>(bytes), source,
        static_cast<std::size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST);
}

int cann_free(void*, void* pointer) {
    return aclrtFree(pointer);
}

int cann_deregister_window(
    void*, std::uintptr_t team, std::uintptr_t window) {
    return HcclTeamWindowDeregister(
        team_handle(team), window_handle(window));
}

int cann_destroy_team(void*, std::uintptr_t team) {
    return HcclTeamDestroy(team_handle(team));
}

int cann_host_barrier(void*, std::int64_t comm) {
    return HcclBarrier(comm_handle(comm), nullptr);
}

CannHostApi default_api() {
    return {
        nullptr,
        cann_get_rank,
        cann_get_size,
        cann_create_world_team,
        cann_register_window,
        cann_create_channels,
        cann_allocate,
        cann_zero,
        cann_copy,
        cann_copy_from_device,
        cann_free,
        cann_deregister_window,
        cann_destroy_team,
        cann_host_barrier,
    };
}

#else

CannHostApi default_api() { return {}; }

#endif

TransportStatus validate_config(const TransportConfig& config) {
    if (config.world_size <= 0)
        return TransportStatus::invalid(
            "make_cann_transport", "world_size must be positive");
    if (config.rank < 0 || config.rank >= config.world_size)
        return TransportStatus::invalid(
            "make_cann_transport", "rank must be in [0, world_size)");
    if (config.communicator_handle == 0)
        return TransportStatus::invalid(
            "make_cann_transport", "communicator_handle must not be zero");
    if (!config.cpu_communicator_empty)
        return TransportStatus::invalid(
            "make_cann_transport", "cpu communicator must be empty");
    if (config.device_buffer_bytes <= 0)
        return TransportStatus::invalid(
            "make_cann_transport", "device_buffer_bytes must be positive");
    if (config.cpu_buffer_bytes != 0)
        return TransportStatus::invalid(
            "make_cann_transport",
            "cpu_buffer_bytes must be zero because mapped CPU memory is unsupported");
    if (config.requested_channels != 1)
        return TransportStatus::invalid(
            "make_cann_transport", "exactly one channel is required");
    TransportTopology topology;
    auto topology_status = build_configured_transport_topology(
        config, &topology);
    if (!topology_status.ok()) {
        topology_status.operation = "make_cann_transport";
        return topology_status;
    }
    std::uint32_t command_capacity = 0;
    if (!checked_scale_up_command_capacity(
            config.world_size, &command_capacity))
        return TransportStatus::invalid(
            "make_cann_transport",
            "rank count exceeds transport command capacity");
    return TransportStatus::success();
}

}  // namespace

TransportStatus query_cann_communicator_size(
    std::int64_t communicator_handle, std::uint32_t* world_size,
    const CannHostApi& api) {
    if (communicator_handle == 0)
        return TransportStatus::invalid(
            "query_communicator_size",
            "communicator_handle must not be zero");
    if (world_size == nullptr)
        return TransportStatus::invalid(
            "query_communicator_size", "world_size must not be null");
    if (api.get_size == nullptr)
        return TransportStatus::runtime_failure(
            "query_communicator_size", 0,
            "CANN public host headers are unavailable at build time");

    std::uint32_t queried_size = 0;
    const int result = api.get_size(
        api.user_data, communicator_handle, &queried_size);
    if (result != 0)
        return backend_failure("query_communicator_size", result);
    if (queried_size == 0 ||
        queried_size > static_cast<std::uint32_t>(
            std::numeric_limits<int>::max()))
        return TransportStatus::runtime_failure(
            "query_communicator_size", 0,
            "CANN returned an invalid communicator size");
    *world_size = queried_size;
    return TransportStatus::success();
}

TransportStatus query_cann_communicator_size(
    std::int64_t communicator_handle, std::uint32_t* world_size) {
    return query_cann_communicator_size(
        communicator_handle, world_size, default_api());
}

TransportCreateResult make_cann_transport(
    const TransportConfig& config, const CannHostApi& api) {
    auto status = validate_config(config);
    if (!status.ok())
        return {std::move(status), nullptr};
    if (!valid_api(api))
        return {
            TransportStatus::invalid(
                "make_cann_transport", "CannHostApi is incomplete"),
            nullptr};

    std::uint32_t rank = 0;
    int result = api.get_rank(
        api.user_data, config.communicator_handle, &rank);
    if (result != 0)
        return {backend_failure("get_rank", result), nullptr};
    std::uint32_t world_size = 0;
    result = api.get_size(
        api.user_data, config.communicator_handle, &world_size);
    if (result != 0)
        return {backend_failure("get_size", result), nullptr};
    if (rank != static_cast<std::uint32_t>(config.rank) ||
        world_size != static_cast<std::uint32_t>(config.world_size))
        return {
            TransportStatus::invalid(
                "make_cann_transport",
                "communicator rank/size do not match TransportConfig"),
            nullptr};

    std::uint32_t command_capacity = 0;
    if (!checked_scale_up_command_capacity(
            config.world_size, &command_capacity))
        return {
            TransportStatus::invalid(
                "make_cann_transport",
                "rank count exceeds transport command capacity"),
            nullptr};

    std::vector<std::uint32_t> rank_ids(world_size);
    for (std::uint32_t index = 0; index < world_size; ++index)
        rank_ids[index] = index;
    std::uintptr_t team = 0;
    result = api.create_world_team(
        api.user_data, config.communicator_handle, rank, world_size,
        rank_ids.data(), sync_layout::kWorldTeamSignalCount,
        sync_layout::kWorldTeamBarrierCount, &team);
    if (result != 0)
        return {backend_failure("create_world_team", result), nullptr};
    if (team == 0)
        return {
            TransportStatus::runtime_failure(
                "create_world_team", 0, "CANN returned a null team handle"),
            nullptr};

    auto transport = std::make_unique<CannHostTransport>(
        config, api, rank, world_size, command_capacity, team);
    return {TransportStatus::success(), std::move(transport)};
}

TransportCreateResult make_cann_transport(const TransportConfig& config) {
    const auto api = default_api();
    if (!valid_api(api))
        return {
            TransportStatus::runtime_failure(
                "make_cann_transport", 0,
                "CANN public host headers are unavailable at build time"),
            nullptr};
    return make_cann_transport(config, api);
}

}  // namespace deep_ep::ascend::transport
