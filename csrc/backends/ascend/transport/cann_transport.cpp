#include "cann_transport.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "transport_commands.hpp"

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

constexpr std::uint32_t kSignalCount = 4;
constexpr std::uint32_t kBarrierCount = 1;
constexpr std::uint32_t kCommandCapacity = 256;
constexpr std::uint64_t kDefaultRetryLimit = 1ULL << 20U;

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
        api.free_device != nullptr && api.deregister_window != nullptr &&
        api.destroy_team != nullptr;
}

class CannHostTransport final : public HostTransport {
public:
    CannHostTransport(
        TransportConfig config, CannHostApi api, std::uint32_t rank,
        std::uint32_t world_size, std::uintptr_t team)
        : config_(std::move(config)), api_(api), rank_(rank),
          world_size_(world_size), team_(team) {}

    ~CannHostTransport() override { (void)destroy(); }

    TransportStatus initialize_resources() {
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
        status = allocate_zero(
            static_cast<std::uint64_t>(world_size_) * sizeof(std::uint64_t),
            &fetch_results_, "allocate_fetch_results");
        if (!status.ok()) return status;

        TransportServiceState service_state;
        service_state.default_retry_limit = kDefaultRetryLimit;
        status = copy_to_device(
            service_, service_state, "initialize_service");
        if (!status.ok()) return status;

        TransportCommandQueue queue_state;
        queue_state.capacity = kCommandCapacity;
        queue_state.commands = pointer_value(commands_);
        queue_state.service_state = pointer_value(service_);
        queue_state.diagnostic = pointer_value(diagnostic_);
        status = copy_to_device(queue_, queue_state, "initialize_queue");
        if (!status.ok()) return status;

        return publish_staged_context("initialize_context");
    }

    TransportCapabilities capabilities() const noexcept override {
        return kNoCapabilities;
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
        if (destroyed_)
            return TransportStatus::invalid(
                "register_symmetric_window", "transport is destroyed");
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
        const auto status = publish_staged_context("publish_window");
        if (!status.ok()) {
            (void)api_.deregister_window(api_.user_data, team_, window_);
            window_ = 0;
            local_window_base_ = 0;
            window_bytes_ = 0;
        }
        return status;
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
        return publish_staged_context("clear_window");
    }

    TransportStatus get_peer_base_pointer(
        TransportTeam, int, std::uintptr_t*) override {
        return TransportStatus::unsupported(
            "get_peer_base_pointer",
            "direct peer pointers are not validated for staged URMA");
    }

    TransportStatus acquire_channels(
        int count, CooperationScope scope) override {
        if (destroyed_)
            return TransportStatus::invalid(
                "acquire_channels", "transport is destroyed");
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
        channels_active_ = true;
        return publish_staged_context("publish_channels");
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
        if (destroyed_ || window_ == 0 || !channels_active_)
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
        destroyed_ = true;
        channels_active_ = false;

        TransportStatus first_error = TransportStatus::success();
        free_resource(fetch_results_, "free_fetch_results", first_error);
        free_resource(staged_, "free_context", first_error);
        free_resource(diagnostic_, "free_diagnostic", first_error);
        free_resource(service_, "free_service", first_error);
        free_resource(queue_, "free_queue", first_error);
        free_resource(commands_, "free_commands", first_error);

        if (window_ != 0) {
            const int result = api_.deregister_window(
                api_.user_data, team_, window_);
            if (result != 0 && first_error.ok())
                first_error = backend_failure(
                    "unregister_symmetric_window", result);
            window_ = 0;
        }
        if (team_ != 0) {
            const int result = api_.destroy_team(api_.user_data, team_);
            if (result != 0 && first_error.ok())
                first_error = backend_failure("destroy_team", result);
            team_ = 0;
        }
        return first_error;
    }

private:
    static constexpr std::uint64_t command_capacity_bytes() {
        return static_cast<std::uint64_t>(kCommandCapacity) *
            sizeof(TransportCommand);
    }

    static std::uintptr_t pointer_value(const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer);
    }

    TransportTopology topology_value() const {
        TransportTopology topology;
        topology.world_rank = static_cast<int>(rank_);
        topology.world_size = static_cast<int>(world_size_);
        topology.scale_up_rank = static_cast<int>(rank_);
        topology.scale_up_size = static_cast<int>(world_size_);
        topology.scale_out_rank = 0;
        topology.scale_out_size = 1;
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
        context.fetch_results = pointer_value(fetch_results_);
        context.fetch_result_bytes =
            static_cast<std::uint64_t>(world_size_) * sizeof(std::uint64_t);
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

    TransportConfig config_;
    CannHostApi api_;
    std::uint32_t rank_ = 0;
    std::uint32_t world_size_ = 0;
    std::uintptr_t team_ = 0;
    std::uintptr_t window_ = 0;
    std::uintptr_t local_window_base_ = 0;
    std::uint64_t window_bytes_ = 0;
    bool channels_created_ = false;
    bool channels_active_ = false;
    bool destroyed_ = false;

    void* commands_ = nullptr;
    void* queue_ = nullptr;
    void* service_ = nullptr;
    void* diagnostic_ = nullptr;
    void* staged_ = nullptr;
    void* fetch_results_ = nullptr;
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
    std::uint32_t signal_count, std::uint32_t barrier_count,
    std::uintptr_t* team) {
    HcclTeamCreateDesc desc;
    auto result = HcclTeamCreateDescInit(&desc);
    if (result != HCCL_SUCCESS)
        return result;
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
    if (config.cpu_buffer_bytes != 0 || config.allow_hybrid_mode)
        return TransportStatus::invalid(
            "make_cann_transport", "hybrid host memory is unsupported");
    if (config.requested_channels != 1)
        return TransportStatus::invalid(
            "make_cann_transport", "exactly one channel is required");
    return TransportStatus::success();
}

}  // namespace

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

    std::uintptr_t team = 0;
    result = api.create_world_team(
        api.user_data, config.communicator_handle, rank, world_size,
        kSignalCount, kBarrierCount, &team);
    if (result != 0)
        return {backend_failure("create_world_team", result), nullptr};
    if (team == 0)
        return {
            TransportStatus::runtime_failure(
                "create_world_team", 0, "CANN returned a null team handle"),
            nullptr};

    auto transport = std::make_unique<CannHostTransport>(
        config, api, rank, world_size, team);
    status = transport->initialize_resources();
    if (!status.ok()) {
        (void)transport->destroy();
        return {std::move(status), nullptr};
    }
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
