#pragma once

#include <cstdlib>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if __has_include(<c10/core/StreamGuard.h>) && \
    !defined(DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR)
#include <c10/core/StreamGuard.h>
#define DEEP_EP_ASCEND_HAS_GENERIC_STREAM_GUARD 1
#else
#define DEEP_EP_ASCEND_HAS_GENERIC_STREAM_GUARD 0
#endif

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "elastic/layout.hpp"
#include "elastic/barrier_state.hpp"
#include "elastic/combine_state.hpp"
#include "elastic/dispatch_state.hpp"
#include "elastic/async_state.hpp"
#include "elastic/operation_coordinator.hpp"
#include "elastic/runtime.hpp"
#include "runtime/cann_runtime.hpp"
#include "transport/topology_config.hpp"

namespace deep_ep::ascend {

[[noreturn]] inline void raise_unsupported(
    const char* operation, const std::string& detail) {
    const std::string message =
        std::string("DeepEP Ascend backend: ") + operation + " " + detail;
    PyErr_SetString(PyExc_NotImplementedError, message.c_str());
    throw pybind11::error_already_set();
}

[[noreturn]] inline void raise_transport_status(
    const transport::TransportStatus& status, int rank) {
    if (status.code == transport::TransportStatusCode::kUnsupportedCapability)
        raise_unsupported(status.operation.c_str(), status.message);
    if (status.code == transport::TransportStatusCode::kRuntimeFailure) {
        TORCH_CHECK(false, "DeepEP Ascend backend: ", status.operation,
                    " failed on rank ", rank, " with backend error ",
                    status.backend_code, ": ", status.message);
    }
    TORCH_CHECK(false, "DeepEP Ascend backend: ", status.operation,
                " ", status.message);
}

[[noreturn]] inline void raise_event_status(
    const transport::TransportStatus& status) {
    if (status.code == transport::TransportStatusCode::kRuntimeFailure) {
        TORCH_CHECK(false, "DeepEP Ascend backend: ", status.operation,
                    " failed with backend error ", status.backend_code,
                    ": ", status.message);
    }
    TORCH_CHECK(false, "DeepEP Ascend backend: ", status.operation,
                " ", status.message);
}

struct EventHandle {
private:
    static constexpr std::uint64_t kEventWaitTimeoutMs = 5000;

    struct SharedState {
        explicit SharedState(
            std::shared_ptr<runtime::NativeEventState> native_event,
            std::shared_ptr<elastic::PendingOperation> pending = nullptr,
            std::shared_ptr<elastic::AsyncBufferState> async = nullptr)
            : event(std::move(native_event)),
              pending_operation(std::move(pending)),
              async_state(std::move(async)) {}

        ~SharedState() {
            try {
                if (pending_operation != nullptr) {
                    (void)pending_operation->finish(kEventWaitTimeoutMs);
                } else if (event != nullptr) {
                    (void)event->finish(kEventWaitTimeoutMs);
                }
            } catch (...) {
            }
        }

        transport::TransportStatus finish_on_current_stream() {
            runtime::StreamIdentity stream;
            if (event == nullptr)
                return transport::TransportStatus::invalid(
                    "current_stream_wait",
                    "operation event is unavailable");
            const auto stream_status = event->current_stream(&stream);
            if (!stream_status.ok())
                return stream_status;

            std::lock_guard<std::mutex> lock(mutex);
            return pending_operation != nullptr ?
                pending_operation->finish(kEventWaitTimeoutMs) :
                event->finish(kEventWaitTimeoutMs);
        }

        std::shared_ptr<runtime::NativeEventState> event;
        std::shared_ptr<elastic::PendingOperation> pending_operation;
        std::shared_ptr<elastic::AsyncBufferState> async_state;
        std::mutex mutex;
    };

    static std::shared_ptr<SharedState> capture_current_event() {
        const auto api = runtime::make_stream_event_api();
        if (api.current_stream == nullptr)
            raise_event_status(transport::TransportStatus::invalid(
                "capture", "Torch NPU stream runtime is unavailable"));
        runtime::StreamIdentity stream;
        const int result = api.current_stream(api.user_data, &stream);
        if (result != 0)
            raise_event_status(transport::TransportStatus::runtime_failure(
                "capture", result, "Torch NPU current stream lookup failed"));
        if (stream.raw == nullptr || stream.device_index < 0)
            raise_event_status(transport::TransportStatus::invalid(
                "capture", "Torch NPU returned an invalid current stream"));
        auto created = runtime::create_native_event(api, stream.device_index);
        if (!created.status.ok())
            raise_event_status(created.status);
        const auto record_status = created.event->record(stream);
        if (!record_status.ok())
            raise_event_status(record_status);
        return std::make_shared<SharedState>(std::move(created.event));
    }

    std::shared_ptr<SharedState> state_;

public:
    EventHandle() : state_(capture_current_event()) {}

    EventHandle(
        std::shared_ptr<runtime::NativeEventState> event,
        std::shared_ptr<elastic::PendingOperation> pending_operation,
        std::shared_ptr<elastic::AsyncBufferState> async_state)
        : state_(std::make_shared<SharedState>(
              std::move(event), std::move(pending_operation),
              std::move(async_state))) {
        TORCH_CHECK(state_->event != nullptr,
                    "DeepEP Ascend backend: operation event is unavailable");
        TORCH_CHECK(state_->pending_operation != nullptr &&
                        state_->async_state != nullptr,
                    "DeepEP Ascend backend: operation event state is unavailable");
    }

    EventHandle(const EventHandle&) = default;
    EventHandle& operator=(const EventHandle&) = default;

    void current_stream_wait() const {
        TORCH_CHECK(state_ != nullptr,
                    "DeepEP Ascend backend: event state is unavailable");
        transport::TransportStatus status;
        {
            [[maybe_unused]] pybind11::gil_scoped_release release;
            status = state_->finish_on_current_stream();
        }
        if (!status.ok())
            raise_event_status(status);
    }

private:
    elastic::EventDependency dependency() const {
        TORCH_CHECK(state_ != nullptr && state_->event != nullptr,
                    "DeepEP Ascend backend: predecessor event is unavailable");
        return {state_->event, state_->pending_operation};
    }

    friend class ElasticBuffer;
};

class ElasticAsyncCompletionResources final
    : public elastic::AsyncCompletionResources {
public:
    ElasticAsyncCompletionResources(
        std::unique_ptr<runtime::CannRuntimeResources> resources,
        std::uint64_t dispatch_family,
        std::uint64_t last_dispatch_generation)
        : resources_(std::move(resources)), dispatch_family_(dispatch_family),
          last_dispatch_generation_(last_dispatch_generation) {}

    runtime::CannRuntimeResources* runtime() const noexcept {
        return resources_.get();
    }

    std::uint64_t dispatch_family() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return dispatch_family_;
    }

    std::uint64_t last_dispatch_generation() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_dispatch_generation_;
    }

    void stage_dispatch_descriptor(
        std::uint64_t generation, torch::Tensor tensor,
        elastic::DispatchHandleDescriptor descriptor) {
        std::lock_guard<std::mutex> lock(mutex_);
        staged_dispatch_ = StagedDispatch{
            generation, std::move(tensor), descriptor};
    }

    void commit_dispatch_generation(std::uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_dispatch_generation_ = generation;
    }

    transport::TransportStatus read_diagnostic(
        elastic::BufferOperationKind, std::uint64_t,
        transport::DeviceTransportDiagnostic* output) override {
        if (resources_ == nullptr || resources_->transport() == nullptr ||
            output == nullptr)
            return transport::TransportStatus::invalid(
                "read_diagnostic", "completion resources are unavailable");
        return resources_->transport()->read_diagnostic(output);
    }

    transport::TransportStatus read_completion(
        elastic::BufferOperationKind kind, std::uint64_t completion_offset,
        std::uint64_t* output) override {
        if (resources_ == nullptr || output == nullptr)
            return transport::TransportStatus::invalid(
                "read_completion", "completion resources are unavailable");
        const auto* address = reinterpret_cast<const void*>(
            reinterpret_cast<std::uintptr_t>(resources_->window_base()) +
            completion_offset);
        auto status = resources_->copy_to_host(
            output, address, sizeof(*output));
        if (!status.ok() || kind != elastic::BufferOperationKind::kDispatch)
            return status;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!staged_dispatch_.has_value() ||
            staged_dispatch_->generation != *output)
            return transport::TransportStatus::invalid(
                "dispatch", "generation-bound descriptor commit is unavailable");
        status = resources_->copy_from_host(
            staged_dispatch_->tensor.data_ptr(), &staged_dispatch_->descriptor,
            sizeof(staged_dispatch_->descriptor));
        if (!status.ok())
            return status;
        last_dispatch_generation_ = staged_dispatch_->generation;
        staged_dispatch_.reset();
        return transport::TransportStatus::success();
    }

    transport::TransportStatus destroy() override {
        if (resources_ == nullptr)
            return transport::TransportStatus::success();
        const auto status = resources_->destroy();
        if (status.ok())
            resources_.reset();
        return status;
    }

private:
    struct StagedDispatch {
        std::uint64_t generation = 0;
        torch::Tensor tensor;
        elastic::DispatchHandleDescriptor descriptor{};
    };

    std::unique_ptr<runtime::CannRuntimeResources> resources_;
    mutable std::mutex mutex_;
    std::uint64_t dispatch_family_ = 0;
    std::uint64_t last_dispatch_generation_ = 0;
    std::optional<StagedDispatch> staged_dispatch_;
};

class ElasticBuffer {
    int rank_idx_;
    int num_ranks_;
    int64_t num_buffer_bytes_;
    bool allow_hybrid_mode_;
    bool allow_multiple_reduction_;
    runtime::CannRuntimeResources* resources_ = nullptr;
    std::shared_ptr<ElasticAsyncCompletionResources> completion_resources_;
    std::shared_ptr<elastic::AsyncBufferState> async_state_;
    std::uint64_t barrier_timeout_cycles_ = 0;

    inline static std::atomic_uint64_t next_dispatch_family_{0};

    static constexpr auto kDispatchCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(transport::TransportCapability::kScaleUpTeam);

    transport::HostTransport* host_transport() const {
        TORCH_CHECK(resources_ != nullptr && resources_->transport() != nullptr,
                    "DeepEP Ascend backend: runtime is destroyed");
        return resources_->transport();
    }

    elastic::BufferOperationCoordinator::OperationLease reserve_operation(
        elastic::BufferOperationKind kind, const char* operation) const {
        TORCH_CHECK(async_state_ != nullptr,
                    "DeepEP Ascend backend: runtime is destroyed");
        auto lease = async_state_->coordinator().reserve(kind);
        if (lease.valid())
            return lease;
        switch (lease.status()) {
            case elastic::LeaseStatus::kBusy:
                TORCH_CHECK(false, "DeepEP Ascend backend: ", operation,
                            " is busy on this buffer");
            case elastic::LeaseStatus::kPoisoned:
            case elastic::LeaseStatus::kGenerationExhausted:
                TORCH_CHECK(false, "DeepEP Ascend backend: ", operation,
                            " cannot continue because this buffer is poisoned");
            case elastic::LeaseStatus::kDestroyed:
                TORCH_CHECK(false, "DeepEP Ascend backend: runtime is destroyed");
            case elastic::LeaseStatus::kAdmitted:
                break;
        }
        TORCH_CHECK(false, "DeepEP Ascend backend: ", operation,
                    " operation admission failed");
    }

    static std::uint64_t activate_operation(
        elastic::BufferOperationCoordinator::OperationLease& lease,
        const char* operation) {
        TORCH_CHECK(
            lease.activate(), "DeepEP Ascend backend: ", operation,
            " generation space is exhausted; this buffer is poisoned");
        return lease.generation();
    }

    static constexpr auto kCombineCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(transport::TransportCapability::kScaleUpTeam);

    transport::TransportCapabilities operation_capabilities(
        transport::TransportCapabilities base) const {
        return resources_->device_context().topology.kind ==
                transport::TransportTopologyKind::kPhysical2D ?
            base | transport::capability_bit(
                transport::TransportCapability::kScaleOutTeam) : base;
    }

    void require_transport(
        const char* operation, transport::TransportCapabilities required) const {
        const auto status =
            host_transport()->require_capabilities(required, operation);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
    }

    static const char* diagnostic_name(
        transport::DeviceTransportError error) {
        switch (error) {
            case transport::DeviceTransportError::kNone: return "none";
            case transport::DeviceTransportError::kInvalidAbi:
                return "invalid_abi";
            case transport::DeviceTransportError::kInvalidRank:
                return "invalid_rank";
            case transport::DeviceTransportError::kInvalidChannel:
                return "invalid_channel";
            case transport::DeviceTransportError::kInvalidAddress:
                return "invalid_address";
            case transport::DeviceTransportError::kInvalidProtocol:
                return "invalid_protocol";
            case transport::DeviceTransportError::kInvalidQueue:
                return "invalid_queue";
            case transport::DeviceTransportError::kUnsupportedOperation:
                return "unsupported_operation";
            case transport::DeviceTransportError::kCommandOverflow:
                return "command_overflow";
            case transport::DeviceTransportError::kCompletionTimeout:
                return "completion_timeout";
            case transport::DeviceTransportError::kCompletionFailure:
                return "completion_failure";
        }
        return "unknown";
    }

    struct CombineLifecycleSnapshot {
        std::uint64_t scratch_status = 0;
        transport::TransportCommandQueue queue{};
        transport::TransportServiceState service{};
    };

    transport::TransportStatus read_combine_lifecycle_snapshot(
        std::uint64_t scratch_status_offset,
        CombineLifecycleSnapshot* snapshot) const {
        *snapshot = {};
        const auto* scratch_status = static_cast<const std::uint8_t*>(
            resources_->workspace()) + scratch_status_offset;
        auto status = resources_->copy_to_host(
            &snapshot->scratch_status, scratch_status,
            sizeof(snapshot->scratch_status));
        if (!status.ok())
            return status;

        transport::StagedTransportContext staged{};
        const auto& context = resources_->device_context();
        if (context.backend_context == 0)
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "staged transport context is unavailable");
        status = resources_->copy_to_host(
            &staged, reinterpret_cast<const void*>(context.backend_context),
            sizeof(staged));
        if (!status.ok())
            return status;
        if (!transport::command::valid_staged_context_header(
                staged.abi_version, staged.struct_size,
                staged.cann_compatibility, staged.command_queue))
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "malformed staged transport context");

        status = resources_->copy_to_host(
            &snapshot->queue,
            reinterpret_cast<const void*>(staged.command_queue),
            sizeof(snapshot->queue));
        if (!status.ok())
            return status;
        if (!transport::command::valid_command_queue_header(
                snapshot->queue.abi_version, snapshot->queue.struct_size,
                snapshot->queue.commands, snapshot->queue.capacity,
                snapshot->queue.count, snapshot->queue.service_state,
                snapshot->queue.diagnostic))
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "malformed transport command queue");
        if (!transport::command::valid_registration_cookie(
                staged.reserved, staged.command_queue,
                snapshot->queue.commands, snapshot->queue.service_state,
                snapshot->queue.diagnostic, snapshot->queue.capacity))
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "malformed transport registration cookie");

        status = resources_->copy_to_host(
            &snapshot->service,
            reinterpret_cast<const void*>(snapshot->queue.service_state),
            sizeof(snapshot->service));
        if (!status.ok())
            return status;
        if (!transport::command::valid_service_state_header(
                snapshot->service.abi_version,
                snapshot->service.struct_size))
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "malformed transport service state");
        return transport::TransportStatus::success();
    }

    [[noreturn]] void raise_barrier_diagnostic(
        const transport::DeviceTransportDiagnostic& diagnostic,
        const char* detail) const {
        auto message = std::string("device diagnostic ") + detail +
            " error=" + diagnostic_name(diagnostic.error) +
            " command_index=" + std::to_string(diagnostic.command_index) +
            " opcode=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.opcode)) +
            " peer=" + std::to_string(diagnostic.peer) +
            " world_peer=" + std::to_string(diagnostic.world_peer) +
            " team=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.team)) +
            " channel=" + std::to_string(diagnostic.channel) +
            " backend_status=" + std::to_string(diagnostic.backend_status) +
            " reserved=" + std::to_string(diagnostic.reserved) +
            " generation=" + std::to_string(diagnostic.generation);
        raise_transport_status(
            transport::TransportStatus::runtime_failure(
                "barrier", static_cast<int>(diagnostic.backend_status),
                std::move(message)),
            rank_idx_);
    }

    [[noreturn]] void raise_dispatch_diagnostic(
        const transport::DeviceTransportDiagnostic& diagnostic,
        const char* detail) const {
        auto message = std::string("device diagnostic ") + detail +
            " error=" + diagnostic_name(diagnostic.error) +
            " command_index=" + std::to_string(diagnostic.command_index) +
            " opcode=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.opcode)) +
            " peer=" + std::to_string(diagnostic.peer) +
            " world_peer=" + std::to_string(diagnostic.world_peer) +
            " team=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.team)) +
            " channel=" + std::to_string(diagnostic.channel) +
            " backend_status=" + std::to_string(diagnostic.backend_status) +
            " reserved=" + std::to_string(diagnostic.reserved) +
            " generation=" + std::to_string(diagnostic.generation);
        raise_transport_status(
            transport::TransportStatus::runtime_failure(
                "dispatch", static_cast<int>(diagnostic.backend_status),
                std::move(message)),
            rank_idx_);
    }

    [[noreturn]] void raise_combine_diagnostic(
        const transport::DeviceTransportDiagnostic& diagnostic,
        const char* detail, std::uint64_t scratch_status_offset) const {
        auto message = std::string("device diagnostic ") + detail +
            " error=" + diagnostic_name(diagnostic.error) +
            " command_index=" + std::to_string(diagnostic.command_index) +
            " opcode=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.opcode)) +
            " peer=" + std::to_string(diagnostic.peer) +
            " world_peer=" + std::to_string(diagnostic.world_peer) +
            " team=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.team)) +
            " channel=" + std::to_string(diagnostic.channel) +
            " backend_status=" + std::to_string(diagnostic.backend_status) +
            " reserved=" + std::to_string(diagnostic.reserved) +
            " generation=" + std::to_string(diagnostic.generation) +
            " diagnostic_generation=" +
                std::to_string(diagnostic.generation) +
            " diagnostic_error=" + diagnostic_name(diagnostic.error);
        CombineLifecycleSnapshot snapshot{};
        const auto snapshot_status = read_combine_lifecycle_snapshot(
            scratch_status_offset, &snapshot);
        if (snapshot_status.ok()) {
            const auto encoded_peer = snapshot.scratch_status >> 32U;
            const auto scratch_peer = encoded_peer == 0 ? std::int64_t{-1} :
                static_cast<std::int64_t>(encoded_peer - 1);
            message +=
                " lifecycle_snapshot=available scratch_status=" +
                std::to_string(snapshot.scratch_status) +
                " scratch_peer=" + std::to_string(scratch_peer) +
                " scratch_error=" + std::to_string(
                    static_cast<std::uint32_t>(snapshot.scratch_status)) +
                " queue_generation=" +
                std::to_string(snapshot.queue.generation) +
                " queue_count=" + std::to_string(snapshot.queue.count) +
                " consumed_generation=" +
                std::to_string(snapshot.service.consumed_generation);
        } else {
            message +=
                " lifecycle_snapshot=unavailable snapshot_operation=" +
                snapshot_status.operation +
                " snapshot_backend_code=" +
                std::to_string(snapshot_status.backend_code) +
                " snapshot_error=" + snapshot_status.message;
        }
        raise_transport_status(
            transport::TransportStatus::runtime_failure(
                "combine", static_cast<int>(diagnostic.backend_status),
                std::move(message)),
            rank_idx_);
    }

    elastic::CoreTiling build_barrier_tiling() const {
        const auto& context = resources_->device_context();
        elastic::CoreTilingInput input{};
        input.operation = elastic::OperationKind::kBarrier;
        input.topology.world_rank = context.topology.world_rank;
        input.topology.world_size = context.topology.world_size;
        input.topology.scale_up_rank = context.topology.scale_up_rank;
        input.topology.scale_up_size = context.topology.scale_up_size;
        input.topology.scale_out_rank = context.topology.scale_out_rank;
        input.topology.scale_out_size = context.topology.scale_out_size;
        input.topology.kind = context.topology.kind;
        input.topology.epoch = context.topology.epoch;
        elastic::CoreTiling tiling{};
        const auto status = elastic::build_core_tiling(input, &tiling);
        TORCH_CHECK(status.ok(), "DeepEP Ascend backend: barrier ",
                    status.message);
        tiling.transport_context = context;
        return tiling;
    }

    elastic::CoreTiling build_dispatch_tiling(
        std::uint64_t num_tokens, std::uint64_t hidden,
        std::uint64_t num_experts, std::uint64_t num_topk,
        std::uint64_t expert_alignment,
        std::uint64_t num_max_tokens_per_rank,
        elastic::CoreModeFlags mode_flags,
        elastic::ElementKind element_kind,
        std::uint64_t num_scale_factor_packs) const {
        const auto& context = resources_->device_context();
        elastic::CoreTilingInput input{};
        input.operation = elastic::OperationKind::kDispatch;
        input.element_kind = element_kind;
        input.mode_flags = mode_flags;
        input.num_tokens = num_tokens;
        input.hidden = hidden;
        input.num_experts = num_experts;
        input.num_topk = num_topk;
        input.expert_alignment = expert_alignment;
        input.num_max_tokens_per_rank = num_max_tokens_per_rank;
        input.num_scale_factor_packs = num_scale_factor_packs;
        input.scale_factor_pack_bytes = num_scale_factor_packs == 0 ? 0 : 4;
        input.has_reusable_slots =
            elastic::has_mode(mode_flags, elastic::CoreMode::kCached);
        input.topology.world_rank = context.topology.world_rank;
        input.topology.world_size = context.topology.world_size;
        input.topology.scale_up_rank = context.topology.scale_up_rank;
        input.topology.scale_up_size = context.topology.scale_up_size;
        input.topology.scale_out_rank = context.topology.scale_out_rank;
        input.topology.scale_out_size = context.topology.scale_out_size;
        input.topology.kind = context.topology.kind;
        input.topology.epoch = context.topology.epoch;
        elastic::CoreTiling tiling{};
        const auto status = elastic::build_core_tiling(input, &tiling);
        TORCH_CHECK(status.ok(), "DeepEP Ascend backend: dispatch ",
                    status.message);
        tiling.transport_context = context;
        return tiling;
    }

    elastic::CoreTiling build_combine_tiling(
        const elastic::DispatchHandleDescriptor& descriptor,
        elastic::CoreModeFlags mode_flags) const {
        const auto& context = resources_->device_context();
        elastic::CoreTilingInput input{};
        input.operation = elastic::OperationKind::kCombine;
        input.element_kind = elastic::ElementKind::kBFloat16;
        input.mode_flags = mode_flags;
        input.num_tokens = descriptor.num_tokens;
        input.hidden = descriptor.hidden;
        input.num_experts = descriptor.num_experts;
        input.num_topk = descriptor.num_topk;
        input.expert_alignment = descriptor.expert_alignment;
        input.num_max_tokens_per_rank = descriptor.num_max_tokens_per_rank;
        input.topology = descriptor.topology;
        elastic::CoreTiling tiling{};
        const auto status = elastic::build_core_tiling(input, &tiling);
        TORCH_CHECK(status.ok(), "DeepEP Ascend backend: combine ",
                    status.message);
        tiling.transport_context = context;
        return tiling;
    }

    static void validate_npu_tensor(
        const torch::Tensor& tensor, int64_t dimensions,
        torch::ScalarType type, const torch::Device& device,
        const char* name) {
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                        tensor.device() == device &&
                        tensor.is_contiguous() &&
                        tensor.dim() == dimensions &&
                        tensor.scalar_type() == type,
                    "DeepEP Ascend backend: dispatch requires contiguous NPU ",
                    name, " with the expected rank and dtype");
    }

    static bool align_without_overflow(
        std::uint64_t value, std::uint64_t alignment,
        std::uint64_t* output) {
        if (alignment == 0 ||
            value > std::numeric_limits<std::uint64_t>::max() -
                        (alignment - 1))
            return false;
        *output = ((value + alignment - 1) / alignment) * alignment;
        return true;
    }

    static void raise_launch_status(
        const elastic::CoreRuntimeStatus& status, int rank) {
        const auto transport_status = status.code ==
                elastic::CoreRuntimeStatusCode::kLaunchFailure ?
            transport::TransportStatus::runtime_failure(
                "dispatch", status.backend_code, status.message) :
            transport::TransportStatus::invalid("dispatch", status.message);
        raise_transport_status(transport_status, rank);
    }

    static void raise_combine_launch_status(
        const elastic::CoreRuntimeStatus& status, int rank) {
        const auto transport_status = status.code ==
                elastic::CoreRuntimeStatusCode::kLaunchFailure ?
            transport::TransportStatus::runtime_failure(
                "combine", status.backend_code, status.message) :
            transport::TransportStatus::invalid("combine", status.message);
        raise_transport_status(transport_status, rank);
    }

#if DEEP_EP_ASCEND_TESTING
    struct TestingTag {};
    ElasticBuffer(TestingTag, int rank, int num_ranks,
                  std::unique_ptr<runtime::CannRuntimeResources> resources,
                  std::int64_t buffer_bytes, std::uint64_t timeout_cycles,
                  bool allow_multiple_reduction,
                  std::uint64_t dispatch_family,
                  std::uint64_t last_dispatch_generation,
                  bool allow_hybrid_mode)
        : rank_idx_(rank), num_ranks_(num_ranks),
          num_buffer_bytes_(buffer_bytes),
          allow_hybrid_mode_(allow_hybrid_mode),
          allow_multiple_reduction_(allow_multiple_reduction),
          barrier_timeout_cycles_(timeout_cycles) {
        completion_resources_ =
            std::make_shared<ElasticAsyncCompletionResources>(
                std::move(resources), dispatch_family,
                last_dispatch_generation);
        resources_ = completion_resources_->runtime();
        async_state_ = std::make_shared<elastic::AsyncBufferState>(
            completion_resources_, 5000);
    }
#endif

public:
    using cpu_comm_t = std::vector<std::pair<int, int>>;

    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& comm_handle, const cpu_comm_t& cpu_comm,
                  const int64_t& num_buffer_bytes,
                  const int64_t& num_cpu_buffer_bytes,
                  const bool& allow_hybrid_mode,
                  const bool& allow_multiple_reduction, const bool&,
                  const int& sl_idx, const int& num_allocated_qps,
                  const int&, const int& num_gpu_timeout_secs, const bool&)
        : rank_idx_(rank_idx), num_ranks_(num_ranks),
          num_buffer_bytes_(num_buffer_bytes),
          allow_hybrid_mode_(allow_hybrid_mode),
          allow_multiple_reduction_(allow_multiple_reduction) {
        TORCH_CHECK(num_ranks >= 2,
                    "DeepEP Ascend backend: world_size must be at least two");
        TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks,
                    "DeepEP Ascend backend: rank must be in [0, world_size)");
        TORCH_CHECK(comm_handle != 0,
                    "DeepEP Ascend backend: communicator_handle must be nonzero");
        TORCH_CHECK(cpu_comm.empty(),
                    "DeepEP Ascend backend: cpu communicator must be empty");
        TORCH_CHECK(num_cpu_buffer_bytes == 0,
                    "DeepEP Ascend backend: cpu_buffer_bytes must be zero");
        TORCH_CHECK(num_allocated_qps == 0,
                    "DeepEP Ascend backend: CUDA QP count must be zero");
        TORCH_CHECK(
            elastic::timeout_cycles_from_seconds(
                num_gpu_timeout_secs, &barrier_timeout_cycles_),
            "DeepEP Ascend backend: num_gpu_timeout_secs must be positive");
        TORCH_CHECK(
            num_buffer_bytes >=
                    static_cast<std::int64_t>(
                        elastic::kPublicElasticBufferAlignment) &&
                num_buffer_bytes % static_cast<std::int64_t>(
                    elastic::kPublicElasticBufferAlignment) == 0,
            "DeepEP Ascend backend: device buffer must be positive and "
            "2 MiB-aligned");
        const auto previous_family = next_dispatch_family_.fetch_add(
            1, std::memory_order_relaxed);
        TORCH_CHECK(previous_family != std::numeric_limits<std::uint64_t>::max(),
                    "DeepEP Ascend backend: dispatch family space is exhausted");
        const auto dispatch_family = previous_family + 1;

        transport::TransportConfig config{
            rank_idx, num_ranks, comm_handle, cpu_comm.empty(), num_buffer_bytes,
            num_cpu_buffer_bytes, allow_hybrid_mode, sl_idx, 1};
        const auto topology_status =
            transport::configure_transport_topology_from_environment(&config);
        if (!topology_status.ok())
            raise_transport_status(topology_status, rank_idx_);
        auto resources = std::make_unique<runtime::CannRuntimeResources>();
        const auto status = resources->initialize(
            config, 2 * elastic::kPublicElasticBufferAlignment);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        completion_resources_ =
            std::make_shared<ElasticAsyncCompletionResources>(
                std::move(resources), dispatch_family, 0);
        resources_ = completion_resources_->runtime();
        async_state_ = std::make_shared<elastic::AsyncBufferState>(
            completion_resources_, 5000);
        if (allow_hybrid_mode_) {
            const auto& topology = resources_->device_context().topology;
            TORCH_CHECK(topology.scale_up_size == 2 &&
                            topology.scale_out_size == 2,
                        "DeepEP Ascend backend: hybrid mode requires the "
                        "supported 2x2 scale-out/scale-up topology");
        }
    }

#if DEEP_EP_ASCEND_TESTING
    static std::unique_ptr<ElasticBuffer> make_testing_buffer(
        int rank, std::unique_ptr<runtime::CannRuntimeResources> resources,
        std::int64_t buffer_bytes, std::uint64_t timeout_cycles,
        bool allow_multiple_reduction = true,
        std::uint64_t dispatch_family = 7,
        int num_ranks = 2,
        std::uint64_t last_dispatch_generation = 0,
        bool allow_hybrid_mode = false) {
        TORCH_CHECK(num_ranks >= 2 && rank >= 0 && rank < num_ranks,
                    "DeepEP Ascend backend: invalid testing topology");
        const auto topology_matches = [&] {
            if (resources == nullptr || !resources->initialized())
                return false;
            const auto& topology = resources->device_context().topology;
            if (topology.world_rank != rank ||
                topology.world_size != num_ranks)
                return false;
            return allow_hybrid_mode ?
                topology.scale_up_size == 2 &&
                    topology.scale_out_size == 2 &&
                    topology.scale_up_rank == rank % 2 &&
                    topology.scale_out_rank == rank / 2 :
                topology.scale_up_rank == rank &&
                    topology.scale_up_size == num_ranks &&
                    topology.scale_out_rank == 0 &&
                    topology.scale_out_size == 1;
        }();
        TORCH_CHECK(topology_matches,
                    "DeepEP Ascend backend: testing topology must match runtime resources");
        return std::unique_ptr<ElasticBuffer>(new ElasticBuffer(
            TestingTag{}, rank, num_ranks, std::move(resources), buffer_bytes,
            timeout_cycles, allow_multiple_reduction, dispatch_family,
            last_dispatch_generation, allow_hybrid_mode));
    }

    std::size_t testing_dispatch_validation_state_bytes() const noexcept {
        return 2 * sizeof(std::uint64_t);
    }

    std::uint64_t testing_operation_generation() const noexcept {
        return async_state_ == nullptr ? 0 :
            async_state_->coordinator().last_generation();
    }
#endif

    void destroy() {
        if (async_state_ == nullptr)
            return;
        const auto status = async_state_->destroy();
        if (!status.ok() && status.operation == "destroy_async_state" &&
            status.message == "operation coordinator is busy")
            TORCH_CHECK(false,
                        "DeepEP Ascend backend: destroy is busy on this buffer");
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        resources_ = nullptr;
        completion_resources_.reset();
        async_state_.reset();
    }

    c10::Stream get_comm_stream() const {
        TORCH_CHECK(resources_ != nullptr,
                    "DeepEP Ascend backend: runtime is destroyed");
        const auto& stream = resources_->comm_stream();
        TORCH_CHECK(stream.raw != nullptr &&
                        stream.device_index == resources_->owning_device(),
                    "DeepEP Ascend backend: communication stream is unavailable");
        return c10::Stream::unpack3(
            static_cast<c10::StreamId>(stream.stream_id),
            static_cast<c10::DeviceIndex>(stream.device_index),
            static_cast<c10::DeviceType>(stream.device_type));
    }

    std::tuple<int, int> get_physical_domain_size() const {
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kTopologyQuery,
            "get_physical_domain_size");
        transport::TransportTopology topology;
        auto status = host_transport()->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_physical_domain_size";
            raise_transport_status(status, rank_idx_);
        }
        const auto domain = transport::physical_transport_domain_size(
            topology, host_transport()->capabilities());
        lease.complete();
        return {domain.first, domain.second};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kTopologyQuery,
            "get_logical_domain_size");
        transport::TransportTopology topology;
        auto status = host_transport()->query_topology(&topology);
        if (!status.ok()) {
            status.operation = "get_logical_domain_size";
            raise_transport_status(status, rank_idx_);
        }
        lease.complete();
        return {topology.scale_out_size, topology.scale_up_size};
    }

    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync,
                 const bool& sequential) {
        TORCH_CHECK(sequential,
                    "DeepEP Ascend backend: barrier requires sequential=True");
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kBarrier, "barrier");
        require_transport("barrier", elastic::kBarrierTransportCapabilities);

        if (with_cpu_sync) {
            const auto status = resources_->synchronize_device();
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }

        auto tiling = build_barrier_tiling();
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_),
            resources_->workspace_bytes()};
        runtime::StreamIdentity stream;
        auto status = resources_->current_stream(&stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        elastic::EventDependency dependency;
        if (use_comm_stream) {
            auto created = resources_->create_event();
            if (!created.status.ok())
                raise_transport_status(created.status, rank_idx_);
            status = created.event->record(stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            dependency.event = std::move(created.event);
            stream = resources_->comm_stream();
            status = dependency.event->wait(stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        const auto generation = activate_operation(lease, "barrier");
        const elastic::BarrierArguments arguments{
            resources_->workspace(), generation, barrier_timeout_cycles_};
        const auto launch_status = elastic::launch_internal_barrier(
            arguments, tiling, storage, stream.raw);
        if (!launch_status.ok()) {
            const auto status = launch_status.code ==
                    elastic::CoreRuntimeStatusCode::kLaunchFailure ?
                transport::TransportStatus::runtime_failure(
                    "barrier", launch_status.backend_code,
                    launch_status.message) :
                transport::TransportStatus::invalid(
                    "barrier", launch_status.message);
            raise_transport_status(status, rank_idx_);
        }

        if (use_comm_stream) {
            std::uint64_t completion_offset = 0;
            TORCH_CHECK(
                elastic::checked_rank_slot_offset(
                    tiling.symmetric_window_layout.barrier_completion_offset,
                    tiling.symmetric_window_layout.barrier_completion_count,
                    rank_idx_, &completion_offset),
                "Invalid Ascend barrier completion slot for rank ", rank_idx_);
            auto completion = resources_->create_event();
            if (!completion.status.ok())
                raise_transport_status(completion.status, rank_idx_);
            status = completion.event->record(stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            std::vector<elastic::EventDependency> predecessors;
            predecessors.emplace_back(std::move(dependency));
            auto published = async_state_->publish(
                std::move(lease), completion.event,
                {elastic::BufferOperationKind::kBarrier, generation,
                 completion_offset,
                 tiling.workspace_layout.scratch_status_offset},
                {}, std::move(predecessors));
            if (!published.status.ok())
                raise_transport_status(published.status, rank_idx_);
            status = published.operation->finish(5000);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            if (with_cpu_sync) {
                status = resources_->synchronize_device();
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
            }
            return;
        }

        status = resources_->synchronize_stream(stream.raw);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        transport::DeviceTransportDiagnostic diagnostic{};
        status = host_transport()->read_diagnostic(&diagnostic);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
#if DEEP_EP_ASCEND_TESTING
        if (std::getenv("DEEP_EP_ASCEND_TEST_DIAGNOSTIC") != nullptr) {
            diagnostic.abi_version = transport::kTransportCommandAbiVersion;
            diagnostic.error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic.command_index = 0;
            diagnostic.opcode = transport::TransportCommandOpcode::kBarrier;
            diagnostic.peer = static_cast<std::uint32_t>(rank_idx_);
            diagnostic.channel = 0;
            diagnostic.backend_status = 0;
            diagnostic.generation = generation;
        }
#endif

        std::uint64_t completion = 0;
        std::uint64_t completion_offset = 0;
        TORCH_CHECK(
            elastic::checked_rank_slot_offset(
                tiling.symmetric_window_layout.barrier_completion_offset,
                tiling.symmetric_window_layout.barrier_completion_count,
                rank_idx_, &completion_offset),
            "Invalid Ascend barrier completion slot for rank ", rank_idx_);
        const auto* completion_address = reinterpret_cast<const void*>(
            reinterpret_cast<std::uintptr_t>(resources_->window_base()) +
            completion_offset);
        status = resources_->copy_to_host(
            &completion, completion_address, sizeof(completion));
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        if (with_cpu_sync) {
            status = resources_->synchronize_device();
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }

        if (diagnostic.abi_version !=
                transport::kTransportCommandAbiVersion ||
            diagnostic.error != transport::DeviceTransportError::kNone)
            raise_barrier_diagnostic(diagnostic, "reported failure");
        if (diagnostic.generation != generation)
            raise_barrier_diagnostic(diagnostic, "generation mismatch");
        if (completion != generation)
            raise_barrier_diagnostic(diagnostic, "completion mismatch");
        lease.complete();
    }

    static int64_t calculate_buffer_size(
        const int64_t& comm_handle, const int& num_max_tokens_per_rank,
        const int& hidden, int num_topk, const bool& use_fp8_dispatch,
        const bool& allow_hybrid_mode,
        const bool& allow_multiple_reduction) {
        TORCH_CHECK(
            comm_handle != 0,
            "DeepEP Ascend backend: calculate_elastic_buffer_size "
            "communicator_handle must be nonzero");
        TORCH_CHECK(
            num_max_tokens_per_rank > 0 && hidden > 0 && num_topk >= 0,
            "DeepEP Ascend backend: calculate_elastic_buffer_size requires "
            "positive token capacity and hidden size and nonnegative top-k");

        std::uint32_t world_size = 0;
        const auto query_status =
            transport::query_cann_communicator_size(comm_handle, &world_size);
        TORCH_CHECK(
            query_status.ok(),
            "DeepEP Ascend backend: calculate_elastic_buffer_size ",
            query_status.operation, " failed with backend error ",
            query_status.backend_code, ": ", query_status.message);
        return calculate_buffer_size_for_world_size(
            world_size, num_max_tokens_per_rank, hidden, num_topk,
            use_fp8_dispatch, allow_multiple_reduction, allow_hybrid_mode);
    }

#if DEEP_EP_ASCEND_TESTING
    static int64_t calculate_buffer_size_for_testing(
        const int64_t& comm_handle, const int& num_max_tokens_per_rank,
        const int& hidden, int num_topk, const bool& use_fp8_dispatch,
        const bool& allow_hybrid_mode,
        const bool& allow_multiple_reduction,
        const transport::CannHostApi& host_api) {
        TORCH_CHECK(
            comm_handle != 0,
            "DeepEP Ascend backend: calculate_elastic_buffer_size "
            "communicator_handle must be nonzero");
        TORCH_CHECK(
            num_max_tokens_per_rank > 0 && hidden > 0 && num_topk >= 0,
            "DeepEP Ascend backend: calculate_elastic_buffer_size requires "
            "positive token capacity and hidden size and nonnegative top-k");
        std::uint32_t world_size = 0;
        const auto query_status = transport::query_cann_communicator_size(
            comm_handle, &world_size, host_api);
        TORCH_CHECK(
            query_status.ok(),
            "DeepEP Ascend backend: calculate_elastic_buffer_size ",
            query_status.operation, " failed with backend error ",
            query_status.backend_code, ": ", query_status.message);
        return calculate_buffer_size_for_world_size(
            world_size, num_max_tokens_per_rank, hidden, num_topk,
            use_fp8_dispatch, allow_multiple_reduction, allow_hybrid_mode);
    }
#endif

private:
    static int64_t calculate_buffer_size_for_world_size(
        std::uint32_t world_size, int num_max_tokens_per_rank, int hidden,
        int num_topk, bool use_fp8_dispatch,
        bool allow_multiple_reduction, bool hybrid) {
        elastic::SymmetricWindowInput input{};
        input.world_size = static_cast<int>(world_size);
        input.num_max_tokens_per_rank =
            static_cast<std::uint64_t>(num_max_tokens_per_rank);
        input.hidden = static_cast<std::uint64_t>(hidden);
        input.num_topk = static_cast<std::uint64_t>(num_topk);
        input.element_bytes = use_fp8_dispatch ? 1 : 2;
        input.scale_factor_bytes = use_fp8_dispatch ?
            ((static_cast<std::uint64_t>(hidden) + 31U) / 32U) * 4U : 0U;
        input.expanded = true;
        input.allow_multiple_reduction = allow_multiple_reduction;
        input.hybrid = hybrid;
        input.hybrid_route_capacity =
            static_cast<std::uint64_t>(world_size) *
            static_cast<std::uint64_t>(num_max_tokens_per_rank);
        elastic::SymmetricWindowLayout layout{};
        const auto status =
            elastic::build_symmetric_window_layout(input, &layout);
        TORCH_CHECK(
            status.ok(), "DeepEP Ascend backend: calculate_elastic_buffer_size ",
            status.message);
        TORCH_CHECK(
            layout.total_bytes <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()),
            "DeepEP Ascend backend: calculate_elastic_buffer_size overflow");
        return static_cast<std::int64_t>(layout.total_bytes);
    }

public:

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, int, int, std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank, const int& num_experts,
             const int& expert_alignment, const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream, const bool& do_handle_copy,
             const bool& do_cpu_sync, const bool& do_expand,
             const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf) const {
        TORCH_CHECK(!cumulative_local_expert_recv_stats.has_value(),
                    "DeepEP Ascend backend: dispatch does not support "
                    "cumulative expert stats");
        const bool cached_mode = cached_num_recv_tokens.has_value();
        TORCH_CHECK(!previous_event_before_epilogue.has_value(),
                    "DeepEP Ascend backend: dispatch does not support "
                    "previous_event_before_epilogue");
        TORCH_CHECK(!previous_event.has_value() || allocate_on_comm_stream,
                    "DeepEP Ascend backend: dispatch previous_event requires "
                    "allocate_on_comm_stream=True");
        const bool stream_mode = previous_event.has_value() ||
            async_with_compute_stream || allocate_on_comm_stream;
        TORCH_CHECK(!stream_mode ||
                        (cached_mode && !sf.has_value() && !allow_hybrid_mode_),
                    "DeepEP Ascend backend: dispatch stream overlap requires "
                    "cached BF16 pure-scale-up mode");
        TORCH_CHECK(do_cpu_sync || cached_mode,
                    "DeepEP Ascend backend: dispatch requires do_cpu_sync unless cached");
        TORCH_CHECK(num_sms == 1 && num_qps == 0,
                    "DeepEP Ascend backend: dispatch requires num_sms=1 and num_qps=0");
        TORCH_CHECK(!do_zero_padding || do_expand,
                    "DeepEP Ascend backend: dispatch zero padding requires expansion");
        TORCH_CHECK(num_max_tokens_per_rank > 0 && num_experts > 0 &&
                        expert_alignment > 0 &&
                        num_experts % num_ranks_ == 0 &&
                        num_max_tokens_per_rank <=
                            std::numeric_limits<int>::max() / num_ranks_,
                    "DeepEP Ascend backend: dispatch requires positive, "
                    "rank-partitioned expert capacity");
        const auto device = x.device();
        const bool fp8_dispatch = x.scalar_type() == torch::kFloat8_e4m3fn;
        TORCH_CHECK(
            (fp8_dispatch && sf.has_value()) ||
                (x.scalar_type() == torch::kBFloat16 && !sf.has_value()),
            "DeepEP Ascend backend: dispatch requires BF16 x without scale "
            "factors or E4M3 x with scale factors");
        validate_npu_tensor(
            x, 2, fp8_dispatch ? torch::kFloat8_e4m3fn : torch::kBFloat16,
            device, fp8_dispatch ? "E4M3 x" : "BF16 x");
        TORCH_CHECK(fp8_dispatch || !use_tma_aligned_col_major_sf,
                    "DeepEP Ascend backend: dispatch column-major scale "
                    "factors require FP8");
        validate_npu_tensor(topk_idx, 2, torch::kLong, device, "int64 topk_idx");
        TORCH_CHECK(x.size(0) == topk_idx.size(0) && x.size(0) >= 0 &&
                        x.size(0) <= num_max_tokens_per_rank && x.size(1) > 0 &&
                        topk_idx.size(1) > 0 && topk_idx.size(1) <= num_experts,
                    "DeepEP Ascend backend: dispatch input shapes exceed capacity");
        if (topk_weights.has_value()) {
            validate_npu_tensor(
                *topk_weights, 2, torch::kFloat, device, "float32 topk_weights");
            TORCH_CHECK(topk_weights->sizes() == topk_idx.sizes(),
                        "DeepEP Ascend backend: dispatch topk_weights shape mismatch");
        }
        std::uint64_t num_scale_factor_packs = 0;
        if (sf.has_value()) {
            TORCH_CHECK(
                sf->device().type() == c10::DeviceType::PrivateUse1 &&
                    sf->device() == device && sf->dim() == 2 &&
                    (sf->scalar_type() == torch::kFloat ||
                     sf->scalar_type() == torch::kInt) &&
                    sf->size(0) == x.size(0) && sf->size(1) > 0 &&
                    sf->stride(0) > 0 && sf->stride(1) > 0,
                "DeepEP Ascend backend: dispatch scale factors require a "
                "same-device rank-2 float32 or int32 tensor with matching "
                "tokens, positive pack count, and positive strides");
            num_scale_factor_packs =
                static_cast<std::uint64_t>(sf->size(1));
            const auto max_scale_factor_packs =
                (static_cast<std::uint64_t>(x.size(1)) + 31U) / 32U;
            TORCH_CHECK(num_scale_factor_packs <= max_scale_factor_packs,
                        "DeepEP Ascend backend: dispatch scale factor packs "
                        "exceed the supported hidden-width bound");
        }
        int current_device = -1;
        auto device_status = resources_->current_device(&current_device);
        if (!device_status.ok())
            raise_transport_status(device_status, rank_idx_);
        TORCH_CHECK(device.index() == resources_->owning_device() &&
                        current_device == resources_->owning_device(),
                    "DeepEP Ascend backend: dispatch tensors and current NPU "
                    "must match the buffer device");
        if (allow_hybrid_mode_)
            require_transport("dispatch",
                              operation_capabilities(kDispatchCapabilities));
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kDispatch, "dispatch");
        if (!allow_hybrid_mode_)
            require_transport("dispatch",
                              operation_capabilities(kDispatchCapabilities));
        const bool any_cached_tensor = cached_num_expanded_tokens.has_value() ||
            cached_num_recv_tokens_per_expert_list.has_value() ||
            cached_psum_num_recv_tokens_per_scaleup_rank.has_value() ||
            cached_psum_num_recv_tokens_per_expert.has_value() ||
            cached_num_unaligned_recv_tokens_per_expert.has_value() ||
            cached_dst_buffer_slot_idx.has_value() ||
            cached_token_metadata_at_forward.has_value() ||
            cached_recv_src_metadata.has_value();
        TORCH_CHECK(cached_mode || !any_cached_tensor,
                    "DeepEP Ascend backend: dispatch cached handles require cached counts");
        TORCH_CHECK(!cached_channel_linked_list.has_value(),
                    "DeepEP Ascend backend: dispatch does not support channel handles");

        elastic::CoreModeFlags mode_flags = 0;
        if (cached_mode)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kCached);
        if (do_expand)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kExpanded);
        if (do_zero_padding)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kZeroPadding);
        if (allow_hybrid_mode_)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kHybrid);
        const auto num_tokens = static_cast<std::uint64_t>(x.size(0));
        const auto hidden = static_cast<std::uint64_t>(x.size(1));
        const auto num_topk = static_cast<std::uint64_t>(topk_idx.size(1));
        const auto capacity = static_cast<std::uint64_t>(num_max_tokens_per_rank);
        const auto experts = static_cast<std::uint64_t>(num_experts);
        const auto alignment = static_cast<std::uint64_t>(expert_alignment);
        const auto tiling = build_dispatch_tiling(
            num_tokens, hidden, experts, num_topk, alignment, capacity,
            mode_flags,
            fp8_dispatch ? elastic::ElementKind::kFloat8E4M3 :
                           elastic::ElementKind::kBFloat16,
            num_scale_factor_packs);
        TORCH_CHECK(tiling.communication_buffer_bytes <=
                        static_cast<std::uint64_t>(num_buffer_bytes_) &&
                        tiling.workspace_bytes <= resources_->workspace_bytes(),
                    "DeepEP Ascend backend: dispatch capacity exceeds runtime storage");

        const auto descriptor_mode_flags = mode_flags & ~elastic::mode_bit(
            elastic::CoreMode::kCached);
        const auto routing_mode = allow_hybrid_mode_ ?
            elastic::DispatchRoutingMode::kHybrid :
            elastic::DispatchRoutingMode::kDirect;
        const auto cached_route_count = allow_hybrid_mode_ && cached_mode ?
            static_cast<std::uint64_t>(*cached_num_recv_tokens) : 0;
        const auto int_options = x.options().dtype(torch::kInt);
        const auto metadata_options = x.options().dtype(torch::kByte);
        const auto max_recv_tokens = allow_hybrid_mode_ ?
            tiling.hybrid_route_capacity :
            capacity * static_cast<std::uint64_t>(num_ranks_);
        const auto local_experts = experts / static_cast<std::uint64_t>(num_ranks_);
        const auto expanded_records = tiling.dispatch_output_capacity;
        TORCH_CHECK(max_recv_tokens <= static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max()) &&
                        expanded_records <= static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()),
                    "DeepEP Ascend backend: dispatch output count overflow");

        runtime::StreamIdentity dispatch_stream;
        elastic::EventDependency predecessor;
        if (cached_mode) {
            auto status = resources_->current_stream(&dispatch_stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            if (previous_event.has_value()) {
                predecessor = previous_event->dependency();
            } else {
                auto created = resources_->create_event();
                if (!created.status.ok())
                    raise_transport_status(created.status, rank_idx_);
                status = created.event->record(dispatch_stream);
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
                predecessor.event = std::move(created.event);
            }
            dispatch_stream = resources_->comm_stream();
            status = predecessor.event->wait(dispatch_stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
#if DEEP_EP_ASCEND_HAS_GENERIC_STREAM_GUARD
        std::optional<c10::StreamGuard> allocation_guard;
        if (allocate_on_comm_stream)
            allocation_guard.emplace(get_comm_stream());
#endif

        torch::Tensor rank_prefix;
        torch::Tensor expert_prefix;
        torch::Tensor unaligned;
        torch::Tensor destination_slots;
        torch::Tensor source_metadata;
        auto kernel_expert_prefix = torch::empty(
            {num_experts + 1}, int_options);
        auto kernel_unaligned = torch::empty(
            {num_experts}, int_options);
        std::vector<std::int32_t> host_rank_prefix(num_ranks_);
        std::vector<std::int32_t> host_kernel_expert_prefix(num_experts + 1);
        std::vector<std::int32_t> host_kernel_unaligned(num_experts);
        std::vector<std::int32_t> host_expert_prefix(local_experts);
        std::vector<std::int32_t> host_unaligned(local_experts);
        std::vector<elastic::HybridRouteRecord> host_route_records;
        std::vector<int> per_expert_list;
        int num_expanded_tokens = 0;
        const int first_local_expert =
            rank_idx_ * static_cast<int>(local_experts);
        if (cached_mode) {
            TORCH_CHECK(cached_num_expanded_tokens.has_value() &&
                            cached_num_recv_tokens_per_expert_list.has_value() &&
                            cached_psum_num_recv_tokens_per_scaleup_rank.has_value() &&
                            cached_psum_num_recv_tokens_per_expert.has_value() &&
                            cached_num_unaligned_recv_tokens_per_expert.has_value() &&
                            cached_dst_buffer_slot_idx.has_value() &&
                            cached_token_metadata_at_forward.has_value() &&
                            cached_recv_src_metadata.has_value(),
                        "DeepEP Ascend backend: dispatch cached mode requires all handles");
            TORCH_CHECK(*cached_num_recv_tokens >= 0 &&
                            *cached_num_expanded_tokens >= 0 &&
                            *cached_num_recv_tokens <= static_cast<int>(max_recv_tokens) &&
                            *cached_num_expanded_tokens <= static_cast<int>(expanded_records) &&
                            cached_num_recv_tokens_per_expert_list->size() == local_experts,
                        "DeepEP Ascend backend: dispatch cached counts are invalid");
            validate_npu_tensor(*cached_psum_num_recv_tokens_per_scaleup_rank,
                                1, torch::kInt, device, "cached rank prefix");
            validate_npu_tensor(*cached_psum_num_recv_tokens_per_expert,
                                1, torch::kInt, device, "cached expert prefix");
            validate_npu_tensor(*cached_num_unaligned_recv_tokens_per_expert,
                                1, torch::kInt, device, "cached unaligned counts");
            validate_npu_tensor(*cached_dst_buffer_slot_idx,
                                2, torch::kInt, device, "cached destination slots");
            validate_npu_tensor(*cached_recv_src_metadata,
                                2, torch::kInt, device, "cached source metadata");
            validate_npu_tensor(*cached_token_metadata_at_forward,
                                1, torch::kByte, device, "cached descriptor");
            TORCH_CHECK(cached_psum_num_recv_tokens_per_scaleup_rank->size(0) ==
                                static_cast<int64_t>(num_ranks_) &&
                            cached_psum_num_recv_tokens_per_expert->size(0) ==
                                static_cast<int64_t>(local_experts) &&
                            cached_num_unaligned_recv_tokens_per_expert->size(0) ==
                                static_cast<int64_t>(local_experts) &&
                            cached_dst_buffer_slot_idx->sizes() == topk_idx.sizes() &&
                            cached_recv_src_metadata->size(0) == *cached_num_recv_tokens &&
                            cached_recv_src_metadata->size(1) ==
                                static_cast<int64_t>(num_topk + 2) &&
                            cached_token_metadata_at_forward->numel() ==
                                static_cast<int64_t>(
                                    sizeof(elastic::DispatchHandleDescriptor) +
                                    cached_route_count *
                                        sizeof(elastic::HybridRouteRecord)),
                        "DeepEP Ascend backend: dispatch cached handle shape mismatch");
            elastic::DispatchHandleDescriptor descriptor{};
            auto status = resources_->copy_to_host(
                &descriptor, cached_token_metadata_at_forward->data_ptr(),
                sizeof(descriptor));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            if (allow_hybrid_mode_) {
                host_route_records.resize(
                    static_cast<std::size_t>(cached_route_count));
                if (!host_route_records.empty()) {
                    const auto* device_records =
                        static_cast<const std::uint8_t*>(
                            cached_token_metadata_at_forward->data_ptr()) +
                        sizeof(elastic::DispatchHandleDescriptor);
                    status = resources_->copy_to_host(
                        host_route_records.data(), device_records,
                        host_route_records.size() *
                            sizeof(elastic::HybridRouteRecord));
                    if (!status.ok())
                        raise_transport_status(status, rank_idx_);
                }
                const auto expected_descriptor =
                    elastic::make_attested_hybrid_dispatch_handle_descriptor(
                        completion_resources_->dispatch_family(), tiling.topology,
                        completion_resources_->last_dispatch_generation(),
                        num_tokens, hidden, experts,
                        num_topk, alignment, capacity, descriptor_mode_flags,
                        elastic::kHybridRouteLayoutVersion, cached_route_count,
                        sizeof(elastic::HybridRouteRecord),
                        elastic::kHybridRouteCompleteStageFlags,
                        {host_route_records.data(),
                         host_route_records.size()});
                const auto descriptor_status =
                    elastic::validate_dispatch_handle(
                        expected_descriptor, descriptor);
                TORCH_CHECK(
                    descriptor_status.ok(),
                    "DeepEP Ascend backend: dispatch handle or hybrid route "
                    "record does not match the current call");
                const auto route_status = elastic::validate_hybrid_route_table(
                    descriptor,
                    {host_route_records.data(), host_route_records.size()},
                    capacity, max_recv_tokens, local_experts);
                TORCH_CHECK(route_status.ok(),
                            "DeepEP Ascend backend: dispatch ",
                            route_status.message);

                std::vector<std::int32_t> host_source_metadata(
                    host_route_records.size() *
                    static_cast<std::size_t>(num_topk + 2));
                std::vector<std::int64_t> host_received_topk(
                    host_route_records.size() *
                        static_cast<std::size_t>(num_topk),
                    -1);
                if (!host_route_records.empty()) {
                    status = resources_->copy_to_host(
                        host_source_metadata.data(),
                        cached_recv_src_metadata->data_ptr(),
                        host_source_metadata.size() * sizeof(std::int32_t));
                    if (!status.ok())
                        raise_transport_status(status, rank_idx_);
                    for (std::size_t index = 0;
                         index < host_route_records.size(); ++index) {
                        const auto encoded_lane = host_source_metadata[
                            index * static_cast<std::size_t>(num_topk + 2) + 1];
                        if (encoded_lane < 0)
                            continue;
                        const auto master_lane =
                            static_cast<std::uint64_t>(encoded_lane) % num_topk;
                        host_received_topk[
                            index * static_cast<std::size_t>(num_topk) +
                            static_cast<std::size_t>(master_lane)] =
                                host_route_records[index]
                                    .destination_local_expert;
                    }
                }
                const elastic::HybridRouteBindingView bindings{
                    host_source_metadata.data(), host_received_topk.data(),
                    host_route_records.size(), num_topk, capacity};
                const auto binding_status =
                    elastic::validate_hybrid_route_bindings(
                        descriptor,
                        {host_route_records.data(), host_route_records.size()},
                        bindings);
                TORCH_CHECK(binding_status.ok(),
                            "DeepEP Ascend backend: dispatch ",
                            binding_status.message);
            } else {
                const auto expected_descriptor =
                    elastic::make_attested_dispatch_handle_descriptor(
                        completion_resources_->dispatch_family(), tiling.topology,
                        completion_resources_->last_dispatch_generation(),
                        num_tokens, hidden, experts,
                        num_topk, alignment, capacity, descriptor_mode_flags);
                const auto descriptor_status =
                    elastic::validate_dispatch_handle(
                        expected_descriptor, descriptor);
                TORCH_CHECK(descriptor_status.ok(),
                            "DeepEP Ascend backend: dispatch ",
                            descriptor_status.message);
            }
            rank_prefix = *cached_psum_num_recv_tokens_per_scaleup_rank;
            expert_prefix = *cached_psum_num_recv_tokens_per_expert;
            unaligned = *cached_num_unaligned_recv_tokens_per_expert;
            destination_slots = *cached_dst_buffer_slot_idx;
            source_metadata = *cached_recv_src_metadata;
            status = resources_->copy_to_host(
                host_rank_prefix.data(), rank_prefix.data_ptr(),
                host_rank_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_expert_prefix.data(), expert_prefix.data_ptr(),
                host_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_unaligned.data(), unaligned.data_ptr(),
                host_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            TORCH_CHECK(host_rank_prefix.back() == *cached_num_recv_tokens,
                        "DeepEP Ascend backend: dispatch cached prefix tail mismatch");
            for (std::size_t index = 0; index < host_rank_prefix.size(); ++index)
                TORCH_CHECK(host_rank_prefix[index] >= 0 &&
                                (index == 0 || host_rank_prefix[index] >=
                                                    host_rank_prefix[index - 1]),
                            "DeepEP Ascend backend: dispatch cached rank prefix is invalid");
            std::int32_t expanded_tail = 0;
            for (int expert = 0; expert < num_experts; ++expert) {
                host_kernel_expert_prefix[expert] = expanded_tail;
                if (expert < first_local_expert ||
                    expert >= first_local_expert +
                                  static_cast<int>(local_experts))
                    continue;
                const int local_expert = expert - first_local_expert;
                const std::int32_t actual = host_unaligned[local_expert];
                std::uint64_t aligned_actual = 0;
                TORCH_CHECK(actual >= 0 && align_without_overflow(
                                static_cast<std::uint64_t>(actual), alignment,
                                &aligned_actual) &&
                                aligned_actual <= static_cast<std::uint64_t>(
                                    std::numeric_limits<int>::max()) &&
                                (*cached_num_recv_tokens_per_expert_list)[local_expert] ==
                                    static_cast<int>(aligned_actual),
                            "DeepEP Ascend backend: dispatch cached expert counts mismatch");
                TORCH_CHECK(aligned_actual <= static_cast<std::uint64_t>(
                                std::numeric_limits<int>::max() - expanded_tail),
                            "DeepEP Ascend backend: dispatch cached expert counts mismatch");
                const std::int32_t expected_public_prefix = do_expand ?
                    expanded_tail + actual :
                    expanded_tail + static_cast<std::int32_t>(aligned_actual);
                TORCH_CHECK(host_expert_prefix[local_expert] ==
                                expected_public_prefix,
                            "DeepEP Ascend backend: dispatch cached expert prefix is invalid");
                host_kernel_unaligned[expert] = actual;
                expanded_tail += static_cast<std::int32_t>(aligned_actual);
            }
            host_kernel_expert_prefix[num_experts] = expanded_tail;
            TORCH_CHECK(expanded_tail == *cached_num_expanded_tokens,
                        "DeepEP Ascend backend: dispatch cached prefix tail mismatch");
            status = resources_->copy_from_host(
                kernel_expert_prefix.data_ptr(), host_kernel_expert_prefix.data(),
                host_kernel_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_from_host(
                kernel_unaligned.data_ptr(), host_kernel_unaligned.data(),
                host_kernel_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            per_expert_list = *cached_num_recv_tokens_per_expert_list;
            num_expanded_tokens = *cached_num_expanded_tokens;
        } else {
            rank_prefix = torch::empty({num_ranks_}, int_options);
            expert_prefix = torch::empty(
                {static_cast<int64_t>(local_experts)}, int_options);
            unaligned = torch::empty(
                {static_cast<int64_t>(local_experts)}, int_options);
            destination_slots = torch::empty({x.size(0), topk_idx.size(1)}, int_options);
            source_metadata = torch::empty(
                {static_cast<int64_t>(max_recv_tokens), topk_idx.size(1) + 2},
                int_options);
        }

        auto recv_x = torch::empty(
            {static_cast<int64_t>(do_expand ? expanded_records : max_recv_tokens),
             x.size(1)}, x.options());
        auto recv_sf = std::optional<torch::Tensor>();
        if (sf.has_value()) {
            const auto allocated_tokens = static_cast<int64_t>(
                do_expand ? expanded_records : max_recv_tokens);
            const auto sf_token_stride = use_tma_aligned_col_major_sf ?
                int64_t{1} : static_cast<int64_t>(num_scale_factor_packs);
            std::uint64_t aligned_tokens = 0;
            TORCH_CHECK(
                !use_tma_aligned_col_major_sf ||
                    align_without_overflow(
                        static_cast<std::uint64_t>(allocated_tokens), 4,
                        &aligned_tokens),
                "DeepEP Ascend backend: dispatch scale factor output stride overflow");
            const auto sf_pack_stride = use_tma_aligned_col_major_sf ?
                static_cast<int64_t>(aligned_tokens) : int64_t{1};
            recv_sf = torch::empty_strided(
                {allocated_tokens,
                 static_cast<int64_t>(num_scale_factor_packs)},
                {sf_token_stride, sf_pack_stride}, sf->options());
        }
        auto recv_topk_indices = torch::empty(
            {static_cast<int64_t>(max_recv_tokens), topk_idx.size(1)},
            topk_idx.options());
        auto recv_topk_weights = std::optional<torch::Tensor>();
        if (topk_weights.has_value()) {
            recv_topk_weights = do_expand ?
                torch::empty(
                    {static_cast<int64_t>(expanded_records)},
                    topk_weights->options()) :
                torch::empty(
                    {static_cast<int64_t>(max_recv_tokens), topk_idx.size(1)},
                    topk_weights->options());
        }
        auto copied_topk_idx = !cached_mode && do_handle_copy ?
            std::optional<torch::Tensor>(topk_idx.clone()) :
            std::optional<torch::Tensor>();
        auto descriptor_tensor = cached_mode ?
            *cached_token_metadata_at_forward :
            torch::empty(
                {static_cast<int64_t>(
                    sizeof(elastic::DispatchHandleDescriptor) +
                    (allow_hybrid_mode_ ?
                         max_recv_tokens * sizeof(elastic::HybridRouteRecord) :
                         0))},
                metadata_options);
        std::vector<std::optional<torch::Tensor>> retained_tensors;
        if (cached_mode) {
            const auto retain = [&retained_tensors](
                                    const std::optional<torch::Tensor>& tensor) {
                retained_tensors.emplace_back(tensor);
            };
            retain(x);
            retain(sf);
            retain(topk_idx);
            retain(topk_weights);
            retain(cumulative_local_expert_recv_stats);
            retain(cached_psum_num_recv_tokens_per_scaleup_rank);
            retain(cached_psum_num_recv_tokens_per_expert);
            retain(cached_num_unaligned_recv_tokens_per_expert);
            retain(cached_dst_buffer_slot_idx);
            retain(cached_token_metadata_at_forward);
            retain(cached_recv_src_metadata);
            retain(cached_channel_linked_list);
            retain(kernel_expert_prefix);
            retain(kernel_unaligned);
            retain(rank_prefix);
            retain(expert_prefix);
            retain(unaligned);
            retain(destination_slots);
            retain(source_metadata);
            retain(recv_x);
            retain(recv_topk_indices);
            retain(recv_topk_weights);
            retain(copied_topk_idx);
            retain(descriptor_tensor);
#ifndef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
            for (const auto& tensor : retained_tensors) {
                if (!tensor.has_value())
                    continue;
                const auto record_status = resources_->record_tensor_stream(
                    *tensor, dispatch_stream);
                if (!record_status.ok())
                    raise_transport_status(record_status, rank_idx_);
            }
#endif
        }
        auto status = transport::TransportStatus{};

        elastic::DispatchArguments arguments{};
        arguments.x = x.data_ptr();
        arguments.scale_factors = sf.has_value() ? sf->data_ptr() : nullptr;
        arguments.scale_factor_token_stride = sf.has_value() ?
            static_cast<std::uint64_t>(sf->stride(0)) : 0;
        arguments.scale_factor_pack_stride = sf.has_value() ?
            static_cast<std::uint64_t>(sf->stride(1)) : 0;
        arguments.topk_indices = topk_idx.data_ptr<std::int64_t>();
        arguments.topk_weights = topk_weights.has_value() ?
            topk_weights->data_ptr<float>() : nullptr;
        arguments.communication_buffer = resources_->window_base();
        arguments.workspace = resources_->workspace();
        arguments.recv_x = recv_x.data_ptr();
        arguments.recv_scale_factors = recv_sf.has_value() ?
            recv_sf->data_ptr() : nullptr;
        arguments.recv_scale_factor_token_stride = recv_sf.has_value() ?
            static_cast<std::uint64_t>(recv_sf->stride(0)) : 0;
        arguments.recv_scale_factor_pack_stride = recv_sf.has_value() ?
            static_cast<std::uint64_t>(recv_sf->stride(1)) : 0;
        arguments.recv_topk_indices = recv_topk_indices.data_ptr<std::int64_t>();
        arguments.recv_topk_weights = recv_topk_weights.has_value() ?
            recv_topk_weights->data_ptr<float>() : nullptr;
        arguments.prefix_per_rank = rank_prefix.data_ptr<std::int32_t>();
        arguments.prefix_per_expert =
            kernel_expert_prefix.data_ptr<std::int32_t>();
        arguments.unaligned_per_expert =
            kernel_unaligned.data_ptr<std::int32_t>();
        arguments.destination_slots = destination_slots.data_ptr<std::int32_t>();
        arguments.source_metadata = source_metadata.data_ptr<std::int32_t>();
        if (allow_hybrid_mode_) {
            arguments.route_records =
                reinterpret_cast<elastic::HybridRouteRecord*>(
                    static_cast<std::uint8_t*>(descriptor_tensor.data_ptr()) +
                    sizeof(elastic::DispatchHandleDescriptor));
            arguments.route_record_capacity = cached_mode ?
                cached_route_count : tiling.hybrid_route_capacity;
        }
        arguments.timeout_cycles = barrier_timeout_cycles_;
        runtime::StreamIdentity stream = dispatch_stream;
        if (!cached_mode) {
            status = resources_->current_stream(&stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_), resources_->workspace_bytes()};
        const auto generation = activate_operation(lease, "dispatch");
        arguments.generation = generation;
        const auto launch_status = elastic::launch_internal_dispatch(
            arguments, tiling, storage, stream.raw);
        if (!launch_status.ok())
            raise_launch_status(launch_status, rank_idx_);
        if (cached_mode) {
            auto completion = resources_->create_event();
            if (!completion.status.ok())
                raise_transport_status(completion.status, rank_idx_);
            status = completion.event->record(stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            const auto committed_descriptor =
                elastic::make_attested_dispatch_handle_descriptor(
                    completion_resources_->dispatch_family(), tiling.topology,
                    generation, num_tokens, hidden, experts, num_topk,
                    alignment, capacity, descriptor_mode_flags);
            completion_resources_->stage_dispatch_descriptor(
                generation, descriptor_tensor, committed_descriptor);
            const auto completion_offset =
                tiling.symmetric_window_layout.control_offset +
                offsetof(elastic::SymmetricControlHeader, dispatch_generation);
            std::vector<elastic::EventDependency> predecessors;
            predecessors.emplace_back(std::move(predecessor));
            auto published = async_state_->publish(
                std::move(lease), completion.event,
                {elastic::BufferOperationKind::kDispatch, generation,
                 completion_offset,
                 tiling.workspace_layout.scratch_status_offset},
                std::move(retained_tensors), std::move(predecessors));
            if (!published.status.ok())
                raise_transport_status(published.status, rank_idx_);

            std::optional<EventHandle> event;
            if (async_with_compute_stream) {
                event.emplace(
                    completion.event, published.operation, async_state_);
            } else {
                status = published.operation->finish(5000);
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
            }
            const int num_recv_tokens = *cached_num_recv_tokens;
            const int output_tokens = do_expand ?
                *cached_num_expanded_tokens : num_recv_tokens;
            auto narrowed_x = recv_x.narrow(0, 0, output_tokens);
            auto narrowed_topk_idx = do_expand ?
                std::optional<torch::Tensor>() :
                std::optional<torch::Tensor>(
                    recv_topk_indices.narrow(0, 0, output_tokens));
            auto narrowed_topk_weights = recv_topk_weights.has_value() ?
                std::optional<torch::Tensor>(
                    recv_topk_weights->narrow(0, 0, output_tokens)) :
                std::optional<torch::Tensor>();
            auto narrowed_metadata = source_metadata.narrow(
                0, 0, num_recv_tokens);
            return {narrowed_x, std::nullopt, narrowed_topk_idx,
                    narrowed_topk_weights, copied_topk_idx, num_recv_tokens,
                    *cached_num_expanded_tokens, per_expert_list, rank_prefix,
                    expert_prefix, unaligned, narrowed_metadata,
                    destination_slots, descriptor_tensor, std::nullopt,
                    std::move(event)};
        }
        status = resources_->synchronize_stream(stream.raw);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        transport::DeviceTransportDiagnostic diagnostic{};
        status = host_transport()->read_diagnostic(&diagnostic);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
#if DEEP_EP_ASCEND_TESTING
        if (std::getenv("DEEP_EP_ASCEND_TEST_DISPATCH_DIAGNOSTIC") != nullptr) {
            diagnostic.abi_version = transport::kTransportCommandAbiVersion;
            diagnostic.error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic.generation = generation;
        }
#endif
        if (diagnostic.abi_version != transport::kTransportCommandAbiVersion ||
            diagnostic.error != transport::DeviceTransportError::kNone ||
            diagnostic.generation != generation)
            raise_dispatch_diagnostic(diagnostic, "reported failure");
        if (!cached_mode) {
            status = resources_->copy_to_host(
                host_rank_prefix.data(), rank_prefix.data_ptr(),
                host_rank_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_kernel_expert_prefix.data(),
                kernel_expert_prefix.data_ptr(),
                host_kernel_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_to_host(
                host_kernel_unaligned.data(), kernel_unaligned.data_ptr(),
                host_kernel_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            std::int32_t expanded_tail = 0;
            for (int expert = 0; expert < num_experts; ++expert) {
                const std::int32_t actual = host_kernel_unaligned[expert];
                const bool local = expert >= first_local_expert &&
                    expert < first_local_expert +
                                 static_cast<int>(local_experts);
                std::uint64_t aligned_actual = 0;
                TORCH_CHECK(actual >= 0 &&
                                (local || actual == 0) &&
                                host_kernel_expert_prefix[expert] ==
                                    expanded_tail &&
                                align_without_overflow(
                                    static_cast<std::uint64_t>(actual),
                                    alignment, &aligned_actual) &&
                                aligned_actual <= static_cast<std::uint64_t>(
                                    std::numeric_limits<int>::max() -
                                    expanded_tail),
                            "DeepEP Ascend backend: dispatch returned invalid expert counts");
                if (local) {
                    const int local_expert = expert - first_local_expert;
                    host_unaligned[local_expert] = actual;
                    host_expert_prefix[local_expert] = do_expand ?
                        expanded_tail + actual :
                        expanded_tail +
                            static_cast<std::int32_t>(aligned_actual);
                    per_expert_list.push_back(
                        static_cast<int>(aligned_actual));
                }
                expanded_tail += static_cast<std::int32_t>(aligned_actual);
            }
            TORCH_CHECK(host_kernel_expert_prefix[num_experts] ==
                            expanded_tail,
                        "DeepEP Ascend backend: dispatch returned invalid expert counts");
            num_expanded_tokens = expanded_tail;
            status = resources_->copy_from_host(
                expert_prefix.data_ptr(), host_expert_prefix.data(),
                host_expert_prefix.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            status = resources_->copy_from_host(
                unaligned.data_ptr(), host_unaligned.data(),
                host_unaligned.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        const int num_recv_tokens = host_rank_prefix.back();
        TORCH_CHECK(num_recv_tokens >= 0 &&
                        num_recv_tokens <= static_cast<int>(max_recv_tokens) &&
                        num_expanded_tokens >= 0 &&
                        num_expanded_tokens <= static_cast<int>(expanded_records),
                    "DeepEP Ascend backend: dispatch returned invalid output counts");
        elastic::DispatchHandleDescriptor committed_descriptor{};
        if (allow_hybrid_mode_) {
            host_route_records.resize(
                static_cast<std::size_t>(num_recv_tokens));
            if (!host_route_records.empty()) {
                const auto* device_records =
                    static_cast<const std::uint8_t*>(descriptor_tensor.data_ptr()) +
                    sizeof(elastic::DispatchHandleDescriptor);
                status = resources_->copy_to_host(
                    host_route_records.data(), device_records,
                    host_route_records.size() *
                        sizeof(elastic::HybridRouteRecord));
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
            }
            committed_descriptor = elastic::make_dispatch_handle_descriptor(
                0, tiling.topology, generation, num_tokens, hidden, experts,
                num_topk, alignment, capacity, descriptor_mode_flags,
                routing_mode, elastic::kHybridRouteLayoutVersion,
                static_cast<std::uint64_t>(num_recv_tokens),
                sizeof(elastic::HybridRouteRecord), generation,
                elastic::kHybridRouteCompleteStageFlags);
            const auto route_status = elastic::validate_hybrid_route_table(
                committed_descriptor,
                {host_route_records.data(), host_route_records.size()},
                capacity, max_recv_tokens, local_experts);
            TORCH_CHECK(route_status.ok(), "DeepEP Ascend backend: dispatch ",
                        route_status.message);

            std::vector<std::int32_t> host_source_metadata(
                static_cast<std::size_t>(num_recv_tokens) *
                static_cast<std::size_t>(num_topk + 2));
            std::vector<std::int64_t> host_received_topk(
                static_cast<std::size_t>(num_recv_tokens) *
                static_cast<std::size_t>(num_topk));
            if (num_recv_tokens != 0) {
                status = resources_->copy_to_host(
                    host_source_metadata.data(), source_metadata.data_ptr(),
                    host_source_metadata.size() * sizeof(std::int32_t));
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
                status = resources_->copy_to_host(
                    host_received_topk.data(), recv_topk_indices.data_ptr(),
                    host_received_topk.size() * sizeof(std::int64_t));
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
            }
            const elastic::HybridRouteBindingView bindings{
                host_source_metadata.data(), host_received_topk.data(),
                static_cast<std::uint64_t>(num_recv_tokens), num_topk,
                capacity};
            const auto binding_status = elastic::validate_hybrid_route_bindings(
                committed_descriptor,
                {host_route_records.data(), host_route_records.size()}, bindings);
            TORCH_CHECK(binding_status.ok(),
                        "DeepEP Ascend backend: dispatch ",
                        binding_status.message);
            committed_descriptor =
                elastic::make_attested_hybrid_dispatch_handle_descriptor(
                    completion_resources_->dispatch_family(), tiling.topology,
                    generation, num_tokens,
                    hidden, experts, num_topk, alignment, capacity,
                    descriptor_mode_flags, elastic::kHybridRouteLayoutVersion,
                    static_cast<std::uint64_t>(num_recv_tokens),
                    sizeof(elastic::HybridRouteRecord),
                    elastic::kHybridRouteCompleteStageFlags,
                    {host_route_records.data(), host_route_records.size()});
        } else {
            committed_descriptor =
                elastic::make_attested_dispatch_handle_descriptor(
                    completion_resources_->dispatch_family(), tiling.topology,
                    generation, num_tokens,
                    hidden, experts, num_topk, alignment, capacity,
                    descriptor_mode_flags);
        }
        status = resources_->copy_from_host(
            descriptor_tensor.data_ptr(), &committed_descriptor,
            sizeof(committed_descriptor));
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        completion_resources_->commit_dispatch_generation(generation);
        lease.complete();
        const int output_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto narrowed_x = recv_x.narrow(0, 0, output_tokens);
        auto narrowed_sf = recv_sf.has_value() ?
            std::optional<torch::Tensor>(recv_sf->narrow(0, 0, output_tokens)) :
            std::optional<torch::Tensor>();
        if (narrowed_sf.has_value() && use_tma_aligned_col_major_sf) {
            std::uint64_t aligned_output_tokens = 0;
            TORCH_CHECK(
                align_without_overflow(
                    static_cast<std::uint64_t>(output_tokens), 4,
                    &aligned_output_tokens),
                "DeepEP Ascend backend: dispatch scale factor output stride overflow");
            auto exact_stride_sf = torch::empty_strided(
                {output_tokens, static_cast<int64_t>(num_scale_factor_packs)},
                {int64_t{1}, static_cast<int64_t>(aligned_output_tokens)},
                narrowed_sf->options());
            exact_stride_sf.copy_(*narrowed_sf);
            narrowed_sf = std::move(exact_stride_sf);
        }
        auto narrowed_topk_idx = do_expand ? std::optional<torch::Tensor>() :
            std::optional<torch::Tensor>(recv_topk_indices.narrow(0, 0, output_tokens));
        auto narrowed_topk_weights = recv_topk_weights.has_value() ?
            std::optional<torch::Tensor>(
                recv_topk_weights->narrow(0, 0, output_tokens)) :
            std::optional<torch::Tensor>();
        auto narrowed_metadata = source_metadata.narrow(0, 0, num_recv_tokens);
        auto handle_tensor = allow_hybrid_mode_ && !cached_mode ?
            descriptor_tensor.narrow(
                0, 0, static_cast<int64_t>(
                    sizeof(elastic::DispatchHandleDescriptor) +
                    static_cast<std::uint64_t>(num_recv_tokens) *
                        sizeof(elastic::HybridRouteRecord))) :
            descriptor_tensor;
        return {narrowed_x, narrowed_sf, narrowed_topk_idx,
                narrowed_topk_weights, copied_topk_idx, num_recv_tokens,
                num_expanded_tokens, per_expert_list, rank_prefix,
                expert_prefix, unaligned, narrowed_metadata,
                destination_slots, handle_tensor, std::nullopt, std::nullopt};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const std::optional<torch::Tensor>& token_metadata_at_forward,
            const std::optional<torch::Tensor>& channel_linked_list,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms, const int& num_qps,
            const std::optional<EventHandle>& previous_event,
            const std::optional<EventHandle>& previous_event_before_epilogue,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout) const {
        TORCH_CHECK(!previous_event.has_value() &&
                        !previous_event_before_epilogue.has_value() &&
                        !async_with_compute_stream && !allocate_on_comm_stream,
                    "DeepEP Ascend backend: combine is synchronous");
        TORCH_CHECK(!channel_linked_list.has_value(),
                    "DeepEP Ascend backend: combine does not support channel handles");
        TORCH_CHECK(num_sms == 1,
                    "DeepEP Ascend backend: combine requires num_sms=1");
        TORCH_CHECK(num_qps == 0,
                    "DeepEP Ascend backend: combine requires num_qps=0");
        TORCH_CHECK(num_experts > 0 && num_experts % num_ranks_ == 0 &&
                        num_max_tokens_per_rank > 0 &&
                        num_max_tokens_per_rank <=
                            std::numeric_limits<std::int32_t>::max() /
                                num_ranks_,
                    "DeepEP Ascend backend: combine requires positive, "
                    "rank-partitioned expert capacity");
        TORCH_CHECK(token_metadata_at_forward.has_value(),
                    "DeepEP Ascend backend: combine requires a dispatch handle");
        const auto device = x.device();
        int current_device = -1;
        auto status = resources_->current_device(&current_device);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        TORCH_CHECK(device.index() == resources_->owning_device() &&
                        current_device == resources_->owning_device(),
                    "DeepEP Ascend backend: combine tensors and current NPU "
                    "must match the buffer device");
        if (allow_hybrid_mode_)
            require_transport("combine",
                              operation_capabilities(kCombineCapabilities));
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kCombine, "combine");
        if (!allow_hybrid_mode_)
            require_transport("combine",
                              operation_capabilities(kCombineCapabilities));
        const auto validate = [&](const torch::Tensor& tensor,
                                  int64_t dimensions,
                                  torch::ScalarType type,
                                  const char* name) {
            TORCH_CHECK(
                tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.device() == device && tensor.is_contiguous() &&
                    tensor.dim() == dimensions && tensor.scalar_type() == type,
                "DeepEP Ascend backend: combine requires contiguous NPU ", name,
                " with the expected rank and dtype");
        };
        validate(x, 2, torch::kBFloat16, "BF16 x");
        validate(src_metadata, 2, torch::kInt, "int32 source metadata");
        validate(combined_topk_idx, 2, torch::kLong,
                 "int64 combined top-k indices");
        validate(psum_num_recv_tokens_per_scaleup_rank, 1, torch::kInt,
                 "int32 rank prefix");
        validate(*token_metadata_at_forward, 1, torch::kByte,
                 "dispatch handle descriptor");
        TORCH_CHECK(x.size(0) >= 0 && x.size(1) > 0 &&
                        src_metadata.size(0) >= 0 &&
                        combined_topk_idx.size(0) >= 0 &&
                        combined_topk_idx.size(1) > 0 &&
                        combined_topk_idx.size(1) <= num_experts &&
                        psum_num_recv_tokens_per_scaleup_rank.size(0) ==
                            num_ranks_ &&
                        token_metadata_at_forward->numel() >=
                            static_cast<int64_t>(
                                sizeof(elastic::DispatchHandleDescriptor)),
                    "DeepEP Ascend backend: combine handle tensor shape mismatch");
        const auto num_source_rows =
            static_cast<std::uint64_t>(src_metadata.size(0));
        const auto num_input_rows = static_cast<std::uint64_t>(x.size(0));
        const auto num_topk =
            static_cast<std::uint64_t>(combined_topk_idx.size(1));
        const auto capacity =
            static_cast<std::uint64_t>(num_max_tokens_per_rank);
        const auto maximum_source_rows =
            capacity * static_cast<std::uint64_t>(num_ranks_);
        TORCH_CHECK(src_metadata.size(1) ==
                        static_cast<int64_t>(num_topk + 2) &&
                        num_source_rows <= maximum_source_rows &&
                        (!use_expanded_layout ?
                             num_input_rows == num_source_rows : true),
                    "DeepEP Ascend backend: combine source metadata shape or "
                    "capacity mismatch");
        if (topk_weights.has_value()) {
            validate(*topk_weights, use_expanded_layout ? 1 : 2,
                     torch::kFloat, "float32 top-k weights");
            TORCH_CHECK(
                (!use_expanded_layout &&
                 topk_weights->size(0) == x.size(0) &&
                 topk_weights->size(1) ==
                     combined_topk_idx.size(1)) ||
                    (use_expanded_layout &&
                     topk_weights->size(0) == x.size(0)),
                "DeepEP Ascend backend: combine top-k weights shape mismatch");
            TORCH_CHECK(!use_expanded_layout || allow_multiple_reduction_,
                        "DeepEP Ascend backend: expanded combine weights require "
                        "allow_multiple_reduction");
        }
        for (const auto* bias : {&bias_0, &bias_1}) {
            if (!bias->has_value())
                continue;
            validate(**bias, 2, torch::kBFloat16, "BF16 bias");
            TORCH_CHECK((*bias)->sizes() == std::vector<int64_t>({
                            combined_topk_idx.size(0), x.size(1)}),
                        "DeepEP Ascend backend: combine bias shape mismatch");
        }

        elastic::DispatchHandleDescriptor descriptor{};
        status = resources_->copy_to_host(
            &descriptor, token_metadata_at_forward->data_ptr(),
            sizeof(descriptor));
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        const auto& context = resources_->device_context();
        const auto expanded_dispatch_mode =
            elastic::mode_bit(elastic::CoreMode::kExpanded);
        const auto zero_padding_dispatch_mode =
            elastic::mode_bit(elastic::CoreMode::kZeroPadding);
        const auto hybrid_dispatch_mode =
            elastic::mode_bit(elastic::CoreMode::kHybrid);
        elastic::CoreModeFlags dispatch_mode = use_expanded_layout ?
            expanded_dispatch_mode : 0;
        if (allow_hybrid_mode_)
            dispatch_mode |= hybrid_dispatch_mode;
        const auto descriptor_base_mode = allow_hybrid_mode_ ?
            hybrid_dispatch_mode : 0;
        const bool compatible_descriptor_mode = use_expanded_layout ?
            descriptor.mode_flags ==
                    (descriptor_base_mode | expanded_dispatch_mode) ||
                descriptor.mode_flags ==
                    (descriptor_base_mode | expanded_dispatch_mode |
                     zero_padding_dispatch_mode) :
            descriptor.mode_flags == descriptor_base_mode;
        TORCH_CHECK(
            compatible_descriptor_mode,
            "DeepEP Ascend backend: combine dispatch handle does not match "
            "the current call");
        TORCH_CHECK(
            token_metadata_at_forward->numel() == static_cast<int64_t>(
                sizeof(elastic::DispatchHandleDescriptor) +
                (allow_hybrid_mode_ ?
                     num_source_rows * sizeof(elastic::HybridRouteRecord) :
                     0)),
            "DeepEP Ascend backend: combine handle tensor shape mismatch");
        std::vector<elastic::HybridRouteRecord> host_route_records;
        if (allow_hybrid_mode_) {
            host_route_records.resize(
                static_cast<std::size_t>(num_source_rows));
            if (!host_route_records.empty()) {
                const auto* device_records =
                    static_cast<const std::uint8_t*>(
                        token_metadata_at_forward->data_ptr()) +
                    sizeof(elastic::DispatchHandleDescriptor);
                status = resources_->copy_to_host(
                    host_route_records.data(), device_records,
                    host_route_records.size() *
                        sizeof(elastic::HybridRouteRecord));
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
            }
        }
        const auto expected_descriptor = allow_hybrid_mode_ ?
            elastic::make_attested_hybrid_dispatch_handle_descriptor(
                completion_resources_->dispatch_family(),
                elastic::core_topology_from_transport(context.topology),
                completion_resources_->last_dispatch_generation(),
                static_cast<std::uint64_t>(combined_topk_idx.size(0)),
                static_cast<std::uint64_t>(x.size(1)),
                static_cast<std::uint64_t>(num_experts), num_topk,
                descriptor.expert_alignment, capacity,
                descriptor.mode_flags, elastic::kHybridRouteLayoutVersion,
                num_source_rows, sizeof(elastic::HybridRouteRecord),
                elastic::kHybridRouteCompleteStageFlags,
                {host_route_records.data(), host_route_records.size()}) :
            elastic::make_attested_dispatch_handle_descriptor(
                completion_resources_->dispatch_family(),
                elastic::core_topology_from_transport(context.topology),
                completion_resources_->last_dispatch_generation(),
                static_cast<std::uint64_t>(combined_topk_idx.size(0)),
                static_cast<std::uint64_t>(x.size(1)),
                static_cast<std::uint64_t>(num_experts), num_topk,
                descriptor.expert_alignment, capacity,
                descriptor.mode_flags);
        const auto descriptor_status = elastic::validate_dispatch_handle(
            expected_descriptor, descriptor);
        TORCH_CHECK(descriptor_status.ok(), "DeepEP Ascend backend: combine ",
                    descriptor_status.message);
        if (allow_hybrid_mode_) {
            const auto route_status = elastic::validate_hybrid_route_table(
                descriptor,
                {host_route_records.data(), host_route_records.size()},
                capacity, maximum_source_rows,
                static_cast<std::uint64_t>(num_experts / num_ranks_));
            TORCH_CHECK(route_status.ok(), "DeepEP Ascend backend: combine ",
                        route_status.message);
        }

        elastic::CoreModeFlags combine_mode = dispatch_mode;
        if (use_expanded_layout && allow_multiple_reduction_)
            combine_mode |= elastic::mode_bit(
                elastic::CoreMode::kAllowMultipleReduction);
        auto tiling = build_combine_tiling(descriptor, combine_mode);
        const std::uint64_t maximum_input_rows = use_expanded_layout ?
            tiling.dispatch_output_capacity : maximum_source_rows;
        TORCH_CHECK(
            num_input_rows <= maximum_input_rows,
            "DeepEP Ascend backend: combine source metadata shape or "
            "capacity mismatch");

        std::vector<std::int32_t> host_prefix(num_ranks_);
        std::vector<std::int32_t> host_metadata(
            static_cast<std::size_t>(num_source_rows * (num_topk + 2)));
        status = resources_->copy_to_host(
            host_prefix.data(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr(),
            host_prefix.size() * sizeof(std::int32_t));
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        if (!host_metadata.empty()) {
            status = resources_->copy_to_host(
                host_metadata.data(), src_metadata.data_ptr(),
                host_metadata.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        std::uint64_t previous_end = 0;
        for (int destination_rank = 0; destination_rank < num_ranks_;
             ++destination_rank) {
            const std::int32_t encoded_end = host_prefix[destination_rank];
            TORCH_CHECK(encoded_end >= 0 &&
                            static_cast<std::uint64_t>(encoded_end) >=
                                previous_end &&
                            static_cast<std::uint64_t>(encoded_end) <=
                                num_source_rows,
                        "DeepEP Ascend backend: combine rank prefix is invalid");
            const auto end = static_cast<std::uint64_t>(encoded_end);
            std::uint64_t records = 0;
            for (std::uint64_t row = previous_end; row < end; ++row) {
                const auto* metadata = host_metadata.data() +
                    row * (num_topk + 2);
                TORCH_CHECK(
                    elastic::is_valid_combine_source_identity(
                        metadata[0], destination_rank, rank_idx_, capacity,
                        descriptor.num_tokens) &&
                        elastic::decode_dispatch_source_rank(
                            metadata[1], num_topk) == destination_rank &&
                        elastic::is_dispatch_local_index(
                            elastic::decode_dispatch_local_index(
                                metadata[1], num_topk), num_topk),
                    "DeepEP Ascend backend: combine source metadata is invalid");
                std::uint64_t valid_slots = 0;
                for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
                    const auto input_row = metadata[2 + lane];
                    TORCH_CHECK(
                        use_expanded_layout ?
                            elastic::combine_expanded_input_row_is_valid(
                                input_row, num_input_rows) :
                            input_row == -1,
                        "DeepEP Ascend backend: combine expanded slot is invalid");
                    valid_slots += input_row == -1 ? 0 : 1;
                }
                records += !use_expanded_layout ? 1 :
                    (allow_multiple_reduction_ ?
                         (valid_slots == 0 ? 0 : 1) : valid_slots);
            }
            const auto record_capacity =
                use_expanded_layout && !allow_multiple_reduction_ ?
                    capacity * num_topk : capacity;
            TORCH_CHECK(records <= record_capacity,
                        "DeepEP Ascend backend: combine source metadata "
                        "exceeds capacity");
            previous_end = end;
        }
        TORCH_CHECK(previous_end == num_source_rows,
                    "DeepEP Ascend backend: combine rank prefix tail mismatch");

        TORCH_CHECK(tiling.launch.num_blocks == 1 &&
                        tiling.communication_buffer_bytes <=
                            static_cast<std::uint64_t>(num_buffer_bytes_) &&
                        tiling.workspace_bytes <= resources_->workspace_bytes(),
                    "DeepEP Ascend backend: combine capacity exceeds runtime storage");
        auto combined_x = torch::empty(
            {combined_topk_idx.size(0), x.size(1)}, x.options());
        std::optional<torch::Tensor> combined_weights;
        if (topk_weights.has_value())
            combined_weights = torch::empty(
                {combined_topk_idx.size(0), combined_topk_idx.size(1)},
                topk_weights->options());

        runtime::StreamIdentity stream;
        status = resources_->current_stream(&stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        elastic::CombineArguments arguments{};
        arguments.x = x.data_ptr();
        arguments.topk_weights = topk_weights.has_value() ?
            topk_weights->data_ptr<float>() : nullptr;
        arguments.source_metadata = src_metadata.data_ptr<std::int32_t>();
        if (allow_hybrid_mode_) {
            arguments.route_records =
                reinterpret_cast<const elastic::HybridRouteRecord*>(
                    static_cast<const std::uint8_t*>(
                        token_metadata_at_forward->data_ptr()) +
                    sizeof(elastic::DispatchHandleDescriptor));
            arguments.route_record_count = descriptor.route_record_count;
        }
        arguments.combined_topk_indices =
            combined_topk_idx.data_ptr<std::int64_t>();
        arguments.prefix_per_rank =
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<std::int32_t>();
        arguments.bias_0 = bias_0.has_value() ? bias_0->data_ptr() : nullptr;
        arguments.bias_1 = bias_1.has_value() ? bias_1->data_ptr() : nullptr;
        arguments.communication_buffer = resources_->window_base();
        arguments.workspace = resources_->workspace();
        arguments.combined_x = combined_x.data_ptr();
        arguments.combined_topk_weights = combined_weights.has_value() ?
            combined_weights->data_ptr<float>() : nullptr;
        const auto generation = activate_operation(lease, "combine");
        arguments.generation = generation;
        arguments.timeout_cycles = barrier_timeout_cycles_;
        arguments.num_source_rows = num_source_rows;
        arguments.num_input_rows = num_input_rows;
        arguments.local_window_base = reinterpret_cast<std::uintptr_t>(
            resources_->window_base());
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_),
            resources_->workspace_bytes()};
        const auto launch_status = elastic::launch_internal_combine(
            arguments, tiling, storage, stream.raw);
        if (!launch_status.ok())
            raise_combine_launch_status(launch_status, rank_idx_);
        status = resources_->synchronize_stream(stream.raw);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        transport::DeviceTransportDiagnostic diagnostic{};
        status = host_transport()->read_diagnostic(&diagnostic);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        if (diagnostic.abi_version !=
                transport::kTransportCommandAbiVersion ||
            diagnostic.error != transport::DeviceTransportError::kNone ||
            diagnostic.generation != generation)
            raise_combine_diagnostic(
                diagnostic, "reported failure",
                tiling.workspace_layout.scratch_status_offset);
        std::uint64_t completion = 0;
        const auto completion_offset =
            tiling.symmetric_window_layout.control_offset +
            offsetof(elastic::SymmetricControlHeader, combine_generation);
        const auto* completion_address = reinterpret_cast<const void*>(
            reinterpret_cast<std::uintptr_t>(resources_->window_base()) +
            completion_offset);
        status = resources_->copy_to_host(
            &completion, completion_address, sizeof(completion));
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        if (completion != generation)
            raise_combine_diagnostic(
                diagnostic, "completion mismatch",
                tiling.workspace_layout.scratch_status_offset);
        lease.complete();
        return {combined_x, combined_weights, std::nullopt};
    }
};

}  // namespace deep_ep::ascend
