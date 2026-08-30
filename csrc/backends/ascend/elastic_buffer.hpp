#pragma once

#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <cstring>
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
#include "elastic/dispatch_pipeline_config.hpp"
#include "elastic/dispatch_token_fanout.hpp"
#include "elastic/async_state.hpp"
#include "elastic/operation_coordinator.hpp"
#include "elastic/runtime.hpp"
#include "runtime/cann_runtime.hpp"
#include "runtime/host_timeline.hpp"
#include "transport/topology_config.hpp"

namespace deep_ep::ascend {

inline bool environment_is(const char* name, const char* expected) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, expected) == 0;
}

#if DEEP_EP_ASCEND_TESTING
inline bool testing_environment_is(
    const char* name, const char* expected) {
    return environment_is(name, expected);
}

inline void inject_testing_completion_mismatch(std::uint64_t* output) {
    if (output != nullptr && testing_environment_is(
            "DEEP_EP_ASCEND_TEST_COMPLETION_FAULT",
            "completion_mismatch"))
        *output ^= 1U;
}

inline void inject_testing_post_wait_failure(const char* stage) {
    TORCH_CHECK(
        !testing_environment_is("DEEP_EP_ASCEND_TEST_POST_WAIT_FAULT", stage),
        "DeepEP Ascend backend: injected post-wait ", stage, " failure");
}
#endif

inline std::vector<std::uint8_t> dispatch_descriptor_snapshot(
    const elastic::DispatchHandleDescriptor& descriptor,
    const std::vector<elastic::HybridRouteRecord>& route_records) {
    std::vector<std::uint8_t> snapshot(
        sizeof(descriptor) +
        route_records.size() * sizeof(elastic::HybridRouteRecord));
    std::memcpy(snapshot.data(), &descriptor, sizeof(descriptor));
    if (!route_records.empty())
        std::memcpy(
            snapshot.data() + sizeof(descriptor), route_records.data(),
            route_records.size() * sizeof(elastic::HybridRouteRecord));
    return snapshot;
}

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
        return {state_->event, state_->pending_operation, nullptr};
    }

    friend class ElasticBuffer;
};

class ElasticAsyncCompletionResources final
    : public elastic::AsyncCompletionResources {
public:
    ElasticAsyncCompletionResources(
        std::unique_ptr<runtime::CannRuntimeResources> resources,
        std::uint64_t dispatch_family,
        std::uint64_t last_dispatch_generation,
        int rank_idx)
        : resources_(std::move(resources)), dispatch_family_(dispatch_family),
          last_dispatch_generation_(last_dispatch_generation),
          rank_idx_(rank_idx) {}

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
        elastic::DispatchHandleDescriptor descriptor,
        std::vector<std::uint8_t> descriptor_bytes) {
#if DEEP_EP_ASCEND_TESTING
        inject_testing_post_wait_failure("dispatch_staging");
#endif
        std::lock_guard<std::mutex> lock(mutex_);
        staged_dispatch_ = StagedDispatch{
            generation, std::move(tensor), descriptor,
            std::move(descriptor_bytes)};
    }

    void commit_dispatch_descriptor(
        std::uint64_t generation, const torch::Tensor& descriptor_tensor,
        std::vector<std::uint8_t> descriptor_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_dispatch_generation_ = generation;
        committed_dispatch_tensor_ = descriptor_tensor;
        committed_dispatch_bytes_ = std::move(descriptor_bytes);
    }

    std::uint64_t dispatch_handle_generation(
        const torch::Tensor& descriptor_tensor) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (resources_ == nullptr ||
            !committed_dispatch_tensor_.has_value() ||
            committed_dispatch_bytes_.empty() ||
            descriptor_tensor.scalar_type() != torch::kByte ||
            descriptor_tensor.data_ptr() !=
                committed_dispatch_tensor_->data_ptr())
            return 0;
        const auto observed_size = static_cast<std::size_t>(
            descriptor_tensor.numel());
        if (observed_size != committed_dispatch_bytes_.size())
            return 0;
        std::vector<std::uint8_t> observed(observed_size);
        const auto status = resources_->copy_to_host(
            observed.data(), descriptor_tensor.data_ptr(), observed.size());
        if (!status.ok() || std::memcmp(
                observed.data(), committed_dispatch_bytes_.data(),
                observed.size()) != 0)
            return 0;
        return last_dispatch_generation_;
    }

    void stage_combine_completion(
        std::uint64_t generation, std::uint64_t scratch_status_offset) {
#if DEEP_EP_ASCEND_TESTING
        inject_testing_post_wait_failure("combine_staging");
#endif
        std::lock_guard<std::mutex> lock(mutex_);
        staged_combine_ = StagedCombine{
            generation, scratch_status_offset, std::nullopt};
    }

    transport::TransportStatus read_diagnostic(
        elastic::BufferOperationKind kind, std::uint64_t,
        transport::DeviceTransportDiagnostic* output) override {
        if (resources_ == nullptr || resources_->transport() == nullptr ||
            output == nullptr)
            return transport::TransportStatus::invalid(
                "read_diagnostic", "completion resources are unavailable");
        auto status = resources_->transport()->read_diagnostic(output);
        if (status.ok() && kind == elastic::BufferOperationKind::kCombine) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (staged_combine_.has_value())
                staged_combine_->diagnostic = *output;
        }
#if DEEP_EP_ASCEND_TESTING
        if (status.ok() && kind == elastic::BufferOperationKind::kBarrier &&
            testing_environment_is(
                "DEEP_EP_ASCEND_TEST_DIAGNOSTIC", "completion_timeout")) {
            auto& diagnostic = *output;
            diagnostic.abi_version = transport::kTransportCommandAbiVersion;
            diagnostic.error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic.command_index = 0;
            diagnostic.opcode = transport::TransportCommandOpcode::kBarrier;
            diagnostic.peer = static_cast<std::uint32_t>(rank_idx_);
            diagnostic.channel = 0;
            diagnostic.backend_status = 0;
        }
#endif
        return status;
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
        if (!status.ok())
            return status;

        if (kind == elastic::BufferOperationKind::kCombine) {
            std::optional<StagedCombine> staged;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                staged = staged_combine_;
            }
            if (staged.has_value() && *output != staged->generation)
                return combine_completion_failure(*staged);
#if DEEP_EP_ASCEND_TESTING
            inject_testing_completion_mismatch(output);
#endif
            return status;
        }
        if (kind != elastic::BufferOperationKind::kDispatch)
            return status;
#if DEEP_EP_ASCEND_TESTING
        inject_testing_completion_mismatch(output);
#endif
        return status;
    }

    transport::TransportStatus commit_completion(
        elastic::BufferOperationKind kind,
        std::uint64_t generation) override {
        if (kind == elastic::BufferOperationKind::kCombine) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!staged_combine_.has_value() ||
                staged_combine_->generation != generation)
                return transport::TransportStatus::invalid(
                    "combine", "generation-bound completion is unavailable");
            staged_combine_.reset();
            return transport::TransportStatus::success();
        }
        if (kind != elastic::BufferOperationKind::kDispatch)
            return transport::TransportStatus::success();

        std::lock_guard<std::mutex> lock(mutex_);
        if (!staged_dispatch_.has_value() ||
            staged_dispatch_->generation != generation)
            return transport::TransportStatus::invalid(
                "dispatch", "generation-bound descriptor commit is unavailable");
        auto status = resources_->copy_from_host(
            staged_dispatch_->tensor.data_ptr(), &staged_dispatch_->descriptor,
            sizeof(staged_dispatch_->descriptor));
        if (!status.ok())
            return status;
        last_dispatch_generation_ = generation;
        committed_dispatch_tensor_ = staged_dispatch_->tensor;
        committed_dispatch_bytes_ = std::move(staged_dispatch_->descriptor_bytes);
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
        std::vector<std::uint8_t> descriptor_bytes;
    };

    struct StagedCombine {
        std::uint64_t generation = 0;
        std::uint64_t scratch_status_offset = 0;
        std::optional<transport::DeviceTransportDiagnostic> diagnostic;
    };

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

        transport::StagedTransportContext context{};
        const auto backend_context =
            resources_->device_context().backend_context;
        if (backend_context == 0)
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "staged transport context is unavailable");
        status = resources_->copy_to_host(
            &context, reinterpret_cast<const void*>(backend_context),
            sizeof(context));
        if (!status.ok())
            return status;
        if (!transport::command::valid_staged_context_header(
                context.abi_version, context.struct_size,
                context.cann_compatibility, context.command_queue))
            return transport::TransportStatus::invalid(
                "combine_lifecycle_snapshot",
                "malformed staged transport context");

        status = resources_->copy_to_host(
            &snapshot->queue,
            reinterpret_cast<const void*>(context.command_queue),
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
                context.reserved, context.command_queue,
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

    transport::TransportStatus combine_completion_failure(
        const StagedCombine& staged) const {
        const auto diagnostic = staged.diagnostic.value_or(
            transport::DeviceTransportDiagnostic{});
        auto message = std::string("device diagnostic completion mismatch") +
            " error=" + elastic::diagnostic_name(diagnostic.error) +
            " command_index=" + std::to_string(diagnostic.command_index) +
            " opcode=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.opcode)) +
            " peer=" + std::to_string(diagnostic.peer) +
            " world_peer=" + std::to_string(diagnostic.world_peer) +
            " team=" + std::to_string(
                static_cast<std::uint32_t>(diagnostic.team)) +
            " channel=" + std::to_string(diagnostic.channel) +
            " backend_status=" +
                std::to_string(diagnostic.backend_status) +
            " reserved=" + std::to_string(diagnostic.reserved) +
            " generation=" + std::to_string(diagnostic.generation) +
            " diagnostic_generation=" +
                std::to_string(diagnostic.generation) +
            " diagnostic_error=" +
                elastic::diagnostic_name(diagnostic.error);
        CombineLifecycleSnapshot snapshot{};
        const auto snapshot_status = read_combine_lifecycle_snapshot(
            staged.scratch_status_offset, &snapshot);
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
        return transport::TransportStatus::runtime_failure(
            "combine", static_cast<int>(diagnostic.backend_status),
            std::move(message));
    }

    std::unique_ptr<runtime::CannRuntimeResources> resources_;
    mutable std::mutex mutex_;
    std::uint64_t dispatch_family_ = 0;
    std::uint64_t last_dispatch_generation_ = 0;
    int rank_idx_ = -1;
    std::optional<StagedDispatch> staged_dispatch_;
    std::optional<torch::Tensor> committed_dispatch_tensor_;
    std::vector<std::uint8_t> committed_dispatch_bytes_;
    std::optional<StagedCombine> staged_combine_;
};

#if DEEP_EP_ASCEND_TESTING
struct ElasticBufferTestingLifecycleControl {
    std::mutex mutex;
    std::condition_variable cv;
    int destroy_attempt_count = 0;
    int active_owner_count = 0;
    int maximum_active_owner_count = 0;
    int destroy_attempts_required = 0;
    bool getter_waits_for_destroy_attempt = false;

    void note_destroy_attempt() {
        std::lock_guard<std::mutex> lock(mutex);
        ++destroy_attempt_count;
        cv.notify_all();
    }

    void enter_owner(bool getter) {
        std::unique_lock<std::mutex> lock(mutex);
        ++active_owner_count;
        if (active_owner_count > maximum_active_owner_count)
            maximum_active_owner_count = active_owner_count;
        cv.notify_all();
        if (getter && getter_waits_for_destroy_attempt) {
            cv.wait(lock, [&] { return destroy_attempt_count >= 1; });
        } else if (!getter && destroy_attempts_required > 0) {
            cv.wait(lock, [&] {
                return destroy_attempt_count >= destroy_attempts_required;
            });
        }
    }

    void leave_owner() {
        std::lock_guard<std::mutex> lock(mutex);
        --active_owner_count;
        cv.notify_all();
    }

    void wait_for_active_owner() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return active_owner_count > 0; });
    }

    int destroy_attempts() {
        std::lock_guard<std::mutex> lock(mutex);
        return destroy_attempt_count;
    }

    int maximum_active_owners() {
        std::lock_guard<std::mutex> lock(mutex);
        return maximum_active_owner_count;
    }
};

class ElasticBufferTestingLifecycleOwner {
public:
    ElasticBufferTestingLifecycleOwner(
        std::shared_ptr<ElasticBufferTestingLifecycleControl> control,
        bool getter)
        : control_(std::move(control)) {
        if (control_ != nullptr)
            control_->enter_owner(getter);
    }

    ~ElasticBufferTestingLifecycleOwner() {
        if (control_ != nullptr)
            control_->leave_owner();
    }

private:
    std::shared_ptr<ElasticBufferTestingLifecycleControl> control_;
};
#endif

class ElasticBuffer {
    int rank_idx_;
    int num_ranks_;
    int64_t num_buffer_bytes_;
    bool allow_hybrid_mode_;
    bool allow_multiple_reduction_;
    mutable std::mutex lifecycle_mutex_;
    runtime::CannRuntimeResources* resources_ = nullptr;
    std::shared_ptr<ElasticAsyncCompletionResources> completion_resources_;
    std::shared_ptr<elastic::AsyncBufferState> async_state_;
    std::uint64_t barrier_timeout_cycles_ = 0;
    std::uint64_t completion_timeout_ms_ = 5000;
    bool stage_profile_enabled_ = false;
    mutable runtime::HostTimelineProfile host_timeline_profile_{};
#if DEEP_EP_ASCEND_TESTING
    std::shared_ptr<ElasticBufferTestingLifecycleControl>
        testing_lifecycle_control_;
#endif

    inline static std::atomic_uint64_t next_dispatch_family_{0};

    static constexpr auto kDispatchCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kAsyncCompletion) |
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
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
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

    std::uint64_t host_profile_start() const noexcept {
        return stage_profile_enabled_ ? runtime::host_timestamp_ns() : 0;
    }

    void host_profile_record(
        runtime::HostTimelinePhase phase, std::uint64_t start_ns) const noexcept {
        if (stage_profile_enabled_)
            (void)host_timeline_profile_.record(
                phase, start_ns, runtime::host_timestamp_ns());
    }

    static constexpr auto kCombineCapabilities =
        transport::capability_bit(transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(transport::TransportCapability::kDevicePut) |
        transport::capability_bit(transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kAsyncCompletion) |
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

    [[noreturn]] void raise_barrier_diagnostic(
        const transport::DeviceTransportDiagnostic& diagnostic,
        const char* detail) const {
        raise_transport_status(
            elastic::diagnostic_failure_status(
                elastic::BufferOperationKind::kBarrier, diagnostic, detail),
            rank_idx_);
    }

    [[noreturn]] void raise_dispatch_diagnostic(
        const transport::DeviceTransportDiagnostic& diagnostic,
        const char* detail) const {
        raise_transport_status(
            elastic::diagnostic_failure_status(
                elastic::BufferOperationKind::kDispatch, diagnostic, detail),
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
        std::uint64_t num_scale_factor_packs,
        std::uint32_t data_num_blocks) const {
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
        input.data_num_blocks = data_num_blocks;
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
        elastic::CoreModeFlags mode_flags,
        std::uint32_t data_num_blocks) const {
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
        input.data_num_blocks = data_num_blocks;
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
                  bool allow_hybrid_mode,
                  std::uint64_t completion_timeout_ms,
                  std::shared_ptr<ElasticBufferTestingLifecycleControl>
                      lifecycle_control)
        : rank_idx_(rank), num_ranks_(num_ranks),
          num_buffer_bytes_(buffer_bytes),
          allow_hybrid_mode_(allow_hybrid_mode),
          allow_multiple_reduction_(allow_multiple_reduction),
          barrier_timeout_cycles_(timeout_cycles),
          completion_timeout_ms_(completion_timeout_ms),
          testing_lifecycle_control_(std::move(lifecycle_control)) {
        completion_resources_ =
            std::make_shared<ElasticAsyncCompletionResources>(
                std::move(resources), dispatch_family,
                last_dispatch_generation, rank);
        resources_ = completion_resources_->runtime();
        stage_profile_enabled_ = transport::has_capability(
            resources_->transport()->capabilities(),
            transport::TransportCapability::kStageProfile);
        async_state_ = std::make_shared<elastic::AsyncBufferState>(
            completion_resources_, completion_timeout_ms);
    }
#endif

public:
    using cpu_comm_t = std::vector<std::pair<int, int>>;
#if DEEP_EP_ASCEND_TESTING
    using TestingLifecycleControl = ElasticBufferTestingLifecycleControl;
#endif

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
        config.stage_profile_enabled =
            environment_is("DEEP_EP_ASCEND_PROFILE_STAGES", "1");
        stage_profile_enabled_ = config.stage_profile_enabled;
        const auto topology_status =
            transport::configure_transport_topology_from_environment(&config);
        if (!topology_status.ok())
            raise_transport_status(topology_status, rank_idx_);
        auto resources = std::make_unique<runtime::CannRuntimeResources>();
        const auto status = resources->initialize(
            config, elastic::default_elastic_runtime_workspace_bytes());
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        completion_resources_ =
            std::make_shared<ElasticAsyncCompletionResources>(
                std::move(resources), dispatch_family, 0, rank_idx_);
        resources_ = completion_resources_->runtime();
        async_state_ = std::make_shared<elastic::AsyncBufferState>(
            completion_resources_, completion_timeout_ms_);
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
        bool allow_hybrid_mode = false,
        std::uint64_t completion_timeout_ms = 5000,
        std::shared_ptr<ElasticBufferTestingLifecycleControl>
            lifecycle_control = nullptr) {
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
            last_dispatch_generation, allow_hybrid_mode,
            completion_timeout_ms, std::move(lifecycle_control)));
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
#if DEEP_EP_ASCEND_TESTING
        if (testing_lifecycle_control_ != nullptr)
            testing_lifecycle_control_->note_destroy_attempt();
#endif
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (async_state_ == nullptr)
            return;
#if DEEP_EP_ASCEND_TESTING
        ElasticBufferTestingLifecycleOwner testing_owner(
            testing_lifecycle_control_, false);
#endif
        TORCH_CHECK(!async_state_->finalization_in_progress(),
                    "DeepEP Ascend backend: destroy is busy on this buffer");
        const auto status = async_state_->destroy();
        if (completion_resources_ == nullptr ||
            completion_resources_->runtime() == nullptr) {
            resources_ = nullptr;
            completion_resources_.reset();
            async_state_.reset();
        }
        if (!status.ok() && status.operation == "destroy_async_state" &&
            status.message == "operation coordinator is busy")
            TORCH_CHECK(false,
                        "DeepEP Ascend backend: destroy is busy on this buffer");
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
    }

    bool is_destroyed() const {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        return async_state_ == nullptr;
    }

    std::uint64_t get_dispatch_handle_generation(
        const torch::Tensor& descriptor_tensor) const {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        if (completion_resources_ == nullptr)
            return 0;
        return completion_resources_->dispatch_handle_generation(
            descriptor_tensor);
    }

    void reset_stage_profile() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        TORCH_CHECK(resources_ != nullptr,
                    "DeepEP Ascend backend: runtime is destroyed");
        host_timeline_profile_.reset(0);
        const auto status = host_transport()->reset_stage_profile();
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
    }

    pybind11::dict get_stage_profile() const {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
        pybind11::dict result;
        if (resources_ == nullptr || async_state_ == nullptr) {
            result["available"] = false;
            result["reason"] = "runtime_destroyed";
            return result;
        }

        transport::TransportStageProfile profile{};
        const auto status = host_transport()->read_stage_profile(&profile);
        if (status.code ==
                transport::TransportStatusCode::kUnsupportedCapability) {
            result["available"] = false;
            result["reason"] = "disabled";
            return result;
        }
        if (!status.ok())
            raise_transport_status(status, rank_idx_);

        const auto unavailable = [&result](const char* reason) {
            result["available"] = false;
            result["reason"] = reason;
        };
        if (profile.abi_version !=
                transport::kTransportStageProfileAbiVersion ||
            profile.struct_size != sizeof(transport::TransportStageProfile)) {
            unavailable("abi_mismatch");
            return result;
        }
        if (profile.operation !=
                transport::TransportProfileOperation::kDispatch &&
            profile.operation !=
                transport::TransportProfileOperation::kCombine) {
            unavailable("operation_unavailable");
            return result;
        }
        const auto expected_generation =
            async_state_->coordinator().last_generation();
        if (profile.generation == 0 ||
            profile.generation != expected_generation) {
            unavailable("stale_generation");
            return result;
        }
        if (profile.completion_generation != profile.generation) {
            unavailable("partial_generation");
            return result;
        }
        pybind11::dict command_metrics;
        command_metrics["command_count"] = profile.command_count;
        command_metrics["put_command_count"] = profile.put_command_count;
        command_metrics["payload_bytes"] = profile.command_bytes;
        command_metrics["sq_depth"] = profile.sq_depth;
        command_metrics["cq_depth"] = profile.cq_depth;
        command_metrics["sq_high_watermark"] = profile.sq_high_watermark;
        command_metrics["cq_high_watermark"] = profile.cq_high_watermark;
        result["command_metrics"] = command_metrics;
        pybind11::dict raw_service;
        raw_service["start"] = profile.service_start_cycles;
        raw_service["end"] = profile.service_end_cycles;
        raw_service["wait_cycles"] = profile.wait_cycles;
        raw_service["payload_command_cycles"] =
            profile.payload_command_cycles;
        raw_service["control_command_cycles"] =
            profile.control_command_cycles;
        raw_service["flush_command_cycles"] =
            profile.flush_command_cycles;
        raw_service["barrier_command_cycles"] =
            profile.barrier_command_cycles;
        raw_service["barrier_poll_cycles"] = profile.barrier_poll_cycles;
        result["service"] = raw_service;
        const auto command_metrics_status =
            transport::transport_stage_profile_command_metrics_status(
                profile, true);
        if (command_metrics_status !=
                transport::TransportStageProfileCommandMetricsStatus::kValid) {
            unavailable(
                transport::transport_stage_profile_command_metrics_reason(
                    command_metrics_status));
            return result;
        }
        const auto mask_status = transport::stage_profile_mask_status(
            profile.operation, profile.valid_stage_mask);
        if (mask_status != transport::TransportStageProfileMaskStatus::kValid) {
            unavailable(mask_status ==
                        transport::TransportStageProfileMaskStatus::kNoStages ?
                    "no_stages" : mask_status ==
                        transport::TransportStageProfileMaskStatus::kInvalidMask ?
                    "invalid_stage_mask" : "partial_stage_mask");
            return result;
        }

        const bool dispatch = profile.operation ==
            transport::TransportProfileOperation::kDispatch;
        constexpr const char* dispatch_stage_names[] = {
            "full", "producer_control", "producer_group",
            "producer_prefix", "producer_record", "release_payload",
            "epilogue_acquire", "epilogue_validate",
            "epilogue_validate_reduce", "epilogue_expert_count",
            "epilogue_expert_prefix", "epilogue_metadata",
            "epilogue_copy", "epilogue_complete", "release_control",
            "release_barrier",
        };
        constexpr const char* combine_stage_names[] = {
            "full", "producer_control", "producer_plan",
            "producer_plan_prefix", "producer_record", "release_payload",
            "epilogue_acquire", "epilogue_validate",
            "epilogue_validate_reduce", "epilogue_reduce",
            "epilogue_weights", "epilogue_complete", "release_control",
            "release_barrier", "producer_local_copy",
        };
        const auto stage_name = [
            dispatch, &dispatch_stage_names,
            &combine_stage_names](std::uint32_t stage) {
            if (dispatch &&
                stage < sizeof(dispatch_stage_names) /
                            sizeof(dispatch_stage_names[0]))
                return dispatch_stage_names[stage];
            if (!dispatch &&
                stage < sizeof(combine_stage_names) /
                            sizeof(combine_stage_names[0]))
                return combine_stage_names[stage];
            return "unknown";
        };

        std::uint64_t stage_spans[transport::kTransportProfileStageCount]{};
        pybind11::list stages;
        for (std::uint32_t stage = 0;
             stage < transport::kTransportProfileStageCount; ++stage) {
            if ((profile.valid_stage_mask & (std::uint64_t{1} << stage)) == 0)
                continue;
            const auto& record = profile.stages[stage];
            if (record.block_count == 0 ||
                record.block_count > transport::kTransportProfileMaxBlocks) {
                unavailable("invalid_block_count");
                result["stage"] = stage;
                result["block"] = 0;
                result["start"] = record.blocks[0].start;
                result["end"] = record.blocks[0].end;
                return result;
            }
            std::uint64_t first = std::numeric_limits<std::uint64_t>::max();
            std::uint64_t last = 0;
            pybind11::list blocks;
            for (std::uint32_t block = 0; block < record.block_count; ++block) {
                const auto& cycles = record.blocks[block];
                if (cycles.start == 0 || cycles.end < cycles.start) {
                    unavailable("partial_stage");
                    result["stage"] = stage;
                    result["block"] = block;
                    result["start"] = cycles.start;
                    result["end"] = cycles.end;
                    return result;
                }
                first = std::min(first, cycles.start);
                last = std::max(last, cycles.end);
                pybind11::dict block_record;
                block_record["block"] = block;
                block_record["start"] = cycles.start;
                block_record["end"] = cycles.end;
                blocks.append(block_record);
            }
            stage_spans[stage] = last - first;
            pybind11::dict stage_record;
            stage_record["id"] = stage;
            stage_record["name"] = stage_name(stage);
            stage_record["block_count"] = record.block_count;
            stage_record["start"] = first;
            stage_record["end"] = last;
            stage_record["span_cycles"] = stage_spans[stage];
            stage_record["blocks"] = blocks;
            stages.append(stage_record);
        }

        if (!transport::transport_stage_profile_service_cycles_valid(
                profile.service_start_cycles, profile.service_end_cycles,
                profile.wait_cycles, profile.barrier_poll_cycles)) {
            unavailable("invalid_service_cycles");
            return result;
        }
        const auto service_cycles =
            profile.service_end_cycles - profile.service_start_cycles;
        const auto phases = transport::derive_stage_profile_phase_cycles(
            profile.operation, profile.valid_stage_mask, stage_spans,
            profile.service_start_cycles, profile.service_end_cycles,
            profile.wait_cycles, profile.barrier_poll_cycles);

        pybind11::dict phase_cycles;
        phase_cycles["producer"] = phases.producer;
        phase_cycles["publication"] = phases.publication;
        phase_cycles["service_submit"] = phases.service_submit;
        phase_cycles["cq_wait"] = phases.cq_wait;
        phase_cycles["barrier_wait"] = phases.barrier_wait;
        phase_cycles["consumer_wait"] = phases.consumer_wait;
        phase_cycles["consumer_compute"] = phases.consumer_compute;
        phase_cycles["epilogue"] = phases.epilogue;

        raw_service["cycles"] = service_cycles;

        result["available"] = true;
        result["abi_version"] = profile.abi_version;
        result["operation"] = dispatch ? "dispatch" : "combine";
        result["generation"] = profile.generation;
        result["completion_generation"] = profile.completion_generation;
        result["stages"] = stages;
        result["service"] = raw_service;
        result["phase_cycles"] = phase_cycles;
        if (host_timeline_profile_.generation == profile.generation) {
            pybind11::dict host_timeline_ns;
            for (std::size_t index = 0;
                 index < runtime::HostTimelineProfile::kPhaseCount; ++index) {
                const auto phase = static_cast<runtime::HostTimelinePhase>(index);
                host_timeline_ns[runtime::host_timeline_phase_name(phase)] =
                    host_timeline_profile_.phase_ns(phase);
            }
            host_timeline_ns["total"] = host_timeline_profile_.total_ns();
            result["host_timeline_ns"] = host_timeline_ns;
        }
        return result;
    }

    c10::Stream get_comm_stream() const {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
#if DEEP_EP_ASCEND_TESTING
        ElasticBufferTestingLifecycleOwner testing_owner(
            testing_lifecycle_control_, true);
#endif
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
            status = dependency.event->wait(
                stream, &dependency.wait_lease);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        runtime::NativeEventCreateResult completion{};
        std::vector<elastic::EventDependency> predecessors;
        std::uint64_t completion_offset = 0;
        TORCH_CHECK(
            elastic::checked_rank_slot_offset(
                tiling.symmetric_window_layout.barrier_completion_offset,
                tiling.symmetric_window_layout.barrier_completion_count,
                rank_idx_, &completion_offset),
            "Invalid Ascend barrier completion slot for rank ", rank_idx_);
        completion = resources_->create_event();
        if (!completion.status.ok())
            raise_transport_status(completion.status, rank_idx_);
        if (dependency.event != nullptr)
            predecessors.emplace_back(std::move(dependency));
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

        auto published = async_state_->publish(
            std::move(lease), completion.event,
            {elastic::BufferOperationKind::kBarrier, generation,
             completion_offset,
             tiling.workspace_layout.scratch_status_offset},
            {}, std::move(predecessors));
        if (!published.status.ok())
            raise_transport_status(published.status, rank_idx_);
        status = completion.event->record(stream);
        if (!status.ok()) {
            (void)published.operation->finish(0);
            raise_transport_status(status, rank_idx_);
        }
        status = async_state_->finish_pending();
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        if (with_cpu_sync) {
            status = resources_->synchronize_device();
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
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
        if (stage_profile_enabled_)
            host_timeline_profile_.reset(0);
        const auto dispatch_prelaunch_start_ns = host_profile_start();
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
        elastic::DispatchDevicePrefixConfig device_prefix_config{};
        const auto device_prefix_config_status =
            elastic::select_dispatch_device_prefix_config(
                std::getenv("DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX"),
                cached_mode, do_cpu_sync, allow_hybrid_mode_, stream_mode,
                &device_prefix_config);
        TORCH_CHECK(
            device_prefix_config_status !=
                elastic::DispatchDevicePrefixConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX must be 0 or 1");
        const bool split_dispatch = !cached_mode && do_cpu_sync &&
            !allow_hybrid_mode_ && !device_prefix_config.enabled;
        elastic::DispatchConsumerTileConfig consumer_tile_config{};
        const auto consumer_tile_config_status =
            elastic::select_dispatch_consumer_tile_config(
                std::getenv(
                    "DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES"),
                device_prefix_config.enabled, cached_mode, do_cpu_sync,
                do_expand, allow_hybrid_mode_, stream_mode,
                &consumer_tile_config);
        TORCH_CHECK(
            consumer_tile_config_status !=
                elastic::DispatchConsumerTileConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES must be one of "
            "512, 1024, 2048, or 4096");
        elastic::DispatchParallelPrefixConfig parallel_prefix_config{};
        const auto parallel_prefix_config_status =
            elastic::select_dispatch_parallel_prefix_config(
                std::getenv(
                    "DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX"),
                device_prefix_config.enabled, cached_mode, do_cpu_sync,
                do_expand, allow_hybrid_mode_, stream_mode,
                &parallel_prefix_config);
        TORCH_CHECK(
            parallel_prefix_config_status !=
                elastic::DispatchParallelPrefixConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX must be 0 or 1");
        TORCH_CHECK(!stream_mode ||
                        ((cached_mode || split_dispatch) &&
                         !allow_hybrid_mode_),
                    "DeepEP Ascend backend: dispatch stream overlap requires "
                    "pure-scale-up mode with cached metadata or CPU sync");
        TORCH_CHECK(do_cpu_sync || cached_mode,
                    "DeepEP Ascend backend: dispatch requires do_cpu_sync unless cached");
        TORCH_CHECK(num_sms >= 1 &&
                        num_sms <= static_cast<int>(
                            elastic::kAscendMaxDataBlocks) &&
                        (!allow_hybrid_mode_ || num_sms == 1) &&
                        num_qps == 0,
                    "DeepEP Ascend backend: dispatch requires num_sms in "
                    "[1, 72] for direct scale-up, num_sms=1 for hybrid, "
                    "and num_qps=0");
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
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kDispatch, "dispatch");
        require_transport("dispatch",
                          operation_capabilities(kDispatchCapabilities));
        int current_device = -1;
        auto device_status = resources_->current_device(&current_device);
        if (!device_status.ok())
            raise_transport_status(device_status, rank_idx_);
        TORCH_CHECK(device.index() == resources_->owning_device() &&
                        current_device == resources_->owning_device(),
                    "DeepEP Ascend backend: dispatch tensors and current NPU "
                    "must match the buffer device");
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
        if (allow_multiple_reduction_)
            mode_flags |= elastic::mode_bit(
                elastic::CoreMode::kAllowMultipleReduction);
        if (allow_hybrid_mode_)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kHybrid);
        if (split_dispatch)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kCpuSync);
        if (split_dispatch && async_with_compute_stream)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kAsyncEvent);
        const auto num_tokens = static_cast<std::uint64_t>(x.size(0));
        const auto hidden = static_cast<std::uint64_t>(x.size(1));
        const auto num_topk = static_cast<std::uint64_t>(topk_idx.size(1));
        const auto capacity = static_cast<std::uint64_t>(num_max_tokens_per_rank);
        const auto experts = static_cast<std::uint64_t>(num_experts);
        const auto alignment = static_cast<std::uint64_t>(expert_alignment);
        elastic::DispatchPipelineConfig pipeline_config{};
        const auto pipeline_config_status =
            elastic::select_dispatch_pipeline_config(
                std::getenv(
                    "DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_SLOTS"),
                cached_mode, split_dispatch, stream_mode, allow_hybrid_mode_,
                num_ranks_, capacity, &pipeline_config);
        TORCH_CHECK(
            pipeline_config_status !=
                elastic::DispatchPipelineConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_SLOTS must be a "
            "positive decimal integer");
        elastic::DispatchSourcePipelineConfig source_pipeline_config{};
        const auto source_pipeline_config_status =
            elastic::select_dispatch_source_pipeline_config(
                std::getenv(
                    "DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES"),
                device_prefix_config.enabled, cached_mode, do_cpu_sync,
                do_expand, allow_hybrid_mode_, stream_mode, num_ranks_,
                num_tokens, &source_pipeline_config);
        TORCH_CHECK(
            source_pipeline_config_status !=
                elastic::DispatchSourcePipelineConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES must be a "
            "positive decimal integer");
        TORCH_CHECK(
            !(source_pipeline_config.enabled && pipeline_config.enabled),
            "DeepEP Ascend backend: dispatch source-tile and slot pipelines "
            "are mutually exclusive");
        if (source_pipeline_config.enabled || pipeline_config.enabled)
            mode_flags |= elastic::mode_bit(elastic::CoreMode::kPipeline);
        const auto tiling = build_dispatch_tiling(
            num_tokens, hidden, experts, num_topk, alignment, capacity,
            mode_flags,
            fp8_dispatch ? elastic::ElementKind::kFloat8E4M3 :
                           elastic::ElementKind::kBFloat16,
            num_scale_factor_packs,
            static_cast<std::uint32_t>(num_sms));
        elastic::DispatchTokenFanoutConfig token_fanout_config{};
        const auto token_fanout_config_status =
            elastic::select_dispatch_token_fanout_config(
                std::getenv("DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT"),
                !allow_hybrid_mode_ && !cached_mode && !do_expand &&
                    tiling.data_launch.num_blocks > 1,
                fp8_dispatch, cached_mode, do_expand, allow_hybrid_mode_,
                stream_mode,
                pipeline_config.enabled || source_pipeline_config.enabled,
                num_topk, num_ranks_,
                tiling.token_layout.hidden_bytes, &token_fanout_config);
        TORCH_CHECK(
            token_fanout_config_status !=
                elastic::DispatchTokenFanoutConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT must be 0 or 1");
        elastic::DispatchEarlyRoutePlanConfig early_route_plan_config{};
        const auto early_route_plan_config_status =
            elastic::select_dispatch_early_route_plan_config(
                std::getenv(
                    "DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN"),
                !allow_hybrid_mode_ && !cached_mode && !do_expand &&
                    tiling.data_launch.num_blocks > 1,
                fp8_dispatch, cached_mode, do_expand, allow_hybrid_mode_,
                stream_mode, pipeline_config.enabled, experts, num_topk,
                num_ranks_, &early_route_plan_config);
        TORCH_CHECK(
            early_route_plan_config_status !=
                elastic::DispatchEarlyRoutePlanConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN must be 0 or 1");
        if (early_route_plan_config.enabled &&
            tiling.symmetric_window_layout.dispatch_route_plan_slot_bytes >
                tiling.symmetric_window_layout.dispatch_staging_shard_bytes)
            early_route_plan_config.enabled = false;
        TORCH_CHECK(tiling.communication_buffer_bytes <=
                        static_cast<std::uint64_t>(num_buffer_bytes_) &&
                        tiling.workspace_bytes <= resources_->workspace_bytes(),
                    "DeepEP Ascend backend: dispatch capacity exceeds runtime storage");

        const auto descriptor_mode_flags = mode_flags &
            ~(elastic::mode_bit(elastic::CoreMode::kCached) |
              elastic::mode_bit(
                  elastic::CoreMode::kAllowMultipleReduction) |
              elastic::mode_bit(elastic::CoreMode::kCpuSync) |
              elastic::mode_bit(elastic::CoreMode::kAsyncEvent) |
              elastic::mode_bit(elastic::CoreMode::kPipeline));
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
        constexpr std::uint64_t kCountBridgeAlignmentElements = 16;
        const auto count_bridge_layout = elastic::dispatch_count_bridge_layout(
            static_cast<std::uint64_t>(num_ranks_), experts, local_experts,
            kCountBridgeAlignmentElements);
        TORCH_CHECK(max_recv_tokens <= static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max()) &&
                        expanded_records <= static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) &&
                        count_bridge_layout.valid &&
                        count_bridge_layout.kernel_elements <=
                            static_cast<std::uint64_t>(
                                std::numeric_limits<int64_t>::max()) &&
                        count_bridge_layout.public_elements <=
                            static_cast<std::uint64_t>(
                                std::numeric_limits<int64_t>::max()),
                    "DeepEP Ascend backend: dispatch output count overflow");

        runtime::StreamIdentity dispatch_stream;
        elastic::EventDependency predecessor;
        elastic::EnqueuedEventDependencyGuard predecessor_guard;
        const bool use_comm_stream = cached_mode || stream_mode;
        if (use_comm_stream) {
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
            auto retirement = resources_->create_event();
            if (!retirement.status.ok())
                raise_transport_status(retirement.status, rank_idx_);
            predecessor.retirement_event = std::move(retirement.event);
            predecessor.retirement_stream = dispatch_stream;
            predecessor_guard.adopt(
                std::move(predecessor), async_state_);
            status = predecessor_guard.dependency().event->wait(
                dispatch_stream,
                &predecessor_guard.dependency().wait_lease);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            predecessor_guard.arm();
            status = predecessor_guard.dependency().retirement_event->record(
                dispatch_stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            predecessor_guard.mark_retirement_recorded();
#if DEEP_EP_ASCEND_TESTING
            inject_testing_post_wait_failure("allocation");
#endif
        }
#if DEEP_EP_ASCEND_HAS_GENERIC_STREAM_GUARD
        std::optional<c10::StreamGuard> allocation_guard;
        if (allocate_on_comm_stream)
            allocation_guard.emplace(get_comm_stream());
#endif

        torch::Tensor rank_prefix;
        torch::Tensor expert_prefix;
        torch::Tensor unaligned;
        torch::Tensor public_count_bridge;
        torch::Tensor destination_slots;
        torch::Tensor source_metadata;
        auto kernel_count_bridge = torch::empty(
            {static_cast<int64_t>(count_bridge_layout.kernel_elements)},
            int_options);
        auto kernel_expert_prefix = kernel_count_bridge.narrow(
            0,
            static_cast<int64_t>(
                count_bridge_layout.kernel_expert_prefix_offset),
            num_experts + 1);
        auto kernel_unaligned = kernel_count_bridge.narrow(
            0,
            static_cast<int64_t>(count_bridge_layout.kernel_unaligned_offset),
            num_experts);
        std::vector<std::int32_t> host_rank_prefix(num_ranks_);
        std::vector<std::int32_t> host_kernel_expert_prefix(num_experts + 1);
        std::vector<std::int32_t> host_kernel_unaligned(num_experts);
        std::vector<std::int32_t> host_expert_prefix(local_experts);
        std::vector<std::int32_t> host_unaligned(local_experts);
        std::vector<std::int32_t> host_kernel_count_bridge(
            static_cast<std::size_t>(count_bridge_layout.kernel_elements));
        std::vector<std::int32_t> host_public_count_bridge(
            static_cast<std::size_t>(count_bridge_layout.public_elements));
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
            rank_prefix = kernel_count_bridge.narrow(
                0,
                static_cast<int64_t>(count_bridge_layout.rank_prefix_offset),
                num_ranks_);
            public_count_bridge = torch::empty(
                {static_cast<int64_t>(count_bridge_layout.public_elements)},
                int_options);
            expert_prefix = public_count_bridge.narrow(
                0,
                static_cast<int64_t>(
                    count_bridge_layout.public_expert_prefix_offset),
                static_cast<int64_t>(local_experts));
            unaligned = public_count_bridge.narrow(
                0,
                static_cast<int64_t>(
                    count_bridge_layout.public_unaligned_offset),
                static_cast<int64_t>(local_experts));
            destination_slots = torch::empty({x.size(0), topk_idx.size(1)}, int_options);
            source_metadata = torch::empty(
                {static_cast<int64_t>(split_dispatch ? 0 : max_recv_tokens),
                 topk_idx.size(1) + 2},
                int_options);
        }

        const auto initial_recv_tokens = static_cast<int64_t>(
            split_dispatch ? 0 : max_recv_tokens);
        const auto initial_output_tokens = static_cast<int64_t>(
            split_dispatch ? 0 :
            (do_expand ? expanded_records : max_recv_tokens));
        auto recv_x = torch::empty(
            {initial_output_tokens, x.size(1)}, x.options());
        const auto allocate_recv_sf = [&](int64_t output_tokens) {
            if (!sf.has_value())
                return std::optional<torch::Tensor>();
            const auto sf_token_stride = use_tma_aligned_col_major_sf ?
                int64_t{1} : static_cast<int64_t>(num_scale_factor_packs);
            std::uint64_t aligned_tokens = 0;
            TORCH_CHECK(
                !use_tma_aligned_col_major_sf ||
                    align_without_overflow(
                        static_cast<std::uint64_t>(output_tokens), 4,
                        &aligned_tokens),
                "DeepEP Ascend backend: dispatch scale factor output stride overflow");
            const auto sf_pack_stride = use_tma_aligned_col_major_sf ?
                static_cast<int64_t>(aligned_tokens) : int64_t{1};
            return std::optional<torch::Tensor>(torch::empty_strided(
                {output_tokens, static_cast<int64_t>(num_scale_factor_packs)},
                {sf_token_stride, sf_pack_stride}, sf->options()));
        };
        const auto initial_sf_tokens = cached_mode ?
            static_cast<int64_t>(do_expand ? *cached_num_expanded_tokens :
                                             *cached_num_recv_tokens) :
            initial_output_tokens;
        auto recv_sf = allocate_recv_sf(initial_sf_tokens);
        auto recv_topk_indices = torch::empty(
            {initial_recv_tokens, topk_idx.size(1)},
            topk_idx.options());
        auto recv_topk_weights = std::optional<torch::Tensor>();
        if (topk_weights.has_value()) {
            recv_topk_weights = do_expand ?
                torch::empty(
                    {initial_output_tokens},
                    topk_weights->options()) :
                torch::empty(
                    {initial_recv_tokens, topk_idx.size(1)},
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
        const auto retain = [&](const std::optional<torch::Tensor>& tensor) {
            if (!use_comm_stream)
                return;
            retained_tensors.emplace_back(tensor);
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
            if (tensor.has_value())
                torch::deep_ep_ascend_test_record_tensor_stream(
                    *tensor, dispatch_stream.raw);
#else
            if (tensor.has_value()) {
                const auto record_status = resources_->record_tensor_stream(
                    *tensor, dispatch_stream);
                if (!record_status.ok())
                    raise_transport_status(record_status, rank_idx_);
            }
#endif
        };
        if (use_comm_stream) {
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
            if (!split_dispatch) {
                retain(source_metadata);
                retain(recv_x);
                retain(recv_sf);
                retain(recv_topk_indices);
                retain(recv_topk_weights);
            }
            retain(copied_topk_idx);
            retain(descriptor_tensor);
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
        arguments.num_recv_tokens = static_cast<std::uint64_t>(
            cached_mode ? *cached_num_recv_tokens : initial_recv_tokens);
        arguments.num_output_tokens = static_cast<std::uint64_t>(
            cached_mode ?
                (do_expand ? *cached_num_expanded_tokens :
                             *cached_num_recv_tokens) :
                initial_output_tokens);
        if (allow_hybrid_mode_) {
            arguments.route_records =
                reinterpret_cast<elastic::HybridRouteRecord*>(
                    static_cast<std::uint8_t*>(descriptor_tensor.data_ptr()) +
                    sizeof(elastic::DispatchHandleDescriptor));
            arguments.route_record_capacity = cached_mode ?
                cached_route_count : tiling.hybrid_route_capacity;
        }
        arguments.timeout_cycles = barrier_timeout_cycles_;
        arguments.pipeline_chunk_slots = pipeline_config.chunk_slots;
        arguments.pipeline_chunk_tiles = source_pipeline_config.chunk_tiles;
        arguments.consumer_tile_bytes = consumer_tile_config.tile_bytes;
        arguments.parallel_prefix =
            parallel_prefix_config.enabled ? 1U : 0U;
        arguments.token_fanout =
            token_fanout_config.enabled ? 1U : 0U;
        arguments.early_route_plan =
            early_route_plan_config.enabled ? 1U : 0U;
        runtime::StreamIdentity stream = dispatch_stream;
        if (!use_comm_stream) {
            status = resources_->current_stream(&stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
        }
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_), resources_->workspace_bytes()};
        runtime::NativeEventCreateResult completion{};
        std::vector<elastic::EventDependency> predecessors;
        if (cached_mode || split_dispatch) {
            if (use_comm_stream)
                predecessors.resize(1);
        }
        if (cached_mode) {
            completion = resources_->create_event();
            if (!completion.status.ok())
                raise_transport_status(completion.status, rank_idx_);
        }
#if DEEP_EP_ASCEND_TESTING
        if (cached_mode)
            inject_testing_post_wait_failure("activation");
#endif
        const auto generation = activate_operation(lease, "dispatch");
        arguments.generation = generation;
        host_profile_record(
            runtime::HostTimelinePhase::kDispatchPrelaunchSetup,
            dispatch_prelaunch_start_ns);
        if (stage_profile_enabled_)
            (void)host_timeline_profile_.bind_generation(generation);
        if (cached_mode) {
            const auto committed_descriptor = allow_hybrid_mode_ ?
                elastic::make_attested_hybrid_dispatch_handle_descriptor(
                    completion_resources_->dispatch_family(), tiling.topology,
                    generation, num_tokens, hidden, experts, num_topk,
                    alignment, capacity, descriptor_mode_flags,
                    elastic::kHybridRouteLayoutVersion,
                    host_route_records.size(),
                    sizeof(elastic::HybridRouteRecord),
                    elastic::kHybridRouteCompleteStageFlags,
                    {host_route_records.data(), host_route_records.size()}) :
                elastic::make_attested_dispatch_handle_descriptor(
                    completion_resources_->dispatch_family(), tiling.topology,
                    generation, num_tokens, hidden, experts, num_topk,
                    alignment, capacity, descriptor_mode_flags);
            completion_resources_->stage_dispatch_descriptor(
                generation, descriptor_tensor, committed_descriptor,
                dispatch_descriptor_snapshot(
                    committed_descriptor, host_route_records));
        }
        const auto launch_status =
            (source_pipeline_config.enabled || pipeline_config.enabled) ?
            elastic::launch_internal_dispatch_pipeline(
                arguments, tiling, storage, stream.raw,
                resources_->comm_stream().raw) :
            elastic::launch_internal_dispatch(
                arguments, tiling, storage, stream.raw);
        if (!launch_status.ok())
            raise_launch_status(launch_status, rank_idx_);
        if (cached_mode) {
            predecessor_guard.copy_to(predecessors.front());
            const auto completion_offset =
                tiling.symmetric_window_layout.control_offset +
                offsetof(elastic::SymmetricControlHeader, dispatch_generation);
            auto published = async_state_->publish(
                std::move(lease), completion.event,
                {elastic::BufferOperationKind::kDispatch, generation,
                 completion_offset,
                 tiling.workspace_layout.scratch_status_offset},
                std::move(retained_tensors), std::move(predecessors));
            if (!published.status.ok())
                raise_transport_status(published.status, rank_idx_);
            predecessor_guard.dismiss();
            status = completion.event->record(stream);
            if (!status.ok()) {
                (void)published.operation->finish(0);
                raise_transport_status(status, rank_idx_);
            }

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
            auto narrowed_sf = recv_sf.has_value() ?
                std::optional<torch::Tensor>(
                    recv_sf->narrow(0, 0, output_tokens)) :
                std::optional<torch::Tensor>();
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
            return {narrowed_x, narrowed_sf, narrowed_topk_idx,
                    narrowed_topk_weights, copied_topk_idx, num_recv_tokens,
                    *cached_num_expanded_tokens, per_expert_list, rank_prefix,
                    expert_prefix, unaligned, narrowed_metadata,
                    destination_slots, descriptor_tensor, std::nullopt,
                    std::move(event)};
        }
        auto host_phase_start_ns = host_profile_start();
        status = resources_->synchronize_stream(stream.raw);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        transport::DeviceTransportDiagnostic diagnostic{};
        status = host_transport()->read_diagnostic(&diagnostic);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
#if DEEP_EP_ASCEND_TESTING
        if (testing_environment_is(
                "DEEP_EP_ASCEND_TEST_DISPATCH_DIAGNOSTIC",
                "completion_timeout")) {
            diagnostic.abi_version = transport::kTransportCommandAbiVersion;
            diagnostic.error = transport::DeviceTransportError::kCompletionTimeout;
            diagnostic.generation = generation;
        }
#endif
        if (diagnostic.abi_version != transport::kTransportCommandAbiVersion ||
            diagnostic.error != transport::DeviceTransportError::kNone ||
            diagnostic.generation != generation)
            raise_dispatch_diagnostic(diagnostic, "reported failure");
        host_profile_record(
            runtime::HostTimelinePhase::kDispatchSynchronize,
            host_phase_start_ns);
        if (!cached_mode) {
            host_phase_start_ns = host_profile_start();
            status = resources_->copy_to_host(
                host_kernel_count_bridge.data(),
                kernel_count_bridge.data_ptr(),
                host_kernel_count_bridge.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            std::memcpy(
                host_rank_prefix.data(),
                host_kernel_count_bridge.data() +
                    count_bridge_layout.rank_prefix_offset,
                host_rank_prefix.size() * sizeof(std::int32_t));
            std::memcpy(
                host_kernel_expert_prefix.data(),
                host_kernel_count_bridge.data() +
                    count_bridge_layout.kernel_expert_prefix_offset,
                host_kernel_expert_prefix.size() * sizeof(std::int32_t));
            std::memcpy(
                host_kernel_unaligned.data(),
                host_kernel_count_bridge.data() +
                    count_bridge_layout.kernel_unaligned_offset,
                host_kernel_unaligned.size() * sizeof(std::int32_t));
            host_profile_record(
                runtime::HostTimelinePhase::kDispatchCountsToHost,
                host_phase_start_ns);
            host_phase_start_ns = host_profile_start();
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
                            "DeepEP Ascend backend: dispatch returned invalid expert counts: expert=",
                            expert, ", actual=", actual, ", local=", local,
                            ", prefix=", host_kernel_expert_prefix[expert],
                            ", expected_prefix=", expanded_tail,
                            ", alignment=", alignment);
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
                        "DeepEP Ascend backend: dispatch returned invalid expert counts: tail=",
                        host_kernel_expert_prefix[num_experts],
                        ", expected_tail=", expanded_tail);
            num_expanded_tokens = expanded_tail;
            host_profile_record(
                runtime::HostTimelinePhase::kDispatchHostPrefix,
                host_phase_start_ns);
            host_phase_start_ns = host_profile_start();
            std::fill(
                host_public_count_bridge.begin(),
                host_public_count_bridge.end(), 0);
            std::memcpy(
                host_public_count_bridge.data() +
                    count_bridge_layout.public_expert_prefix_offset,
                host_expert_prefix.data(),
                host_expert_prefix.size() * sizeof(std::int32_t));
            std::memcpy(
                host_public_count_bridge.data() +
                    count_bridge_layout.public_unaligned_offset,
                host_unaligned.data(),
                host_unaligned.size() * sizeof(std::int32_t));
            status = resources_->copy_from_host(
                public_count_bridge.data_ptr(),
                host_public_count_bridge.data(),
                host_public_count_bridge.size() * sizeof(std::int32_t));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            host_profile_record(
                runtime::HostTimelinePhase::kDispatchPrefixToDevice,
                host_phase_start_ns);
        }
        const int num_recv_tokens = host_rank_prefix.back();
        TORCH_CHECK(num_recv_tokens >= 0 &&
                        num_recv_tokens <= static_cast<int>(max_recv_tokens) &&
                        num_expanded_tokens >= 0 &&
                        num_expanded_tokens <= static_cast<int>(expanded_records),
                    "DeepEP Ascend backend: dispatch returned invalid output counts");
        std::uint64_t epilogue_setup_start_ns = 0;
        if (split_dispatch) {
            host_phase_start_ns = host_profile_start();
            const auto output_tokens = static_cast<int64_t>(
                do_expand ? num_expanded_tokens : num_recv_tokens);
            recv_x = torch::empty({output_tokens, x.size(1)}, x.options());
            recv_sf = allocate_recv_sf(output_tokens);
            recv_topk_indices = torch::empty(
                {static_cast<int64_t>(num_recv_tokens), topk_idx.size(1)},
                topk_idx.options());
            recv_topk_weights = topk_weights.has_value() ?
                std::optional<torch::Tensor>(do_expand ?
                    torch::empty({output_tokens}, topk_weights->options()) :
                    torch::empty(
                        {static_cast<int64_t>(num_recv_tokens),
                         topk_idx.size(1)}, topk_weights->options())) :
                std::optional<torch::Tensor>();
            source_metadata = torch::empty(
                {static_cast<int64_t>(num_recv_tokens),
                 topk_idx.size(1) + 2}, int_options);

            arguments.recv_x = recv_x.data_ptr();
            arguments.recv_scale_factors = recv_sf.has_value() ?
                recv_sf->data_ptr() : nullptr;
            arguments.recv_scale_factor_token_stride = recv_sf.has_value() ?
                static_cast<std::uint64_t>(recv_sf->stride(0)) : 0;
            arguments.recv_scale_factor_pack_stride = recv_sf.has_value() ?
                static_cast<std::uint64_t>(recv_sf->stride(1)) : 0;
            arguments.recv_topk_indices =
                recv_topk_indices.data_ptr<std::int64_t>();
            arguments.recv_topk_weights = recv_topk_weights.has_value() ?
                recv_topk_weights->data_ptr<float>() : nullptr;
            arguments.source_metadata =
                source_metadata.data_ptr<std::int32_t>();
            arguments.num_recv_tokens = static_cast<std::uint64_t>(
                num_recv_tokens);
            arguments.num_output_tokens = static_cast<std::uint64_t>(
                output_tokens);

            retain(source_metadata);
            retain(recv_x);
            retain(recv_sf);
            retain(recv_topk_indices);
            retain(recv_topk_weights);

            host_profile_record(
                runtime::HostTimelinePhase::kDispatchOutputAllocation,
                host_phase_start_ns);

            epilogue_setup_start_ns = host_profile_start();
            completion = resources_->create_event();
            if (!completion.status.ok())
                raise_transport_status(completion.status, rank_idx_);
#if DEEP_EP_ASCEND_TESTING
            inject_testing_post_wait_failure("activation");
#endif
        }
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
        std::optional<EventHandle> event;
        if (split_dispatch) {
            completion_resources_->stage_dispatch_descriptor(
                generation, descriptor_tensor, committed_descriptor,
                dispatch_descriptor_snapshot(
                    committed_descriptor, host_route_records));
            host_profile_record(
                runtime::HostTimelinePhase::kDispatchEpilogueSetup,
                epilogue_setup_start_ns);
            host_phase_start_ns = host_profile_start();
            const auto epilogue_status =
                elastic::launch_internal_dispatch_epilogue(
                    arguments, tiling, storage, stream.raw);
            if (!epilogue_status.ok())
                raise_launch_status(epilogue_status, rank_idx_);
            host_profile_record(
                runtime::HostTimelinePhase::kDispatchEpilogueSubmit,
                host_phase_start_ns);
            host_phase_start_ns = host_profile_start();
            if (use_comm_stream)
                predecessor_guard.copy_to(predecessors.front());
            const auto completion_offset =
                tiling.symmetric_window_layout.control_offset +
                offsetof(elastic::SymmetricControlHeader, dispatch_generation);
            auto published = async_state_->publish(
                std::move(lease), completion.event,
                {elastic::BufferOperationKind::kDispatch, generation,
                 completion_offset,
                 tiling.workspace_layout.scratch_status_offset},
                std::move(retained_tensors), std::move(predecessors));
            if (!published.status.ok())
                raise_transport_status(published.status, rank_idx_);
            if (use_comm_stream)
                predecessor_guard.dismiss();
            status = completion.event->record(stream);
            if (!status.ok()) {
                (void)published.operation->finish(0);
                raise_transport_status(status, rank_idx_);
            }
            host_profile_record(
                runtime::HostTimelinePhase::kDispatchCompletionRecord,
                host_phase_start_ns);
            if (async_with_compute_stream) {
                event.emplace(
                    completion.event, published.operation, async_state_);
            } else {
                host_phase_start_ns = host_profile_start();
                status = published.operation->finish(5000);
                if (!status.ok())
                    raise_transport_status(status, rank_idx_);
                host_profile_record(
                    runtime::HostTimelinePhase::kDispatchCompletionWait,
                    host_phase_start_ns);
            }
        } else {
            status = resources_->copy_from_host(
                descriptor_tensor.data_ptr(), &committed_descriptor,
                sizeof(committed_descriptor));
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            completion_resources_->commit_dispatch_descriptor(
                generation, descriptor_tensor,
                dispatch_descriptor_snapshot(
                    committed_descriptor, host_route_records));
            lease.complete();
        }
        const int output_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto narrowed_x = recv_x.narrow(0, 0, output_tokens);
        auto narrowed_sf = recv_sf.has_value() ?
            std::optional<torch::Tensor>(recv_sf->narrow(0, 0, output_tokens)) :
            std::optional<torch::Tensor>();
        if (narrowed_sf.has_value() && use_tma_aligned_col_major_sf &&
            !split_dispatch) {
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
                destination_slots, handle_tensor, std::nullopt,
                std::move(event)};
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
        if (stage_profile_enabled_)
            host_timeline_profile_.reset(0);
        auto combine_host_phase_start_ns = host_profile_start();
        TORCH_CHECK(!previous_event_before_epilogue.has_value(),
                    "DeepEP Ascend backend: combine does not support "
                    "previous_event_before_epilogue");
        TORCH_CHECK(!previous_event.has_value() || allocate_on_comm_stream,
                    "DeepEP Ascend backend: combine previous_event requires "
                    "allocate_on_comm_stream=True");
        const bool stream_mode = previous_event.has_value() ||
            async_with_compute_stream || allocate_on_comm_stream;
        TORCH_CHECK(!stream_mode || !allow_hybrid_mode_,
                    "DeepEP Ascend backend: combine stream overlap requires "
                    "BF16 pure-scale-up mode");
        TORCH_CHECK(!channel_linked_list.has_value(),
                    "DeepEP Ascend backend: combine does not support channel handles");
        TORCH_CHECK(num_sms >= 1 &&
                        num_sms <= static_cast<int>(
                            elastic::kAscendMaxDataBlocks) &&
                        (!allow_hybrid_mode_ || num_sms == 1),
                    "DeepEP Ascend backend: combine requires num_sms in "
                    "[1, 72] for direct scale-up and num_sms=1 for hybrid");
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
        auto lease = reserve_operation(
            elastic::BufferOperationKind::kCombine, "combine");
        require_transport("combine",
                          operation_capabilities(kCombineCapabilities));
        TORCH_CHECK(
            !stream_mode ||
                resources_->device_context().topology.scale_out_size == 1,
            "DeepEP Ascend backend: combine stream overlap requires "
            "BF16 pure-scale-up mode");

        const auto device = x.device();
        int current_device = -1;
        auto status = resources_->current_device(&current_device);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        TORCH_CHECK(device.index() == resources_->owning_device() &&
                        current_device == resources_->owning_device(),
                    "DeepEP Ascend backend: combine tensors and current NPU "
                    "must match the buffer device");
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
        elastic::CombineExpandedVectorReduceConfig expanded_reduce_config{};
        const auto expanded_reduce_config_status =
            elastic::select_combine_expanded_vector_reduce_config(
                std::getenv(
                    "DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE"),
                !allow_hybrid_mode_, use_expanded_layout,
                allow_multiple_reduction_, allow_hybrid_mode_, num_topk,
                &expanded_reduce_config);
        TORCH_CHECK(
            expanded_reduce_config_status !=
                elastic::CombineExpandedVectorReduceConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE must be 0 or 1");
        elastic::CombineLocalCopyDataCopyConfig local_copy_config{};
        const auto local_copy_config_status =
            elastic::select_combine_local_copy_datacopy_config(
                std::getenv(
                    "DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY"),
                !allow_hybrid_mode_, allow_hybrid_mode_, &local_copy_config);
        TORCH_CHECK(
            local_copy_config_status !=
                elastic::CombineLocalCopyDataCopyConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY must be 0, 1, "
            "512, 1024, 2048, 4096, 8192, 16384, or 32768");
        elastic::CombineVectorReduceTileConfig vector_reduce_tile_config{};
        const auto vector_reduce_tile_config_status =
            elastic::select_combine_vector_reduce_tile_config(
                std::getenv(
                    "DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE"),
                !allow_hybrid_mode_, allow_hybrid_mode_,
                &vector_reduce_tile_config);
        TORCH_CHECK(
            vector_reduce_tile_config_status !=
                elastic::CombineVectorReduceTileConfigStatus::kInvalid,
            "DeepEP Ascend backend: "
            "DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE must be 0, 1, or 512");
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

        host_profile_record(
            runtime::HostTimelinePhase::kCombinePrelaunchSetup,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
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
        host_profile_record(
            runtime::HostTimelinePhase::kCombineHandleToHost,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
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
        auto tiling = build_combine_tiling(
            descriptor, combine_mode, static_cast<std::uint32_t>(num_sms));
        const std::uint64_t maximum_input_rows = use_expanded_layout ?
            tiling.dispatch_output_capacity : maximum_source_rows;
        TORCH_CHECK(
            num_input_rows <= maximum_input_rows,
            "DeepEP Ascend backend: combine source metadata shape or "
            "capacity mismatch");

        host_profile_record(
            runtime::HostTimelinePhase::kCombineHostValidation,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
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
        host_profile_record(
            runtime::HostTimelinePhase::kCombineMetadataToHost,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
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

        TORCH_CHECK(tiling.control_launch.num_blocks == 1 &&
                        tiling.data_launch.num_blocks ==
                            static_cast<std::uint32_t>(num_sms) &&
                        tiling.communication_buffer_bytes <=
                            static_cast<std::uint64_t>(num_buffer_bytes_) &&
                        tiling.workspace_bytes <= resources_->workspace_bytes(),
                    "DeepEP Ascend backend: combine capacity exceeds runtime storage");
        host_profile_record(
            runtime::HostTimelinePhase::kCombineHostValidation,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
        runtime::StreamIdentity current_stream;
        status = resources_->current_stream(&current_stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        elastic::EventDependency predecessor;
        elastic::EnqueuedEventDependencyGuard predecessor_guard;
        if (previous_event.has_value()) {
            predecessor = previous_event->dependency();
        } else {
            auto created = resources_->create_event();
            if (!created.status.ok())
                raise_transport_status(created.status, rank_idx_);
            status = created.event->record(current_stream);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            predecessor.event = std::move(created.event);
        }
        const auto stream = resources_->comm_stream();
        auto retirement = resources_->create_event();
        if (!retirement.status.ok())
            raise_transport_status(retirement.status, rank_idx_);
        predecessor.retirement_event = std::move(retirement.event);
        predecessor.retirement_stream = stream;
        predecessor_guard.adopt(
            std::move(predecessor), async_state_);
        status = predecessor_guard.dependency().event->wait(
            stream, &predecessor_guard.dependency().wait_lease);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        predecessor_guard.arm();
        status = predecessor_guard.dependency().retirement_event->record(stream);
        if (!status.ok())
            raise_transport_status(status, rank_idx_);
        predecessor_guard.mark_retirement_recorded();
#if DEEP_EP_ASCEND_TESTING
        inject_testing_post_wait_failure("allocation");
#endif
#if DEEP_EP_ASCEND_HAS_GENERIC_STREAM_GUARD
        std::optional<c10::StreamGuard> allocation_guard;
        if (allocate_on_comm_stream)
            allocation_guard.emplace(get_comm_stream());
#endif

        host_profile_record(
            runtime::HostTimelinePhase::kCombineStreamSetup,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
        auto combined_x = torch::empty(
            {combined_topk_idx.size(0), x.size(1)}, x.options());
        std::optional<torch::Tensor> combined_weights;
        if (topk_weights.has_value())
            combined_weights = torch::empty(
                {combined_topk_idx.size(0), combined_topk_idx.size(1)},
                topk_weights->options());
        host_profile_record(
            runtime::HostTimelinePhase::kCombineOutputAllocation,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();

        std::vector<std::optional<torch::Tensor>> retained_tensors;
        const auto retain = [&retained_tensors](
                                const std::optional<torch::Tensor>& tensor) {
            retained_tensors.emplace_back(tensor);
        };
        retain(x);
        retain(topk_weights);
        retain(bias_0);
        retain(bias_1);
        retain(src_metadata);
        retain(combined_topk_idx);
        retain(psum_num_recv_tokens_per_scaleup_rank);
        retain(token_metadata_at_forward);
        retain(channel_linked_list);
        retain(combined_x);
        retain(combined_weights);
#ifndef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
        for (const auto& tensor : retained_tensors) {
            if (!tensor.has_value())
                continue;
            const auto record_status = resources_->record_tensor_stream(
                *tensor, stream);
            if (!record_status.ok())
                raise_transport_status(record_status, rank_idx_);
        }
#endif

        auto completion = resources_->create_event();
        if (!completion.status.ok())
            raise_transport_status(completion.status, rank_idx_);
        std::vector<elastic::EventDependency> predecessors(1);
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
#if DEEP_EP_ASCEND_TESTING
        inject_testing_post_wait_failure("activation");
#endif
        const auto generation = activate_operation(lease, "combine");
        arguments.generation = generation;
        if (stage_profile_enabled_)
            (void)host_timeline_profile_.bind_generation(generation);
        arguments.timeout_cycles = barrier_timeout_cycles_;
        arguments.num_source_rows = num_source_rows;
        arguments.num_input_rows = num_input_rows;
        arguments.local_window_base = reinterpret_cast<std::uintptr_t>(
            resources_->window_base());
        arguments.expanded_vector_reduce =
            expanded_reduce_config.enabled ? 1U : 0U;
        arguments.local_copy_datacopy = local_copy_config.enabled ?
            local_copy_config.tile_bytes : 0U;
        arguments.vector_reduce_tile_elements =
            vector_reduce_tile_config.enabled ?
                vector_reduce_tile_config.tile_elements : 0U;
        const elastic::CoreLaunchStorage storage{
            static_cast<std::uint64_t>(num_buffer_bytes_),
            resources_->workspace_bytes()};
        completion_resources_->stage_combine_completion(
            generation, tiling.workspace_layout.scratch_status_offset);
        const auto launch_status = elastic::launch_internal_combine(
            arguments, tiling, storage, stream.raw);
        if (!launch_status.ok())
            raise_combine_launch_status(launch_status, rank_idx_);
        predecessor_guard.copy_to(predecessors.front());
        const auto completion_offset =
            tiling.symmetric_window_layout.control_offset +
            offsetof(elastic::SymmetricControlHeader, combine_generation);
        auto published = async_state_->publish(
            std::move(lease), completion.event,
            {elastic::BufferOperationKind::kCombine, generation,
             completion_offset,
             tiling.workspace_layout.scratch_status_offset},
            std::move(retained_tensors), std::move(predecessors));
        if (!published.status.ok())
            raise_transport_status(published.status, rank_idx_);
        predecessor_guard.dismiss();
        host_profile_record(
            runtime::HostTimelinePhase::kCombineSubmit,
            combine_host_phase_start_ns);
        combine_host_phase_start_ns = host_profile_start();
        status = completion.event->record(stream);
        if (!status.ok()) {
            (void)published.operation->finish(0);
            raise_transport_status(status, rank_idx_);
        }
        host_profile_record(
            runtime::HostTimelinePhase::kCombineCompletionRecord,
            combine_host_phase_start_ns);

        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            event.emplace(
                completion.event, published.operation, async_state_);
        } else {
            combine_host_phase_start_ns = host_profile_start();
            status = published.operation->finish(5000);
            if (!status.ok())
                raise_transport_status(status, rank_idx_);
            host_profile_record(
                runtime::HostTimelinePhase::kCombineCompletionWait,
                combine_host_phase_start_ns);
        }
        return {combined_x, combined_weights, std::move(event)};
    }
};

}  // namespace deep_ep::ascend
