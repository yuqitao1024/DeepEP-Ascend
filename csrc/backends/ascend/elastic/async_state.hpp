#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
#include <condition_variable>
#include <mutex>
#endif

#include "operation_coordinator.hpp"
#include "../runtime/stream_event.hpp"
#include "../transport/transport_commands.hpp"
#include "../transport/types.hpp"

namespace torch {
class Tensor;
}

namespace deep_ep::ascend::elastic {

enum class PendingState : std::uint8_t {
    kLaunched,
    kFinalizing,
    kSucceeded,
    kFailed,
};

struct CompletionRecipe {
    BufferOperationKind kind = BufferOperationKind::kTopologyQuery;
    std::uint64_t generation = 0;
    std::uint64_t completion_offset = 0;
    std::uint64_t scratch_status_offset = 0;
};

const char* diagnostic_name(
    transport::DeviceTransportError error) noexcept;
transport::TransportStatus diagnostic_failure_status(
    BufferOperationKind kind,
    const transport::DeviceTransportDiagnostic& diagnostic,
    const char* detail);

class AsyncCompletionResources {
public:
    virtual ~AsyncCompletionResources() = default;

    virtual transport::TransportStatus read_diagnostic(
        BufferOperationKind kind, std::uint64_t scratch_status_offset,
        transport::DeviceTransportDiagnostic* output) = 0;
    virtual transport::TransportStatus read_completion(
        BufferOperationKind kind, std::uint64_t completion_offset,
        std::uint64_t* output) = 0;
    virtual transport::TransportStatus commit_completion(
        BufferOperationKind kind, std::uint64_t generation) = 0;
    virtual transport::TransportStatus destroy() = 0;
};

class PendingOperation {
public:
    ~PendingOperation();

    PendingOperation(const PendingOperation&) = delete;
    PendingOperation& operator=(const PendingOperation&) = delete;

    transport::TransportStatus finish(std::uint64_t timeout_ms);
    PendingState state() const noexcept;
    std::uint64_t generation() const noexcept;

private:
    struct Impl;

    explicit PendingOperation(std::shared_ptr<Impl> impl);
    transport::TransportStatus teardown(std::uint64_t timeout_ms);
    bool replaceable() const noexcept;
    bool teardown_complete() const noexcept;

    std::shared_ptr<Impl> impl_;

    friend class AsyncBufferState;
};

struct EventDependency {
    EventDependency() = default;

    EventDependency(
        std::shared_ptr<runtime::NativeEventState> dependency_event,
        std::shared_ptr<PendingOperation> operation,
        std::shared_ptr<runtime::NativeEventWaitLease> lease)
        : event(std::move(dependency_event)),
          pending_operation(std::move(operation)),
          wait_lease(std::move(lease)) {}

    std::shared_ptr<runtime::NativeEventState> event;
    std::shared_ptr<PendingOperation> pending_operation;
    std::shared_ptr<runtime::NativeEventWaitLease> wait_lease;
    std::shared_ptr<runtime::NativeEventState> retirement_event;
    runtime::StreamIdentity retirement_stream;
    bool retirement_recorded = false;
};

class AsyncBufferState;

class EnqueuedEventDependencyGuard {
public:
    EnqueuedEventDependencyGuard() = default;
    ~EnqueuedEventDependencyGuard();

    EnqueuedEventDependencyGuard(const EnqueuedEventDependencyGuard&) = delete;
    EnqueuedEventDependencyGuard& operator=(
        const EnqueuedEventDependencyGuard&) = delete;

    void adopt(
        EventDependency dependency, std::shared_ptr<AsyncBufferState> owner);
    EventDependency& dependency() noexcept;
    void arm() noexcept;
    void mark_retirement_recorded() noexcept;
    void copy_to(EventDependency& output) const noexcept;
    void dismiss() noexcept;

private:
    std::unique_ptr<EventDependency> dependency_;
    std::shared_ptr<AsyncBufferState> owner_;
    bool quarantine_on_destroy_ = false;
};

struct PendingOperationCreateResult {
    transport::TransportStatus status;
    std::shared_ptr<PendingOperation> operation;
};

#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
struct AsyncStateHostTestControl {
    std::mutex mutex;
    std::condition_variable cv;
    bool pause_destroy_after_snapshot = false;
    bool destroy_paused_after_snapshot = false;
    bool resume_destroy = false;
    std::uint64_t finalization_waiter_entries = 0;
    bool pause_after_terminal_state = false;
    bool terminal_state_published = false;
    bool resume_terminal_publication = false;
};
#endif

class AsyncBufferState {
public:
    explicit AsyncBufferState(
        std::shared_ptr<AsyncCompletionResources> resources,
        std::uint64_t owned_timeout_ms = 5000,
        std::uint64_t last_generation = 0
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
        , std::shared_ptr<AsyncStateHostTestControl> host_test_control = nullptr
#endif
        );
    ~AsyncBufferState();

    AsyncBufferState(const AsyncBufferState&) = delete;
    AsyncBufferState& operator=(const AsyncBufferState&) = delete;

    BufferOperationCoordinator& coordinator() noexcept;

    PendingOperationCreateResult publish(
        BufferOperationCoordinator::OperationLease lease,
        std::shared_ptr<runtime::NativeEventState> event,
        CompletionRecipe recipe,
        std::vector<std::optional<torch::Tensor>> retained_tensors,
        std::vector<EventDependency> predecessors);
    transport::TransportStatus finish_pending();
    transport::TransportStatus destroy();
    bool finalization_in_progress() const;
    std::optional<transport::TransportStatus> terminal_failure() const;
    void retire_or_quarantine(
        std::unique_ptr<EventDependency> dependency) noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace deep_ep::ascend::elastic
