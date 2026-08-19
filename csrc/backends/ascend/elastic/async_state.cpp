#ifndef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
#include <torch/extension.h>
#endif

#include "async_state.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

namespace deep_ep::ascend::elastic {

using transport::TransportStatus;

struct SharedAsyncContext {
    SharedAsyncContext(
        std::shared_ptr<AsyncCompletionResources> completion_resources,
        std::uint64_t last_generation
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
        , std::shared_ptr<AsyncStateHostTestControl> test_control
#endif
        )
        : coordinator(last_generation),
          resources(std::move(completion_resources))
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
          , host_test_control(std::move(test_control))
#endif
          {}

    void record_terminal_failure(const TransportStatus& status) {
        if (status.ok())
            return;
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (!first_terminal_failure.has_value())
            first_terminal_failure = status;
    }

    std::optional<TransportStatus> terminal_failure() const {
        std::lock_guard<std::mutex> lock(failure_mutex);
        return first_terminal_failure;
    }

    BufferOperationCoordinator coordinator;
    std::shared_ptr<AsyncCompletionResources> resources;
    mutable std::mutex failure_mutex;
    std::optional<TransportStatus> first_terminal_failure;
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
    std::shared_ptr<AsyncStateHostTestControl> host_test_control;
#endif
};

namespace {

static_assert(std::is_trivially_copyable_v<CompletionRecipe>);

const char* operation_name(BufferOperationKind kind) {
    switch (kind) {
        case BufferOperationKind::kTopologyQuery: return "topology_query";
        case BufferOperationKind::kBarrier: return "barrier";
        case BufferOperationKind::kDispatch: return "dispatch";
        case BufferOperationKind::kCombine: return "combine";
    }
    return "async_completion";
}

TransportStatus completion_failure(
    BufferOperationKind kind, const char* detail, int backend_code = 0) {
    return TransportStatus::runtime_failure(
        operation_name(kind), backend_code, detail);
}

TransportStatus finalizer_wait_timeout() {
    return TransportStatus::runtime_failure(
        "finish_pending", -1, "timed out waiting for the active finalizer");
}

TransportStatus destroy_wait_timeout() {
    return TransportStatus::runtime_failure(
        "destroy_async_state", -1,
        "timed out waiting for the active destroy attempt");
}

bool terminal(PendingState state) {
    return state == PendingState::kSucceeded || state == PendingState::kFailed;
}

TransportStatus invalid_publish(const char* message) {
    return TransportStatus::invalid("publish_pending", message);
}

}  // namespace

struct PendingOperation::Impl {
    Impl(
        std::shared_ptr<SharedAsyncContext> shared_context,
        BufferOperationCoordinator::OperationLease operation_lease,
        std::shared_ptr<runtime::NativeEventState> completion_event,
        CompletionRecipe completion_recipe,
        std::vector<std::optional<torch::Tensor>> tensors,
        std::vector<EventDependency> dependencies)
        : context(std::move(shared_context)), lease(std::move(operation_lease)),
          event(std::move(completion_event)), recipe(completion_recipe),
          retained_tensors(std::move(tensors)),
          predecessors(std::move(dependencies)) {}

    TransportStatus finalize(std::uint64_t timeout_ms) {
        auto status = event->finish(timeout_ms);
        if (!status.ok()) {
            lease.abandon();
            return status;
        }
        event_completed = true;

        transport::DeviceTransportDiagnostic diagnostic{};
        status = context->resources->read_diagnostic(
            recipe.kind, recipe.scratch_status_offset, &diagnostic);
        if (status.ok() &&
            diagnostic.abi_version != transport::kTransportCommandAbiVersion) {
            status = completion_failure(
                recipe.kind, "device diagnostic ABI mismatch");
        }
        if (status.ok() &&
            diagnostic.error != transport::DeviceTransportError::kNone) {
            status = completion_failure(
                recipe.kind, "device diagnostic reported failure",
                static_cast<int>(diagnostic.backend_status));
        }
        if (status.ok() && diagnostic.generation != recipe.generation) {
            status = completion_failure(
                recipe.kind, "device diagnostic generation mismatch");
        }

        std::uint64_t completion = 0;
        if (status.ok()) {
            status = context->resources->read_completion(
                recipe.kind, recipe.completion_offset, &completion);
        }
        if (status.ok() && completion != recipe.generation) {
            status = completion_failure(
                recipe.kind, "device completion generation mismatch");
        }

        const auto destroy_status = event->destroy();
        if (destroy_status.ok()) {
            release_retained_owners();
        } else if (status.ok()) {
            status = destroy_status;
        }

        if (status.ok())
            lease.complete();
        else
            lease.abandon();
        return status;
    }

    TransportStatus retry_teardown(std::uint64_t timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lock(mutex);
        if (retained_lifetime_safe)
            return TransportStatus::success();
        if (teardown_in_progress) {
            if (!cv.wait_until(lock, deadline, [&] {
                    return !teardown_in_progress;
                }))
                return destroy_wait_timeout();
            if (retained_lifetime_safe)
                return TransportStatus::success();
        }
        teardown_in_progress = true;
        const bool needs_finish = !event_completed;
        auto retained_event = event;
        lock.unlock();

        auto status = TransportStatus::success();
        if (needs_finish)
            status = retained_event->finish(timeout_ms);
        if (status.ok()) {
            lock.lock();
            event_completed = true;
            lock.unlock();
            status = retained_event->destroy();
        }

        lock.lock();
        if (status.ok())
            release_retained_owners();
        teardown_in_progress = false;
        lock.unlock();
        cv.notify_all();
        return status;
    }

    void release_retained_owners() {
        event.reset();
        retained_tensors.clear();
        predecessors.clear();
        retained_lifetime_safe = true;
    }

    std::shared_ptr<SharedAsyncContext> context;
    BufferOperationCoordinator::OperationLease lease;
    std::shared_ptr<runtime::NativeEventState> event;
    CompletionRecipe recipe;
    std::vector<std::optional<torch::Tensor>> retained_tensors;
    std::vector<EventDependency> predecessors;
    mutable std::mutex mutex;
    std::condition_variable cv;
    PendingState state = PendingState::kLaunched;
    TransportStatus terminal_status = TransportStatus::success();
    std::chrono::steady_clock::time_point finalization_deadline{};
    bool event_completed = false;
    bool retained_lifetime_safe = false;
    bool teardown_in_progress = false;
};

PendingOperation::PendingOperation(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

PendingOperation::~PendingOperation() = default;

TransportStatus PendingOperation::finish(std::uint64_t timeout_ms) {
    const auto requested_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (terminal(impl_->state))
        return impl_->terminal_status;
    if (impl_->state == PendingState::kFinalizing) {
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
        if (impl_->context->host_test_control != nullptr) {
            auto control = impl_->context->host_test_control;
            {
                std::lock_guard<std::mutex> control_lock(control->mutex);
                ++control->finalization_waiter_entries;
            }
            control->cv.notify_all();
        }
#endif
        const auto deadline = impl_->finalization_deadline;
        if (!impl_->cv.wait_until(lock, deadline, [&] {
                return terminal(impl_->state);
            }))
            return finalizer_wait_timeout();
        return impl_->terminal_status;
    }

    impl_->state = PendingState::kFinalizing;
    impl_->finalization_deadline = requested_deadline;
    lock.unlock();
    const auto status = impl_->finalize(timeout_ms);
    lock.lock();
    impl_->terminal_status = status;
    impl_->state = status.ok() ? PendingState::kSucceeded : PendingState::kFailed;
    lock.unlock();
    if (!status.ok())
        impl_->context->record_terminal_failure(status);
    impl_->cv.notify_all();
    return status;
}

PendingState PendingOperation::state() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->state;
}

std::uint64_t PendingOperation::generation() const noexcept {
    return impl_->recipe.generation;
}

TransportStatus PendingOperation::teardown(std::uint64_t timeout_ms) {
    return impl_->retry_teardown(timeout_ms);
}

bool PendingOperation::lifetime_safe() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->retained_lifetime_safe;
}

struct AsyncBufferState::Impl {
    enum class DestroyState : std::uint8_t {
        kAlive,
        kDestroying,
        kDestroyed,
    };

    Impl(
        std::shared_ptr<AsyncCompletionResources> resources,
        std::uint64_t timeout_ms, std::uint64_t last_generation
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
        , std::shared_ptr<AsyncStateHostTestControl> test_control
#endif
        )
        : context(std::make_shared<SharedAsyncContext>(
              std::move(resources), last_generation
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
              , test_control
#endif
              )),
          owned_timeout_ms(timeout_ms)
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
          , host_test_control(std::move(test_control))
#endif
          {}

    void finish_destroy_attempt() {
        std::lock_guard<std::mutex> lock(mutex);
        destroy_reserved = false;
        destroy_state = DestroyState::kAlive;
        cv.notify_all();
    }

    std::shared_ptr<SharedAsyncContext> context;
    const std::uint64_t owned_timeout_ms;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<PendingOperation> pending;
    DestroyState destroy_state = DestroyState::kAlive;
    TransportStatus destroy_result = TransportStatus::success();
    bool destroy_reserved = false;
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
    std::shared_ptr<AsyncStateHostTestControl> host_test_control;
#endif
};

AsyncBufferState::AsyncBufferState(
    std::shared_ptr<AsyncCompletionResources> resources,
    std::uint64_t owned_timeout_ms, std::uint64_t last_generation
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
    , std::shared_ptr<AsyncStateHostTestControl> host_test_control
#endif
    )
    : impl_(std::make_shared<Impl>(
          std::move(resources), owned_timeout_ms, last_generation
#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
          , std::move(host_test_control)
#endif
          )) {}

AsyncBufferState::~AsyncBufferState() {
    try {
        (void)destroy();
    } catch (...) {
    }
}

BufferOperationCoordinator& AsyncBufferState::coordinator() noexcept {
    return impl_->context->coordinator;
}

PendingOperationCreateResult AsyncBufferState::publish(
    BufferOperationCoordinator::OperationLease lease,
    std::shared_ptr<runtime::NativeEventState> event,
    CompletionRecipe recipe,
    std::vector<std::optional<torch::Tensor>> retained_tensors,
    std::vector<EventDependency> predecessors) {
    TransportStatus status = TransportStatus::success();
    if (impl_->context->resources == nullptr)
        status = invalid_publish("completion resources are unavailable");
    else if (!lease.valid() || !lease.active())
        status = invalid_publish("operation lease is not active");
    else if (!lease.belongs_to(impl_->context->coordinator))
        status = invalid_publish("operation lease belongs to another buffer");
    else if (event == nullptr)
        status = invalid_publish("completion event is unavailable");
    else if (recipe.generation == 0 ||
             recipe.generation != lease.generation())
        status = invalid_publish("completion generation does not match the lease");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (status.ok() && impl_->destroy_state != Impl::DestroyState::kAlive) {
        status = invalid_publish("async state is being destroyed");
    } else if (status.ok() && impl_->pending != nullptr &&
               (!terminal(impl_->pending->state()) ||
                !impl_->pending->lifetime_safe())) {
        status = invalid_publish("another operation is still pending");
    }
    if (!status.ok()) {
        lease.abandon();
        impl_->context->record_terminal_failure(status);
        if (event != nullptr && !impl_->destroy_reserved &&
            (impl_->pending == nullptr ||
             (terminal(impl_->pending->state()) &&
              impl_->pending->lifetime_safe()))) {
            auto failed_impl = std::make_shared<PendingOperation::Impl>(
                impl_->context, std::move(lease), std::move(event), recipe,
                std::move(retained_tensors), std::move(predecessors));
            failed_impl->terminal_status = status;
            failed_impl->state = PendingState::kFailed;
            impl_->pending = std::shared_ptr<PendingOperation>(
                new PendingOperation(std::move(failed_impl)));
        }
        return {status, nullptr};
    }

    auto operation = std::shared_ptr<PendingOperation>(new PendingOperation(
        std::make_shared<PendingOperation::Impl>(
            impl_->context, std::move(lease), std::move(event), recipe,
            std::move(retained_tensors), std::move(predecessors))));
    impl_->pending = operation;
    return {TransportStatus::success(), std::move(operation)};
}

TransportStatus AsyncBufferState::finish_pending() {
    std::shared_ptr<PendingOperation> pending;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        pending = impl_->pending;
    }
    return pending == nullptr ? TransportStatus::success() :
        pending->finish(impl_->owned_timeout_ms);
}

TransportStatus AsyncBufferState::destroy() {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(impl_->owned_timeout_ms);
    std::shared_ptr<PendingOperation> pending;
    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (impl_->destroy_state == Impl::DestroyState::kDestroyed)
            return impl_->destroy_result;
        if (impl_->destroy_state == Impl::DestroyState::kDestroying) {
            if (!impl_->cv.wait_until(lock, deadline, [&] {
                    return impl_->destroy_state != Impl::DestroyState::kDestroying;
                }))
                return destroy_wait_timeout();
            if (impl_->destroy_state == Impl::DestroyState::kDestroyed)
                return impl_->destroy_result;
        }
        impl_->destroy_state = Impl::DestroyState::kDestroying;
        pending = impl_->pending;
    }

#ifdef DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR
    if (impl_->host_test_control != nullptr) {
        auto control = impl_->host_test_control;
        std::unique_lock<std::mutex> lock(control->mutex);
        if (control->pause_destroy_after_snapshot) {
            control->destroy_paused_after_snapshot = true;
            control->cv.notify_all();
            (void)control->cv.wait_for(
                lock, std::chrono::seconds(5), [&] {
                    return control->resume_destroy;
                });
        }
    }
#endif

    auto pending_status = TransportStatus::success();
    BufferOperationCoordinator::DestroyLease teardown;
    while (true) {
        if (pending != nullptr) {
            const bool was_terminal = terminal(pending->state());
            pending_status = pending->finish(impl_->owned_timeout_ms);
            if (!pending->lifetime_safe()) {
                if (!was_terminal) {
                    impl_->finish_destroy_attempt();
                    return pending_status;
                }
                const auto teardown_status = pending->teardown(
                    impl_->owned_timeout_ms);
                if (!teardown_status.ok()) {
                    impl_->finish_destroy_attempt();
                    return teardown_status;
                }
            }
        }

        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (impl_->pending != pending) {
            pending = impl_->pending;
            continue;
        }
        teardown = impl_->context->coordinator.reserve_destroy();
        if (!teardown.valid()) {
            lock.unlock();
            impl_->finish_destroy_attempt();
            return TransportStatus::invalid(
                "destroy_async_state", "operation coordinator is busy");
        }
        impl_->destroy_reserved = true;
        break;
    }

    const auto resource_status = impl_->context->resources == nullptr ?
        TransportStatus::invalid(
            "destroy_async_state", "completion resources are unavailable") :
        impl_->context->resources->destroy();
    if (!resource_status.ok()) {
        teardown.fail();
        impl_->finish_destroy_attempt();
        return resource_status;
    }
    teardown.complete();

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->pending.reset();
        impl_->destroy_result = pending_status;
        impl_->destroy_state = Impl::DestroyState::kDestroyed;
    }
    impl_->cv.notify_all();
    return pending_status;
}

std::optional<TransportStatus> AsyncBufferState::terminal_failure() const {
    return impl_->context->terminal_failure();
}

}  // namespace deep_ep::ascend::elastic
