#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#include "csrc/backends/ascend/elastic/operation_coordinator.hpp"

#define CHECK(expression) do { if (!(expression)) return __LINE__; } while (0)

using deep_ep::ascend::elastic::BufferOperationCoordinator;
using deep_ep::ascend::elastic::CoordinatorState;
using deep_ep::ascend::elastic::LeaseStatus;
using deep_ep::ascend::elastic::BufferOperationKind;

struct Rendezvous {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    bool admitted = false;
    bool release = false;
};

int check_two_buffers_can_be_active() {
    BufferOperationCoordinator first;
    BufferOperationCoordinator second;
    Rendezvous rendezvous;
    bool thread_ok = false;

    std::thread worker([&] {
        auto lease = first.reserve(BufferOperationKind::kDispatch);
        const bool admitted = lease.valid() && lease.activate();
        {
            std::lock_guard<std::mutex> lock(rendezvous.mutex);
            rendezvous.ready = true;
            rendezvous.admitted = admitted;
        }
        rendezvous.cv.notify_one();
        if (!admitted)
            return;
        {
            std::unique_lock<std::mutex> lock(rendezvous.mutex);
            rendezvous.cv.wait(lock, [&] { return rendezvous.release; });
        }
        lease.complete();
        thread_ok = true;
    });

    bool worker_admitted = false;
    {
        std::unique_lock<std::mutex> lock(rendezvous.mutex);
        rendezvous.cv.wait(lock, [&] { return rendezvous.ready; });
        worker_admitted = rendezvous.admitted;
    }
    if (!worker_admitted) worker.join();
    CHECK(worker_admitted);
    auto lease = second.reserve(BufferOperationKind::kCombine);
    CHECK(lease.valid());
    CHECK(lease.activate());
    CHECK(first.state() == CoordinatorState::kActive);
    CHECK(second.state() == CoordinatorState::kActive);
    lease.complete();
    {
        std::lock_guard<std::mutex> lock(rendezvous.mutex);
        rendezvous.release = true;
    }
    rendezvous.cv.notify_one();
    worker.join();
    CHECK(thread_ok);
    CHECK(first.state() == CoordinatorState::kIdle);
    CHECK(second.state() == CoordinatorState::kIdle);
    return 0;
}

int check_same_buffer_busy_and_deferred_poison() {
    BufferOperationCoordinator coordinator;

    {
        auto reserved = coordinator.reserve(BufferOperationKind::kBarrier);
        CHECK(reserved.valid());
        auto busy = coordinator.reserve(BufferOperationKind::kDispatch);
        CHECK(!busy.valid());
        CHECK(busy.status() == LeaseStatus::kBusy);
        CHECK(coordinator.state() == CoordinatorState::kReserved);
        reserved.complete();
    }
    CHECK(coordinator.state() == CoordinatorState::kIdle);
    CHECK(coordinator.last_generation() == 0);

    BufferOperationCoordinator deferred;
    auto reserved = deferred.reserve(BufferOperationKind::kDispatch);
    CHECK(reserved.valid());
    auto reserved_busy = deferred.reserve(BufferOperationKind::kCombine);
    CHECK(reserved_busy.status() == LeaseStatus::kBusy);
    CHECK(reserved.activate());
    reserved.complete();
    CHECK(deferred.state() == CoordinatorState::kPoisoned);

    Rendezvous rendezvous;
    bool thread_ok = false;
    std::thread worker([&] {
        auto active = coordinator.reserve(BufferOperationKind::kBarrier);
        const bool admitted = active.valid() && active.activate() &&
            active.generation() == 1;
        {
            std::lock_guard<std::mutex> lock(rendezvous.mutex);
            rendezvous.ready = true;
            rendezvous.admitted = admitted;
        }
        rendezvous.cv.notify_one();
        if (!admitted)
            return;
        {
            std::unique_lock<std::mutex> lock(rendezvous.mutex);
            rendezvous.cv.wait(lock, [&] { return rendezvous.release; });
        }
        active.complete();
        thread_ok = true;
    });
    bool worker_admitted = false;
    {
        std::unique_lock<std::mutex> lock(rendezvous.mutex);
        rendezvous.cv.wait(lock, [&] { return rendezvous.ready; });
        worker_admitted = rendezvous.admitted;
    }
    if (!worker_admitted) worker.join();
    CHECK(worker_admitted);
    auto busy = coordinator.reserve(BufferOperationKind::kCombine);
    CHECK(!busy.valid());
    CHECK(busy.status() == LeaseStatus::kBusy);
    CHECK(coordinator.state() == CoordinatorState::kActive);
    {
        std::lock_guard<std::mutex> lock(rendezvous.mutex);
        rendezvous.release = true;
    }
    rendezvous.cv.notify_one();
    worker.join();
    CHECK(thread_ok);
    CHECK(coordinator.state() == CoordinatorState::kPoisoned);
    auto rejected = coordinator.reserve(BufferOperationKind::kTopologyQuery);
    CHECK(rejected.status() == LeaseStatus::kPoisoned);
    return 0;
}

int check_post_activation_failure_is_buffer_local() {
    BufferOperationCoordinator failed;
    BufferOperationCoordinator healthy;
    {
        auto lease = failed.reserve(BufferOperationKind::kDispatch);
        CHECK(lease.valid());
        CHECK(lease.activate());
    }
    CHECK(failed.state() == CoordinatorState::kPoisoned);
    auto lease = healthy.reserve(BufferOperationKind::kDispatch);
    CHECK(lease.valid());
    CHECK(lease.activate());
    lease.complete();
    CHECK(healthy.state() == CoordinatorState::kIdle);
    return 0;
}

int check_destroy_race_and_retry() {
    BufferOperationCoordinator coordinator;
    Rendezvous rendezvous;
    bool operation_ok = false;

    std::thread worker([&] {
        auto lease = coordinator.reserve(BufferOperationKind::kTopologyQuery);
        const bool admitted = lease.valid() && lease.activate();
        {
            std::lock_guard<std::mutex> lock(rendezvous.mutex);
            rendezvous.ready = true;
            rendezvous.admitted = admitted;
        }
        rendezvous.cv.notify_one();
        if (!admitted)
            return;
        {
            std::unique_lock<std::mutex> lock(rendezvous.mutex);
            rendezvous.cv.wait(lock, [&] { return rendezvous.release; });
        }
        lease.complete();
        operation_ok = true;
    });
    bool worker_admitted = false;
    {
        std::unique_lock<std::mutex> lock(rendezvous.mutex);
        rendezvous.cv.wait(lock, [&] { return rendezvous.ready; });
        worker_admitted = rendezvous.admitted;
    }
    if (!worker_admitted) worker.join();
    CHECK(worker_admitted);
    auto busy_destroy = coordinator.reserve_destroy();
    CHECK(!busy_destroy.valid());
    CHECK(busy_destroy.status() == LeaseStatus::kBusy);
    CHECK(coordinator.state() == CoordinatorState::kActive);
    {
        std::lock_guard<std::mutex> lock(rendezvous.mutex);
        rendezvous.release = true;
    }
    rendezvous.cv.notify_one();
    worker.join();
    CHECK(operation_ok);
    CHECK(coordinator.state() == CoordinatorState::kIdle);

    {
        auto destroy = coordinator.reserve_destroy();
        CHECK(destroy.valid());
        CHECK(coordinator.state() == CoordinatorState::kDestroying);
        auto busy = coordinator.reserve(BufferOperationKind::kBarrier);
        CHECK(busy.status() == LeaseStatus::kBusy);
        destroy.fail();
    }
    CHECK(coordinator.state() == CoordinatorState::kPoisoned);
    {
        auto retry = coordinator.reserve_destroy();
        CHECK(retry.valid());
        retry.complete();
    }
    CHECK(coordinator.state() == CoordinatorState::kDestroyed);
    auto repeated = coordinator.reserve_destroy();
    CHECK(!repeated.valid());
    CHECK(repeated.status() == LeaseStatus::kDestroyed);
    auto rejected = coordinator.reserve(BufferOperationKind::kDispatch);
    CHECK(rejected.status() == LeaseStatus::kDestroyed);
    return 0;
}

int check_generation_overflow() {
    BufferOperationCoordinator coordinator(
        std::numeric_limits<std::uint64_t>::max() - 1);
    {
        auto last = coordinator.reserve(BufferOperationKind::kDispatch);
        CHECK(last.valid());
        CHECK(last.activate());
        CHECK(last.generation() == std::numeric_limits<std::uint64_t>::max());
        last.complete();
    }
    auto overflow = coordinator.reserve(BufferOperationKind::kCombine);
    CHECK(overflow.valid());
    CHECK(!overflow.activate());
    CHECK(overflow.status() == LeaseStatus::kGenerationExhausted);
    CHECK(coordinator.state() == CoordinatorState::kPoisoned);
    CHECK(coordinator.last_generation() ==
          std::numeric_limits<std::uint64_t>::max());
    return 0;
}

int check_active_lease_move_retires_exactly_once() {
    BufferOperationCoordinator coordinator;
    auto lease = coordinator.reserve(BufferOperationKind::kDispatch);
    CHECK(lease.valid());
    CHECK(lease.activate());
    CHECK(coordinator.completed_operation_count() == 0);
    CHECK(coordinator.abandoned_operation_count() == 0);

    auto moved = std::move(lease);
    CHECK(!lease.valid());
    CHECK(moved.valid());
    CHECK(moved.active());
    CHECK(coordinator.state() == CoordinatorState::kActive);
    CHECK(coordinator.completed_operation_count() == 0);
    CHECK(coordinator.abandoned_operation_count() == 0);

    moved.complete();
    moved.complete();
    CHECK(coordinator.state() == CoordinatorState::kIdle);
    CHECK(coordinator.completed_operation_count() == 1);
    CHECK(coordinator.abandoned_operation_count() == 0);
    return 0;
}

int main() {
    if (const int status = check_two_buffers_can_be_active()) return status;
    if (const int status = check_same_buffer_busy_and_deferred_poison())
        return status;
    if (const int status = check_post_activation_failure_is_buffer_local())
        return status;
    if (const int status = check_destroy_race_and_retry()) return status;
    if (const int status = check_generation_overflow()) return status;
    return check_active_lease_move_retires_exactly_once();
}
