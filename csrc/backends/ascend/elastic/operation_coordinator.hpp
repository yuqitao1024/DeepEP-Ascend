#pragma once

#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

namespace deep_ep::ascend::elastic {

enum class BufferOperationKind : std::uint8_t {
    kTopologyQuery,
    kBarrier,
    kDispatch,
    kCombine,
};

enum class CoordinatorState : std::uint8_t {
    kIdle,
    kReserved,
    kActive,
    kPoisoned,
    kDestroying,
    kDestroyed,
};

enum class LeaseStatus : std::uint8_t {
    kAdmitted,
    kBusy,
    kPoisoned,
    kDestroyed,
    kGenerationExhausted,
};

class BufferOperationCoordinator {
public:
    class OperationLease {
    public:
        OperationLease() = default;

        ~OperationLease() { abandon(); }

        OperationLease(const OperationLease&) = delete;
        OperationLease& operator=(const OperationLease&) = delete;

        OperationLease(OperationLease&& other) noexcept {
            move_from(std::move(other));
        }

        OperationLease& operator=(OperationLease&& other) noexcept {
            if (this != &other) {
                abandon();
                move_from(std::move(other));
            }
            return *this;
        }

        bool valid() const noexcept {
            return coordinator_ != nullptr && status_ == LeaseStatus::kAdmitted;
        }

        LeaseStatus status() const noexcept { return status_; }
        std::uint64_t generation() const noexcept { return generation_; }

        bool activate() noexcept {
            if (!valid() || active_)
                return false;
            if (!coordinator_->activate_operation(token_, &generation_)) {
                status_ = LeaseStatus::kGenerationExhausted;
                coordinator_ = nullptr;
                return false;
            }
            active_ = true;
            return true;
        }

        void complete() noexcept {
            if (!valid())
                return;
            coordinator_->complete_operation(token_, active_);
            coordinator_ = nullptr;
        }

    private:
        friend class BufferOperationCoordinator;

        OperationLease(BufferOperationCoordinator* coordinator,
                       LeaseStatus status, std::uint64_t token) noexcept
            : coordinator_(coordinator), status_(status), token_(token) {}

        void abandon() noexcept {
            if (coordinator_ == nullptr)
                return;
            coordinator_->abandon_operation(token_, active_);
            coordinator_ = nullptr;
        }

        void move_from(OperationLease&& other) noexcept {
            coordinator_ = std::exchange(other.coordinator_, nullptr);
            status_ = other.status_;
            token_ = other.token_;
            generation_ = other.generation_;
            active_ = other.active_;
        }

        BufferOperationCoordinator* coordinator_ = nullptr;
        LeaseStatus status_ = LeaseStatus::kBusy;
        std::uint64_t token_ = 0;
        std::uint64_t generation_ = 0;
        bool active_ = false;
    };

    class DestroyLease {
    public:
        DestroyLease() = default;

        ~DestroyLease() { fail(); }

        DestroyLease(const DestroyLease&) = delete;
        DestroyLease& operator=(const DestroyLease&) = delete;

        DestroyLease(DestroyLease&& other) noexcept {
            move_from(std::move(other));
        }

        DestroyLease& operator=(DestroyLease&& other) noexcept {
            if (this != &other) {
                fail();
                move_from(std::move(other));
            }
            return *this;
        }

        bool valid() const noexcept {
            return coordinator_ != nullptr && status_ == LeaseStatus::kAdmitted;
        }

        LeaseStatus status() const noexcept { return status_; }

        void complete() noexcept {
            if (coordinator_ == nullptr)
                return;
            coordinator_->complete_destroy(token_, true);
            coordinator_ = nullptr;
        }

        void fail() noexcept {
            if (coordinator_ == nullptr)
                return;
            coordinator_->complete_destroy(token_, false);
            coordinator_ = nullptr;
        }

    private:
        friend class BufferOperationCoordinator;

        DestroyLease(BufferOperationCoordinator* coordinator,
                     LeaseStatus status, std::uint64_t token) noexcept
            : coordinator_(coordinator), status_(status), token_(token) {}

        void move_from(DestroyLease&& other) noexcept {
            coordinator_ = std::exchange(other.coordinator_, nullptr);
            status_ = other.status_;
            token_ = other.token_;
        }

        BufferOperationCoordinator* coordinator_ = nullptr;
        LeaseStatus status_ = LeaseStatus::kBusy;
        std::uint64_t token_ = 0;
    };

    BufferOperationCoordinator() = default;

    explicit BufferOperationCoordinator(std::uint64_t last_generation) noexcept
        : last_generation_(last_generation) {}

    BufferOperationCoordinator(const BufferOperationCoordinator&) = delete;
    BufferOperationCoordinator& operator=(const BufferOperationCoordinator&) =
        delete;

    OperationLease reserve(BufferOperationKind) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == CoordinatorState::kIdle) {
            state_ = CoordinatorState::kReserved;
            return OperationLease(this, LeaseStatus::kAdmitted, next_token());
        }
        if (state_ == CoordinatorState::kActive) {
            poison_requested_ = true;
            return OperationLease(nullptr, LeaseStatus::kBusy, 0);
        }
        if (state_ == CoordinatorState::kReserved) {
            poison_requested_ = true;
            return OperationLease(nullptr, LeaseStatus::kBusy, 0);
        }
        if (state_ == CoordinatorState::kDestroying)
            return OperationLease(nullptr, LeaseStatus::kBusy, 0);
        if (state_ == CoordinatorState::kPoisoned)
            return OperationLease(nullptr, LeaseStatus::kPoisoned, 0);
        return OperationLease(nullptr, LeaseStatus::kDestroyed, 0);
    }

    DestroyLease reserve_destroy() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == CoordinatorState::kDestroyed)
            return DestroyLease(nullptr, LeaseStatus::kDestroyed, 0);
        if (state_ == CoordinatorState::kReserved ||
            state_ == CoordinatorState::kActive ||
            state_ == CoordinatorState::kDestroying)
            return DestroyLease(nullptr, LeaseStatus::kBusy, 0);
        state_ = CoordinatorState::kDestroying;
        poison_requested_ = false;
        return DestroyLease(this, LeaseStatus::kAdmitted, next_token());
    }

    CoordinatorState state() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    std::uint64_t last_generation() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_generation_;
    }

private:
    std::uint64_t next_token() noexcept {
        ++lease_token_;
        if (lease_token_ == 0)
            ++lease_token_;
        return lease_token_;
    }

    bool activate_operation(
        std::uint64_t token, std::uint64_t* generation) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != CoordinatorState::kReserved || token != lease_token_ ||
            generation == nullptr)
            return false;
        if (last_generation_ == std::numeric_limits<std::uint64_t>::max()) {
            state_ = CoordinatorState::kPoisoned;
            return false;
        }
        ++last_generation_;
        *generation = last_generation_;
        state_ = CoordinatorState::kActive;
        return true;
    }

    void complete_operation(std::uint64_t token, bool active) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto expected = active ? CoordinatorState::kActive :
            CoordinatorState::kReserved;
        if (state_ != expected || token != lease_token_)
            return;
        state_ = active && poison_requested_ ? CoordinatorState::kPoisoned :
            CoordinatorState::kIdle;
        poison_requested_ = false;
    }

    void abandon_operation(std::uint64_t token, bool active) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto expected = active ? CoordinatorState::kActive :
            CoordinatorState::kReserved;
        if (state_ != expected || token != lease_token_)
            return;
        state_ = active ? CoordinatorState::kPoisoned :
            CoordinatorState::kIdle;
        poison_requested_ = false;
    }

    void complete_destroy(std::uint64_t token, bool success) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != CoordinatorState::kDestroying || token != lease_token_)
            return;
        state_ = success ? CoordinatorState::kDestroyed :
            CoordinatorState::kPoisoned;
    }

    mutable std::mutex mutex_;
    CoordinatorState state_ = CoordinatorState::kIdle;
    std::uint64_t last_generation_ = 0;
    std::uint64_t lease_token_ = 0;
    bool poison_requested_ = false;
};

}  // namespace deep_ep::ascend::elastic
