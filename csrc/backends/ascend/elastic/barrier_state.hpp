#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kAscend950SystemCounterHz = 1000000000ULL;

constexpr bool timeout_cycles_from_seconds(
    int seconds, std::uint64_t* timeout_cycles) {
    if (seconds <= 0 || timeout_cycles == nullptr)
        return false;
    const auto value = static_cast<std::uint64_t>(seconds);
    if (value > std::numeric_limits<std::uint64_t>::max() /
                    kAscend950SystemCounterHz)
        return false;
    *timeout_cycles = value * kAscend950SystemCounterHz;
    return true;
}

class BarrierAttempt;

class BarrierSequence {
public:
    bool poisoned() const noexcept { return poisoned_; }

private:
    friend class BarrierAttempt;

    bool begin(std::uint64_t* generation) noexcept {
        if (generation == nullptr || poisoned_ || in_progress_)
            return false;
        ++generation_;
        if (generation_ == 0)
            ++generation_;
        *generation = generation_;
        in_progress_ = true;
        return true;
    }

    void complete() noexcept { in_progress_ = false; }

    void fail() noexcept {
        in_progress_ = false;
        poisoned_ = true;
    }

    std::uint64_t generation_ = 0;
    bool in_progress_ = false;
    bool poisoned_ = false;
};

class BarrierAttempt {
public:
    explicit BarrierAttempt(BarrierSequence& sequence) noexcept
        : sequence_(&sequence) {
        if (!sequence_->begin(&generation_))
            sequence_ = nullptr;
    }

    ~BarrierAttempt() {
        if (sequence_ != nullptr)
            sequence_->fail();
    }

    BarrierAttempt(const BarrierAttempt&) = delete;
    BarrierAttempt& operator=(const BarrierAttempt&) = delete;

    bool valid() const noexcept { return sequence_ != nullptr; }
    std::uint64_t generation() const noexcept { return generation_; }

    void complete() noexcept {
        if (sequence_ == nullptr)
            return;
        sequence_->complete();
        sequence_ = nullptr;
    }

private:
    BarrierSequence* sequence_ = nullptr;
    std::uint64_t generation_ = 0;
};

}  // namespace deep_ep::ascend::elastic
