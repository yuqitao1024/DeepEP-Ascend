#pragma once

#include <cstdint>
#include <limits>

#include "layout.hpp"

namespace deep_ep::ascend::elastic {

inline constexpr std::uint32_t kDispatchHandleDescriptorAbiVersion = 1;

class DispatchAttempt;

class DispatchSequence {
public:
    DispatchSequence() = default;

    explicit DispatchSequence(std::uint64_t last_generation) noexcept
        : generation_(last_generation) {}

    bool poisoned() const noexcept { return poisoned_; }

private:
    friend class DispatchAttempt;

    bool begin(std::uint64_t* generation) noexcept {
        if (generation == nullptr || poisoned_ || in_progress_)
            return false;
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            poisoned_ = true;
            return false;
        }
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

class DispatchAttempt {
public:
    explicit DispatchAttempt(DispatchSequence& sequence) noexcept
        : sequence_(&sequence) {
        if (!sequence_->begin(&generation_))
            sequence_ = nullptr;
    }

    ~DispatchAttempt() {
        if (sequence_ != nullptr)
            sequence_->fail();
    }

    DispatchAttempt(const DispatchAttempt&) = delete;
    DispatchAttempt& operator=(const DispatchAttempt&) = delete;

    bool valid() const noexcept { return sequence_ != nullptr; }
    std::uint64_t generation() const noexcept { return generation_; }

    void complete() noexcept {
        if (sequence_ == nullptr)
            return;
        sequence_->complete();
        sequence_ = nullptr;
    }

private:
    DispatchSequence* sequence_ = nullptr;
    std::uint64_t generation_ = 0;
};

struct DispatchHandleDescriptor {
    std::uint32_t abi_version = kDispatchHandleDescriptorAbiVersion;
    std::uint32_t struct_size = sizeof(DispatchHandleDescriptor);
    std::uint64_t family = 0;
    CoreTopology topology{};
    std::uint64_t num_tokens = 0;
    std::uint64_t hidden = 0;
    std::uint64_t num_experts = 0;
    std::uint64_t num_topk = 0;
    std::uint64_t expert_alignment = 0;
    std::uint64_t num_max_tokens_per_rank = 0;
    CoreModeFlags mode_flags = 0;
};

enum class DispatchHandleStatusCode : std::uint8_t {
    kSuccess,
    kInvalidDescriptor,
    kMismatch,
};

struct DispatchHandleStatus {
    DispatchHandleStatusCode code = DispatchHandleStatusCode::kSuccess;
    const char* message = "";

    constexpr bool ok() const {
        return code == DispatchHandleStatusCode::kSuccess;
    }
};

inline DispatchHandleDescriptor make_dispatch_handle_descriptor(
    std::uint64_t family, const CoreTopology& topology,
    std::uint64_t num_tokens, std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags) {
    DispatchHandleDescriptor descriptor{};
    descriptor.family = family;
    descriptor.topology = topology;
    descriptor.num_tokens = num_tokens;
    descriptor.hidden = hidden;
    descriptor.num_experts = num_experts;
    descriptor.num_topk = num_topk;
    descriptor.expert_alignment = expert_alignment;
    descriptor.num_max_tokens_per_rank = num_max_tokens_per_rank;
    descriptor.mode_flags = mode_flags;
    return descriptor;
}

constexpr bool same_topology(
    const CoreTopology& lhs, const CoreTopology& rhs) {
    return lhs.world_rank == rhs.world_rank &&
           lhs.world_size == rhs.world_size &&
           lhs.scale_up_rank == rhs.scale_up_rank &&
           lhs.scale_up_size == rhs.scale_up_size &&
           lhs.scale_out_rank == rhs.scale_out_rank &&
           lhs.scale_out_size == rhs.scale_out_size;
}

inline DispatchHandleStatus validate_dispatch_handle(
    const DispatchHandleDescriptor& expected,
    const DispatchHandleDescriptor& actual) {
    if (expected.abi_version != kDispatchHandleDescriptorAbiVersion ||
        expected.struct_size != sizeof(DispatchHandleDescriptor) ||
        actual.abi_version != kDispatchHandleDescriptorAbiVersion ||
        actual.struct_size != sizeof(DispatchHandleDescriptor))
        return {DispatchHandleStatusCode::kInvalidDescriptor,
                "invalid dispatch handle descriptor"};
    if (expected.family != actual.family ||
        !same_topology(expected.topology, actual.topology) ||
        expected.num_tokens != actual.num_tokens ||
        expected.hidden != actual.hidden ||
        expected.num_experts != actual.num_experts ||
        expected.num_topk != actual.num_topk ||
        expected.expert_alignment != actual.expert_alignment ||
        expected.num_max_tokens_per_rank != actual.num_max_tokens_per_rank ||
        expected.mode_flags != actual.mode_flags)
        return {DispatchHandleStatusCode::kMismatch,
                "dispatch handle does not match the current call"};
    return {};
}

}  // namespace deep_ep::ascend::elastic
