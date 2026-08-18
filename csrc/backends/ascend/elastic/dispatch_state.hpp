#pragma once

#include <cstdint>
#include <limits>

#include "layout.hpp"

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE \
    __SIMT_DEVICE_FUNCTIONS_DECL__
#else
#define DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE
#endif

namespace deep_ep::ascend::elastic {

inline constexpr std::uint32_t kDispatchHandleDescriptorAbiVersion = 3;

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
    std::uint64_t generation = 0;
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
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags) {
    DispatchHandleDescriptor descriptor{};
    descriptor.family = family;
    descriptor.topology = topology;
    descriptor.generation = generation;
    descriptor.num_tokens = num_tokens;
    descriptor.hidden = hidden;
    descriptor.num_experts = num_experts;
    descriptor.num_topk = num_topk;
    descriptor.expert_alignment = expert_alignment;
    descriptor.num_max_tokens_per_rank = num_max_tokens_per_rank;
    descriptor.mode_flags = mode_flags;
    return descriptor;
}

constexpr std::uint64_t mix_dispatch_handle_attestation(
    std::uint64_t state, std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return state ^ (value + 0x9e3779b97f4a7c15ULL +
                    (state << 6U) + (state >> 2U));
}

constexpr CoreModeFlags dispatch_handle_identity_mode_flags(
    CoreModeFlags flags) noexcept {
    return flags & ~mode_bit(CoreMode::kZeroPadding);
}

// Bind public descriptor geometry to its buffer without retaining handles.
constexpr std::uint64_t attest_dispatch_handle_family(
    std::uint64_t buffer_family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank,
    CoreModeFlags mode_flags) noexcept {
    std::uint64_t state = mix_dispatch_handle_attestation(
        buffer_family, kDispatchHandleDescriptorAbiVersion);
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.world_rank));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.world_size));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.scale_up_rank));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.scale_up_size));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.scale_out_rank));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.scale_out_size));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(topology.kind));
    state = mix_dispatch_handle_attestation(state, topology.epoch);
    state = mix_dispatch_handle_attestation(state, generation);
    state = mix_dispatch_handle_attestation(state, num_tokens);
    state = mix_dispatch_handle_attestation(state, hidden);
    state = mix_dispatch_handle_attestation(state, num_experts);
    state = mix_dispatch_handle_attestation(state, num_topk);
    state = mix_dispatch_handle_attestation(state, expert_alignment);
    state = mix_dispatch_handle_attestation(
        state, num_max_tokens_per_rank);
    return mix_dispatch_handle_attestation(
        state, dispatch_handle_identity_mode_flags(mode_flags));
}

inline DispatchHandleDescriptor make_attested_dispatch_handle_descriptor(
    std::uint64_t buffer_family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags) {
    return make_dispatch_handle_descriptor(
        attest_dispatch_handle_family(
            buffer_family, topology, generation, num_tokens, hidden, num_experts,
            num_topk, expert_alignment, num_max_tokens_per_rank, mode_flags),
        topology, generation, num_tokens, hidden, num_experts, num_topk,
        expert_alignment, num_max_tokens_per_rank, mode_flags);
}

constexpr bool same_topology(
    const CoreTopology& lhs, const CoreTopology& rhs) {
    return lhs.world_rank == rhs.world_rank &&
           lhs.world_size == rhs.world_size &&
           lhs.scale_up_rank == rhs.scale_up_rank &&
           lhs.scale_up_size == rhs.scale_up_size &&
           lhs.scale_out_rank == rhs.scale_out_rank &&
           lhs.scale_out_size == rhs.scale_out_size &&
           lhs.kind == rhs.kind && lhs.epoch == rhs.epoch;
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr bool
is_dispatch_expert_local(
    std::int64_t expert, std::uint64_t first_local_expert,
    std::uint64_t num_local_experts) noexcept {
    if (expert < 0)
        return false;
    const auto unsigned_expert = static_cast<std::uint64_t>(expert);
    return unsigned_expert >= first_local_expert &&
           unsigned_expert - first_local_expert < num_local_experts;
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr bool
is_dispatch_local_index(
    std::int32_t local_index, std::uint64_t extent) noexcept {
    return local_index >= 0 &&
           static_cast<std::uint64_t>(local_index) < extent;
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr std::int32_t
encode_dispatch_source_index(
    std::uint64_t source_rank, std::uint64_t extent,
    std::uint64_t local_index) noexcept {
    return static_cast<std::int32_t>(source_rank * extent + local_index);
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr int
decode_dispatch_source_rank(
    std::int32_t encoded_index, std::uint64_t extent) noexcept {
    return encoded_index < 0 || extent == 0 ? -1 :
        static_cast<int>(
            static_cast<std::uint64_t>(encoded_index) / extent);
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr std::int32_t
decode_dispatch_local_index(
    std::int32_t encoded_index, std::uint64_t extent) noexcept {
    return encoded_index < 0 || extent == 0 ? -1 :
        static_cast<std::int32_t>(
            static_cast<std::uint64_t>(encoded_index) % extent);
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr std::int64_t
localize_dispatch_expert(
    std::int64_t expert, std::uint64_t first_local_expert,
    std::uint64_t num_local_experts) noexcept {
    return is_dispatch_expert_local(
               expert, first_local_expert, num_local_experts) ?
        expert - static_cast<std::int64_t>(first_local_expert) : -1;
}

inline DispatchHandleStatus validate_dispatch_handle(
    const DispatchHandleDescriptor& expected,
    const DispatchHandleDescriptor& actual) {
    if (expected.abi_version != kDispatchHandleDescriptorAbiVersion ||
        expected.struct_size != sizeof(DispatchHandleDescriptor) ||
        actual.abi_version != kDispatchHandleDescriptorAbiVersion ||
        actual.struct_size != sizeof(DispatchHandleDescriptor) ||
        expected.generation == 0 || actual.generation == 0)
        return {DispatchHandleStatusCode::kInvalidDescriptor,
                "invalid dispatch handle descriptor"};
    if (expected.family != actual.family ||
        !same_topology(expected.topology, actual.topology) ||
        expected.generation != actual.generation ||
        expected.num_tokens != actual.num_tokens ||
        expected.hidden != actual.hidden ||
        expected.num_experts != actual.num_experts ||
        expected.num_topk != actual.num_topk ||
        expected.expert_alignment != actual.expert_alignment ||
        expected.num_max_tokens_per_rank != actual.num_max_tokens_per_rank ||
        dispatch_handle_identity_mode_flags(expected.mode_flags) !=
            dispatch_handle_identity_mode_flags(actual.mode_flags))
        return {DispatchHandleStatusCode::kMismatch,
                "dispatch handle does not match the current call"};
    return {};
}

}  // namespace deep_ep::ascend::elastic

#undef DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE
