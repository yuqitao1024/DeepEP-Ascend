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

constexpr std::uint64_t scale_factor_byte_offset(
    std::uint64_t token, std::uint64_t pack,
    std::uint64_t token_stride, std::uint64_t pack_stride,
    std::uint64_t pack_bytes) {
    return (token * token_stride + pack * pack_stride) * pack_bytes;
}

inline constexpr std::uint32_t kDispatchHandleDescriptorAbiVersion = 4;
inline constexpr std::uint32_t kHybridRouteLayoutVersion = 1;
inline constexpr std::uint64_t kInvalidHybridRouteSlot =
    std::numeric_limits<std::uint64_t>::max();

enum class DispatchRoutingMode : std::uint32_t { kDirect = 0, kHybrid = 1 };

enum class HybridRouteStage : std::uint32_t {
    kIngressComplete = 0,
    kForwardComplete = 1,
};

enum class DispatchProtocolError : std::uint32_t {
    kNone = 0,
    kInvalidTopk,
    kInvalidCachedSlot,
    kInvalidCachedPrefix,
    kInvalidCachedMetadata,
    kCapacityOverflow,
    kInvalidControl,
    kInvalidLayout,
};

enum class DispatchProtocolStage : std::uint32_t {
    kProducer = 1,
    kForward = 2,
    kPrepareEpilogue = 3,
    kEpilogue = 4,
    kRouteRecord = 5,
};

struct DispatchProtocolFailure {
    int world_rank = -1;
    std::uint64_t generation = 0;
    std::uint64_t scratch_status = 0;
    std::uint32_t backend_status = 0;
};

struct DecodedDispatchProtocolFailure {
    int world_rank = -1;
    DispatchProtocolError error = DispatchProtocolError::kNone;
    bool valid = false;
};

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr DispatchProtocolError
validate_hybrid_route_control(
    std::uint64_t expected_generation, std::uint64_t observed_generation,
    std::uint64_t count, std::uint64_t capacity) noexcept {
    if (observed_generation != expected_generation)
        return DispatchProtocolError::kInvalidControl;
    return count <= capacity ? DispatchProtocolError::kNone :
        DispatchProtocolError::kCapacityOverflow;
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr DispatchProtocolFailure
make_dispatch_protocol_failure(
    int world_rank, DispatchProtocolStage stage, std::uint64_t generation,
    DispatchProtocolError error) noexcept {
    return {
        world_rank,
        generation,
        (static_cast<std::uint64_t>(world_rank + 1) << 32U) |
            static_cast<std::uint32_t>(error),
        (static_cast<std::uint32_t>(stage) << 16U) |
            static_cast<std::uint32_t>(error),
    };
}

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr
DecodedDispatchProtocolFailure decode_dispatch_protocol_scratch(
    std::uint64_t scratch_status) noexcept {
    const std::uint64_t encoded_rank = scratch_status >> 32U;
    const std::uint32_t encoded_error =
        static_cast<std::uint32_t>(scratch_status);
    if (encoded_rank == 0 ||
        encoded_rank >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1U ||
        encoded_error <=
            static_cast<std::uint32_t>(DispatchProtocolError::kNone) ||
        encoded_error >
            static_cast<std::uint32_t>(DispatchProtocolError::kInvalidLayout))
        return {};
    return {
        static_cast<int>(encoded_rank - 1U),
        static_cast<DispatchProtocolError>(encoded_error), true};
}

using HybridRouteStageFlags = std::uint32_t;

constexpr HybridRouteStageFlags hybrid_stage_bit(HybridRouteStage stage) {
    return HybridRouteStageFlags{1} << static_cast<std::uint32_t>(stage);
}

inline constexpr HybridRouteStageFlags kHybridRouteCompleteStageFlags =
    hybrid_stage_bit(HybridRouteStage::kIngressComplete) |
    hybrid_stage_bit(HybridRouteStage::kForwardComplete);

struct HybridRouteRecord {
    std::int32_t origin_world_rank;
    std::int32_t destination_world_rank;
    std::int32_t ingress_world_rank;
    std::int32_t destination_local_expert;
    std::uint64_t origin_source_row;
    std::uint64_t ingress_slot;
    std::uint64_t forwarded_slot;
    std::uint64_t generation;
    std::uint64_t topology_epoch;
    HybridRouteStageFlags stage_flags;
    std::uint32_t reserved;
};

static_assert(sizeof(HybridRouteRecord) == kHybridRouteRecordBytes);

struct HybridRouteTableView {
    const HybridRouteRecord* records;
    std::uint64_t count;
};

struct HybridRouteBindingView {
    const std::int32_t* source_metadata = nullptr;
    const std::int64_t* received_topk_indices = nullptr;
    std::uint64_t row_count = 0;
    std::uint64_t num_topk = 0;
    std::uint64_t shard_capacity = 0;
};

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
    DispatchRoutingMode routing_mode = DispatchRoutingMode::kDirect;
    std::uint32_t route_layout_version = 0;
    std::uint64_t route_record_count = 0;
    std::uint64_t route_record_stride = 0;
    std::uint64_t dispatch_generation = 0;
    HybridRouteStageFlags route_stage_flags = 0;
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
    descriptor.dispatch_generation = generation;
    return descriptor;
}

inline DispatchHandleDescriptor make_dispatch_handle_descriptor(
    std::uint64_t family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags,
    DispatchRoutingMode routing_mode, std::uint32_t route_layout_version,
    std::uint64_t route_record_count, std::uint64_t route_record_stride,
    std::uint64_t dispatch_generation,
    HybridRouteStageFlags route_stage_flags) {
    DispatchHandleDescriptor descriptor = make_dispatch_handle_descriptor(
        family, topology, generation, num_tokens, hidden, num_experts, num_topk,
        expert_alignment, num_max_tokens_per_rank, mode_flags);
    descriptor.routing_mode = routing_mode;
    descriptor.route_layout_version = route_layout_version;
    descriptor.route_record_count = route_record_count;
    descriptor.route_record_stride = route_record_stride;
    descriptor.dispatch_generation = dispatch_generation;
    descriptor.route_stage_flags = route_stage_flags;
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
    CoreModeFlags mode_flags, DispatchRoutingMode routing_mode,
    std::uint32_t route_layout_version,
    std::uint64_t route_record_count, std::uint64_t route_record_stride,
    std::uint64_t dispatch_generation,
    HybridRouteStageFlags route_stage_flags) noexcept {
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
    state = mix_dispatch_handle_attestation(
        state, dispatch_handle_identity_mode_flags(mode_flags));
    state = mix_dispatch_handle_attestation(
        state, static_cast<std::uint32_t>(routing_mode));
    state = mix_dispatch_handle_attestation(state, route_layout_version);
    state = mix_dispatch_handle_attestation(state, route_record_count);
    state = mix_dispatch_handle_attestation(state, route_record_stride);
    state = mix_dispatch_handle_attestation(state, dispatch_generation);
    return mix_dispatch_handle_attestation(state, route_stage_flags);
}

inline std::uint64_t attest_hybrid_route_table(
    std::uint64_t state, HybridRouteTableView route_table) noexcept {
    state = mix_dispatch_handle_attestation(state, route_table.count);
    if (route_table.count != 0 && route_table.records == nullptr)
        return mix_dispatch_handle_attestation(
            state, kInvalidHybridRouteSlot);

    for (std::uint64_t index = 0; index < route_table.count; ++index) {
        const auto& record = route_table.records[index];
        state = mix_dispatch_handle_attestation(state, index);
        state = mix_dispatch_handle_attestation(
            state, static_cast<std::uint32_t>(record.origin_world_rank));
        state = mix_dispatch_handle_attestation(
            state, static_cast<std::uint32_t>(record.destination_world_rank));
        state = mix_dispatch_handle_attestation(
            state, static_cast<std::uint32_t>(record.ingress_world_rank));
        state = mix_dispatch_handle_attestation(
            state, static_cast<std::uint32_t>(
                       record.destination_local_expert));
        state = mix_dispatch_handle_attestation(
            state, record.origin_source_row);
        state = mix_dispatch_handle_attestation(state, record.ingress_slot);
        state = mix_dispatch_handle_attestation(state, record.forwarded_slot);
        state = mix_dispatch_handle_attestation(state, record.generation);
        state = mix_dispatch_handle_attestation(state, record.topology_epoch);
        state = mix_dispatch_handle_attestation(state, record.stage_flags);
        state = mix_dispatch_handle_attestation(state, record.reserved);
    }
    return state;
}

inline std::uint64_t attest_hybrid_dispatch_handle_family(
    std::uint64_t buffer_family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden, std::uint64_t num_experts,
    std::uint64_t num_topk, std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags,
    std::uint32_t route_layout_version,
    std::uint64_t route_record_count, std::uint64_t route_record_stride,
    HybridRouteStageFlags route_stage_flags,
    HybridRouteTableView route_table) noexcept {
    const auto geometry = attest_dispatch_handle_family(
        buffer_family, topology, generation, num_tokens, hidden, num_experts,
        num_topk, expert_alignment, num_max_tokens_per_rank, mode_flags,
        DispatchRoutingMode::kHybrid, route_layout_version,
        route_record_count, route_record_stride, generation,
        route_stage_flags);
    return attest_hybrid_route_table(geometry, route_table);
}

constexpr std::uint64_t attest_dispatch_handle_family(
    std::uint64_t buffer_family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank,
    CoreModeFlags mode_flags) noexcept {
    return attest_dispatch_handle_family(
        buffer_family, topology, generation, num_tokens, hidden, num_experts,
        num_topk, expert_alignment, num_max_tokens_per_rank, mode_flags,
        DispatchRoutingMode::kDirect, 0, 0, 0, generation, 0);
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

inline DispatchHandleDescriptor make_attested_dispatch_handle_descriptor(
    std::uint64_t buffer_family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden,
    std::uint64_t num_experts, std::uint64_t num_topk,
    std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags,
    DispatchRoutingMode routing_mode, std::uint32_t route_layout_version,
    std::uint64_t route_record_count, std::uint64_t route_record_stride,
    std::uint64_t dispatch_generation,
    HybridRouteStageFlags route_stage_flags) {
    return make_dispatch_handle_descriptor(
        attest_dispatch_handle_family(
            buffer_family, topology, generation, num_tokens, hidden, num_experts,
            num_topk, expert_alignment, num_max_tokens_per_rank, mode_flags,
            routing_mode, route_layout_version, route_record_count,
            route_record_stride, dispatch_generation, route_stage_flags),
        topology, generation, num_tokens, hidden, num_experts, num_topk,
        expert_alignment, num_max_tokens_per_rank, mode_flags, routing_mode,
        route_layout_version, route_record_count, route_record_stride,
        dispatch_generation, route_stage_flags);
}

inline DispatchHandleDescriptor make_attested_hybrid_dispatch_handle_descriptor(
    std::uint64_t buffer_family, const CoreTopology& topology,
    std::uint64_t generation, std::uint64_t num_tokens,
    std::uint64_t hidden, std::uint64_t num_experts,
    std::uint64_t num_topk, std::uint64_t expert_alignment,
    std::uint64_t num_max_tokens_per_rank, CoreModeFlags mode_flags,
    std::uint32_t route_layout_version,
    std::uint64_t route_record_count, std::uint64_t route_record_stride,
    HybridRouteStageFlags route_stage_flags,
    HybridRouteTableView route_table) {
    return make_dispatch_handle_descriptor(
        attest_hybrid_dispatch_handle_family(
            buffer_family, topology, generation, num_tokens, hidden,
            num_experts, num_topk, expert_alignment,
            num_max_tokens_per_rank, mode_flags, route_layout_version,
            route_record_count, route_record_stride, route_stage_flags,
            route_table),
        topology, generation, num_tokens, hidden, num_experts, num_topk,
        expert_alignment, num_max_tokens_per_rank, mode_flags,
        DispatchRoutingMode::kHybrid, route_layout_version,
        route_record_count, route_record_stride, generation,
        route_stage_flags);
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

DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE constexpr bool
is_complete_hybrid_route_stage_flags(
    HybridRouteStageFlags flags) noexcept {
    return flags == kHybridRouteCompleteStageFlags;
}

inline bool is_valid_dispatch_handle_descriptor(
    const DispatchHandleDescriptor& descriptor) noexcept {
    if (descriptor.abi_version != kDispatchHandleDescriptorAbiVersion ||
        descriptor.struct_size != sizeof(DispatchHandleDescriptor) ||
        descriptor.generation == 0 || descriptor.dispatch_generation == 0 ||
        descriptor.dispatch_generation != descriptor.generation)
        return false;

    if (descriptor.routing_mode == DispatchRoutingMode::kDirect)
        return !has_mode(descriptor.mode_flags, CoreMode::kHybrid) &&
               descriptor.route_layout_version == 0 &&
               descriptor.route_record_count == 0 &&
               descriptor.route_record_stride == 0 &&
               descriptor.route_stage_flags == 0;

    return descriptor.routing_mode == DispatchRoutingMode::kHybrid &&
           has_mode(descriptor.mode_flags, CoreMode::kHybrid) &&
           descriptor.route_layout_version == kHybridRouteLayoutVersion &&
           descriptor.route_record_stride == sizeof(HybridRouteRecord) &&
           descriptor.route_stage_flags == kHybridRouteCompleteStageFlags;
}

inline DispatchHandleStatus validate_dispatch_handle(
    const DispatchHandleDescriptor& expected,
    const DispatchHandleDescriptor& actual) {
    if (!is_valid_dispatch_handle_descriptor(expected) ||
        !is_valid_dispatch_handle_descriptor(actual))
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
            dispatch_handle_identity_mode_flags(actual.mode_flags) ||
        expected.routing_mode != actual.routing_mode ||
        expected.route_layout_version != actual.route_layout_version ||
        expected.route_record_count != actual.route_record_count ||
        expected.route_record_stride != actual.route_record_stride ||
        expected.dispatch_generation != actual.dispatch_generation ||
        expected.route_stage_flags != actual.route_stage_flags)
        return {DispatchHandleStatusCode::kMismatch,
                "dispatch handle does not match the current call"};
    return {};
}

inline DispatchHandleStatus validate_hybrid_route_table(
    const DispatchHandleDescriptor& descriptor, HybridRouteTableView route_table,
    std::uint64_t max_source_rows, std::uint64_t max_slots,
    std::uint64_t num_local_experts) {
    if (!is_valid_dispatch_handle_descriptor(descriptor) ||
        descriptor.routing_mode != DispatchRoutingMode::kHybrid ||
        route_table.count != descriptor.route_record_count ||
        (route_table.count != 0 && route_table.records == nullptr))
        return {DispatchHandleStatusCode::kInvalidDescriptor,
                "invalid hybrid route table descriptor"};

    for (std::uint64_t index = 0; index < route_table.count; ++index) {
        const auto& record = route_table.records[index];
        if (record.origin_world_rank < 0 ||
            record.destination_world_rank < 0 ||
            record.ingress_world_rank < 0 ||
            record.origin_world_rank >= descriptor.topology.world_size ||
            record.destination_world_rank >= descriptor.topology.world_size ||
            record.ingress_world_rank >= descriptor.topology.world_size ||
            record.destination_local_expert < 0 ||
            record.origin_source_row >= max_source_rows ||
            static_cast<std::uint64_t>(record.destination_local_expert) >=
                num_local_experts ||
            (record.ingress_slot != kInvalidHybridRouteSlot &&
             record.ingress_slot >= max_slots) ||
            (record.forwarded_slot != kInvalidHybridRouteSlot &&
             record.forwarded_slot >= max_slots) ||
            record.generation != descriptor.dispatch_generation ||
            record.topology_epoch != descriptor.topology.epoch ||
            record.stage_flags != kHybridRouteCompleteStageFlags)
            return {DispatchHandleStatusCode::kMismatch,
                    "hybrid route record does not match the dispatch handle"};

        for (std::uint64_t other_index = 0;
             other_index < route_table.count; ++other_index) {
            if (other_index == index)
                continue;
            const auto& other = route_table.records[other_index];
            if ((record.ingress_slot != kInvalidHybridRouteSlot &&
                 record.ingress_slot == other.ingress_slot &&
                 record.ingress_world_rank == other.ingress_world_rank) ||
                (record.forwarded_slot != kInvalidHybridRouteSlot &&
                 record.forwarded_slot == other.forwarded_slot &&
                 record.destination_world_rank ==
                     other.destination_world_rank))
                return {DispatchHandleStatusCode::kMismatch,
                        "hybrid route table contains duplicate slots"};
        }
    }
    return {};
}

inline DispatchHandleStatus validate_hybrid_route_bindings(
    const DispatchHandleDescriptor& descriptor,
    HybridRouteTableView route_table,
    const HybridRouteBindingView& bindings) {
    if (descriptor.routing_mode != DispatchRoutingMode::kHybrid ||
        descriptor.topology.world_size <= 0 ||
        descriptor.topology.scale_up_size <= 0 ||
        descriptor.topology.world_rank < 0 ||
        descriptor.topology.world_rank >= descriptor.topology.world_size ||
        descriptor.topology.world_size %
                descriptor.topology.scale_up_size != 0 ||
        route_table.count != descriptor.route_record_count ||
        bindings.row_count != route_table.count || bindings.num_topk == 0 ||
        bindings.shard_capacity == 0 ||
        (route_table.count != 0 &&
         (route_table.records == nullptr || bindings.source_metadata == nullptr ||
          bindings.received_topk_indices == nullptr)))
        return {DispatchHandleStatusCode::kInvalidDescriptor,
                "invalid hybrid route binding descriptor"};

    const int scale_up_size = descriptor.topology.scale_up_size;
    for (std::uint64_t index = 0; index < route_table.count; ++index) {
        const auto& record = route_table.records[index];
        const auto* metadata = bindings.source_metadata +
            index * (bindings.num_topk + 2);
        const int metadata_origin = metadata[0] < 0 ? -1 :
            static_cast<int>(static_cast<std::uint64_t>(metadata[0]) /
                             bindings.shard_capacity);
        const std::int32_t metadata_row = metadata[0] < 0 ? -1 :
            static_cast<std::int32_t>(
                static_cast<std::uint64_t>(metadata[0]) %
                bindings.shard_capacity);
        const int lane_origin = metadata[1] < 0 ? -1 :
            static_cast<int>(static_cast<std::uint64_t>(metadata[1]) /
                             bindings.num_topk);
        const std::int32_t master_lane = metadata[1] < 0 ? -1 :
            static_cast<std::int32_t>(
                static_cast<std::uint64_t>(metadata[1]) % bindings.num_topk);
        if (metadata_origin != record.origin_world_rank ||
            lane_origin != record.origin_world_rank || metadata_row < 0 ||
            static_cast<std::uint64_t>(metadata_row) !=
                record.origin_source_row ||
            master_lane < 0 ||
            static_cast<std::uint64_t>(master_lane) >= bindings.num_topk ||
            record.destination_world_rank != descriptor.topology.world_rank)
            return {DispatchHandleStatusCode::kMismatch,
                    "hybrid route record does not match source metadata"};

        const auto expected_local_expert = bindings.received_topk_indices[
            index * bindings.num_topk +
            static_cast<std::uint64_t>(master_lane)];
        if (expected_local_expert < 0 ||
            expected_local_expert != record.destination_local_expert)
            return {DispatchHandleStatusCode::kMismatch,
                    "hybrid route record does not match received top-k"};

        const int origin_domain = record.origin_world_rank / scale_up_size;
        const int destination_domain =
            record.destination_world_rank / scale_up_size;
        const int origin_rail = record.origin_world_rank % scale_up_size;
        const int destination_rail =
            record.destination_world_rank % scale_up_size;
        const bool diagonal = origin_domain != destination_domain &&
            origin_rail != destination_rail;
        const int expected_ingress = diagonal ?
            destination_domain * scale_up_size + origin_rail :
            record.destination_world_rank;
        if (record.ingress_world_rank != expected_ingress ||
            (diagonal &&
             (record.ingress_slot == kInvalidHybridRouteSlot ||
              record.forwarded_slot == kInvalidHybridRouteSlot)) ||
            (!diagonal &&
             (record.ingress_slot != kInvalidHybridRouteSlot ||
              record.forwarded_slot != kInvalidHybridRouteSlot)))
            return {DispatchHandleStatusCode::kMismatch,
                    "hybrid route record has invalid stage slots"};

        if (diagonal) {
            std::uint64_t prior_origin_rows = 0;
            for (std::uint64_t prior = 0; prior < index; ++prior) {
                if (route_table.records[prior].origin_world_rank ==
                    record.origin_world_rank)
                    ++prior_origin_rows;
            }
            const std::uint64_t expected_forwarded_slot =
                static_cast<std::uint64_t>(record.origin_world_rank) *
                    bindings.shard_capacity +
                prior_origin_rows;
            if (record.ingress_slot != expected_forwarded_slot ||
                record.forwarded_slot != expected_forwarded_slot)
                return {DispatchHandleStatusCode::kMismatch,
                        "hybrid route record has invalid staged slot"};
        }
    }
    return {};
}

}  // namespace deep_ep::ascend::elastic

#undef DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE
