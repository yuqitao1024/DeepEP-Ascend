#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kDispatchTokenFanoutAlignmentBytes = 32;
inline constexpr std::uint64_t kDispatchTokenFanoutBufferBytes = 7168;
inline constexpr std::uint64_t kDispatchTokenFanoutMaximumTopk = 8;
inline constexpr int kDispatchTokenFanoutMaximumWorldSize = 8;

enum class DispatchTokenFanoutConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchTokenFanoutPlan {
    std::uint64_t vector_bytes = 0;
    std::uint64_t scalar_begin = 0;
    std::uint32_t source_loads_per_token = 0;
    bool valid = false;
};

struct DispatchTokenFanoutConfig {
    std::uint64_t vector_bytes = 0;
    bool enabled = false;
};

constexpr DispatchTokenFanoutPlan build_dispatch_token_fanout_plan(
    std::uint64_t hidden_bytes, std::uint64_t alignment_bytes,
    std::uint64_t buffer_bytes) noexcept {
    DispatchTokenFanoutPlan plan{};
    if (alignment_bytes == 0 ||
        (alignment_bytes & (alignment_bytes - 1)) != 0 ||
        buffer_bytes < alignment_bytes)
        return plan;
    const std::uint64_t vector_bytes =
        hidden_bytes - hidden_bytes % alignment_bytes;
    if (vector_bytes == 0 || vector_bytes > buffer_bytes)
        return plan;
    plan.vector_bytes = vector_bytes;
    plan.scalar_begin = vector_bytes;
    plan.source_loads_per_token = 1;
    plan.valid = true;
    return plan;
}

inline DispatchTokenFanoutConfigStatus select_dispatch_token_fanout_config(
    const char* value, bool grouping_enabled, bool fp8_dispatch,
    bool cached_mode, bool expanded, bool hybrid_mode, bool stream_mode,
    bool pipeline_enabled, std::uint64_t num_topk, int world_size,
    std::uint64_t hidden_bytes, DispatchTokenFanoutConfig* output) noexcept {
    if (output == nullptr)
        return DispatchTokenFanoutConfigStatus::kInvalid;
    *output = {};
    const auto plan = build_dispatch_token_fanout_plan(
        hidden_bytes, kDispatchTokenFanoutAlignmentBytes,
        kDispatchTokenFanoutBufferBytes);
    const bool eligible = grouping_enabled && fp8_dispatch && !cached_mode &&
        !expanded && !hybrid_mode && !stream_mode && !pipeline_enabled &&
        num_topk != 0 && num_topk <= kDispatchTokenFanoutMaximumTopk &&
        world_size >= 1 && world_size <= kDispatchTokenFanoutMaximumWorldSize &&
        plan.valid && plan.vector_bytes == kDispatchTokenFanoutBufferBytes;
    if (value == nullptr) {
        if (eligible) {
            output->vector_bytes = plan.vector_bytes;
            output->enabled = true;
        }
        return eligible ? DispatchTokenFanoutConfigStatus::kEnabled :
                          DispatchTokenFanoutConfigStatus::kDisabled;
    }
    if (value[0] == '0' && value[1] == '\0')
        return DispatchTokenFanoutConfigStatus::kDisabled;
    if (value[0] != '1' || value[1] != '\0')
        return DispatchTokenFanoutConfigStatus::kInvalid;

    if (!eligible)
        return DispatchTokenFanoutConfigStatus::kDisabled;

    output->vector_bytes = plan.vector_bytes;
    output->enabled = true;
    return DispatchTokenFanoutConfigStatus::kEnabled;
}

constexpr bool dispatch_token_fanout_work_bytes(
    std::uint64_t num_tokens, std::uint64_t destination_records,
    DispatchTokenFanoutPlan plan, std::uint64_t* source_bytes,
    std::uint64_t* destination_bytes) noexcept {
    if (source_bytes != nullptr)
        *source_bytes = 0;
    if (destination_bytes != nullptr)
        *destination_bytes = 0;
    if (source_bytes == nullptr || destination_bytes == nullptr ||
        !plan.valid || plan.vector_bytes == 0 ||
        num_tokens > std::numeric_limits<std::uint64_t>::max() /
                         plan.vector_bytes ||
        destination_records >
            std::numeric_limits<std::uint64_t>::max() / plan.vector_bytes)
        return false;
    *source_bytes = num_tokens * plan.vector_bytes;
    *destination_bytes = destination_records * plan.vector_bytes;
    return true;
}

}  // namespace deep_ep::ascend::elastic
