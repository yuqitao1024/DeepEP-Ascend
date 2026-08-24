#pragma once

#include <cstdint>
#include <limits>

#include "kernels.hpp"

namespace deep_ep::ascend::elastic {

inline constexpr std::uint32_t kMaximumDispatchPipelineChunks = 8;

enum class DispatchPipelineConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchPipelineConfig {
    bool enabled = false;
    std::uint64_t chunk_slots = 0;
    std::uint32_t chunk_count = 0;
};

inline DispatchPipelineConfigStatus select_dispatch_pipeline_config(
    const char* value, bool cached_mode, bool split_dispatch,
    bool stream_mode, bool hybrid_mode, int world_size,
    std::uint64_t shard_capacity, DispatchPipelineConfig* output) noexcept {
    if (output == nullptr)
        return DispatchPipelineConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr)
        return DispatchPipelineConfigStatus::kDisabled;
    if (*value == '\0')
        return DispatchPipelineConfigStatus::kInvalid;

    std::uint64_t chunk_slots = 0;
    for (const char* cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9')
            return DispatchPipelineConfigStatus::kInvalid;
        const auto digit = static_cast<std::uint64_t>(*cursor - '0');
        if (chunk_slots >
            (std::numeric_limits<std::uint64_t>::max() - digit) / 10U)
            return DispatchPipelineConfigStatus::kInvalid;
        chunk_slots = chunk_slots * 10U + digit;
    }
    if (chunk_slots == 0)
        return DispatchPipelineConfigStatus::kInvalid;
    if (cached_mode || !split_dispatch || stream_mode || hybrid_mode ||
        world_size < 2)
        return DispatchPipelineConfigStatus::kDisabled;

    DispatchChunkPlan plan{};
    if (!build_dispatch_chunk_plan(shard_capacity, chunk_slots, &plan) ||
        plan.chunk_count < 2 ||
        plan.chunk_count > kMaximumDispatchPipelineChunks)
        return DispatchPipelineConfigStatus::kDisabled;

    output->enabled = true;
    output->chunk_slots = chunk_slots;
    output->chunk_count = plan.chunk_count;
    return DispatchPipelineConfigStatus::kEnabled;
}

}  // namespace deep_ep::ascend::elastic
