#pragma once

#include <cstdint>
#include <limits>

#include "kernels.hpp"

namespace deep_ep::ascend::elastic {

inline constexpr std::uint32_t kMaximumDispatchPipelineChunks = 8;

enum class DispatchParallelPrefixConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchParallelPrefixConfig {
    bool enabled = false;
};

inline DispatchParallelPrefixConfigStatus
select_dispatch_parallel_prefix_config(
    const char* value, bool device_prefix_enabled, bool cached_mode,
    bool cpu_sync, bool expanded, bool hybrid_mode, bool stream_mode,
    DispatchParallelPrefixConfig* output) noexcept {
    if (output == nullptr)
        return DispatchParallelPrefixConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr || (value[0] == '0' && value[1] == '\0'))
        return DispatchParallelPrefixConfigStatus::kDisabled;
    if (value[0] != '1' || value[1] != '\0')
        return DispatchParallelPrefixConfigStatus::kInvalid;
    if (!device_prefix_enabled || cached_mode || !cpu_sync || expanded ||
        hybrid_mode || stream_mode)
        return DispatchParallelPrefixConfigStatus::kDisabled;
    output->enabled = true;
    return DispatchParallelPrefixConfigStatus::kEnabled;
}

enum class DispatchConsumerTileConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchConsumerTileConfig {
    std::uint32_t tile_bytes = 512;
};

inline DispatchConsumerTileConfigStatus select_dispatch_consumer_tile_config(
    const char* value, bool device_prefix_enabled, bool cached_mode,
    bool cpu_sync, bool expanded, bool hybrid_mode, bool stream_mode,
    DispatchConsumerTileConfig* output) noexcept {
    if (output == nullptr)
        return DispatchConsumerTileConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr)
        return DispatchConsumerTileConfigStatus::kDisabled;
    if (*value == '\0')
        return DispatchConsumerTileConfigStatus::kInvalid;

    std::uint32_t tile_bytes = 0;
    for (const char* cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9')
            return DispatchConsumerTileConfigStatus::kInvalid;
        const auto digit = static_cast<std::uint32_t>(*cursor - '0');
        if (tile_bytes >
            (std::numeric_limits<std::uint32_t>::max() - digit) / 10U)
            return DispatchConsumerTileConfigStatus::kInvalid;
        tile_bytes = tile_bytes * 10U + digit;
    }
    if (tile_bytes != 512 && tile_bytes != 1024 && tile_bytes != 2048 &&
        tile_bytes != 4096)
        return DispatchConsumerTileConfigStatus::kInvalid;
    if (tile_bytes == 512 || !device_prefix_enabled || cached_mode ||
        !cpu_sync || expanded || hybrid_mode || stream_mode)
        return DispatchConsumerTileConfigStatus::kDisabled;
    output->tile_bytes = tile_bytes;
    return DispatchConsumerTileConfigStatus::kEnabled;
}

enum class DispatchDevicePrefixConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchDevicePrefixConfig {
    bool enabled = false;
};

inline DispatchDevicePrefixConfigStatus select_dispatch_device_prefix_config(
    const char* value, bool cached_mode, bool cpu_sync, bool hybrid_mode,
    bool stream_mode, DispatchDevicePrefixConfig* output) noexcept {
    if (output == nullptr)
        return DispatchDevicePrefixConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr || (value[0] == '0' && value[1] == '\0'))
        return DispatchDevicePrefixConfigStatus::kDisabled;
    if (value[0] != '1' || value[1] != '\0')
        return DispatchDevicePrefixConfigStatus::kInvalid;
    if (cached_mode || !cpu_sync || hybrid_mode || stream_mode)
        return DispatchDevicePrefixConfigStatus::kDisabled;
    output->enabled = true;
    return DispatchDevicePrefixConfigStatus::kEnabled;
}

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

enum class DispatchSourcePipelineConfigStatus : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalid,
};

struct DispatchSourcePipelineConfig {
    bool enabled = false;
    std::uint32_t chunk_tiles = 0;
    std::uint32_t chunk_count = 0;
};

inline DispatchSourcePipelineConfigStatus
select_dispatch_source_pipeline_config(
    const char* value, bool device_prefix_enabled, bool cached_mode,
    bool cpu_sync, bool expanded, bool hybrid_mode, bool stream_mode,
    int world_size, std::uint64_t num_tokens,
    DispatchSourcePipelineConfig* output) noexcept {
    if (output == nullptr)
        return DispatchSourcePipelineConfigStatus::kInvalid;
    *output = {};
    if (value == nullptr)
        return DispatchSourcePipelineConfigStatus::kDisabled;
    if (*value == '\0')
        return DispatchSourcePipelineConfigStatus::kInvalid;

    std::uint64_t chunk_tiles = 0;
    for (const char* cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9')
            return DispatchSourcePipelineConfigStatus::kInvalid;
        const auto digit = static_cast<std::uint64_t>(*cursor - '0');
        if (chunk_tiles >
            (std::numeric_limits<std::uint64_t>::max() - digit) / 10U)
            return DispatchSourcePipelineConfigStatus::kInvalid;
        chunk_tiles = chunk_tiles * 10U + digit;
    }
    if (chunk_tiles == 0)
        return DispatchSourcePipelineConfigStatus::kInvalid;
    if (!device_prefix_enabled || cached_mode || !cpu_sync || expanded ||
        hybrid_mode || stream_mode || world_size < 2)
        return DispatchSourcePipelineConfigStatus::kDisabled;

    DispatchSourceChunkPlan plan{};
    if (!build_dispatch_source_chunk_plan(
            num_tokens, kDispatchGroupingTokensPerTile, chunk_tiles, &plan) ||
        plan.chunk_count < 2 ||
        plan.chunk_count > kMaximumDispatchPipelineChunks)
        return DispatchSourcePipelineConfigStatus::kDisabled;
    output->enabled = true;
    output->chunk_tiles = plan.chunk_tiles;
    output->chunk_count = plan.chunk_count;
    return DispatchSourcePipelineConfigStatus::kEnabled;
}

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
