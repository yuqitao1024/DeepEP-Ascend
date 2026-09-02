#pragma once

#include <cstdint>

#include "tiling.hpp"

namespace deep_ep::ascend::elastic {

struct HybridRouteRecord;

#ifndef DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE
#define DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE inline constexpr
#define DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE_LOCAL 1
#endif

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_KERNEL_CALLEE __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#else
#define DEEP_EP_ASCEND_KERNEL_CALLEE inline constexpr
#endif

struct DispatchChunkPlan {
    std::uint64_t shard_capacity = 0;
    std::uint64_t chunk_slots = 0;
    std::uint32_t chunk_count = 0;
};

struct DispatchSourceChunkPlan {
    std::uint32_t token_count = 0;
    std::uint32_t tile_count = 0;
    std::uint32_t tokens_per_tile = 0;
    std::uint32_t chunk_tiles = 0;
    std::uint32_t chunk_count = 0;
};

inline constexpr bool build_dispatch_source_chunk_plan(
    std::uint64_t token_count, std::uint64_t tokens_per_tile,
    std::uint64_t chunk_tiles, DispatchSourceChunkPlan* output) noexcept {
    if (output == nullptr || token_count == 0 || tokens_per_tile == 0 ||
        chunk_tiles == 0 || token_count > UINT32_MAX ||
        tokens_per_tile > UINT32_MAX || chunk_tiles > UINT32_MAX)
        return false;
    const auto tile_quotient = token_count / tokens_per_tile;
    const auto tile_remainder = token_count % tokens_per_tile;
    const auto tile_count = tile_quotient + (tile_remainder != 0);
    if (tile_count == 0 || tile_count > UINT32_MAX)
        return false;
    const auto chunk_quotient = tile_count / chunk_tiles;
    const auto chunk_remainder = tile_count % chunk_tiles;
    const auto chunk_count = chunk_quotient + (chunk_remainder != 0);
    if (chunk_count == 0 || chunk_count > UINT32_MAX)
        return false;
    *output = {
        static_cast<std::uint32_t>(token_count),
        static_cast<std::uint32_t>(tile_count),
        static_cast<std::uint32_t>(tokens_per_tile),
        static_cast<std::uint32_t>(chunk_tiles),
        static_cast<std::uint32_t>(chunk_count),
    };
    return true;
}

inline constexpr bool dispatch_source_chunk_bounds(
    const DispatchSourceChunkPlan& plan, std::uint32_t chunk_index,
    std::uint32_t* tile_begin, std::uint32_t* tile_end,
    std::uint32_t* token_begin, std::uint32_t* token_end) noexcept {
    if (tile_begin == nullptr || tile_end == nullptr ||
        token_begin == nullptr || token_end == nullptr ||
        plan.token_count == 0 || plan.tile_count == 0 ||
        plan.tokens_per_tile == 0 || plan.chunk_tiles == 0 ||
        chunk_index >= plan.chunk_count)
        return false;
    const auto begin = static_cast<std::uint64_t>(chunk_index) *
        plan.chunk_tiles;
    const auto end = plan.chunk_tiles > plan.tile_count - begin ?
        plan.tile_count : begin + plan.chunk_tiles;
    const auto first_token = begin * plan.tokens_per_tile;
    const auto last_token = end * plan.tokens_per_tile;
    *tile_begin = static_cast<std::uint32_t>(begin);
    *tile_end = static_cast<std::uint32_t>(end);
    *token_begin = static_cast<std::uint32_t>(first_token);
    *token_end = static_cast<std::uint32_t>(
        last_token < plan.token_count ? last_token : plan.token_count);
    return *tile_begin < *tile_end && *token_begin < *token_end;
}

inline constexpr bool dispatch_source_chunk_peer_range(
    const DispatchSourceChunkPlan& plan, std::uint32_t chunk_index,
    std::uint64_t prefix_at_begin, std::uint64_t prefix_at_end,
    std::uint64_t destination_count, std::uint64_t* chunk_begin,
    std::uint64_t* chunk_count) noexcept {
    if (chunk_begin == nullptr || chunk_count == nullptr)
        return false;
    std::uint32_t tile_begin = 0;
    std::uint32_t tile_end = 0;
    std::uint32_t token_begin = 0;
    std::uint32_t token_end = 0;
    if (!dispatch_source_chunk_bounds(
            plan, chunk_index, &tile_begin, &tile_end,
            &token_begin, &token_end))
        return false;
    const std::uint64_t end = tile_end == plan.tile_count ?
        destination_count : prefix_at_end;
    if (prefix_at_begin > end || end > destination_count)
        return false;
    *chunk_begin = prefix_at_begin;
    *chunk_count = end - prefix_at_begin;
    return true;
}

inline constexpr bool build_dispatch_chunk_plan(
    std::uint64_t shard_capacity, std::uint64_t chunk_slots,
    DispatchChunkPlan* output) noexcept {
    if (output == nullptr || shard_capacity == 0 || chunk_slots == 0)
        return false;
    const auto quotient = shard_capacity / chunk_slots;
    const auto remainder = shard_capacity % chunk_slots;
    if (quotient > UINT32_MAX ||
        (quotient == UINT32_MAX && remainder != 0))
        return false;
    *output = {shard_capacity, chunk_slots,
               static_cast<std::uint32_t>(quotient + (remainder != 0))};
    return output->chunk_count != 0;
}

inline constexpr std::uint32_t dispatch_pipeline_slot(
    std::uint32_t chunk_index) noexcept {
    return chunk_index % kDispatchPipelineSlotCount;
}

inline constexpr bool dispatch_pipeline_slot_reusable(
    const DispatchPipelineSlot& slot, std::uint32_t chunk_index) noexcept {
    if (slot.state == DispatchPipelineSlotState::kEmpty)
        return chunk_index < kDispatchPipelineSlotCount;
    return slot.state == DispatchPipelineSlotState::kCompleted &&
        slot.published_chunk <= UINT32_MAX - kDispatchPipelineSlotCount &&
        slot.published_chunk + kDispatchPipelineSlotCount == chunk_index;
}

inline constexpr bool dispatch_pipeline_producer_must_wait_for_reuse(
    std::uint32_t chunk_index) noexcept {
    return chunk_index >= kDispatchPipelineSlotCount;
}

inline constexpr bool dispatch_pipeline_completion_is_last(
    std::uint32_t completed_blocks,
    std::uint32_t producer_blocks) noexcept {
    return producer_blocks != 0 && completed_blocks < UINT32_MAX &&
        completed_blocks + 1 == producer_blocks;
}

inline constexpr bool dispatch_pipeline_slot_ready_for_release(
    const DispatchPipelineSlot& slot, std::uint64_t expected_generation,
    std::uint64_t observed_generation, std::uint32_t chunk_index,
    std::uint32_t producer_blocks) noexcept {
    return expected_generation != 0 &&
        observed_generation == expected_generation &&
        slot.state == DispatchPipelineSlotState::kReady &&
        slot.chunk_index == chunk_index &&
        slot.published_chunk == chunk_index &&
        slot.scalar_blocks_completed == producer_blocks &&
        slot.hidden_blocks_completed == producer_blocks;
}

inline constexpr bool dispatch_pipeline_block_progress_ready(
    const DispatchPipelineBlockProgress& progress,
    std::uint64_t generation,
    std::uint32_t chunk_index) noexcept {
    return generation != 0 && chunk_index < UINT32_MAX &&
        progress.generation == generation &&
        progress.completed_chunk == chunk_index;
}

// Scalar progress is indexed by source chunk, so generation is its sole
// publication value.  Keeping the chunk field out of the protocol avoids
// observing two independently published words from one cache line.
inline constexpr bool dispatch_pipeline_scalar_progress_ready(
    std::uint64_t observed_generation,
    std::uint64_t generation) noexcept {
    return generation != 0 && observed_generation == generation;
}

inline constexpr bool dispatch_pipeline_release_batch_pending(
    std::uint32_t batch_target, std::uint32_t batch_consumed) noexcept {
    return batch_target > batch_consumed;
}

inline constexpr bool dispatch_pipeline_release_batch_acknowledged(
    std::uint32_t batch_target, std::uint32_t batch_consumed) noexcept {
    return batch_target != 0 && batch_consumed >= batch_target;
}

inline constexpr bool dispatch_pipeline_wait_timed_out(
    std::uint64_t start_cycles, std::uint64_t current_cycles,
    std::uint64_t timeout_cycles) noexcept {
    return timeout_cycles != 0 &&
        current_cycles - start_cycles >= timeout_cycles;
}

inline constexpr std::uint32_t dispatch_persistent_producer_blocks(
    std::uint32_t requested_blocks) noexcept {
    // Keep one AICore block available for the persistent release service.
    return requested_blocks >= kAscendMaxDataBlocks ?
        kAscendMaxDataBlocks - 1U : requested_blocks;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
DEEP_EP_ASCEND_KERNEL_CALLEE std::uint32_t
dispatch_simt_pipeline_slot(std::uint32_t chunk_index) noexcept {
    return chunk_index % kDispatchPipelineSlotCount;
}

DEEP_EP_ASCEND_KERNEL_CALLEE bool
dispatch_simt_pipeline_producer_must_wait_for_reuse(
    std::uint32_t chunk_index) noexcept {
    return chunk_index >= kDispatchPipelineSlotCount;
}

DEEP_EP_ASCEND_KERNEL_CALLEE bool
dispatch_simt_pipeline_completion_is_last(
    std::uint32_t completed_blocks,
    std::uint32_t producer_blocks) noexcept {
    return producer_blocks != 0 && completed_blocks < UINT32_MAX &&
        completed_blocks + 1 == producer_blocks;
}

DEEP_EP_ASCEND_KERNEL_CALLEE bool
dispatch_simt_pipeline_release_batch_acknowledged(
    std::uint32_t batch_target, std::uint32_t batch_consumed) noexcept {
    return batch_target != 0 && batch_consumed >= batch_target;
}

DEEP_EP_ASCEND_KERNEL_CALLEE bool dispatch_simt_pipeline_wait_timed_out(
    std::uint64_t start_cycles, std::uint64_t current_cycles,
    std::uint64_t timeout_cycles) noexcept {
    return timeout_cycles != 0 &&
        current_cycles - start_cycles >= timeout_cycles;
}
#endif

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE std::uint32_t
dispatch_aicore_pipeline_slot(std::uint32_t chunk_index) noexcept {
    return chunk_index % kDispatchPipelineSlotCount;
}

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE bool
dispatch_aicore_pipeline_release_batch_pending(
    std::uint32_t batch_target, std::uint32_t batch_consumed) noexcept {
    return batch_target > batch_consumed;
}

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE bool
dispatch_aicore_pipeline_wait_timed_out(
    std::uint64_t start_cycles, std::uint64_t current_cycles,
    std::uint64_t timeout_cycles) noexcept {
    return timeout_cycles != 0 &&
        current_cycles - start_cycles >= timeout_cycles;
}

inline constexpr bool dispatch_chunk_peer_range(
    const DispatchChunkPlan& plan, std::uint32_t chunk_index,
    std::uint64_t peer_count, std::uint64_t* chunk_begin,
    std::uint64_t* chunk_count) noexcept {
    if (chunk_begin == nullptr || chunk_count == nullptr ||
        plan.chunk_slots == 0 || chunk_index >= plan.chunk_count ||
        peer_count > plan.shard_capacity)
        return false;
    const auto begin = static_cast<std::uint64_t>(chunk_index) *
        plan.chunk_slots;
    const auto end = plan.chunk_slots > plan.shard_capacity - begin ?
        plan.shard_capacity : begin + plan.chunk_slots;
    *chunk_begin = begin;
    *chunk_count = peer_count <= begin ? 0 :
        (peer_count < end ? peer_count : end) - begin;
    return true;
}

enum class DirectReleaseSegment : std::uint8_t {
    kNone,
    kAll,
    kPayload,
    kControl,
    kBarrier,
};

enum class DirectDispatchStage : std::uint8_t {
    kFull,
    kProducerControl,
    kProducerGroup,
    kProducerPrefix,
    kProducerRecord,
    kProducerRelease,
    kEpilogueAcquire,
    kEpiloguePrepare = kEpilogueAcquire,
    kEpilogueValidate,
    kEpilogueValidateReduce,
    kEpilogueExpertCount,
    kEpilogueExpertPrefix,
    kEpilogueMetadata,
    kEpilogueCopy,
    kEpilogueComplete,
    kProducerReleaseControl,
    kProducerReleaseBarrier,
    kProducerRecordPipeline,
    kProducerReleasePipeline,
};

inline constexpr DirectDispatchStage kFirstDirectDispatchEpilogueStage =
    DirectDispatchStage::kEpilogueAcquire;
inline constexpr DirectDispatchStage kLastDirectDispatchEpilogueStage =
    DirectDispatchStage::kEpilogueComplete;

struct DirectDispatchPipeline {
    DirectDispatchStage stages[15]{};
    std::uint32_t count = 0;
};

inline constexpr DirectDispatchPipeline direct_dispatch_pipeline(
    bool cpu_sync) noexcept {
    DirectDispatchPipeline pipeline{{
        DirectDispatchStage::kProducerControl,
        DirectDispatchStage::kProducerGroup,
        DirectDispatchStage::kProducerPrefix,
        DirectDispatchStage::kProducerRecord,
        DirectDispatchStage::kProducerRelease,
        DirectDispatchStage::kEpilogueAcquire,
        DirectDispatchStage::kEpilogueValidate,
        DirectDispatchStage::kEpilogueValidateReduce,
        DirectDispatchStage::kEpilogueExpertCount,
        DirectDispatchStage::kEpilogueExpertPrefix,
        DirectDispatchStage::kEpilogueMetadata,
        DirectDispatchStage::kEpilogueCopy,
        DirectDispatchStage::kEpilogueComplete,
    }, cpu_sync ? 10U : 13U};
    return pipeline;
}

inline constexpr DirectDispatchPipeline direct_dispatch_profile_pipeline(
    bool cpu_sync) noexcept {
    DirectDispatchPipeline pipeline{{
        DirectDispatchStage::kProducerControl,
        DirectDispatchStage::kProducerGroup,
        DirectDispatchStage::kProducerPrefix,
        DirectDispatchStage::kProducerRecord,
        DirectDispatchStage::kProducerRelease,
        DirectDispatchStage::kProducerReleaseControl,
        DirectDispatchStage::kProducerReleaseBarrier,
        DirectDispatchStage::kEpilogueAcquire,
        DirectDispatchStage::kEpilogueValidate,
        DirectDispatchStage::kEpilogueValidateReduce,
        DirectDispatchStage::kEpilogueExpertCount,
        DirectDispatchStage::kEpilogueExpertPrefix,
        DirectDispatchStage::kEpilogueMetadata,
        DirectDispatchStage::kEpilogueCopy,
        DirectDispatchStage::kEpilogueComplete,
    }, cpu_sync ? 12U : 15U};
    return pipeline;
}

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE DirectReleaseSegment
direct_dispatch_release_segment(
    DirectDispatchStage stage, bool profile_enabled) noexcept {
    if (stage == DirectDispatchStage::kFull)
        return DirectReleaseSegment::kAll;
    if (stage == DirectDispatchStage::kProducerRelease)
        return profile_enabled ? DirectReleaseSegment::kPayload :
                                 DirectReleaseSegment::kAll;
    if (profile_enabled &&
        stage == DirectDispatchStage::kProducerReleaseControl)
        return DirectReleaseSegment::kControl;
    if (profile_enabled &&
        stage == DirectDispatchStage::kProducerReleaseBarrier)
        return DirectReleaseSegment::kBarrier;
    return DirectReleaseSegment::kNone;
}

inline constexpr DirectDispatchPipeline
direct_dispatch_epilogue_pipeline() noexcept {
    DirectDispatchPipeline pipeline{{
        DirectDispatchStage::kEpilogueMetadata,
        DirectDispatchStage::kEpilogueCopy,
        DirectDispatchStage::kEpilogueComplete,
    }, 3U};
    return pipeline;
}

inline constexpr bool direct_dispatch_epilogue_stage(
    DirectDispatchStage stage) noexcept {
    return static_cast<std::uint8_t>(stage) >=
               static_cast<std::uint8_t>(
                   kFirstDirectDispatchEpilogueStage) &&
           static_cast<std::uint8_t>(stage) <=
               static_cast<std::uint8_t>(
                   kLastDirectDispatchEpilogueStage);
}

inline constexpr bool direct_dispatch_data_stage(
    DirectDispatchStage stage) noexcept {
    return stage == DirectDispatchStage::kProducerGroup ||
           stage == DirectDispatchStage::kProducerRecord ||
           stage == DirectDispatchStage::kProducerRecordPipeline ||
           stage == DirectDispatchStage::kEpilogueValidate ||
           stage == DirectDispatchStage::kEpilogueExpertCount ||
           stage == DirectDispatchStage::kEpilogueMetadata ||
           stage == DirectDispatchStage::kEpilogueCopy;
}

inline constexpr CoreLaunchShape direct_dispatch_stage_launch(
    const CoreTiling& tiling, DirectDispatchStage stage) noexcept {
    return direct_dispatch_data_stage(stage) ?
        tiling.data_launch : tiling.control_launch;
}

enum class DirectCombineStage : std::uint8_t {
    kFull,
    kProducerControl,
    kProducerPlan,
    kProducerPlanPrefix,
    kProducerRecord,
    kProducerRelease,
    kEpilogueAcquire,
    kEpiloguePrepare = kEpilogueAcquire,
    kEpilogueValidate,
    kEpilogueValidateReduce,
    kEpilogueReduce,
    kEpilogueWeights,
    kEpilogueComplete,
    kProducerReleaseControl,
    kProducerReleaseBarrier,
    kProducerLocalCopy,
};

struct DirectCombinePipeline {
    DirectCombineStage stages[14]{};
    std::uint32_t count = 0;
};

inline constexpr DirectCombinePipeline direct_combine_pipeline() noexcept {
    return {{
        DirectCombineStage::kProducerControl,
        DirectCombineStage::kProducerPlan,
        DirectCombineStage::kProducerPlanPrefix,
        DirectCombineStage::kProducerRecord,
        DirectCombineStage::kProducerRelease,
        DirectCombineStage::kEpilogueAcquire,
        DirectCombineStage::kEpilogueValidate,
        DirectCombineStage::kEpilogueValidateReduce,
        DirectCombineStage::kEpilogueReduce,
        DirectCombineStage::kEpilogueWeights,
        DirectCombineStage::kEpilogueComplete,
    }, 11U};
}

inline constexpr DirectCombinePipeline
direct_combine_profile_pipeline() noexcept {
    return {{
        DirectCombineStage::kProducerControl,
        DirectCombineStage::kProducerPlan,
        DirectCombineStage::kProducerPlanPrefix,
        DirectCombineStage::kProducerRecord,
        DirectCombineStage::kProducerLocalCopy,
        DirectCombineStage::kProducerRelease,
        DirectCombineStage::kProducerReleaseControl,
        DirectCombineStage::kProducerReleaseBarrier,
        DirectCombineStage::kEpilogueAcquire,
        DirectCombineStage::kEpilogueValidate,
        DirectCombineStage::kEpilogueValidateReduce,
        DirectCombineStage::kEpilogueReduce,
        DirectCombineStage::kEpilogueWeights,
        DirectCombineStage::kEpilogueComplete,
    }, 14U};
}

DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE DirectReleaseSegment
direct_combine_release_segment(
    DirectCombineStage stage, bool profile_enabled) noexcept {
    if (stage == DirectCombineStage::kFull)
        return DirectReleaseSegment::kAll;
    if (stage == DirectCombineStage::kProducerRelease)
        return profile_enabled ? DirectReleaseSegment::kPayload :
                                 DirectReleaseSegment::kAll;
    if (profile_enabled &&
        stage == DirectCombineStage::kProducerReleaseControl)
        return DirectReleaseSegment::kControl;
    if (profile_enabled &&
        stage == DirectCombineStage::kProducerReleaseBarrier)
        return DirectReleaseSegment::kBarrier;
    return DirectReleaseSegment::kNone;
}

#if defined(DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE_LOCAL)
#undef DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE_LOCAL
#undef DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE
#endif

inline constexpr bool direct_combine_data_stage(
    DirectCombineStage stage) noexcept {
    return stage == DirectCombineStage::kProducerPlan ||
           stage == DirectCombineStage::kProducerRecord ||
           stage == DirectCombineStage::kEpilogueValidate ||
           stage == DirectCombineStage::kEpilogueReduce ||
           stage == DirectCombineStage::kEpilogueWeights;
}

inline constexpr CoreLaunchShape direct_combine_stage_launch(
    const CoreTiling& tiling, DirectCombineStage stage) noexcept {
    return direct_combine_data_stage(stage) ?
        tiling.data_launch : tiling.control_launch;
}

enum class WorldRouteKind : std::uint32_t {
    kLocal,
    kScaleUp,
    kScaleOut,
    kDiagonal,
};

struct WorldRoute {
    WorldRouteKind kind = WorldRouteKind::kLocal;
    int ingress_world_rank = 0;
};

struct ReleaseBoundary {
    int control_slot_world_rank = 0;
    int signal_sender_world_rank = 0;
    bool remote_acquire_required = false;
};

struct DirectDataGridStride {
    std::uint32_t first = 0;
    std::uint32_t stride = 0;
};

struct DirectSubgroupGridStride {
    std::uint32_t first = 0;
    std::uint32_t stride = 0;
    std::uint32_t lane = 0;
};

DEEP_EP_ASCEND_KERNEL_CALLEE DirectDataGridStride direct_data_grid_stride(
    std::uint32_t block_index, std::uint32_t thread_index,
    std::uint32_t num_blocks, std::uint32_t num_threads) noexcept {
    return {
        block_index * num_threads + thread_index,
        num_blocks * num_threads,
    };
}

DEEP_EP_ASCEND_KERNEL_CALLEE DirectDataGridStride
direct_block_distributed_grid_stride(
    std::uint32_t block_index, std::uint32_t thread_index,
    std::uint32_t num_blocks, std::uint32_t num_threads) noexcept {
    return {
        thread_index * num_blocks + block_index,
        num_blocks * num_threads,
    };
}

DEEP_EP_ASCEND_KERNEL_CALLEE DirectSubgroupGridStride
direct_subgroup_grid_stride(
    std::uint32_t block_index, std::uint32_t thread_index,
    std::uint32_t num_blocks, std::uint32_t num_threads,
    std::uint32_t subgroup_width) noexcept {
    const std::uint32_t subgroups_per_block =
        num_threads / subgroup_width;
    const std::uint32_t subgroup_in_block =
        thread_index / subgroup_width;
    return {
        block_index * subgroups_per_block + subgroup_in_block,
        num_blocks * subgroups_per_block,
        thread_index % subgroup_width,
    };
}

DEEP_EP_ASCEND_KERNEL_CALLEE WorldRoute classify_world_route(
    int origin_world_rank, int destination_world_rank,
    int scale_up_size) noexcept {
    if (origin_world_rank == destination_world_rank)
        return {WorldRouteKind::kLocal, destination_world_rank};
    const int origin_domain = origin_world_rank / scale_up_size;
    const int destination_domain = destination_world_rank / scale_up_size;
    const int origin_rail = origin_world_rank % scale_up_size;
    const int destination_rail = destination_world_rank % scale_up_size;
    if (origin_domain == destination_domain)
        return {WorldRouteKind::kScaleUp, destination_world_rank};
    if (origin_rail == destination_rail)
        return {WorldRouteKind::kScaleOut, destination_world_rank};
    return {WorldRouteKind::kDiagonal,
            destination_domain * scale_up_size + origin_rail};
}

DEEP_EP_ASCEND_KERNEL_CALLEE int final_release_sender_world_rank(
    const WorldRoute& route, int direct_sender_world_rank) noexcept {
    return route.kind == WorldRouteKind::kDiagonal ?
        route.ingress_world_rank : direct_sender_world_rank;
}

DEEP_EP_ASCEND_KERNEL_CALLEE ReleaseBoundary dispatch_release_boundary(
    int source_world_rank, int destination_world_rank,
    int scale_up_size) noexcept {
    const auto route = classify_world_route(
        source_world_rank, destination_world_rank, scale_up_size);
    return {
        source_world_rank,
        final_release_sender_world_rank(route, source_world_rank),
        route.kind != WorldRouteKind::kLocal &&
            route.kind != WorldRouteKind::kDiagonal,
    };
}

DEEP_EP_ASCEND_KERNEL_CALLEE ReleaseBoundary
dispatch_prepare_release_boundary(
    int source_world_rank, int destination_world_rank,
    int scale_up_size) noexcept {
    const auto route = classify_world_route(
        source_world_rank, destination_world_rank, scale_up_size);
    return {
        source_world_rank,
        route.ingress_world_rank,
        route.kind == WorldRouteKind::kDiagonal,
    };
}

DEEP_EP_ASCEND_KERNEL_CALLEE ReleaseBoundary combine_release_boundary(
    int origin_world_rank, int contributor_world_rank,
    int scale_up_size) noexcept {
    const auto route = classify_world_route(
        origin_world_rank, contributor_world_rank, scale_up_size);
    return {
        contributor_world_rank,
        final_release_sender_world_rank(route, contributor_world_rank),
        route.kind != WorldRouteKind::kLocal &&
            route.kind != WorldRouteKind::kDiagonal,
    };
}

DEEP_EP_ASCEND_KERNEL_CALLEE ReleaseBoundary
combine_prepare_release_boundary(
    int origin_world_rank, int contributor_world_rank,
    int scale_up_size) noexcept {
    const auto route = classify_world_route(
        origin_world_rank, contributor_world_rank, scale_up_size);
    return {
        contributor_world_rank,
        route.ingress_world_rank,
        route.kind == WorldRouteKind::kDiagonal,
    };
}

struct BarrierArguments {
    void* workspace = nullptr;
    std::uint64_t generation = 0;
    std::uint64_t timeout_cycles = 0;
};

struct DispatchArguments {
    const void* x = nullptr;
    const void* scale_factors = nullptr;
    std::uint64_t scale_factor_token_stride = 0;
    std::uint64_t scale_factor_pack_stride = 0;
    const std::int64_t* topk_indices = nullptr;
    const float* topk_weights = nullptr;
    void* communication_buffer = nullptr;
    void* workspace = nullptr;
    void* recv_x = nullptr;
    void* recv_scale_factors = nullptr;
    std::uint64_t recv_scale_factor_token_stride = 0;
    std::uint64_t recv_scale_factor_pack_stride = 0;
    std::int64_t* recv_topk_indices = nullptr;
    float* recv_topk_weights = nullptr;
    std::int32_t* prefix_per_rank = nullptr;
    std::int32_t* prefix_per_expert = nullptr;
    std::int32_t* unaligned_per_expert = nullptr;
    std::int32_t* destination_slots = nullptr;
    std::int32_t* source_metadata = nullptr;
    HybridRouteRecord* route_records = nullptr;
    std::uint64_t route_record_capacity = 0;
    std::uint64_t num_recv_tokens = 0;
    std::uint64_t num_output_tokens = 0;
    std::uint64_t generation = 0;
    std::uint64_t timeout_cycles = 0;
    std::uint64_t pipeline_chunk_begin = 0;
    std::uint64_t pipeline_chunk_end = 0;
    std::uint32_t pipeline_chunk_index = 0;
    std::uint32_t pipeline_final_chunk = 1;
    std::uint64_t pipeline_chunk_slots = 0;
    std::uint32_t pipeline_chunk_tiles = 0;
    std::uint32_t pipeline_source_chunk = 0;
    std::uint32_t consumer_tile_bytes = 512;
    std::uint32_t parallel_prefix = 0;
    std::uint32_t token_fanout = 0;
    std::uint32_t early_route_plan = 0;
};

struct CombineArguments {
    const void* x = nullptr;
    const float* topk_weights = nullptr;
    const std::int32_t* source_metadata = nullptr;
    const HybridRouteRecord* route_records = nullptr;
    std::uint64_t route_record_count = 0;
    const std::int64_t* combined_topk_indices = nullptr;
    const std::int32_t* prefix_per_rank = nullptr;
    const void* bias_0 = nullptr;
    const void* bias_1 = nullptr;
    void* communication_buffer = nullptr;
    void* workspace = nullptr;
    void* combined_x = nullptr;
    float* combined_topk_weights = nullptr;
    std::uint64_t generation = 0;
    std::uint64_t timeout_cycles = 0;
    std::uint64_t num_source_rows = 0;
    std::uint64_t num_input_rows = 0;
    std::uintptr_t local_window_base = 0;
    std::uint32_t expanded_vector_reduce = 0;
    std::uint32_t local_copy_datacopy = 0;
    std::uint32_t direct_local_placement = 0;
    std::uint32_t metadata_buckets = 0;
    std::uint32_t vector_reduce_tile_elements = 0;
};

}  // namespace deep_ep::ascend::elastic

#undef DEEP_EP_ASCEND_KERNEL_CALLEE

extern "C" int deep_ep_ascend_launch_barrier(
    deep_ep::ascend::elastic::BarrierArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
extern "C" int deep_ep_ascend_launch_dispatch(
    deep_ep::ascend::elastic::DispatchArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
extern "C" int deep_ep_ascend_launch_dispatch_pipeline(
    deep_ep::ascend::elastic::DispatchArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* producer_stream,
    void* communication_stream);
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    deep_ep::ascend::elastic::DispatchArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
extern "C" int deep_ep_ascend_launch_combine(
    deep_ep::ascend::elastic::CombineArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
extern "C" int deep_ep_ascend_launch_combine_epilogue(
    deep_ep::ascend::elastic::CombineArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
