#pragma once

#include <cstdint>

#include "tiling.hpp"

namespace deep_ep::ascend::elastic {

struct HybridRouteRecord;

struct DispatchChunkPlan {
    std::uint64_t shard_capacity = 0;
    std::uint64_t chunk_slots = 0;
    std::uint32_t chunk_count = 0;
};

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
};

inline constexpr DirectDispatchStage kFirstDirectDispatchEpilogueStage =
    DirectDispatchStage::kEpilogueAcquire;
inline constexpr DirectDispatchStage kLastDirectDispatchEpilogueStage =
    DirectDispatchStage::kEpilogueComplete;

struct DirectDispatchPipeline {
    DirectDispatchStage stages[13]{};
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
};

struct DirectCombinePipeline {
    DirectCombineStage stages[11]{};
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

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_KERNEL_CALLEE __SIMT_DEVICE_FUNCTIONS_DECL__ inline
#else
#define DEEP_EP_ASCEND_KERNEL_CALLEE inline constexpr
#endif

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
