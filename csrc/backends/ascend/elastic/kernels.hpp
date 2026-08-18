#pragma once

#include <cstdint>

#include "tiling.hpp"

namespace deep_ep::ascend::elastic {

struct HybridRouteRecord;

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
    const std::int64_t* topk_indices = nullptr;
    const float* topk_weights = nullptr;
    void* communication_buffer = nullptr;
    void* workspace = nullptr;
    void* recv_x = nullptr;
    void* recv_scale_factors = nullptr;
    std::int64_t* recv_topk_indices = nullptr;
    float* recv_topk_weights = nullptr;
    std::int32_t* prefix_per_rank = nullptr;
    std::int32_t* prefix_per_expert = nullptr;
    std::int32_t* unaligned_per_expert = nullptr;
    std::int32_t* destination_slots = nullptr;
    std::int32_t* source_metadata = nullptr;
    HybridRouteRecord* route_records = nullptr;
    std::uint64_t route_record_capacity = 0;
    std::uint64_t generation = 0;
    std::uint64_t timeout_cycles = 0;
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
extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    deep_ep::ascend::elastic::DispatchArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
extern "C" int deep_ep_ascend_launch_combine(
    deep_ep::ascend::elastic::CombineArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
extern "C" int deep_ep_ascend_launch_combine_epilogue(
    deep_ep::ascend::elastic::CombineArguments arguments,
    deep_ep::ascend::elastic::CoreTiling tiling, void* stream);
