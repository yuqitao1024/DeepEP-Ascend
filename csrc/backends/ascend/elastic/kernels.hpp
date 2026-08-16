#pragma once

#include <cstdint>

#include "tiling.hpp"

namespace deep_ep::ascend::elastic {

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
};

struct CombineArguments {
    const void* x = nullptr;
    const float* topk_weights = nullptr;
    const std::int32_t* source_metadata = nullptr;
    const std::int64_t* combined_topk_indices = nullptr;
    const std::int32_t* prefix_per_rank = nullptr;
    const void* bias_0 = nullptr;
    const void* bias_1 = nullptr;
    void* communication_buffer = nullptr;
    void* workspace = nullptr;
    void* combined_x = nullptr;
    float* combined_topk_weights = nullptr;
};

}  // namespace deep_ep::ascend::elastic

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
