#pragma once

extern "C" int deep_ep_ascend_launch_combine_producer(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);
extern "C" int deep_ep_ascend_launch_direct_combine_producer_control(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_producer_plan(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_producer_plan_prefix(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_producer_record(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_producer_local_copy(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_producer_release(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_hybrid_combine_return(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_hybrid_combine_prepare_epilogue(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_vf_combine_epilogue(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_epilogue_acquire(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_epilogue_clear_index(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_epilogue_validate(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_epilogue_reduce_errors(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);

extern "C" int deep_ep_ascend_launch_direct_combine_epilogue_prepare_vector_slots(
    CombineArguments arguments, CoreTiling tiling,
    DirectCombineStage stage, bool profile_enabled,
    void* stream, CoreLaunchShape launch);
