#pragma once

#include "combine_vf_launchers.hpp"

inline constexpr DirectReleaseSegment host_combine_release_segment(
    DirectCombineStage stage, bool profile_enabled) noexcept {
    if (stage == DirectCombineStage::kFull)
        return DirectReleaseSegment::kAll;
    if (stage == DirectCombineStage::kProducerRelease)
        return profile_enabled ? DirectReleaseSegment::kPayload :
                                 DirectReleaseSegment::kAll;
    if (profile_enabled && stage == DirectCombineStage::kProducerReleaseControl)
        return DirectReleaseSegment::kControl;
    if (profile_enabled && stage == DirectCombineStage::kProducerReleaseBarrier)
        return DirectReleaseSegment::kBarrier;
    return DirectReleaseSegment::kNone;
}
inline int launch_combine_producer(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_combine_producer(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_producer_control(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_producer_control(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_producer_plan(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_producer_plan(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_producer_plan_prefix(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_producer_plan_prefix(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_producer_record(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_producer_record(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_producer_local_copy(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_producer_local_copy(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_producer_release(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_producer_release(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_hybrid_combine_return(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_hybrid_combine_return(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_hybrid_combine_prepare_epilogue(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_hybrid_combine_prepare_epilogue(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_combine_epilogue(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_vf_combine_epilogue(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_epilogue_acquire(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_epilogue_acquire(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_epilogue_clear_index(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_epilogue_clear_index(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_epilogue_validate(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_epilogue_validate(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_epilogue_reduce_errors(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_epilogue_reduce_errors(
        arguments, tiling, stage, profile_enabled, stream, launch);
}

inline int launch_direct_combine_epilogue_prepare_vector_slots(
    CombineArguments arguments, CoreTiling tiling, void* stream,
    CoreLaunchShape launch, DirectCombineStage stage,
    bool profile_enabled) {
    return deep_ep_ascend_launch_direct_combine_epilogue_prepare_vector_slots(
        arguments, tiling, stage, profile_enabled, stream, launch);
}
