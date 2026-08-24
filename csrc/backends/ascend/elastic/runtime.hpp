#pragma once

#include <cstdint>

#include "kernels.hpp"

namespace deep_ep::ascend::elastic {

enum class CoreRuntimeStatusCode : std::uint8_t {
    kSuccess,
    kInvalidArgument,
    kUnsupportedTopology,
    kUnsupportedMode,
    kInsufficientStorage,
    kLaunchFailure,
};

struct CoreRuntimeStatus {
    CoreRuntimeStatusCode code = CoreRuntimeStatusCode::kSuccess;
    int backend_code = 0;
    const char* message = "";

    constexpr bool ok() const {
        return code == CoreRuntimeStatusCode::kSuccess;
    }
};

constexpr CoreTopology core_topology_from_transport(
    const transport::TransportTopology& topology) noexcept {
    return {
        topology.world_rank,
        topology.world_size,
        topology.scale_up_rank,
        topology.scale_up_size,
        topology.scale_out_rank,
        topology.scale_out_size,
        topology.kind,
        topology.epoch,
    };
}

struct CoreLaunchStorage {
    std::uint64_t communication_buffer_bytes = 0;
    std::uint64_t workspace_bytes = 0;
};

constexpr CoreLaunchStorage merge_core_launch_storage(
    const CoreLaunchStorage& lhs, const CoreLaunchStorage& rhs) noexcept {
    return {
        lhs.communication_buffer_bytes >= rhs.communication_buffer_bytes ?
            lhs.communication_buffer_bytes : rhs.communication_buffer_bytes,
        lhs.workspace_bytes >= rhs.workspace_bytes ?
            lhs.workspace_bytes : rhs.workspace_bytes,
    };
}

CoreLaunchStorage required_core_launch_storage(const CoreTiling& tiling);

CoreRuntimeStatus validate_internal_launch(
    const CoreTiling& tiling, const CoreLaunchStorage& storage);

CoreRuntimeStatus launch_internal_barrier(
    const BarrierArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

CoreRuntimeStatus launch_internal_dispatch(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

CoreRuntimeStatus launch_internal_dispatch_pipeline(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* producer_stream,
    void* communication_stream);

CoreRuntimeStatus launch_internal_dispatch_epilogue(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

CoreRuntimeStatus launch_internal_combine(
    const CombineArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

}  // namespace deep_ep::ascend::elastic
