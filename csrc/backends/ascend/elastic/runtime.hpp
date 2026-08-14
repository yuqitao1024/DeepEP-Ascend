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

struct CoreLaunchStorage {
    std::uint64_t communication_buffer_bytes = 0;
    std::uint64_t workspace_bytes = 0;
};

CoreLaunchStorage required_core_launch_storage(const CoreTiling& tiling);

CoreRuntimeStatus validate_internal_launch(
    const CoreTiling& tiling, const CoreLaunchStorage& storage);

CoreRuntimeStatus launch_internal_barrier(
    const BarrierArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

CoreRuntimeStatus launch_internal_dispatch(
    const DispatchArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

CoreRuntimeStatus launch_internal_combine(
    const CombineArguments& arguments, const CoreTiling& tiling,
    const CoreLaunchStorage& storage, void* stream);

}  // namespace deep_ep::ascend::elastic
