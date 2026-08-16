#include <cstdint>

#include "csrc/backends/ascend/elastic/runtime.hpp"

using namespace deep_ep::ascend::elastic;
namespace transport = deep_ep::ascend::transport;

namespace {

enum LaunchId : int {
    kBarrierLaunch = 1,
    kDispatchLaunch,
    kDispatchEpilogueLaunch,
    kCombineLaunch,
    kCombineEpilogueLaunch,
};

int launch_trace[8] = {};
int launch_trace_size = 0;
int failing_launch = 0;
int failing_code = 0;
std::uint64_t barrier_generation = 0;

int record_launch(LaunchId launch) {
    launch_trace[launch_trace_size++] = launch;
    return failing_launch == launch ? failing_code : 0;
}

void reset_launches(LaunchId fail = static_cast<LaunchId>(0), int code = 0) {
    launch_trace_size = 0;
    failing_launch = fail;
    failing_code = code;
}

bool trace_is(int first, int second = 0) {
    const int expected_size = second == 0 ? 1 : 2;
    return launch_trace_size == expected_size && launch_trace[0] == first &&
           (second == 0 || launch_trace[1] == second);
}

CoreTiling valid_tiling(
    OperationKind operation,
    ElementKind element_kind = ElementKind::kBFloat16,
    int world_rank = 0, int world_size = 1) {
    CoreTilingInput input{};
    input.operation = operation;
    input.element_kind = element_kind;
    input.num_tokens = operation == OperationKind::kBarrier ? 0 : 4;
    input.hidden = 64;
    input.num_experts = 4;
    input.num_topk = 2;
    input.expert_alignment = 4;
    input.num_max_tokens_per_rank = 8;
    input.topology.world_rank = world_rank;
    input.topology.world_size = world_size;
    input.topology.scale_up_rank = world_rank;
    input.topology.scale_up_size = world_size;
    input.topology.scale_out_rank = 0;
    input.topology.scale_out_size = 1;
    if (element_kind == ElementKind::kFloat8E4M3) {
        input.num_scale_factor_packs = 4;
        input.scale_factor_pack_bytes = 4;
    }
    CoreTiling tiling{};
    if (!build_core_tiling(input, &tiling).ok())
        return {};
    return tiling;
}

CoreTiling valid_barrier_tiling(int world_rank) {
    auto tiling = valid_tiling(
        OperationKind::kBarrier, ElementKind::kBFloat16, world_rank, 2);
    tiling.transport_context.capabilities =
        transport::capability_bit(
            transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(
            transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(
            transport::TransportCapability::kScaleUpTeam);
    tiling.transport_context.local_window_base = 0x200000;
    tiling.transport_context.peer_address_table = 0x300000;
    tiling.transport_context.channel_table = 0x400000;
    tiling.transport_context.backend_context = 0x500000;
    return tiling;
}

}  // namespace

extern "C" int deep_ep_ascend_launch_barrier(
    BarrierArguments arguments, CoreTiling, void*) {
    barrier_generation = arguments.generation;
    return record_launch(kBarrierLaunch);
}

extern "C" int deep_ep_ascend_launch_dispatch(
    DispatchArguments, CoreTiling, void*) {
    return record_launch(kDispatchLaunch);
}

extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    DispatchArguments, CoreTiling, void*) {
    return record_launch(kDispatchEpilogueLaunch);
}

extern "C" int deep_ep_ascend_launch_combine(
    CombineArguments, CoreTiling, void*) {
    return record_launch(kCombineLaunch);
}

extern "C" int deep_ep_ascend_launch_combine_epilogue(
    CombineArguments, CoreTiling, void*) {
    return record_launch(kCombineEpilogueLaunch);
}

int main() {
    auto barrier_rank_zero = valid_barrier_tiling(0);
    auto barrier_rank_one = valid_barrier_tiling(1);
    const auto barrier_storage =
        required_core_launch_storage(barrier_rank_zero);
    if (!validate_internal_launch(barrier_rank_zero, barrier_storage).ok() ||
        !validate_internal_launch(barrier_rank_one, barrier_storage).ok())
        return 30;

    auto malformed_barrier = barrier_rank_zero;
    malformed_barrier.topology.world_size = 3;
    malformed_barrier.topology.scale_up_size = 3;
    malformed_barrier.transport_context.topology.world_size = 3;
    malformed_barrier.transport_context.topology.scale_up_size = 3;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kUnsupportedTopology)
        return 31;
    malformed_barrier = barrier_rank_zero;
    malformed_barrier.topology.scale_out_size = 2;
    malformed_barrier.transport_context.topology.scale_out_size = 2;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kUnsupportedTopology)
        return 32;
    malformed_barrier = barrier_rank_zero;
    malformed_barrier.transport_context.topology.world_rank = 1;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 33;
    malformed_barrier = barrier_rank_zero;
    malformed_barrier.transport_context.capabilities = 0;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 34;
    malformed_barrier = barrier_rank_zero;
    malformed_barrier.transport_context.backend_context = 0;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 39;
    malformed_barrier = barrier_rank_zero;
    malformed_barrier.symmetric_window_layout.control_offset +=
        kAscendElasticAlignment;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 40;

    auto insufficient_barrier = barrier_storage;
    --insufficient_barrier.workspace_bytes;
    if (validate_internal_launch(
            barrier_rank_zero, insufficient_barrier).code !=
        CoreRuntimeStatusCode::kInsufficientStorage)
        return 35;
    insufficient_barrier = barrier_storage;
    --insufficient_barrier.communication_buffer_bytes;
    if (validate_internal_launch(
            barrier_rank_one, insufficient_barrier).code !=
        CoreRuntimeStatusCode::kInsufficientStorage)
        return 36;

    auto two_rank_dispatch = valid_tiling(
        OperationKind::kDispatch, ElementKind::kBFloat16, 0, 2);
    auto two_rank_combine = valid_tiling(
        OperationKind::kCombine, ElementKind::kBFloat16, 1, 2);
    if (validate_internal_launch(
            two_rank_dispatch,
            required_core_launch_storage(two_rank_dispatch)).code !=
            CoreRuntimeStatusCode::kUnsupportedTopology ||
        validate_internal_launch(
            two_rank_combine,
            required_core_launch_storage(two_rank_combine)).code !=
            CoreRuntimeStatusCode::kUnsupportedTopology)
        return 37;

    auto dispatch_tiling = valid_tiling(OperationKind::kDispatch);
    if (dispatch_tiling.struct_size != sizeof(CoreTiling))
        return 1;
    const auto storage = required_core_launch_storage(dispatch_tiling);
    if (!validate_internal_launch(dispatch_tiling, storage).ok())
        return 2;

    auto invalid_topology = dispatch_tiling;
    invalid_topology.topology.world_size = 2;
    if (validate_internal_launch(invalid_topology, storage).code !=
        CoreRuntimeStatusCode::kUnsupportedTopology)
        return 3;
    invalid_topology = dispatch_tiling;
    invalid_topology.topology.scale_up_size = 2;
    if (validate_internal_launch(invalid_topology, storage).code !=
        CoreRuntimeStatusCode::kUnsupportedTopology)
        return 4;
    invalid_topology = dispatch_tiling;
    invalid_topology.topology.scale_out_size = 2;
    if (validate_internal_launch(invalid_topology, storage).code !=
        CoreRuntimeStatusCode::kUnsupportedTopology)
        return 5;
    invalid_topology = dispatch_tiling;
    invalid_topology.topology.world_rank = 1;
    if (validate_internal_launch(invalid_topology, storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 6;

    auto unsupported = dispatch_tiling;
    unsupported.mode_flags |= mode_bit(CoreMode::kAsyncEvent);
    if (validate_internal_launch(unsupported, storage).code !=
        CoreRuntimeStatusCode::kUnsupportedMode)
        return 7;
    unsupported = dispatch_tiling;
    unsupported.mode_flags |= mode_bit(CoreMode::kCpuSync) |
                              mode_bit(CoreMode::kHybrid) |
                              mode_bit(CoreMode::kPipeline) |
                              mode_bit(CoreMode::kEngram);
    if (validate_internal_launch(unsupported, storage).code !=
        CoreRuntimeStatusCode::kUnsupportedMode)
        return 8;
    if (validate_internal_launch(
            valid_tiling(OperationKind::kDispatch,
                         ElementKind::kFloat8E4M3),
            storage).code != CoreRuntimeStatusCode::kUnsupportedMode ||
        validate_internal_launch(
            valid_tiling(OperationKind::kCombine,
                         ElementKind::kFloat8E4M3),
            storage).code != CoreRuntimeStatusCode::kUnsupportedMode)
        return 9;

    auto malformed = dispatch_tiling;
    malformed.workspace_bytes = 0;
    if (validate_internal_launch(
            malformed, required_core_launch_storage(malformed)).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 10;
    malformed = dispatch_tiling;
    ++malformed.token_layout.hidden_offset;
    if (validate_internal_launch(malformed, storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 11;
    malformed = dispatch_tiling;
    malformed.launch.num_threads = 0;
    if (validate_internal_launch(malformed, storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 12;
    malformed = dispatch_tiling;
    malformed.transport_context.topology.world_rank = 1;
    if (validate_internal_launch(malformed, storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 13;

    auto insufficient = storage;
    --insufficient.workspace_bytes;
    if (validate_internal_launch(dispatch_tiling, insufficient).code !=
        CoreRuntimeStatusCode::kInsufficientStorage)
        return 14;
    insufficient = storage;
    --insufficient.communication_buffer_bytes;
    if (validate_internal_launch(dispatch_tiling, insufficient).code !=
        CoreRuntimeStatusCode::kInsufficientStorage)
        return 15;

    alignas(kAscendElasticAlignment) std::uint8_t bytes[128] = {};
    alignas(kAscendElasticAlignment) std::int64_t indices[8] = {};
    alignas(kAscendElasticAlignment) float weights[8] = {};
    alignas(kAscendElasticAlignment) std::int32_t integers[64] = {};
    DispatchArguments dispatch{};
    dispatch.x = bytes;
    dispatch.topk_indices = indices;
    dispatch.topk_weights = weights;
    dispatch.communication_buffer = bytes;
    dispatch.workspace = bytes;
    dispatch.recv_x = bytes;
    dispatch.recv_topk_indices = indices;
    dispatch.recv_topk_weights = weights;
    dispatch.prefix_per_rank = integers;
    dispatch.prefix_per_expert = integers;
    dispatch.unaligned_per_expert = integers;
    dispatch.destination_slots = integers;
    dispatch.source_metadata = integers;

    reset_launches();
    if (!launch_internal_dispatch(
             dispatch, dispatch_tiling, storage, nullptr).ok() ||
        !trace_is(kDispatchLaunch, kDispatchEpilogueLaunch))
        return 16;
    reset_launches(kDispatchLaunch, 5);
    auto status = launch_internal_dispatch(
        dispatch, dispatch_tiling, storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kLaunchFailure ||
        status.backend_code != 5 || !trace_is(kDispatchLaunch))
        return 17;
    reset_launches(kDispatchEpilogueLaunch, 7);
    status = launch_internal_dispatch(dispatch, dispatch_tiling, storage,
                                      nullptr);
    if (status.code != CoreRuntimeStatusCode::kLaunchFailure ||
        status.backend_code != 7 ||
        !trace_is(kDispatchLaunch, kDispatchEpilogueLaunch))
        return 18;
    reset_launches();
    auto misaligned_dispatch = dispatch;
    misaligned_dispatch.workspace = bytes + 1;
    if (launch_internal_dispatch(
            misaligned_dispatch, dispatch_tiling, storage, nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 19;

    auto combine_tiling = valid_tiling(OperationKind::kCombine);
    const auto combine_storage = required_core_launch_storage(combine_tiling);
    CombineArguments combine{};
    combine.x = bytes;
    combine.topk_weights = weights;
    combine.source_metadata = integers;
    combine.combined_topk_indices = indices;
    combine.prefix_per_rank = integers;
    combine.communication_buffer = bytes;
    combine.workspace = bytes;
    combine.combined_x = bytes;
    combine.combined_topk_weights = weights;
    reset_launches();
    if (!launch_internal_combine(
             combine, combine_tiling, combine_storage, nullptr).ok() ||
        !trace_is(kCombineLaunch, kCombineEpilogueLaunch))
        return 20;
    reset_launches(kCombineLaunch, 8);
    status = launch_internal_combine(
        combine, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kLaunchFailure ||
        status.backend_code != 8 || !trace_is(kCombineLaunch))
        return 21;
    reset_launches(kCombineEpilogueLaunch, 9);
    status = launch_internal_combine(
        combine, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kLaunchFailure ||
        status.backend_code != 9 ||
        !trace_is(kCombineLaunch, kCombineEpilogueLaunch))
        return 22;

    auto barrier_tiling = valid_barrier_tiling(0);
    BarrierArguments barrier{bytes, 7, 1000000000ULL};
    reset_launches();
    if (launch_internal_barrier(
            barrier, barrier_tiling,
            required_core_launch_storage(barrier_tiling), nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 38;
    reset_launches();
    auto zero_generation = barrier;
    zero_generation.generation = 0;
    if (launch_internal_barrier(
            zero_generation, barrier_tiling,
            required_core_launch_storage(barrier_tiling),
            reinterpret_cast<void*>(0x6161)).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 41;
    reset_launches();
    auto zero_timeout = barrier;
    zero_timeout.timeout_cycles = 0;
    if (launch_internal_barrier(
            zero_timeout, barrier_tiling,
            required_core_launch_storage(barrier_tiling),
            reinterpret_cast<void*>(0x6161)).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 42;
    reset_launches();
    if (!launch_internal_barrier(
             barrier, barrier_tiling,
             required_core_launch_storage(barrier_tiling),
             reinterpret_cast<void*>(0x6161)).ok() ||
        !trace_is(kBarrierLaunch) || barrier_generation != 7)
        return 23;
    reset_launches();
    barrier.workspace = bytes + 1;
    if (launch_internal_barrier(
            barrier, barrier_tiling,
            required_core_launch_storage(barrier_tiling),
            reinterpret_cast<void*>(0x6161)).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 24;

    return 0;
}
