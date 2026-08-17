#include <cstdint>
#include <cstring>

#include "csrc/backends/ascend/elastic/combine_state.hpp"
#include "csrc/backends/ascend/elastic/dispatch_state.hpp"
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
std::uint64_t dispatch_generation = 0;
std::uint64_t dispatch_timeout_cycles = 0;
int dispatch_world_rank = -1;
std::uint64_t dispatch_receive_offset = 0;
std::uint64_t dispatch_staging_offset = 0;
std::uint64_t combine_generation = 0;
std::uint64_t combine_timeout_cycles = 0;
std::uint64_t combine_num_source_rows = 0;
std::uint64_t combine_num_input_rows = 0;
std::uintptr_t combine_local_window_base = 0;
int combine_world_rank = -1;
std::uint64_t combine_receive_offset = 0;
std::uint64_t combine_staging_offset = 0;

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
    int world_rank = 0, int world_size = 1, CoreModeFlags mode_flags = 0,
    std::uint64_t num_tokens = 4) {
    CoreTilingInput input{};
    input.operation = operation;
    input.element_kind = element_kind;
    input.mode_flags = mode_flags;
    input.num_tokens = operation == OperationKind::kBarrier ? 0 : num_tokens;
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

void export_transport(CoreTiling* tiling) {
    tiling->transport_context.capabilities =
        transport::capability_bit(
            transport::TransportCapability::kSymmetricWindow) |
        transport::capability_bit(
            transport::TransportCapability::kDevicePut) |
        transport::capability_bit(
            transport::TransportCapability::kDevicePutValue) |
        transport::capability_bit(
            transport::TransportCapability::kRemoteSignal) |
        transport::capability_bit(
            transport::TransportCapability::kSystemMemoryOrdering) |
        transport::capability_bit(
            transport::TransportCapability::kDeviceBarrier) |
        transport::capability_bit(
            transport::TransportCapability::kScaleUpTeam);
    tiling->transport_context.local_window_base = 0x200000;
    tiling->transport_context.peer_address_table = 0x300000;
    tiling->transport_context.channel_table = 0x400000;
    tiling->transport_context.backend_context = 0x500000;
}

CoreTiling context_derived_combine_tiling(int world_rank) {
    auto context = transport::make_device_transport_context();
    context.topology.world_rank = world_rank;
    context.topology.world_size = 2;
    context.topology.scale_up_rank = world_rank;
    context.topology.scale_up_size = 2;
    context.topology.scale_out_rank = 0;
    context.topology.scale_out_size = 1;

    CoreTilingInput input{};
    input.operation = OperationKind::kCombine;
    input.element_kind = ElementKind::kBFloat16;
    input.num_tokens = 4;
    input.hidden = 64;
    input.num_experts = 4;
    input.num_topk = 2;
    input.expert_alignment = 4;
    input.num_max_tokens_per_rank = 8;
    input.topology = core_topology_from_transport(context.topology);

    CoreTiling tiling{};
    if (!build_core_tiling(input, &tiling).ok())
        return {};
    tiling.transport_context = context;
    export_transport(&tiling);
    return tiling;
}

bool combine_context_topology_contract_matches() {
    const auto rank_zero = context_derived_combine_tiling(0);
    const auto rank_one = context_derived_combine_tiling(1);
    if (rank_zero.topology.world_rank != 0 ||
        rank_zero.topology.scale_up_rank != 0 ||
        rank_one.topology.world_rank != 1 ||
        rank_one.topology.scale_up_rank != 1 ||
        !validate_internal_launch(
            rank_zero, required_core_launch_storage(rank_zero)).ok() ||
        !validate_internal_launch(
            rank_one, required_core_launch_storage(rank_one)).ok())
        return false;

    auto mismatch = rank_one;
    mismatch.transport_context.topology.world_rank = 0;
    return validate_internal_launch(
               mismatch, required_core_launch_storage(mismatch)).code ==
           CoreRuntimeStatusCode::kInvalidArgument;
}

bool cached_mixed_rank_prefix_fixture_matches() {
    constexpr int world_rank = 0;
    constexpr int world_size = 2;
    constexpr std::uint64_t num_experts = 4;
    constexpr std::uint64_t num_topk = 2;
    constexpr std::uint64_t expert_alignment = 4;
    constexpr std::uint64_t num_records = 2;
    constexpr std::int64_t records[num_records * num_topk] = {
        0, 2,
        1, 3,
    };
    constexpr std::int32_t cached_counts[num_experts] = {1, 1, 0, 0};
    constexpr std::int32_t cached_prefix[num_experts + 1] = {0, 4, 8, 8, 8};
    const std::uint64_t num_local_experts =
        num_experts / static_cast<std::uint64_t>(world_size);
    const std::uint64_t first_local_expert =
        static_cast<std::uint64_t>(world_rank) * num_local_experts;
    std::int32_t derived_prefix = 0;
    for (std::uint64_t expert = 0; expert < num_experts; ++expert) {
        std::int32_t actual_count = 0;
        if (is_dispatch_expert_local(
                static_cast<std::int64_t>(expert), first_local_expert,
                num_local_experts)) {
            for (std::uint64_t record = 0; record < num_records; ++record) {
                for (std::uint64_t lane = 0; lane < num_topk; ++lane) {
                    if (records[record * num_topk + lane] ==
                        static_cast<std::int64_t>(expert))
                        ++actual_count;
                }
            }
        }
        if (actual_count != cached_counts[expert] ||
            derived_prefix != cached_prefix[expert])
            return false;
        derived_prefix +=
            ((actual_count + static_cast<std::int32_t>(expert_alignment) - 1) /
             static_cast<std::int32_t>(expert_alignment)) *
            static_cast<std::int32_t>(expert_alignment);
    }
    return derived_prefix == cached_prefix[num_experts];
}

bool padded_dispatch_capacity_fixture_matches() {
    CoreTilingInput input{};
    input.operation = OperationKind::kDispatch;
    input.element_kind = ElementKind::kBFloat16;
    input.mode_flags = mode_bit(CoreMode::kExpanded) |
                       mode_bit(CoreMode::kZeroPadding);
    input.num_tokens = 4;
    input.hidden = 64;
    input.num_experts = 4;
    input.num_topk = 1;
    input.expert_alignment = 4;
    input.num_max_tokens_per_rank = 4;
    input.topology.world_rank = 0;
    input.topology.world_size = 2;
    input.topology.scale_up_rank = 0;
    input.topology.scale_up_size = 2;
    input.topology.scale_out_rank = 0;
    input.topology.scale_out_size = 1;

    CoreTiling tiling{};
    if (!build_core_tiling(input, &tiling).ok())
        return false;

    constexpr std::uint64_t raw_lane_count = 4 * 2 * 1;
    constexpr std::uint64_t aligned_expert_zero = 8;
    constexpr std::uint64_t aligned_expert_one = 4;
    constexpr std::uint64_t required_rows =
        aligned_expert_zero + aligned_expert_one;
    return raw_lane_count == 8 && required_rows == 12 &&
           required_rows > raw_lane_count &&
           tiling.dispatch_output_capacity == 16;
}

bool combine_state_helpers_match() {
    constexpr std::uint64_t generation = 13;
    if (!is_clean_combine_transport_completion(
            generation, generation, transport::kTransportCommandAbiVersion,
            generation, transport::DeviceTransportError::kNone) ||
        is_clean_combine_transport_completion(
            generation, generation - 1,
            transport::kTransportCommandAbiVersion, generation,
            transport::DeviceTransportError::kNone) ||
        is_clean_combine_transport_completion(
            generation, generation,
            transport::kTransportCommandAbiVersion + 1, generation,
            transport::DeviceTransportError::kNone) ||
        is_clean_combine_transport_completion(
            generation, generation,
            transport::kTransportCommandAbiVersion, generation - 1,
            transport::DeviceTransportError::kNone) ||
        is_clean_combine_transport_completion(
            generation, generation,
            transport::kTransportCommandAbiVersion, generation,
            transport::DeviceTransportError::kCompletionFailure))
        return false;

    constexpr std::uintptr_t base = 0x100000;
    constexpr std::uint64_t receive_offset = 0x2000;
    constexpr std::uint64_t receive_shard_bytes = 0x400;
    constexpr std::uint64_t staging_offset = 0x3000;
    constexpr std::uint64_t staging_shard_bytes = 0x800;
    std::uint16_t capacity_output = 0x5a5a;
    if (is_valid_combine_token_extent(9, 8))
        capacity_output = 0;
    return is_valid_combine_token_extent(8, 8) &&
           !is_valid_combine_token_extent(9, 8) &&
           capacity_output == 0x5a5a &&
           combine_receive_shard_address(
               base, receive_offset, 0, receive_shard_bytes) == 0x102000 &&
           combine_receive_shard_address(
               base, receive_offset, 1, receive_shard_bytes) == 0x102400 &&
           combine_staging_shard_address(
               base, staging_offset, 0, staging_shard_bytes) == 0x103000 &&
           combine_staging_shard_address(
               base, staging_offset, 1, staging_shard_bytes) == 0x103800 &&
           is_valid_combine_origin_token(3, 4, 8) &&
           !is_valid_combine_origin_token(4, 4, 8) &&
           !is_valid_combine_origin_token(8, 16, 8) &&
           !is_valid_combine_origin_token(-1, 4, 8);
}

bool combine_runner_allocation_contract_matches() {
    CoreTilingInput input{};
    input.operation = OperationKind::kDispatch;
    input.element_kind = ElementKind::kBFloat16;
    input.num_tokens = 7;
    input.hidden = 64;
    input.num_experts = 4;
    input.num_topk = 2;
    input.expert_alignment = 4;
    input.num_max_tokens_per_rank = 16;
    CoreTiling dispatch{};
    if (!build_core_tiling(input, &dispatch).ok())
        return false;

    input.operation = OperationKind::kCombine;
    input.topology.world_size = 2;
    input.topology.scale_up_size = 2;
    CoreTiling combine{};
    if (!build_core_tiling(input, &combine).ok())
        return false;

    const auto allocation = merge_core_launch_storage(
        required_core_launch_storage(dispatch),
        required_core_launch_storage(combine));
    return dispatch.communication_buffer_bytes == 3584 &&
           dispatch.workspace_bytes == 416 &&
           combine.communication_buffer_bytes == 2097152 &&
           combine.workspace_bytes == 416 &&
           allocation.communication_buffer_bytes == 2097152 &&
           allocation.workspace_bytes == 416 &&
           combine.topology.world_size == 2;
}

}  // namespace

extern "C" int deep_ep_ascend_launch_barrier(
    BarrierArguments arguments, CoreTiling, void*) {
    barrier_generation = arguments.generation;
    return record_launch(kBarrierLaunch);
}

extern "C" int deep_ep_ascend_launch_dispatch(
    DispatchArguments arguments, CoreTiling tiling, void*) {
    dispatch_generation = arguments.generation;
    dispatch_timeout_cycles = arguments.timeout_cycles;
    dispatch_world_rank = tiling.transport_context.topology.world_rank;
    dispatch_receive_offset =
        tiling.symmetric_window_layout.dispatch_receive_offset;
    dispatch_staging_offset =
        tiling.symmetric_window_layout.dispatch_staging_offset;
    return record_launch(kDispatchLaunch);
}

extern "C" int deep_ep_ascend_launch_dispatch_epilogue(
    DispatchArguments, CoreTiling, void*) {
    return record_launch(kDispatchEpilogueLaunch);
}

extern "C" int deep_ep_ascend_launch_combine(
    CombineArguments arguments, CoreTiling tiling, void*) {
    combine_generation = arguments.generation;
    combine_timeout_cycles = arguments.timeout_cycles;
    combine_num_source_rows = arguments.num_source_rows;
    combine_num_input_rows = arguments.num_input_rows;
    combine_local_window_base = arguments.local_window_base;
    combine_world_rank = tiling.transport_context.topology.world_rank;
    combine_receive_offset =
        tiling.symmetric_window_layout.combine_receive_offset;
    combine_staging_offset =
        tiling.symmetric_window_layout.combine_staging_offset;
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
    malformed_barrier = barrier_rank_zero;
    malformed_barrier.symmetric_window_layout.combine_receive_offset +=
        kAscendElasticAlignment;
    if (validate_internal_launch(malformed_barrier, barrier_storage).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 46;

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
    if (validate_internal_launch(
            two_rank_dispatch,
            required_core_launch_storage(two_rank_dispatch)).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 43;
    export_transport(&two_rank_dispatch);
    auto two_rank_combine = valid_tiling(
        OperationKind::kCombine, ElementKind::kBFloat16, 1, 2);
    export_transport(&two_rank_combine);
    if (!validate_internal_launch(
            two_rank_dispatch,
            required_core_launch_storage(two_rank_dispatch)).ok() ||
        !validate_internal_launch(
            two_rank_combine,
            required_core_launch_storage(two_rank_combine)).ok())
        return 37;

    auto single_rank_combine = valid_tiling(OperationKind::kCombine);
    if (validate_internal_launch(
            single_rank_combine,
            required_core_launch_storage(single_rank_combine)).code !=
        CoreRuntimeStatusCode::kUnsupportedTopology)
        return 48;

    auto dispatch_tiling = valid_tiling(OperationKind::kDispatch);
    if (dispatch_tiling.struct_size != sizeof(CoreTiling))
        return 1;
    const auto storage = required_core_launch_storage(dispatch_tiling);
    if (!validate_internal_launch(dispatch_tiling, storage).ok())
        return 2;

    CoreTilingInput excessive_input{};
    excessive_input.operation = OperationKind::kDispatch;
    excessive_input.element_kind = ElementKind::kBFloat16;
    excessive_input.num_tokens = 9;
    excessive_input.hidden = 64;
    excessive_input.num_experts = 4;
    excessive_input.num_topk = 2;
    excessive_input.expert_alignment = 4;
    excessive_input.num_max_tokens_per_rank = 8;
    CoreTiling excessive_tiling{};
    if (build_core_tiling(excessive_input, &excessive_tiling).code !=
        TilingStatusCode::kInvalidArgument)
        return 44;
    auto excessive_combine_input = excessive_input;
    excessive_combine_input.operation = OperationKind::kCombine;
    excessive_combine_input.topology.world_size = 2;
    excessive_combine_input.topology.scale_up_size = 2;
    if (build_core_tiling(
            excessive_combine_input, &excessive_tiling).code !=
        TilingStatusCode::kInvalidArgument)
        return 56;
    auto excessive_descriptor = dispatch_tiling;
    excessive_descriptor.num_tokens =
        excessive_descriptor.num_max_tokens_per_rank + 1;
    if (validate_internal_launch(
            excessive_descriptor,
            required_core_launch_storage(excessive_descriptor)).code !=
        CoreRuntimeStatusCode::kInvalidArgument)
        return 45;

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
            storage).code != CoreRuntimeStatusCode::kUnsupportedTopology)
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
    dispatch.generation = 11;
    dispatch.timeout_cycles = 101;

    reset_launches();
    if (!launch_internal_dispatch(
             dispatch, dispatch_tiling, storage, nullptr).ok() ||
        !trace_is(kDispatchLaunch) || dispatch_generation != 11 ||
        dispatch_timeout_cycles != 101)
        return 16;
    reset_launches(kDispatchLaunch, 5);
    auto status = launch_internal_dispatch(
        dispatch, dispatch_tiling, storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kLaunchFailure ||
        status.backend_code != 5 || !trace_is(kDispatchLaunch))
        return 17;
    reset_launches();
    auto zero_dispatch_generation = dispatch;
    zero_dispatch_generation.generation = 0;
    status = launch_internal_dispatch(
        zero_dispatch_generation, dispatch_tiling, storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 18;
    reset_launches();
    auto zero_dispatch_timeout = dispatch;
    zero_dispatch_timeout.timeout_cycles = 0;
    status = launch_internal_dispatch(
        zero_dispatch_timeout, dispatch_tiling, storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 41;
    reset_launches();
    auto rank_one_dispatch = valid_tiling(
        OperationKind::kDispatch, ElementKind::kBFloat16, 1, 2);
    export_transport(&rank_one_dispatch);
    if (!launch_internal_dispatch(
            dispatch, rank_one_dispatch,
            required_core_launch_storage(rank_one_dispatch), nullptr).ok() ||
        !trace_is(kDispatchLaunch) || dispatch_world_rank != 1 ||
        dispatch_receive_offset != rank_one_dispatch.symmetric_window_layout
                                       .dispatch_receive_offset ||
        dispatch_staging_offset != rank_one_dispatch.symmetric_window_layout
                                       .dispatch_staging_offset)
        return 42;
    reset_launches();
    auto misaligned_dispatch = dispatch;
    misaligned_dispatch.workspace = bytes + 1;
    if (launch_internal_dispatch(
            misaligned_dispatch, dispatch_tiling, storage, nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 19;

    auto combine_tiling = valid_tiling(
        OperationKind::kCombine, ElementKind::kBFloat16, 1, 2);
    export_transport(&combine_tiling);
    const auto combine_storage = required_core_launch_storage(combine_tiling);
    CombineArguments combine{};
    combine.x = bytes;
    combine.source_metadata = integers;
    combine.combined_topk_indices = indices;
    combine.prefix_per_rank = integers;
    combine.communication_buffer = bytes;
    combine.workspace = bytes;
    combine.combined_x = bytes;
    combine.generation = 13;
    combine.timeout_cycles = 103;
    combine.num_source_rows = 3;
    combine.num_input_rows = 3;
    combine.local_window_base =
        combine_tiling.transport_context.local_window_base;
    auto excessive_combine_tiling = combine_tiling;
    excessive_combine_tiling.num_tokens =
        excessive_combine_tiling.num_max_tokens_per_rank + 1;
    reset_launches();
    status = launch_internal_combine(
        combine, excessive_combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 57;
    reset_launches();
    if (!launch_internal_combine(
             combine, combine_tiling, combine_storage, nullptr).ok() ||
        !trace_is(kCombineLaunch) || combine_generation != 13 ||
        combine_timeout_cycles != 103 || combine_num_source_rows != 3 ||
        combine_num_input_rows != 3 ||
        combine_local_window_base !=
            combine_tiling.transport_context.local_window_base ||
        combine_world_rank != 1 ||
        combine_receive_offset !=
            combine_tiling.symmetric_window_layout.combine_receive_offset ||
        combine_staging_offset !=
            combine_tiling.symmetric_window_layout.combine_staging_offset)
        return 20;

    auto zero_token_combine_tiling = valid_tiling(
        OperationKind::kCombine, ElementKind::kBFloat16, 1, 2, 0, 0);
    export_transport(&zero_token_combine_tiling);
    const auto zero_token_combine_storage =
        required_core_launch_storage(zero_token_combine_tiling);
    auto zero_token_combine = combine;
    zero_token_combine.combined_topk_indices = nullptr;
    zero_token_combine.combined_x = nullptr;
    zero_token_combine.num_source_rows = 0;
    zero_token_combine.num_input_rows = 0;
    zero_token_combine.local_window_base =
        zero_token_combine_tiling.transport_context.local_window_base;
    reset_launches();
    if (!launch_internal_combine(
             zero_token_combine, zero_token_combine_tiling,
             zero_token_combine_storage, nullptr).ok() ||
        !trace_is(kCombineLaunch))
        return 61;

    reset_launches();
    auto zero_token_missing_prefix = zero_token_combine;
    zero_token_missing_prefix.prefix_per_rank = nullptr;
    if (launch_internal_combine(
            zero_token_missing_prefix, zero_token_combine_tiling,
            zero_token_combine_storage, nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 62;
    reset_launches();
    auto zero_token_missing_workspace = zero_token_combine;
    zero_token_missing_workspace.workspace = nullptr;
    if (launch_internal_combine(
            zero_token_missing_workspace, zero_token_combine_tiling,
            zero_token_combine_storage, nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 63;
    reset_launches();
    auto zero_token_missing_window = zero_token_combine;
    zero_token_missing_window.local_window_base = 0;
    if (launch_internal_combine(
            zero_token_missing_window, zero_token_combine_tiling,
            zero_token_combine_storage, nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 64;
    reset_launches();
    auto nonempty_missing_indices = combine;
    nonempty_missing_indices.combined_topk_indices = nullptr;
    if (launch_internal_combine(
            nonempty_missing_indices, combine_tiling, combine_storage,
            nullptr).code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 65;
    reset_launches();
    auto nonempty_missing_output = combine;
    nonempty_missing_output.combined_x = nullptr;
    if (launch_internal_combine(
            nonempty_missing_output, combine_tiling, combine_storage,
            nullptr).code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 66;

    reset_launches(kCombineLaunch, 8);
    status = launch_internal_combine(
        combine, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kLaunchFailure ||
        status.backend_code != 8 || !trace_is(kCombineLaunch))
        return 21;
    reset_launches();
    auto zero_combine_generation = combine;
    zero_combine_generation.generation = 0;
    status = launch_internal_combine(
        zero_combine_generation, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 22;
    reset_launches();
    auto zero_combine_timeout = combine;
    zero_combine_timeout.timeout_cycles = 0;
    status = launch_internal_combine(
        zero_combine_timeout, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 49;
    reset_launches();
    auto mismatched_combine_window = combine;
    mismatched_combine_window.local_window_base += kAscendElasticAlignment;
    status = launch_internal_combine(
        mismatched_combine_window, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 50;
    reset_launches();
    auto mismatched_combine_rows = combine;
    ++mismatched_combine_rows.num_input_rows;
    status = launch_internal_combine(
        mismatched_combine_rows, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 51;
    reset_launches();
    auto excessive_combine_rows = combine;
    excessive_combine_rows.num_source_rows =
        combine_tiling.num_max_tokens_per_rank * 2 + 1;
    excessive_combine_rows.num_input_rows =
        excessive_combine_rows.num_source_rows;
    status = launch_internal_combine(
        excessive_combine_rows, combine_tiling, combine_storage, nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 52;
    reset_launches();
    auto weighted_combine = combine;
    weighted_combine.topk_weights = weights;
    weighted_combine.combined_topk_weights = weights;
    if (!launch_internal_combine(
            weighted_combine, combine_tiling, combine_storage, nullptr).ok() ||
        !trace_is(kCombineLaunch))
        return 53;
    reset_launches();
    auto expanded_single_tiling = valid_tiling(
        OperationKind::kCombine, ElementKind::kBFloat16, 1, 2,
        mode_bit(CoreMode::kExpanded));
    export_transport(&expanded_single_tiling);
    auto expanded_weighted_combine = weighted_combine;
    expanded_weighted_combine.local_window_base =
        expanded_single_tiling.transport_context.local_window_base;
    status = launch_internal_combine(
        expanded_weighted_combine, expanded_single_tiling,
        required_core_launch_storage(expanded_single_tiling), nullptr);
    if (status.code != CoreRuntimeStatusCode::kInvalidArgument ||
        std::strcmp(
            status.message,
            "expanded combine weights require allow_multiple_reduction") != 0 ||
        launch_trace_size != 0)
        return 58;
    auto expanded_multiple_tiling = valid_tiling(
        OperationKind::kCombine, ElementKind::kBFloat16, 1, 2,
        mode_bit(CoreMode::kExpanded) |
            mode_bit(CoreMode::kAllowMultipleReduction));
    export_transport(&expanded_multiple_tiling);
    auto maximum_expanded_input = weighted_combine;
    maximum_expanded_input.local_window_base =
        expanded_multiple_tiling.transport_context.local_window_base;
    maximum_expanded_input.num_source_rows = 16;
    maximum_expanded_input.num_input_rows = 40;
    reset_launches();
    if (expanded_multiple_tiling.dispatch_output_capacity != 40 ||
        !launch_internal_combine(
            maximum_expanded_input, expanded_multiple_tiling,
            required_core_launch_storage(expanded_multiple_tiling), nullptr).ok() ||
        !trace_is(kCombineLaunch))
        return 59;
    ++maximum_expanded_input.num_input_rows;
    reset_launches();
    if (launch_internal_combine(
            maximum_expanded_input, expanded_multiple_tiling,
            required_core_launch_storage(expanded_multiple_tiling), nullptr).code !=
            CoreRuntimeStatusCode::kInvalidArgument ||
        launch_trace_size != 0)
        return 60;

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

    if (!cached_mixed_rank_prefix_fixture_matches())
        return 46;
    if (!padded_dispatch_capacity_fixture_matches())
        return 47;
    if (!combine_state_helpers_match())
        return 54;
    if (!combine_runner_allocation_contract_matches())
        return 58;
    if (!combine_context_topology_contract_matches())
        return 59;

    return 0;
}
