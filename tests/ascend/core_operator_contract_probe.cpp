#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/combine_parallel.hpp"
#include "csrc/backends/ascend/elastic/dispatch_parallel.hpp"
#include "csrc/backends/ascend/elastic/dispatch_early_route_plan.hpp"
#include "csrc/backends/ascend/elastic/dispatch_pipeline_config.hpp"
#include "csrc/backends/ascend/elastic/dispatch_token_fanout.hpp"
#include "csrc/backends/ascend/elastic/layout.hpp"
#include "csrc/backends/ascend/elastic/kernels.hpp"
#include "csrc/backends/ascend/elastic/tiling.hpp"
#include "csrc/backends/ascend/elastic/topk_grouping.hpp"
#include "csrc/backends/ascend/transport/sync_layout.hpp"

using namespace deep_ep::ascend::elastic;

static_assert(std::is_standard_layout_v<TokenLayout>);
static_assert(std::is_trivially_copyable_v<TokenLayout>);
static_assert(std::is_standard_layout_v<WorkspaceLayout>);
static_assert(std::is_trivially_copyable_v<WorkspaceLayout>);
static_assert(std::is_standard_layout_v<SymmetricWindowLayout>);
static_assert(std::is_trivially_copyable_v<SymmetricWindowLayout>);
static_assert(std::is_standard_layout_v<CoreTiling>);
static_assert(std::is_trivially_copyable_v<CoreTiling>);
static_assert(std::is_standard_layout_v<DispatchPipelineState>);
static_assert(std::is_trivially_copyable_v<DispatchPipelineState>);
static_assert(offsetof(DispatchPipelineState, release_completed_generation) ==
              64);
static_assert(offsetof(DispatchPipelineState, slots) == 128);
static_assert(offsetof(DispatchPipelineState, scalar_progress) == 384);
static_assert(offsetof(DispatchPipelineState, hidden_progress) == 4992);
static_assert(
    offsetof(DispatchPipelineState, release_completed_generation) / 64 !=
    offsetof(DispatchPipelineState, generation) / 64);

namespace {

CoreTilingInput valid_input() {
    CoreTilingInput input{};
    input.operation = OperationKind::kDispatch;
    input.element_kind = ElementKind::kBFloat16;
    input.num_tokens = 7;
    input.hidden = 64;
    input.num_experts = 4;
    input.num_topk = 2;
    input.expert_alignment = 4;
    input.num_max_tokens_per_rank = 16;
    input.data_num_blocks = 1;
    input.topology.world_size = 1;
    input.topology.scale_up_size = 1;
    input.topology.scale_out_size = 1;
    input.topology.world_rank = 0;
    input.topology.scale_up_rank = 0;
    input.topology.scale_out_rank = 0;
    return input;
}

bool has_tiling_error(
    const TilingStatus& status, const char* expected_message) {
    return status.code == TilingStatusCode::kInvalidArgument &&
        status.message != nullptr &&
        std::strcmp(status.message, expected_message) == 0;
}

int direct_device_index_boundary_contract() {
    constexpr std::uint64_t kLimit = 0x7fffffffULL;
    constexpr const char* kShapeError =
        "shape exceeds 32-bit device index range";
    constexpr const char* kCapacityError =
        "dispatch output exceeds 32-bit device index range";
    CoreTiling tiling{};

    auto input = valid_input();
    input.num_experts = 1;
    input.num_topk = 1;
    input.expert_alignment = 1;
    input.num_max_tokens_per_rank = kLimit;
    auto status = build_core_tiling(input, &tiling);
    if (!status.ok() || tiling.dispatch_output_capacity != kLimit)
        return 79;

    input = valid_input();
    input.hidden = kLimit + 1;
    if (!has_tiling_error(build_core_tiling(input, &tiling), kShapeError))
        return 80;

    input = valid_input();
    input.num_experts = kLimit + 1;
    if (!has_tiling_error(build_core_tiling(input, &tiling), kShapeError))
        return 81;

    input = valid_input();
    input.num_max_tokens_per_rank = kLimit + 1;
    if (!has_tiling_error(build_core_tiling(input, &tiling), kShapeError))
        return 82;

    input = valid_input();
    input.num_tokens = kLimit + 1;
    input.num_max_tokens_per_rank = kLimit + 1;
    if (!has_tiling_error(build_core_tiling(input, &tiling), kShapeError))
        return 83;

    input = valid_input();
    input.num_experts = kLimit + 1;
    input.num_topk = kLimit + 1;
    input.expert_alignment = 1;
    if (!has_tiling_error(build_core_tiling(input, &tiling), kShapeError))
        return 84;

    input = valid_input();
    input.num_experts = 4;
    input.expert_alignment = 1;
    input.num_max_tokens_per_rank = 536870912ULL;
    input.topology.world_size = 2;
    input.topology.scale_up_size = 2;
    if (!has_tiling_error(build_core_tiling(input, &tiling), kCapacityError))
        return 85;

    return 0;
}

bool aligned(std::uint64_t value) {
    return value % kAscendElasticAlignment == 0;
}

bool topk_grouping_reference_contract() {
    {
        const std::int32_t keys[] = {0, 0, 1, 1, 5, -1};
        const std::int32_t slots[] = {7, -1, 3, -1, 9, -1};
        const auto result =
            group_combine_contributors_reference<6>(keys, slots, 6);
        if (result.contributor_count != 3 ||
            result.entries[0].contributor_rank != 0 ||
            result.entries[0].contribution_lane != 0 ||
            result.entries[0].receive_slot != 7 ||
            result.entries[1].contributor_rank != 1 ||
            result.entries[1].contribution_lane != 2 ||
            result.entries[1].receive_slot != 3 ||
            result.entries[2].contributor_rank != 5 ||
            result.entries[2].contribution_lane != 4 ||
            result.entries[2].receive_slot != 9)
            return false;
        const std::int32_t expected_slots[] = {7, 7, 3, 3, 9, -1};
        for (std::uint32_t lane = 0; lane < 6; ++lane)
            if (result.resolved_slots[lane] != expected_slots[lane])
                return false;
    }
    {
        const std::int32_t keys[] = {5, 2, 5, -1, 2, 5};
        const std::int32_t slots[] = {9, 3, -1, -1, -1, -1};
        const auto result =
            group_combine_contributors_reference<6>(keys, slots, 6);
        if (result.contributor_count != 2 ||
            result.entries[0].contributor_rank != 2 ||
            result.entries[0].contribution_lane != 1 ||
            result.entries[0].receive_slot != 3 ||
            result.entries[1].contributor_rank != 5 ||
            result.entries[1].contribution_lane != 0 ||
            result.entries[1].receive_slot != 9)
            return false;
        const std::int32_t expected_slots[] = {9, 3, 9, -1, 3, 9};
        for (std::uint32_t lane = 0; lane < 6; ++lane)
            if (result.resolved_slots[lane] != expected_slots[lane])
                return false;
    }
    {
        const std::int32_t keys[] = {4, 4, -1, 7};
        const std::int32_t slots[] = {-1, 8, -1, 6};
        const auto result =
            group_combine_contributors_reference<4>(keys, slots, 4);
        if (result.contributor_count != 1 ||
            result.entries[0].contributor_rank != 7 ||
            result.entries[0].contribution_lane != 3 ||
            result.entries[0].receive_slot != 6)
            return false;
        const std::int32_t expected_slots[] = {-1, -1, -1, 6};
        for (std::uint32_t lane = 0; lane < 4; ++lane)
            if (result.resolved_slots[lane] != expected_slots[lane])
                return false;
    }
    {
        std::int32_t keys[33]{};
        std::int32_t slots[33]{};
        for (std::uint32_t lane = 0; lane < 33; ++lane) {
            keys[lane] = static_cast<std::int32_t>(lane);
            slots[lane] = static_cast<std::int32_t>(lane + 10);
        }
        const auto result =
            group_combine_contributors_reference<33>(keys, slots, 33);
        if (result.contributor_count != 33 ||
            result.entries[0].contributor_rank != 0 ||
            result.entries[0].receive_slot != 10 ||
            result.entries[32].contributor_rank != 32 ||
            result.entries[32].receive_slot != 42 ||
            result.resolved_slots[32] != 42)
            return false;
    }
    {
        const std::int32_t keys[] = {3, 99, 99, 99};
        const std::int32_t slots[] = {11, 12, 13, 14};
        const auto result =
            group_combine_contributors_reference<4>(keys, slots, 1);
        if (result.contributor_count != 1 ||
            result.entries[0].contributor_rank != 3 ||
            result.entries[0].contribution_lane != 0 ||
            result.resolved_slots[0] != 11 ||
            result.resolved_slots[1] != -1 ||
            result.resolved_slots[3] != -1)
            return false;
    }
    return true;
}

}  // namespace

int main() {
    static_assert(
        deep_ep::ascend::transport::sync_layout::
            kDispatchRouteReadySignalIndex == 2);
    std::uint64_t canonical_source_count = 99;
    if (!handoff_dispatch_route_source_count(
            true, 5, 5, 16, &canonical_source_count) ||
        canonical_source_count != 5)
        return 135;
    canonical_source_count = 99;
    if (handoff_dispatch_route_source_count(
            true, 7, 5, 16, &canonical_source_count) ||
        canonical_source_count != 99 ||
        handoff_dispatch_route_source_count(
            true, 17, 17, 16, &canonical_source_count) ||
        canonical_source_count != 99 ||
        !handoff_dispatch_route_source_count(
            false, 7, 5, 16, &canonical_source_count) ||
        canonical_source_count != 5)
        return 136;
    DispatchEarlyRoutePlanConfig early_route_config{};
    if (select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, false,
            256, 8, 8, &early_route_config) !=
                DispatchEarlyRoutePlanConfigStatus::kEnabled ||
        !early_route_config.enabled)
        return 127;
    for (const char* value : {static_cast<const char*>(nullptr), "0"}) {
        if (select_dispatch_early_route_plan_config(
                value, true, true, false, false, false, false, false,
                256, 8, 8, &early_route_config) !=
                    DispatchEarlyRoutePlanConfigStatus::kDisabled ||
            early_route_config.enabled)
            return 128;
    }
    for (const char* value : {"", "2", "true", "01"}) {
        if (select_dispatch_early_route_plan_config(
                value, true, true, false, false, false, false, false,
                256, 8, 8, &early_route_config) !=
                    DispatchEarlyRoutePlanConfigStatus::kInvalid)
            return 129;
    }
    const DispatchEarlyRoutePlanConfigStatus early_route_disabled_cases[] = {
        select_dispatch_early_route_plan_config(
            "1", false, true, false, false, false, false, false,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, false, false, false, false, false, false,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, true, false, false, false, false,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, true, false, false, false,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, true, false, false,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, true, false,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, true,
            256, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, false,
            257, 8, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, false,
            256, 9, 8, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, false,
            256, 8, 9, &early_route_config),
        select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, false,
            255, 8, 8, &early_route_config),
    };
    for (const auto status : early_route_disabled_cases) {
        if (status != DispatchEarlyRoutePlanConfigStatus::kDisabled ||
            early_route_config.enabled)
            return 130;
    }
    if (select_dispatch_early_route_plan_config(
            "1", true, true, false, false, false, false, false,
            256, 8, 8, nullptr) !=
                DispatchEarlyRoutePlanConfigStatus::kInvalid)
        return 131;
    {
        constexpr std::int64_t topk[] = {
            0, 1, 4, -1,
            0, 0, 5, 7,
            -1, 3, 3, 3,
        };
        std::uint64_t rank_counts[2]{};
        std::uint64_t expert_counts[8]{};
        for (std::uint32_t token = 0; token < 3; ++token) {
            bool selected_ranks[2]{};
            for (std::uint32_t lane = 0; lane < 4; ++lane) {
                const auto coordinate = dispatch_route_plan_coordinate(
                    topk[token * 4 + lane], 8, 2);
                if (!coordinate.valid)
                    continue;
                selected_ranks[coordinate.destination_rank] = true;
                ++expert_counts[
                    coordinate.destination_rank * 4 +
                    coordinate.local_expert];
            }
            for (std::uint32_t rank = 0; rank < 2; ++rank)
                rank_counts[rank] += selected_ranks[rank] ? 1 : 0;
        }
        constexpr std::uint64_t expected_rank_counts[] = {3, 2};
        constexpr std::uint64_t expected_expert_counts[] = {
            3, 1, 0, 3, 1, 1, 0, 1,
        };
        for (std::uint32_t rank = 0; rank < 2; ++rank)
            if (rank_counts[rank] != expected_rank_counts[rank])
                return 132;
        for (std::uint32_t expert = 0; expert < 8; ++expert)
            if (expert_counts[expert] != expected_expert_counts[expert])
                return 133;
        if (dispatch_route_plan_coordinate(-1, 8, 2).valid ||
            dispatch_route_plan_coordinate(8, 8, 2).valid ||
            dispatch_route_plan_coordinate(0, 0, 2).valid ||
            dispatch_route_plan_coordinate(0, 8, 0).valid ||
            dispatch_route_plan_coordinate(0, 7, 2).valid)
            return 134;
    }

    DispatchTokenFanoutConfig token_fanout_config{};
    if (select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 8, 7168, &token_fanout_config) !=
                DispatchTokenFanoutConfigStatus::kEnabled ||
        !token_fanout_config.enabled ||
        token_fanout_config.vector_bytes != 7168)
        return 118;
    if (select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 8, 7184, &token_fanout_config) !=
                DispatchTokenFanoutConfigStatus::kEnabled ||
        !token_fanout_config.enabled ||
        token_fanout_config.vector_bytes != 7168)
        return 126;
    if (select_dispatch_token_fanout_config(
            nullptr, true, true, false, false, false, false, false,
            8, 8, 7168, &token_fanout_config) !=
                DispatchTokenFanoutConfigStatus::kEnabled ||
        !token_fanout_config.enabled ||
        token_fanout_config.vector_bytes != 7168)
        return 119;
    for (const char* value : {"0"}) {
        if (select_dispatch_token_fanout_config(
                value, true, true, false, false, false, false, false,
                8, 8, 7168, &token_fanout_config) !=
                    DispatchTokenFanoutConfigStatus::kDisabled ||
            token_fanout_config.enabled ||
            token_fanout_config.vector_bytes != 0)
            return 119;
    }
    for (const char* value : {"", "2", "true", "01"}) {
        if (select_dispatch_token_fanout_config(
                value, true, true, false, false, false, false, false,
                8, 8, 7168, &token_fanout_config) !=
                    DispatchTokenFanoutConfigStatus::kInvalid)
            return 120;
    }
    const DispatchTokenFanoutConfigStatus token_fanout_disabled_cases[] = {
        select_dispatch_token_fanout_config(
            "1", false, true, false, false, false, false, false,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, false, false, false, false, false, false,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, true, false, false, false, false,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, true, false, false, false,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, true, false, false,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, true, false,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, true,
            8, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            9, 8, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 9, 7168, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 8, 31, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 8, 4096, &token_fanout_config),
        select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 8, 7200, &token_fanout_config),
    };
    for (const auto status : token_fanout_disabled_cases) {
        if (status != DispatchTokenFanoutConfigStatus::kDisabled ||
            token_fanout_config.enabled ||
            token_fanout_config.vector_bytes != 0)
            return 121;
    }
    if (select_dispatch_token_fanout_config(
            "1", true, true, false, false, false, false, false,
            8, 8, 7168, nullptr) !=
                DispatchTokenFanoutConfigStatus::kInvalid)
        return 122;

    const auto common_fanout =
        build_dispatch_token_fanout_plan(7168, 32, 7168);
    const auto tailed_fanout =
        build_dispatch_token_fanout_plan(7184, 32, 7168);
    const auto oversized_fanout =
        build_dispatch_token_fanout_plan(7200, 32, 7168);
    const auto invalid_alignment_fanout =
        build_dispatch_token_fanout_plan(7168, 48, 7168);
    if (!common_fanout.valid || common_fanout.vector_bytes != 7168 ||
        common_fanout.scalar_begin != 7168 ||
        common_fanout.source_loads_per_token != 1 ||
        !tailed_fanout.valid || tailed_fanout.vector_bytes != 7168 ||
        tailed_fanout.scalar_begin != 7168 || oversized_fanout.valid ||
        invalid_alignment_fanout.valid)
        return 123;
    std::uint64_t source_bytes = 0;
    std::uint64_t destination_bytes = 0;
    if (!dispatch_token_fanout_work_bytes(
            8192, 43325, common_fanout,
            &source_bytes, &destination_bytes) ||
        source_bytes != 8192ULL * 7168ULL ||
        destination_bytes != 43325ULL * 7168ULL)
        return 124;
    source_bytes = 17;
    destination_bytes = 19;
    if (dispatch_token_fanout_work_bytes(
            std::numeric_limits<std::uint64_t>::max(), 1,
            common_fanout, &source_bytes, &destination_bytes) ||
        source_bytes != 0 || destination_bytes != 0 ||
        dispatch_token_fanout_work_bytes(
            1, 1, common_fanout, nullptr, &destination_bytes) ||
        dispatch_token_fanout_work_bytes(
            1, 1, common_fanout, &source_bytes, nullptr))
        return 125;

    DispatchDevicePrefixConfig device_prefix_config{};
    if (select_dispatch_device_prefix_config(
            "1", false, true, false, false,
            &device_prefix_config) !=
                DispatchDevicePrefixConfigStatus::kEnabled ||
        !device_prefix_config.enabled)
        return 95;
    if (select_dispatch_device_prefix_config(
            nullptr, false, true, false, false,
            &device_prefix_config) !=
                DispatchDevicePrefixConfigStatus::kEnabled ||
        !device_prefix_config.enabled ||
        select_dispatch_device_prefix_config(
            "0", false, true, false, false,
            &device_prefix_config) !=
                DispatchDevicePrefixConfigStatus::kDisabled ||
        device_prefix_config.enabled)
        return 96;
    for (const auto invalid_value : {"", "2", "true", "01"}) {
        if (select_dispatch_device_prefix_config(
                invalid_value, false, true, false, false,
                &device_prefix_config) !=
                    DispatchDevicePrefixConfigStatus::kInvalid)
            return 97;
    }
    const DispatchDevicePrefixConfigStatus device_prefix_disabled_cases[] = {
        select_dispatch_device_prefix_config(
            "1", true, true, false, false, &device_prefix_config),
        select_dispatch_device_prefix_config(
            "1", false, false, false, false, &device_prefix_config),
        select_dispatch_device_prefix_config(
            "1", false, true, true, false, &device_prefix_config),
        select_dispatch_device_prefix_config(
            "1", false, true, false, true, &device_prefix_config),
    };
    for (const auto status : device_prefix_disabled_cases) {
        if (status != DispatchDevicePrefixConfigStatus::kDisabled ||
            device_prefix_config.enabled)
            return 98;
    }
    DispatchConsumerTileConfig consumer_tile_config{};
    if (select_dispatch_consumer_tile_config(
            "1024", true, false, true, false, false, false,
            &consumer_tile_config) !=
                DispatchConsumerTileConfigStatus::kEnabled ||
        consumer_tile_config.tile_bytes != 1024)
        return 99;
    if (select_dispatch_consumer_tile_config(
            nullptr, true, false, true, false, false, false,
            &consumer_tile_config) !=
                DispatchConsumerTileConfigStatus::kEnabled ||
        consumer_tile_config.tile_bytes != 8192)
        return 100;
    for (const auto baseline_value : {"512"}) {
        if (select_dispatch_consumer_tile_config(
                baseline_value, true, false, true, false, false, false,
                &consumer_tile_config) !=
                    DispatchConsumerTileConfigStatus::kDisabled ||
            consumer_tile_config.tile_bytes != 512)
            return 100;
    }
    for (const auto candidate_value : {"1024", "2048", "4096", "8192"}) {
        if (select_dispatch_consumer_tile_config(
                candidate_value, true, false, true, false, false, false,
                &consumer_tile_config) !=
                    DispatchConsumerTileConfigStatus::kEnabled)
            return 101;
    }
    for (const auto invalid_value :
         {"", "0", "256", "16384", " 1024", "+1024", "1024x"}) {
        if (select_dispatch_consumer_tile_config(
                invalid_value, true, false, true, false, false, false,
                &consumer_tile_config) !=
                    DispatchConsumerTileConfigStatus::kInvalid)
            return 102;
    }
    const DispatchConsumerTileConfigStatus consumer_tile_disabled_cases[] = {
        select_dispatch_consumer_tile_config(
            "1024", false, false, true, false, false, false,
            &consumer_tile_config),
        select_dispatch_consumer_tile_config(
            "1024", true, true, true, false, false, false,
            &consumer_tile_config),
        select_dispatch_consumer_tile_config(
            "1024", true, false, false, false, false, false,
            &consumer_tile_config),
        select_dispatch_consumer_tile_config(
            "1024", true, false, true, true, false, false,
            &consumer_tile_config),
        select_dispatch_consumer_tile_config(
            "1024", true, false, true, false, true, false,
            &consumer_tile_config),
        select_dispatch_consumer_tile_config(
            "1024", true, false, true, false, false, true,
            &consumer_tile_config),
    };
    for (const auto status : consumer_tile_disabled_cases) {
        if (status != DispatchConsumerTileConfigStatus::kDisabled ||
            consumer_tile_config.tile_bytes != 512)
            return 103;
    }
    const auto common_consumer_copy =
        dispatch_consumer_copy_plan(7168, 1024, 32);
    const auto tailed_consumer_copy =
        dispatch_consumer_copy_plan(7184, 1024, 32);
    const auto invalid_consumer_copy =
        dispatch_consumer_copy_plan(7168, 1000, 32);
    if (!common_consumer_copy.valid ||
        common_consumer_copy.vector_bytes != 7168 ||
        common_consumer_copy.tile_count != 7 ||
        common_consumer_copy.scalar_begin != 7168 ||
        !tailed_consumer_copy.valid ||
        tailed_consumer_copy.vector_bytes != 7168 ||
        tailed_consumer_copy.tile_count != 7 ||
        tailed_consumer_copy.scalar_begin != 7168 ||
        invalid_consumer_copy.valid)
        return 104;
    DispatchParallelPrefixConfig parallel_prefix_config{};
    if (select_dispatch_parallel_prefix_config(
            "1", true, false, true, false, false, false,
            &parallel_prefix_config) !=
                DispatchParallelPrefixConfigStatus::kEnabled ||
        !parallel_prefix_config.enabled)
        return 105;
    if (select_dispatch_parallel_prefix_config(
            nullptr, true, false, true, false, false, false,
            &parallel_prefix_config) !=
                DispatchParallelPrefixConfigStatus::kEnabled ||
        !parallel_prefix_config.enabled)
        return 106;
    for (const auto baseline_value : {"0"}) {
        if (select_dispatch_parallel_prefix_config(
                baseline_value, true, false, true, false, false, false,
                &parallel_prefix_config) !=
                    DispatchParallelPrefixConfigStatus::kDisabled ||
            parallel_prefix_config.enabled)
            return 106;
    }
    for (const auto invalid_value : {"", "2", "true", "01"}) {
        if (select_dispatch_parallel_prefix_config(
                invalid_value, true, false, true, false, false, false,
                &parallel_prefix_config) !=
                    DispatchParallelPrefixConfigStatus::kInvalid)
            return 107;
    }
    const DispatchParallelPrefixConfigStatus
        parallel_prefix_disabled_cases[] = {
            select_dispatch_parallel_prefix_config(
                "1", false, false, true, false, false, false,
                &parallel_prefix_config),
            select_dispatch_parallel_prefix_config(
                "1", true, true, true, false, false, false,
                &parallel_prefix_config),
            select_dispatch_parallel_prefix_config(
                "1", true, false, false, false, false, false,
                &parallel_prefix_config),
            select_dispatch_parallel_prefix_config(
                "1", true, false, true, true, false, false,
                &parallel_prefix_config),
            select_dispatch_parallel_prefix_config(
                "1", true, false, true, false, true, false,
                &parallel_prefix_config),
            select_dispatch_parallel_prefix_config(
                "1", true, false, true, false, false, true,
                &parallel_prefix_config),
        };
    for (const auto status : parallel_prefix_disabled_cases) {
        if (status != DispatchParallelPrefixConfigStatus::kDisabled ||
            parallel_prefix_config.enabled)
            return 108;
    }
    const auto representative_prefix_workers =
        dispatch_expert_prefix_worker_plan(256, 8, 512);
    const auto invalid_expert_divisibility =
        dispatch_expert_prefix_worker_plan(255, 8, 512);
    const auto insufficient_prefix_threads =
        dispatch_expert_prefix_worker_plan(256, 8, 31);
    const auto invalid_prefix_world =
        dispatch_expert_prefix_worker_plan(256, 0, 512);
    if (!representative_prefix_workers.valid ||
        representative_prefix_workers.local_experts != 32 ||
        representative_prefix_workers.active_threads != 32 ||
        invalid_expert_divisibility.valid ||
        insufficient_prefix_threads.valid || invalid_prefix_world.valid)
        return 109;
    DispatchPipelineConfig pipeline_config{};
    if (select_dispatch_pipeline_config(
            "2048", false, true, false, false, 8, 8192,
            &pipeline_config) != DispatchPipelineConfigStatus::kEnabled ||
        !pipeline_config.enabled || pipeline_config.chunk_slots != 2048 ||
        pipeline_config.chunk_count != 4)
        return 91;
    if (select_dispatch_pipeline_config(
            nullptr, false, true, false, false, 8, 8192,
            &pipeline_config) != DispatchPipelineConfigStatus::kDisabled ||
        pipeline_config.enabled)
        return 92;
    for (const auto invalid_value : {"", "0", "-1", "2048x"}) {
        if (select_dispatch_pipeline_config(
                invalid_value, false, true, false, false, 8, 8192,
                &pipeline_config) != DispatchPipelineConfigStatus::kInvalid)
            return 93;
    }
    const DispatchPipelineConfigStatus disabled_cases[] = {
        select_dispatch_pipeline_config(
            "2048", true, true, false, false, 8, 8192,
            &pipeline_config),
        select_dispatch_pipeline_config(
            "2048", false, false, false, false, 8, 8192,
            &pipeline_config),
        select_dispatch_pipeline_config(
            "2048", false, true, true, false, 8, 8192,
            &pipeline_config),
        select_dispatch_pipeline_config(
            "2048", false, true, false, true, 8, 8192,
            &pipeline_config),
        select_dispatch_pipeline_config(
            "2048", false, true, false, false, 1, 8192,
            &pipeline_config),
        select_dispatch_pipeline_config(
            "8192", false, true, false, false, 8, 8192,
            &pipeline_config),
        select_dispatch_pipeline_config(
            "512", false, true, false, false, 8, 8192,
            &pipeline_config),
    };
    for (const auto status : disabled_cases) {
        if (status != DispatchPipelineConfigStatus::kDisabled)
            return 94;
    }
    if (!topk_grouping_reference_contract())
        return 52;
    DispatchChunkPlan chunk_plan{};
    if (!build_dispatch_chunk_plan(8192, 2048, &chunk_plan) ||
        chunk_plan.chunk_count != 4 ||
        chunk_plan.chunk_slots != 2048)
        return 86;
    const std::uint32_t expected_pipeline_slots[] = {0, 1, 0, 1};
    const std::uint64_t expected_peer_counts[] = {2048, 2048, 904, 0};
    for (std::uint32_t chunk = 0; chunk < chunk_plan.chunk_count; ++chunk) {
        std::uint64_t begin = 0;
        std::uint64_t count = 0;
        if (dispatch_pipeline_slot(chunk) != expected_pipeline_slots[chunk] ||
            !dispatch_chunk_peer_range(
                chunk_plan, chunk, 5000, &begin, &count) ||
            begin != static_cast<std::uint64_t>(chunk) * 2048 ||
            count != expected_peer_counts[chunk])
            return 87;
        if (!dispatch_chunk_peer_range(
                chunk_plan, chunk, 0, &begin, &count) || count != 0)
            return 88;
    }
    if (build_dispatch_chunk_plan(8192, 0, &chunk_plan) ||
        dispatch_chunk_peer_range(chunk_plan, 4, 5000, nullptr, nullptr))
        return 89;
    DispatchSourceChunkPlan source_chunk_plan{};
    if (!build_dispatch_source_chunk_plan(
            8192, kDispatchGroupingTokensPerTile, 512,
            &source_chunk_plan) ||
        source_chunk_plan.token_count != 8192 ||
        source_chunk_plan.tile_count != 2048 ||
        source_chunk_plan.chunk_tiles != 512 ||
        source_chunk_plan.chunk_count != 4)
        return 166;
    std::uint32_t expected_token_begin = 0;
    for (std::uint32_t chunk = 0;
         chunk < source_chunk_plan.chunk_count; ++chunk) {
        std::uint32_t tile_begin = 0;
        std::uint32_t tile_end = 0;
        std::uint32_t token_begin = 0;
        std::uint32_t token_end = 0;
        if (!dispatch_source_chunk_bounds(
                source_chunk_plan, chunk, &tile_begin, &tile_end,
                &token_begin, &token_end) ||
            tile_begin != chunk * 512 ||
            tile_end != (chunk + 1) * 512 ||
            token_begin != expected_token_begin ||
            token_end != (chunk + 1) * 2048)
            return 167;
        expected_token_begin = token_end;
    }
    DispatchSourceChunkPlan tailed_source_chunk_plan{};
    if (!build_dispatch_source_chunk_plan(
            9, kDispatchGroupingTokensPerTile, 2,
            &tailed_source_chunk_plan) ||
        tailed_source_chunk_plan.tile_count != 3 ||
        tailed_source_chunk_plan.chunk_count != 2)
        return 168;
    std::uint32_t tail_tile_begin = 0;
    std::uint32_t tail_tile_end = 0;
    std::uint32_t tail_token_begin = 0;
    std::uint32_t tail_token_end = 0;
    if (!dispatch_source_chunk_bounds(
            tailed_source_chunk_plan, 1, &tail_tile_begin,
            &tail_tile_end, &tail_token_begin, &tail_token_end) ||
        tail_tile_begin != 2 || tail_tile_end != 3 ||
        tail_token_begin != 8 || tail_token_end != 9)
        return 169;
    std::uint64_t source_slot_begin = 0;
    std::uint64_t source_slot_count = 0;
    if (!dispatch_source_chunk_peer_range(
            source_chunk_plan, 0, 0, 11, 18,
            &source_slot_begin, &source_slot_count) ||
        source_slot_begin != 0 || source_slot_count != 11 ||
        !dispatch_source_chunk_peer_range(
            source_chunk_plan, 3, 13, 0, 18,
            &source_slot_begin, &source_slot_count) ||
        source_slot_begin != 13 || source_slot_count != 5 ||
        dispatch_source_chunk_peer_range(
            source_chunk_plan, 0, 12, 11, 18,
            &source_slot_begin, &source_slot_count) ||
        dispatch_source_chunk_peer_range(
            source_chunk_plan, 4, 0, 0, 0, nullptr, nullptr))
        return 170;
    DispatchSourcePipelineConfig source_pipeline_config{};
    if (select_dispatch_source_pipeline_config(
            "512", true, false, true, false, false, false, 2,
            8192, &source_pipeline_config) !=
                DispatchSourcePipelineConfigStatus::kEnabled ||
        !source_pipeline_config.enabled ||
        source_pipeline_config.chunk_tiles != 512 ||
        source_pipeline_config.chunk_count != 4)
        return 171;
    if (select_dispatch_source_pipeline_config(
            nullptr, true, false, true, false, false, false, 2,
            8192, &source_pipeline_config) !=
                DispatchSourcePipelineConfigStatus::kDisabled ||
        source_pipeline_config.enabled)
        return 172;
    for (const char* invalid_value : {"", "0", "-1", "512x"}) {
        if (select_dispatch_source_pipeline_config(
                invalid_value, true, false, true, false, false, false, 2,
                8192, &source_pipeline_config) !=
            DispatchSourcePipelineConfigStatus::kInvalid)
            return 173;
    }
    const DispatchSourcePipelineConfigStatus source_pipeline_disabled[] = {
        select_dispatch_source_pipeline_config(
            "512", false, false, true, false, false, false, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "512", true, true, true, false, false, false, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "512", true, false, false, false, false, false, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "512", true, false, true, true, false, false, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "512", true, false, true, false, true, false, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "512", true, false, true, false, false, true, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "512", true, false, true, false, false, false, 1,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "2048", true, false, true, false, false, false, 2,
            8192, &source_pipeline_config),
        select_dispatch_source_pipeline_config(
            "128", true, false, true, false, false, false, 2,
            8192, &source_pipeline_config),
    };
    for (const auto status : source_pipeline_disabled) {
        if (status != DispatchSourcePipelineConfigStatus::kDisabled ||
            source_pipeline_config.enabled)
            return 174;
    }
    auto pipeline_input = valid_input();
    pipeline_input.mode_flags = mode_bit(CoreMode::kPipeline);
    pipeline_input.data_num_blocks = 72;
    CoreTiling pipeline_tiling{};
    if (!build_core_tiling(pipeline_input, &pipeline_tiling).ok() ||
        pipeline_tiling.workspace_layout.dispatch_pipeline_bytes !=
            sizeof(DispatchPipelineState) ||
        pipeline_tiling.workspace_layout.dispatch_pipeline_offset %
                alignof(DispatchPipelineState) != 0 ||
        pipeline_tiling.workspace_layout.dispatch_pipeline_offset +
                pipeline_tiling.workspace_layout.dispatch_pipeline_bytes >
            pipeline_tiling.workspace_layout.scratch_offset +
                pipeline_tiling.workspace_layout.scratch_bytes)
        return 90;
    DispatchPipelineState pipeline_state{};
    if (&pipeline_state.slots[0].request ==
        &pipeline_state.slots[1].request)
        return 91;
    constexpr std::uint64_t pipeline_generation = 17;
    constexpr std::uint32_t pipeline_blocks = 72;
    if (dispatch_pipeline_producer_must_wait_for_reuse(0) ||
        dispatch_pipeline_producer_must_wait_for_reuse(1) ||
        !dispatch_pipeline_producer_must_wait_for_reuse(2) ||
        !dispatch_pipeline_producer_must_wait_for_reuse(3))
        return 189;
    pipeline_state.generation = pipeline_generation;
    for (std::uint32_t chunk = 0; chunk < 4; ++chunk) {
        auto& slot = pipeline_state.slots[dispatch_pipeline_slot(chunk)];
        if (!dispatch_pipeline_slot_reusable(slot, chunk))
            return 179;
        slot.state = DispatchPipelineSlotState::kProducing;
        slot.chunk_index = chunk;
        slot.published_chunk = chunk;
        slot.scalar_blocks_completed = 0;
        slot.hidden_blocks_completed = 0;
        for (std::uint32_t block = 0; block < pipeline_blocks; ++block) {
            const bool publish = dispatch_pipeline_completion_is_last(
                slot.scalar_blocks_completed, pipeline_blocks);
            ++slot.scalar_blocks_completed;
            if (publish != (block + 1 == pipeline_blocks))
                return 180;
        }
        slot.state = DispatchPipelineSlotState::kScalarReady;
        for (std::uint32_t block = 0; block < pipeline_blocks; ++block) {
            auto& scalar_progress = pipeline_state.scalar_progress[block];
            auto& hidden_progress = pipeline_state.hidden_progress[block];
            if (&scalar_progress == &hidden_progress ||
                reinterpret_cast<std::uintptr_t>(&scalar_progress) % 64 != 0 ||
                reinterpret_cast<std::uintptr_t>(&hidden_progress) % 64 != 0)
                return 187;
            scalar_progress.generation = pipeline_generation;
            scalar_progress.completed_chunk = chunk;
            if (!dispatch_pipeline_block_progress_ready(
                    scalar_progress, pipeline_generation, chunk) ||
                dispatch_pipeline_block_progress_ready(
                    scalar_progress, pipeline_generation + 1U, chunk) ||
                dispatch_pipeline_block_progress_ready(
                    scalar_progress, pipeline_generation, chunk + 1U))
                return 188;
            hidden_progress.generation = pipeline_generation;
            hidden_progress.completed_chunk = chunk;
            if (!dispatch_pipeline_block_progress_ready(
                    hidden_progress, pipeline_generation, chunk))
                return 184;
            ++slot.hidden_blocks_completed;
        }
        slot.state = DispatchPipelineSlotState::kReady;
        if (!dispatch_pipeline_slot_ready_for_release(
                slot, pipeline_generation, pipeline_state.generation,
                chunk, pipeline_blocks))
            return 181;
        slot.state = DispatchPipelineSlotState::kCompleted;
    }
    auto& wrapped_slot = pipeline_state.slots[0];
    wrapped_slot.state = DispatchPipelineSlotState::kInFlight;
    if (dispatch_pipeline_slot_reusable(wrapped_slot, 4))
        return 182;
    wrapped_slot.state = DispatchPipelineSlotState::kCompleted;
    wrapped_slot.published_chunk = 3;
    if (dispatch_pipeline_slot_reusable(wrapped_slot, 4))
        return 183;
    if (dispatch_pipeline_release_batch_pending(7, 7) ||
        !dispatch_pipeline_release_batch_pending(8, 7) ||
        dispatch_pipeline_release_batch_acknowledged(8, 7) ||
        !dispatch_pipeline_release_batch_acknowledged(8, 8) ||
        !dispatch_pipeline_release_batch_acknowledged(8, 9))
        return 185;
    if (dispatch_pipeline_wait_timed_out(100, 109, 10) ||
        !dispatch_pipeline_wait_timed_out(100, 110, 10) ||
        dispatch_pipeline_wait_timed_out(
            std::numeric_limits<std::uint64_t>::max() - 4, 3, 9) ||
        !dispatch_pipeline_wait_timed_out(
            std::numeric_limits<std::uint64_t>::max() - 4, 4, 9) ||
        dispatch_pipeline_wait_timed_out(100, 200, 0))
        return 186;
    pipeline_state.diagnostic_stage =
        DispatchPipelineDiagnosticStage::kReleaseBatchTargetPublished;
    pipeline_state.diagnostic_detail = 7;
    if (pipeline_state.diagnostic_stage !=
            DispatchPipelineDiagnosticStage::kReleaseBatchTargetPublished ||
        pipeline_state.diagnostic_detail != 7)
        return 187;
    if (const int boundary_error = direct_device_index_boundary_contract();
        boundary_error != 0)
        return boundary_error;
    std::uint64_t combine_slot = 0;
    if (!combine_record_slot_index(0, 0, 4, 6, &combine_slot) ||
        combine_slot != 0 ||
        !combine_record_slot_index(3, 5, 4, 6, &combine_slot) ||
        combine_slot != 23 ||
        combine_record_slot_index(4, 0, 4, 6, &combine_slot) ||
        combine_record_slot_index(0, 6, 4, 6, &combine_slot) ||
        combine_record_slot_index(0, 0, 4, 6, nullptr) ||
        combine_record_slot_index(
            std::numeric_limits<std::uint64_t>::max() - 1, 1,
            std::numeric_limits<std::uint64_t>::max(), 2,
            &combine_slot))
        return 34;

    std::uint64_t bitmap_words = 0;
    if (!dispatch_bitmap_words(0, &bitmap_words) || bitmap_words != 0 ||
        !dispatch_bitmap_words(1, &bitmap_words) || bitmap_words != 1 ||
        !dispatch_bitmap_words(64, &bitmap_words) || bitmap_words != 1 ||
        !dispatch_bitmap_words(65, &bitmap_words) || bitmap_words != 2 ||
        dispatch_bitmap_words(1, nullptr) ||
        !dispatch_owner_bitmap_words(3, 65, &bitmap_words) ||
        bitmap_words != 6 ||
        dispatch_owner_bitmap_words(
            std::numeric_limits<std::uint64_t>::max(), 65,
            &bitmap_words))
        return 29;
    std::uint64_t bitmap_word = 0;
    std::uint64_t bitmap_mask = 0;
    if (!dispatch_bitmap_location(
            0, 65, &bitmap_word, &bitmap_mask) ||
        bitmap_word != 0 || bitmap_mask != 1 ||
        !dispatch_bitmap_location(
            63, 65, &bitmap_word, &bitmap_mask) ||
        bitmap_word != 0 || bitmap_mask != (std::uint64_t{1} << 63U) ||
        !dispatch_bitmap_location(
            64, 65, &bitmap_word, &bitmap_mask) ||
        bitmap_word != 1 || bitmap_mask != 1 ||
        dispatch_bitmap_location(
            65, 65, &bitmap_word, &bitmap_mask) ||
        dispatch_bitmap_location(
            0, 65, nullptr, &bitmap_mask) ||
        dispatch_bitmap_location(
            0, 65, &bitmap_word, nullptr))
        return 33;
    std::uint64_t owner_word = 0;
    std::uint64_t owner_words = 0;
    if (!dispatch_owner_bitmap_range(
            0, 3, 65, &owner_word, &owner_words) ||
        owner_word != 0 || owner_words != 2 ||
        !dispatch_owner_bitmap_range(
            2, 3, 65, &owner_word, &owner_words) ||
        owner_word != 4 || owner_words != 2 ||
        dispatch_owner_bitmap_range(
            3, 3, 65, &owner_word, &owner_words) ||
        dispatch_owner_bitmap_range(
            0, 3, 65, nullptr, &owner_words) ||
        dispatch_owner_bitmap_range(
            0, 3, 65, &owner_word, nullptr))
        return 34;

    std::uint64_t logical_record = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t source_slot = 0;
    if (!dispatch_receive_logical_record(
            0, 0, 2, 16, &logical_record) || logical_record != 0 ||
        !dispatch_receive_logical_record(
            1, 0, 2, 16, &logical_record) || logical_record != 16 ||
        !dispatch_receive_logical_record(
            1, 15, 2, 16, &logical_record) || logical_record != 31 ||
        dispatch_receive_logical_record(
            2, 0, 2, 16, &logical_record) ||
        dispatch_receive_logical_record(
            0, 16, 2, 16, &logical_record) ||
        dispatch_receive_logical_record(
            0, 0, 2, 0, &logical_record) ||
        dispatch_receive_logical_record(
            std::numeric_limits<std::uint64_t>::max() - 1, 1,
            std::numeric_limits<std::uint64_t>::max(), 2,
            &logical_record) ||
        !dispatch_receive_record_coordinates(
            0, 2, 16, &source_rank, &source_slot) ||
        source_rank != 0 || source_slot != 0 ||
        !dispatch_receive_record_coordinates(
            16, 2, 16, &source_rank, &source_slot) ||
        source_rank != 1 || source_slot != 0 ||
        !dispatch_receive_record_coordinates(
            31, 2, 16, &source_rank, &source_slot) ||
        source_rank != 1 || source_slot != 15 ||
        dispatch_receive_record_coordinates(
            32, 2, 16, &source_rank, &source_slot) ||
        dispatch_receive_record_coordinates(
            0, 2, 0, &source_rank, &source_slot) ||
        dispatch_receive_record_coordinates(
            0, 2, 16, nullptr, &source_slot) ||
        dispatch_receive_record_coordinates(
            0, 2, 16, &source_rank, nullptr))
        return 72;
    if (!direct_dispatch_cached_bitmap_owner(0) ||
        direct_dispatch_cached_bitmap_owner(1) ||
        direct_dispatch_cached_bitmap_owner(71))
        return 78;
    const std::uint64_t compact_source_bases[] = {0, 2, 2, 5};
    const std::uint64_t compact_source_counts[] = {2, 0, 3, 1};
    if (!dispatch_compact_record_coordinates(
            0, compact_source_bases, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        source_rank != 0 || source_slot != 0 ||
        !dispatch_compact_record_coordinates(
            1, compact_source_bases, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        source_rank != 0 || source_slot != 1 ||
        !dispatch_compact_record_coordinates(
            2, compact_source_bases, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        source_rank != 2 || source_slot != 0 ||
        !dispatch_compact_record_coordinates(
            4, compact_source_bases, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        source_rank != 2 || source_slot != 2 ||
        !dispatch_compact_record_coordinates(
            5, compact_source_bases, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        source_rank != 3 || source_slot != 0 ||
        dispatch_compact_record_coordinates(
            6, compact_source_bases, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        dispatch_compact_record_coordinates(
            0, nullptr, compact_source_counts, 4,
            &source_rank, &source_slot) ||
        dispatch_compact_record_coordinates(
            0, compact_source_bases, compact_source_counts, 0,
            &source_rank, &source_slot))
        return 95;
    const auto count_bridge = dispatch_count_bridge_layout(8, 256, 32, 16);
    const auto invalid_count_bridge =
        dispatch_count_bridge_layout(0, 256, 32, 16);
    if (!count_bridge.valid || count_bridge.rank_prefix_offset != 0 ||
        count_bridge.kernel_expert_prefix_offset != 16 ||
        count_bridge.kernel_unaligned_offset != 288 ||
        count_bridge.kernel_elements != 544 ||
        count_bridge.public_expert_prefix_offset != 0 ||
        count_bridge.public_unaligned_offset != 32 ||
        count_bridge.public_elements != 64 || invalid_count_bridge.valid)
        return 96;
    std::uint64_t expert_tile_index = 0;
    std::uint64_t destination = 0;
    if (!dispatch_expert_tile_index(
            0, 0, 3, 2, &expert_tile_index) || expert_tile_index != 0 ||
        !dispatch_expert_tile_index(
            1, 0, 3, 2, &expert_tile_index) || expert_tile_index != 2 ||
        !dispatch_expert_tile_index(
            2, 1, 3, 2, &expert_tile_index) || expert_tile_index != 5 ||
        dispatch_expert_tile_index(
            3, 0, 3, 2, &expert_tile_index) ||
        dispatch_expert_tile_index(
            0, 2, 3, 2, &expert_tile_index) ||
        !dispatch_expert_destination(
            128, 3, 2, 256, &destination) || destination != 133 ||
        dispatch_expert_destination(
            128, 3, 2, 133, &destination) ||
        dispatch_expert_destination(
            std::numeric_limits<std::uint64_t>::max(), 1, 0,
            std::numeric_limits<std::uint64_t>::max(), &destination))
        return 73;
    std::uint64_t combine_index = 0;
    const std::int32_t uneven_prefix[] = {0, 3, 3, 7};
    std::uint64_t combine_rank = 0;
    std::uint64_t combine_begin = 0;
    std::uint64_t combine_end = 0;
    std::uint64_t combine_destination_slot = 0;
    if (!combine_tile_rank_index(0, 0, 3, 2, &combine_index) ||
        combine_index != 0 ||
        !combine_tile_rank_index(2, 1, 3, 2, &combine_index) ||
        combine_index != 5 ||
        combine_tile_rank_index(3, 0, 3, 2, &combine_index) ||
        combine_tile_rank_index(0, 2, 3, 2, &combine_index) ||
        !combine_receive_record_coordinates(
            31, 2, 16, &source_rank, &source_slot) ||
        source_rank != 1 || source_slot != 15 ||
        combine_receive_record_coordinates(
            32, 2, 16, &source_rank, &source_slot) ||
        !combine_rank_prefix_range(
            uneven_prefix, 4, 7, 0, &combine_begin, &combine_end) ||
        combine_begin != 0 || combine_end != 0 ||
        !combine_rank_prefix_range(
            uneven_prefix, 4, 7, 1, &combine_begin, &combine_end) ||
        combine_begin != 0 || combine_end != 3 ||
        !combine_rank_prefix_range(
            uneven_prefix, 4, 7, 2, &combine_begin, &combine_end) ||
        combine_begin != 3 || combine_end != 3 ||
        !combine_rank_prefix_range(
            uneven_prefix, 4, 7, 3, &combine_begin, &combine_end) ||
        combine_begin != 3 || combine_end != 7 ||
        combine_rank_prefix_range(
            uneven_prefix, 4, 6, 3, &combine_begin, &combine_end) ||
        !combine_destination_rank_for_row(
            0, uneven_prefix, 4, 7, &combine_rank) || combine_rank != 1 ||
        !combine_destination_rank_for_row(
            2, uneven_prefix, 4, 7, &combine_rank) || combine_rank != 1 ||
        !combine_destination_rank_for_row(
            3, uneven_prefix, 4, 7, &combine_rank) || combine_rank != 3 ||
        combine_destination_rank_for_row(
            7, uneven_prefix, 4, 7, &combine_rank) ||
        !combine_record_destination_slot(
            5, 2, 8, &combine_destination_slot) ||
        combine_destination_slot != 7 ||
        combine_record_destination_slot(
            5, 3, 8, &combine_destination_slot))
        return 74;

    const auto common_normal_payload = combine_producer_payload_copy_plan(
        7168, 256, 16, false);
    const auto tail_normal_payload = combine_producer_payload_copy_plan(
        272, 256, 16, false);
    const auto unaligned_normal_payload = combine_producer_payload_copy_plan(
        257, 256, 16, false);
    const auto expanded_payload = combine_producer_payload_copy_plan(
        7168, 256, 16, true);
    const auto invalid_payload = combine_producer_payload_copy_plan(
        7168, 0, 16, false);
    if (!common_normal_payload.valid ||
        common_normal_payload.vector_elements != 7168 ||
        common_normal_payload.scalar_begin != 7168 ||
        !tail_normal_payload.valid ||
        tail_normal_payload.vector_elements != 256 ||
        tail_normal_payload.scalar_begin != 256 ||
        !unaligned_normal_payload.valid ||
        unaligned_normal_payload.vector_elements != 0 ||
        unaligned_normal_payload.scalar_begin != 0 ||
        !expanded_payload.valid || expanded_payload.vector_elements != 0 ||
        expanded_payload.scalar_begin != 0 || invalid_payload.valid)
        return 79;

    CombineExpandedVectorReduceConfig expanded_reduce_config{};
    if (select_combine_expanded_vector_reduce_config(
            nullptr, true, true, true, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        expanded_reduce_config.enabled ||
        select_combine_expanded_vector_reduce_config(
            "0", true, true, true, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, true, true, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kEnabled ||
        !expanded_reduce_config.enabled ||
        select_combine_expanded_vector_reduce_config(
            "2", true, true, true, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kInvalid ||
        select_combine_expanded_vector_reduce_config(
            "1", false, true, true, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, false, true, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, true, false, false, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, true, true, true, 8,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, true, true, false, 0,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, true, true, false, 33,
            &expanded_reduce_config) !=
            CombineExpandedVectorReduceConfigStatus::kDisabled ||
        select_combine_expanded_vector_reduce_config(
            "1", true, true, true, false, 8, nullptr) !=
            CombineExpandedVectorReduceConfigStatus::kInvalid)
        return 87;

    const auto aligned_expanded_reduce =
        combine_expanded_producer_payload_plan(7168, 256, 16, true);
    const auto tail_expanded_reduce =
        combine_expanded_producer_payload_plan(7184, 256, 16, true);
    const auto disabled_expanded_reduce =
        combine_expanded_producer_payload_plan(7168, 256, 16, false);
    const auto invalid_expanded_reduce =
        combine_expanded_producer_payload_plan(7168, 255, 16, true);
    if (!aligned_expanded_reduce.valid ||
        aligned_expanded_reduce.vector_elements != 7168 ||
        aligned_expanded_reduce.scalar_begin != 7168 ||
        !tail_expanded_reduce.valid ||
        tail_expanded_reduce.vector_elements != 7168 ||
        tail_expanded_reduce.scalar_begin != 7168 ||
        !disabled_expanded_reduce.valid ||
        disabled_expanded_reduce.vector_elements != 0 ||
        disabled_expanded_reduce.scalar_begin != 0 ||
        invalid_expanded_reduce.valid)
        return 88;

    CombineLocalCopyDataCopyConfig local_copy_config{};
    if (select_combine_local_copy_datacopy_config(
            nullptr, true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kDisabled ||
        local_copy_config.enabled ||
        select_combine_local_copy_datacopy_config(
            "0", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kDisabled ||
        select_combine_local_copy_datacopy_config(
            "1", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        !local_copy_config.enabled || local_copy_config.tile_bytes != 32768 ||
        select_combine_local_copy_datacopy_config(
            "512", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 512 ||
        select_combine_local_copy_datacopy_config(
            "1024", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 1024 ||
        select_combine_local_copy_datacopy_config(
            "2048", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 2048 ||
        select_combine_local_copy_datacopy_config(
            "4096", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 4096 ||
        select_combine_local_copy_datacopy_config(
            "8192", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 8192 ||
        select_combine_local_copy_datacopy_config(
            "16384", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 16384 ||
        select_combine_local_copy_datacopy_config(
            "32768", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kEnabled ||
        local_copy_config.tile_bytes != 32768 ||
        select_combine_local_copy_datacopy_config(
            "256", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kInvalid ||
        select_combine_local_copy_datacopy_config(
            "2", true, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kInvalid ||
        select_combine_local_copy_datacopy_config(
            "1", false, false, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kDisabled ||
        select_combine_local_copy_datacopy_config(
            "1", true, true, &local_copy_config) !=
            CombineLocalCopyDataCopyConfigStatus::kDisabled ||
        select_combine_local_copy_datacopy_config(
            "1", true, false, nullptr) !=
            CombineLocalCopyDataCopyConfigStatus::kInvalid)
        return 89;

    CombineVectorReduceTileConfig vector_reduce_tile_config{};
    if (select_combine_vector_reduce_tile_config(
            nullptr, true, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kEnabled ||
        !vector_reduce_tile_config.enabled ||
        vector_reduce_tile_config.tile_elements != 512 ||
        select_combine_vector_reduce_tile_config(
            nullptr, false, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kDisabled ||
        vector_reduce_tile_config.enabled ||
        select_combine_vector_reduce_tile_config(
            nullptr, true, true, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kDisabled ||
        vector_reduce_tile_config.enabled ||
        select_combine_vector_reduce_tile_config(
            "0", true, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kDisabled ||
        select_combine_vector_reduce_tile_config(
            "1", true, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kEnabled ||
        !vector_reduce_tile_config.enabled ||
        vector_reduce_tile_config.tile_elements != 512 ||
        select_combine_vector_reduce_tile_config(
            "512", true, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kEnabled ||
        vector_reduce_tile_config.tile_elements != 512 ||
        select_combine_vector_reduce_tile_config(
            "256", true, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kInvalid ||
        select_combine_vector_reduce_tile_config(
            "1", false, false, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kDisabled ||
        select_combine_vector_reduce_tile_config(
            "1", true, true, &vector_reduce_tile_config) !=
            CombineVectorReduceTileConfigStatus::kDisabled ||
        select_combine_vector_reduce_tile_config(
            "1", true, false, nullptr) !=
            CombineVectorReduceTileConfigStatus::kInvalid)
        return 90;

    const auto aligned_local_copy =
        combine_local_copy_plan(14336, 256, 32, true);
    const auto tail_local_copy =
        combine_local_copy_plan(14368, 256, 32, true);
    const auto disabled_local_copy =
        combine_local_copy_plan(14368, 256, 32, false);
    const auto invalid_local_copy =
        combine_local_copy_plan(14368, 255, 32, true);
    if (!aligned_local_copy.valid ||
        aligned_local_copy.vector_bytes != 14336 ||
        aligned_local_copy.scalar_begin != 14336 ||
        !tail_local_copy.valid ||
        tail_local_copy.vector_bytes != 14336 ||
        tail_local_copy.scalar_begin != 14336 ||
        !disabled_local_copy.valid ||
        disabled_local_copy.vector_bytes != 0 ||
        disabled_local_copy.scalar_begin != 0 ||
        invalid_local_copy.valid)
        return 90;

    CoreTiling tiling{};
    auto input = valid_input();
    auto status = build_core_tiling(input, &tiling);
    if (!status.ok())
        return 1;
    if (tiling.abi_version != kCoreTilingAbiVersion ||
        tiling.struct_size != sizeof(CoreTiling))
        return 2;
    if (tiling.control_launch.num_blocks != 1 ||
        tiling.control_launch.num_threads != 512 ||
        tiling.control_launch.dynamic_ub_bytes != 0 ||
        tiling.data_launch.num_blocks != 1 ||
        tiling.data_launch.num_threads != 512 ||
        tiling.data_launch.dynamic_ub_bytes != 0)
        return 35;

    input.data_num_blocks = 72;
    status = build_core_tiling(input, &tiling);
    if (!status.ok())
        return 42;
    const auto direct_pipeline = direct_dispatch_pipeline(false);
    const DirectDispatchStage expected_direct_stages[] = {
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
    };
    if (direct_pipeline.count != 13)
        return 43;
    for (std::uint32_t index = 0; index < direct_pipeline.count; ++index) {
        if (direct_pipeline.stages[index] != expected_direct_stages[index])
            return 44;
        const auto launch = direct_dispatch_stage_launch(
            tiling, direct_pipeline.stages[index]);
        const bool data_stage =
            index == 1 || index == 3 || index == 6 || index == 8 ||
            index == 10 || index == 11;
        if (launch.num_blocks != (data_stage ? 72U : 1U) ||
            launch.num_threads != 512 || launch.dynamic_ub_bytes != 0)
            return 45;
    }
    const auto cpu_sync_pipeline = direct_dispatch_pipeline(true);
    if (cpu_sync_pipeline.count != 10 ||
        cpu_sync_pipeline.stages[9] !=
            DirectDispatchStage::kEpilogueExpertPrefix)
        return 46;
    const auto dispatch_profile_pipeline =
        direct_dispatch_profile_pipeline(false);
    const DirectDispatchStage expected_dispatch_profile_stages[] = {
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
    };
    if (dispatch_profile_pipeline.count != 15)
        return 80;
    for (std::uint32_t index = 0;
         index < dispatch_profile_pipeline.count; ++index) {
        if (dispatch_profile_pipeline.stages[index] !=
            expected_dispatch_profile_stages[index])
            return 81;
    }
    const auto cpu_sync_profile_pipeline =
        direct_dispatch_profile_pipeline(true);
    if (cpu_sync_profile_pipeline.count != 12 ||
        cpu_sync_profile_pipeline.stages[11] !=
            DirectDispatchStage::kEpilogueExpertPrefix)
        return 82;
    if (direct_dispatch_release_segment(
            DirectDispatchStage::kFull, true) !=
        DirectReleaseSegment::kAll ||
        direct_dispatch_release_segment(
            DirectDispatchStage::kProducerRelease, false) !=
            DirectReleaseSegment::kAll ||
        direct_dispatch_release_segment(
            DirectDispatchStage::kProducerRelease, true) !=
            DirectReleaseSegment::kPayload ||
        direct_dispatch_release_segment(
            DirectDispatchStage::kProducerReleaseControl, true) !=
            DirectReleaseSegment::kControl ||
        direct_dispatch_release_segment(
            DirectDispatchStage::kProducerReleaseBarrier, true) !=
            DirectReleaseSegment::kBarrier)
        return 83;
    const auto epilogue_pipeline = direct_dispatch_epilogue_pipeline();
    if (epilogue_pipeline.count != 3 ||
        epilogue_pipeline.stages[0] !=
            DirectDispatchStage::kEpilogueMetadata ||
        epilogue_pipeline.stages[1] !=
            DirectDispatchStage::kEpilogueCopy ||
        epilogue_pipeline.stages[2] !=
            DirectDispatchStage::kEpilogueComplete)
        return 47;
#if !defined(DEEP_EP_ASCEND_SIMT_DEVICE)
    if (direct_dispatch_epilogue_stage(
            DirectDispatchStage::kProducerRelease) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueAcquire) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueValidate) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueValidateReduce) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueExpertCount) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueExpertPrefix) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueMetadata) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueCopy) ||
        !direct_dispatch_epilogue_stage(
            DirectDispatchStage::kEpilogueComplete))
        return 48;
#endif
    const auto combine_pipeline = direct_combine_pipeline();
    const DirectCombineStage expected_combine_stages[] = {
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
    };
    if (combine_pipeline.count != 11)
        return 49;
    for (std::uint32_t index = 0; index < combine_pipeline.count; ++index) {
        if (combine_pipeline.stages[index] != expected_combine_stages[index])
            return 50;
        const auto launch = direct_combine_stage_launch(
            tiling, combine_pipeline.stages[index]);
        const bool data_stage =
            index == 1 || index == 3 || index == 6 || index == 8 ||
            index == 9;
        if (launch.num_blocks != (data_stage ? 72U : 1U) ||
            launch.num_threads != 512 || launch.dynamic_ub_bytes != 0)
            return 51;
    }
    const auto combine_profile_pipeline = direct_combine_profile_pipeline();
    const DirectCombineStage expected_combine_profile_stages[] = {
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
    };
    if (combine_profile_pipeline.count != 14)
        return 84;
    for (std::uint32_t index = 0;
         index < combine_profile_pipeline.count; ++index) {
        if (combine_profile_pipeline.stages[index] !=
            expected_combine_profile_stages[index])
            return 85;
    }
    const auto combine_local_copy_launch = direct_combine_stage_launch(
        tiling, DirectCombineStage::kProducerLocalCopy);
    if (combine_local_copy_launch.num_blocks != 1U ||
        combine_local_copy_launch.num_threads != 512U ||
        combine_local_copy_launch.dynamic_ub_bytes != 0U)
        return 116;
    if (direct_combine_release_segment(
            DirectCombineStage::kFull, true) !=
            DirectReleaseSegment::kAll ||
        direct_combine_release_segment(
            DirectCombineStage::kProducerRelease, false) !=
            DirectReleaseSegment::kAll ||
        direct_combine_release_segment(
            DirectCombineStage::kProducerRelease, true) !=
            DirectReleaseSegment::kPayload ||
        direct_combine_release_segment(
            DirectCombineStage::kProducerReleaseControl, true) !=
            DirectReleaseSegment::kControl ||
        direct_combine_release_segment(
            DirectCombineStage::kProducerReleaseBarrier, true) !=
            DirectReleaseSegment::kBarrier ||
        direct_combine_release_segment(
            DirectCombineStage::kProducerLocalCopy, true) !=
            DirectReleaseSegment::kNone)
        return 86;
    auto representative_input = input;
    representative_input.num_tokens = 8192;
    representative_input.num_max_tokens_per_rank = 8192;
    representative_input.num_experts = 256;
    representative_input.num_topk = 8;
    representative_input.element_kind = ElementKind::kFloat8E4M3;
    representative_input.num_scale_factor_packs = 224;
    representative_input.scale_factor_pack_bytes = 4;
    representative_input.expert_alignment = 128;
    representative_input.topology.world_size = 8;
    representative_input.topology.scale_up_size = 8;
    representative_input.topology.world_rank = 0;
    representative_input.topology.scale_up_rank = 0;
    status = build_core_tiling(representative_input, &tiling);
    if (!status.ok())
        return 68;
    const auto default_workspace_bytes =
        default_elastic_runtime_workspace_bytes();
    if (default_workspace_bytes % kPublicElasticBufferAlignment != 0 ||
        default_workspace_bytes < tiling.workspace_bytes)
        return 95;
    const auto& workspace_layout = tiling.workspace_layout;
    const auto workspace_end = workspace_layout.scratch_offset +
        workspace_layout.scratch_bytes;
    const auto region_is_valid = [workspace_end](
        std::uint64_t offset, std::uint64_t bytes,
        std::uint64_t alignment) {
        return offset % alignment == 0 && offset <= workspace_end &&
            bytes <= workspace_end - offset;
    };
    const auto regions_overlap = [](
        std::uint64_t lhs_offset, std::uint64_t lhs_bytes,
        std::uint64_t rhs_offset, std::uint64_t rhs_bytes) {
        return lhs_offset < rhs_offset + rhs_bytes &&
               rhs_offset < lhs_offset + lhs_bytes;
    };
    if (workspace_layout.dispatch_receive_tile_count != 512 ||
        workspace_layout.dispatch_expert_tile_count != 512 ||
        workspace_layout.dispatch_group_expert_count != 256 ||
        workspace_layout.dispatch_group_expert_count_bytes !=
            256ULL * sizeof(std::uint32_t) ||
        workspace_layout.dispatch_route_source_counts_bytes !=
            8ULL * sizeof(std::uint64_t) ||
        !region_is_valid(
            workspace_layout.dispatch_receive_tile_error_offset,
            workspace_layout.dispatch_receive_tile_error_bytes,
            alignof(std::uint64_t)) ||
        !region_is_valid(
            workspace_layout.dispatch_group_expert_count_offset,
            workspace_layout.dispatch_group_expert_count_bytes,
            alignof(std::uint32_t)) ||
        !region_is_valid(
            workspace_layout.dispatch_route_source_counts_offset,
            workspace_layout.dispatch_route_source_counts_bytes,
            alignof(std::uint64_t)) ||
        regions_overlap(
            workspace_layout.scratch_rank_counts_offset,
            8ULL * sizeof(std::uint64_t),
            workspace_layout.dispatch_route_source_counts_offset,
            workspace_layout.dispatch_route_source_counts_bytes) ||
        !region_is_valid(
            workspace_layout.dispatch_expert_tile_count_offset,
            workspace_layout.dispatch_expert_tile_count_bytes,
            alignof(std::uint64_t)))
        return 69;

    auto combine_input = representative_input;
    combine_input.operation = OperationKind::kCombine;
    combine_input.mode_flags = mode_bit(CoreMode::kExpanded);
    status = build_core_tiling(combine_input, &tiling);
    if (!status.ok())
        return 70;
    const auto& combine_workspace = tiling.workspace_layout;
    const auto combine_workspace_end = combine_workspace.scratch_offset +
        combine_workspace.scratch_bytes;
    const auto combine_region_is_valid = [combine_workspace_end](
        std::uint64_t offset, std::uint64_t bytes,
        std::uint64_t alignment) {
        return offset % alignment == 0 && offset <= combine_workspace_end &&
            bytes <= combine_workspace_end - offset;
    };
    if (combine_workspace.combine_producer_tile_count != 512 ||
        combine_workspace.combine_receive_tile_count != 4096 ||
        !combine_region_is_valid(
            combine_workspace.combine_producer_tile_rank_count_offset,
            combine_workspace.combine_producer_tile_rank_count_bytes,
            alignof(std::uint64_t)) ||
        !combine_region_is_valid(
            combine_workspace.combine_producer_tile_error_offset,
            combine_workspace.combine_producer_tile_error_bytes,
            alignof(std::uint64_t)) ||
        !combine_region_is_valid(
            combine_workspace.combine_receive_tile_error_offset,
            combine_workspace.combine_receive_tile_error_bytes,
            alignof(std::uint64_t)) ||
        !combine_region_is_valid(
            combine_workspace.combine_receive_record_index_offset,
            combine_workspace.combine_receive_record_index_bytes,
            alignof(std::uint64_t)))
        return 71;
#if !defined(DEEP_EP_ASCEND_SIMT_DEVICE)
    for (const std::uint64_t item_count : {
             0ULL, 1ULL, 511ULL, 512ULL, 513ULL, 36871ULL}) {
        for (const std::uint32_t blocks : {1U, 72U}) {
            std::uint64_t visits[36871]{};
            for (std::uint32_t block = 0; block < blocks; ++block) {
                for (std::uint32_t thread = 0; thread < 512; ++thread) {
                    const auto work = direct_data_grid_stride(
                        block, thread, blocks, 512);
                    for (std::uint64_t item = work.first;
                         item < item_count; item += work.stride)
                        ++visits[item];
                }
            }
            for (std::uint64_t item = 0; item < item_count; ++item)
                if (visits[item] != 1)
                    return 48;
        }
    }
    {
        std::uint64_t visits[512]{};
        bool active_blocks[72]{};
        for (std::uint32_t block = 0; block < 72; ++block) {
            for (std::uint32_t thread = 0; thread < 512; ++thread) {
                const auto work = direct_block_distributed_grid_stride(
                    block, thread, 72, 512);
                for (std::uint64_t tile = work.first;
                     tile < 512; tile += work.stride) {
                    ++visits[tile];
                    active_blocks[block] = true;
                }
            }
        }
        for (std::uint64_t tile = 0; tile < 512; ++tile)
            if (visits[tile] != 1)
                return 75;
        for (const bool active : active_blocks)
            if (!active)
                return 76;
        const auto second_thread = direct_block_distributed_grid_stride(
            0, 1, 72, 512);
        if (second_thread.first != 72 || second_thread.stride != 36864)
            return 77;
    }
    for (const std::uint64_t item_count : {
             0ULL, 1ULL, 15ULL, 16ULL, 17ULL, 511ULL, 512ULL, 513ULL}) {
        for (const std::uint32_t blocks : {1U, 72U}) {
            std::uint32_t visits[513][32]{};
            for (std::uint32_t block = 0; block < blocks; ++block) {
                for (std::uint32_t thread = 0; thread < 512; ++thread) {
                    const auto work = direct_subgroup_grid_stride(
                        block, thread, blocks, 512, 32);
                    if (work.lane != thread % 32)
                        return 53;
                    for (std::uint64_t item = work.first;
                         item < item_count; item += work.stride)
                        ++visits[item][work.lane];
                }
            }
            for (std::uint64_t item = 0; item < item_count; ++item)
                for (std::uint32_t lane = 0; lane < 32; ++lane)
                    if (visits[item][lane] != 1)
                        return 54;
        }
    }
#endif

    for (const auto operation : {
             OperationKind::kDispatch, OperationKind::kCombine}) {
        input = valid_input();
        input.operation = operation;
        input.data_num_blocks = 72;
        status = build_core_tiling(input, &tiling);
        if (!status.ok() || tiling.control_launch.num_blocks != 1 ||
            tiling.data_launch.num_blocks != 72)
            return 36;

        input.data_num_blocks = 0;
        if (build_core_tiling(input, &tiling).code !=
            TilingStatusCode::kInvalidArgument)
            return 37;
        input.data_num_blocks = 73;
        if (build_core_tiling(input, &tiling).code !=
            TilingStatusCode::kInvalidArgument)
            return 38;
    }

    input = valid_input();
    input.operation = OperationKind::kBarrier;
    input.data_num_blocks = 72;
    if (build_core_tiling(input, &tiling).code !=
        TilingStatusCode::kInvalidArgument)
        return 39;
    input = valid_input();
    status = build_core_tiling(input, &tiling);
    if (!status.ok())
        return 41;
    if (tiling.token_layout.hidden_offset != 0 ||
        tiling.token_layout.hidden_bytes != 128 ||
        tiling.token_layout.scale_factor_bytes != 0 ||
        tiling.token_layout.topk_index_bytes != 2 * sizeof(std::int64_t) ||
        tiling.token_layout.topk_weight_bytes != 2 * sizeof(float) ||
        tiling.token_layout.stride_bytes == 0 ||
        !aligned(tiling.token_layout.stride_bytes))
        return 3;

    const auto& workspace = tiling.workspace_layout;
    if (!aligned(workspace.barrier_offset) ||
        !aligned(workspace.block_count_offset) ||
        !aligned(workspace.reduced_count_offset) ||
        !aligned(workspace.prefix_sum_offset) ||
        !aligned(workspace.slot_offset) ||
        !aligned(workspace.source_metadata_offset) ||
        !aligned(workspace.scratch_offset) ||
        !aligned(workspace.total_bytes) || workspace.total_bytes == 0)
        return 4;
    if (tiling.communication_buffer_bytes == 0 ||
        tiling.workspace_bytes != workspace.total_bytes)
        return 5;

    input.mode_flags = mode_bit(CoreMode::kExpanded) |
                       mode_bit(CoreMode::kCached) |
                       mode_bit(CoreMode::kZeroPadding);
    input.has_reusable_slots = true;
    status = build_core_tiling(input, &tiling);
    if (!status.ok() || !has_mode(tiling.mode_flags, CoreMode::kExpanded) ||
        !has_mode(tiling.mode_flags, CoreMode::kCached))
        return 6;

    input = valid_input();
    input.mode_flags = mode_bit(CoreMode::kZeroPadding);
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kInvalidMode)
        return 7;

    input = valid_input();
    input.mode_flags = mode_bit(CoreMode::kCached);
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kInvalidMode)
        return 8;

    input = valid_input();
    input.element_kind = ElementKind::kFloat8E4M3;
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kInvalidArgument)
        return 9;
    input.num_scale_factor_packs = 2;
    input.scale_factor_pack_bytes = 8;
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kInvalidArgument)
        return 10;
    input.scale_factor_pack_bytes = 4;
    status = build_core_tiling(input, &tiling);
    if (!status.ok() || tiling.token_layout.hidden_bytes != 64 ||
        tiling.token_layout.scale_factor_bytes != 8)
        return 17;
    input = valid_input();
    input.num_scale_factor_packs = 1;
    input.scale_factor_pack_bytes = 4;
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kInvalidArgument)
        return 18;

    input = valid_input();
    input.topology.world_size = 3;
    input.topology.scale_up_size = 3;
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kInvalidArgument)
        return 11;

    input = valid_input();
    input.hidden = std::numeric_limits<std::uint64_t>::max();
    status = build_core_tiling(input, &tiling);
    if (status.code != TilingStatusCode::kOverflow)
        return 12;

    if (build_core_tiling(valid_input(), nullptr).code !=
        TilingStatusCode::kInvalidArgument)
        return 13;

    input = valid_input();
    status = build_core_tiling(input, &tiling);
    if (!status.ok() || !validate_single_rank(tiling).ok())
        return 14;
    tiling.topology.world_size = 2;
    if (validate_single_rank(tiling).code !=
        TilingStatusCode::kUnsupportedTopology)
        return 15;

    CoreTilingInput barrier{};
    barrier.operation = OperationKind::kBarrier;
    barrier.topology.world_size = 2;
    barrier.topology.scale_up_size = 2;
    status = build_core_tiling(barrier, &tiling);
    if (!status.ok() ||
        tiling.symmetric_window_layout.abi_version !=
            kSymmetricWindowAbiVersion ||
        tiling.symmetric_window_layout.struct_size !=
            sizeof(SymmetricWindowLayout) ||
        tiling.communication_buffer_bytes !=
            tiling.symmetric_window_layout.total_bytes ||
        tiling.communication_buffer_bytes % kPublicElasticBufferAlignment != 0)
        return 16;

    std::uint64_t logical_window_bytes = 0;
    for (const auto operation : {
             OperationKind::kBarrier, OperationKind::kDispatch,
             OperationKind::kCombine}) {
        for (int rank = 0; rank < 4; ++rank) {
            input = valid_input();
            input.operation = operation;
            input.num_experts = 8;
            if (operation == OperationKind::kBarrier) {
                input.num_tokens = 0;
                input.hidden = 0;
                input.num_experts = 0;
                input.num_topk = 0;
                input.num_max_tokens_per_rank = 0;
            }
            input.topology.world_rank = rank;
            input.topology.world_size = 4;
            input.topology.scale_up_rank = rank % 2;
            input.topology.scale_up_size = 2;
            input.topology.scale_out_rank = rank / 2;
            input.topology.scale_out_size = 2;
            input.topology.kind =
                deep_ep::ascend::transport::TransportTopologyKind::
                    kLogicalSimulation;
            input.topology.epoch = 17;
            status = build_core_tiling(input, &tiling);
            if (!status.ok() ||
                tiling.symmetric_window_layout.struct_size !=
                    sizeof(SymmetricWindowLayout) ||
                tiling.symmetric_window_layout.barrier_generation_count != 4 ||
                tiling.symmetric_window_layout.barrier_completion_count != 4 ||
                (operation != OperationKind::kBarrier &&
                 (tiling.symmetric_window_layout.dispatch_receive_shard_count != 4 ||
                  tiling.symmetric_window_layout.dispatch_staging_shard_count != 4 ||
                  tiling.symmetric_window_layout.combine_receive_shard_count != 4 ||
                  tiling.symmetric_window_layout.combine_staging_shard_count != 4)) ||
                tiling.communication_buffer_bytes !=
                    tiling.symmetric_window_layout.total_bytes ||
                tiling.communication_buffer_bytes == 0)
                return 19;
            if (logical_window_bytes == 0)
                logical_window_bytes = tiling.communication_buffer_bytes;
            else if (tiling.communication_buffer_bytes != logical_window_bytes)
                return 20;
        }
        logical_window_bytes = 0;
    }

    struct ScratchFixture {
        int world_size;
        std::uint64_t dispatch_bytes;
        std::uint64_t dispatch_error_count;
        std::uint64_t dispatch_rank_bitmap_bytes;
        std::uint64_t dispatch_expert_bitmap_bytes;
        std::uint64_t combine_bytes;
    };
    for (const auto fixture : {
             ScratchFixture{2, 288, 4, 16, 32, 512},
             ScratchFixture{4, 448, 6, 32, 48, 960},
             ScratchFixture{8, 800, 10, 64, 80, 1856}}) {
        for (const auto operation : {
                 OperationKind::kDispatch, OperationKind::kCombine}) {
            input = valid_input();
            input.operation = operation;
            input.topology.world_size = fixture.world_size;
            input.topology.scale_up_size = fixture.world_size;
            input.num_experts =
                static_cast<std::uint64_t>(fixture.world_size) * 2;
            status = build_core_tiling(input, &tiling);
            if (!status.ok())
                return 17;
            const auto& rank_scratch = tiling.workspace_layout;
            const auto count =
                static_cast<std::uint64_t>(fixture.world_size);
            const bool dispatch = operation == OperationKind::kDispatch;
            const auto expected_bytes = dispatch ?
                fixture.dispatch_bytes : fixture.combine_bytes;
            if (rank_scratch.scratch_bytes != expected_bytes ||
                rank_scratch.scratch_rank_count != count ||
                rank_scratch.scratch_outbound_ingress_counts_offset != 0 ||
                rank_scratch.scratch_outbound_ingress_count != 0 ||
                rank_scratch.scratch_status_offset !=
                    rank_scratch.scratch_offset ||
                rank_scratch.scratch_local_count_offset !=
                    rank_scratch.scratch_status_offset + sizeof(std::uint64_t) ||
                rank_scratch.scratch_rank_counts_offset !=
                    rank_scratch.scratch_local_count_offset +
                        sizeof(std::uint64_t) ||
                rank_scratch.scratch_rank_values_offset !=
                    rank_scratch.scratch_rank_counts_offset +
                        count * sizeof(std::uint64_t) ||
                rank_scratch.scratch_rank_indices_offset !=
                    rank_scratch.scratch_rank_values_offset +
                        count * sizeof(std::uint64_t) ||
                rank_scratch.scratch_rank_flags_offset !=
                    rank_scratch.scratch_rank_indices_offset +
                        count * sizeof(std::int32_t) ||
                rank_scratch.scratch_rank_flags_offset + count >
                    rank_scratch.scratch_offset + rank_scratch.scratch_bytes) {
                std::cerr << "scratch mismatch operation="
                          << static_cast<int>(operation)
                          << " world_size=" << fixture.world_size
                          << " actual_bytes=" << rank_scratch.scratch_bytes
                          << " expected_bytes=" << expected_bytes << '\n';
                return 32;
            }
            if (dispatch) {
                if (rank_scratch.dispatch_error_count !=
                        fixture.dispatch_error_count ||
                    rank_scratch.dispatch_group_owner_bytes == 0 ||
                    rank_scratch.dispatch_group_tile_count != 2 ||
                    rank_scratch.dispatch_group_tile_bytes == 0 ||
                    rank_scratch.dispatch_group_error_bytes == 0 ||
                    rank_scratch.dispatch_rank_bitmap_bytes !=
                        fixture.dispatch_rank_bitmap_bytes ||
                    rank_scratch.dispatch_expert_bitmap_bytes !=
                        fixture.dispatch_expert_bitmap_bytes ||
                    rank_scratch.dispatch_error_offset %
                            alignof(std::uint64_t) != 0 ||
                    rank_scratch.dispatch_rank_bitmap_offset <
                        rank_scratch.dispatch_error_offset +
                            rank_scratch.dispatch_error_count *
                                sizeof(std::uint64_t) ||
                    rank_scratch.dispatch_expert_bitmap_offset <
                        rank_scratch.dispatch_rank_bitmap_offset +
                            rank_scratch.dispatch_rank_bitmap_bytes ||
                    rank_scratch.dispatch_expert_bitmap_offset +
                            rank_scratch.dispatch_expert_bitmap_bytes >
                        rank_scratch.scratch_offset +
                            rank_scratch.scratch_bytes ||
                    rank_scratch.combine_record_slots_offset != 0 ||
                    rank_scratch.combine_record_slots_bytes != 0)
                    return 30;
            } else if (rank_scratch.combine_record_slots_offset == 0 ||
                       rank_scratch.combine_record_slots_offset %
                               alignof(std::int32_t) != 0 ||
                       rank_scratch.combine_record_slots_bytes !=
                           tiling.dispatch_output_capacity *
                               sizeof(std::int32_t) ||
                       rank_scratch.combine_record_slots_offset +
                               rank_scratch.combine_record_slots_bytes >
                           rank_scratch.scratch_offset +
                               rank_scratch.scratch_bytes ||
                       rank_scratch.dispatch_error_offset != 0 ||
                       rank_scratch.dispatch_error_count != 0 ||
                       rank_scratch.dispatch_group_owner_offset != 0 ||
                       rank_scratch.dispatch_group_owner_bytes != 0 ||
                       rank_scratch.dispatch_group_tile_offset != 0 ||
                       rank_scratch.dispatch_group_tile_bytes != 0 ||
                       rank_scratch.dispatch_group_tile_count != 0 ||
                       rank_scratch.dispatch_group_error_offset != 0 ||
                       rank_scratch.dispatch_group_error_bytes != 0 ||
                       rank_scratch.dispatch_rank_bitmap_offset != 0 ||
                       rank_scratch.dispatch_rank_bitmap_bytes != 0 ||
                       rank_scratch.dispatch_expert_bitmap_offset != 0 ||
                       rank_scratch.dispatch_expert_bitmap_bytes != 0) {
                return 31;
            }
        }
    }

    input = valid_input();
    input.mode_flags = mode_bit(CoreMode::kHybrid);
    input.num_experts = 8;
    input.topology.world_rank = 0;
    input.topology.world_size = 4;
    input.topology.scale_up_rank = 0;
    input.topology.scale_up_size = 2;
    input.topology.scale_out_rank = 0;
    input.topology.scale_out_size = 2;
    input.topology.kind =
        deep_ep::ascend::transport::TransportTopologyKind::kLogicalSimulation;
    input.data_num_blocks = 72;
    if (build_core_tiling(input, &tiling).code !=
        TilingStatusCode::kInvalidArgument)
        return 40;
    input.data_num_blocks = 1;
    status = build_core_tiling(input, &tiling);
    if (!status.ok())
        return 21;
    const auto& hybrid_scratch = tiling.workspace_layout;
    if (hybrid_scratch.scratch_outbound_ingress_count != 4 ||
        hybrid_scratch.scratch_outbound_ingress_counts_offset !=
            hybrid_scratch.scratch_rank_values_offset +
                4 * sizeof(std::uint64_t) ||
        hybrid_scratch.scratch_rank_indices_offset !=
            hybrid_scratch.scratch_outbound_ingress_counts_offset +
                4 * sizeof(std::uint64_t) ||
        hybrid_scratch.scratch_bytes != 160 ||
        hybrid_scratch.dispatch_error_offset != 0 ||
        hybrid_scratch.dispatch_error_count != 0 ||
        hybrid_scratch.dispatch_group_owner_offset != 0 ||
        hybrid_scratch.dispatch_group_owner_bytes != 0 ||
        hybrid_scratch.dispatch_group_tile_offset != 0 ||
        hybrid_scratch.dispatch_group_tile_bytes != 0 ||
        hybrid_scratch.dispatch_group_tile_count != 0 ||
        hybrid_scratch.dispatch_group_error_offset != 0 ||
        hybrid_scratch.dispatch_group_error_bytes != 0 ||
        hybrid_scratch.dispatch_rank_bitmap_offset != 0 ||
        hybrid_scratch.dispatch_rank_bitmap_bytes != 0 ||
        hybrid_scratch.dispatch_expert_bitmap_offset != 0 ||
        hybrid_scratch.dispatch_expert_bitmap_bytes != 0 ||
        hybrid_scratch.scratch_outbound_ingress_counts_offset +
                4 * sizeof(std::uint64_t) >
            hybrid_scratch.scratch_offset + hybrid_scratch.scratch_bytes)
        return 22;

    return 0;
}
