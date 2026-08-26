#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/combine_parallel.hpp"
#include "csrc/backends/ascend/elastic/dispatch_parallel.hpp"
#include "csrc/backends/ascend/elastic/dispatch_pipeline_config.hpp"
#include "csrc/backends/ascend/elastic/layout.hpp"
#include "csrc/backends/ascend/elastic/kernels.hpp"
#include "csrc/backends/ascend/elastic/tiling.hpp"
#include "csrc/backends/ascend/elastic/topk_grouping.hpp"

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
                DispatchDevicePrefixConfigStatus::kDisabled ||
        device_prefix_config.enabled ||
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
    for (const auto baseline_value : {static_cast<const char*>(nullptr),
                                      "512"}) {
        if (select_dispatch_consumer_tile_config(
                baseline_value, true, false, true, false, false, false,
                &consumer_tile_config) !=
                    DispatchConsumerTileConfigStatus::kDisabled ||
            consumer_tile_config.tile_bytes != 512)
            return 100;
    }
    for (const auto candidate_value : {"1024", "2048", "4096"}) {
        if (select_dispatch_consumer_tile_config(
                candidate_value, true, false, true, false, false, false,
                &consumer_tile_config) !=
                    DispatchConsumerTileConfigStatus::kEnabled)
            return 101;
    }
    for (const auto invalid_value :
         {"", "0", "256", "8192", " 1024", "+1024", "1024x"}) {
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
    for (const auto baseline_value : {static_cast<const char*>(nullptr),
                                      "0"}) {
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
    if (combine_profile_pipeline.count != 13)
        return 84;
    for (std::uint32_t index = 0;
         index < combine_profile_pipeline.count; ++index) {
        if (combine_profile_pipeline.stages[index] !=
            expected_combine_profile_stages[index])
            return 85;
    }
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
            DirectReleaseSegment::kBarrier)
        return 86;
    auto representative_input = input;
    representative_input.num_tokens = 8192;
    representative_input.num_max_tokens_per_rank = 8192;
    representative_input.num_experts = 256;
    representative_input.num_topk = 8;
    representative_input.expert_alignment = 128;
    representative_input.topology.world_size = 8;
    representative_input.topology.scale_up_size = 8;
    representative_input.topology.world_rank = 0;
    representative_input.topology.scale_up_rank = 0;
    status = build_core_tiling(representative_input, &tiling);
    if (!status.ok())
        return 68;
    const auto& workspace_layout = tiling.workspace_layout;
    const auto workspace_end = workspace_layout.scratch_offset +
        workspace_layout.scratch_bytes;
    const auto region_is_valid = [workspace_end](
        std::uint64_t offset, std::uint64_t bytes,
        std::uint64_t alignment) {
        return offset % alignment == 0 && offset <= workspace_end &&
            bytes <= workspace_end - offset;
    };
    if (workspace_layout.dispatch_receive_tile_count != 512 ||
        workspace_layout.dispatch_expert_tile_count != 512 ||
        !region_is_valid(
            workspace_layout.dispatch_receive_tile_error_offset,
            workspace_layout.dispatch_receive_tile_error_bytes,
            alignof(std::uint64_t)) ||
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
