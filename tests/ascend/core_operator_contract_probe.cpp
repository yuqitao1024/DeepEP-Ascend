#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/combine_parallel.hpp"
#include "csrc/backends/ascend/elastic/dispatch_parallel.hpp"
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
    if (!topk_grouping_reference_contract())
        return 52;
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
    if (cpu_sync_pipeline.count != 6 ||
        cpu_sync_pipeline.stages[5] !=
            DirectDispatchStage::kEpilogueAcquire)
        return 46;
    const auto epilogue_pipeline = direct_dispatch_epilogue_pipeline();
    if (epilogue_pipeline.count != 8 ||
        epilogue_pipeline.stages[0] !=
            DirectDispatchStage::kEpilogueAcquire ||
        epilogue_pipeline.stages[1] !=
            DirectDispatchStage::kEpilogueValidate ||
        epilogue_pipeline.stages[2] !=
            DirectDispatchStage::kEpilogueValidateReduce ||
        epilogue_pipeline.stages[3] !=
            DirectDispatchStage::kEpilogueExpertCount ||
        epilogue_pipeline.stages[4] !=
            DirectDispatchStage::kEpilogueExpertPrefix ||
        epilogue_pipeline.stages[5] !=
            DirectDispatchStage::kEpilogueMetadata ||
        epilogue_pipeline.stages[6] !=
            DirectDispatchStage::kEpilogueCopy ||
        epilogue_pipeline.stages[7] !=
            DirectDispatchStage::kEpilogueComplete)
        return 47;
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
