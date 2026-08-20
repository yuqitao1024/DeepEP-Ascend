#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/dispatch_parallel.hpp"
#include "csrc/backends/ascend/elastic/layout.hpp"
#include "csrc/backends/ascend/elastic/tiling.hpp"

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

}  // namespace

int main() {
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

    CoreTiling tiling{};
    auto input = valid_input();
    auto status = build_core_tiling(input, &tiling);
    if (!status.ok())
        return 1;
    if (tiling.abi_version != kCoreTilingAbiVersion ||
        tiling.struct_size != sizeof(CoreTiling))
        return 2;
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
             ScratchFixture{2, 160, 4, 16, 32, 64},
             ScratchFixture{4, 256, 6, 32, 48, 128},
             ScratchFixture{8, 416, 10, 64, 80, 192}}) {
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
                            rank_scratch.scratch_bytes)
                    return 30;
            } else if (rank_scratch.dispatch_error_offset != 0 ||
                       rank_scratch.dispatch_error_count != 0 ||
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
