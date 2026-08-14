#include <cstdint>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/layout.hpp"
#include "csrc/backends/ascend/elastic/tiling.hpp"

using namespace deep_ep::ascend::elastic;

static_assert(std::is_standard_layout_v<TokenLayout>);
static_assert(std::is_trivially_copyable_v<TokenLayout>);
static_assert(std::is_standard_layout_v<WorkspaceLayout>);
static_assert(std::is_trivially_copyable_v<WorkspaceLayout>);
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
    if (!status.ok() || tiling.token_layout.hidden_bytes != 64 ||
        tiling.token_layout.scale_factor_bytes != 16)
        return 10;

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

    return 0;
}
