#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "csrc/backends/ascend/elastic/combine_state.hpp"

using namespace deep_ep::ascend::elastic;

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    static_assert(std::is_standard_layout_v<SymmetricWindowLayout>);
    static_assert(std::is_trivially_copyable_v<SymmetricWindowLayout>);
    static_assert(std::is_standard_layout_v<SymmetricControlHeader>);
    static_assert(std::is_trivially_copyable_v<SymmetricControlHeader>);
    static_assert(sizeof(SymmetricControlHeader) == 32);
    static_assert(kSymmetricWindowAbiVersion == 7);
    static_assert(offsetof(SymmetricWindowLayout, abi_version) == 0);
    static_assert(offsetof(SymmetricWindowLayout, struct_size) == 4);
    static_assert(offsetof(SymmetricWindowLayout, control_offset) == 8);
    static_assert(offsetof(SymmetricWindowLayout, control_bytes) == 16);
    static_assert(offsetof(SymmetricWindowLayout, dispatch_offset) == 24);
    static_assert(offsetof(SymmetricWindowLayout, dispatch_record_bytes) == 32);
    static_assert(
        offsetof(SymmetricWindowLayout, dispatch_source_shard_bytes) == 40);
    static_assert(
        offsetof(SymmetricWindowLayout, dispatch_source_shard_count) == 48);
    static_assert(offsetof(SymmetricWindowLayout, dispatch_bytes) == 56);
    static_assert(offsetof(SymmetricWindowLayout, combine_offset) == 64);
    static_assert(offsetof(SymmetricWindowLayout, combine_record_bytes) == 72);
    static_assert(offsetof(
        SymmetricWindowLayout, combine_contributor_shard_bytes) == 80);
    static_assert(offsetof(
        SymmetricWindowLayout, combine_contributor_shard_count) == 88);
    static_assert(offsetof(SymmetricWindowLayout, combine_bytes) == 96);
    static_assert(offsetof(SymmetricWindowLayout, reserve_offset) == 104);
    static_assert(offsetof(SymmetricWindowLayout, reserve_bytes) == 112);
    static_assert(offsetof(SymmetricWindowLayout, total_bytes) == 120);
    static_assert(
        offsetof(SymmetricWindowLayout, combine_weight_offset) == 288);

    SymmetricWindowInput input{};
    input.world_size = 2;
    input.num_max_tokens_per_rank = 8;
    input.hidden = 64;
    input.num_topk = 2;
    input.element_bytes = 2;

    SymmetricWindowLayout layout{};
    CHECK(build_symmetric_window_layout(input, &layout).ok());
    CHECK(layout.abi_version == kSymmetricWindowAbiVersion);
    CHECK(layout.struct_size == sizeof(SymmetricWindowLayout));
    CHECK(layout.dispatch_source_shard_count == 2);
    CHECK(layout.dispatch_receive_shard_count == 2);
    CHECK(layout.dispatch_staging_shard_count == 2);
    CHECK(layout.combine_contributor_shard_count == 2);
    CHECK(layout.dispatch_receive_shard_bytes >=
          input.num_max_tokens_per_rank *
              layout.dispatch_record_bytes);
    CHECK(layout.dispatch_staging_shard_bytes >=
          input.num_max_tokens_per_rank *
              layout.dispatch_record_bytes);
    CHECK(layout.dispatch_control_bytes ==
          input.world_size * sizeof(DispatchControlSlot));
    CHECK(layout.combine_contributor_shard_bytes >=
          input.num_max_tokens_per_rank * layout.combine_record_bytes);
    CHECK(layout.combine_control_bytes ==
          input.world_size * sizeof(CombineControlSlot));
    CHECK(layout.combine_receive_shard_count == 2);
    CHECK(layout.combine_staging_shard_count == 2);
    CHECK(layout.combine_weight_offset % kAscendElasticAlignment == 0);
    CHECK(layout.combine_weight_offset >= input.hidden * input.element_bytes);
    CHECK(layout.combine_weight_offset + input.num_topk * sizeof(float) <=
          layout.combine_record_bytes - kAscendElasticAlignment);
    CHECK(layout.combine_receive_shard_bytes >=
          input.num_max_tokens_per_rank * layout.combine_record_bytes);
    CHECK(layout.combine_staging_shard_bytes >=
          input.num_max_tokens_per_rank * layout.combine_record_bytes);
    CHECK(layout.control_offset % kAscendElasticAlignment == 0);
    CHECK(layout.control_bytes >= sizeof(SymmetricControlHeader));
    CHECK(layout.barrier_generation_count == input.world_size);
    CHECK(layout.barrier_generation_bytes ==
          input.world_size * sizeof(std::uint64_t));
    CHECK(layout.barrier_completion_count == input.world_size);
    CHECK(layout.barrier_completion_bytes ==
          input.world_size * sizeof(std::uint64_t));
    CHECK(layout.barrier_generation_offset % kAscendElasticAlignment == 0);
    CHECK(layout.barrier_completion_offset % kAscendElasticAlignment == 0);
    CHECK(layout.control_offset + sizeof(SymmetricControlHeader) <=
          layout.barrier_generation_offset);
    CHECK(layout.barrier_generation_offset +
              layout.barrier_generation_bytes <=
          layout.barrier_completion_offset);
    CHECK(layout.barrier_completion_offset +
              layout.barrier_completion_bytes <=
          layout.dispatch_control_offset);
    std::uint64_t rank_slot = 0;
    CHECK(checked_rank_slot_offset(
        layout.barrier_completion_offset,
        layout.barrier_completion_count, 1, &rank_slot));
    CHECK(rank_slot ==
          layout.barrier_completion_offset + sizeof(std::uint64_t));
    CHECK(!checked_rank_slot_offset(
        layout.barrier_completion_offset,
        layout.barrier_completion_count, -1, &rank_slot));
    CHECK(!checked_rank_slot_offset(
        layout.barrier_completion_offset,
        layout.barrier_completion_count,
        static_cast<int>(layout.barrier_completion_count), &rank_slot));
    CHECK(!checked_rank_slot_offset(
        std::numeric_limits<std::uint64_t>::max() - 4, 2, 1,
        &rank_slot));
    CHECK(layout.dispatch_offset % kAscendElasticAlignment == 0);
    CHECK(layout.dispatch_receive_offset % kAscendElasticAlignment == 0);
    CHECK(layout.dispatch_staging_offset % kAscendElasticAlignment == 0);
    CHECK(layout.combine_offset % kAscendElasticAlignment == 0);
    CHECK(layout.combine_control_offset % kAscendElasticAlignment == 0);
    CHECK(layout.combine_receive_offset % kAscendElasticAlignment == 0);
    CHECK(layout.combine_staging_offset % kAscendElasticAlignment == 0);
    CHECK(layout.reserve_offset % kAscendElasticAlignment == 0);
    CHECK(layout.control_offset + layout.control_bytes <=
          layout.dispatch_offset);
    CHECK(layout.dispatch_offset <= layout.dispatch_receive_offset);
    CHECK(layout.dispatch_receive_offset + layout.dispatch_receive_bytes <=
          layout.dispatch_staging_offset);
    CHECK(layout.dispatch_staging_offset + layout.dispatch_staging_bytes <=
          layout.dispatch_offset + layout.dispatch_bytes);
    CHECK(layout.dispatch_offset + layout.dispatch_bytes <=
          layout.combine_offset);
    CHECK(layout.combine_offset <= layout.combine_control_offset);
    CHECK(layout.combine_control_offset + layout.combine_control_bytes <=
          layout.combine_receive_offset);
    CHECK(layout.combine_receive_offset + layout.combine_receive_bytes <=
          layout.combine_staging_offset);
    CHECK(layout.combine_staging_offset + layout.combine_staging_bytes <=
          layout.reserve_offset);
    CHECK(layout.combine_offset + layout.combine_bytes <=
          layout.reserve_offset);
    CHECK(layout.total_bytes % kPublicElasticBufferAlignment == 0);

    // These literals preserve the direct ABI-v4 geometry before hybrid tails.
    SymmetricWindowInput direct_input{};
    direct_input.world_size = 4;
    direct_input.num_max_tokens_per_rank = 8;
    direct_input.hidden = 128;
    direct_input.num_topk = 2;
    direct_input.element_bytes = 2;
    SymmetricWindowLayout direct_layout{};
    CHECK(build_symmetric_window_layout(direct_input, &direct_layout).ok());
    CHECK(direct_layout.control_offset == 0);
    CHECK(direct_layout.control_bytes == 160);
    CHECK(direct_layout.dispatch_offset == 160);
    CHECK(direct_layout.dispatch_record_bytes == 352);
    CHECK(direct_layout.dispatch_source_shard_bytes == 2816);
    CHECK(direct_layout.dispatch_source_shard_count == 4);
    CHECK(direct_layout.dispatch_bytes == 22528);
    CHECK(direct_layout.combine_offset == 22688);
    CHECK(direct_layout.combine_record_bytes == 320);
    CHECK(direct_layout.combine_contributor_shard_bytes == 2560);
    CHECK(direct_layout.combine_contributor_shard_count == 4);
    CHECK(direct_layout.combine_bytes == 20544);
    CHECK(direct_layout.reserve_offset == 43232);
    CHECK(direct_layout.reserve_bytes == 32);
    CHECK(direct_layout.total_bytes == 2097152);
    CHECK(direct_layout.dispatch_control_offset == 96);
    CHECK(direct_layout.dispatch_control_bytes == 64);
    CHECK(direct_layout.dispatch_receive_offset == 160);
    CHECK(direct_layout.dispatch_receive_shard_bytes == 2816);
    CHECK(direct_layout.dispatch_receive_shard_count == 4);
    CHECK(direct_layout.dispatch_receive_bytes == 11264);
    CHECK(direct_layout.dispatch_staging_offset == 11424);
    CHECK(direct_layout.dispatch_staging_shard_bytes == 2816);
    CHECK(direct_layout.dispatch_staging_shard_count == 4);
    CHECK(direct_layout.dispatch_staging_bytes == 11264);
    CHECK(direct_layout.combine_control_offset == 22688);
    CHECK(direct_layout.combine_control_bytes == 64);
    CHECK(direct_layout.combine_receive_offset == 22752);
    CHECK(direct_layout.combine_receive_shard_bytes == 2560);
    CHECK(direct_layout.combine_receive_shard_count == 4);
    CHECK(direct_layout.combine_receive_bytes == 10240);
    CHECK(direct_layout.combine_staging_offset == 32992);
    CHECK(direct_layout.combine_staging_shard_bytes == 2560);
    CHECK(direct_layout.combine_staging_shard_count == 4);
    CHECK(direct_layout.combine_staging_bytes == 10240);
    CHECK(direct_layout.combine_weight_offset == 256);
    CHECK(direct_layout.barrier_generation_offset == 32);
    CHECK(direct_layout.barrier_generation_bytes == 32);
    CHECK(direct_layout.barrier_generation_count == 4);
    CHECK(direct_layout.barrier_completion_offset == 64);
    CHECK(direct_layout.barrier_completion_bytes == 32);
    CHECK(direct_layout.barrier_completion_count == 4);
    CHECK(direct_layout.hybrid_route_record_offset == 0);
    CHECK(direct_layout.hybrid_route_record_bytes == 0);
    CHECK(direct_layout.hybrid_route_record_count == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_control_offset == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_control_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_control_count == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_shard_offset == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_shard_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_shard_count == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_control_offset == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_control_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_control_count == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_shard_offset == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_shard_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_shard_count == 0);
    CHECK(direct_layout.hybrid_dispatch_forward_bytes == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_control_offset == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_control_bytes == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_control_count == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_shard_offset == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_shard_bytes == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_shard_count == 0);
    CHECK(direct_layout.hybrid_combine_reverse_forward_bytes == 0);
    CHECK(direct_layout.hybrid_combine_return_control_offset == 0);
    CHECK(direct_layout.hybrid_combine_return_control_bytes == 0);
    CHECK(direct_layout.hybrid_combine_return_control_count == 0);
    CHECK(direct_layout.hybrid_combine_return_shard_offset == 0);
    CHECK(direct_layout.hybrid_combine_return_shard_bytes == 0);
    CHECK(direct_layout.hybrid_combine_return_shard_count == 0);
    CHECK(direct_layout.hybrid_combine_return_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_staging_offset == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_staging_shard_bytes == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_staging_shard_count == 0);
    CHECK(direct_layout.hybrid_dispatch_ingress_staging_bytes == 0);

    auto small_direct_input = direct_input;
    small_direct_input.world_size = 2;
    small_direct_input.num_max_tokens_per_rank = 1;
    small_direct_input.hidden = 1;
    small_direct_input.num_topk = 1;
    SymmetricWindowLayout small_direct_layout{};
    CHECK(build_symmetric_window_layout(
              small_direct_input, &small_direct_layout).ok());
    CHECK(small_direct_layout.combine_record_bytes == 96);
    auto small_hybrid_input = small_direct_input;
    small_hybrid_input.hybrid = true;
    small_hybrid_input.hybrid_route_capacity = 2;
    SymmetricWindowLayout small_hybrid_layout{};
    CHECK(build_symmetric_window_layout(
              small_hybrid_input, &small_hybrid_layout).ok());
    CHECK(small_hybrid_layout.combine_record_bytes == 128);

    auto hybrid_input = direct_input;
    hybrid_input.hybrid = true;
    hybrid_input.hybrid_route_capacity = 40;
    SymmetricWindowLayout hybrid_layout{};
    CHECK(build_symmetric_window_layout(hybrid_input, &hybrid_layout).ok());
    CHECK(hybrid_layout.control_offset == direct_layout.control_offset);
    CHECK(hybrid_layout.control_bytes == direct_layout.control_bytes);
    CHECK(hybrid_layout.dispatch_offset == direct_layout.dispatch_offset);
    CHECK(hybrid_layout.dispatch_bytes == direct_layout.dispatch_bytes);
    CHECK(hybrid_layout.combine_offset == direct_layout.combine_offset);
    CHECK(hybrid_layout.combine_record_bytes == 352);
    CHECK(hybrid_layout.combine_contributor_shard_bytes == 2816);
    CHECK(hybrid_layout.combine_receive_shard_bytes == 2816);
    CHECK(hybrid_layout.combine_staging_shard_bytes == 2816);
    CHECK(hybrid_layout.combine_bytes == 22592);
    CHECK(hybrid_layout.reserve_offset == 45280);
    CHECK(hybrid_layout.reserve_bytes == direct_layout.reserve_bytes);
    CHECK(hybrid_layout.total_bytes >= direct_layout.total_bytes);
    CHECK(hybrid_layout.hybrid_route_record_offset == 45312);
    CHECK(hybrid_layout.hybrid_route_record_count == 40);
    CHECK(hybrid_layout.hybrid_route_record_bytes == 2560);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_control_offset == 47872);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_control_count == 4);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_control_bytes == 64);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_shard_offset == 47936);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_shard_count == 4);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_shard_bytes == 2816);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_bytes == 11264);
    CHECK(hybrid_layout.hybrid_dispatch_forward_control_offset == 59200);
    CHECK(hybrid_layout.hybrid_dispatch_forward_control_count == 4);
    CHECK(hybrid_layout.hybrid_dispatch_forward_control_bytes == 64);
    CHECK(hybrid_layout.hybrid_dispatch_forward_shard_offset == 59264);
    CHECK(hybrid_layout.hybrid_dispatch_forward_shard_count == 4);
    CHECK(hybrid_layout.hybrid_dispatch_forward_shard_bytes == 2816);
    CHECK(hybrid_layout.hybrid_dispatch_forward_bytes == 11264);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_control_offset == 70528);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_control_count == 4);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_control_bytes == 64);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_shard_offset == 70592);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_shard_count == 4);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_shard_bytes == 2816);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_bytes == 11264);
    CHECK(hybrid_layout.hybrid_combine_return_control_offset == 81856);
    CHECK(hybrid_layout.hybrid_combine_return_control_count == 4);
    CHECK(hybrid_layout.hybrid_combine_return_control_bytes == 64);
    CHECK(hybrid_layout.hybrid_combine_return_shard_offset == 81920);
    CHECK(hybrid_layout.hybrid_combine_return_shard_count == 4);
    CHECK(hybrid_layout.hybrid_combine_return_shard_bytes == 2816);
    CHECK(hybrid_layout.hybrid_combine_return_bytes == 11264);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_staging_offset == 93184);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_staging_shard_count == 4);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_staging_shard_bytes == 2816);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_staging_bytes == 11264);
    CHECK(hybrid_layout.hybrid_route_record_offset >=
          hybrid_layout.reserve_offset + hybrid_layout.reserve_bytes);
    CHECK(hybrid_layout.hybrid_route_record_offset +
              hybrid_layout.hybrid_route_record_bytes <=
          hybrid_layout.hybrid_dispatch_ingress_control_offset);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_control_offset +
              hybrid_layout.hybrid_dispatch_ingress_control_bytes <=
          hybrid_layout.hybrid_dispatch_ingress_shard_offset);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_shard_offset +
              hybrid_layout.hybrid_dispatch_ingress_bytes <=
          hybrid_layout.hybrid_dispatch_forward_control_offset);
    CHECK(hybrid_layout.hybrid_dispatch_forward_control_offset +
              hybrid_layout.hybrid_dispatch_forward_control_bytes <=
          hybrid_layout.hybrid_dispatch_forward_shard_offset);
    CHECK(hybrid_layout.hybrid_dispatch_forward_shard_offset +
              hybrid_layout.hybrid_dispatch_forward_bytes <=
          hybrid_layout.hybrid_combine_reverse_forward_control_offset);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_control_offset +
              hybrid_layout.hybrid_combine_reverse_forward_control_bytes <=
          hybrid_layout.hybrid_combine_reverse_forward_shard_offset);
    CHECK(hybrid_layout.hybrid_combine_reverse_forward_shard_offset +
              hybrid_layout.hybrid_combine_reverse_forward_bytes <=
          hybrid_layout.hybrid_combine_return_control_offset);
    CHECK(hybrid_layout.hybrid_combine_return_control_offset +
              hybrid_layout.hybrid_combine_return_control_bytes <=
          hybrid_layout.hybrid_combine_return_shard_offset);
    CHECK(hybrid_layout.hybrid_combine_return_shard_offset +
              hybrid_layout.hybrid_combine_return_bytes <=
          hybrid_layout.hybrid_dispatch_ingress_staging_offset);
    CHECK(hybrid_layout.hybrid_dispatch_ingress_staging_offset +
              hybrid_layout.hybrid_dispatch_ingress_staging_bytes <=
          hybrid_layout.total_bytes);

    auto larger = input;
    larger.num_max_tokens_per_rank *= 2;
    SymmetricWindowLayout larger_layout{};
    CHECK(build_symmetric_window_layout(larger, &larger_layout).ok());
    CHECK(larger_layout.dispatch_receive_shard_bytes >
          layout.dispatch_receive_shard_bytes);
    CHECK(larger_layout.dispatch_staging_shard_bytes >
          layout.dispatch_staging_shard_bytes);
    CHECK(larger_layout.combine_contributor_shard_bytes >
          layout.combine_contributor_shard_bytes);
    CHECK(larger_layout.combine_receive_shard_bytes >
          layout.combine_receive_shard_bytes);
    CHECK(larger_layout.combine_staging_shard_bytes >
          layout.combine_staging_shard_bytes);
    CHECK(larger_layout.total_bytes >= layout.total_bytes);

    auto conservative = input;
    conservative.num_topk = 0;
    SymmetricWindowLayout conservative_layout{};
    CHECK(build_symmetric_window_layout(
              conservative, &conservative_layout).ok());
    CHECK(conservative_layout.dispatch_source_shard_bytes > 0);

    auto expanded_single_reduction = input;
    expanded_single_reduction.expanded = true;
    expanded_single_reduction.allow_multiple_reduction = false;
    SymmetricWindowLayout expanded_single_reduction_layout{};
    CHECK(build_symmetric_window_layout(
              expanded_single_reduction,
              &expanded_single_reduction_layout).ok());
    CHECK(expanded_single_reduction_layout.combine_receive_shard_bytes >=
          input.num_max_tokens_per_rank * input.num_topk *
              expanded_single_reduction_layout.combine_record_bytes);
    CHECK(expanded_single_reduction_layout.combine_staging_shard_bytes >=
          input.num_max_tokens_per_rank * input.num_topk *
              expanded_single_reduction_layout.combine_record_bytes);

    auto expanded_multiple_reduction = expanded_single_reduction;
    expanded_multiple_reduction.allow_multiple_reduction = true;
    SymmetricWindowLayout expanded_multiple_reduction_layout{};
    CHECK(build_symmetric_window_layout(
              expanded_multiple_reduction,
              &expanded_multiple_reduction_layout).ok());
    CHECK(expanded_multiple_reduction_layout.combine_receive_shard_bytes >=
          input.num_max_tokens_per_rank *
              expanded_multiple_reduction_layout.combine_record_bytes);
    CHECK(expanded_multiple_reduction_layout.combine_receive_shard_bytes <
          expanded_single_reduction_layout.combine_receive_shard_bytes);

    std::uint64_t previous_control_bytes = 0;
    for (const std::uint32_t world_size : {2U, 3U, 4U, 8U}) {
        auto parameterized = input;
        parameterized.world_size = world_size;
        SymmetricWindowLayout parameterized_layout{};
        CHECK(build_symmetric_window_layout(
                  parameterized, &parameterized_layout).ok());
        CHECK(parameterized_layout.barrier_generation_count == world_size);
        CHECK(parameterized_layout.barrier_generation_bytes ==
              world_size * sizeof(std::uint64_t));
        CHECK(parameterized_layout.barrier_completion_count == world_size);
        CHECK(parameterized_layout.barrier_completion_bytes ==
              world_size * sizeof(std::uint64_t));
        CHECK(parameterized_layout.dispatch_control_bytes ==
              world_size * sizeof(DispatchControlSlot));
        CHECK(parameterized_layout.dispatch_receive_shard_count == world_size);
        CHECK(parameterized_layout.dispatch_staging_shard_count == world_size);
        CHECK(parameterized_layout.combine_receive_shard_count == world_size);
        CHECK(parameterized_layout.combine_staging_shard_count == world_size);
        CHECK(parameterized_layout.control_bytes >= previous_control_bytes);
        previous_control_bytes = parameterized_layout.control_bytes;
    }

    auto invalid = input;
    invalid.world_size = 1;
    CHECK(build_symmetric_window_layout(invalid, &layout).code ==
          LayoutStatusCode::kInvalidArgument);
    invalid.world_size = 0;
    CHECK(build_symmetric_window_layout(invalid, &layout).code ==
          LayoutStatusCode::kInvalidArgument);
    CHECK(build_symmetric_window_layout(input, nullptr).code ==
          LayoutStatusCode::kInvalidArgument);

    auto overflow = input;
    overflow.hidden = std::numeric_limits<std::uint64_t>::max();
    CHECK(build_symmetric_window_layout(overflow, &layout).code ==
          LayoutStatusCode::kOverflow);
    overflow = input;
    overflow.num_max_tokens_per_rank =
        std::numeric_limits<std::uint64_t>::max();
    CHECK(build_symmetric_window_layout(overflow, &layout).code ==
          LayoutStatusCode::kOverflow);
    overflow = expanded_single_reduction;
    overflow.num_max_tokens_per_rank =
        std::numeric_limits<std::uint64_t>::max() / input.num_topk + 1;
    CHECK(build_symmetric_window_layout(overflow, &layout).code ==
          LayoutStatusCode::kOverflow);
    overflow = direct_input;
    overflow.hybrid = true;
    overflow.hybrid_route_capacity = std::numeric_limits<std::uint64_t>::max();
    CHECK(build_symmetric_window_layout(overflow, &layout).code ==
          LayoutStatusCode::kOverflow);

    SymmetricWindowInput barrier{};
    barrier.world_size = 2;
    SymmetricWindowLayout barrier_layout{};
    CHECK(build_symmetric_window_layout(barrier, &barrier_layout).ok());
    CHECK(barrier_layout.control_bytes > 0);
    CHECK(barrier_layout.dispatch_bytes == 0);
    CHECK(barrier_layout.combine_bytes == 0);
    CHECK(barrier_layout.total_bytes % kPublicElasticBufferAlignment == 0);
    return 0;
}
