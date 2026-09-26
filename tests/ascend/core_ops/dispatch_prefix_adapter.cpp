// Exercise the production prefix launcher with synthetic grouping outputs.
#include "csrc/backends/ascend/elastic/kernels.hpp"

namespace deep_ep::ascend::elastic {
extern "C" int deep_ep_ascend_launch_dispatch_producer_prefix(
    DispatchArguments, CoreTiling, void*, CoreLaunchShape, std::uint32_t);
}

extern "C" int probe_dispatch_prefix(
    void* entry, std::uint8_t* workspace, std::uint8_t* staging,
    int rank, int world, std::uint64_t tiles, std::uint64_t capacity,
    std::uint32_t threads, std::uint32_t early_route, void* stream) {
    using namespace deep_ep::ascend;
    elastic::DispatchArguments arguments{};
    arguments.workspace = workspace;
    arguments.generation = 7;
    arguments.early_route_plan = early_route;
    elastic::CoreTiling tiling{};
    tiling.num_topk = 8;
    tiling.num_experts = world * 4;
    tiling.num_max_tokens_per_rank = capacity;
    tiling.launch.num_threads = threads;
    tiling.transport_context = transport::make_device_transport_context();
    tiling.transport_context.topology.world_rank = rank;
    tiling.transport_context.topology.world_size = world;
    tiling.transport_context.local_window_base =
        reinterpret_cast<std::uintptr_t>(staging);
    auto& layout = tiling.workspace_layout;
    layout.scratch_status_offset = 0;
    layout.scratch_rank_counts_offset = 256;
    layout.scratch_rank_values_offset = 512;
    layout.dispatch_error_offset = 768;
    layout.dispatch_group_expert_count_offset = 1024;
    layout.dispatch_group_expert_count = world * 4;
    layout.dispatch_group_tile_offset = 4096;
    layout.dispatch_group_tile_count = tiles;
    layout.dispatch_group_error_offset = 4096 + tiles * world * 8 + 256;
    auto& window = tiling.symmetric_window_layout;
    window.dispatch_staging_offset = 256;
    window.dispatch_staging_shard_bytes = 128;
    window.dispatch_route_plan_slot_bytes = 64;
    window.dispatch_route_plan_expert_capacity = 4;
    elastic::CoreLaunchShape launch{};
    launch.num_blocks = 1;
    const auto function = reinterpret_cast<decltype(
        &elastic::deep_ep_ascend_launch_dispatch_producer_prefix)>(entry);
    return function(arguments, tiling, stream, launch, 0);
}
