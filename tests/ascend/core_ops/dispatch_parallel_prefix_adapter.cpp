// Exercise the production epilogue parallel-prefix launcher with synthetic
// expert tile counts.
#include "csrc/backends/ascend/elastic/kernels.hpp"

namespace deep_ep::ascend::elastic {
#include "csrc/backends/ascend/elastic/dispatch_vf_launchers.hpp"
}

extern "C" int probe_dispatch_parallel_prefix(
    void* entry, std::uint8_t* workspace,
    std::int32_t* prefix_per_rank, std::int32_t* prefix_per_expert,
    std::int32_t* unaligned_per_expert,
    int rank, int world, std::uint64_t experts, std::uint64_t alignment,
    std::uint64_t capacity, std::uint64_t tiles, std::uint32_t blocks,
    std::uint32_t threads, void* stream, std::uint64_t status_offset,
    std::uint64_t rank_counts_offset, std::uint64_t tile_counts_offset) {
    using namespace deep_ep::ascend;
    auto context = transport::make_device_transport_context();
    context.topology.world_rank = rank;
    context.topology.world_size = world;
    context.topology.scale_up_rank = rank;
    context.topology.scale_up_size = world;
    context.topology.scale_out_rank = 0;
    context.topology.scale_out_size = 1;
    context.topology.kind = transport::TransportTopologyKind::kFlatScaleUp;
    context.topology.epoch = 1;
    context.local_window_base = reinterpret_cast<std::uintptr_t>(workspace);

    elastic::CoreLaunchShape launch{};
    launch.num_blocks = blocks;
    launch.num_threads = threads;
    const auto function = reinterpret_cast<decltype(
        &elastic::deep_ep_ascend_launch_direct_dispatch_epilogue_parallel_prefix)>(
            entry);
    return function(
        workspace, prefix_per_rank, prefix_per_expert, unaligned_per_expert,
        context.abi_version, context.struct_size, context.capabilities,
        context.local_window_base, context.channel_table,
        context.peer_address_table, context.topology.abi_version,
        context.topology.struct_size, context.topology.world_rank,
        context.topology.world_size, context.topology.scale_up_rank,
        context.topology.scale_up_size, context.topology.scale_out_rank,
        context.topology.scale_out_size,
        static_cast<std::uint32_t>(context.topology.scale_up_direct),
        static_cast<std::uint32_t>(context.topology.kind),
        context.topology.epoch, context.backend_context,
        7U, experts, alignment, capacity, status_offset, rank_counts_offset,
        tile_counts_offset, tiles, stream, launch, threads);
}
