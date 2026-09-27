// Exercise the production epilogue metadata launcher with synthetic records.
#include "csrc/backends/ascend/elastic/kernels.hpp"

namespace deep_ep::ascend::elastic {
#include "csrc/backends/ascend/elastic/dispatch_vf_launchers.hpp"
}

extern "C" int probe_dispatch_metadata(
    void* entry, std::uint8_t* records, std::uint8_t* workspace,
    std::int64_t* recv_topk_indices, std::int32_t* source_metadata,
    int rank, int world, std::uint32_t mode, std::uint64_t experts,
    std::uint64_t topk, std::uint64_t capacity, std::uint64_t stride,
    std::uint64_t topk_offset, std::uint64_t metadata_offset,
    std::uint32_t blocks, std::uint32_t threads, void* stream,
    std::uint64_t status_offset, std::uint64_t rank_counts_offset,
    std::uint64_t rank_values_offset,
    std::uint8_t* recv_scale_factors, float* recv_topk_weights,
    std::uint64_t sf_offset, std::uint64_t sf_bytes,
    std::uint64_t sf_token_stride, std::uint64_t sf_pack_stride,
    std::uint64_t weight_offset) {
    using namespace deep_ep::ascend;
    elastic::CoreLaunchShape launch{};
    launch.num_blocks = blocks;
    launch.num_threads = threads;
    const auto function = reinterpret_cast<decltype(
        &elastic::deep_ep_ascend_launch_direct_dispatch_epilogue_metadata)>(
            entry);
    return function(
        records, workspace, recv_topk_indices, source_metadata,
        reinterpret_cast<std::uintptr_t>(records), rank, world, mode,
        experts, topk, capacity, 0, capacity * stride, status_offset,
        rank_counts_offset, rank_values_offset, stride, topk_offset,
        metadata_offset, recv_scale_factors, recv_topk_weights,
        sf_offset, sf_bytes, sf_token_stride, sf_pack_stride, weight_offset,
        stream, launch, threads);
}
