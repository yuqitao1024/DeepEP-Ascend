// Exercise the production epilogue expert-count launcher with synthetic records.
#include "csrc/backends/ascend/elastic/kernels.hpp"

namespace deep_ep::ascend::elastic {
extern "C" int deep_ep_ascend_launch_direct_dispatch_epilogue_count_experts(
    std::uint8_t*, std::uint8_t*, std::uintptr_t, int, int,
    std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t, std::uint64_t, std::uint64_t, void*, CoreLaunchShape,
    std::uint32_t);
}  // namespace deep_ep::ascend::elastic

extern "C" int probe_dispatch_count(
    void* entry, std::uint8_t* records, std::uint8_t* workspace,
    int rank, int world, std::uint64_t experts, std::uint64_t topk,
    std::uint64_t capacity, std::uint64_t stride,
    std::uint32_t blocks, std::uint32_t threads, void* stream) {
    using namespace deep_ep::ascend;

    elastic::CoreLaunchShape launch{};
    launch.num_blocks = blocks;
    launch.num_threads = threads;
    const auto function = reinterpret_cast<decltype(
        &elastic::deep_ep_ascend_launch_direct_dispatch_epilogue_count_experts)>(
            entry);

    // Use a fixed production-compatible scratch layout. The offsets must
    // match the runner's word indices exactly.
    constexpr std::uint64_t kRankCountsOffsetBytes = 256;
    constexpr std::uint64_t kOutputOffsetBytes = 8192;
    const std::uint64_t tile_count =
        (static_cast<std::uint64_t>(world) * capacity + 127) / 128;
    return function(
        records,
        workspace,
        reinterpret_cast<std::uintptr_t>(records),
        rank,
        world,
        experts,
        topk,
        capacity,
        0,
        capacity * stride,
        0,
        kRankCountsOffsetBytes,
        kOutputOffsetBytes,
        tile_count,
        stride,
        0,
        stream,
        launch,
        threads);
}
