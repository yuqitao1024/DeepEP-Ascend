// Host adapter for exercising the production receive-validation launcher.
// The function address comes from the loaded Ascend extension, so the test
// runs the same device kernel as Dispatch without needing faulted peer traffic.
#include "csrc/backends/ascend/elastic/kernels.hpp"

namespace deep_ep::ascend::elastic {
#include "csrc/backends/ascend/elastic/dispatch_vf_launchers.hpp"
}

extern "C" int probe_dispatch_validate(
    void* entry, std::uint8_t* records, std::uint8_t* workspace,
    const std::int32_t* metadata, int rank, int world,
    std::uint64_t capacity, std::uint64_t topk, std::uint64_t stride,
    std::uint32_t mode, std::uint32_t blocks, std::uint32_t threads,
    void* stream) {
    using namespace deep_ep::ascend;
    auto context = transport::make_device_transport_context();
    context.topology.world_rank = rank;
    context.topology.world_size = world;
    context.local_window_base = reinterpret_cast<std::uintptr_t>(records);
    elastic::CoreLaunchShape launch{};
    launch.num_blocks = blocks;
    const auto function = reinterpret_cast<decltype(
        &elastic::deep_ep_ascend_launch_direct_dispatch_epilogue_validate_records)>(entry);
    return function(records, workspace, metadata, context, mode,
                    7, world * 4, topk, capacity,
                    0, capacity * stride, 0, 32, 128, 256, 8192, 8200,
                    (world * capacity + 127) / 128, stride, 0, topk * 8,
                    stream, launch, threads);
}
