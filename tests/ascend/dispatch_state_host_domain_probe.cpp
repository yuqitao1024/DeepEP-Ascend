#include <cstdint>

#define DEEP_EP_ASCEND_SIMT_DEVICE 1
#define __SIMT_DEVICE_FUNCTIONS_DECL__ \
    __attribute__((error("device helper called from host domain")))
#include "csrc/backends/ascend/elastic/dispatch_state.hpp"

using namespace deep_ep::ascend::elastic;

int main() {
    CoreTopology topology{};
    topology.world_rank = 3;
    topology.world_size = 4;
    topology.scale_up_rank = 1;
    topology.scale_up_size = 2;
    topology.scale_out_rank = 1;
    topology.scale_out_size = 2;
    topology.epoch = 9;

    const auto descriptor = make_dispatch_handle_descriptor(
        7, topology, 11, 8, 16, 8, 2, 1, 8,
        mode_bit(CoreMode::kHybrid), DispatchRoutingMode::kHybrid,
        kHybridRouteLayoutVersion, 1, sizeof(HybridRouteRecord), 11,
        kHybridRouteCompleteStageFlags);
    const HybridRouteRecord record{
        0, 3, 2, 1, 7, 0, 0, 11, 9,
        kHybridRouteCompleteStageFlags, 0};
    const std::int32_t source_metadata[]{
        7, 1, -1, -1,
    };
    const std::int64_t received_topk[]{-1, 1};
    const HybridRouteBindingView bindings{
        source_metadata, received_topk, 1, 2, 8};

    return validate_hybrid_route_bindings(
               descriptor, {&record, 1}, bindings).ok() ? 0 : 1;
}
