#include "csrc/backends/ascend/elastic_buffer.hpp"

extern "C" int deep_ep_ascend_launch_barrier(
    deep_ep::ascend::elastic::BarrierArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_dispatch(
    deep_ep::ascend::elastic::DispatchArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine(
    deep_ep::ascend::elastic::CombineArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }
extern "C" int deep_ep_ascend_launch_combine_epilogue(
    deep_ep::ascend::elastic::CombineArguments,
    deep_ep::ascend::elastic::CoreTiling, void*) { return 0; }

int main() {
    // The test-only factory supplies tensor/runtime/transport doubles and
    // exercises tuple narrowing, cache reuse, empty input, and diagnostics.
    return deep_ep::ascend::testing::run_public_dispatch_probe();
}
