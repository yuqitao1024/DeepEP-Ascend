#include <cstdint>

#define __aicore__ __attribute__((noinline))
#define DEEP_EP_ASCEND_AICORE_URMA_SERVICE 1
#define __SIMT_DEVICE_FUNCTIONS_DECL__ __attribute__((noinline))
#define DEEP_EP_ASCEND_SIMT_DEVICE 1

#include "csrc/backends/ascend/elastic/combine_parallel.hpp"

namespace elastic = deep_ep::ascend::elastic;

static_assert(__builtin_has_attribute(
    elastic::aicore_combine_producer_payload_copy_plan, noinline));
static_assert(__builtin_has_attribute(
    elastic::simt_combine_producer_payload_copy_plan, noinline));

__aicore__ bool combine_parallel_aicore_callee_probe() {
    const auto common =
        elastic::aicore_combine_producer_payload_copy_plan(
            7168, 256, 16, false);
    const auto unaligned =
        elastic::aicore_combine_producer_payload_copy_plan(
            257, 256, 16, false);
    const auto expanded =
        elastic::aicore_combine_producer_payload_copy_plan(
            7168, 256, 16, true);
    const auto simt_common =
        elastic::simt_combine_producer_payload_copy_plan(
            7168, 256, 16, false);
    return common.valid && common.vector_elements == 7168 &&
        common.scalar_begin == 7168 && unaligned.valid &&
        unaligned.vector_elements == 0 && unaligned.scalar_begin == 0 &&
        expanded.valid && expanded.vector_elements == 0 &&
        expanded.scalar_begin == 0 && simt_common.valid &&
        simt_common.vector_elements == 7168 &&
        simt_common.scalar_begin == 7168;
}

int main() {
    return combine_parallel_aicore_callee_probe() ? 0 : 1;
}
