#include "tests/ascend/simt_urma/runtime_probe.hpp"

namespace probe = deep_ep::ascend::transport::runtime_probe;

int main() {
    static_assert(probe::queue_wrap_batch_operations(0) == 0);
    static_assert(probe::queue_wrap_batch_operations(1) == 0);
    static_assert(probe::queue_wrap_batch_operations(2) == 0);
    static_assert(probe::queue_wrap_batch_operations(3) == 1);
    static_assert(probe::queue_wrap_batch_operations(6) == 4);
    static_assert(probe::queue_wrap_batch_operations(66) == 64);
    static_assert(probe::queue_wrap_batch_operations(128) == 64);
    static_assert(probe::runtime_case_records_transport_profile(
        probe::RuntimeCase::kPut));
    static_assert(probe::runtime_case_records_transport_profile(
        probe::RuntimeCase::kQueueWrap));
    static_assert(!probe::runtime_case_records_transport_profile(
        probe::RuntimeCase::kPhaseBoundary));
    return 0;
}
