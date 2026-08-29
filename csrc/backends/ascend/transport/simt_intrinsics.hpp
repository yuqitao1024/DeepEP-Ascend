#pragma once

#include <type_traits>

#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"

#include "device_transport.hpp"

namespace deep_ep::ascend::transport::simt {

inline constexpr std::uint32_t kPollingNopCycles = 55;

template <typename T>
inline constexpr bool kSupportedDeviceScalar =
    std::is_integral_v<T> && (sizeof(T) == 1 || sizeof(T) == 2 ||
                              sizeof(T) == 4 || sizeof(T) == 8);

DEEP_EP_ASCEND_SIMT_CALLEE void system_fence() {
    asc_threadfence();
}

DEEP_EP_ASCEND_SIMT_CALLEE void poll_nop() {
    for (std::uint32_t cycle = 0; cycle < kPollingNopCycles; ++cycle)
        asc_nop();
}

template <typename T>
DEEP_EP_ASCEND_SIMT_CALLEE T load_observed(const __gm__ T* address) {
    static_assert(kSupportedDeviceScalar<T>);
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, NON_CACHEABLE>(
        const_cast<__gm__ T*>(address));
}

template <typename T>
DEEP_EP_ASCEND_SIMT_CALLEE void store_published(
    __gm__ T* address, T value) {
    static_assert(kSupportedDeviceScalar<T>);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, NON_CACHEABLE>(address,
                                                                  value);
}

}  // namespace deep_ep::ascend::transport::simt
