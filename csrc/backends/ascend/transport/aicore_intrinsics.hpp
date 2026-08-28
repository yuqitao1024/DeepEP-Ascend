#pragma once

#include <type_traits>

#include "kernel_operator.h"

#include "stage_profile.hpp"

namespace deep_ep::ascend::transport::aicore {

template <typename T>
inline constexpr bool kSupportedDeviceScalar =
    std::is_integral_v<T> && (sizeof(T) == 1 || sizeof(T) == 2 ||
                              sizeof(T) == 4 || sizeof(T) == 8);

__aicore__ inline void system_fence() {
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void flush_cacheline(__gm__ void* address) {
    pipe_barrier(PIPE_ALL);
    dcci(address, cache_line_t::SINGLE_CACHE_LINE,
         dcci_dst_t::CACHELINE_OUT);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void flush_stage_profile_header(
    __gm__ TransportStageProfile* profile) {
    auto* bytes = reinterpret_cast<__gm__ std::uint8_t*>(profile);
    for (std::size_t line = 0;
         line < kTransportStageProfileHeaderCacheLineCount; ++line) {
        flush_cacheline(bytes +
                        line * kTransportStageProfileCacheLineBytes);
    }
}

template <AscendC::HardEvent Event>
__aicore__ inline void sync_event() {
    auto* pipe = GetTPipePtr();
    AscendC::TEventID event_id = 0;
    if (pipe != nullptr)
        event_id = pipe->FetchEventID(Event);
    AscendC::SetFlag<Event>(event_id);
    AscendC::WaitFlag<Event>(event_id);
}

template <typename T>
__aicore__ inline T load_device(const __gm__ T* address) {
    static_assert(kSupportedDeviceScalar<T>);
    return static_cast<T>(
        ld_dev(const_cast<__gm__ T*>(address), 0));
}

// Control words published by a SIMT VF must bypass the AICore DCache. A
// cached ld_dev can retain the pre-publication value after another execution
// context has written the same GM line.
template <typename T>
__aicore__ inline T load_published(const __gm__ T* address) {
    static_assert(kSupportedDeviceScalar<T>);
    return AscendC::ReadGmByPassDCache<T>(
        const_cast<__gm__ T*>(address));
}

template <typename T>
__aicore__ inline void store_device(__gm__ T* address, T value) {
    static_assert(kSupportedDeviceScalar<T>);
    st_dev(value, address, 0);
}

}  // namespace deep_ep::ascend::transport::aicore
