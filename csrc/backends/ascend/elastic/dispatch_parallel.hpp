#pragma once

#include <cstdint>
#include <limits>

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)
#define DEEP_EP_ASCEND_DISPATCH_PARALLEL_CALLEE \
    __SIMT_DEVICE_FUNCTIONS_DECL__
#else
#define DEEP_EP_ASCEND_DISPATCH_PARALLEL_CALLEE
#endif

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kDispatchBitmapWordBits = 64;

DEEP_EP_ASCEND_DISPATCH_PARALLEL_CALLEE constexpr bool
dispatch_bitmap_words(
    std::uint64_t bits, std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    *words = bits / kDispatchBitmapWordBits +
        (bits % kDispatchBitmapWordBits != 0 ? 1 : 0);
    return true;
}

DEEP_EP_ASCEND_DISPATCH_PARALLEL_CALLEE constexpr bool
dispatch_owner_bitmap_words(
    std::uint64_t owners, std::uint64_t bits_per_owner,
    std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    std::uint64_t words_per_owner = 0;
    if (!dispatch_bitmap_words(bits_per_owner, &words_per_owner) ||
        (words_per_owner != 0 &&
         owners > std::numeric_limits<std::uint64_t>::max() /
             words_per_owner))
        return false;
    *words = owners * words_per_owner;
    return true;
}

}  // namespace deep_ep::ascend::elastic

#undef DEEP_EP_ASCEND_DISPATCH_PARALLEL_CALLEE
