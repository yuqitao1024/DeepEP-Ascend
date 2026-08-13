#pragma once

#include <string_view>

#if defined(DEEP_EP_PLATFORM_CUDA) && defined(DEEP_EP_PLATFORM_ASCEND)
#error "Define exactly one DeepEP platform macro"
#elif !defined(DEEP_EP_PLATFORM_CUDA) && !defined(DEEP_EP_PLATFORM_ASCEND)
#error "Define exactly one DeepEP platform macro"
#endif

namespace deep_ep::platform {

#if defined(DEEP_EP_PLATFORM_CUDA)
inline constexpr std::string_view kName = "cuda";
#else
inline constexpr std::string_view kName = "ascend";
#endif

}  // namespace deep_ep::platform
