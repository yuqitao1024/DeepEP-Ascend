#pragma once

#include "../platform/config.hpp"
#include "binding.hpp"

#if defined(DEEP_EP_PLATFORM_CUDA)
#include "../utils/event.hpp"
#include "buffer.hpp"
#else
#include "../backends/ascend/elastic_buffer.hpp"
#endif

namespace deep_ep::elastic {

inline void register_apis(pybind11::module_& m) {
#if defined(DEEP_EP_PLATFORM_CUDA)
    binding::register_event<deep_ep::EventHandle>(m);
    auto cls = binding::register_common_apis<ElasticBuffer>(m);
    register_cuda_only_apis(m, cls);
#else
    binding::register_event<deep_ep::ascend::EventHandle>(m);
    binding::register_common_apis<deep_ep::ascend::ElasticBuffer>(m);
#endif
}

}  // namespace deep_ep::elastic
