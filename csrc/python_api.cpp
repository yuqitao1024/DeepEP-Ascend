#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <cstdint>
#include <string>

#include "platform/config.hpp"
#include "elastic/api.hpp"

#if defined(DEEP_EP_PLATFORM_CUDA)
#include <deep_ep/common/compiled.cuh>

#include "jit/api.hpp"
#include "legacy/buffer.hpp"
#endif

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif

#if EP_NUM_TOPK_IDX_BITS == 8
using module_topk_idx_t = int8_t;
#elif EP_NUM_TOPK_IDX_BITS == 16
using module_topk_idx_t = int16_t;
#elif EP_NUM_TOPK_IDX_BITS == 32
using module_topk_idx_t = int32_t;
#elif EP_NUM_TOPK_IDX_BITS == 64
using module_topk_idx_t = int64_t;
#else
#error "EP_NUM_TOPK_IDX_BITS must be one of 8, 16, 32, or 64"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    m.def("get_platform", []() { return std::string(deep_ep::platform::kName); });

#if defined(DEEP_EP_PLATFORM_CUDA)
    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });
#endif

    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<module_topk_idx_t>::value);

    deep_ep::elastic::register_apis(m);

#if defined(DEEP_EP_PLATFORM_CUDA)
    deep_ep::jit::register_apis(m);
    deep_ep::legacy::register_apis(m);
#endif
}
