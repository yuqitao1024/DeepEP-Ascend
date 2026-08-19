#include <pybind11/pybind11.h>

#include "csrc/backends/ascend/runtime/stream_event.hpp"

namespace runtime = deep_ep::ascend::runtime;

bool production_stream_event_callbacks_ready() {
    const auto api = runtime::make_stream_event_api();
    return api.current_device != nullptr && api.current_stream != nullptr &&
        api.pool_stream != nullptr && api.create_event != nullptr &&
        api.record_event != nullptr && api.query_event != nullptr &&
        api.wait_event != nullptr && api.synchronize_event != nullptr &&
        api.destroy_event != nullptr;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("production_stream_event_callbacks_ready",
               &production_stream_event_callbacks_ready);
}
