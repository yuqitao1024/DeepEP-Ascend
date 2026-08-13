#pragma once

#include <cstdint>

#include <pybind11/pybind11.h>

namespace deep_ep::elastic::binding {

template <typename EventHandle>
void register_event(pybind11::module_& m) {
    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);
}

template <typename ElasticBuffer>
pybind11::class_<ElasticBuffer> register_common_apis(pybind11::module_& m) {
    pybind11::class_<ElasticBuffer> cls(m, "ElasticBuffer");
    cls.def(pybind11::init<int, int, int64_t, int64_t,
                          bool, bool, bool, bool, int, int, int, int, bool>())
       .def("destroy", &ElasticBuffer::destroy)
       .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
       .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
       .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
       .def("barrier", &ElasticBuffer::barrier)
       .def("dispatch", &ElasticBuffer::dispatch)
       .def("combine", &ElasticBuffer::combine);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
    return cls;
}

}  // namespace deep_ep::elastic::binding
