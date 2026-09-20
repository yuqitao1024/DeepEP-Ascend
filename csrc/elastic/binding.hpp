#pragma once

#include <cstdint>
#include <type_traits>

#include <pybind11/pybind11.h>

namespace deep_ep::elastic::binding {

template <typename EventHandle>
void register_event(pybind11::module_& m) {
    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("__copy__", [](const EventHandle& event) {
            return event;
        })
        .def("current_stream_wait", [](const EventHandle& event) {
            event.current_stream_wait();
        });
}

template <typename ElasticBuffer, bool ReleaseGilOnDestroy = false>
pybind11::class_<ElasticBuffer> register_common_apis(pybind11::module_& m) {
    pybind11::class_<ElasticBuffer> cls(m, "ElasticBuffer");
    if constexpr (std::is_constructible_v<
                      ElasticBuffer, const int&, const int&, const int64_t&,
                      const typename ElasticBuffer::cpu_comm_t&,
                      const int64_t&, const int64_t&, const bool&,
                      const bool&, const bool&, const int&, const int&,
                      const int&, const int&, const bool&>) {
        cls.def(pybind11::init<int, int, int64_t,
                              typename ElasticBuffer::cpu_comm_t,
                              int64_t, int64_t,
                              bool, bool, bool, int, int, int, int, bool>());
    }
    if constexpr (std::is_constructible_v<
                      ElasticBuffer, const int&, const int&,
                      const pybind11::bytes&,
                      const typename ElasticBuffer::cpu_comm_t&,
                      const int64_t&, const int64_t&, const int64_t&, const bool&,
                      const bool&, const bool&, const int&, const int&,
                      const int&, const int&, const bool&>) {
        cls.def(pybind11::init<int, int, pybind11::bytes,
                              typename ElasticBuffer::cpu_comm_t,
                              int64_t, int64_t, int64_t,
                              bool, bool, bool, int, int, int, int, bool>());
    }
    if constexpr (ReleaseGilOnDestroy) {
        cls.def("destroy", [](ElasticBuffer& buffer) {
            pybind11::gil_scoped_release release;
            buffer.destroy();
        });
    } else {
        cls.def("destroy", &ElasticBuffer::destroy);
    }
    cls.def("get_comm_stream", &ElasticBuffer::get_comm_stream)
       .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
       .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
       .def("barrier", &ElasticBuffer::barrier)
       .def("dispatch", &ElasticBuffer::dispatch)
       .def("combine", &ElasticBuffer::combine);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
    return cls;
}

}  // namespace deep_ep::elastic::binding
