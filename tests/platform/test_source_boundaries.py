import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class RegistrationBoundaryTest(unittest.TestCase):
    def read(self, path: str) -> str:
        return (ROOT / path).read_text()

    def test_python_module_has_one_elastic_entrypoint(self):
        source = self.read("csrc/python_api.cpp")
        self.assertIn('#include "elastic/api.hpp"', source)
        self.assertNotIn('#include "elastic/buffer.hpp"', source)

    def test_legacy_no_longer_registers_event_handle(self):
        source = self.read("csrc/legacy/buffer.hpp")
        self.assertNotIn('class_<EventHandle>(m, "EventHandle")', source)

    def test_common_bindings_are_not_owned_by_cuda_buffer(self):
        source = self.read("csrc/elastic/buffer.hpp")
        self.assertNotIn('class_<ElasticBuffer>(m, "ElasticBuffer")', source)
        self.assertIn("register_cuda_only_apis", source)

    def test_ascend_transport_is_owned_by_ascend_backend(self):
        source = self.read("csrc/backends/ascend/elastic_buffer.hpp")
        self.assertIn('#include "runtime/cann_runtime.hpp"', source)
        self.assertNotIn('stub_transport.hpp', source)
        self.assertIn("CannRuntimeResources", source)
        for cuda_source in (
                "csrc/elastic/buffer.hpp",
                "csrc/kernels/backend/api.cuh",
                "deep_ep/include/deep_ep/common/handle.cuh"):
            self.assertNotIn("backends/ascend/transport", self.read(cuda_source))

    def test_ascend_events_and_comm_stream_use_runtime_owned_identity(self):
        source = self.read("csrc/backends/ascend/elastic_buffer.hpp")
        self.assertIn('#include "elastic/async_state.hpp"', source)
        self.assertIn("runtime::make_stream_event_api()", source)
        self.assertIn("create_native_event", source)
        self.assertIn("pybind11::gil_scoped_release", source)
        self.assertIn("std::shared_ptr<elastic::AsyncBufferState>", source)
        self.assertIn("resources_->comm_stream()", source)
        self.assertIn("c10::Stream::unpack3", source)

    def test_ascend_production_target_does_not_import_cuda_sources(self):
        source = self.read("CMakeLists.txt")
        marker = 'if(DEEP_EP_PLATFORM STREQUAL "ascend")'
        self.assertIn(marker, source)
        ascend = source[source.index(marker):]
        for forbidden in (".cu", "nvshmem", "nccl", "CUDAToolkit"):
            self.assertNotIn(forbidden, ascend)
        self.assertIn("runtime/cann_runtime.cpp", ascend)
        self.assertIn("TORCH_NPU_ROOT", ascend)
        self.assertIn("torch_npu", ascend)


if __name__ == "__main__":
    unittest.main()
