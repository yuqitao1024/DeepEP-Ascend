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
        self.assertIn('#include "transport/stub_transport.hpp"', source)
        for cuda_source in (
                "csrc/elastic/buffer.hpp",
                "csrc/kernels/backend/api.cuh",
                "deep_ep/include/deep_ep/common/handle.cuh"):
            self.assertNotIn("backends/ascend/transport", self.read(cuda_source))


if __name__ == "__main__":
    unittest.main()
