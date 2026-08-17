import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "platform"))
from extension_loader import load_extension


_C = load_extension()


ALIGNMENT = 2 * 1024 * 1024
ARGS = (0, 2, 1, [], ALIGNMENT, 0, False, True, True, 3, 0, 300, 100, True)


class AscendStubTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(_C.get_platform(), "ascend")

    def assert_transport_error(self, operation, call, detail):
        message = f"DeepEP Ascend backend: {operation} {detail}"
        with self.assertRaises(NotImplementedError) as context:
            call()
        exception = context.exception
        self.assertIs(type(exception), NotImplementedError)
        self.assertEqual(str(exception), message)

    def test_constructor_rejects_invalid_production_preconditions(self):
        with self.assertRaisesRegex(RuntimeError, "world_size must be at least two"):
            _C.ElasticBuffer(0, 1, *ARGS[2:])
        with self.assertRaisesRegex(RuntimeError, r"rank must be in \[0, world_size\)"):
            _C.ElasticBuffer(2, 2, *ARGS[2:])
        with self.assertRaisesRegex(RuntimeError,
                                    "communicator_handle must be nonzero"):
            _C.ElasticBuffer(0, 2, 0, *ARGS[3:])
        with self.assertRaisesRegex(RuntimeError,
                                    "cpu communicator must be empty"):
            _C.ElasticBuffer(0, 2, 1, [(1, 2)], *ARGS[4:])
        with self.assertRaisesRegex(
                RuntimeError, "device buffer must be positive and 2 MiB-aligned"):
            _C.ElasticBuffer(0, 2, 1, [], 0, *ARGS[5:])
        with self.assertRaisesRegex(RuntimeError,
                                    "cpu_buffer_bytes must be zero"):
            _C.ElasticBuffer(0, 2, 1, [], ALIGNMENT, 1, *ARGS[6:])
        with self.assertRaisesRegex(RuntimeError, "hybrid mode is unsupported"):
            _C.ElasticBuffer(*ARGS[:6], True, *ARGS[7:])
        with self.assertRaisesRegex(RuntimeError, "CUDA QP count must be zero"):
            _C.ElasticBuffer(*ARGS[:10], 1, *ARGS[11:])

    def test_event_wait_remains_unavailable(self):
        unavailable = "is unavailable until the Ascend device transport is implemented"
        self.assert_transport_error("current_stream_wait",
                                    _C.EventHandle().current_stream_wait, unavailable)

    def test_size_calculation_enforces_production_preconditions(self):
        with self.assertRaisesRegex(RuntimeError,
                                    "communicator_handle must be nonzero"):
            _C.calculate_elastic_buffer_size(
                0, 128, 7168, 8, False, False, True)
        self.assert_transport_error(
            "calculate_elastic_buffer_size",
            lambda: _C.calculate_elastic_buffer_size(
                7, 128, 7168, 8, True, False, True),
            "does not support FP8")
        self.assert_transport_error(
            "calculate_elastic_buffer_size",
            lambda: _C.calculate_elastic_buffer_size(
                7, 128, 7168, 8, False, True, True),
            "does not support hybrid mode")


if __name__ == "__main__":
    unittest.main()
