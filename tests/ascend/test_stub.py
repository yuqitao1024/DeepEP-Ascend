import pathlib
import sys
import unittest

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "platform"))
from extension_loader import load_extension


_C = load_extension()


ARGS = (0, 1, 0, [], 4096, 0, False, True, True, 3, 0, 300, 100, True)


class AscendStubTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(_C.get_platform(), "ascend")
        self.buffer = _C.ElasticBuffer(*ARGS)

    def assert_phase_error(self, operation, call):
        message = f"DeepEP Ascend backend: {operation} is not implemented in phase 1"
        with self.assertRaises(NotImplementedError) as context:
            call()
        exception = context.exception
        self.assertIs(type(exception), NotImplementedError)
        self.assertEqual(str(exception), message)

    def test_constructor_validation_and_idempotent_destroy(self):
        with self.assertRaisesRegex(RuntimeError, "num_ranks must be positive"):
            _C.ElasticBuffer(0, 0, *ARGS[2:])
        with self.assertRaisesRegex(RuntimeError, "rank_idx must be in"):
            _C.ElasticBuffer(1, 1, *ARGS[2:])
        with self.assertRaisesRegex(RuntimeError, "comm_handle must be zero"):
            _C.ElasticBuffer(0, 1, 7, *ARGS[3:])
        with self.assertRaisesRegex(RuntimeError, "cpu_comm must be empty"):
            _C.ElasticBuffer(0, 1, 0, [(1, 2)], *ARGS[4:])
        with self.assertRaisesRegex(RuntimeError, "num_buffer_bytes must be positive"):
            _C.ElasticBuffer(0, 1, 0, [], 0, *ARGS[5:])
        with self.assertRaisesRegex(RuntimeError, "num_cpu_buffer_bytes must be zero"):
            _C.ElasticBuffer(0, 1, 0, [], 4096, 4096, *ARGS[6:])
        self.buffer.destroy()
        self.buffer.destroy()

    def test_runtime_primitives_raise(self):
        self.assert_phase_error("barrier", lambda: self.buffer.barrier(True, False, True))
        self.assert_phase_error("get_comm_stream", self.buffer.get_comm_stream)
        self.assert_phase_error("get_physical_domain_size", self.buffer.get_physical_domain_size)
        self.assert_phase_error("get_logical_domain_size", self.buffer.get_logical_domain_size)
        self.assert_phase_error("current_stream_wait", _C.EventHandle().current_stream_wait)

    def test_size_calculation_raises(self):
        self.assert_phase_error(
            "calculate_elastic_buffer_size",
            lambda: _C.calculate_elastic_buffer_size(0, 128, 7168, 8, False, True, True))

    def test_dispatch_and_combine_raise_before_device_validation(self):
        x = torch.empty((1, 16), dtype=torch.bfloat16)
        topk = torch.zeros((1, 1), dtype=torch.int64)
        none = None
        dispatch_args = (x, none, topk, none, none, none, none, none, none,
                         none, none, none, none, none, none,
                         1, 1, 1, 1, 0, none, none,
                         False, False, True, True, False, False, False)
        self.assert_phase_error("dispatch", lambda: self.buffer.dispatch(*dispatch_args))
        combine_args = (x, none, none, none, topk, topk, topk[:, 0].to(torch.int32),
                        none, none, 1, 1, 1, 0, none, none, False, False, False)
        self.assert_phase_error("combine", lambda: self.buffer.combine(*combine_args))


if __name__ == "__main__":
    unittest.main()
