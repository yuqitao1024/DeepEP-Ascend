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

    def assert_transport_error(self, operation, call, detail):
        message = f"DeepEP Ascend backend: {operation} {detail}"
        with self.assertRaises(NotImplementedError) as context:
            call()
        exception = context.exception
        self.assertIs(type(exception), NotImplementedError)
        self.assertEqual(str(exception), message)

    def test_constructor_validation_and_idempotent_destroy(self):
        with self.assertRaisesRegex(RuntimeError, "world_size must be positive"):
            _C.ElasticBuffer(0, 0, *ARGS[2:])
        with self.assertRaisesRegex(RuntimeError, r"rank must be in \[0, world_size\)"):
            _C.ElasticBuffer(1, 1, *ARGS[2:])
        with self.assertRaisesRegex(RuntimeError,
                                    "communicator_handle must be zero in Phase 2A"):
            _C.ElasticBuffer(0, 1, 7, *ARGS[3:])
        with self.assertRaisesRegex(RuntimeError,
                                    "cpu_communicator must be empty in Phase 2A"):
            _C.ElasticBuffer(0, 1, 0, [(1, 2)], *ARGS[4:])
        with self.assertRaisesRegex(RuntimeError, "device_buffer_bytes must be positive"):
            _C.ElasticBuffer(0, 1, 0, [], 0, *ARGS[5:])
        with self.assertRaisesRegex(RuntimeError,
                                    "cpu_buffer_bytes must be zero in Phase 2A"):
            _C.ElasticBuffer(0, 1, 0, [], 4096, 4096, *ARGS[6:])
        self.buffer.destroy()
        self.buffer.destroy()

    def test_runtime_primitives_raise(self):
        unavailable = "is unavailable until the Ascend device transport is implemented"
        self.assert_transport_error(
            "barrier", lambda: self.buffer.barrier(True, False, True),
            "requires unavailable device transport capabilities: remote_signal, "
            "system_memory_ordering, device_barrier")
        self.assert_transport_error("get_comm_stream", self.buffer.get_comm_stream, unavailable)
        self.assert_transport_error("get_physical_domain_size",
                                    self.buffer.get_physical_domain_size, unavailable)
        self.assert_transport_error("get_logical_domain_size",
                                    self.buffer.get_logical_domain_size, unavailable)
        self.assert_transport_error("current_stream_wait",
                                    _C.EventHandle().current_stream_wait, unavailable)

    def test_hybrid_runtime_primitives_raise(self):
        hybrid_buffer = _C.ElasticBuffer(*ARGS[:6], True, *ARGS[7:])
        self.assert_transport_error(
            "barrier", lambda: hybrid_buffer.barrier(True, False, True),
            "requires unavailable device transport capabilities: remote_signal, "
            "system_memory_ordering, device_barrier, scale_up_team, scale_out_team")

        x = torch.empty((1, 16), dtype=torch.bfloat16)
        topk = torch.zeros((1, 1), dtype=torch.int64)
        none = None
        dispatch_args = (x, none, topk, none, none, none, none, none, none,
                         none, none, none, none, none, none,
                         1, 1, 1, 1, 0, none, none,
                         False, False, True, True, False, False, False)
        self.assert_transport_error(
            "dispatch", lambda: hybrid_buffer.dispatch(*dispatch_args),
            "requires unavailable device transport capabilities: symmetric_window, "
            "direct_peer_pointer, device_put, device_put_value, "
            "remote_atomic_add_release, remote_signal, system_memory_ordering, "
            "device_barrier, scale_up_team, scale_out_team")
        combine_args = (x, none, none, none, topk, topk, topk[:, 0].to(torch.int32),
                        none, none, 1, 1, 1, 0, none, none, False, False, False)
        self.assert_transport_error(
            "combine", lambda: hybrid_buffer.combine(*combine_args),
            "requires unavailable device transport capabilities: symmetric_window, "
            "direct_peer_pointer, device_put, remote_atomic_add_release, remote_signal, "
            "async_completion, system_memory_ordering, device_barrier, "
            "scale_up_team, scale_out_team")
        hybrid_buffer.destroy()

    def test_size_calculation_raises(self):
        self.assert_transport_error(
            "calculate_elastic_buffer_size",
            lambda: _C.calculate_elastic_buffer_size(0, 128, 7168, 8, False, True, True),
            "is unavailable until the Ascend device transport is implemented")

    def test_dispatch_and_combine_raise_before_device_validation(self):
        x = torch.empty((1, 16), dtype=torch.bfloat16)
        topk = torch.zeros((1, 1), dtype=torch.int64)
        none = None
        dispatch_args = (x, none, topk, none, none, none, none, none, none,
                         none, none, none, none, none, none,
                         1, 1, 1, 1, 0, none, none,
                         False, False, True, True, False, False, False)
        self.assert_transport_error(
            "dispatch", lambda: self.buffer.dispatch(*dispatch_args),
            "requires unavailable device transport capabilities: symmetric_window, "
            "direct_peer_pointer, device_put, device_put_value, "
            "remote_atomic_add_release, remote_signal, system_memory_ordering, "
            "device_barrier")
        combine_args = (x, none, none, none, topk, topk, topk[:, 0].to(torch.int32),
                        none, none, 1, 1, 1, 0, none, none, False, False, False)
        self.assert_transport_error(
            "combine", lambda: self.buffer.combine(*combine_args),
            "requires unavailable device transport capabilities: symmetric_window, "
            "direct_peer_pointer, device_put, remote_atomic_add_release, remote_signal, "
            "system_memory_ordering, device_barrier")


if __name__ == "__main__":
    unittest.main()
