import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/ascend/device_transport_stub_probe.cpp"
SIMT_PROBE = ROOT / "tests/ascend/device_transport_simt_probe.asc"
SIMT_CMAKE = ROOT / "tests/ascend/simt/CMakeLists.txt"
TRANSPORT = ROOT / "csrc/backends/ascend/transport"


class AscendDeviceTransportStubTest(unittest.TestCase):
    def test_facade_preserves_local_addressing_and_stub_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "device_transport_stub_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PROBE), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_facade_and_stub_headers_have_no_vendor_dependencies(self):
        forbidden = ("cuda", "nccl", "nvshmem", "ain", "hcomm", "hccl",
                     "urma", "torch_npu", "kernel_operator")
        for name in ("device_transport_facade.hpp", "device_transport_stub.hpp"):
            header = TRANSPORT / name
            self.assertTrue(header.is_file(), str(header))
            includes = [line.strip().lower()
                        for line in header.read_text().splitlines()
                        if line.lstrip().startswith("#include")]
            for token in forbidden:
                self.assertFalse(any(token in include for include in includes),
                                 f"{header}: {token}")

    def test_simt_probe_covers_the_transport_surface(self):
        self.assertTrue(SIMT_PROBE.is_file(), str(SIMT_PROBE))
        self.assertTrue(SIMT_CMAKE.is_file(), str(SIMT_CMAKE))

        source = SIMT_PROBE.read_text()
        for operation in ("is_peer_directly_accessible", "get_symmetric_offset",
                          "get_symmetric_pointer", "put(", "get(", "put_value",
                          "remote_add_release", "signal(", "read_signal",
                          "wait_signal", "flush(", "flush_async", "wait(",
                          "load_acquire", "store_release", "system_fence",
                          "device_barrier"):
            self.assertIn(operation, source)

        cmake = SIMT_CMAKE.read_text()
        self.assertIn("dav-3510", cmake)
        self.assertIn("device_transport_simt_probe.asc", cmake)

        forbidden = ("ain", "hcomm", "hccl", "cuda", "nccl", "nvshmem")
        includes = [line.strip().lower() for line in source.splitlines()
                    if line.lstrip().startswith("#include")]
        for token in forbidden:
            self.assertFalse(any(token in include for include in includes), token)


if __name__ == "__main__":
    unittest.main()
