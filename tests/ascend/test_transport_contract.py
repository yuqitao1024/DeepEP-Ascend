import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/ascend/transport_contract_probe.cpp"
TRANSPORT = ROOT / "csrc/backends/ascend/transport"


class AscendTransportContractTest(unittest.TestCase):
    def test_pure_cpp_contract_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "transport_contract_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", f"-I{ROOT}", str(PROBE),
                 "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_public_transport_headers_have_no_vendor_dependencies(self):
        forbidden = ("cuda", "nccl", "nvshmem", "acl/", "hccl", "cann",
                     "torch_npu", "kernel_operator")
        headers = [TRANSPORT / name for name in (
            "types.hpp", "host_transport.hpp", "device_transport.hpp",
            "device_transport_facade.hpp", "device_transport_stub.hpp",
            "stub_transport.hpp")]
        for header in headers:
            self.assertTrue(header.is_file(), str(header))
            includes = [line.strip().lower() for line in header.read_text().splitlines()
                        if line.lstrip().startswith("#include")]
            for name in forbidden:
                self.assertFalse(any(name in include for include in includes),
                                 f"{header}: {name}")


if __name__ == "__main__":
    unittest.main()
