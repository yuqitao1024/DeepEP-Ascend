import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSPORT = ROOT / "csrc/backends/ascend/transport"
SIMT_URMA = ROOT / "tests/ascend/simt_urma"


class AscendSimtUrmaTransportTest(unittest.TestCase):
    def test_mixed_phase_primitive_probe_builds_with_cann(self):
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            self.skipTest("ASCEND_HOME_PATH is not configured")

        with tempfile.TemporaryDirectory() as directory:
            configure = subprocess.run(
                ["cmake", "-S", str(SIMT_URMA), "-B", directory],
                capture_output=True, text=True, check=False)
            self.assertEqual(configure.returncode, 0, configure.stderr)

            build = subprocess.run(
                ["cmake", "--build", directory, "--verbose"],
                capture_output=True, text=True, check=False)
            self.assertEqual(build.returncode, 0, build.stderr)

    def test_production_transport_does_not_include_cann_internal_headers(self):
        for header in TRANSPORT.glob("*.hpp"):
            includes = [
                line.strip().lower()
                for line in header.read_text().splitlines()
                if line.lstrip().startswith("#include")
            ]
            self.assertFalse(
                any("asc/impl" in include for include in includes), str(header))


if __name__ == "__main__":
    unittest.main()
