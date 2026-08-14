import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSPORT = ROOT / "csrc/backends/ascend/transport"
SIMT_URMA = ROOT / "tests/ascend/simt_urma"


class AscendSimtUrmaTransportTest(unittest.TestCase):
    def test_cann_92_device_abi_matches_package_layouts(self):
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            self.skipTest("ASCEND_HOME_PATH is not configured")

        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "cann_abi_probe"
            include = pathlib.Path(ascend_home) / "include"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    "-I", str(include),
                    "-I", str(include / "hcomm"),
                    "-I", str(pathlib.Path(ascend_home) /
                              "x86_64-linux/pkg_inc"),
                    "-I", str(pathlib.Path(ascend_home) /
                              "x86_64-linux/asc/include"),
                    "-I", str(pathlib.Path(ascend_home) /
                              "x86_64-linux/asc/impl"),
                    str(ROOT / "tests/ascend/cann_abi_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_transport_command_abi_and_queue_model(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "transport_commands_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/transport_commands_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

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
