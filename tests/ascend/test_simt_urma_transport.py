import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSPORT = ROOT / "csrc/backends/ascend/transport"
SIMT_URMA = ROOT / "tests/ascend/simt_urma"


class AscendSimtUrmaTransportTest(unittest.TestCase):
    def test_two_rank_runtime_harness_contract(self):
        describe = subprocess.run(
            [sys.executable, str(SIMT_URMA / "run_two_rank_probe.py"),
             "--describe"],
            capture_output=True, text=True, check=False)
        self.assertEqual(describe.returncode, 0, describe.stderr)
        contract = json.loads(describe.stdout)

        expected_cases = {
            "put", "put-value64", "faa64", "signal", "flush",
            "payload-signal-order", "barrier-repeat", "queue-wrap",
            "phase-boundary", "teardown",
        }
        self.assertEqual(set(contract["cases"]), expected_cases)
        self.assertEqual(
            contract["communicator"],
            "backend.get_hccl_comm(local_rank)")
        self.assertEqual(contract["runner_loading"],
                         "in-process-shared-library")
        self.assertGreater(contract["timeout_ms"], 0)
        self.assertGreaterEqual(
            contract["cases"]["payload-signal-order"]["iterations"], 1000)
        self.assertTrue(contract["cases"]["queue-wrap"]["requires_sq_wrap"])
        for name in expected_cases - {"teardown"}:
            self.assertEqual(
                contract["cases"][name]["phases"],
                ["producer", "service", "consumer"])

    def test_cann_host_transport_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "cann_transport_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/cann_transport_probe.cpp"),
                    str(TRANSPORT / "cann_transport.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_aicore_service_ordering_and_timeout_model(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "transport_service_model"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/transport_service_model_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_urma_work_request_words_and_queue_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "urma_wqe_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/urma_wqe_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

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

    def test_runtime_probe_builds_with_cann(self):
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            self.skipTest("ASCEND_HOME_PATH is not configured")

        with tempfile.TemporaryDirectory() as directory:
            configure = subprocess.run(
                ["cmake", "-S", str(SIMT_URMA), "-B", directory,
                 "-DDEEP_EP_ASCEND_STAGED_URMA=ON",
                 "-DDEEP_EP_BUILD_URMA_RUNTIME=ON"],
                capture_output=True, text=True, check=False)
            self.assertEqual(configure.returncode, 0, configure.stderr)

            build = subprocess.run(
                ["cmake", "--build", directory, "--target",
                 "deep_ep_ascend_urma_runner", "--verbose"],
                capture_output=True, text=True, check=False)
            self.assertEqual(build.returncode, 0, build.stderr)

            smoke = subprocess.run(
                [sys.executable, str(SIMT_URMA / "run_two_rank_probe.py"),
                 "--runner", str(pathlib.Path(directory) /
                                   "deep_ep_ascend_urma_runner"),
                 "--local-smoke"],
                capture_output=True, text=True, check=False)
            self.assertEqual(smoke.returncode, 0, smoke.stderr)

    def test_production_transport_does_not_include_cann_internal_headers(self):
        for source in list(TRANSPORT.glob("*.hpp")) + list(
                TRANSPORT.glob("*.cpp")):
            includes = [
                line.strip().lower()
                for line in source.read_text().splitlines()
                if line.lstrip().startswith("#include")
            ]
            self.assertFalse(
                any("asc/impl" in include for include in includes), str(source))


if __name__ == "__main__":
    unittest.main()
