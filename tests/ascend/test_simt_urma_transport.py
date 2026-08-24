import contextlib
import importlib.util
import io
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRANSPORT = ROOT / "csrc/backends/ascend/transport"
SIMT_URMA = ROOT / "tests/ascend/simt_urma"


class AscendSimtUrmaTransportTest(unittest.TestCase):
    def test_failed_primitive_tears_down_before_process_group_destroy(self):
        module_path = SIMT_URMA / "run_two_rank_probe.py"
        spec = importlib.util.spec_from_file_location(
            "run_two_rank_probe", module_path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        events = []

        class FakeDist:
            class distributed_c10d:
                @staticmethod
                def _get_default_group():
                    return types.SimpleNamespace(
                        get_hccl_comm=lambda local_rank: 0x1234)

            @staticmethod
            def init_process_group(**_):
                events.append("init")

            @staticmethod
            def barrier():
                events.append("barrier")

            @staticmethod
            def all_gather_object(results, local_result):
                events.append("gather")
                results[:] = [local_result, local_result]

            @staticmethod
            def destroy_process_group():
                events.append("destroy")

        torch = types.ModuleType("torch")
        torch.npu = types.SimpleNamespace(
            set_device=lambda local_rank: events.append(
                f"device:{local_rank}"))
        torch.distributed = FakeDist

        def run_case(_, __, ___, case_name, ____, error, _____):
            name = case_name.decode()
            events.append(f"runner:{name}")
            if name == "put":
                error.value = b"put failed"
                return 1
            return 0

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.dict(
                sys.modules,
                {"torch": torch, "torch.distributed": FakeDist,
                 "torch_npu": types.ModuleType("torch_npu")}), \
             mock.patch.dict(
                 os.environ,
                 {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "2"},
                 clear=False), \
             mock.patch.object(probe, "load_runner", return_value=run_case), \
             contextlib.redirect_stdout(stdout), \
             contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, "case put failed") as error:
                probe.run_runtime(
                    types.SimpleNamespace(runner="unused"),
                    ["put", "teardown"])

        self.assertIn("case put failed", str(error.exception))
        self.assertNotIn("case teardown failed", str(error.exception))
        self.assertLess(events.index("runner:teardown"), events.index("destroy"))
        self.assertEqual(stderr.getvalue(), "")

    def test_teardown_failure_is_reported_without_earlier_error(self):
        module_path = SIMT_URMA / "run_two_rank_probe.py"
        spec = importlib.util.spec_from_file_location(
            "run_two_rank_probe", module_path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        class FakeDist:
            @staticmethod
            def barrier():
                pass

            @staticmethod
            def all_gather_object(results, local_result):
                results[:] = [local_result, local_result]

        def run_case(_, __, ___, case_name, ____, error, _____):
            if case_name == b"teardown":
                error.value = b"teardown failed"
                return 1
            return 0

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, "case teardown failed"):
                probe.run_case_sequence(
                    FakeDist, run_case, 0x1234, 0, 2,
                    ["put", "teardown"], probe.runtime_contract()["cases"])
        self.assertIn("teardown after failed runtime case also failed", stderr.getvalue())

    def test_successful_final_teardown_is_not_repeated(self):
        module_path = SIMT_URMA / "run_two_rank_probe.py"
        spec = importlib.util.spec_from_file_location(
            "run_two_rank_probe", module_path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        calls = []

        class FakeDist:
            @staticmethod
            def barrier():
                pass

            @staticmethod
            def all_gather_object(results, local_result):
                results[:] = [local_result, local_result]

        def run_case(_, __, ___, case_name, ____, _____, ______):
            calls.append(case_name.decode())
            return 0

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(
                probe.run_case_sequence(
                    FakeDist, run_case, 0x1234, 0, 2,
                    ["put", "teardown"], probe.runtime_contract()["cases"]),
                0)
        self.assertEqual(calls, ["put", "teardown"])
        self.assertEqual(stderr.getvalue(), "")

    def test_runtime_case_list_requires_final_teardown(self):
        module_path = SIMT_URMA / "run_two_rank_probe.py"
        spec = importlib.util.spec_from_file_location(
            "run_two_rank_probe", module_path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        valid_cases = "put,put-value64,faa64,teardown"
        self.assertEqual(
            probe.selected_cases(valid_cases), valid_cases.split(","))
        with self.assertRaisesRegex(ValueError, "teardown case is required"):
            probe.selected_cases("put,faa64")
        with self.assertRaisesRegex(
                ValueError, "teardown must be the final selected case"):
            probe.selected_cases("put,teardown,faa64")

    def test_two_rank_runtime_harness_contract(self):
        describe = subprocess.run(
            [sys.executable, str(SIMT_URMA / "run_two_rank_probe.py"),
             "--describe"],
            capture_output=True, text=True, check=False)
        self.assertEqual(describe.returncode, 0, describe.stderr)
        contract = json.loads(describe.stdout)

        expected_cases = {
            "put", "put-value64", "faa64", "signal", "signal-set", "flush",
            "async-lifecycle", "payload-signal-order", "barrier-repeat", "queue-wrap",
            "profile-mixed", "phase-boundary", "teardown",
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
        self.assertEqual(
            contract["cases"]["profile-mixed"]["command_metrics"], {
                "command_count": 6,
                "put_command_count": 1,
                "payload_bytes": 24,
            })
        self.assertEqual(
            contract["cases"]["profile-mixed"]["queue_invariants"], {
                "forced_capacity_drain": True,
                "final_sq_depth": 0,
                "final_cq_depth": 0,
                "equal_nonzero_high_watermarks": True,
                "positive_wait_cycles": True,
            })
        for name in expected_cases - {"teardown"}:
            self.assertEqual(
                contract["cases"][name]["phases"],
                ["producer", "service", "consumer"])

    def test_local_phase_boundary_launch_disables_profile_pressure(self):
        runtime = (SIMT_URMA / "runtime_probe_main.cpp").read_text()
        self.assertIn(
            "probe::RuntimeCase::kPhaseBoundary, 0, 1,\n"
            "            1, false, stream);",
            runtime)

    def test_profile_final_launch_stream_orders_rank_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "runtime_launch_sequence"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/"
                        "runtime_probe_launch_sequence_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_profile_final_launch_preserves_six_command_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "runtime_command_sequence"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/"
                        "runtime_probe_command_sequence_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

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

    def test_production_runtime_resource_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "production_lifecycle_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/production_lifecycle_probe.cpp"),
                    str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                    str(ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"),
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

    def test_queue_wrap_batch_respects_command_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "queue_wrap_batch_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/queue_wrap_batch_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_urma_wqe_publication_uses_mte3(self):
        service = (
            TRANSPORT / "aicore_transport_service.hpp").read_text()
        intrinsics = (TRANSPORT / "aicore_intrinsics.hpp").read_text()
        self.assertIn("sync_event<AscendC::HardEvent::S_MTE3>", service)
        self.assertIn("DataCopy(sq_global", service)
        self.assertIn("sync_event<AscendC::HardEvent::MTE3_S>", service)
        self.assertNotIn("destination[word] = source", service)
        self.assertIn("AscendC::SetFlag<Event>", intrinsics)
        self.assertIn("AscendC::WaitFlag<Event>", intrinsics)

    def test_urma_cqe_polling_uses_mte2(self):
        service = (
            TRANSPORT / "aicore_transport_service.hpp").read_text()
        self.assertIn("sync_event<AscendC::HardEvent::S_MTE2>", service)
        self.assertIn("DataCopy(cqe_scratch", service)
        self.assertIn("sync_event<AscendC::HardEvent::MTE2_S>", service)
        self.assertNotIn("word0 = cqe->words[0]", service)

    def test_barrier_signal_polling_uses_mte2(self):
        service = (
            TRANSPORT / "aicore_transport_service.hpp").read_text()
        self.assertIn("DataCopyPad(signal_scratch", service)
        self.assertIn("observed = signal_scratch.GetValue(0)", service)
        self.assertNotIn("if (*signal >= generation)", service)

    def test_stage_recording_does_not_reset_from_one_kernel_block(self):
        service = (
            TRANSPORT / "aicore_transport_service.hpp").read_text()
        begin = service.index("__aicore__ inline void record_stage_start(")
        end = service.index("\n}\n", begin)
        record_stage_start = service[begin:end]

        self.assertNotIn("begin_profile(", record_stage_start)
        self.assertIn("profile->generation = generation", record_stage_start)
        self.assertIn("profile->operation = operation", record_stage_start)

    def test_cann_92_device_abi_matches_package_layouts(self):
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            self.skipTest("ASCEND_HOME_PATH is not configured")

        architecture_root = {
            "x86_64": "x86_64-linux",
            "aarch64": "aarch64-linux",
        }.get(platform.machine().lower())
        if architecture_root is None:
            self.skipTest(
                f"unsupported CANN host architecture: {platform.machine()}")

        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "cann_abi_probe"
            include = pathlib.Path(ascend_home) / "include"
            include_paths = []
            hcomm_root = os.environ.get("HCOMM_ROOT")
            if hcomm_root:
                hcomm_arch = pathlib.Path(hcomm_root) / architecture_root
                hcomm_include = hcomm_arch / "include"
                include_paths.extend([
                    hcomm_arch / "pkg_inc",
                    hcomm_include,
                    hcomm_include / "hcomm",
                ])
            include_paths.extend([
                include,
                include / "hcomm",
                pathlib.Path(ascend_home) / architecture_root / "pkg_inc",
                pathlib.Path(ascend_home) / architecture_root / "asc/include",
                pathlib.Path(ascend_home) / architecture_root / "asc/impl",
            ])
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    *(flag for path in include_paths for flag in ("-I", str(path))),
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

    def test_transport_helpers_split_host_and_aicore_callees(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "aicore_callee_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/transport_command_aicore_callee_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_barrier_predicate_uses_simt_callee(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "simt_callee_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/transport_command_simt_callee_probe.cpp"),
                    "-o", str(executable),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(
                compile_probe.returncode, 0, compile_probe.stderr)

            run_probe = subprocess.run(
                [str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run_probe.returncode, 0, run_probe.stderr)

    def test_release_protocol_uses_simt_callee_route_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "simt_route_probe"
            compile_probe = subprocess.run(
                [
                    "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT),
                    str(ROOT / "tests/ascend/release_protocol_simt_callee_probe.cpp"),
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
