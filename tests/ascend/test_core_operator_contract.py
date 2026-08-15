import pathlib
import re
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/ascend/core_operator_contract_probe.cpp"
RUNTIME_PROBE = ROOT / "tests/ascend/core_runtime_contract_probe.cpp"
PRODUCTION_LAYOUT_PROBE = ROOT / "tests/ascend/production_layout_probe.cpp"
ELASTIC = ROOT / "csrc/backends/ascend/elastic"
CORE_OPS = ROOT / "tests/ascend/core_ops"


class AscendCoreOperatorContractTest(unittest.TestCase):
    def test_simt_vf_arguments_have_explicit_device_abi(self):
        sources = {
            name: path.read_text()
            for name, path in {
                "barrier.asc": ELASTIC / "barrier.asc",
                "dispatch.asc": ELASTIC / "dispatch.asc",
                "combine.asc": ELASTIC / "combine.asc",
                "core_operator_compile_probe.asc":
                    CORE_OPS / "core_operator_compile_probe.asc",
            }.items()
        }
        signatures = {}
        for source_name, source in sources.items():
            matches = re.findall(
                r"__simt_vf__\s+inline\s+void\s+(\w+)\s*\((.*?)\)\s*\{",
                source, flags=re.DOTALL)
            self.assertTrue(matches, source_name)
            for function_name, arguments in matches:
                normalized = " ".join(arguments.split())
                signatures[function_name] = normalized
                self.assertNotIn("CoreTiling", normalized, function_name)
                self.assertNotIn("BarrierArguments", normalized, function_name)
                for argument in arguments.split(","):
                    if "*" in argument:
                        self.assertTrue(
                            "__gm__" in argument or "__ubuf__" in argument,
                            f"{function_name}: {argument.strip()}")
                    else:
                        self.assertRegex(
                            " ".join(argument.split()),
                            r"^(transport::DeviceTransportContext|CoreModeFlags|"
                            r"ElementKind|std::uint(32|64)_t|int) \w+$",
                            function_name)

        for function_name in (
                "barrier_producer_vf", "barrier_continuation_vf"):
            signature = signatures[function_name]
            for argument in (
                    "transport::DeviceTransportContext context",
                    "std::uint64_t control_offset", "int world_rank",
                    "std::uint64_t generation"):
                self.assertIn(argument, signature, function_name)
        self.assertIn(
            "ElementKind element_kind",
            signatures["core_operator_compile_probe_vf"])

    def test_production_symmetric_window_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_layout_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_LAYOUT_PROBE), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)

            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_pure_cpp_layout_and_tiling_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "core_operator_contract_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PROBE), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_contract_headers_are_vendor_neutral(self):
        forbidden = ("cuda", "nccl", "nvshmem", "acl/", "ain", "hcomm",
                     "hccl", "urma", "torch_npu", "kernel_operator")
        headers = [ELASTIC / "layout.hpp", ELASTIC / "tiling.hpp"]
        for header in headers:
            self.assertTrue(header.is_file(), str(header))
            includes = [line.strip().lower()
                        for line in header.read_text().splitlines()
                        if line.lstrip().startswith("#include")]
            for token in forbidden:
                self.assertFalse(any(token in include for include in includes),
                                 f"{header}: {token}")

    def test_pure_cpp_runtime_contract(self):
        runtime = ELASTIC / "runtime.cpp"
        self.assertTrue(runtime.is_file(), str(runtime))
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "core_runtime_contract_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(RUNTIME_PROBE), str(runtime),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_aot_kernel_surface_and_build_contract(self):
        header = ELASTIC / "kernels.hpp"
        self.assertTrue(header.is_file(), str(header))
        surface = header.read_text()
        for launch in ("deep_ep_ascend_launch_barrier",
                       "deep_ep_ascend_launch_dispatch",
                       "deep_ep_ascend_launch_dispatch_epilogue",
                       "deep_ep_ascend_launch_combine",
                       "deep_ep_ascend_launch_combine_epilogue"):
            self.assertIn(launch, surface)

        cmake_path = CORE_OPS / "CMakeLists.txt"
        probe_path = CORE_OPS / "core_operator_compile_probe.asc"
        self.assertTrue(cmake_path.is_file(), str(cmake_path))
        self.assertTrue(probe_path.is_file(), str(probe_path))
        cmake = cmake_path.read_text()
        self.assertIn("dav-3510", cmake)
        self.assertNotIn("--enable-simt", cmake)
        for source in ("barrier.asc", "dispatch.asc", "combine.asc"):
            self.assertIn(source, cmake)
        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for case_name in ("adapter-launch-error", "dispatch-normal", "dispatch-expanded",
                          "dispatch-cached", "dispatch-zero-padding",
                          "dispatch-empty", "combine-normal",
                          "combine-expanded", "combine-multiple",
                          "combine-single-reduction", "combine-weights",
                          "combine-bias0", "combine-bias01", "round-trip"):
            self.assertIn(case_name, runner)
        self.assertIn("barrier-local", runner)
        probe = probe_path.read_text()
        self.assertIn("ElementKind", probe)

    def test_kernels_schedule_transport_through_the_facade(self):
        required = {
            "barrier.asc": ("DeviceTransportFacade", "device_barrier"),
            "dispatch.asc": ("DeviceTransportFacade", "put(", "put_value(",
                             "signal(", "device_barrier("),
            "combine.asc": ("DeviceTransportFacade", "put(",
                            "remote_add_release(", "flush(",
                            "device_barrier("),
        }
        forbidden = ("nccl", "nvshmem", "cuda", "ain", "hcomm", "hccl",
                     "urma")
        for name, operations in required.items():
            path = ELASTIC / name
            self.assertTrue(path.is_file(), str(path))
            source = path.read_text()
            for operation in operations:
                self.assertIn(operation, source, f"{name}: {operation}")
            includes = [line.strip().lower()
                        for line in source.splitlines()
                        if line.lstrip().startswith("#include")]
            for token in forbidden:
                self.assertFalse(any(token in include for include in includes),
                                 f"{path}: {token}")

    def test_barrier_has_staged_service_boundaries(self):
        source = (ELASTIC / "barrier.asc").read_text()
        ordered_markers = (
            "service::reset",
            "asc_vf_call<barrier_producer_vf>",
            "service::execute",
            "asc_vf_call<barrier_continuation_vf>",
        )
        for marker in ordered_markers:
            self.assertIn(marker, source)
        positions = [source.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))
        for marker in (
                "arguments.generation", "tiling.launch.num_blocks",
                "threadIdx.x != 0", "DeviceTransportFacade transport(",
                "context, 0", "transport.device_barrier("):
            self.assertIn(marker, source)
        for forbidden in ("HcclBarrier", "HcclAllReduce", "HcclAllGather"):
            self.assertNotIn(forbidden, source)

    def test_production_api_does_not_bypass_transport_gate(self):
        production = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        bindings = (ROOT / "csrc/python_api.cpp").read_text()
        self.assertNotIn("DEEP_EP_ASCEND_TESTING", bindings)
        self.assertNotIn("DEEP_EP_ASCEND_TEST_DIAGNOSTIC", bindings)
        self.assertIn("launch_internal_barrier", production)
        self.assertNotIn("launch_internal_dispatch", production)
        self.assertNotIn("launch_internal_combine", production)
        for marker in ("read_diagnostic", "copy_to_host",
                       "barrier_completion", "barrier_generation_"):
            self.assertIn(marker, production)
        self.assertIn("#if DEEP_EP_ASCEND_TESTING", production)
        self.assertIn("DEEP_EP_ASCEND_TEST_DIAGNOSTIC", production)
        for marker in (
                "diagnostic.abi_version = transport::kTransportCommandAbiVersion",
                "DeviceTransportError::kCompletionTimeout",
                "diagnostic.command_index = 0",
                "TransportCommandOpcode::kBarrier",
                "diagnostic.peer = static_cast<std::uint32_t>(rank_idx_)",
                "diagnostic.channel = 0",
                "diagnostic.backend_status = 0",
                "diagnostic.generation = barrier_generation_"):
            self.assertIn(marker, production)
        for operation in ("barrier", "dispatch", "combine"):
            marker = f'require_transport("{operation}"'
            self.assertIn(marker, production)


if __name__ == "__main__":
    unittest.main()
