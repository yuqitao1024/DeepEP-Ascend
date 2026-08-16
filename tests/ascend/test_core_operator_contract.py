import json
import pathlib
import re
import subprocess
import tempfile
import unittest

from tests.ascend.test_stub_source import PYBIND11_HEADER, TORCH_HEADER


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/ascend/core_operator_contract_probe.cpp"
RUNTIME_PROBE = ROOT / "tests/ascend/core_runtime_contract_probe.cpp"
PRODUCTION_LAYOUT_PROBE = ROOT / "tests/ascend/production_layout_probe.cpp"
PRODUCTION_COMBINE_STATE_PROBE = \
    ROOT / "tests/ascend/production_combine_state_probe.cpp"
PRODUCTION_DISPATCH_STATE_PROBE = \
    ROOT / "tests/ascend/production_dispatch_state_probe.cpp"
PRODUCTION_BARRIER_STATE_PROBE = \
    ROOT / "tests/ascend/production_barrier_state_probe.cpp"
TWO_RANK_DISPATCH = \
    ROOT / "tests/ascend/production/run_two_rank_dispatch.py"
ELASTIC = ROOT / "csrc/backends/ascend/elastic"
CORE_OPS = ROOT / "tests/ascend/core_ops"


class AscendCoreOperatorContractTest(unittest.TestCase):
    def test_simt_vf_arguments_have_explicit_device_abi(self):
        def split_arguments(arguments):
            result = []
            start = 0
            depth = 0
            for index, character in enumerate(arguments):
                if character in "([{":
                    depth += 1
                elif character in ")]}":
                    depth -= 1
                elif character == "," and depth == 0:
                    result.append(arguments[start:index].strip())
                    start = index + 1
            result.append(arguments[start:].strip())
            return result

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
        signature_arguments = {}
        calls_by_function = {}
        for source_name, source in sources.items():
            matches = re.findall(
                r"__simt_vf__\s+inline\s+void\s+(\w+)\s*\((.*?)\)\s*\{",
                source, flags=re.DOTALL)
            self.assertTrue(matches, source_name)
            for function_name, arguments in matches:
                normalized = " ".join(arguments.split())
                signatures[function_name] = normalized
                signature_arguments[function_name] = split_arguments(arguments)
                self.assertNotIn("CoreTiling", normalized, function_name)
                self.assertNotIn("BarrierArguments", normalized, function_name)
                self.assertNotIn(
                    "DeviceTransportContext", normalized, function_name)
                for argument in arguments.split(","):
                    if "*" in argument:
                        self.assertTrue(
                            "__gm__" in argument or "__ubuf__" in argument,
                            f"{function_name}: {argument.strip()}")
                    else:
                        self.assertRegex(
                            " ".join(argument.split()),
                            r"^(CoreModeFlags|ElementKind|"
                            r"std::uintptr_t|std::uint(32|64)_t|int) \w+$",
                            function_name)

            calls = re.findall(
                r"asc_vf_call<(\w+)>\s*\((.*?)\);",
                source, flags=re.DOTALL)
            self.assertTrue(calls, source_name)
            for function_name, arguments in calls:
                calls_by_function[function_name] = arguments
                self.assertNotRegex(
                    arguments,
                    r"(?<![\w.])tiling\.transport_context(?![\w.])",
                    function_name)

        for function_name in (
                "barrier_producer_vf", "barrier_continuation_vf"):
            signature = signatures[function_name]
            for argument in (
                    "std::uint32_t transport_abi_version",
                    "std::uint32_t transport_struct_size",
                    "std::uintptr_t transport_local_window_base",
                    "std::uintptr_t transport_backend_context",
                    "std::uint64_t control_offset", "int world_rank",
                    "std::uint64_t generation"):
                self.assertIn(argument, signature, function_name)
        self.assertIn(
            "std::uint64_t timeout_cycles",
            signatures["barrier_producer_vf"])
        for function_name in ("dispatch_producer_vf", "combine_producer_vf"):
            signature = signatures[function_name]
            for argument in (
                    "std::uint32_t transport_abi_version",
                    "std::uint32_t transport_struct_size",
                    "int transport_world_size",
                    "int transport_scale_up_size",
                    "std::uintptr_t transport_backend_context"):
                self.assertIn(argument, signature, function_name)

        expected_transport_arguments = {
            "barrier_producer_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.local_window_base",
                "tiling.transport_context.backend_context",
            },
            "barrier_continuation_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.local_window_base",
                "tiling.transport_context.backend_context",
            },
            "dispatch_producer_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.local_window_base",
                "tiling.transport_context.topology.world_rank",
                "tiling.transport_context.topology.world_size",
                "tiling.transport_context.topology.scale_up_size",
                "tiling.transport_context.backend_context",
            },
            "combine_producer_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.topology.world_rank",
                "tiling.transport_context.topology.world_size",
                "tiling.transport_context.topology.scale_up_size",
                "tiling.transport_context.backend_context",
            },
        }
        for function_name, expected in expected_transport_arguments.items():
            call = calls_by_function[function_name]
            observed = set(re.findall(
                r"tiling\.transport_context(?:\.topology)?\.\w+", call))
            self.assertEqual(observed, expected, function_name)
        self.assertIn(
            "ElementKind element_kind",
            signatures["core_operator_compile_probe_vf"])
        dispatch_kernel_match = re.search(
            r"__global__\s+__vector__\s+void\s+dispatch_kernel\s*"
            r"\((.*?)\)\s*\{", sources["dispatch.asc"], flags=re.DOTALL)
        self.assertIsNotNone(dispatch_kernel_match)
        kernel_parameters = split_arguments(dispatch_kernel_match.group(1))
        self.assertNotIn("DispatchArguments", dispatch_kernel_match.group(1))
        kernel_pointer_names = set()
        for parameter in kernel_parameters:
            if "*" not in parameter:
                continue
            self.assertIn("__gm__", parameter, parameter)
            kernel_pointer_names.add(parameter.rsplit(maxsplit=1)[-1])
        for function_name in (
                "dispatch_producer_vf", "dispatch_epilogue_vf"):
            parameters = signature_arguments[function_name]
            call_arguments = split_arguments(calls_by_function[function_name])
            self.assertEqual(len(call_arguments), len(parameters) + 1)
            for parameter, call_argument in zip(
                    parameters, call_arguments[1:]):
                if "*" in parameter:
                    self.assertIn(
                        call_argument, kernel_pointer_names,
                        f"{function_name}: {call_argument}")

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

    def test_production_combine_state_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_combine_state_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_COMBINE_STATE_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_production_dispatch_state_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_dispatch_state_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_DISPATCH_STATE_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_production_barrier_sequence_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_barrier_state_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_BARRIER_STATE_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
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
        headers = [ELASTIC / "dispatch_state.hpp", ELASTIC / "layout.hpp",
                   ELASTIC / "tiling.hpp"]
        for header in headers:
            self.assertTrue(header.is_file(), str(header))
            includes = [line.strip().lower()
                        for line in header.read_text().splitlines()
                        if line.lstrip().startswith("#include")]
            for token in forbidden:
                self.assertFalse(any(token in include for include in includes),
                                 f"{header}: {token}")

    def test_dispatch_locality_helper_is_device_callable(self):
        """Catches a host-only helper call from dispatch SIMT functions."""
        probe_source = r'''
#include <cstdint>

#define DEEP_EP_ASCEND_SIMT_DEVICE 1
#define __SIMT_DEVICE_FUNCTIONS_DECL__ extern "C"
#include "csrc/backends/ascend/elastic/dispatch_state.hpp"

auto* dispatch_locality_definition =
    &deep_ep::ascend::elastic::is_dispatch_expert_local;

extern "C" bool is_dispatch_expert_local(
    std::int64_t, std::uint64_t, std::uint64_t) noexcept;

int main() {
    return dispatch_locality_definition(1, 0, 2) &&
        is_dispatch_expert_local(1, 0, 2) ? 0 : 1;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            source = directory / "dispatch_state_device_probe.cpp"
            binary = directory / "dispatch_state_device_probe"
            source.write_text(probe_source)
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(source), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

        include_contracts = (
            (ELASTIC / "dispatch.asc", '#include "dispatch_state.hpp"'),
            (CORE_OPS / "core_operator_runner.asc",
             '#include "csrc/backends/ascend/elastic/dispatch_state.hpp"'),
        )
        for path, state_include in include_contracts:
            source = path.read_text()
            self.assertLess(
                source.index("#define DEEP_EP_ASCEND_SIMT_DEVICE 1"),
                source.index(state_include), str(path))

    def test_dispatch_public_identity_helpers_execute_on_host(self):
        """Catches receiver-ranked slots, local metadata, and global routes."""
        probe_source = r'''
#include <cstdint>

#include "csrc/backends/ascend/elastic/dispatch_state.hpp"

using namespace deep_ep::ascend::elastic;

int main() {
    return encode_dispatch_source_index(1, 4, 3) == 7 &&
        decode_dispatch_source_rank(7, 4) == 1 &&
        decode_dispatch_local_index(7, 4) == 3 &&
        encode_dispatch_source_index(1, 2, 1) == 3 &&
        localize_dispatch_expert(2, 2, 2) == 0 &&
        localize_dispatch_expert(3, 2, 2) == 1 &&
        localize_dispatch_expert(0, 2, 2) == -1 &&
        localize_dispatch_expert(-1, 2, 2) == -1 &&
        is_dispatch_local_index(3, 4) &&
        !is_dispatch_local_index(4, 4) &&
        !is_dispatch_local_index(-1, 4) ? 0 : 1;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            source = directory / "dispatch_public_identity_probe.cpp"
            binary = directory / "dispatch_public_identity_probe"
            source.write_text(probe_source)
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(source), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

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
        for definition in (
                "DEEP_EP_ASCEND_STAGED_URMA=1",
                "DEEP_EP_ASCEND_AICORE_URMA_SERVICE=1",
                "DEEP_EP_ASCEND_AICORE_WQE_CALLEE=__aicore__"):
            self.assertIn(definition, cmake)
        for source in ("barrier.asc", "dispatch.asc", "combine.asc"):
            self.assertIn(source, cmake)
        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for case_name in ("adapter-launch-error", "dispatch-normal", "dispatch-expanded",
                          "dispatch-cached", "dispatch-zero-padding",
                          "dispatch-public-identity",
                          "dispatch-empty", "combine-normal",
                          "combine-state-probe",
                          "combine-expanded", "combine-multiple",
                          "combine-single-reduction", "combine-weights",
                          "combine-bias0", "combine-bias01", "round-trip"):
            self.assertIn(case_name, runner)
        self.assertIn("barrier-local", runner)
        probe = probe_path.read_text()
        self.assertIn("ElementKind", probe)

    def test_kernels_schedule_transport_through_the_facade(self):
        required = {
            "barrier.asc": ("DeviceTransportFacade", "store_release(",
                            "device_barrier"),
            "dispatch.asc": ("DeviceTransportFacade", "put(", "put_value(",
                             "signal(", "device_barrier("),
            "combine.asc": ("DeviceTransportFacade", "put(", "put_value(",
                            "signal(", "device_barrier("),
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
                "context, 0", "arguments.timeout_cycles",
                "transport.device_barrier("):
            self.assertIn(marker, source)
        self.assertNotIn("transport.put_value(", source)
        self.assertNotIn("transport.load_acquire(", source)
        for forbidden in ("HcclBarrier", "HcclAllReduce", "HcclAllGather"):
            self.assertNotIn(forbidden, source)

    def test_dispatch_has_fixed_shard_service_boundaries(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        ordered_markers = (
            "service::reset",
            "asc_vf_call<dispatch_producer_vf>",
            "service::execute",
            "asc_vf_call<dispatch_epilogue_vf>",
        )
        for marker in ordered_markers:
            self.assertIn(marker, source)
        positions = [source.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))
        for marker in (
                "transport_world_rank", "transport_local_window_base",
                "arguments.generation", "arguments.timeout_cycles",
                "dispatch_control_offset", "dispatch_control_bytes",
                "dispatch_receive_offset", "dispatch_receive_shard_bytes",
                "dispatch_receive_shard_count", "dispatch_receive_bytes",
                "dispatch_staging_offset", "dispatch_staging_shard_bytes",
                "dispatch_staging_shard_count", "dispatch_staging_bytes",
                "destination_rank",
                "transport.device_barrier(", "transport.consumed_generation()"):
            self.assertIn(marker, source)
        self.assertNotRegex(
            source,
            r"transport\.(?:put|put_value|signal)\(\s*"
            r"transport::TransportTeam::k(?:World|ScaleUp),\s*0,")
        for legacy in ("dispatch_source_shard_bytes",
                       "dispatch_source_shard_count"):
            self.assertNotIn(legacy, source)

    def test_combine_has_fixed_shard_service_boundaries(self):
        """Catches local-only combine and publication before service execute."""
        source = (ELASTIC / "combine.asc").read_text()
        ordered_markers = (
            "service::reset",
            "asc_vf_call<combine_producer_vf>",
            "service::execute",
            "asc_vf_call<combine_epilogue_vf>",
        )
        for marker in ordered_markers:
            self.assertIn(marker, source)
        positions = [source.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))

        for marker in (
                "transport_world_rank", "transport_local_window_base",
                "arguments.generation", "arguments.timeout_cycles",
                "arguments.num_source_rows", "arguments.num_input_rows",
                "arguments.local_window_base",
                "combine_control_offset", "combine_control_bytes",
                "combine_receive_offset", "combine_receive_shard_bytes",
                "combine_receive_shard_count", "combine_receive_bytes",
                "combine_staging_offset", "combine_staging_shard_bytes",
                "combine_staging_shard_count", "combine_staging_bytes",
                "decode_dispatch_source_rank(",
                "decode_dispatch_local_index(", "destination_rank",
                "CombineRecordHeader", "CombineControlSlot",
                "CombineProtocolError::kInvalidPrefix",
                "CombineProtocolError::kDuplicateRecord",
                "raw_record_capacity", "count * combine_record_bytes",
                "header->origin_token", "header->contributor_rank",
                "header->contribution_lane != header->master_lane",
                "record_count", "remote_count, count",
                "remote_generation, generation", "transport.signal(",
                "transport.device_barrier(",
                "transport.consumed_generation()",
                "float value = 0.0F",
                "value += static_cast<float>(payload[hidden])",
                "combined_topk_weights[index] = 0.0F"):
            self.assertIn(marker, source)

        self.assertNotRegex(
            source,
            r"transport\.(?:put|put_value|signal)\(\s*"
            r"transport::TransportTeam::k(?:World|ScaleUp),\s*0,")
        for forbidden in (
                "transport.remote_add_release(",
                "transport.get_symmetric_pointer(", "peer_address_table"):
            self.assertNotIn(forbidden, source)
        self.assertNotRegex(
            source, r"transport\.device_barrier\([^;]*,\s*0\s*\)")

        execute_position = source.index("service::execute")
        epilogue_position = source.index(
            "asc_vf_call<combine_epilogue_vf>")
        self.assertLess(execute_position, epilogue_position)

    def test_combine_completion_gate_and_runner_probe(self):
        """Catches output publication after failed transport execution."""
        source = (ELASTIC / "combine.asc").read_text()
        epilogue_begin = source.index(
            "__simt_vf__ inline void combine_epilogue_vf")
        epilogue = source[epilogue_begin:]
        self.assertIn("is_clean_combine_transport_completion(", epilogue)
        gate_position = epilogue.index(
            "is_clean_combine_transport_completion(")
        output_position = epilogue.index(
            "for (std::uint64_t token = 0; token < num_tokens; ++token)")
        completion_position = epilogue.index("&control->combine_generation")
        self.assertLess(gate_position, output_position)
        self.assertLess(gate_position, completion_position)
        for marker in (
                "transport::device::detail::command_queue(context)",
                "transport::device::detail::diagnostic(queue)",
                "diagnostic->abi_version", "diagnostic->generation",
                "diagnostic->error"):
            self.assertIn(marker, epilogue)

        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for marker in (
                "combine-state-probe", "combine_state_probe_kernel<<<",
                "is_clean_combine_transport_completion(",
                "combine_receive_shard_address(",
                "combine_staging_shard_address("):
            self.assertIn(marker, runner)

        tiling_begin = runner.index("CoreTiling make_combine_tiling")
        tiling_end = runner.index("bool run_combine_case", tiling_begin)
        combine_tiling = runner[tiling_begin:tiling_end]
        for assignment in (
                "input.topology.world_rank = 0",
                "input.topology.world_size = 2",
                "input.topology.scale_up_rank = 0",
                "input.topology.scale_up_size = 2",
                "input.topology.scale_out_rank = 0",
                "input.topology.scale_out_size = 1"):
            self.assertIn(assignment, combine_tiling)

        normal_begin = runner.index(
            'else if (std::strcmp(argv[2], "combine-normal") == 0)')
        normal_end = runner.index("else if", normal_begin + 1)
        normal_case = runner[normal_begin:normal_end]
        self.assertIn(
            "run_combine_case(stream, false, false, 0)", normal_case)

    def test_dispatch_preflights_protocol_state_before_publication(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        for marker in (
                "DispatchProtocolError::kInvalidTopk",
                "DispatchProtocolError::kInvalidCachedSlot",
                "DispatchProtocolError::kInvalidCachedPrefix",
                "DispatchProtocolError::kInvalidCachedMetadata",
                "DispatchProtocolError::kCapacityOverflow",
                "DeviceTransportError::kInvalidProtocol",
                "record_dispatch_protocol_error",
                "validate_cached_dispatch_state",
                "dispatch_output_capacity",
                "arguments.workspace",
                "tiling.workspace_layout.scratch_offset"):
            self.assertIn(marker, source)
        self.assertIn("observed_transport_error", source)
        diagnostic_check = source.index("observed_transport_error")
        backend_status_write = source.index("&diagnostic->backend_status")
        self.assertLess(diagnostic_check, backend_status_write)
        self.assertNotIn("RemoteAction::signal_add", source)
        self.assertIn("RemoteAction::signal_increment", source)
        count_position = source.index("remote_count, count")
        generation_position = source.index("remote_generation, generation")
        signal_position = source.index("transport.signal(")
        barrier_position = source.index("transport.device_barrier(")
        self.assertLess(count_position, generation_position)
        self.assertLess(generation_position, signal_position)
        self.assertLess(signal_position, barrier_position)

        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for case_name in ("dispatch-invalid-topk",
                          "dispatch-invalid-cache"):
            self.assertIn(case_name, runner)

    def test_cached_dispatch_validates_private_counts_and_local_prefixes(self):
        """Catches public count repair and nonlocal cached expert counting."""
        source = (ELASTIC / "dispatch.asc").read_text()
        producer_begin = source.index(
            "__simt_vf__ inline void dispatch_producer_vf")
        epilogue_begin = source.index(
            "__simt_vf__ inline void dispatch_epilogue_vf")
        producer = source[producer_begin:epilogue_begin]
        self.assertNotIn("prefix_per_rank[", producer)
        self.assertIn("local_count_address", producer)
        epilogue = source[epilogue_begin:]
        self.assertIn(
            "source_counts[0] = transport.load_acquire(local_count_address)",
            epilogue)

        derived_begin = source.index("std::int32_t derived_prefix")
        derived_end = source.index("if (cached && expanded)", derived_begin)
        derived_prefix = source[derived_begin:derived_end]
        self.assertIn("expert_is_local", derived_prefix)
        self.assertIn("is_dispatch_expert_local(", derived_prefix)
        self.assertIn("if (expert_is_local)", derived_prefix)
        self.assertIn("first_local_expert", derived_prefix)
        self.assertIn("num_local_experts", derived_prefix)

        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for case_name in ("dispatch-invalid-prefix",
                          "dispatch-invalid-slot",
                          "dispatch-cached-mixed-rank"):
            self.assertIn(case_name, runner)
        for mutation in (
                "Mutation caught: producer-side cached rank-prefix repair.",
                "Mutation caught: cached slot validation after publication.",
                "Mutation caught: nonlocal lanes counted in cached expert prefixes."):
            self.assertIn(mutation, runner)
        mixed_begin = runner.index(
            "bool run_dispatch_cached_mixed_rank(aclrtStream stream)")
        mixed_end = runner.index("std::uint16_t float_to_bfloat16", mixed_begin)
        mixed_case = runner[mixed_begin:mixed_end]
        self.assertIn("cached_mixed_rank_prefix_probe_kernel<<<", mixed_case)
        self.assertNotIn("make_dispatch_tiling", mixed_case)
        self.assertIn("kMixedRankWorldSize", runner)
        self.assertIn("kMixedRankWorldRank", runner)

    def test_dispatch_kernel_uses_public_source_identity_contract(self):
        """Catches receiver-ranked slots, receiver token bounds, and global routes."""
        source = (ELASTIC / "dispatch.asc").read_text()
        producer_begin = source.index(
            "__simt_vf__ inline void dispatch_producer_vf")
        epilogue_begin = source.index(
            "__simt_vf__ inline void dispatch_epilogue_vf")
        producer = source[:epilogue_begin]
        epilogue = source[epilogue_begin:]
        self.assertIn("encode_dispatch_source_index(", producer)
        self.assertIn("decode_dispatch_source_rank(", producer)
        self.assertIn("decode_dispatch_local_index(", producer)
        self.assertIn("is_dispatch_local_index(", epilogue)
        self.assertIn("encode_dispatch_source_index(", epilogue)
        self.assertIn("localize_dispatch_expert(", epilogue)

        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        self.assertIn("dispatch-public-identity", runner)
        for mutation in (
                "Mutation caught: destination-ranked cached slots.",
                "Mutation caught: receiver-local source token bounds.",
                "Mutation caught: global normal-mode receive routes."):
            self.assertIn(mutation, runner)

    def test_production_api_does_not_bypass_transport_gate(self):
        production = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        bindings = (ROOT / "csrc/python_api.cpp").read_text()
        self.assertNotIn("DEEP_EP_ASCEND_TESTING", bindings)
        self.assertNotIn("DEEP_EP_ASCEND_TEST_DIAGNOSTIC", bindings)
        self.assertIn("launch_internal_barrier", production)
        self.assertIn("launch_internal_dispatch", production)
        self.assertNotIn("launch_internal_combine", production)
        for marker in ("read_diagnostic", "copy_to_host",
                       "barrier_completion", "BarrierSequence",
                       "timeout_cycles_from_seconds"):
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
                       "diagnostic.generation = attempt.generation()",
                       "attempt.complete()"):
            self.assertIn(marker, production)
        for operation in ("barrier", "dispatch", "combine"):
            marker = f'require_transport("{operation}"'
            self.assertIn(marker, production)

        dispatch_capabilities = production[
            production.index("kDispatchCapabilities"):production.index(
                "kCombineCapabilities")]
        self.assertNotIn("kDirectPeerPointer", dispatch_capabilities)
        self.assertNotIn("kRemoteAtomicAddRelease", dispatch_capabilities)
        for marker in (
                "DispatchSequence", "DispatchHandleDescriptor",
                "copy_from_host", "copy_to_host", "synchronize_stream",
                "read_diagnostic", "num_sms == 1", "num_qps == 0",
                "do_expand", "do_zero_padding",
                "use_tma_aligned_col_major_sf"):
            self.assertIn(marker, production)

    def test_public_dispatch_probe_executes(self):
        probe = ROOT / "tests/ascend/production_dispatch_probe.cpp"
        self.assertTrue(probe.is_file(), str(probe))
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            (directory / "pybind11").mkdir()
            (directory / "torch").mkdir()
            (directory / "pybind11/pybind11.h").write_text(PYBIND11_HEADER)
            (directory / "torch/python.h").write_text(TORCH_HEADER)
            binary = directory / "production_dispatch_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-DDEEP_EP_ASCEND_TESTING=1", f"-I{directory}", f"-I{ROOT}", str(probe),
                 str(ELASTIC / "runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/transport/cann_transport.cpp"),
                 "-o", str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_two_rank_dispatch_harness_contract(self):
        expected_cases = [
            "asymmetric-routing",
            "empty-input",
            "negative-one-route",
            "duplicate-destination-rank",
            "multiple-experts",
            "optional-weights",
            "expanded",
            "aligned-zero-padding",
            "aligned-near-capacity",
            "cached-reuse",
            "cached-aligned-near-capacity",
            "sequential-100-generations",
            "combine-gated",
            "invalid-expert-diagnostics",
        ]
        result = subprocess.run(
            ["python3", str(TWO_RANK_DISPATCH), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "case_names": expected_cases,
                "contract_checks": [
                    "literal-gather",
                    "independent-reference",
                    "cached-handle-reuse",
                    "rank-barriers",
                    "distributed-failure-reduction",
                    "buffer-cleanup",
                ],
                "dispatch_surface": "Buffer.dispatch",
                "expected_world_size": 2,
                "reference": "gathered-literal-inputs",
            })


if __name__ == "__main__":
    unittest.main()
