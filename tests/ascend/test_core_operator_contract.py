import json
import importlib.util
import contextlib
import io
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests.ascend.test_stub_source import PYBIND11_HEADER, TORCH_HEADER


ROOT = pathlib.Path(__file__).resolve().parents[2]
HOST_SANITIZER_FLAGS = (
    ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    if os.environ.get("DEEP_EP_ASCEND_HOST_SANITIZERS") == "1" else [])
PROBE = ROOT / "tests/ascend/core_operator_contract_probe.cpp"
RUNTIME_PROBE = ROOT / "tests/ascend/core_runtime_contract_probe.cpp"
PRODUCTION_LAYOUT_PROBE = ROOT / "tests/ascend/production_layout_probe.cpp"
HYBRID_BUFFER_OWNERSHIP_PROBE = \
    ROOT / "tests/ascend/hybrid_buffer_ownership_probe.cpp"
PRODUCTION_COMBINE_STATE_PROBE = \
    ROOT / "tests/ascend/production_combine_state_probe.cpp"
PRODUCTION_COMBINE_SEMANTICS_PROBE = \
    ROOT / "tests/ascend/production_combine_semantics_probe.cpp"
PRODUCTION_COMBINE_PRODUCER_PROBE = \
    ROOT / "tests/ascend/production_combine_producer_probe.cpp"
PRODUCTION_COMBINE_DEVICE_POINTER_PROBE = \
    ROOT / "tests/ascend/production_combine_device_pointer_probe.cpp"
PRODUCTION_DISPATCH_STATE_PROBE = \
    ROOT / "tests/ascend/production_dispatch_state_probe.cpp"
DISPATCH_STATE_HOST_DOMAIN_PROBE = \
    ROOT / "tests/ascend/dispatch_state_host_domain_probe.cpp"
PRODUCTION_BARRIER_STATE_PROBE = \
    ROOT / "tests/ascend/production_barrier_state_probe.cpp"
PRODUCTION_OPERATION_COORDINATOR_PROBE = \
    ROOT / "tests/ascend/production_operation_coordinator_probe.cpp"
PRODUCTION_BUFFER_LIFECYCLE_PROBE = \
    ROOT / "tests/ascend/production_buffer_lifecycle_probe.cpp"
PRODUCTION_RUNTIME_LIFECYCLE_PROBE = \
    ROOT / "tests/ascend/production_lifecycle_probe.cpp"
TWO_RANK_DISPATCH = \
    ROOT / "tests/ascend/production/run_two_rank_dispatch.py"
TWO_RANK_COMBINE = \
    ROOT / "tests/ascend/production/run_two_rank_combine.py"
SCALE_UP_SMOKE = \
    ROOT / "tests/ascend/production/run_scale_up_smoke.py"
ASYNC_OVERLAP = \
    ROOT / "tests/ascend/production/run_async_overlap.py"
STREAM_EVENT_CAPABILITY_PROBE = \
    ROOT / "tests/ascend/stream_event/capability_probe.cpp"
STREAM_EVENT_CAPABILITY_RUNNER = \
    ROOT / "tests/ascend/stream_event/run_capability_probe.py"
ASYNC_RUNTIME_PROBE = ROOT / "tests/ascend/async_runtime_probe.cpp"
ELASTIC = ROOT / "csrc/backends/ascend/elastic"
CORE_OPS = ROOT / "tests/ascend/core_ops"


class AscendCoreOperatorContractTest(unittest.TestCase):
    def test_async_runtime_stream_event_contract(self):
        runtime = ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "async_runtime_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(ASYNC_RUNTIME_PROBE), str(runtime),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_async_runtime_uses_an_explicit_current_stream_device(self):
        runtime = (
            ROOT / "csrc/backends/ascend/runtime/stream_event.cpp").read_text()
        self.assertIn("c10_npu::getCurrentNPUStream(device_index)", runtime)
        self.assertNotIn("c10_npu::getCurrentNPUStream()", runtime)
        self.assertIn(
            "runtime_current_device(nullptr, &device_index)", runtime)
        self.assertLess(
            runtime.index("#include <torch_npu/csrc/core/npu/NPUStream.h>"),
            runtime.index("#include <acl/acl_rt.h>"))

    def test_stream_event_capability_probe_contract(self):
        probe = STREAM_EVENT_CAPABILITY_PROBE.read_text()
        runner = STREAM_EVENT_CAPABILITY_RUNNER.read_text()

        self.assertIn("#include <acl/acl_rt.h>", probe)
        self.assertIn(
            "#include <torch_npu/csrc/core/npu/NPUGuard.h>", probe)
        for call in (
                "c10_npu::getStreamFromPool(true, device_index)",
                "c10_npu::getCurrentNPUStream(device_index)",
                "c10_npu::NPUStreamGuard guard(comm_stream)",
                "aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC)",
                "aclrtRecordEvent(event, compute_stream.stream(false))",
                "aclrtStreamWaitEvent(comm_stream.stream(false), event)",
                "aclrtQueryEventStatus(event, &status)",
                "wait_for_event(event, timeout_ms)",
                "aclrtDestroyEvent(event)",
                "c10_npu::NPUCachingAllocator::recordStream("):
            self.assertIn(call, probe)

        self.assertIn("ACL_EVENT_RECORDED_STATUS_NOT_READY", probe)
        self.assertIn("std::chrono::steady_clock", probe)

        self.assertIn("auto source = torch::zeros({16}, options);", probe)
        self.assertIn("source.fill_(kExpectedSourceValue);", probe)
        self.assertLess(
            probe.index("source.fill_(kExpectedSourceValue);"),
            probe.index("aclrtRecordEvent(event, compute_stream.stream(false))"))
        self.assertIn("destination.to(torch::kCPU)", probe)
        self.assertIn("torch::full_like(observed, kExpectedSourceValue)", probe)
        self.assertLess(
            probe.index("wait_for_event(completion_event, timeout_ms);"),
            probe.index("destination.to(torch::kCPU)"))

        self.assertIn(
            "torch.npu.Stream(stream_id=stream_id, device_index=device_index, "
            "device_type=device_type)", runner)
        self.assertIn("with torch.npu.stream(stream):", runner)
        self.assertIn('os.environ["ASCEND_HOME_PATH"]', runner)
        self.assertIn('cann_root / "aarch64-linux" / "include"', runner)
        self.assertIn('cann_root / "aarch64-linux" / "lib64"', runner)
        self.assertNotIn("aclrtSynchronizeDevice", probe)
        self.assertNotIn("aclrtSynchronizeEventWithTimeout", probe)
        self.assertNotIn("torch.npu.synchronize", runner)

    def test_route_aware_kernels_receive_complete_topology_and_timeout(self):
        required_parameters = (
            "transport_topology_abi_version",
            "transport_topology_struct_size",
            "transport_world_rank",
            "transport_world_size",
            "transport_scale_up_rank",
            "transport_scale_up_size",
            "transport_scale_out_rank",
            "transport_scale_out_size",
            "transport_scale_up_direct",
            "transport_topology_kind",
            "transport_topology_epoch",
            "std::uint64_t transport_capabilities",
            "std::uintptr_t transport_channel_table",
            "std::uintptr_t transport_peer_address_table",
            "std::uint64_t timeout_cycles",
        )
        required_arguments = (
            "topology.abi_version",
            "topology.struct_size",
            "topology.world_rank",
            "topology.world_size",
            "topology.scale_up_rank",
            "topology.scale_up_size",
            "topology.scale_out_rank",
            "topology.scale_out_size",
            "topology.scale_up_direct",
            "topology.kind",
            "topology.epoch",
            "tiling.transport_context.capabilities",
            "tiling.transport_context.channel_table",
            "tiling.transport_context.peer_address_table",
            "timeout_cycles",
        )
        for source_name in ("dispatch.asc", "combine.asc"):
            source = (ELASTIC / source_name).read_text()
            for suffix in ("producer_vf", "epilogue_vf"):
                function_name = source_name.removesuffix(".asc") + "_" + suffix
                signature = re.search(
                    rf"__simt_vf__\s+inline\s+void\s+{function_name}\s*"
                    rf"\((.*?)\)\s*\{{", source, flags=re.DOTALL)
                self.assertIsNotNone(signature, function_name)
                for parameter in required_parameters:
                    self.assertIn(parameter, signature.group(1), function_name)
                call = re.search(
                    rf"asc_vf_call<{function_name}>\s*\((.*?)\);",
                    source, flags=re.DOTALL)
                self.assertIsNotNone(call, function_name)
                for argument in required_arguments:
                    self.assertIn(argument, call.group(1), function_name)
            operation = source_name.removesuffix(".asc")
            self.assertRegex(
                source,
                rf"(?s)extern \"C\" int deep_ep_ascend_launch_{operation}"
                rf"\s*\(.*?arguments\.generation,\s*"
                rf"arguments\.timeout_cycles,",
                operation)

    def test_remote_operator_commands_reuse_checked_team_peer(self):
        release = (ELASTIC / "release_protocol.hpp").read_text()
        for source_name, signal_name in (
                ("dispatch.asc", "kDispatchReleaseSignalIndex"),
                ("combine.asc", "kCombineReleaseSignalIndex")):
            source = (ELASTIC / source_name).read_text()
            self.assertIn("checked_device_team_peer_for_world_rank(", source,
                          source_name)
            self.assertIn("route.team", source, source_name)
            self.assertIn("route.peer", source, source_name)
            self.assertIn(
                "release_protocol::publish_control_and_release(",
                source, source_name)
            self.assertIn(
                "release_protocol::observe_release_control(",
                source, source_name)
            self.assertNotIn(
                "TransportTeam::kScaleUp, destination_rank", source,
                source_name)
            self.assertIn(signal_name, source, source_name)
            self.assertEqual(source.count("transport.device_barrier("), 2,
                             source_name)
        barrier = (ELASTIC / "barrier.asc").read_text()
        self.assertEqual(barrier.count("transport.device_barrier("), 1)
        for marker in (
                "checked_device_team_peer_for_world_rank(", "route.team",
                "route.peer", "facade.put_value(", "facade.signal(",
                "facade.wait_signal(", "facade.read_signal("):
            self.assertIn(marker, release)

    def test_production_combine_producer_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_combine_producer"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_COMBINE_PRODUCER_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def _run_production_combine_semantics_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_combine_semantics"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_COMBINE_SEMANTICS_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_production_combine_device_pointer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_combine_device_pointer"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_COMBINE_DEVICE_POINTER_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

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

        def braced_block_at(source, statement_start):
            opening = source.index("{", statement_start)
            depth = 0
            for index in range(opening, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return source[opening + 1:index], index + 1
            raise AssertionError("unterminated producer block")

        def brace_depth_at(source, position):
            return source[:position].count("{") - source[:position].count("}")

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

        barrier_region_arguments = {
            "barrier_producer_vf": (
                "std::uint64_t generation_offset",
                "std::uint64_t generation_count"),
            "barrier_continuation_vf": (
                "std::uint64_t completion_offset",
                "std::uint64_t completion_count"),
        }
        for function_name, region_arguments in barrier_region_arguments.items():
            signature = signatures[function_name]
            for argument in (
                    "std::uint32_t transport_abi_version",
                    "std::uint32_t transport_struct_size",
                    "std::uint64_t transport_capabilities",
                    "std::uintptr_t transport_local_window_base",
                    "std::uintptr_t transport_channel_table",
                    "std::uintptr_t transport_peer_address_table",
                    "std::uintptr_t transport_backend_context",
                    *region_arguments, "int world_rank",
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

        for function_name in barrier_region_arguments:
            signature = signatures[function_name]
            for argument in (
                "std::uint32_t transport_topology_abi_version",
                "std::uint32_t transport_topology_struct_size",
                "int transport_world_size",
                "int transport_scale_up_rank",
                "int transport_scale_up_size",
                "int transport_scale_out_rank",
                "int transport_scale_out_size",
                "std::uint32_t transport_scale_up_direct",
                "std::uint32_t transport_topology_kind",
                "std::uint64_t transport_topology_epoch"):
                self.assertIn(
                    argument, signature,
                    f"{function_name} must reconstruct the complete topology")

        expected_transport_arguments = {
            "barrier_producer_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.capabilities",
                "tiling.transport_context.local_window_base",
                "tiling.transport_context.channel_table",
                "tiling.transport_context.peer_address_table",
                "tiling.transport_context.topology.abi_version",
                "tiling.transport_context.topology.struct_size",
                "tiling.transport_context.topology.world_rank",
                "tiling.transport_context.topology.world_size",
                "tiling.transport_context.topology.scale_up_rank",
                "tiling.transport_context.topology.scale_up_size",
                "tiling.transport_context.topology.scale_out_rank",
                "tiling.transport_context.topology.scale_out_size",
                "tiling.transport_context.topology.scale_up_direct",
                "tiling.transport_context.topology.kind",
                "tiling.transport_context.topology.epoch",
                "tiling.transport_context.backend_context",
            },
            "barrier_continuation_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.capabilities",
                "tiling.transport_context.local_window_base",
                "tiling.transport_context.channel_table",
                "tiling.transport_context.peer_address_table",
                "tiling.transport_context.topology.abi_version",
                "tiling.transport_context.topology.struct_size",
                "tiling.transport_context.topology.world_rank",
                "tiling.transport_context.topology.world_size",
                "tiling.transport_context.topology.scale_up_rank",
                "tiling.transport_context.topology.scale_up_size",
                "tiling.transport_context.topology.scale_out_rank",
                "tiling.transport_context.topology.scale_out_size",
                "tiling.transport_context.topology.scale_up_direct",
                "tiling.transport_context.topology.kind",
                "tiling.transport_context.topology.epoch",
                "tiling.transport_context.backend_context",
            },
            "dispatch_producer_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.capabilities",
                "tiling.transport_context.local_window_base",
                "tiling.transport_context.channel_table",
                "tiling.transport_context.peer_address_table",
                "tiling.transport_context.topology.abi_version",
                "tiling.transport_context.topology.struct_size",
                "tiling.transport_context.topology.world_rank",
                "tiling.transport_context.topology.world_size",
                "tiling.transport_context.topology.scale_up_rank",
                "tiling.transport_context.topology.scale_up_size",
                "tiling.transport_context.topology.scale_out_rank",
                "tiling.transport_context.topology.scale_out_size",
                "tiling.transport_context.topology.scale_up_direct",
                "tiling.transport_context.topology.kind",
                "tiling.transport_context.topology.epoch",
                "tiling.transport_context.backend_context",
            },
            "combine_producer_vf": {
                "tiling.transport_context.abi_version",
                "tiling.transport_context.struct_size",
                "tiling.transport_context.capabilities",
                "tiling.transport_context.channel_table",
                "tiling.transport_context.peer_address_table",
                "tiling.transport_context.topology.abi_version",
                "tiling.transport_context.topology.struct_size",
                "tiling.transport_context.topology.world_rank",
                "tiling.transport_context.topology.world_size",
                "tiling.transport_context.topology.scale_up_rank",
                "tiling.transport_context.topology.scale_up_size",
                "tiling.transport_context.topology.scale_out_rank",
                "tiling.transport_context.topology.scale_out_size",
                "tiling.transport_context.topology.scale_up_direct",
                "tiling.transport_context.topology.kind",
                "tiling.transport_context.topology.epoch",
                "tiling.transport_context.backend_context",
            },
        }
        for function_name, expected in expected_transport_arguments.items():
            call = calls_by_function[function_name]
            observed = set(re.findall(
                r"tiling\.transport_context(?:\.topology)?\.\w+", call))
            self.assertEqual(observed, expected, function_name)
        for source_name in ("barrier.asc", "dispatch.asc", "combine.asc"):
            source = sources[source_name]
            for assignment in (
                    "context.capabilities = transport_capabilities;",
                    "context.channel_table = transport_channel_table;",
                    "context.peer_address_table = transport_peer_address_table;",
                    "context.topology.epoch = transport_topology_epoch;"):
                self.assertEqual(source.count(assignment), 2,
                                 f"{source_name}: {assignment}")
        self.assertIn(
            "ElementKind element_kind",
            signatures["core_operator_compile_probe_vf"])
        expected_combine_producer_parameters = [
            "__gm__ const bfloat16_t* x",
            "__gm__ const float* topk_weights",
            "__gm__ const std::int32_t* source_metadata",
            "__gm__ const HybridRouteRecord* route_records",
            "std::uint64_t route_record_count",
            "__gm__ const std::int32_t* prefix_per_rank",
            "__gm__ std::uint8_t* workspace",
            "std::uint32_t transport_abi_version",
            "std::uint32_t transport_struct_size",
            "std::uint64_t transport_capabilities",
            "std::uintptr_t transport_local_window_base",
            "std::uintptr_t transport_channel_table",
            "std::uintptr_t transport_peer_address_table",
            "std::uint32_t transport_topology_abi_version",
            "std::uint32_t transport_topology_struct_size",
            "int transport_world_rank",
            "int transport_world_size",
            "int transport_scale_up_rank",
            "int transport_scale_up_size",
            "int transport_scale_out_rank",
            "int transport_scale_out_size",
            "std::uint32_t transport_scale_up_direct",
            "std::uint32_t transport_topology_kind",
            "std::uint64_t transport_topology_epoch",
            "std::uintptr_t transport_backend_context",
            "CoreModeFlags mode_flags",
            "std::uint64_t generation",
            "std::uint64_t timeout_cycles",
            "std::uint64_t num_source_rows",
            "std::uint64_t num_input_rows",
            "std::uint64_t num_tokens",
            "std::uint64_t num_topk",
            "std::uint64_t shard_capacity",
            "std::uint64_t dispatch_output_capacity",
            "std::uint64_t combine_record_bytes",
            "std::uint64_t combine_weight_offset",
            "std::uint64_t combine_control_offset",
            "std::uint64_t combine_control_bytes",
            "std::uint64_t combine_receive_offset",
            "std::uint64_t combine_receive_shard_bytes",
            "std::uint64_t combine_receive_shard_count",
            "std::uint64_t combine_receive_bytes",
            "std::uint64_t combine_staging_offset",
            "std::uint64_t combine_staging_shard_bytes",
            "std::uint64_t combine_staging_shard_count",
            "std::uint64_t combine_staging_bytes",
            "std::uint64_t reverse_forward_control_offset",
            "std::uint64_t reverse_forward_control_bytes",
            "std::uint64_t reverse_forward_shard_offset",
            "std::uint64_t reverse_forward_shard_bytes",
            "std::uint64_t reverse_forward_shard_count",
            "std::uint64_t reverse_forward_bytes",
            "std::uint64_t workspace_scratch_status_offset",
            "std::uint64_t workspace_scratch_rank_counts_offset",
            "std::uint64_t workspace_scratch_rank_values_offset",
            "std::uint64_t workspace_scratch_rank_count",
            "std::uint64_t hidden_elements",
        ]
        self.assertEqual(
            [" ".join(argument.split()) for argument in
             signature_arguments["combine_producer_vf"]],
            expected_combine_producer_parameters)
        expected_combine_producer_call = [
            "dim3(tiling.launch.num_threads)",
            "reinterpret_cast<__gm__ const bfloat16_t*>(x)",
            "topk_weights",
            "source_metadata",
            "route_records",
            "route_record_count",
            "prefix_per_rank",
            "workspace",
            "tiling.transport_context.abi_version",
            "tiling.transport_context.struct_size",
            "tiling.transport_context.capabilities",
            "local_window_base",
            "tiling.transport_context.channel_table",
            "tiling.transport_context.peer_address_table",
            "tiling.transport_context.topology.abi_version",
            "tiling.transport_context.topology.struct_size",
            "tiling.transport_context.topology.world_rank",
            "tiling.transport_context.topology.world_size",
            "tiling.transport_context.topology.scale_up_rank",
            "tiling.transport_context.topology.scale_up_size",
            "tiling.transport_context.topology.scale_out_rank",
            "tiling.transport_context.topology.scale_out_size",
            "static_cast<std::uint32_t>( tiling.transport_context.topology.scale_up_direct)",
            "static_cast<std::uint32_t>(tiling.transport_context.topology.kind)",
            "tiling.transport_context.topology.epoch",
            "tiling.transport_context.backend_context",
            "tiling.mode_flags",
            "generation",
            "timeout_cycles",
            "num_source_rows",
            "num_input_rows",
            "tiling.num_tokens",
            "tiling.num_topk",
            "tiling.num_max_tokens_per_rank",
            "tiling.dispatch_output_capacity",
            "tiling.symmetric_window_layout.combine_record_bytes",
            "tiling.symmetric_window_layout.combine_weight_offset",
            "tiling.symmetric_window_layout.combine_control_offset",
            "tiling.symmetric_window_layout.combine_control_bytes",
            "tiling.symmetric_window_layout.combine_receive_offset",
            "tiling.symmetric_window_layout.combine_receive_shard_bytes",
            "tiling.symmetric_window_layout.combine_receive_shard_count",
            "tiling.symmetric_window_layout.combine_receive_bytes",
            "tiling.symmetric_window_layout.combine_staging_offset",
            "tiling.symmetric_window_layout.combine_staging_shard_bytes",
            "tiling.symmetric_window_layout.combine_staging_shard_count",
            "tiling.symmetric_window_layout.combine_staging_bytes",
            "tiling.symmetric_window_layout .hybrid_combine_reverse_forward_control_offset",
            "tiling.symmetric_window_layout .hybrid_combine_reverse_forward_control_bytes",
            "tiling.symmetric_window_layout .hybrid_combine_reverse_forward_shard_offset",
            "tiling.symmetric_window_layout .hybrid_combine_reverse_forward_shard_bytes",
            "tiling.symmetric_window_layout .hybrid_combine_reverse_forward_shard_count",
            "tiling.symmetric_window_layout .hybrid_combine_reverse_forward_bytes",
            "tiling.workspace_layout.scratch_status_offset",
            "tiling.workspace_layout.scratch_rank_counts_offset",
            "tiling.workspace_layout.scratch_rank_values_offset",
            "tiling.workspace_layout.scratch_rank_count",
            "tiling.hidden",
        ]
        self.assertEqual(
            [" ".join(argument.split()) for argument in
             split_arguments(calls_by_function["combine_producer_vf"])],
            expected_combine_producer_call)
        producer_begin = sources["combine.asc"].index(
            "__simt_vf__ inline void combine_producer_vf")
        producer_end = sources["combine.asc"].index(
            "__simt_vf__ inline void combine_epilogue_vf", producer_begin)
        producer = sources["combine.asc"][producer_begin:producer_end]
        fill_calls = re.findall(
            r"combine_fill_normal_record_routing_weights\s*"
            r"\((.*?)\);", producer, flags=re.DOTALL)
        self.assertEqual(len(fill_calls), 1, producer)
        self.assertEqual(
            [" ".join(argument.split()) for argument in
             split_arguments(fill_calls[0])],
            ["topk_weights", "row", "master_lane", "num_topk",
             "record", "combine_weight_offset"])
        fill_marker = "combine_fill_normal_record_routing_weights("
        fill_position = producer.index(fill_marker)
        normal_branch_position = producer.rfind(
            "if (!expanded)", 0, fill_position)
        self.assertNotEqual(normal_branch_position, -1, producer)
        normal_branch, normal_branch_end = braced_block_at(
            producer, normal_branch_position)
        self.assertIn(fill_marker, normal_branch)
        else_match = re.match(r"\s*else\s*", producer[normal_branch_end:])
        self.assertIsNotNone(else_match, producer)
        expanded_branch, expanded_branch_end = braced_block_at(
            producer, normal_branch_end + else_match.end())
        self.assertNotIn(fill_marker, expanded_branch)
        self.assertRegex(
            expanded_branch,
            r"const std::int32_t input_row = metadata\[2 \+ lane\];\s*"
            r"record_weights\[lane\] = topk_weights == nullptr \?\s*"
            r"0\.0F : combine_routing_weight\(\s*topk_weights,\s*"
            r"static_cast<std::uint64_t>\(input_row\)\);")

        record_position = producer.index("__gm__ std::uint8_t* record =")
        zero_weights = re.search(
            r"for\s*\(std::uint64_t lane = 0; lane < num_topk; \+\+lane\)"
            r"\s*record_weights\[lane\] = 0\.0F;",
            producer[record_position:normal_branch_position])
        self.assertIsNotNone(zero_weights, producer)
        zero_weights_position = record_position + zero_weights.start()
        header_position = producer.index(
            "auto* header =", expanded_branch_end)
        self.assertLess(
            record_position, zero_weights_position)
        self.assertLess(
            zero_weights_position, normal_branch_position)
        self.assertLess(normal_branch_position, fill_position)
        self.assertLess(fill_position, header_position)
        per_record_depth = brace_depth_at(producer, record_position)
        self.assertEqual(
            brace_depth_at(producer, zero_weights_position),
            per_record_depth)
        self.assertEqual(
            brace_depth_at(producer, normal_branch_position),
            per_record_depth)
        self.assertEqual(
            brace_depth_at(producer, header_position),
            per_record_depth)
        self.assertNotIn(
            "combine_normal_record_routing_weight(", producer)
        self.assertNotIn("combined_topk_indices", producer)
        self.assertIn("record + combine_weight_offset", producer)
        record_source_begin = sources["combine.asc"].index(
            "struct CombineOriginDeviceRecordSource")
        record_source_end = sources["combine.asc"].index(
            "__simt_vf__ inline void combine_epilogue_vf",
            record_source_begin)
        record_source = sources["combine.asc"][
            record_source_begin:record_source_end]
        self.assertIn("record + weight_offset", record_source)
        self.assertRegex(
            sources["combine.asc"][record_source_end:],
            re.compile(
                r"const CombineOriginDeviceRecordSource origin_records\s*\{"
                r".*?combine_weight_offset,\s*hidden_elements,",
                re.DOTALL))
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

    def test_hybrid_buffer_slots_have_single_writer_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            variants = (
                ("production", [], True),
                ("dual_owned_source",
                 ["-DDEEP_EP_TEST_MUTATE_INGRESS_SOURCE=1"], False),
                ("stale_cached_payload",
                 ["-DDEEP_EP_TEST_MUTATE_CACHED_STALE=1"], False),
            )
            for name, definitions, should_pass in variants:
                with self.subTest(name=name):
                    binary = pathlib.Path(directory) / name
                    compile_result = subprocess.run(
                        ["c++", "-std=c++17", "-Wall", "-Wextra",
                         "-Werror", *definitions, f"-I{ROOT}",
                         str(HYBRID_BUFFER_OWNERSHIP_PROBE), "-o",
                         str(binary)], capture_output=True, text=True,
                        check=False)
                    self.assertEqual(compile_result.returncode, 0,
                                     compile_result.stderr)
                    run_result = subprocess.run(
                        [str(binary)], capture_output=True, text=True,
                        check=False)
                    if should_pass:
                        self.assertEqual(run_result.returncode, 0,
                                         run_result.stderr)
                    else:
                        self.assertNotEqual(
                            run_result.returncode, 0,
                            f"mutation {name} escaped behavioral coverage")

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

        source = (ELASTIC / "combine.asc").read_text()
        producer_begin = source.index(
            "__simt_vf__ inline void combine_producer_vf")
        producer_end = source.index(
            "__simt_vf__ inline void combine_epilogue_vf", producer_begin)
        producer = source[producer_begin:producer_end]
        identity_check = producer.index(
            "is_valid_combine_source_identity(")
        normal_weight_fill = producer.index(
            "combine_fill_normal_record_routing_weights(")
        self.assertLess(identity_check, normal_weight_fill)
        self.assertNotIn("combined_topk_indices", producer)
        self.assertRegex(
            producer,
            r"is_valid_combine_source_identity\(\s*"
            r"metadata\[0\],\s*destination_rank,\s*"
            r"transport_world_rank,\s*shard_capacity,\s*num_tokens\)")
        host = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        self.assertRegex(
            host,
            r"is_valid_combine_source_identity\(\s*"
            r"metadata\[0\],\s*destination_rank,\s*rank_idx_,\s*"
            r"capacity,\s*descriptor\.num_tokens\)")
        epilogue = source[producer_end:]
        self.assertRegex(
            epilogue,
            r"observed_header\s*=\s*"
            r"load_observed_combine_record_header\(header\)")
        self.assertRegex(
            epilogue,
            r"is_valid_combine_origin_token\(\s*"
            r"observed_header\.origin_token,\s*num_tokens,\s*"
            r"shard_capacity\)")
        self.assertNotIn("header->", epilogue)
        self.assertNotIn("prior_header->", epilogue)
        self.assertIn("is_valid_combine_record_lanes(", epilogue)

    def test_hybrid_route_stage_helper_is_device_callable(self):
        """Separates AICore route checks from same-TU host validators."""
        source = (ELASTIC / "dispatch_state.hpp").read_text()
        self.assertRegex(
            source,
            r"DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE\s+constexpr\s+bool\s+"
            r"is_complete_hybrid_route_stage_flags\s*\(")
        descriptor_validator = source[
            source.index("inline bool is_valid_dispatch_handle_descriptor"):
            source.index("inline DispatchHandleStatus validate_dispatch_handle")]
        table_validator = source[
            source.index("inline DispatchHandleStatus validate_hybrid_route_table"):
            source.index("}  // namespace deep_ep::ascend::elastic")]
        self.assertNotIn(
            "is_complete_hybrid_route_stage_flags(", descriptor_validator)
        self.assertNotIn(
            "is_complete_hybrid_route_stage_flags(", table_validator)
        combine = (ELASTIC / "combine.asc").read_text()
        producer = combine[
            combine.index("__simt_vf__ inline void combine_producer_vf"):
            combine.index("__simt_vf__ inline void hybrid_combine_return_vf")]
        self.assertIn("is_complete_hybrid_route_stage_flags(", producer)

    def test_scale_factor_offset_helper_is_device_callable(self):
        """Keeps strided scale-factor addressing callable from AICore code."""
        source = (ELASTIC / "dispatch_state.hpp").read_text()
        self.assertRegex(
            source,
            r"DEEP_EP_ASCEND_DISPATCH_STATE_SIMT_CALLEE\s+constexpr\s+"
            r"std::uint64_t\s+scale_factor_byte_offset\s*\(")

    def test_combine_record_trailer_geometry_is_shared_by_all_stages(self):
        def block_end(source, statement_start):
            opening = source.index("{", statement_start)
            depth = 0
            for index in range(opening, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return index + 1
            raise AssertionError("unterminated route metadata guard")

        source = (ELASTIC / "combine.asc").read_text()
        producer = source[
            source.index("__simt_vf__ inline void combine_producer_vf"):
            source.index("__simt_vf__ inline void hybrid_combine_return_vf")]
        reverse_return = source[
            source.index("__simt_vf__ inline void hybrid_combine_return_vf"):
            source.index(
                "__simt_vf__ inline void hybrid_combine_prepare_epilogue_vf")]
        prepare = source[
            source.index(
                "__simt_vf__ inline void hybrid_combine_prepare_epilogue_vf"):
            source.index("struct CombineOriginDeviceRecordSource")]
        epilogue = source[
            source.index("__simt_vf__ inline void combine_epilogue_vf"):
            source.index("__global__ __vector__ void combine_kernel")]

        self.assertIn(
            "combine_record_trailer_layout(combine_record_bytes, hybrid)",
            producer)
        self.assertIn("record + record_trailer.header_offset", producer)
        metadata_guard = producer.index(
            "if (record_trailer.has_route_metadata)")
        metadata_write = producer.index(
            "record + record_trailer.route_metadata_offset")
        self.assertLess(metadata_guard, metadata_write)
        self.assertLess(metadata_write, block_end(producer, metadata_guard))

        self.assertIn(
            "combine_record_trailer_layout(combine_record_bytes, true)",
            reverse_return)
        self.assertEqual(
            reverse_return.count(
                "record_trailer.route_metadata_offset"), 3)
        self.assertNotIn("combine_record_bytes -", reverse_return)

        self.assertNotIn("CombineRecordHeader", prepare)
        self.assertNotIn("HybridCombineRouteMetadata", prepare)
        self.assertNotIn("combine_record_bytes -", prepare)

        self.assertIn(
            "combine_record_trailer_layout(combine_record_bytes, hybrid)",
            epilogue)
        self.assertIn("record_trailer.header_offset", epilogue)
        self.assertNotIn("combine_record_bytes -", epilogue)
        self.assertNotIn(
            "kAscendElasticAlignment + sizeof(CombineRecordHeader)",
            source)

    def test_rank_indexed_kernel_state_uses_workspace_views(self):
        """Catches fixed-rank SIMT arrays and two-rank device admission."""
        sources = {
            name: (ELASTIC / name).read_text()
            for name in ("dispatch.asc", "combine.asc")
        }
        fixed_rank_array = re.compile(r"\[\s*(?:2|8)\s*\]")
        for name, source in sources.items():
            self.assertIsNone(fixed_rank_array.search(source), name)
            self.assertNotRegex(source, r"(?:world|scale_up)_size\s*!=\s*2")
            for argument in (
                    "workspace_scratch_status_offset",
                    "workspace_scratch_rank_counts_offset",
                    "workspace_scratch_rank_values_offset",
                    "workspace_scratch_rank_count"):
                self.assertIn(argument, source, f"{name}: {argument}")

        dispatch = sources["dispatch.asc"]
        for argument in ("workspace_scratch_local_count_offset",
                         "workspace_scratch_rank_indices_offset",
                         "workspace_scratch_rank_flags_offset"):
            self.assertIn(argument, dispatch, argument)

        combine = sources["combine.asc"]
        self.assertIn(
            "__gm__ const std::uint64_t* contributor_counts", combine)
        self.assertIn("std::uint64_t contributor_count", combine)
        self.assertNotIn("{contributor_counts[0], contributor_counts[1]}",
                         combine)

    def test_production_combine_semantics(self):
        self._run_production_combine_semantics_probe()

    def test_combine_owner_validates_all_headers_before_payload_access(self):
        def block_end(source, statement_start):
            opening = source.index("{", statement_start)
            depth = 0
            for index in range(opening, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return index + 1
            raise AssertionError("unterminated owner validation loop")

        source = (ELASTIC / "combine.asc").read_text()
        epilogue_begin = source.index(
            "__simt_vf__ inline void combine_epilogue_vf")
        epilogue_end = source.index(
            "__global__ __vector__ void combine_kernel", epilogue_begin)
        epilogue = source[epilogue_begin:epilogue_end]
        validation_anchor = epilogue.index(
            "const std::uint64_t num_local_experts")
        validation_begin = epilogue.index(
            "for (int contributor_rank", validation_anchor)
        validation_end = block_end(epilogue, validation_begin)
        record_source = epilogue.index(
            "const CombineOriginDeviceRecordSource origin_records")
        first_reduction = epilogue.index(
            "combine_reduce_origin_records(", record_source)
        first_output_write = epilogue.index("combined_x[")

        self.assertLess(validation_end, record_source)
        self.assertLess(validation_end, first_reduction)
        self.assertLess(validation_end, first_output_write)

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

    def test_hybrid_route_binding_validator_is_host_domain_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "dispatch_state_host_domain_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-fno-inline", f"-I{ROOT}",
                 str(DISPATCH_STATE_HOST_DOMAIN_PROBE), "-o", str(binary)],
                capture_output=True, text=True, check=False)
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

    def test_buffer_operation_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_operation_coordinator"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-pthread", f"-I{ROOT}",
                 str(PRODUCTION_OPERATION_COORDINATOR_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_barrier_buffer_lifecycle_resource_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            (directory / "pybind11").mkdir()
            (directory / "torch").mkdir()
            (directory / "pybind11/pybind11.h").write_text(PYBIND11_HEADER)
            (directory / "torch/python.h").write_text(TORCH_HEADER)
            binary = directory / "production_buffer_lifecycle_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-pthread", "-DDEEP_EP_ASCEND_TESTING=1",
                 "-DDEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR=1",
                 *HOST_SANITIZER_FLAGS,
                 f"-I{directory}", f"-I{ROOT}", "-include",
                 str(directory / "torch/python.h"),
                 str(PRODUCTION_BUFFER_LIFECYCLE_PROBE),
                 str(ELASTIC / "runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/elastic/async_state.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"),
                 str(ROOT / "csrc/backends/ascend/transport/cann_transport.cpp"),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_runtime_resource_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "production_lifecycle_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(PRODUCTION_RUNTIME_LIFECYCLE_PROBE),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"),
                 str(ROOT / "csrc/backends/ascend/transport/cann_transport.cpp"),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_runtime_resource_lifecycle_torch_npu_source_contract(self):
        header = (
            ROOT / "csrc/backends/ascend/runtime/cann_runtime.hpp").read_text()
        source = (
            ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp").read_text()
        runtime_api = header[
            header.index("struct CannRuntimeApi"):
            header.index("CannRuntimeApi make_cann_runtime_api")]

        self.assertNotIn("current_device", runtime_api)
        self.assertNotIn("current_stream", runtime_api)
        self.assertIn("c10_npu::NPUStream::unpack3(", source)
        self.assertIn(
            "c10_npu::NPUCachingAllocator::recordStream(\n"
            "            tensor.storage().data_ptr(), npu_stream);", source)
        self.assertNotIn("destroy_stream", header.lower())
        self.assertNotIn("destroy_stream", source.lower())
        self.assertNotIn("aclrtDestroyStream", source)

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

    def test_hybrid_release_sender_matrices_execute_on_host(self):
        """Catches using the canonical slot key as a diagonal sender."""
        probe_source = r'''
#include "csrc/backends/ascend/elastic/kernels.hpp"

using namespace deep_ep::ascend::elastic;

int main() {
    constexpr int dispatch_senders[4][4] = {
        {0, 1, 2, 1},
        {0, 1, 0, 3},
        {0, 3, 2, 3},
        {2, 1, 2, 3},
    };
    constexpr int combine_senders[4][4] = {
        {0, 1, 2, 2},
        {0, 1, 3, 3},
        {0, 0, 2, 3},
        {1, 1, 2, 3},
    };
    for (int receiver = 0; receiver < 4; ++receiver) {
        for (int origin = 0; origin < 4; ++origin) {
            const auto route = classify_world_route(origin, receiver, 2);
            if (final_release_sender_world_rank(route, origin) !=
                    dispatch_senders[receiver][origin])
                return 1;
        }
        for (int contributor = 0; contributor < 4; ++contributor) {
            const auto route = classify_world_route(receiver, contributor, 2);
            if (final_release_sender_world_rank(route, contributor) !=
                    combine_senders[receiver][contributor])
                return 2;
        }
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            source = directory / "hybrid_release_sender_probe.cpp"
            binary = directory / "hybrid_release_sender_probe"
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

    def test_hybrid_prepare_and_epilogues_use_release_boundaries(self):
        """Catches misplaced ingress acquire and raw canonical publication."""
        dispatch_source = (ELASTIC / "dispatch.asc").read_text()
        dispatch_prepare = dispatch_source[
            dispatch_source.index(
                "__simt_vf__ inline void hybrid_dispatch_prepare_epilogue_vf"):
            dispatch_source.index(
                "__simt_vf__ inline void hybrid_dispatch_record_routes_vf")]
        self.assertIn("dispatch_prepare_release_boundary(", dispatch_prepare)
        self.assertLess(
            dispatch_prepare.index(
                "release_protocol::observe_release_control("),
            dispatch_prepare.index("auto* source"))
        self.assertLess(
            dispatch_prepare.index("destination[byte] = source[byte]"),
            dispatch_prepare.index("release_protocol::publish_local_control("))
        self.assertNotIn(
            "direct_control[origin_rank].count =", dispatch_prepare)
        self.assertNotIn(
            "direct_control[origin_rank].generation =", dispatch_prepare)
        dispatch_epilogue = dispatch_source[
            dispatch_source.index("__simt_vf__ inline void dispatch_epilogue_vf"):
            dispatch_source.index(
                'extern "C" int deep_ep_ascend_launch_dispatch')]
        self.assertIn("dispatch_release_boundary(", dispatch_epilogue)
        self.assertIn(
            "release_protocol::observe_release_control(", dispatch_epilogue)
        self.assertNotIn(
            "release_protocol::acquire_release(", dispatch_epilogue)

        combine_source = (ELASTIC / "combine.asc").read_text()
        combine_prepare = combine_source[
            combine_source.index(
                "__simt_vf__ inline void hybrid_combine_prepare_epilogue_vf"):
            combine_source.index("struct CombineOriginDeviceRecordSource")]
        self.assertIn("combine_prepare_release_boundary(", combine_prepare)
        self.assertLess(
            combine_prepare.index(
                "release_protocol::observe_release_control("),
            combine_prepare.index("auto* source"))
        self.assertLess(
            combine_prepare.index("destination[byte] = source[byte]"),
            combine_prepare.index("release_protocol::publish_local_control("))
        self.assertNotIn(
            "direct_control[contributor_rank].count =", combine_prepare)
        self.assertNotIn(
            "direct_control[contributor_rank].generation =", combine_prepare)
        combine_epilogue = combine_source[
            combine_source.index("__simt_vf__ inline void combine_epilogue_vf"):
            combine_source.index('extern "C" int deep_ep_ascend_launch_combine')]
        self.assertIn("combine_release_boundary(", combine_epilogue)
        self.assertIn(
            "release_protocol::observe_release_control(", combine_epilogue)
        self.assertNotIn(
            "release_protocol::acquire_release(", combine_epilogue)

    def test_hybrid_dispatch_ingress_is_receive_owned(self):
        """Catches local scratch writes racing remote ingress publication."""
        source = (ELASTIC / "dispatch.asc").read_text()
        producer = source[
            source.index("__simt_vf__ inline void dispatch_producer_vf"):
            source.index(
                "DEEP_EP_ASCEND_SIMT_CALLEE transport::DeviceTransportContext")]
        self.assertIn(
            "workspace_scratch_outbound_ingress_counts_offset", producer)
        self.assertIn(
            "workspace_scratch_outbound_ingress_count", producer)
        self.assertIn("outbound_ingress_counts", producer)
        self.assertNotRegex(
            producer,
            r"ingress_control_slots\s*\[[^]]+\]\s*\.count\s*(?:=|\+\+)")
        self.assertIn("outbound_ingress_counts[ingress_rank]", producer)
        self.assertIn(
            "const std::uint64_t count =\n"
            "                outbound_ingress_counts[ingress_rank]",
            producer)
        self.assertIn("hybrid_dispatch_ingress_staging_offset", producer)
        self.assertIn("hybrid_dispatch_ingress_staging_shard_bytes", producer)
        self.assertIn("hybrid_dispatch_ingress_staging_shard_count", producer)
        self.assertIn("hybrid_dispatch_ingress_staging_bytes", producer)
        local_write = producer[producer.index(
            "} else if (diagonal) {"):producer.index(
                "} else if (destination_rank == transport_world_rank) {")]
        self.assertIn("hybrid_dispatch_ingress_staging_offset", local_write)
        self.assertNotIn("hybrid_dispatch_ingress_shard_offset", local_write)
        stage1 = producer[producer.index(
            "release_protocol::put_staged_payload("):producer.index(
                "release_protocol::flush_payload(transport);",
                producer.index("release_protocol::put_staged_payload("))]
        self.assertIn("hybrid_dispatch_ingress_shard_offset", stage1)
        self.assertIn("hybrid_dispatch_ingress_staging_offset", stage1)
        self.assertLess(stage1.index("hybrid_dispatch_ingress_shard_offset"),
                        stage1.index("hybrid_dispatch_ingress_staging_offset"))

    def test_hybrid_dispatch_republishes_masked_producer_failure(self):
        """Catches a second service reset followed by a silent scratch return."""
        source = (ELASTIC / "dispatch.asc").read_text()
        forward = source[
            source.index("__simt_vf__ inline void hybrid_dispatch_forward_vf"):
            source.index(
                "__simt_vf__ inline void hybrid_dispatch_prepare_epilogue_vf")]
        self.assertIn("decode_dispatch_protocol_scratch(", forward)
        self.assertIn("DispatchProtocolStage::kProducer", forward)
        self.assertIn("record_dispatch_protocol_error(", forward)
        self.assertIn(
            "const bool valid_producer_failure =\n"
            "            producer_failure.valid &&\n"
            "            producer_failure.world_rank >= 0 &&\n"
            "            producer_failure.world_rank < transport_world_size;",
            forward)
        self.assertEqual(forward.count("valid_producer_failure ?"), 2)
        self.assertNotIn("producer_failure.valid ?", forward)
        self.assertNotIn(
            "if (transport.load_acquire(status_address) != 0)\n"
            "        return;", forward)

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
            "dispatch.asc": (
                "DeviceTransportFacade", "put(",
                "release_protocol::publish_control_and_release(",
                "release_protocol::observe_release_control(",
                "device_barrier("),
            "combine.asc": (
                "DeviceTransportFacade", "put(",
                "release_protocol::publish_control_and_release(",
                "release_protocol::observe_release_control(",
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
        release = (ELASTIC / "release_protocol.hpp").read_text()
        for operation in (
                "facade.flush(", "facade.put_value(", "facade.signal(",
                "facade.wait_signal(", "facade.read_signal("):
            self.assertIn(operation, release)

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
                "raw_record_capacity", "dispatch_output_capacity",
                "num_input_rows > dispatch_output_capacity",
                "tiling.dispatch_output_capacity",
                "count * combine_record_bytes",
                "header->origin_token", "header->contributor_rank",
                "is_valid_combine_record_lanes(",
                "record_count", "remote_count, count",
                "release_protocol::flush_payload(",
                "release_protocol::publish_control_and_release(",
                "release_protocol::observe_release_control(",
                "kCombineReleaseSignalIndex",
                "transport.device_barrier(",
                "transport.consumed_generation()",
                "CombineOriginDeviceRecordSource",
                "combine_reduce_origin_records(",
                "static_cast<bfloat16_t>(value)"):
            self.assertIn(marker, source)

        self.assertNotRegex(
            source,
            r"transport\.(?:put|put_value|signal)\(\s*"
            r"transport::TransportTeam::k(?:World|ScaleUp),\s*0,")
        for forbidden in (
                "transport.remote_add_release(",
                "transport.get_symmetric_pointer("):
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
        self.assertIn("core_topology_from_transport(", combine_tiling)

        normal_begin = runner.index(
            'else if (std::strcmp(argv[2], "combine-normal") == 0)')
        normal_end = runner.index("else if", normal_begin + 1)
        normal_case = runner[normal_begin:normal_end]
        self.assertIn(
            "run_combine_cli_case(stream, false, false, 0, false)", normal_case)

    def test_combine_runner_uses_real_context_and_defers_standalone_cli(self):
        """Catches rank-0 tiling and context-free remote launches."""
        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        tiling_begin = runner.index("CoreTiling make_combine_tiling")
        tiling_end = runner.index("bool run_combine_case", tiling_begin)
        combine_tiling = runner[tiling_begin:tiling_end]
        topology_marker = "input.topology = core_topology_from_transport("
        self.assertIn(topology_marker, combine_tiling)
        topology_position = combine_tiling.index(topology_marker)
        build_position = combine_tiling.index("build_core_tiling(input")
        self.assertLess(topology_position, build_position)
        self.assertIn("tiling.transport_context = transport_context",
                      combine_tiling)

        combine_begin = runner.index("bool run_combine_case")
        cli_wrapper_begin = runner.index("bool run_combine_cli_case")
        combine_case = runner[combine_begin:cli_wrapper_begin]
        self.assertIn(
            "const deep_ep::ascend::transport::DeviceTransportContext& "
            "transport_context",
            combine_case)
        self.assertIn("launch_internal_combine(", combine_case)

        wrapper_end = runner.index("bool run_barrier_local", cli_wrapper_begin)
        cli_wrapper = runner[cli_wrapper_begin:wrapper_end]
        self.assertIn("transport_context != nullptr", cli_wrapper)
        self.assertIn("run_combine_case(", cli_wrapper)
        self.assertNotIn("run_combine_state_probe(stream)", cli_wrapper)
        self.assertIn("hardware-deferred", cli_wrapper)
        self.assertNotIn("make_device_transport_context", cli_wrapper)

        main = runner[runner.index("int main("):]
        self.assertNotIn("run_combine_case(stream", main)
        self.assertEqual(main.count("run_combine_cli_case(stream"), 7)

    def test_combine_device_capacity_gate_and_runner_allocation(self):
        """Catches device output overflow and undersized shared buffers."""
        source = (ELASTIC / "combine.asc").read_text()
        epilogue_begin = source.index(
            "__simt_vf__ inline void combine_epilogue_vf")
        epilogue = source[epilogue_begin:]
        self.assertIn("is_valid_combine_token_extent(", epilogue)
        capacity_position = epilogue.index(
            "is_valid_combine_token_extent(")
        output_position = epilogue.index(
            "for (std::uint64_t token = 0; token < num_tokens; ++token)")
        weight_position = epilogue.index("combined_topk_weights != nullptr")
        completion_position = epilogue.index("&control->combine_generation")
        self.assertLess(capacity_position, output_position)
        self.assertLess(capacity_position, weight_position)
        self.assertLess(capacity_position, completion_position)
        self.assertIn("CombineProtocolError::kCapacityOverflow", epilogue)

        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for marker in (
                "is_valid_combine_token_extent(",
                "merge_core_launch_storage(",
                "allocation_storage.communication_buffer_bytes",
                "allocation_storage.workspace_bytes",
                "prefix_rank_count"):
            self.assertIn(marker, runner)
        allocation_begin = runner.index("bool allocate_dispatch_buffers")
        allocation_end = runner.index(
            "CoreTiling make_dispatch_tiling", allocation_begin)
        allocation = runner[allocation_begin:allocation_end]
        self.assertIn(
            "allocate(&buffers->prefix_rank, prefix_rank_count)", allocation)
        self.assertNotIn("allocate(&buffers->prefix_rank, 1)", allocation)

        combine_begin = runner.index("bool run_combine_case")
        combine_end = runner.index("bool run_barrier_local", combine_begin)
        combine_case = runner[combine_begin:combine_end]
        self.assertIn("allocation_storage", combine_case)
        self.assertIn(
            "static_cast<std::uint64_t>(combine_tiling.topology.world_size)",
            combine_case)
        self.assertIn(
            "const deep_ep::ascend::transport::DeviceTransportContext& "
            "transport_context",
            combine_case)
        self.assertIn(
            "combine_tiling.transport_context.local_window_base",
            combine_case)
        self.assertIn("arguments.communication_buffer = combine_communication",
                      combine_case)
        self.assertIn("observed_output[2] == 0x5a5a", runner)

    def test_combine_expanded_semantics(self):
        """Executes expanded lane planning and padding-safe reduction."""
        self._run_production_combine_semantics_probe()

    def test_combine_weights_and_bias_semantics(self):
        """Executes routing-weight identity and post-reduction bias checks."""
        self._run_production_combine_semantics_probe()

    def test_dispatch_preflights_protocol_state_before_publication(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        release = (ELASTIC / "release_protocol.hpp").read_text()
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
                "tiling.workspace_layout.scratch_status_offset"):
            self.assertIn(marker, source)
        self.assertIn("make_dispatch_protocol_failure", source)
        self.assertIn("failure.backend_status, failure.generation", source)
        failure_build = source.index("make_dispatch_protocol_failure")
        diagnostic_write = source.index(
            "failure.backend_status, failure.generation")
        self.assertLess(failure_build, diagnostic_write)
        self.assertNotIn("RemoteAction::signal_add", source + release)
        self.assertNotIn("RemoteAction::signal_increment", source + release)
        self.assertIn(
            "RemoteAction::signal_set(signal_index, generation)", release)
        flush_position = source.index("release_protocol::flush_payload(")
        publish_position = source.index(
            "release_protocol::publish_control_and_release(")
        barrier_position = source.index("transport.device_barrier(")
        self.assertLess(flush_position, publish_position)
        self.assertLess(publish_position, barrier_position)

        publish_begin = release.index("void publish_control_and_release(")
        publish_end = release.index("template <typename Transport>",
                                    publish_begin)
        publish = release[publish_begin:publish_end]
        count_position = publish.index("count_address, count")
        generation_position = publish.index(
            "generation_address, generation")
        signal_position = publish.index("facade.signal(")
        self.assertLess(count_position, generation_position)
        self.assertLess(generation_position, signal_position)

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
        self.assertIn("launch_internal_combine", production)
        for marker in ("read_diagnostic", "copy_to_host",
                       "barrier_completion", "BufferOperationCoordinator",
                       "timeout_cycles_from_seconds"):
            self.assertIn(marker, production)

    def test_public_runtime_rank_parameterization_source_contract(self):
        production = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        runtime = (ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp").read_text()
        transport = (ROOT / "csrc/backends/ascend/transport/cann_transport.cpp").read_text()
        self.assertGreaterEqual(production.count("num_experts % num_ranks_"), 2)
        self.assertIn(
            "capacity * static_cast<std::uint64_t>(num_ranks_)", production)
        self.assertIn("query_cann_communicator_size(comm_handle", production)
        self.assertNotIn("num_ranks == 2", production)
        self.assertNotIn("input.world_size = 2", production)
        self.assertNotIn("config.world_size != 2", runtime)
        self.assertIn("config.world_size < 2", runtime)
        self.assertIn("query_cann_communicator_size", transport)
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
                       "diagnostic.generation = generation",
                       "lease.complete()"):
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
                "BufferOperationCoordinator", "DispatchHandleDescriptor",
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
                 "-DDEEP_EP_ASCEND_TESTING=1",
                 "-DDEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR=1",
                 *HOST_SANITIZER_FLAGS,
                 f"-I{directory}", f"-I{ROOT}", "-include",
                 str(directory / "torch/python.h"), str(probe),
                 str(ELASTIC / "runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/elastic/async_state.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"),
                 str(ROOT / "csrc/backends/ascend/transport/cann_transport.cpp"),
                 "-o", str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_public_combine_async_probe_executes(self):
        probe = ROOT / "tests/ascend/production_combine_probe.cpp"
        self.assertTrue(probe.is_file(), str(probe))
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            (directory / "pybind11").mkdir()
            (directory / "torch").mkdir()
            (directory / "pybind11/pybind11.h").write_text(PYBIND11_HEADER)
            (directory / "torch/python.h").write_text(TORCH_HEADER)
            binary = directory / "production_combine_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-DDEEP_EP_ASCEND_TESTING=1",
                 "-DDEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR=1",
                 *HOST_SANITIZER_FLAGS,
                 f"-I{directory}", f"-I{ROOT}", "-include",
                 str(directory / "torch/python.h"),
                 str(probe), str(ELASTIC / "runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/elastic/async_state.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/cann_runtime.cpp"),
                 str(ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"),
                 str(ROOT / "csrc/backends/ascend/transport/cann_transport.cpp"),
                 "-o", str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_async_overlap_harness_contract(self):
        expected_cases = [
            "capture-current-stream",
            "cached-dispatch-sync-allocate-false",
            "cached-dispatch-sync-allocate-true",
            "cached-dispatch-async-allocate-false",
            "cached-dispatch-async-allocate-true",
            "previous-event-allocate-true",
            "combine-sync-allocate-false",
            "combine-sync-allocate-true",
            "combine-async-allocate-false",
            "combine-async-allocate-true",
            "empty-route",
            "asymmetric-route",
            "100-generations",
            "two-independent-buffers",
            "record-failure",
            "event-timeout",
            "diagnostic-failure",
            "completion-mismatch",
            "drop-event",
            "destroy-pending-retry",
            "overlap-vs-serialized",
        ]
        result = subprocess.run(
            ["python3", str(ASYNC_OVERLAP), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "case_names": expected_cases,
                "case_timeout_seconds": 30,
                "contract_checks": [
                    "literal-bf16-torch-reference",
                    "rank-qualified-failure-aggregation",
                    "per-case-process-timeout",
                    "finally-buffer-before-process-group-teardown",
                    "zero-global-synchronization",
                    "npu-profiler-overlap-interval",
                ],
                "event_cases": [
                    "capture-current-stream",
                    "record-failure",
                    "event-timeout",
                    "drop-event",
                    "destroy-pending-retry",
                ],
                "full_cases": expected_cases,
                "matrix_groups": [
                    "capture-current-stream",
                    "cached-dispatch sync/async x allocation false/true",
                    "previous-event + allocation true",
                    "combine sync/async x allocation false/true",
                    "empty-route",
                    "asymmetric-route",
                    "100-generations",
                    "two-independent-buffers",
                    "record-failure",
                    "event-timeout",
                    "diagnostic-failure",
                    "completion-mismatch",
                    "drop-event",
                    "destroy-pending-retry",
                    "overlap-vs-serialized",
                ],
                "overlap": {
                    "compute_iterations": 8,
                    "compute_shape": [4096, 4096],
                    "minimum_median_improvement": 0.05,
                    "profiler_interval_required": True,
                    "repetitions": 7,
                    "report_percentiles": [50, 95],
                    "warmups": 3,
                },
                "reference": "rank-gathered-literal-inputs-and-torch-ops",
                "world_size": 2,
            })

    def test_async_overlap_helpers_enforce_timeout_rank_and_profiler_contract(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(
            module._case_command(
                "cached-dispatch-sync-allocate-false", "/tmp/traces"),
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(ASYNC_OVERLAP),
                "--worker",
                "cached-dispatch-sync-allocate-false",
                "--trace-dir",
                "/tmp/traces",
            ],
        )

        bounded = module._run_bounded(
            ["python3", "-c", "import time; time.sleep(1)"],
            timeout_seconds=0.01)
        self.assertEqual(bounded["status"], "failed")
        self.assertEqual(bounded["failure"], "process timeout after 0.01s")
        self.assertEqual(
            module._aggregate_rank_failures([
                {"rank": 0, "failure": "dispatch backend 17"},
                {"rank": 1, "failure": None},
            ]),
            "rank 0: dispatch backend 17",
        )

        with tempfile.TemporaryDirectory() as directory:
            trace = pathlib.Path(directory) / "trace.json"
            trace.write_text(json.dumps({"traceEvents": [
                {"ph": "X", "cat": "NPU", "name": "matmul",
                 "ts": 100.0, "dur": 40.0,
                 "args": {"Stream Id": 7}},
                {"ph": "X", "cat": "NPU", "name": "dispatch",
                 "ts": 120.0, "dur": 50.0,
                 "args": {"Stream Id": 11}},
            ]}))
            overlap = module._find_npu_overlap_interval(
                (trace,), compute_stream_id=7, comm_stream_id=11)
        self.assertEqual(overlap["overlap_us"], 20.0)
        self.assertEqual(overlap["compute_event"], "matmul")
        self.assertEqual(overlap["communication_event"], "dispatch")

    def test_async_overlap_suite_checkpoints_each_completed_case(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_checkpoint", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        selected = module.EVENT_CASES[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            observed_checkpoint = []

            def run_bounded(_command):
                if output.exists():
                    observed_checkpoint.append(json.loads(output.read_text()))
                return {
                    "status": "passed",
                    "failure": None,
                    "duration_seconds": 0.25,
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                }

            with mock.patch.object(module, "EVENT_CASES", selected), \
                    mock.patch.object(module, "_run_bounded", run_bounded), \
                    contextlib.redirect_stdout(io.StringIO()):
                exit_code = module._run_suite(
                    "event", output, pathlib.Path(directory) / "traces")

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(observed_checkpoint), 1)
        self.assertEqual(
            [row["case"] for row in observed_checkpoint[0]["results"]],
            [selected[0]],
        )

    def test_async_overlap_suite_stops_after_first_failure(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_fail_fast", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        selected = module.EVENT_CASES[:3]
        calls = []

        def run_bounded(_command):
            calls.append(len(calls))
            failed = len(calls) == 2
            return {
                "status": "failed" if failed else "passed",
                "failure": "process exited 17" if failed else None,
                "duration_seconds": 0.25,
                "exit_code": 17 if failed else 0,
                "stdout": "",
                "stderr": "rank 0 traceback" if failed else "",
            }

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            with mock.patch.object(module, "EVENT_CASES", selected), \
                    mock.patch.object(module, "_run_bounded", run_bounded), \
                    contextlib.redirect_stdout(io.StringIO()):
                exit_code = module._run_suite(
                    "event", output, pathlib.Path(directory) / "traces")
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["summary"], {
            "total": 2,
            "selected": 3,
            "executed": 2,
            "passed": 1,
            "failed": 1,
            "not_run": 1,
        })

    def test_async_overlap_suite_streams_failed_child_diagnostic(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_diagnostic", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        failure = {
            "status": "failed",
            "failure": "process exited 17",
            "duration_seconds": 0.25,
            "exit_code": 17,
            "stdout": "",
            "stderr": "rank 0 traceback",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            stdout = io.StringIO()
            with mock.patch.object(
                    module, "EVENT_CASES", module.EVENT_CASES[:1]), \
                    mock.patch.object(
                        module, "_run_bounded", return_value=failure), \
                    contextlib.redirect_stdout(stdout):
                module._run_suite(
                    "event", output, pathlib.Path(directory) / "traces")

        live_output, _separator, _final_report = stdout.getvalue().partition(
            "PHASE3E_SUITE_RESULT ")
        self.assertIn(
            "DIAGNOSTIC capture-current-stream:\nrank 0 traceback\n",
            live_output,
        )

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
            "round-trip-smoke",
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

    def test_two_rank_combine_harness_contract(self):
        expected_cases = [
            "normal",
            "expanded-multiple-reduction",
            "expanded-single-reduction",
            "weights",
            "zero-bias",
            "one-bias",
            "two-bias",
            "duplicate-same-rank-experts",
            "negative-one-route",
            "empty-input",
            "asymmetric-routing",
            "aligned-padding",
            "aligned-near-capacity",
            "expanded-weighted-multiple-reduction",
            "expanded-single-padded-extent",
            "odd-hidden-unweighted",
            "odd-hidden-weighted",
            "cached-dispatch-changed-outputs",
            "sequential-100-generations",
            "cross-buffer-handle",
            "interleaved-dual-buffer",
            "malformed-handle",
            "bounded-peer-diagnostics",
            "repeated-teardown",
        ]
        result = subprocess.run(
            ["python3", str(TWO_RANK_COMBINE), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "case_names": expected_cases,
                "contract_checks": [
                    "public-dispatch-combine",
                    "literal-route-reference",
                    "synthetic-origin-transform",
                    "expanded-metadata-writes",
                    "bf16-tolerance",
                    "exact-float32-weights",
                    "weighted-same-contributor-lanes",
                    "expanded-weighted-multiple-reduction",
                    "padding-expanded-input-capacity",
                    "odd-hidden-record-layout",
                    "case-boundary-barriers",
                    "distributed-failure-aggregation",
                    "buffer-before-group-teardown",
                    "reduction-mode-buffer-recreation",
                    "bounded-peer-diagnostics",
                    "public-handle-mutations",
                    "one-hccl-group",
                    "sub-operation-failure-synchronization",
                    "order-sensitive-public-case",
                    "oracle-behavior-mutations",
                    "local-npu-output-placement",
                    "best-effort-cleanup",
                    "synchronized-buffer-mode-phases",
                    "literal-expanded-layout-counts",
                    "interleaved-dual-buffer-isolation",
                ],
                "behavior_fixtures": {
                    "buffer_modes": {
                        "false_to_true": {
                            "reductions": 2,
                            "trace": ["destroy:false", "construct:true"],
                        },
                        "initial_construction": {
                            "reductions": 1,
                            "trace": ["construct:true"],
                        },
                        "same_mode_reuse": {
                            "reductions": 0,
                            "trace": [],
                        },
                        "teardown_failure": {
                            "construction_blocked": True,
                            "reductions": 1,
                            "trace": ["destroy:false"],
                        },
                        "true_to_false": {
                            "reductions": 2,
                            "trace": ["destroy:true", "construct:false"],
                        },
                    },
                    "cleanup": {
                        "calls": ["buffer-a", "buffer-b", "buffer-c",
                                  "process-group"],
                        "failures": ["buffer-a failed", "buffer-c failed",
                                     "process-group failed"],
                    },
                    "expanded": {
                        "mapped_rows": [2],
                        "padding_rows": [0, 1, 3],
                        "row_2": [1.0, 2.0, 3.0, 4.0],
                    },
                    "expanded_layouts": {
                        "expanded-single-padded-extent": {
                            "rank0": {"padding": 27, "rows": 32},
                            "rank1": {"padding": 27, "rows": 32},
                        },
                        "expanded-weighted-multiple-reduction": {
                            "rank0": {"padding": 27, "rows": 32},
                            "rank1": {"padding": 27, "rows": 32},
                        },
                        "aligned-near-capacity": {
                            "rank0": {"padding": 0, "rows": 8},
                            "rank1": {"padding": 0, "rows": 8},
                        },
                        "aligned-padding": {
                            "rank0": {"padding": 5, "rows": 8},
                            "rank1": {"padding": 6, "rows": 8},
                        },
                    },
                    "mutations_rejected": {
                        "alignment_2": True,
                        "bias_per_contributor": True,
                        "expanded_reads_sentinel": True,
                        "handle_expansion_mode": True,
                        "lane_order": True,
                        "rank_order": True,
                        "surplus_expanded_row": True,
                        "weight_multiplies_activation": True,
                    },
                    "ordering": {
                        "canonical": [0.0, 0.0, 9.0, 12.0],
                        "lane_reversed": [0.5, 0.0, 9.0, 12.0],
                        "rank_reversed": [0.0, 0.5, 9.0, 12.0],
                    },
                    "placement": {
                        "cpu": False,
                        "local_npu": True,
                        "wrong_npu": False,
                    },
                    "same_contributor_weights": {
                        "routes": [
                            [[0, 1], [2, 3]],
                            [[1, 0], [3, 2]],
                        ],
                        "weights": [
                            [[0.125, 0.25], [0.375, 0.5]],
                            [[0.625, 0.75], [0.875, 1.0]],
                        ],
                    },
                    "synchronization": {
                        "local_failure_reductions": 1,
                        "peer_failure_reductions": 1,
                        "peer_result_rejected": True,
                        "post_sync_failure_operation_blocked": True,
                        "sync_failure_reductions": 1,
                    },
                    "weight_bias": {
                        "activation_ignores_weight": [1.0, 2.0, 3.0, 4.0],
                        "bias_once": [9.0, 11.0, 13.0, 15.0],
                        "restored_weights": [0.25, 0.0],
                    },
                    "weight_mismatch": {
                        "actual": [[0.125, 0.0], [0.375, 0.0]],
                        "differences": [{
                            "actual": 0.0,
                            "expected": 0.25,
                            "index": [0, 1],
                        }],
                        "expected": [[0.125, 0.25], [0.375, 0.0]],
                        "rank": 0,
                    },
                },
                "expected_world_size": 2,
                "empty_reference_shape": [0, 4],
                "float32_order_fixture": 0.0,
                "reference": "gathered-original-routes",
                "reference_fixture": {
                    "rank0": [[4.0, 6.0, 8.0, 10.0],
                              [5.0, 6.0, 7.0, 8.0]],
                    "rank1": [[36.0, 38.0, 40.0, 42.0]],
                },
                "system_under_test": ["Buffer.dispatch", "Buffer.combine"],
            })

    def test_rank_parameterized_scale_up_smoke_contract(self):
        result = subprocess.run(
            ["python3", str(SCALE_UP_SMOKE), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "minimum_world_size": 2,
                "rank_limit": None,
                "cases": ["barrier", "bf16-all-to-all-round-trip"],
                "num_experts": "world_size",
                "num_tokens_per_rank": "world_size",
                "system_under_test": [
                    "ElasticBuffer.barrier",
                    "ElasticBuffer.dispatch",
                    "ElasticBuffer.combine",
                ],
            })

    def test_hybrid_core_runner_covers_two_stage_reverse_routes(self):
        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        qualified_context = (
            "deep_ep::ascend::transport::DeviceTransportContext")
        self.assertEqual(runner.count(qualified_context), 3)
        self.assertNotIn("const transport::DeviceTransportContext", runner)
        for marker in (
                "hybrid_route_probe_kernel<<<", "run_hybrid_route_probe(",
                "classify_world_route(", "combine_expanded_record_count("):
            self.assertIn(marker, runner)
        for case_name in (
                "hybrid-route-local", "hybrid-route-scale-up",
                "hybrid-route-scale-out", "hybrid-route-diagonal",
                "hybrid-route-empty", "hybrid-route-cached",
                "hybrid-combine-single", "hybrid-combine-multiple"):
            self.assertIn(case_name, runner)
        for marker in (
                "HybridRouteRecord", "kInvalidHybridRouteSlot",
                "TransportTeam::kScaleOut", "TransportTeam::kScaleUp"):
            self.assertIn(marker, runner)


if __name__ == "__main__":
    unittest.main()
