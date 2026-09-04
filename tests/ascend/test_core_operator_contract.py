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
import time
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
COMBINE_PARALLEL_AICORE_CALLEE_PROBE = \
    ROOT / "tests/ascend/combine_parallel_aicore_callee_probe.cpp"
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
EXPECTED_FP8_ASYNC_CASES = (
    "fp8-cached-normal-fp32-row-async-allocate-false",
    "fp8-cached-normal-packed-column-async-allocate-true",
    "fp8-cached-expanded-fp32-row-async-allocate-false",
    "fp8-cached-expanded-packed-column-async-allocate-true",
    "fp8-uncached-normal-fp32-row-async-allocate-false",
    "fp8-uncached-normal-packed-column-async-allocate-true",
    "fp8-uncached-expanded-fp32-row-async-allocate-false",
    "fp8-uncached-expanded-packed-column-async-allocate-true",
    "fp8-previous-event-allocate-true",
    "fp8-empty-route",
    "fp8-asymmetric-route",
    "fp8-10-generations",
    "fp8-cached-representation-changes",
    "fp8-completion-mismatch",
    "fp8-drop-event",
    "fp8-destroy-pending-retry",
)
STREAM_EVENT_CAPABILITY_PROBE = \
    ROOT / "tests/ascend/stream_event/capability_probe.cpp"
STREAM_EVENT_CAPABILITY_RUNNER = \
    ROOT / "tests/ascend/stream_event/run_capability_probe.py"
ASYNC_RUNTIME_PROBE = ROOT / "tests/ascend/async_runtime_probe.cpp"
HOST_TIMELINE_PROBE = ROOT / "tests/ascend/host_timeline_probe.cpp"
ELASTIC_BINDING_PROBE = ROOT / "tests/ascend/elastic_binding_probe.cpp"
ELASTIC = ROOT / "csrc/backends/ascend/elastic"
CORE_OPS = ROOT / "tests/ascend/core_ops"

BINDING_PYBIND11_HEADER = r"""
#pragma once
#include <functional>
#include <string_view>
#include <utility>

extern bool binding_probe_gil_released;

namespace pybind11 {

class gil_scoped_release {
public:
    gil_scoped_release() { binding_probe_gil_released = true; }
    ~gil_scoped_release() { binding_probe_gil_released = false; }
};

class module_ {
public:
    template <typename Function>
    void def(const char*, Function&&) {}
};

template <typename... Args>
struct init {};

template <typename Type>
class class_ {
public:
    class_(module_&, const char*) {}

    template <typename... Args>
    class_& def(init<Args...>) {
        return *this;
    }

    template <typename Function>
    class_& def(const char* name, Function&& function) {
        if (std::string_view(name) == "destroy") {
            Type instance;
            std::invoke(std::forward<Function>(function), instance);
        }
        return *this;
    }
};

}  // namespace pybind11
"""


def _component_stage_seconds():
    return {
        "communication-only": {
            "async_dispatch_return": 0.01,
            "event_wait": 0.19,
        },
        "compute-only": {
            "compute_enqueue": 0.20,
            "compute_completion": 0.10,
        },
        "serialized": {
            "serialized_dispatch_return": 0.20,
            "compute_enqueue": 0.20,
            "compute_completion": 0.10,
        },
        "overlapped": {
            "async_dispatch_return": 0.01,
            "compute_enqueue": 0.20,
            "event_wait": 0.19,
            "compute_completion": 0.10,
        },
    }


def _async_overlap_worker_stdout(case):
    measurements = {
        "capture-current-stream": {
            "repeated_waits": 2,
            "global_synchronizations": 0,
        },
        "record-failure": {
            "injected_failure": "record_event failed",
            "recovery_event_waited": True,
            "global_synchronizations": 0,
        },
        "event-timeout": {
            "injected_failure": "synchronize_event failed",
            "same_event_recovered": True,
            "recovery_event_waited": True,
            "global_synchronizations": 0,
        },
    }[case]
    return "PHASE3E_WORKER_RESULT " + json.dumps({
        "case": case,
        "measurements": measurements,
    }) + "\n"


def _valid_component_rank(rank):
    return {
        "rank": rank,
        "communication_tokens": 256,
        "compute_input_shape": [1, 64, 8, 32, 32],
        "compute_iterations": 16,
        "compute_variant": "aic-only-conv3d",
        "compute_weight_shape": [64, 64, 3, 3, 3],
        "compute_calibration": {
            "completion_wait": "npu-event-synchronize",
            "initial_iterations": 8,
            "iteration_bounds": [1, 2048],
            "measured_initial_seconds": 0.125 + rank * 0.001,
            "selected_iterations": 16,
            "target_range_seconds": [0.20, 0.30],
            "target_seconds": 0.25,
            "warmup_iterations": 1,
        },
        "buffer_instance": f"rank-{rank}-component-buffer",
        "communication_only_seconds": 0.20,
        "compute_only_seconds": 0.30,
        "serialized_seconds": 0.50,
        "overlapped_seconds": 0.45,
        "serialized_dispatch_return_duration_seconds": 0.20,
        "async_dispatch_return_duration_seconds": 0.01,
        "event_wait_duration_seconds": 0.19,
        "compute_completion_duration_seconds": 0.10,
        "stage_seconds": _component_stage_seconds(),
        "profiler_overlap": {
            "overlap_us": 100000.0 + rank,
            "compute_event": "aclnnConvolution_Conv3DV2",
            "communication_event": "dispatch_kernel",
            "compute_task_type": "KERNEL_AICORE",
            "communication_task_type": "KERNEL_AIVEC",
            "logical_compute_stream_id": 0,
            "logical_communication_stream_id": 143,
            "physical_compute_stream_id": 61,
            "physical_communication_stream_id": 11,
            "auxiliary_aiv_events": [],
            "transdata_events": [],
            "compute_path_aic_only": True,
        },
        "completed_phase": "profiler",
        "active_phase": None,
        "phase_status": "completed",
    }


class AscendCoreOperatorContractTest(unittest.TestCase):
    def test_invalid_stage_profile_exposes_raw_command_metrics(self):
        source = (
            ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        begin = source.index("    pybind11::dict get_stage_profile() const")
        end = source.index("\n    c10::Stream get_comm_stream() const", begin)
        profile = source[begin:end]
        validation = profile.index(
            "const auto command_metrics_status =")
        construction = profile.index("pybind11::dict command_metrics;")
        publication = profile.index(
            'result["command_metrics"] = command_metrics;')
        service_construction = profile.index("pybind11::dict raw_service;")
        service_publication = profile.index(
            'result["service"] = raw_service;')

        self.assertLess(construction, validation)
        self.assertLess(publication, validation)
        self.assertLess(service_construction, validation)
        self.assertLess(service_publication, validation)

    def test_transport_service_profile_avoids_long_lived_interval_state(self):
        source = (
            ROOT /
            "csrc/backends/ascend/transport/aicore_transport_service.hpp"
        ).read_text()
        begin = source.index(
            "__aicore__ inline void execute_body(")
        end = source.index(
            "\ntemplate <bool ProfileEnabled = true>\n"
            "__aicore__ inline void execute(", begin)
        execute = source[begin:end]

        self.assertIn(
            "profile->service_start_cycles =\n"
            "                record_transport_stage_start(", execute)
        command_loop = execute[
            execute.index("for (std::uint32_t index = command_begin;"):
            execute.index("        bool success = true;")
        ]
        self.assertIn("++profile->command_count;", command_loop)
        self.assertIn(
            "profile->service_end_cycles =\n"
            "                record_transport_stage_end(", execute)
        self.assertNotIn("std::uint64_t service_start_cycles = 0;", execute)
        self.assertNotIn("accumulate_transport_service_interval(", execute)

    def test_profiled_transport_service_has_a_noinline_codegen_boundary(self):
        source = (
            ROOT /
            "csrc/backends/ascend/transport/aicore_transport_service.hpp"
        ).read_text()
        self.assertIn(
            "__aicore__ static __attribute__((noinline)) void "
            "execute_profiled(", source)

        begin = source.index(
            "__aicore__ inline void execute(const DeviceTransportContext&")
        execute = source[begin:]
        self.assertIn(
            "if constexpr (ProfileEnabled)\n"
            "        execute_profiled(context);\n"
            "    else\n"
            "        execute_body<false>(context);",
            execute)

    def test_host_timeline_accumulates_literal_phase_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "host_timeline_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(HOST_TIMELINE_PROBE), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_combine_host_timeline_covers_prelaunch_and_completion_path(self):
        source = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        begin = source.index(
            "    combine(const torch::Tensor& x,")
        end = source.index("\n    }\n};", begin)
        combine = source[begin:end]

        self.assertIn("host_timeline_profile_.reset(0);", combine)
        self.assertIn("host_timeline_profile_.bind_generation(generation)", combine)
        expected_phase_counts = {
            "kCombinePrelaunchSetup": 1,
            "kCombineHostValidation": 2,
        }
        for phase in (
                "kCombineHandleToHost",
                "kCombineMetadataToHost",
                "kCombineStreamSetup",
                "kCombineOutputAllocation",
                "kCombineSubmit",
                "kCombineCompletionRecord",
                "kCombineCompletionWait"):
            expected_phase_counts[phase] = 1
        for phase, count in expected_phase_counts.items():
            self.assertEqual(
                combine.count(f"HostTimelinePhase::{phase}"), count, phase)

    def test_direct_device_hot_path_uses_32_bit_control_indices(self):
        """Keeps shape iteration narrow while addresses and offsets stay wide."""
        kernels = (ELASTIC / "kernels.hpp").read_text()
        grid_begin = kernels.index("struct DirectDataGridStride")
        grid_end = kernels.index(
            "DEEP_EP_ASCEND_KERNEL_CALLEE WorldRoute", grid_begin)
        grid = kernels[grid_begin:grid_end]
        self.assertIn("std::uint32_t first", grid)
        self.assertIn("std::uint32_t stride", grid)
        self.assertNotIn("std::uint64_t first", grid)
        self.assertNotIn("std::uint64_t stride", grid)

        tiling = (ELASTIC / "tiling.hpp").read_text()
        self.assertIn("kDirectDeviceIndexLimit", tiling)
        self.assertIn("shape exceeds 32-bit device index range", tiling)
        self.assertIn("dispatch output exceeds 32-bit device index range", tiling)

        dispatch = (ELASTIC / "dispatch.asc").read_text()
        for function_name, markers in {
                "direct_dispatch_producer_group_vf": (
                    "const std::uint32_t num_tokens_u32",
                    "for (std::uint32_t tile = grid.first",
                    "for (std::uint32_t local_token = 0",
                ),
                "direct_dispatch_producer_vector_payload_impl": (
                    "const std::uint32_t num_tokens_u32",
                    "const std::uint32_t block_index",
                    "for (std::uint32_t subgroup = 0",
                    "for (std::uint32_t tile =",
                ),
                "direct_dispatch_producer_record_body": (
                    "const std::uint32_t num_tokens_u32",
                    "for (std::uint32_t tile = source_tile_begin + grid.first",
                    "for (std::uint32_t logical = grid.first",
                ),
        }.items():
            begin = dispatch.index(f"{function_name}(")
            end = dispatch.index("\n}\n", begin)
            function = dispatch[begin:end]
            for marker in markers:
                self.assertIn(marker, function)

    def test_direct_epilogues_keep_shape_iteration_32_bit(self):
        dispatch = (ELASTIC / "dispatch.asc").read_text()
        for function_name, markers in {
                "direct_dispatch_producer_plan_vf": (
                    "const std::uint32_t num_tokens_u32",
                    "const std::uint32_t words_per_rank_u32",
                    "for (std::uint32_t token = 0",
                    "for (std::uint32_t lane = 0",
                ),
                "direct_dispatch_epilogue_validate_records_vf": (
                    "const std::uint32_t tile_count_u32",
                    "for (std::uint32_t tile = work.first",
                    "for (std::uint32_t logical_record = tile_begin",
                    "for (std::uint32_t lane = 0",
                ),
                "direct_dispatch_epilogue_count_experts_vf": (
                    "const std::uint32_t local_experts",
                    "for (std::uint32_t tile = work.first",
                    "for (std::uint32_t logical_record = tile_begin",
                ),
                "direct_dispatch_epilogue_metadata_vf": (
                    "const std::uint32_t logical_count",
                    "for (std::uint32_t tile = work.first",
                    "for (std::uint32_t lane = 0",
                ),
                "direct_dispatch_epilogue_assign_destinations_vf": (
                    "const std::uint32_t tile_count_u32",
                    "for (std::uint32_t tile = work.first",
                    "for (std::uint32_t lane = 0",
                ),
                "direct_dispatch_epilogue_vector_payload_impl": (
                    "const std::uint32_t logical_count",
                    "const std::uint32_t block_index",
                    "for (std::uint32_t logical = block_index",
                ),
                "direct_dispatch_epilogue_copy_outputs_vf": (
                    "const std::uint32_t logical_count",
                    "for (std::uint32_t logical = grid.first",
                    "for (std::uint32_t compact_lane = 0",
                ),
        }.items():
            begin = dispatch.index(f"inline void {function_name}")
            end = dispatch.index("\n}\n", begin)
            function = dispatch[begin:end]
            for marker in markers:
                self.assertIn(marker, function)

        combine = (ELASTIC / "combine.asc").read_text()
        for function_name, markers in {
                "direct_combine_producer_plan_prefix_vf": (
                    "const std::uint32_t rank_count_u32",
                    "const std::uint32_t tile_count_u32",
                    "for (std::uint32_t rank = 0",
                    "for (std::uint32_t tile = 0",
                ),
                "direct_combine_epilogue_clear_index_vf": (
                    "const std::uint32_t slot_count",
                    "for (std::uint32_t index = threadIdx.x",
                    "for (std::uint32_t tile = threadIdx.x",
                ),
                "direct_combine_epilogue_validate_vf": (
                    "const std::uint32_t tile_count_u32",
                    "for (std::uint32_t tile = work.first",
                    "for (std::uint32_t logical = tile_begin",
                ),
                "direct_combine_epilogue_reduce_errors_vf": (
                    "const std::uint32_t tile_count_u32",
                    "for (std::uint32_t tile = 0",
                ),
        }.items():
            begin = combine.index(f"inline void {function_name}")
            end = combine.index("\n}\n", begin)
            function = combine[begin:end]
            for marker in markers:
                self.assertIn(marker, function)

        combine = (ELASTIC / "combine.asc").read_text()
        for function_name, markers in {
                "direct_combine_producer_plan_vf": (
                    "const std::uint32_t num_source_rows_u32",
                    "for (std::uint32_t tile = work.first",
                    "for (std::uint32_t row = begin; row < end; ++row)",
                ),
                "direct_combine_producer_record_vf": (
                    "const std::uint32_t num_source_rows_u32",
                    "for (std::uint32_t row = grid.first",
                    "for (std::uint32_t hidden =",
                ),
                "direct_combine_epilogue_vector_reduce_impl": (
                    "const std::uint32_t num_tokens_u32",
                    "const std::uint32_t hidden_elements_u32",
                    "for (std::uint32_t token = block_index",
                    "for (std::uint32_t hidden = 0",
                ),
                "direct_combine_epilogue_reduce_vf": (
                    "const std::uint32_t num_tokens_u32",
                    "const std::uint32_t hidden_elements_u32",
                    "for (std::uint32_t token = work.first",
                    "for (std::uint32_t hidden = work.lane",
                ),
                "direct_combine_epilogue_weights_vf": (
                    "const std::uint32_t output_count",
                    "for (std::uint32_t logical = grid.first",
                ),
        }.items():
            begin = combine.index(f"inline void {function_name}")
            end = combine.index("\n}\n", begin)
            function = combine[begin:end]
            for marker in markers:
                self.assertIn(marker, function)

    def test_direct_dispatch_launcher_consumes_stage_pipeline(self):
        """Catches accepting 72 blocks without launching split data stages."""
        source = (ELASTIC / "dispatch.asc").read_text()
        launch = source[source.index(
            'extern "C" int deep_ep_ascend_launch_dispatch'):]
        self.assertIn("direct_dispatch_pipeline(", launch)
        self.assertIn("direct_dispatch_epilogue_pipeline(", launch)
        self.assertIn("direct_dispatch_stage_launch(", source)
        self.assertIn("launch_direct_dispatch_stage(", launch)
        self.assertIn("tiling.data_launch.num_blocks == 1", launch)

        for function_name in (
                "direct_dispatch_producer_record_body",
                "direct_dispatch_epilogue_copy_outputs_vf"):
            begin = source.index(f"{function_name}(")
            end = source.index("\n}\n", begin)
            function = source[begin:end]
            self.assertIn("direct_data_grid_stride(", function)

    def test_dispatch_source_pipeline_bounds_producer_work_by_chunk(self):
        """Catches source chunks that rescan tokens outside their tile range."""
        source = (ELASTIC / "dispatch.asc").read_text()
        for function_name in (
                "direct_dispatch_producer_record_body",
                "direct_dispatch_producer_token_fanout_impl"):
            begin = source.index(f"{function_name}(")
            end = source.index("\n}\n", begin)
            function = source[begin:end]
            self.assertIn("pipeline_source_chunk", function)
            self.assertIn("source_tile_begin", function)
            self.assertIn("source_tile_end", function)
            self.assertIn(
                "tile = source_tile_begin +", function)
            self.assertIn("tile < source_tile_end", function)
            self.assertNotIn(
                "tile = grid.first;\n             tile < tile_count_u32",
                function)

        release_begin = source.index(
            "direct_dispatch_producer_release_body(")
        release_end = source.index("\n}\n", release_begin)
        release = source[release_begin:release_end]
        for marker in (
                "dispatch_group_tile_offset",
                "source_tile_begin",
                "source_tile_end",
                "tile_offsets[source_tile_begin * world_size +",
                "tile_offsets[source_tile_end * world_size +"):
            self.assertIn(marker, release)

    def test_dispatch_source_pipeline_selector_and_launch_contract(self):
        """Catches routing source-tile chunks through slot-range launch rules."""
        buffer = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        runtime = (ELASTIC / "runtime.cpp").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        for marker in (
                "DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES",
                "select_dispatch_source_pipeline_config(",
                "arguments.pipeline_chunk_tiles =",
                "source_pipeline_config.enabled || pipeline_config.enabled"):
            self.assertIn(marker, buffer)
        self.assertIn("std::uint32_t pipeline_chunk_tiles = 0;", kernels)
        self.assertIn("std::uint32_t pipeline_source_chunk = 0;", kernels)
        self.assertIn("arguments.pipeline_chunk_slots == 0 &&", runtime)
        self.assertIn("arguments.pipeline_chunk_tiles == 0", runtime)
        self.assertIn("arguments.pipeline_chunk_slots != 0 &&", runtime)
        self.assertIn("arguments.pipeline_chunk_tiles != 0", runtime)

        dispatch = (ELASTIC / "dispatch.asc").read_text()
        launch = dispatch[dispatch.index(
            'extern "C" int deep_ep_ascend_launch_dispatch_pipeline'):]
        for marker in (
                "build_dispatch_source_chunk_plan(",
                "dispatch_source_chunk_bounds(",
                "arguments.pipeline_source_chunk = source_pipeline ? 1U : 0U",
                "direct_dispatch_profile_pipeline(cpu_sync)",
                "DirectDispatchStage::kProducerReleaseControl",
                "DirectDispatchStage::kProducerReleaseBarrier"):
            self.assertIn(marker, launch)

    def test_dispatch_persistent_pipeline_state_contract(self):
        """Catches stale producer or release doorbells in the persistent path."""
        layout = (ELASTIC / "layout.hpp").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        for marker in (
                "struct alignas(64) DispatchPipelineBlockProgress",
                "std::uint32_t scalar_blocks_completed = 0;",
                "std::uint32_t hidden_blocks_completed = 0;",
                "std::uint64_t generation = 0;",
                "std::uint32_t completed_chunk = UINT32_MAX;",
                "std::uint32_t release_batch_target = 0;",
                "std::uint32_t release_batch_consumed = 0;"):
            self.assertIn(marker, layout)
        for marker in (
                "dispatch_pipeline_slot_reusable(",
                "dispatch_pipeline_completion_is_last(",
                "dispatch_pipeline_slot_ready_for_release(",
                "dispatch_pipeline_block_progress_ready(",
                "dispatch_pipeline_release_batch_pending(",
                "dispatch_pipeline_release_batch_acknowledged("):
            self.assertIn(marker, kernels)
        for signature in (
            "DEEP_EP_ASCEND_KERNEL_CALLEE std::uint32_t\n"
                "dispatch_simt_pipeline_slot(",
            "DEEP_EP_ASCEND_KERNEL_CALLEE bool\n"
                "dispatch_simt_pipeline_completion_is_last(",
            "DEEP_EP_ASCEND_KERNEL_CALLEE bool\n"
                "dispatch_simt_pipeline_release_batch_acknowledged(",
            "DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE std::uint32_t\n"
                "dispatch_aicore_pipeline_slot(",
            "DEEP_EP_ASCEND_ELASTIC_AICORE_CALLEE bool\n"
                "dispatch_aicore_pipeline_release_batch_pending("):
            self.assertIn(signature, kernels)

    def test_dispatch_persistent_pipeline_waits_fail_with_diagnostics(self):
        """Catches an unbounded doorbell wait or one-sided worker failure."""
        layout = (ELASTIC / "layout.hpp").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        source = (ELASTIC / "dispatch.asc").read_text()
        for marker in (
                "enum class DispatchPipelineDiagnosticStage",
                "DispatchPipelineDiagnosticStage diagnostic_stage",
                "std::uint32_t diagnostic_detail = 0;"):
            self.assertIn(marker, layout)
        self.assertIn("dispatch_pipeline_wait_timed_out(", kernels)

        producer_name = "direct_dispatch_persistent_producer_vf("
        producer_begin = source.rfind(
            "__simt_vf__", 0, source.index(producer_name))
        producer_end = source.index("\n}\n", producer_begin)
        producer = source[producer_begin:producer_end]
        for marker in (
                "timeout_cycles",
                "__asc_simt_vf::clock()",
                "DispatchPipelineWorkerState::kFailed",
                "DispatchPipelineSlotState::kFailed"):
            self.assertIn(marker, producer)

        release_name = "direct_dispatch_persistent_release_vf("
        release_begin = source.rfind(
            "__simt_vf__", 0, source.index(release_name))
        release_end = source.index("\n}\n", release_begin)
        release = source[release_begin:release_end]
        for marker in (
                "__asc_simt_vf::clock()",
                "record_dispatch_protocol_error(",
                "DispatchPipelineWorkerState::kFailed",
                "DispatchPipelineSlotState::kFailed"):
            self.assertIn(marker, release)

        producer_device_name = "direct_dispatch_persistent_producer_device("
        producer_device_begin = source.rfind(
            "__aicore__", 0, source.index(producer_device_name))
        producer_device_end = source.index("\n}\n", producer_device_begin)
        producer_device = source[producer_device_begin:producer_device_end]
        self.assertIn("__asc_aicore::clock()", producer_device)
        self.assertIn("DispatchPipelineSlotState::kFailed", producer_device)

        release_device_name = "direct_dispatch_persistent_release_device("
        release_device_begin = source.rfind(
            "__aicore__", 0, source.index(release_device_name))
        release_device_end = source.index("\n}\n", release_device_begin)
        release_device = source[release_device_begin:release_device_end]
        self.assertIn(
            "asc_vf_call<direct_dispatch_persistent_release_wait_vf>",
            release_device)
        self.assertIn(
            "asc_vf_call<direct_dispatch_pipeline_wait_vf>", release_device)

    def test_dispatch_persistent_source_producer_contract(self):
        """Catches repeated VF launches or incomplete hidden publication."""
        source = (ELASTIC / "dispatch.asc").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        self.assertIn("kProducerRecordPipeline", kernels)
        function = "direct_dispatch_persistent_producer_vf("
        self.assertIn(function, source)
        begin = source.rfind("__simt_vf__", 0, source.index(function))
        end = source.index("\n}\n", begin)
        producer_vf = source[begin:end]
        for marker in (
                "for (std::uint32_t source_chunk = 0;",
                "direct_dispatch_producer_record_body(",
                "scalar_blocks_completed",
                "hidden_blocks_completed",
                "scalar_progress",
                "generation",
                "transport::simt::system_fence();"):
            self.assertIn(marker, producer_vf)
        self.assertIn(
            "&progress->generation, generation", producer_vf)
        self.assertNotIn(
            "&progress->completed_chunk", producer_vf)
        self.assertNotIn(
            "DispatchPipelineSlotState::kReady", producer_vf)
        self.assertNotIn("asc_atomic_add(", producer_vf)
        self.assertNotIn(
            "DispatchPipelineSlotState::kScalarReady", producer_vf)

        kernel_begin = source.index("__global__ __vector__ void dispatch_kernel")
        kernel_end = source.index(
            "template <bool ProfileEnabled>\ninline int launch_dispatch_kernel",
            kernel_begin)
        kernel = source[kernel_begin:kernel_end]
        producer_begin = kernel.index(
            "stage == DirectDispatchStage::kProducerRecordPipeline")
        release_begin = kernel.index(
            "stage == DirectDispatchStage::kProducerReleasePipeline",
            producer_begin)
        producer = kernel[producer_begin:release_begin]
        self.assertIn(
            "direct_dispatch_persistent_producer_device(", producer)
        helper_begin = source.index(
            "__aicore__ inline void "
            "direct_dispatch_persistent_producer_device(")
        helper_end = source.index("\n}\n", helper_begin)
        helper = source[helper_begin:helper_end]
        self.assertIn(
            "observed_generation == generation", helper)
        launch = "asc_vf_call<direct_dispatch_persistent_producer_vf>"
        self.assertEqual(helper.count(launch), 1)
        self.assertNotIn("direct_dispatch_persistent_chunk_begin_vf", producer)
        self.assertNotIn(
            "direct_dispatch_persistent_chunk_complete_vf", producer)
        self.assertNotIn(
            "direct_dispatch_persistent_release_tail_wait_vf", producer)
        launch_index = helper.index(launch)
        self.assertIn("dim3(tiling.launch.num_threads), x", helper)
        self.assertIn(
            "const std::uint32_t producer_blocks =\n"
            "        tiling.data_launch.num_blocks;", helper)
        for marker in (
                "&pipeline->abi_version",
                "&pipeline->producer_worker_state",
                "&pipeline->generation"):
            self.assertLess(helper.index(marker), launch_index)
        loop_index = helper.index(
            "for (std::uint32_t source_chunk = 0;", launch_index)
        self.assertLess(launch_index, loop_index)
        self.assertNotIn("asc_vf_call<", helper[loop_index:])
        for marker in (
                "DispatchPipelineSlotState::kScalarReady",
                "scalar_progress",
                "hidden_progress",
                "completed_chunk",
                "DispatchPipelineSlotState::kReady",
                "transport::aicore::store_device(",
                "transport::aicore::flush_cacheline("):
            self.assertIn(marker, helper[loop_index:])

    def test_dispatch_persistent_release_contract(self):
        """Keeps release service outside the blocking release VF."""
        source = (ELASTIC / "dispatch.asc").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        layout = (ELASTIC / "layout.hpp").read_text()
        self.assertIn("kProducerReleasePipeline", kernels)
        self.assertIn("direct_dispatch_persistent_release_vf", source)
        self.assertIn(
            "std::uint64_t release_completed_generation = 0;", layout)

        kernel_begin = source.index("__global__ __vector__ void dispatch_kernel")
        kernel_end = source.index(
            "template <bool ProfileEnabled>\ninline int launch_dispatch_kernel",
            kernel_begin)
        kernel = source[kernel_begin:kernel_end]
        stage = kernel.index(
            "stage == DirectDispatchStage::kProducerReleasePipeline")
        release = kernel[stage:]
        self.assertIn(
            "direct_dispatch_persistent_release_device<ProfileEnabled>(",
            release)
        helper_begin = source.index(
            "__aicore__ inline void "
            "direct_dispatch_persistent_release_device(")
        helper_end = source.index("\n}\n", helper_begin)
        helper = source[helper_begin:helper_end]
        self.assertNotIn(
            "asc_vf_call<direct_dispatch_persistent_release_vf>", helper)
        for marker in (
                "asc_vf_call<direct_dispatch_persistent_release_wait_vf>",
                "direct_dispatch_release_chunk_device<ProfileEnabled>(",
                "asc_vf_call<direct_dispatch_pipeline_wait_vf>",
                "asc_vf_call<direct_dispatch_persistent_release_complete_vf>",
                "DirectReleaseSegment::kPayload",
                "DirectReleaseSegment::kControl",
                "DirectReleaseSegment::kBarrier"):
            self.assertIn(marker, helper)
        release_chunk_begin = source.index(
            "__aicore__ inline void direct_dispatch_release_chunk_device(")
        release_chunk_end = source.index("\n}\n", release_chunk_begin)
        release_chunk = source[release_chunk_begin:release_chunk_end]
        self.assertIn(
            "transport::service::execute<ProfileEnabled>", release_chunk)
        # ProducerControl owns the per-dispatch reset.  The communication
        # stream must not reset the shared queue concurrently with producer
        # route-plan publication.
        self.assertNotIn("transport::service::reset(", helper)
        release_wait = helper.index(
            "asc_vf_call<direct_dispatch_persistent_release_wait_vf>")
        release_payload = helper.index(
            "DirectReleaseSegment::kPayload", release_wait)
        request_wait = helper.index(
            "asc_vf_call<direct_dispatch_pipeline_wait_vf>", release_payload)
        self.assertLess(release_wait, release_payload)
        self.assertLess(release_payload, request_wait)
        complete_begin = source.index(
            "direct_dispatch_persistent_release_complete_vf(")
        complete_end = source.index("\n}\n", complete_begin)
        complete = source[complete_begin:complete_end]
        self.assertIn(
            "DispatchPipelineWorkerState::kCompleted", complete)

    def test_dispatch_persistent_pipeline_launch_is_event_free(self):
        """Catches reintroducing host event or sync gaps into source chunks."""
        source = (ELASTIC / "dispatch.asc").read_text()
        signature = "inline int launch_persistent_dispatch_source_pipeline("
        begin = source.index(signature)
        end = source.index("\n}\n", begin)
        launch = source[begin:end]
        for marker in (
                "DirectDispatchStage::kProducerControl",
                "DirectDispatchStage::kProducerGroup",
                "DirectDispatchStage::kProducerPrefix",
                "DirectDispatchStage::kProducerReleasePipeline",
                "DirectDispatchStage::kProducerRecordPipeline",
                "dispatch_persistent_producer_blocks(\n"
                "            tiling.data_launch.num_blocks)",
                "pipeline_tiling.data_launch.num_blocks = producer_blocks",
                "arguments, pipeline_tiling, communication_stream",
                "arguments, pipeline_tiling, producer_stream"):
            self.assertIn(marker, launch)
        for forbidden in (
                "aclrtCreateEventWithFlag",
                "aclrtRecordEvent",
                "aclrtStreamWaitEvent",
                "aclrtSynchronizeStream",
                "aclrtDestroyEvent"):
            self.assertNotIn(forbidden, launch)

    def test_dispatch_persistent_epilogue_uses_producer_stream_completion(self):
        """Keeps epilogue after producer completion and on release acquire."""
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_acquire_vf")
        end = source.index("\n}\n", begin)
        acquire = source[begin:end]
        for marker in (
                "dispatch_pipeline_offset",
                "dispatch_pipeline_bytes",
                "persistent_source_pipeline",
                "transport.consumed_generation()"):
            self.assertIn(marker, acquire)
        self.assertNotIn(
            "pipeline->release_completed_generation", acquire)
        self.assertNotIn("pipeline->reserved2", acquire)
        self.assertIn(
            "persistent_source_pipeline != 0 ?", acquire)

        macro_begin = source.index(
            "#define DEEP_EP_ASCEND_DIRECT_DISPATCH_EPILOGUE")
        macro_end = source.index("#undef DEEP_EP_ASCEND_DIRECT_DISPATCH_EPILOGUE")
        macro = source[macro_begin:macro_end]
        self.assertIn(
            "tiling.workspace_layout.dispatch_pipeline_offset", macro)
        self.assertIn(
            "tiling.workspace_layout.dispatch_pipeline_bytes", macro)
        self.assertIn("(PERSISTENT_SOURCE_PIPELINE)", macro)

        producer_begin = source.index(
            "__aicore__ inline void "
            "direct_dispatch_persistent_producer_device(")
        producer_end = source.index("\n}\n", producer_begin)
        producer = source[producer_begin:producer_end]
        self.assertIn("asc_sync_vec();", producer)
        self.assertEqual(
            producer.count(
                "asc_vf_call<direct_dispatch_persistent_producer_vf>"), 1)

        launch_begin = source.index(
            "inline int launch_persistent_dispatch_source_pipeline(")
        launch_end = source.index("\n}\n", launch_begin)
        launch = source[launch_begin:launch_end]
        self.assertLess(
            launch.index("DirectDispatchStage::kProducerRecordPipeline"),
            launch.index("for (std::uint32_t index = epilogue_begin;"))

    def test_dispatch_persistent_control_acquire_diagnoses_before_wait(self):
        """Makes a missing remote release observable without a long wait."""
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_acquire_vf")
        end = source.index("\n}\n", begin)
        acquire = source[begin:end]
        loop = acquire[acquire.index(
            "for (int source_rank = 0;"):]
        for marker in (
                "observed_control_generation",
                "transport.read_signal(",
                "persistent_control_diagnostic",
                "record_dispatch_protocol_error(",
                "release_protocol::observe_release_control("):
            self.assertIn(marker, loop)
        self.assertLess(
            loop.index("persistent_control_diagnostic"),
            loop.index("release_protocol::observe_release_control("))

    def test_dispatch_pipeline_header_uses_noncacheable_publication(self):
        """Prevents a dirty header cacheline from overwriting the tail gate."""
        probe = PROBE.read_text()
        self.assertIn(
            "offsetof(DispatchPipelineState, "
            "release_completed_generation) ==\n              64",
            probe)
        self.assertIn(
            "offsetof(DispatchPipelineState, slots) == 128", probe)
        source = (ELASTIC / "dispatch.asc").read_text()
        for forbidden in (
                "pipeline->abi_version =",
                "pipeline->struct_size =",
                "pipeline->generation =",
                "pipeline->chunk_count =",
                "pipeline->completed_chunks =",
                "pipeline->chunk_slots =",
                "pipeline->terminal_error =",
                "pipeline->release_completed_generation =",
                "++pipeline->completed_chunks"):
            self.assertNotIn(forbidden, source)
        for field in (
                "abi_version", "struct_size", "generation", "chunk_count",
                "completed_chunks", "chunk_slots",
                "release_completed_generation"):
            self.assertRegex(
                source,
                rf"transport::simt::store_published\(\s*"
                rf"&pipeline->{field}")
        self.assertRegex(
            source,
            r"reinterpret_cast<__gm__ std::uint32_t\*>\(\s*"
            r"&pipeline->terminal_error\)")

        external = source[source.index(
            'extern "C" int deep_ep_ascend_launch_dispatch_pipeline'):]
        source_branch = external.index("if (source_pipeline)")
        persistent_call = external.index(
            "launch_persistent_dispatch_source_pipeline(", source_branch)
        legacy_events = external.index("aclrtEvent ready[")
        self.assertLess(source_branch, persistent_call)
        self.assertLess(persistent_call, legacy_events)

    def test_direct_dispatch_receive_validation_is_rank_major_tiled(self):
        """Catches restoring one serial scan per source rank."""
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_validate_records_vf")
        end = source.index("\n}\n", begin)
        validation = source[begin:end]
        for marker in (
                "direct_block_distributed_grid_stride(",
                "kDispatchReceiveRecordsPerTile",
                "dispatch_simt_receive_record_coordinates(",
                "source_slot >= source_counts[source_rank]",
                "tile_errors[tile] ="):
            self.assertIn(marker, validation)
        self.assertNotIn("record_dispatch_protocol_error(", validation)

        launch = source[source.index(
            "#define DEEP_EP_ASCEND_DIRECT_DISPATCH_EPILOGUE"):]
        validate = launch.index(
            "(STAGE) == DirectDispatchStage::kEpilogueValidate")
        validate_call = launch.index(
            "asc_vf_call<direct_dispatch_epilogue_validate_records_vf>",
            validate)
        reduce = launch.index(
            "(STAGE) == DirectDispatchStage::kEpilogueValidateReduce",
            validate_call)
        reduce_call = launch.index(
            "asc_vf_call<direct_dispatch_reduce_errors_vf>", reduce)
        self.assertLess(validate, validate_call)
        self.assertLess(validate_call, reduce)
        self.assertLess(reduce, reduce_call)

    def test_direct_dispatch_expert_histogram_prefix_scatter_is_tiled(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        count_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_count_experts_vf")
        prefix_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_prefix_vf", count_begin)
        count = source[count_begin:prefix_begin]
        for marker in (
                "direct_block_distributed_grid_stride(",
                "kDispatchReceiveRecordsPerTile",
                "static_cast<std::uint64_t>(tile) * local_experts +"):
            self.assertIn(marker, count)
        self.assertNotIn(
            "for (std::uint64_t local_expert = threadIdx.x", count)

        scatter_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_assign_destinations_vf")
        scatter_end = source.index("\n}\n", scatter_begin)
        scatter = source[scatter_begin:scatter_end]
        self.assertIn("if (!cached) {", scatter)
        self.assertIn("direct_block_distributed_grid_stride(", scatter)
        self.assertIn("tile_cursors[index]++", scatter)
        self.assertIn("owner_bitmap[word] & mask", scatter)

    def test_direct_dispatch_grouping_reuse_contract(self):
        """Catches restoring destination-by-destination top-k rescans."""
        source = (ELASTIC / "dispatch.asc").read_text()
        device_macro = "#define DEEP_EP_ASCEND_TOPK_GROUPING_DEVICE 1"
        self.assertIn(device_macro, source)
        self.assertLess(
            source.index(device_macro),
            source.index('#include "topk_grouping.hpp"'))
        self.assertIn('#include "topk_grouping.hpp"', source)

        group_signature = (
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_group_vf")
        prefix_signature = (
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_prefix_vf")
        group_begin = source.index(group_signature)
        group_end = source.index("\n}\n", group_begin)
        grouping = source[group_begin:group_end]
        for marker in (
                "direct_subgroup_grid_stride(", "group_topk_subgroup(",
                "dispatch_group_owner_offset", "dispatch_group_tile_offset",
                "dispatch_group_error_offset"):
            self.assertIn(marker, grouping)

        prefix_begin = source.index(prefix_signature)
        prefix_end = source.index("\n}\n", prefix_begin)
        prefix = source[prefix_begin:prefix_end]
        self.assertIn("dispatch_group_tile_offset", prefix)
        self.assertIn("dispatch_group_error_offset", prefix)

        record_begin = source.index(
            "direct_dispatch_producer_record_body(")
        record_end = source.index("\n}\n", record_begin)
        record = source[record_begin:record_end]
        for marker in (
                "num_topk <= kTopkSubgroupWidth",
                "transport_world_size <= "
                "static_cast<int>(kTopkSubgroupWidth)",
                "dispatch_group_owner_offset",
                "dispatch_group_tile_offset"):
            self.assertIn(marker, record)
        self.assertIn("direct_data_grid_stride(", record)

        grouped_begin = record.index("if (grouped) {")
        grouped_end = record.index("        return;", grouped_begin)
        grouped_record = record[grouped_begin:grouped_end]
        self.assertIn("owners[token * world_size + lane]", grouped_record)
        self.assertNotIn(
            "for (int destination_rank = 0;", grouped_record)
        self.assertNotIn(
            "for (std::uint64_t lane = 0; lane < num_topk; ++lane)",
            grouped_record)

    def test_direct_dispatch_vector_payload_contract(self):
        """Catches payload copies racing record writers on another AIV."""
        source = (ELASTIC / "dispatch.asc").read_text()
        for marker in (
                '#include "c_api/sync/sync.h"',
                '#include "simt_api/device_sync_functions.h"',
                "kDispatchVectorTileBytes",
                "direct_dispatch_producer_vector_payload_impl",
                "direct_dispatch_epilogue_vector_payload_impl",
                "direct_dispatch_store_scale_factor_pack",
                "direct_dispatch_load_scale_factor_pack"):
            self.assertIn(marker, source)

        producer_begin = source.index(
            "__aicore__ inline void "
            "direct_dispatch_producer_vector_payload_impl")
        producer_end = source.index("\n}\n", producer_begin)
        producer = source[producer_begin:producer_end]
        for marker in (
                "AscendC::GetBlockIdx()", "AscendC::GetBlockNum()",
                "AscendC::DataCopy", "dispatch_group_owner_offset",
                "dispatch_group_tile_count", "launch_num_threads",
                "subgroups_per_block", "kTopkSubgroupWidth",
                "kDispatchGroupingTokensPerTile", "destination_slots",
                "AscendC::TBuf", "EVENT_ID0"):
            self.assertIn(marker, producer)
        self.assertIn(
            "launch_num_threads / kTopkSubgroupWidth", producer)
        self.assertIn(
            "block_index * subgroups_per_block + subgroup", producer)

        self.assertIn(
            "block_count * subgroups_per_block", producer)
        self.assertIn("tile < source_tile_end", producer)
        self.assertIn(
            "tile * static_cast<std::uint32_t>(\n"
            "                    kDispatchGroupingTokensPerTile)", producer)
        self.assertNotIn("logical_count = num_tokens * world_size", producer)
        self.assertNotIn("input_queue", producer)
        for ready, release in (
                ("HardEvent::MTE2_MTE3", "HardEvent::MTE3_MTE2"),):
            self.assertIn(ready, producer)
            self.assertIn(release, producer)
            self.assertLess(producer.index(ready), producer.index(release))

        epilogue_begin = source.index(
            "__aicore__ inline void "
            "direct_dispatch_epilogue_vector_payload_impl")
        epilogue_end = source.index("\n}\n", epilogue_begin)
        epilogue = source[epilogue_begin:epilogue_end]
        for marker in (
                "AscendC::GetBlockIdx()", "AscendC::GetBlockNum()",
                "AscendC::DataCopy", "source_metadata", "expanded",
                "AscendC::TBuf", "EVENT_ID0"):
            self.assertIn(marker, epilogue)
        self.assertNotIn("input_queue", epilogue)
        self.assertIn("HardEvent::MTE2_MTE3", epilogue)
        self.assertIn("HardEvent::MTE3_MTE2", epilogue)
        self.assertLess(
            epilogue.index("HardEvent::MTE2_MTE3"),
            epilogue.index("HardEvent::MTE3_MTE2"))

    def test_dispatch_token_fanout_selector_and_launch_contract(self):
        """Catches accepting the selector without reaching the AICore path."""
        source = (ELASTIC / "dispatch.asc").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        buffer = (
            ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        dispatch = buffer[
            buffer.index("    dispatch(const torch::Tensor& x"):
            buffer.index("    combine(const torch::Tensor& x")
        ]

        for marker in (
                '"DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT"',
                "select_dispatch_token_fanout_config(",
                "DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT must be 0 or 1",
                "arguments.token_fanout =\n"
                "            token_fanout_config.enabled ? 1U : 0U;"):
            self.assertIn(marker, dispatch)
        self.assertIn("std::uint32_t token_fanout = 0;", kernels)

        kernel_begin = source.index(
            "__global__ __vector__ void dispatch_kernel")
        kernel_body = source.index("{", kernel_begin)
        kernel_signature = source[kernel_begin:kernel_body]
        self.assertIn("std::uint32_t token_fanout", kernel_signature)

        launch_begin = source.index(
            "template <bool ProfileEnabled>\ninline int "
            "launch_dispatch_kernel")
        launch_end = source.index("\n}\n", launch_begin)
        launcher = source[launch_begin:launch_end]
        self.assertIn("arguments.token_fanout", launcher)

    def test_dispatch_token_fanout_is_disabled_for_either_pipeline(self):
        """Keeps unvalidated fan-out out of both persistent pipeline paths."""
        buffer = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        dispatch = buffer[
            buffer.index("    dispatch(const torch::Tensor& x"):
            buffer.index("    combine(const torch::Tensor& x")
        ]
        fanout_call = dispatch[
            dispatch.index("select_dispatch_token_fanout_config("):
            dispatch.index(
                "TORCH_CHECK(\n            token_fanout_config_status",
                dispatch.index("select_dispatch_token_fanout_config("))]
        self.assertIn(
            "pipeline_config.enabled || source_pipeline_config.enabled",
            fanout_call)
        self.assertNotIn(
            "stream_mode, pipeline_config.enabled, num_topk",
            fanout_call)

    def test_dispatch_token_fanout_loads_each_token_before_destination_loop(
            self):
        """Catches reloading a token's hidden row for every destination."""
        source = (ELASTIC / "dispatch.asc").read_text()
        signature = (
            "__aicore__ inline void "
            "direct_dispatch_producer_token_fanout_impl")
        self.assertIn(signature, source)
        begin = source.index(signature)
        end = source.index("\n}\n", begin)
        candidate = source[begin:end]

        self.assertIn(
            "pipe.InitBuffer(\n"
            "        payload_buffer, kDispatchTokenFanoutBufferBytes)",
            candidate)
        source_copy = (
            "AscendC::DataCopy(\n"
            "                    payload_local, input_global, vector_bytes);")
        destination_loop = (
            "for (std::uint32_t index = 0;\n"
            "                     index < destination_count; ++index)")
        destination_copy = (
            "AscendC::DataCopy(\n"
            "                        output_global, payload_local, "
            "vector_bytes);")
        self.assertEqual(candidate.count(source_copy), 1)
        self.assertIn(destination_loop, candidate)
        self.assertLess(
            candidate.index(source_copy), candidate.index(destination_loop))
        rank_body = candidate[candidate.index(destination_loop):]
        self.assertIn(destination_copy, rank_body)
        self.assertNotIn("input_global.SetGlobalBuffer", rank_body)
        self.assertNotIn(source_copy, rank_body)

        record_call = source.index(
            "asc_vf_call<direct_dispatch_producer_record_vf>")
        release_call = source.index(
            "asc_vf_call<direct_dispatch_producer_release_vf>", record_call)
        producer_launch = source[record_call:release_call]
        self.assertIn("if (token_fanout != 0)", producer_launch)
        self.assertIn(
            "direct_dispatch_producer_token_fanout_impl(", producer_launch)
        self.assertIn("} else {", producer_launch)
        self.assertIn(
            "direct_dispatch_producer_vector_payload_impl(", producer_launch)

    def test_dispatch_grouping_builds_rank_and_expert_route_counts_together(
            self):
        """Catches restoring the receiver payload rescan as count source."""
        source = (ELASTIC / "dispatch.asc").read_text()
        group_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_group_vf")
        group_end = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_prefix_vf", group_begin)
        group = source[group_begin:group_end]
        for marker in (
                "dispatch_group_expert_count_offset",
                "dispatch_group_expert_count",
                "expert_counts",
                "asc_atomic_add(",
                "&expert_counts[static_cast<std::uint32_t>(expert)]",
                "1U"):
            self.assertIn(marker, group)
        self.assertNotIn("tile_expert_counts", group)

        prefix_begin = group_end
        prefix_end = source.index("\n}\n", prefix_begin)
        prefix = source[prefix_begin:prefix_end]
        for marker in (
                "dispatch_group_expert_count_offset",
                "dispatch_staging_offset",
                "dispatch_staging_shard_bytes",
                "dispatch_route_plan_slot_bytes",
                "dispatch_route_plan_expert_capacity",
                "expert_counts",
                "route_plan_staging"):
            self.assertIn(marker, prefix)
        self.assertNotIn("tile_expert_counts", prefix)
        self.assertNotIn(
            "for (std::uint32_t tile = 0; tile < tile_count_u32; ++tile)\n"
            "                count +=", prefix)

    def test_dispatch_route_plan_storage_uses_actual_local_expert_count(self):
        """Padded route slots must not change the expert-to-rank mapping."""
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_prefix_vf")
        end = source.index("\n}\n", begin)
        prefix = source[begin:end]
        self.assertIn(
            "const std::uint64_t num_local_experts =\n"
            "        num_experts / world_size;", prefix)
        self.assertIn(
            "storage_expert / dispatch_route_plan_expert_capacity", prefix)
        self.assertIn(
            "storage_expert % dispatch_route_plan_expert_capacity", prefix)
        self.assertIn(
            "destination_rank * num_local_experts + local_expert", prefix)
        self.assertNotIn(
            "const std::uint32_t expert = threadIdx.x;", prefix)

    def test_dispatch_grouping_uses_native_atomic_expert_accumulator(self):
        """Catches restoring the serial per-tile expert-count reduction."""
        source = (ELASTIC / "dispatch.asc").read_text()
        init_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_expert_count_init_vf")
        init_end = source.index("\n}\n", init_begin)
        init = source[init_begin:init_end]
        self.assertIn("threadIdx.x", init)
        self.assertIn("index += blockDim.x", init)
        self.assertIn("expert_counts[index] = 0;", init)

        begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_group_vf")
        end = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_prefix_vf", begin)
        group = source[begin:end]
        self.assertIn("asc_atomic_add(", group)
        self.assertIn(
            "&expert_counts[static_cast<std::uint32_t>(expert)]", group)
        self.assertNotIn("tile_expert_counts", group)
        self.assertNotIn(
            "for (std::uint32_t expert = 0;\n"
            "                 expert < num_experts_u32; ++expert)", group)

        control_stage = source.index(
            "stage == DirectDispatchStage::kProducerControl")
        init_call = source.index(
            "asc_vf_call<direct_dispatch_producer_expert_count_init_vf>",
            control_stage)
        group_stage = source.index(
            "stage == DirectDispatchStage::kProducerGroup", init_call)
        self.assertLess(init_call, group_stage)
        init_launch = source[init_call:group_stage]
        self.assertIn(
            "tiling.workspace_layout.dispatch_group_expert_count_offset",
            init_launch)
        self.assertIn(
            "tiling.workspace_layout.dispatch_group_expert_count", init_launch)

    def test_dispatch_early_route_plan_publishes_before_payload_packing(self):
        """Catches coupling receiver prefix back to payload readiness."""
        source = (ELASTIC / "dispatch.asc").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        buffer = (
            ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        dispatch = buffer[
            buffer.index("    dispatch(const torch::Tensor& x"):
            buffer.index("    combine(const torch::Tensor& x")
        ]
        for marker in (
                '"DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN"',
                "select_dispatch_early_route_plan_config(",
                "DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN must be 0 or 1",
                "arguments.early_route_plan =\n"
                "            early_route_plan_config.enabled ? 1U : 0U;"):
            self.assertIn(marker, dispatch)
        self.assertIn("std::uint32_t early_route_plan = 0;", kernels)

        publish_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_publish_route_plan_vf")
        publish_end = source.index("\n}\n", publish_begin)
        publish = source[publish_begin:publish_end]
        self.assertIn("transport.put(", publish)
        self.assertIn(
            "transport_local_window_base + dispatch_staging_offset",
            publish)
        self.assertNotIn(
            "workspace + dispatch_route_plan_staging_offset", publish)
        flush_call = "release_protocol::flush_payload(transport)"
        self.assertEqual(publish.count(flush_call), 2)
        self.assertIn(
            "transport::sync_layout::kDispatchRouteReadySignalIndex",
            publish)
        self.assertLess(
            publish.index("transport.put("), publish.index(flush_call))
        self.assertLess(
            publish.index(flush_call),
            publish.index("kDispatchRouteReadySignalIndex"))
        self.assertLess(
            publish.index("kDispatchRouteReadySignalIndex"),
            publish.rindex(flush_call))
        self.assertIn("std::uint64_t timeout_cycles", publish)
        self.assertIn("transport.device_barrier(", publish)
        self.assertIn("transport::kWorldTeamMask", publish)
        self.assertLess(
            publish.rindex(flush_call),
            publish.index("transport.device_barrier("))

        acquire_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_acquire_route_plan_vf")
        acquire_end = source.index("\n}\n", acquire_begin)
        acquire = source[acquire_begin:acquire_end]
        self.assertIn("release_protocol::acquire_release(", acquire)
        self.assertIn("prefix_per_rank", acquire)
        self.assertIn("prefix_per_expert", acquire)
        self.assertIn("unaligned_per_expert", acquire)
        self.assertNotIn("dispatch_receive_offset", acquire)

        payload_acquire_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_acquire_vf")
        payload_acquire_end = source.index("\n}\n", payload_acquire_begin)
        payload_acquire = source[payload_acquire_begin:payload_acquire_end]
        self.assertIn("std::uint32_t early_route_plan", payload_acquire)
        self.assertIn(
            "dispatch_simt_handoff_dispatch_route_source_count(",
            payload_acquire)
        self.assertIn(
            "canonical_source_counts[source_rank] = canonical_count;",
            payload_acquire)

        epilogue_macro_begin = source.index(
            "#define DEEP_EP_ASCEND_DIRECT_DISPATCH_EPILOGUE")
        epilogue_macro_end = source.index("\n    } while (0)", epilogue_macro_begin)
        epilogue_macro = source[epilogue_macro_begin:epilogue_macro_end]
        self.assertIn(
            "(EARLY_ROUTE_PLAN) == 0 && \\\n"
            "            ((STAGE) == DirectDispatchStage::kFull || \\\n"
            "             (STAGE) == DirectDispatchStage::kEpilogueExpertCount)",
            epilogue_macro)
        self.assertIn(
            "(EARLY_ROUTE_PLAN) == 0 && \\\n"
            "            ((STAGE) == DirectDispatchStage::kFull || \\\n"
            "             (STAGE) == DirectDispatchStage::kEpilogueExpertPrefix)",
            epilogue_macro)
        self.assertIn(
            "copy_outputs, stage, early_route_plan, pipeline_source_chunk);",
            source)
        self.assertIn(
            "1U, DirectDispatchStage::kFull, 0U, 0U);", source)

        prefix_stage = source.index(
            "stage == DirectDispatchStage::kProducerPrefix")
        publish_call = source.index(
            "asc_vf_call<direct_dispatch_publish_route_plan_vf>",
            prefix_stage)
        service_call = source.index(
            "transport::service::execute<ProfileEnabled>", publish_call)
        publish_launch = source[publish_call:service_call]
        self.assertIn("generation, timeout_cycles,", publish_launch)
        acquire_call = source.index(
            "asc_vf_call<direct_dispatch_acquire_route_plan_vf>",
            service_call)
        record_stage = source.index(
            "stage == DirectDispatchStage::kProducerRecord", acquire_call)
        self.assertLess(publish_call, service_call)
        self.assertLess(service_call, acquire_call)
        self.assertLess(acquire_call, record_stage)

    def test_dispatch_early_route_plan_keeps_outgoing_and_incoming_counts_separate(
            self):
        """Catches route acquire clobbering counts needed by producer release."""
        source = (ELASTIC / "dispatch.asc").read_text()
        layout = (ELASTIC / "layout.hpp").read_text()
        tiling = (ELASTIC / "tiling.hpp").read_text()

        for marker in (
                "dispatch_route_source_counts_offset",
                "dispatch_route_source_counts_bytes"):
            self.assertIn(marker, layout)
            self.assertIn(marker, tiling)

        acquire_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_acquire_route_plan_vf")
        acquire_end = source.index("\n}\n", acquire_begin)
        acquire = source[acquire_begin:acquire_end]
        self.assertIn("workspace_route_source_counts_offset", acquire)
        self.assertIn(
            "workspace + workspace_route_source_counts_offset", acquire)
        self.assertNotIn(
            "workspace + workspace_rank_counts_offset", acquire)

        payload_acquire_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_acquire_vf")
        payload_acquire_end = source.index("\n}\n", payload_acquire_begin)
        payload_acquire = source[
            payload_acquire_begin:payload_acquire_end]
        self.assertIn("workspace_route_source_counts_offset", payload_acquire)
        self.assertIn(
            "early_route_plan != 0 ?\n"
            "            workspace_route_source_counts_offset :\n"
            "            workspace_rank_counts_offset",
            payload_acquire)

        release_begin = source.index(
            "direct_dispatch_producer_release_body(")
        release_end = source.index("\n}\n", release_begin)
        release = source[release_begin:release_end]
        self.assertIn("workspace + workspace_rank_counts_offset", release)
        self.assertNotIn("workspace_route_source_counts_offset", release)

    def test_dispatch_consumer_tile_specializations(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        buffer = (
            ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        dispatch = buffer[
            buffer.index("    dispatch(const torch::Tensor& x"):
            buffer.index("    combine(const torch::Tensor& x")
        ]

        self.assertIn("std::uint32_t consumer_tile_bytes = 512;", kernels)
        self.assertIn(
            '"DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES"', dispatch)
        self.assertIn("select_dispatch_consumer_tile_config(", dispatch)
        self.assertIn(
            "arguments.consumer_tile_bytes = "
            "consumer_tile_config.tile_bytes;", dispatch)

        vector_signature = (
            "template <std::uint32_t TileBytes>\n"
            "__aicore__ inline void "
            "direct_dispatch_epilogue_vector_payload_impl")
        vector_begin = source.index(vector_signature)
        vector_end = source.index("\n}\n", vector_begin)
        vector_copy = source[vector_begin:vector_end]
        for marker in (
                "dispatch_consumer_copy_plan(",
                "pipe.InitBuffer(payload_buffer, TileBytes)",
                "const std::uint32_t copy_bytes",
                "AscendC::DataCopy(", "copy_bytes"):
            self.assertIn(marker, vector_copy)

        for tile_bytes in (512, 1024, 2048, 4096, 8192):
            self.assertEqual(
                source.count(
                    "direct_dispatch_epilogue_vector_payload_impl<"
                    f"{tile_bytes}>("),
                1,
            )
        self.assertIn(
            "consumer_copy_plan.scalar_begin", source)

    def test_dispatch_producer_uses_2048_byte_vector_tiles(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        self.assertIn(
            "constexpr std::uint32_t kDispatchProducerVectorTileBytes = 2048;",
            source,
        )
        producer_begin = source.index(
            "__aicore__ inline void direct_dispatch_producer_vector_payload_impl")
        producer_end = source.index("\n}\n", producer_begin)
        producer = source[producer_begin:producer_end]
        self.assertIn(
            "payload_buffer, kDispatchProducerVectorTileBytes", producer)
        self.assertIn(
            "byte += kDispatchProducerVectorTileBytes", producer)
        self.assertIn(
            "kDispatchProducerVectorTileBytes);", producer)
        self.assertNotIn(
            "payload_buffer, kDispatchVectorTileBytes", producer)

    def test_dispatch_token_fanout_batches_destination_store_completion(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = source.index(
            "__aicore__ inline void direct_dispatch_producer_token_fanout_impl")
        end = source.index("\n}\n", begin)
        fanout = source[begin:end]
        self.assertEqual(fanout.count("AscendC::SetFlag<\n"
                                      "                    AscendC::HardEvent::MTE3_MTE2>"), 1)
        self.assertEqual(fanout.count("AscendC::WaitFlag<\n"
                                      "                    AscendC::HardEvent::MTE3_MTE2>"), 1)
        store = fanout.index("AscendC::DataCopy(\n                        output_global")
        wait = fanout.index("AscendC::WaitFlag<\n"
                            "                    AscendC::HardEvent::MTE3_MTE2>", store)
        self.assertLess(store, wait)

    def test_dispatch_producer_preserves_nonfanout_hidden_tail(self):
        """Keeps the non-2048-byte hidden suffix on the scalar record path."""
        source = (ELASTIC / "dispatch.asc").read_text()
        for signature in (
                "__aicore__ inline void "
                "direct_dispatch_persistent_producer_device(",
                "__global__ __vector__ void dispatch_kernel("):
            begin = source.index(signature)
            end = source.index("\n}\n", begin)
            function = source[begin:end]
            vector_begin = function.index("const std::uint64_t vector_hidden_bytes")
            vector_expr = function[vector_begin:vector_begin + 420]
            self.assertIn(
                "kDispatchProducerVectorTileBytes", vector_expr)
            self.assertIn(
                "vector_hidden_bytes", function[vector_begin:])

    def test_dispatch_producer_prefix_parallelizes_large_tile_scans(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_producer_prefix_vf")
        end = source.index("\n}\n", begin)
        prefix = source[begin:end]
        for marker in (
                "const std::uint32_t group_width",
                "const std::uint32_t chunks_per_rank",
                "tile_errors[rank * group_width + lane]",
                "asc_syncthreads();",
                "tile_counts[index] += chunk_base"):
            self.assertIn(marker, prefix)
        self.assertIn("tile_count_u32 >= world_size * group_width", prefix)
        self.assertIn("destination_counts[rank] = rank_prefix", prefix)
        branch_index = prefix.index("\n    if (use_parallel_prefix)")
        first_barrier = prefix.index("asc_syncthreads();")
        self.assertLess(first_barrier, branch_index)

        copy_kernel_begin = source.index(
            "__global__ __vector__ void dispatch_copy_kernel")
        copy_kernel_end = source.index("\n}", copy_kernel_begin)
        copy_kernel = source[copy_kernel_begin:copy_kernel_end]
        self.assertIn(
            "std::uint32_t consumer_tile_bytes, "
            "std::uint32_t parallel_prefix,\n    CoreTiling tiling)",
            copy_kernel,
        )
        epilogue_launcher = source[source.index(
            'extern "C" int deep_ep_ascend_launch_dispatch_epilogue'):]
        self.assertIn(
            "arguments.consumer_tile_bytes, arguments.parallel_prefix, "
            "tiling", epilogue_launcher)

        store_begin = source.index(
            "direct_dispatch_store_scale_factor_pack")
        store_end = source.index("\n}", store_begin)
        self.assertIn("std::uint32_t", source[store_begin:store_end])
        load_begin = source.index(
            "direct_dispatch_load_scale_factor_pack")
        load_end = source.index("\n}", load_begin)
        self.assertIn("std::uint32_t", source[load_begin:load_end])

        self.assertIn("direct_dispatch_producer_record_vf", source)
        self.assertIn("direct_dispatch_epilogue_copy_outputs_vf", source)
        self.assertIn("stage == DirectDispatchStage::kFull", source)
        self.assertNotIn("AscendC::SyncAll()", source)

        record_call_begin = source.index(
            "asc_vf_call<direct_dispatch_producer_record_vf>")
        payload_call_begin = source.index(
            "direct_dispatch_producer_vector_payload_impl(",
            record_call_begin)
        record_payload_boundary = source[
            record_call_begin:payload_call_begin]
        self.assertIn("asc_sync_vec();", record_payload_boundary)
        self.assertIn(
            "asc_sync_data_barrier(mem_dsb_t::DSB_DDR);",
            record_payload_boundary)
        self.assertLess(
            record_payload_boundary.index("asc_sync_vec();"),
            record_payload_boundary.index(
                "asc_sync_data_barrier(mem_dsb_t::DSB_DDR);"))

        writer_begin = source.index(
            "DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_write_record")
        writer_body = source.index("{", writer_begin)
        writer_signature = source[writer_begin:writer_body]
        self.assertIn(
            "std::uint64_t hidden_copy_begin", writer_signature)

        hybrid_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "dispatch_producer_vf")
        hybrid_body = source.index("{", hybrid_begin)
        hybrid_signature = source[hybrid_begin:hybrid_body]
        self.assertNotIn("hidden_copy_begin", hybrid_signature)

    def test_dispatch_release_split_preserves_service_drain(self):
        """Keeps split release stages aligned with service draining."""
        kernels = (ELASTIC / "kernels.hpp").read_text()
        source = (ELASTIC / "dispatch.asc").read_text()
        begin = kernels.index(
            "direct_dispatch_release_segment(")
        end = kernels.index("\n}", begin)
        segment = kernels[begin:end]
        self.assertIn(
            "stage == DirectDispatchStage::kProducerRelease)\n"
            "        return profile_enabled ? DirectReleaseSegment::kPayload :\n"
            "                                 DirectReleaseSegment::kAll;",
            segment,
        )
        self.assertIn(
            "stage == DirectDispatchStage::kProducerReleaseBarrier)\n"
            "        return DirectReleaseSegment::kBarrier;",
            segment,
        )
        begin = source.index(
            "DEEP_EP_ASCEND_SIMT_CALLEE void direct_dispatch_producer_release_body(")
        end = source.index("\n}", begin)
        release = source[begin:end]
        self.assertIn(
            "release_segment == DirectReleaseSegment::kControl",
            release,
        )

    def test_transport_barrier_polls_gm_counter_without_mte_round_trip(self):
        """Barrier polling should use the cache-bypassing scalar GM read."""
        source = (ELASTIC / "../transport/aicore_transport_service.hpp").resolve().read_text()
        begin = source.index("template <bool ProfileEnabled = true>\n__aicore__ inline bool execute_barrier(")
        end = source.index("\n}\n\n}  // namespace detail", begin)
        barrier = source[begin:end]
        self.assertIn("aicore::load_published(signal)", barrier)
        self.assertNotIn("AscendC::DataCopyPad(signal_scratch, signal_global", barrier)

    def test_transport_barrier_scans_pending_peers_in_one_poll_loop(self):
        """Avoid serializing independent peer-arrival delays."""
        source = (ELASTIC / "../transport/aicore_transport_service.hpp").resolve().read_text()
        begin = source.index("template <bool ProfileEnabled = true>\n__aicore__ inline bool execute_barrier(")
        end = source.index("\n}\n\n}  // namespace detail", begin)
        barrier = source[begin:end]
        self.assertIn("pending_peers", barrier)
        self.assertIn("pending_peers |=", barrier)
        self.assertIn("pending_peers = observed_pending", barrier)
        self.assertIn("while (pending_peers != 0)", barrier)
        self.assertNotRegex(
            barrier,
            r"for \(std::uint32_t peer = 0;.*?while \(true\)",
        )

    def test_dispatch_barrier_precedes_epilogue_copy(self):
        """Keeps control and barrier visibility before epilogue reads."""
        kernels = (ELASTIC / "kernels.hpp").read_text()
        begin = kernels.index("direct_dispatch_pipeline(")
        end = kernels.index("\n}\n", begin)
        pipeline = kernels[begin:end]
        self.assertNotIn("DirectDispatchStage::kProducerReleaseBarrier", pipeline)
        self.assertIn("}, cpu_sync ? 10U : 13U};", pipeline)
        begin = kernels.index("direct_dispatch_profile_pipeline(")
        end = kernels.index("\n}\n", begin)
        pipeline = kernels[begin:end]
        barrier_index = pipeline.index(
            "DirectDispatchStage::kProducerReleaseBarrier")
        epilogue_index = pipeline.index(
            "DirectDispatchStage::kEpilogueAcquire")
        self.assertLess(barrier_index, epilogue_index)
        self.assertIn("}, cpu_sync ? 12U : 15U};", pipeline)
        self.assertIn(
            "if (stage == DirectDispatchStage::kProducerRelease)\n"
            "        return profile_enabled ? DirectReleaseSegment::kPayload :\n"
            "                                 DirectReleaseSegment::kAll;",
            kernels,
        )
        self.assertIn(
            "stage == DirectDispatchStage::kProducerReleaseControl)\n"
            "        return DirectReleaseSegment::kControl;",
            kernels,
        )

    def test_dispatch_parallel_expert_prefix_candidate(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        kernels = (ELASTIC / "kernels.hpp").read_text()
        buffer = (
            ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        dispatch = buffer[
            buffer.index("    dispatch(const torch::Tensor& x"):
            buffer.index("    combine(const torch::Tensor& x")
        ]

        self.assertIn("std::uint32_t parallel_prefix = 0;", kernels)
        self.assertIn(
            '"DEEP_EP_ASCEND_DISPATCH_PARALLEL_PREFIX"', dispatch)
        self.assertIn("select_dispatch_parallel_prefix_config(", dispatch)
        self.assertRegex(
            dispatch,
            r"arguments\.parallel_prefix =\s*"
            r"parallel_prefix_config\.enabled \? 1U : 0U;",
        )

        signature = (
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_parallel_prefix_vf")
        begin = source.index(signature)
        end = source.index("\n}\n", begin)
        candidate = source[begin:end]
        for marker in (
                "dispatch_simt_expert_prefix_worker_plan(",
                "const std::uint32_t local_expert = threadIdx.x",
                "tile < tile_count_u32", "tile_counts[index] = prefix",
                "unaligned_per_expert[expert] =",
                "asc_threadfence_block();", "asc_syncthreads();"):
            self.assertIn(marker, candidate)
        self.assertLess(
            candidate.index("asc_threadfence_block();"),
            candidate.index("asc_syncthreads();"),
        )
        barrier_end = candidate.index("asc_syncthreads();")
        self.assertNotIn("return;", candidate[:barrier_end])
        self.assertIn("if (threadIdx.x != 0)", candidate[barrier_end:])

        self.assertIn("direct_dispatch_epilogue_prefix_vf", source)
        prefix_launch = source[source.index(
            "(STAGE) == DirectDispatchStage::kEpilogueExpertPrefix"):]
        prefix_launch = prefix_launch[:prefix_launch.index(
            "(STAGE) == DirectDispatchStage::kEpilogueMetadata")]
        self.assertIn("parallel_prefix != 0", prefix_launch)
        self.assertIn(
            "asc_vf_call<direct_dispatch_epilogue_parallel_prefix_vf>",
            prefix_launch,
        )
        self.assertIn(
            "asc_vf_call<direct_dispatch_epilogue_prefix_vf>",
            prefix_launch,
        )

    def test_dispatch_output_copy_uses_compact_receive_domain(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        for function_name in (
                "direct_dispatch_epilogue_vector_payload_impl",
                "direct_dispatch_epilogue_copy_outputs_vf"):
            begin = source.index(f"inline void {function_name}")
            end = source.index("\n}\n", begin)
            function = source[begin:end]
            for marker in (
                    "last_source_rank", "total_records",
                    "copies_per_record", "compact_record"):
                self.assertIn(marker, function)
            self.assertIn(
                "total_records * copies_per_record", function)

        copy_begin = source.index(
            "inline void direct_dispatch_epilogue_copy_outputs_vf")
        copy_end = source.index("\n}\n", copy_begin)
        self.assertIn(
            "direct_dispatch_compact_record_coordinates(",
            source[copy_begin:copy_end])

    def test_direct_combine_launcher_consumes_stage_pipeline(self):
        """Catches keeping direct combine data stages on one AI Vector."""
        source = (ELASTIC / "combine.asc").read_text()
        launch = source[source.index(
            'extern "C" int deep_ep_ascend_launch_combine'):]
        self.assertIn("direct_combine_pipeline(", launch)
        self.assertIn("direct_combine_stage_launch(", source)
        self.assertIn("launch_direct_combine_stage(", launch)
        self.assertIn("tiling.data_launch.num_blocks == 1", launch)
        self.assertIn("expanded && !allow_multiple_reduction", launch)
        self.assertIn("DirectCombineStage::kFull", launch)
        compile_probe = (
            CORE_OPS / "core_operator_compile_probe.asc").read_text()
        self.assertIn(
            "direct_combine_pipeline_compile_probe", compile_probe)
        self.assertIn("direct_combine_stage_launch", compile_probe)

        for function_name in (
                "direct_combine_producer_record_vf",
                "direct_combine_epilogue_weights_vf"):
            begin = source.index(
                f"__simt_vf__ __launch_bounds__(512) inline void {function_name}")
            end = source.index("\n}\n", begin)
            function = source[begin:end]
            self.assertIn("direct_data_grid_stride(", function)

        reduce_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_reduce_vf")
        reduce_end = source.index("\n}\n", reduce_begin)
        reduction = source[reduce_begin:reduce_end]
        self.assertIn("direct_subgroup_grid_stride(", reduction)

    def test_direct_normal_combine_producer_uses_vector_payload_copy(self):
        """Catches restoring per-element BF16 conversion for normal records."""
        source = (ELASTIC / "combine.asc").read_text()
        vector_signature = (
            "__aicore__ inline void "
            "direct_combine_producer_vector_payload_impl")
        vector_begin = source.index(vector_signature)
        vector_end = source.index("\n}\n", vector_begin)
        vector_copy = source[vector_begin:vector_end]
        for marker in (
                "AscendC::GetBlockIdx()", "AscendC::GetBlockNum()",
                "AscendC::GlobalTensor<bfloat16_t>", "AscendC::DataCopy",
                "combine_producer_payload_copy_plan(",
                "combine_producer_tile_rank_count_offset"):
            self.assertIn(marker, vector_copy)

        record_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_producer_record_vf")
        record_end = source.index("\n}\n", record_begin)
        record = source[record_begin:record_end]
        self.assertIn("payload_plan.scalar_begin", record)
        self.assertIn("combine_reduce_expanded_lanes(", record)

        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        record_call = kernel.index(
            "asc_vf_call<direct_combine_producer_record_vf>")
        vector_call = kernel.index(
            "direct_combine_producer_vector_payload_impl(", record_call)
        boundary = kernel[record_call:vector_call]
        self.assertIn("asc_sync_vec();", boundary)
        self.assertIn(
            "asc_sync_data_barrier(mem_dsb_t::DSB_DDR);", boundary)

    def test_reduced_combine_producer_uses_opt_in_vector_reduction(self):
        """Catches rescanning top-k once per hidden element in reduced combine."""
        source = (ELASTIC / "combine.asc").read_text()
        header = (ELASTIC / "kernels.hpp").read_text()
        host = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()

        self.assertIn("expanded_vector_reduce", header)
        self.assertIn(
            '"DEEP_EP_ASCEND_COMBINE_EXPANDED_VECTOR_REDUCE"', host)
        self.assertIn(
            "select_combine_expanded_vector_reduce_config(", host)
        self.assertIn(
            "arguments.expanded_vector_reduce =", host)

        signature = (
            "__aicore__ inline void "
            "direct_combine_producer_expanded_vector_reduce_impl")
        vector_begin = source.index(signature)
        vector_end = source.index("\n}\n", vector_begin)
        vector_reduce = source[vector_begin:vector_end]
        rows = vector_reduce.index("std::int32_t input_rows[32]")
        hidden = vector_reduce.index(
            "for (std::uint32_t hidden = 0;", rows)
        self.assertLess(rows, hidden)
        for marker in (
                "AscendC::Duplicate(", "AscendC::DataCopy(",
                "AscendC::Cast(", "AscendC::Add(",
                "AscendC::RoundMode::CAST_RINT"):
            self.assertIn(marker, vector_reduce)

        record_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_producer_record_vf")
        record_end = source.index("\n}\n", record_begin)
        record = source[record_begin:record_end]
        self.assertIn(
            "combine_expanded_producer_payload_plan(", record)
        self.assertIn("expanded_payload_plan.scalar_begin", record)

        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        self.assertIn(
            "direct_combine_producer_expanded_vector_reduce_impl(", kernel)
        self.assertIn("arguments.expanded_vector_reduce", kernel)

    def test_direct_combine_local_copy_uses_opt_in_datacopy_body(self):
        """Catches restoring a byte-at-a-time copy for the local rank shard."""
        source = (ELASTIC / "combine.asc").read_text()
        header = (ELASTIC / "kernels.hpp").read_text()
        host = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()

        self.assertIn("local_copy_datacopy", header)
        self.assertIn("direct_local_placement", header)
        self.assertIn(
            '"DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY"', host)
        self.assertIn(
            '"DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT"', host)
        self.assertIn(
            "select_combine_local_copy_datacopy_config(", host)
        self.assertIn("arguments.local_copy_datacopy =", host)

        signature = (
            "__aicore__ inline void "
            "direct_combine_producer_local_copy_impl")
        copy_begin = source.index(signature)
        copy_end = source.index("\n}\n", copy_begin)
        local_copy = source[copy_begin:copy_end]
        for marker in (
                "AscendC::GetBlockIdx()", "AscendC::GetBlockNum()",
                "AscendC::GlobalTensor<std::uint8_t>",
                "AscendC::DataCopy(", "combine_local_copy_plan("):
            self.assertIn(marker, local_copy)

        tail_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_producer_local_copy_vf")
        tail_end = source.index("\n}\n", tail_begin)
        tail = source[tail_begin:tail_end]
        self.assertIn("copy_plan.scalar_begin", tail)
        self.assertIn("direct_local_placement", source)
        self.assertIn("combine_receive_offset", local_copy)

        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        datacopy_call = kernel.index(
            "direct_combine_producer_local_copy_impl<512>(")
        tail_call = kernel.index(
            "asc_vf_call<direct_combine_producer_local_copy_vf>",
            datacopy_call)
        release_call = kernel.index(
            "asc_vf_call<direct_combine_producer_release_vf>", tail_call)
        boundary = kernel[datacopy_call:release_call]
        self.assertIn("asc_sync_vec();", boundary)
        self.assertIn(
            "asc_sync_data_barrier(mem_dsb_t::DSB_DDR);", boundary)
        self.assertIn("local_copy_datacopy", boundary)
        self.assertIn("direct_local_placement == 0", kernel)
        for tile_bytes in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            self.assertIn(
                f"direct_combine_producer_local_copy_impl<{tile_bytes}>(",
                boundary)
        self.assertIn(
            "kCombineLocalCopyDefaultTileBytes = 32768", source)

    def test_direct_combine_validation_builds_slots_in_parallel(self):
        """Catches restoring the serial receive-record slot-map pass."""
        source = (ELASTIC / "combine.asc").read_text()
        self.assertIn(
            '#include "simt_api/device_atomic_functions.h"', source)

        validate_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_epilogue_validate_vf")
        validate_end = source.index("\n}\n", validate_begin)
        validate = source[validate_begin:validate_end]
        for marker in (
                "workspace_slot_offset", "auto* slots",
                "asc_atomic_cas(slots + index, -1,",
                "CombineProtocolError::kDuplicateRecord"):
            self.assertIn(marker, validate)

        reduce_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_epilogue_reduce_errors_vf")
        reduce_end = source.index("\n}\n", reduce_begin)
        reduce = source[reduce_begin:reduce_end]
        self.assertNotIn("for (int contributor_rank", reduce)
        self.assertNotIn("record_indices[logical]", reduce)
        self.assertNotIn("slots[index]", reduce)

    def test_topk_grouping_compile_probe_contract(self):
        """Catches an uncompiled ballot adapter or a native match-any dependency."""
        header = (ELASTIC / "topk_grouping.hpp").read_text()
        probe_path = CORE_OPS / "topk_grouping_probe.asc"
        self.assertTrue(probe_path.is_file(), str(probe_path))
        probe = probe_path.read_text()
        cmake = (CORE_OPS / "CMakeLists.txt").read_text()

        self.assertIn("deep_ep_ascend_topk_grouping_probe", cmake)
        self.assertIn("topk_grouping_probe.asc", cmake)
        for symbol in (
                "asc_ballot", "asc_shfl", "laneid", "lanemask_lt",
                "__popc", "__ffs"):
            self.assertIn(symbol, header)
        self.assertNotIn("match_any", header.lower())
        for fixture in (
                "all-equal", "all-distinct", "noncontiguous-duplicates",
                "inactive-lane", "partial-mask"):
            self.assertIn(fixture, probe)

    def test_direct_combine_vector_payload_compile_probe_contract(self):
        """Catches shipping an uncompiled BF16 DataCopy/vector payload path."""
        probe_path = CORE_OPS / "core_operator_compile_probe.asc"
        probe = probe_path.read_text()
        cmake = (CORE_OPS / "CMakeLists.txt").read_text()

        self.assertIn("core_operator_compile_probe.asc", cmake)
        for marker in (
                '#include "kernel_operator.h"',
                "kCombineVectorTileElements = 256",
                "direct_combine_vector_payload_compile_probe",
                "AscendC::TPipe", "AscendC::TQue", "AscendC::TBuf",
                "AscendC::GlobalTensor<bfloat16_t>",
                "AscendC::DataCopy", "AscendC::Cast", "AscendC::Add",
                "AscendC::Duplicate"):
            self.assertIn(marker, probe)

    def test_direct_combine_vector_payload_contract(self):
        """Catches losing the qualified UB payload path or its scalar tail."""
        source = (ELASTIC / "combine.asc").read_text()
        self.assertIn('#include "kernel_operator.h"', source)
        self.assertIn("kCombineVectorTileElements = 256", source)
        self.assertIn("kCombineDataCopyAlignmentElements = 16", source)

        prepare_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_epilogue_prepare_vector_slots_vf")
        prepare_end = source.index("\n}\n", prepare_begin)
        prepare = source[prepare_begin:prepare_end]
        for marker in (
                "direct_subgroup_grid_stride(", "group_topk_subgroup(",
                "topk_broadcast_owner_value(",
                "slots[logical_index] = resolved_slot"):
            self.assertIn(marker, prepare)
        self.assertNotIn("combined_x", prepare)
        self.assertNotIn("combine_receive_shard_address", prepare)

        vector_begin = source.index(
            "__aicore__ inline void direct_combine_epilogue_vector_reduce")
        vector_end = source.index("\n}\n", vector_begin)
        vector_reduce = source[vector_begin:vector_end]
        for marker in (
                "AscendC::GetBlockIdx()", "AscendC::GetBlockNum()",
                "AscendC::TPipe", "AscendC::TQue", "AscendC::TBuf",
                "AscendC::GlobalTensor<bfloat16_t>",
                "AscendC::DataCopy", "AscendC::Cast", "AscendC::Add",
                "AscendC::Duplicate", "for (int contributor_rank = 0;",
                "vector_end", "hidden < vector_end",
                "bias_0 != nullptr", "bias_1 != nullptr"):
            self.assertIn(marker, vector_reduce)
        self.assertLess(
            vector_reduce.index("for (int contributor_rank = 0;"),
            vector_reduce.index("AscendC::DataCopy(\n"
                                "                output_tile"))

        tail_signature = (
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_combine_epilogue_vector_tail_vf")
        self.assertIn(tail_signature, source)
        tail_begin = source.index(tail_signature)
        tail_end = source.index("\n}\n", tail_begin)
        tail = source[tail_begin:tail_end]
        for marker in (
                "direct_subgroup_grid_stride(", "vector_end",
                "hidden = vector_end + work.lane",
                "hidden < hidden_elements",
                "for (int contributor_rank = 0;",
                "combined_x[logical]"):
            self.assertIn(marker, tail)

        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        prepare_stage = kernel.index(
            "stage == DirectCombineStage::kEpilogueValidateReduce")
        prepare_call = kernel.index(
            "asc_vf_call<"
            "direct_combine_epilogue_prepare_vector_slots_vf>",
            prepare_stage)
        reduce_stage = kernel.index(
            "stage == DirectCombineStage::kEpilogueReduce",
            prepare_call)
        vector_call = kernel.index(
            "direct_combine_epilogue_vector_reduce_impl<", reduce_stage)
        fallback_call = kernel.index(
            "asc_vf_call<direct_combine_epilogue_reduce_vf>",
            vector_call)
        weights_stage = kernel.index(
            "stage == DirectCombineStage::kEpilogueWeights",
            fallback_call)
        tail_call = kernel.index(
            "asc_vf_call<direct_combine_epilogue_vector_tail_vf>",
            weights_stage)
        weights_call = kernel.index(
            "asc_vf_call<direct_combine_epilogue_weights_vf>",
            tail_call)
        self.assertLess(prepare_call, vector_call)
        self.assertLess(vector_call, fallback_call)
        self.assertLess(fallback_call, tail_call)
        self.assertLess(tail_call, weights_call)
        self.assertIn(
            "tiling.num_topk <= kTopkSubgroupWidth",
            kernel[prepare_stage:fallback_call])
        self.assertEqual(
            kernel.count(
                "tiling.hidden % kCombineDataCopyAlignmentElements == 0"),
            3)
        self.assertIn(
            "tiling.hidden % kCombineVectorTileElements != 0",
            kernel[weights_stage:tail_call])
        self.assertNotIn("combine_contributor_count_offset", source)
        self.assertNotIn("combine_contributor_entry_offset", source)

    def test_direct_combine_producer_uses_2048_byte_vector_tiles(self):
        """Keeps producer payload tiling independent from C6 reduction tiling."""
        source = (ELASTIC / "combine.asc").read_text()
        self.assertIn(
            "kCombineProducerVectorTileElements = 1024;", source)
        producer_begin = source.index(
            "__aicore__ inline void direct_combine_producer_vector_payload_impl")
        producer_end = source.index("\n}\n", producer_begin)
        producer = source[producer_begin:producer_end]
        for marker in (
                "kCombineProducerVectorTileElements * sizeof(bfloat16_t)",
                "hidden += kCombineProducerVectorTileElements",
                "output_global, payload_local, kCombineProducerVectorTileElements"):
            self.assertIn(marker, producer)
        self.assertNotIn(
            "payload_buffer, kCombineVectorTileElements", producer)

    def test_combine_vector_reduce_tile_selector_contract(self):
        """Catches a fixed 256-element C6 tile with no screening selector."""
        source = (ELASTIC / "combine.asc").read_text()
        header = (ELASTIC / "kernels.hpp").read_text()
        host = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        parallel = (ELASTIC / "combine_parallel.hpp").read_text()

        self.assertIn("vector_reduce_tile_elements", header)
        self.assertIn(
            '"DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE"', host)
        self.assertIn(
            "select_combine_vector_reduce_tile_config(", host)
        self.assertIn("arguments.vector_reduce_tile_elements", host)
        self.assertIn(
            "select_combine_vector_reduce_tile_config(", parallel)
        self.assertIn("TileElements", source)
        self.assertIn(
            "direct_combine_epilogue_vector_reduce_impl<\n"
            "                    kCombineCommonTopk, kCombineCommonHidden,\n"
            "                    kCombineVectorReduceDefaultTileElements>",
            source)
        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        self.assertIn("vector_reduce_tile_elements", kernel)
        self.assertIn("kCombineVectorReduceDefaultTileElements", source)

    def test_direct_combine_common_shape_specialization_contract(self):
        """Catches losing the K=8/H=7168 AOT path or dynamic fallback."""
        source = (ELASTIC / "combine.asc").read_text()
        self.assertIn("kCombineCommonTopk = 8", source)
        self.assertIn("kCombineCommonHidden = 7168", source)
        self.assertIn(
            "template <std::uint64_t StaticNumTopk,\n"
            "          std::uint64_t StaticHiddenElements,\n"
            "          std::uint32_t TileElements",
            source)
        self.assertIn(
            "direct_combine_epilogue_vector_reduce_impl<\n"
            "                    kCombineCommonTopk, kCombineCommonHidden>",
            source)
        self.assertIn(
            "direct_combine_epilogue_vector_reduce_impl<0, 0>", source)

        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        reduce_stage = kernel.index(
            "stage == DirectCombineStage::kEpilogueReduce")
        common_guard = kernel.index(
            "tiling.num_topk == kCombineCommonTopk", reduce_stage)
        common_hidden = kernel.index(
            "tiling.hidden == kCombineCommonHidden", common_guard)
        common_call = kernel.index(
            "direct_combine_epilogue_vector_reduce_impl<", common_hidden)
        dynamic_call = kernel.index(
            "direct_combine_epilogue_vector_reduce_impl<0, 0>", common_call)
        scalar_fallback = kernel.index(
            "asc_vf_call<direct_combine_epilogue_reduce_vf>", dynamic_call)
        self.assertLess(common_guard, common_hidden)
        self.assertLess(common_hidden, common_call)
        self.assertLess(common_call, dynamic_call)
        self.assertLess(dynamic_call, scalar_fallback)

    def test_direct_combine_grouping_contract(self):
        """Catches restoring a separate grouping launch or contributor table."""
        source = (ELASTIC / "combine.asc").read_text()
        self.assertIn('#include "topk_grouping.hpp"', source)
        self.assertNotIn("direct_combine_epilogue_group_vf", source)
        self.assertNotIn("combine_contributor_count_offset", source)
        self.assertNotIn("combine_contributor_entry_offset", source)

        reduce_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_reduce_vf")
        reduce_end = source.index("\n}\n", reduce_begin)
        reduce = source[reduce_begin:reduce_end]
        for marker in (
                "combined_topk_indices", "world_size", "num_experts",
                "slot_offset", "group_topk_subgroup(",
                "topk_broadcast_owner_value(", "entry_owner_mask",
                "topk_compact_owner_ordinal(",
                "slots[logical_index] = resolved_slot"):
            self.assertIn(marker, reduce)
        self.assertNotIn("CombineContributorEntry", reduce)
        self.assertIn("if (num_topk_u32 > kTopkSubgroupWidth)", reduce)
        self.assertIn("direct_data_grid_stride(", reduce)
        self.assertIn("for (int contributor_rank = 0;", reduce)
        self.assertIn("slots[token_base + lane] = owner_slot", reduce)

        ordinal_call = reduce.index("topk_compact_owner_ordinal(")
        owner_iteration = reduce.index(
            "TopkSubgroupMask remaining_owners = entry_owner_mask")
        self.assertLess(
            ordinal_call, owner_iteration,
            "each owner must calculate its rank-order ordinal before broadcast")
        for marker in (
                "broadcast_ordinal", "broadcast_rank", "broadcast_slot",
                "contributor_ranks[broadcast_ordinal]",
                "receive_slots[broadcast_ordinal]"):
            self.assertIn(marker, reduce)

        weights_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_weights_vf")
        weights_end = source.index("\n}\n", weights_begin)
        weights = source[weights_begin:weights_end]
        self.assertNotIn("candidate_lane", weights)
        self.assertIn("slots[logical]", weights)

        launch = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        reduce_errors = launch.index(
            "asc_vf_call<direct_combine_epilogue_reduce_errors_vf>")
        reduce_call = launch.index(
            "asc_vf_call<direct_combine_epilogue_reduce_vf>")
        weights_call = launch.index(
            "asc_vf_call<direct_combine_epilogue_weights_vf>")
        complete_call = launch.index(
            "asc_vf_call<direct_combine_epilogue_complete_vf>")
        self.assertLess(reduce_errors, reduce_call)
        self.assertLess(reduce_call, weights_call)
        self.assertLess(weights_call, complete_call)
        self.assertNotIn(
            "asc_vf_call<direct_combine_epilogue_group_vf>", launch)
        self.assertNotIn("DirectCombineStage::kEpilogueGroup", launch)

    def test_direct_combine_uses_staged_simt_data_paths(self):
        """Catches restoring canonical direct combine to one-thread data paths."""
        source = (ELASTIC / "combine.asc").read_text()
        kernel = source[source.index(
            "__global__ __vector__ void combine_kernel"):]
        markers = (
            "asc_vf_call<direct_combine_producer_control_vf>",
            "asc_vf_call<direct_combine_producer_plan_vf>",
            "asc_vf_call<direct_combine_producer_plan_prefix_vf>",
            "asc_vf_call<direct_combine_producer_record_vf>",
            "asc_vf_call<direct_combine_producer_local_copy_vf>",
            "asc_vf_call<direct_combine_producer_release_vf>",
            "transport::service::execute",
            "asc_vf_call<direct_combine_epilogue_acquire_vf>",
            "asc_vf_call<direct_combine_epilogue_clear_index_vf>",
            "asc_vf_call<direct_combine_epilogue_validate_vf>",
            "asc_vf_call<direct_combine_epilogue_reduce_errors_vf>",
            "asc_vf_call<direct_combine_epilogue_reduce_vf>",
            "asc_vf_call<direct_combine_epilogue_weights_vf>",
            "asc_vf_call<direct_combine_epilogue_complete_vf>",
        )
        positions = [kernel.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

        record = source[source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_producer_record_vf"):
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void direct_combine_producer_local_copy_vf")]
        reduction = source[source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_reduce_vf"):
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_weights_vf")]
        for stage in (record, reduction):
            self.assertIn("threadIdx.x", stage)
            self.assertIn("blockDim.x", stage)
        self.assertNotIn("combine_contributor_count_offset", reduction)
        self.assertNotIn("combine_contributor_entry_offset", reduction)
        self.assertIn("direct_subgroup_grid_stride(", reduction)
        self.assertIn("direct_data_grid_stride(", reduction)
        self.assertIn("kTopkSubgroupWidth", reduction)
        self.assertIn("token = work.first", reduction)
        self.assertIn("token += work.stride", reduction)
        self.assertIn("hidden = work.lane", reduction)
        self.assertIn("hidden += kTopkSubgroupWidth", reduction)
        self.assertNotIn("output_count", reduction)
        self.assertNotIn("logical / hidden_elements", reduction)
        self.assertNotIn("workspace_slot_offset", reduction)
        self.assertNotIn("combine_reduce_origin_records(", reduction)
        self.assertIn("const std::uint32_t owner_ordinal", reduction)
        self.assertIn("TopkSubgroupMask remaining_owners", reduction)
        self.assertIn("const std::uint32_t broadcast_ordinal", reduction)
        self.assertIn("const std::int32_t broadcast_rank", reduction)
        self.assertIn("const std::int32_t broadcast_slot", reduction)
        self.assertNotIn("contributor_counts[token]", reduction)
        self.assertNotIn("entry->contributor_rank", reduction)
        self.assertNotIn("entry->receive_slot", reduction)

        plan = source[source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_producer_plan_vf"):
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void direct_combine_producer_plan_prefix_vf")]
        self.assertIn("direct_block_distributed_grid_stride(", plan)
        self.assertIn("kCombineRecordsPerTile", plan)
        self.assertIn("combine_producer_tile_rank_count_offset", plan)
        self.assertIn("combine_producer_tile_error_offset", plan)
        self.assertIn("combine_direct_status_is_clean(", record)

        validate = source[source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_validate_vf"):
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_reduce_errors_vf")]
        self.assertIn("direct_block_distributed_grid_stride(", validate)
        self.assertIn("kCombineRecordsPerTile", validate)
        self.assertIn("combine_receive_tile_error_offset", validate)
        self.assertIn("workspace_slot_offset", validate)
        self.assertIn("combine_simt_receive_record_coordinates(", validate)
        self.assertIn("asc_atomic_cas(slots + index, -1,", validate)
        self.assertIn("CombineProtocolError::kDuplicateRecord", validate)

        validate_reduce = source[source.index(
            "__simt_vf__ __launch_bounds__(512) inline void direct_combine_epilogue_reduce_errors_vf"):
            source.index(
                "DEEP_EP_ASCEND_SIMT_CALLEE std::uint32_t\n"
                "topk_compact_owner_ordinal")]
        self.assertIn("combine_receive_tile_error_offset", validate_reduce)
        self.assertNotIn("combine_receive_record_index_offset", validate_reduce)
        self.assertNotIn("for (int contributor_rank", validate_reduce)
        self.assertNotIn("slots[index]", validate_reduce)

        producer_control = kernel.index(
            "stage == DirectCombineStage::kProducerControl")
        producer_plan = kernel.index(
            "stage == DirectCombineStage::kProducerPlan", producer_control)
        producer_prefix = kernel.index(
            "stage == DirectCombineStage::kProducerPlanPrefix", producer_plan)
        producer_record = kernel.index(
            "stage == DirectCombineStage::kProducerRecord", producer_prefix)
        self.assertLess(producer_control, producer_plan)
        self.assertLess(producer_plan, producer_prefix)
        self.assertLess(producer_prefix, producer_record)

        epilogue_acquire = kernel.index(
            "stage == DirectCombineStage::kEpilogueAcquire")
        epilogue_validate = kernel.index(
            "stage == DirectCombineStage::kEpilogueValidate",
            epilogue_acquire)
        epilogue_reduce = kernel.index(
            "stage == DirectCombineStage::kEpilogueValidateReduce",
            epilogue_validate)
        self.assertLess(epilogue_acquire, epilogue_validate)
        self.assertLess(epilogue_validate, epilogue_reduce)

        producer_branch = kernel[kernel.index("if (direct_parallel)"):
                                 kernel.index("transport::service::execute")]
        direct_branch, hybrid_branch = producer_branch.split("} else {", 1)
        self.assertNotIn("asc_vf_call<combine_producer_vf>", direct_branch)
        self.assertIn("asc_vf_call<combine_producer_vf>", hybrid_branch)

    def test_async_runtime_stream_event_contract(self):
        runtime = ROOT / "csrc/backends/ascend/runtime/stream_event.cpp"
        with tempfile.TemporaryDirectory() as directory:
            for testing in (0, 1):
                with self.subTest(testing=testing):
                    binary = pathlib.Path(directory) / \
                        f"async_runtime_probe_testing_{testing}"
                    compile_result = subprocess.run(
                        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                         f"-DDEEP_EP_ASCEND_TESTING={testing}",
                         f"-I{ROOT}", str(ASYNC_RUNTIME_PROBE), str(runtime),
                         "-o", str(binary)], capture_output=True, text=True,
                        check=False)
                    self.assertEqual(compile_result.returncode, 0,
                                     compile_result.stderr)
                    run_result = subprocess.run(
                        [str(binary)], capture_output=True, text=True,
                        check=False)
                    self.assertEqual(run_result.returncode, 0,
                                     run_result.stderr)

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
                rf"__simt_vf__\s+(?:__launch_bounds__\(512\)\s+)?"
                rf"inline\s+void\s+{function_name}\s*"
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
            launch_boundary = (
                f"launch_{operation}_kernel"
                if f"launch_{operation}_kernel" in source
                else f"deep_ep_ascend_launch_{operation}")
            self.assertRegex(
                source,
                rf"(?s){launch_boundary}\s*\(.*?arguments\.generation,\s*"
                rf"arguments\.timeout_cycles,",
                operation)

    def test_disabled_stage_profile_does_not_materialize_kernel_block_state(self):
        """Keeps profile-only block state out of disabled production kernels."""
        boundaries = {
            "dispatch.asc": "__global__ __vector__ void dispatch_copy_kernel",
            "combine.asc": "template <bool ProfileEnabled>\n"
                           "inline int launch_combine_kernel",
        }
        for source_name, boundary in boundaries.items():
            operation = source_name.removesuffix(".asc")
            source = (ELASTIC / source_name).read_text()
            begin = source.index(
                "template <bool ProfileEnabled>\n"
                f"__global__ __vector__ void {operation}_kernel")
            kernel = source[begin:source.index(boundary, begin)]

            self.assertGreaterEqual(
                kernel.count("if constexpr (ProfileEnabled)"), 2,
                source_name)
            self.assertEqual(kernel.count("const auto profile_block"), 2,
                             source_name)
            first_guard = kernel.index("if constexpr (ProfileEnabled)")
            first_block = kernel.index("const auto profile_block")
            last_guard = kernel.rindex("if constexpr (ProfileEnabled)")
            last_block = kernel.rindex("const auto profile_block")
            self.assertLess(first_guard, first_block, source_name)
            self.assertLess(last_guard, last_block, source_name)

    def test_remote_operator_commands_reuse_checked_team_peer(self):
        release = (ELASTIC / "release_protocol.hpp").read_text()
        for source_name, signal_name, barrier_calls in (
                ("dispatch.asc", "kDispatchReleaseSignalIndex", 4),
                ("combine.asc", "kCombineReleaseSignalIndex", 3)):
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
            self.assertEqual(source.count("transport.device_barrier("),
                             barrier_calls,
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
                r"__simt_vf__\s+(?:__launch_bounds__\(512\)\s+)?"
                r"inline\s+void\s+(\w+)\s*\((.*?)\)\s*\{",
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
            "__simt_vf__ __launch_bounds__(512) inline void combine_producer_vf")
        producer_end = sources["combine.asc"].index(
            "make_hybrid_combine_context", producer_begin)
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
            "__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf",
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
            "__simt_vf__ __launch_bounds__(512) inline void combine_producer_vf")
        producer_end = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf", producer_begin)
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
            combine.index("__simt_vf__ __launch_bounds__(512) inline void combine_producer_vf"):
            combine.index("__simt_vf__ __launch_bounds__(512) inline void hybrid_combine_return_vf")]
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
            source.index("__simt_vf__ __launch_bounds__(512) inline void combine_producer_vf"):
            source.index("__simt_vf__ __launch_bounds__(512) inline void hybrid_combine_return_vf")]
        reverse_return = source[
            source.index("__simt_vf__ __launch_bounds__(512) inline void hybrid_combine_return_vf"):
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void hybrid_combine_prepare_epilogue_vf")]
        prepare = source[
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void hybrid_combine_prepare_epilogue_vf"):
            source.index("struct CombineOriginDeviceRecordSource")]
        epilogue = source[
            source.index("__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf"):
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
            "__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf")
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

    def test_ascend_destroy_binding_releases_gil_without_changing_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            (directory / "pybind11").mkdir()
            (directory / "pybind11/pybind11.h").write_text(
                BINDING_PYBIND11_HEADER)
            binary = directory / "elastic_binding_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{directory}", f"-I{ROOT}", str(ELASTIC_BINDING_PROBE),
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

    def test_dispatch_device_prefix_strategy_controls_host_split(self):
        source = (
            ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()
        dispatch = source[
            source.index("    dispatch(const torch::Tensor& x"):
            source.index("    combine(const torch::Tensor& x")
        ]

        self.assertIn(
            '"DEEP_EP_ASCEND_DISPATCH_DEVICE_PREFIX"', dispatch)
        self.assertIn("select_dispatch_device_prefix_config(", dispatch)
        self.assertRegex(
            dispatch,
            r"const bool split_dispatch = !cached_mode && do_cpu_sync &&\s+"
            r"!allow_hybrid_mode_ && !device_prefix_config\.enabled;")
        self.assertIn(
            "split_dispatch ? 0 : max_recv_tokens", dispatch)
        self.assertIn(
            "split_dispatch ? 0 :\n"
            "            (do_expand ? expanded_records : max_recv_tokens)",
            dispatch)

    def test_combine_payload_copy_plan_is_aicore_callable(self):
        """Catches calling a host-only copy-plan helper from combine.asc."""
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "combine_parallel_aicore"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 f"-I{ROOT}", str(COMBINE_PARALLEL_AICORE_CALLEE_PROBE),
                 "-o", str(binary)], capture_output=True, text=True,
                check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)
            run_result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_dispatch_parallel_layout_helpers_remain_host_callable(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "dispatch_parallel_host_probe"
            compile_result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-DDEEP_EP_ASCEND_SIMT_DEVICE=1",
                 '-D__SIMT_DEVICE_FUNCTIONS_DECL__='
                 '__attribute__((error("device-only")))',
                 f"-I{ROOT}", str(PROBE), "-o", str(binary)],
                capture_output=True, text=True, check=False)
            self.assertEqual(compile_result.returncode, 0,
                             compile_result.stderr)

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
                "__simt_vf__ __launch_bounds__(512) inline void hybrid_dispatch_prepare_epilogue_vf"):
            dispatch_source.index(
                "__simt_vf__ __launch_bounds__(512) inline void hybrid_dispatch_record_routes_vf")]
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
            dispatch_source.index("__simt_vf__ __launch_bounds__(512) inline void dispatch_epilogue_vf"):
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
                "__simt_vf__ __launch_bounds__(512) inline void hybrid_combine_prepare_epilogue_vf"):
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
            combine_source.index("__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf"):
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
            source.index("__simt_vf__ __launch_bounds__(512) inline void dispatch_producer_vf"):
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
            source.index("__simt_vf__ __launch_bounds__(512) inline void hybrid_dispatch_forward_vf"):
            source.index(
                "__simt_vf__ __launch_bounds__(512) inline void hybrid_dispatch_prepare_epilogue_vf")]
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

    def test_uncached_dispatch_has_count_and_event_epilogue_stages(self):
        source = (ELASTIC / "dispatch.asc").read_text()
        runtime = (ELASTIC / "runtime.cpp").read_text()
        buffer = (ROOT / "csrc/backends/ascend/elastic_buffer.hpp").read_text()

        self.assertIn("std::uint32_t copy_outputs", source)
        self.assertIn("if (copy_outputs == 0)\n        return;", source)
        self.assertIn("void dispatch_copy_kernel(", source)
        producer_call = source[
            source.index("asc_vf_call<dispatch_producer_vf>"):
            source.index("transport::service::execute", source.index(
                "asc_vf_call<dispatch_producer_vf>"))]
        self.assertNotIn("cpu_sync ? 0U : 1U", producer_call)
        dispatch_epilogue_call = source[
            source.index("asc_vf_call<dispatch_epilogue_vf>"):
            source.index("if (hybrid)", source.index(
                "asc_vf_call<dispatch_epilogue_vf>"))]
        self.assertIn("cpu_sync ? 0U : 1U", dispatch_epilogue_call)
        epilogue = source[source.index(
            'extern "C" int deep_ep_ascend_launch_dispatch_epilogue'):]
        self.assertIn("dispatch_copy_kernel<<<", epilogue)
        self.assertIn("aclrtGetLastError(ACL_RT_THREAD_LEVEL)", epilogue)
        self.assertIn("direct_dispatch_epilogue_pipeline()", epilogue)
        self.assertIn("if (status != 0)\n            return status;", epilogue)

        self.assertIn("launch_internal_dispatch_epilogue(", runtime)
        self.assertIn("CoreMode::kCpuSync", buffer)
        self.assertIn("CoreMode::kAsyncEvent", buffer)
        self.assertIn("elastic::launch_internal_dispatch_epilogue(", buffer)
        self.assertIn("completion_resources_->stage_dispatch_descriptor(",
                      buffer)
        self.assertIn("async_state_->publish(", buffer)

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
        kernel = source[source.index(
            "__global__ __vector__ void dispatch_kernel"):]
        ordered_markers = (
            "service::reset",
            "asc_vf_call<dispatch_producer_vf>",
            "service::execute",
            "asc_vf_call<dispatch_epilogue_vf>",
        )
        for marker in ordered_markers:
            self.assertIn(marker, kernel)
        positions = [kernel.index(marker) for marker in ordered_markers]
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
            "__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf")
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
            "__simt_vf__ __launch_bounds__(512) inline void combine_epilogue_vf")
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
        self.assertIn(
            "failure.backend_status,\n        failure.generation", source)
        failure_build = source.index("make_dispatch_protocol_failure")
        diagnostic_write = source.index(
            "failure.backend_status,\n        failure.generation")
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

    def test_direct_release_batches_all_payloads_before_controls(self):
        """Catches per-peer flushes that serialize independent publications."""
        functions = (
            ("dispatch.asc", "direct_dispatch_producer_release_body",
             "#define DEEP_EP_ASCEND_DISPATCH_RELEASE_ARGUMENTS"),
            ("combine.asc", "direct_combine_producer_release_vf",
             "hybrid_combine_return_vf"),
        )
        for filename, function_name, next_function in functions:
            source = (ELASTIC / filename).read_text()
            release = source[
                source.index(function_name):source.index(next_function)]
            first_peer_loop = release.index(
                "for (int destination_rank = 0;")
            flush = release.index("release_protocol::flush_payload(transport);")
            second_peer_loop = release.index(
                "for (int destination_rank = 0;", flush)
            payload = release.index("transport.put(", first_peer_loop)
            control_markers = (
                "release_protocol::publish_control_and_release(",
                "release_protocol::publish_control_slot_and_release(",
            )
            control_positions = [
                release.find(marker, second_peer_loop)
                for marker in control_markers
            ]
            control = min(position for position in control_positions
                          if position >= 0)
            barrier = release.index("transport.device_barrier(")
            self.assertLess(first_peer_loop, payload)
            self.assertLess(payload, flush)
            self.assertLess(flush, second_peer_loop)
            self.assertLess(second_peer_loop, control)
            self.assertLess(control, barrier)
            self.assertEqual(
                release.count("release_protocol::flush_payload(transport);"),
                2)
            terminal_flush = release.index(
                "release_protocol::flush_payload(transport);", control)
            self.assertLess(control, terminal_flush)
            self.assertLess(terminal_flush, barrier)

        combine = (ELASTIC / "combine.asc").read_text()
        fallback = combine[
            combine.index("combine_producer_vf"):
            combine.index("make_hybrid_combine_context")]
        flush = fallback.index("release_protocol::flush_payload(transport);")
        second_peer_loop = fallback.index(
            "for (int destination_rank = 0;", flush)
        self.assertIn(
            "const std::uint64_t count = record_counts[destination_rank];",
            fallback[second_peer_loop:])

        runner = (CORE_OPS / "core_operator_runner.asc").read_text()
        for case_name in ("dispatch-invalid-topk",
                          "dispatch-invalid-cache"):
            self.assertIn(case_name, runner)

    def test_cached_dispatch_validates_private_counts_and_local_prefixes(self):
        """Catches public count repair and nonlocal cached expert counting."""
        source = (ELASTIC / "dispatch.asc").read_text()
        producer_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void dispatch_producer_vf")
        producer_end = source.index("\n}\n", producer_begin)
        producer = source[producer_begin:producer_end]
        self.assertNotIn("prefix_per_rank[", producer)
        self.assertIn("local_count_address", producer)
        epilogue_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void "
            "direct_dispatch_epilogue_acquire_vf")
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
            "__simt_vf__ __launch_bounds__(512) inline void dispatch_producer_vf")
        epilogue_begin = source.index(
            "__simt_vf__ __launch_bounds__(512) inline void dispatch_epilogue_vf")
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

    def test_single_rank_dispatch_epilogue_publishes_generation(self):
        dispatch = (ELASTIC / "dispatch.asc").read_text()
        epilogue = dispatch[
            dispatch.index("__simt_vf__ __launch_bounds__(512) inline void dispatch_epilogue_vf"):
            dispatch.index("__global__ __vector__ void dispatch_kernel")]
        self.assertIn(
            "__gm__ std::uint8_t* control_base = transport_world_size == 1 ?",
            epilogue)
        self.assertIn("communication_buffer :", epilogue)

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
                "case_timeout_seconds": 45,
                "contract_checks": [
                    "literal-bf16-torch-reference",
                    "rank-qualified-failure-aggregation",
                    "per-case-process-timeout",
                    "single-process-distributed-batch",
                    "per-case-process-watchdog",
                    "finally-buffer-before-process-group-teardown",
                    "zero-global-synchronization",
                    "npu-profiler-overlap-interval",
                ],
                "distributed_launches_per_full_suite": 1,
                "event_cases": [
                    "capture-current-stream",
                    "record-failure",
                    "event-timeout",
                ],
                "fp8_async": {
                    "case_names": list(EXPECTED_FP8_ASYNC_CASES),
                    "payload_dtype": "float8_e4m3fn",
                    "scale_factor_dtypes": ["float32", "int32"],
                    "output_scale_layouts": [
                        "row-major", "column-major",
                    ],
                    "dispatch_modes": ["cached", "non-cached"],
                    "completion": "native-event",
                    "combine": "bf16-only",
                    "supported_world_sizes": [2, 4, 8],
                },
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
                    "component_diagnostic": {
                        "classifications": [
                            "unserialized-baseline",
                            "resource-contention",
                            "effective-overlap",
                        ],
                        "communication_tokens": 256,
                        "compute_calibration": {
                            "initial_iterations": 8,
                            "iteration_bounds": [1, 2048],
                            "target_range_seconds": [0.2, 0.3],
                            "target_seconds": 0.25,
                            "warmup_iterations": 1,
                        },
                        "compute_input_shape": [1, 64, 8, 32, 32],
                        "compute_variant": "aic-only-conv3d",
                        "compute_weight_shape": [64, 64, 3, 3, 3],
                        "event_wait_limit_seconds": 5.0,
                        "minimum_wall_gain": 0.05,
                        "repetitions": 1,
                        "timeout_seconds": 120,
                        "warmups": 0,
                    },
                    "communication_hidden": 4096,
                    "communication_num_sms": 1,
                    "communication_tokens": 256,
                    "compute_dilation": [1, 1, 1],
                    "compute_dtype": "bfloat16",
                    "compute_groups": 1,
                    "compute_input_shape": [1, 64, 24, 96, 96],
                    "compute_iterations": 256,
                    "compute_invocations_per_rank": 21,
                    "compute_layout": "NCDHW",
                    "compute_padding": [0, 0, 0],
                    "compute_stride": [1, 1, 1],
                    "compute_variant": "fixed-bf16-conv3d",
                    "compute_weight_shape": [64, 64, 3, 3, 3],
                    "conv3d_launches_per_rank": 5376,
                    "diagnostic_compute_iterations": 256,
                    "diagnostic_compute_shape": [4096, 4096],
                    "diagnostic_repetitions": 1,
                    "diagnostic_timeout_seconds": 150,
                    "diagnostic_tokens": [1024],
                    "diagnostic_warmups": 0,
                    "event_wait_limit_seconds": 5.0,
                    "maximum_auxiliary_aiv_ratio": 0.01,
                    "minimum_median_improvement": 0.05,
                    "profiler_interval_required": True,
                    "repetitions": 7,
                    "report_percentiles": [50, 95],
                    "warmups": 3,
                },
                "reference": "rank-gathered-literal-inputs-and-torch-ops",
                "watchdog_seconds": 45,
                "world_size": 2,
            })

    def test_uncached_async_acceptance_matrix_contract(self):
        spec = importlib.util.spec_from_file_location(
            "run_uncached_async_contract", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = tuple(
            f"uncached-{layout}-{mode}-allocate-{str(allocate).lower()}"
            for layout in ("normal", "expanded")
            for mode in ("sync", "async")
            for allocate in (False, True)
        ) + (
            "uncached-previous-event-allocate-true",
            "uncached-empty-route",
            "uncached-asymmetric-route",
            "uncached-10-generations",
            "uncached-completion-mismatch",
            "uncached-drop-event",
            "uncached-destroy-pending-retry",
        )
        self.assertEqual(module.UNCACHED_CASE_NAMES, expected)
        self.assertTrue(set(expected).issubset(module.ALL_DISTRIBUTED_CASES))
        source = ASYNC_OVERLAP.read_text()
        for marker in (
                "def _uncached_dispatch(",
                "do_cpu_sync=True",
                "do_expand=expanded",
                "async_with_compute_stream=async_mode",
                "allocate_on_comm_stream=allocate",
                "previous_event=previous",
                'suite == "uncached"'):
            self.assertIn(marker, source)

    def test_async_overlap_formal_workload_has_static_watchdog_margin(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_static_budget", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        reference_output_points = 8 * 32 * 32
        formal_output_points = 22 * 94 * 94
        reference_compute_seconds = 0.056503370
        reference_iterations = 2039
        communication_seconds = 0.282109590
        estimated_compute_seconds = (
            reference_compute_seconds *
            formal_output_points / reference_output_points *
            module.OVERLAP_COMPUTE_ITERATIONS / reference_iterations
        )
        combined_upper_bound = communication_seconds + estimated_compute_seconds
        compute_invocations = (
            2 * (module.OVERLAP_WARMUPS + module.OVERLAP_REPETITIONS) + 1)
        estimated_workload_seconds = compute_invocations * combined_upper_bound

        self.assertEqual(module.OVERLAP_COMPUTE_INPUT_SHAPE,
                         (1, 64, 24, 96, 96))
        self.assertEqual(module.OVERLAP_COMPUTE_WEIGHT_SHAPE,
                         (64, 64, 3, 3, 3))
        self.assertEqual(compute_invocations, 21)
        self.assertGreaterEqual(module.OVERLAP_REPETITIONS, 5)
        self.assertLess(estimated_workload_seconds,
                        module.CASE_TIMEOUT_SECONDS)

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
                {"ph": "X", "cat": "NPU",
                 "name": "aclnnMatmul_MatMulV3Common_MatMulV3",
                 "ts": 100.0, "dur": 40.0,
                 "args": {"Physic Stream Id": 61}},
                {"ph": "X", "cat": "NPU",
                 "name": "deep_ep::ascend::elastic::dispatch_kernel",
                 "ts": 120.0, "dur": 50.0,
                 "args": {"Physic Stream Id": 26}},
                {"ph": "X", "cat": "NPU", "name": "unrelated_kernel_a",
                 "ts": 90.0, "dur": 100.0,
                 "args": {"Physic Stream Id": 7}},
                {"ph": "X", "cat": "NPU", "name": "unrelated_kernel_b",
                 "ts": 95.0, "dur": 90.0,
                 "args": {"Physic Stream Id": 11}},
            ]}))
            overlap = module._find_npu_overlap_interval((trace,))
        self.assertEqual(overlap["overlap_us"], 20.0)
        self.assertEqual(
            overlap["compute_event"],
            "aclnnMatmul_MatMulV3Common_MatMulV3",
        )
        self.assertEqual(
            overlap["communication_event"],
            "deep_ep::ascend::elastic::dispatch_kernel",
        )
        self.assertEqual(overlap["physical_compute_stream_id"], 61)
        self.assertEqual(overlap["physical_communication_stream_id"], 26)

    def test_async_overlap_failure_cases_run_through_production_workers(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_production_failures", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertFalse(hasattr(module, "HOST_CONTRACT_CASES"))
        self.assertEqual(module.EVENT_CASES, (
            "capture-current-stream",
            "record-failure",
            "event-timeout",
        ))
        self.assertEqual(
            set(module.DISTRIBUTED_CASES),
            set(module.CASE_NAMES) - set(module.EVENT_CASES),
        )
        for case in module.EVENT_CASES:
            with self.subTest(case=case):
                command = module._case_command(case, "/tmp/traces")
                self.assertNotIn("pytest", command)
                self.assertNotIn("torch.distributed.run", command)
                self.assertEqual(command[command.index("--worker") + 1], case)
        for case in (
                "completion-mismatch", "drop-event",
                "destroy-pending-retry"):
            with self.subTest(case=case):
                command = module._case_command(case, "/tmp/traces")
                self.assertIn("torch.distributed.run", command)
                self.assertIn(case, module.DISTRIBUTED_CASES)

    def test_async_overlap_lifecycle_measurements_preserve_both_ranks(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_lifecycle_measurements", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        reports = [
            {"rank": 1, "measurements": {"local_failure": None}},
            {"rank": 0, "measurements": {"local_failure": "expected"}},
        ]
        for case in (
                "completion-mismatch", "drop-event",
                "destroy-pending-retry"):
            with self.subTest(case=case):
                self.assertEqual(
                    module._case_measurements(case, reports),
                    {"ranks": [
                        {"rank": 0, "local_failure": "expected"},
                        {"rank": 1, "local_failure": None},
                    ]},
                )

    def test_async_overlap_destroy_retry_recovers_transient_event_failure(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_destroy_retry", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fault_name = "DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT"
        original_fault = os.environ.pop(fault_name, None)
        try:
            class RetryBuffer:
                def __init__(self):
                    self.runtime = object()
                    self.destroy_calls = 0

                def destroy(self):
                    self.destroy_calls += 1
                    if self.destroy_calls == 1:
                        self.assert_fault_enabled()
                        raise RuntimeError(
                            "destroy_event failed with backend error -2")
                    self.assert_fault_disabled()
                    self.runtime = None

                @staticmethod
                def assert_fault_enabled():
                    if os.environ.get(fault_name) != "destroy_failure":
                        raise AssertionError("destroy fault was not enabled")

                @staticmethod
                def assert_fault_disabled():
                    if fault_name in os.environ:
                        raise AssertionError("destroy fault leaked into retry")

            buffer = RetryBuffer()
            worker = object.__new__(module.AsyncOverlapWorker)
            worker.buffers = [buffer]
            worker.new_buffer = lambda: buffer
            worker._seed = lambda _buffer: object()
            worker._cached_dispatch = lambda *_args, **_kwargs: (
                object(), object(), object(), object())

            measurements = module.AsyncOverlapWorker._run_destroy_pending_retry(
                worker)

            self.assertEqual(buffer.destroy_calls, 2)
            self.assertIsNone(buffer.runtime)
            self.assertIsNone(measurements["retry_failure"])
            self.assertTrue(measurements["runtime_released_after_retry"])
            self.assertEqual(worker.buffers, [])
        finally:
            os.environ.pop(fault_name, None)
            if original_fault is not None:
                os.environ[fault_name] = original_fault

    def test_async_overlap_completion_mismatch_then_drop_event_reuses_handle(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_lifecycle_sequence", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fault_name = "DEEP_EP_ASCEND_TEST_COMPLETION_FAULT"
        original_fault = os.environ.pop(fault_name, None)
        try:
            class MismatchBuffer:
                def __init__(self):
                    self.runtime = object()

                def destroy(self):
                    self.runtime = None
                    raise RuntimeError(
                        "device completion generation mismatch")

            class MismatchEvent:
                @staticmethod
                def current_stream_wait():
                    self.assertEqual(
                        os.environ.get(fault_name), "completion_mismatch")
                    raise RuntimeError(
                        "device completion generation mismatch")

            class Gather:
                @staticmethod
                def all_gather_object(reports, local, group):
                    reports[:] = [
                        local,
                        {"rank": 1, "failure": None},
                    ]

            mismatch = object.__new__(module.AsyncOverlapWorker)
            mismatch.rank = 0
            mismatch.group = object()
            mismatch.dist = Gather()
            mismatch.buffers = []
            mismatch_buffer = MismatchBuffer()

            def new_mismatch_buffer():
                mismatch.buffers.append(mismatch_buffer)
                return mismatch_buffer

            mismatch.new_buffer = new_mismatch_buffer
            mismatch._seed = lambda _buffer: object()
            mismatch._cached_dispatch = lambda *_args, **_kwargs: (
                object(), MismatchEvent(), object(), object())
            mismatch._assert_tensor = lambda *_args: None

            mismatch_measurements = \
                module.AsyncOverlapWorker._run_completion_mismatch(mismatch)
            self.assertTrue(mismatch_measurements["runtime_released"])
            self.assertIsNone(mismatch_buffer.runtime)
            self.assertEqual(mismatch.buffers, [])
            self.assertNotIn(fault_name, os.environ)

            class Handle:
                pass

            class DroppedEvent:
                pass

            drop = object.__new__(module.AsyncOverlapWorker)
            drop_buffer = object()
            drop.new_buffer = lambda: drop_buffer
            seeded_handle = Handle()
            seed_calls = []

            def seed(buffer):
                seed_calls.append(buffer)
                return seeded_handle

            drop._seed = seed
            cached_calls = []

            def cached_dispatch(buffer, handle, fixture, **options):
                cached_calls.append((buffer, handle, fixture, options))
                if len(cached_calls) == 1:
                    return object(), DroppedEvent(), object(), object()
                return object(), object(), object(), object()

            drop._cached_dispatch = cached_dispatch
            drop._assert_tensor = lambda *_args: None

            drop_measurements = \
                module.AsyncOverlapWorker._run_drop_event(drop)
            self.assertTrue(drop_measurements["event_dropped_without_wait"])
            self.assertTrue(drop_measurements["buffer_reused"])
            self.assertEqual(seed_calls, [drop_buffer])
            self.assertEqual(len(cached_calls), 2)
            self.assertIs(cached_calls[0][0], drop_buffer)
            self.assertIs(cached_calls[1][0], drop_buffer)
            self.assertIs(cached_calls[0][1], seeded_handle)
            self.assertIs(cached_calls[1][1], seeded_handle)
        finally:
            os.environ.pop(fault_name, None)
            if original_fault is not None:
                os.environ[fault_name] = original_fault

    def test_async_overlap_bounded_timeout_terminates_descendants(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_descendant_timeout", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            sentinel = pathlib.Path(directory) / "descendant-survived"
            child = (
                "import pathlib, sys, time; time.sleep(0.4); "
                "pathlib.Path(sys.argv[1]).write_text('survived')"
            )
            parent = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}, "
                f"{str(sentinel)!r}]); time.sleep(5)"
            )
            bounded = module._run_bounded(
                [sys.executable, "-c", parent], timeout_seconds=0.05)
            time.sleep(0.6)

            self.assertEqual(bounded["status"], "failed")
            self.assertFalse(sentinel.exists())

    def test_async_overlap_standalone_worker_marker_is_strict(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_worker_marker", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        prefix = "PHASE3E_WORKER_RESULT "
        valid = {
            "case": "record-failure",
            "measurements": {
                "injected_failure": "record_event failed",
                "recovery_event_waited": True,
                "global_synchronizations": 0,
            },
        }
        self.assertEqual(
            module._worker_payload(prefix + json.dumps(valid),
                                   "record-failure"),
            valid,
        )
        invalid = {
            "missing": "",
            "malformed": prefix + "{",
            "non-object": prefix + "[]",
            "duplicate": (prefix + json.dumps(valid) + "\n" +
                          prefix + json.dumps(valid)),
            "wrong-case": prefix + json.dumps({
                **valid, "case": "event-timeout"}),
            "missing-measurements": prefix + json.dumps({
                "case": "record-failure"}),
            "incomplete-measurements": prefix + json.dumps({
                "case": "record-failure",
                "measurements": {"injected_failure": "record_event failed"},
            }),
        }
        for name, stdout in invalid.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "worker result"):
                module._worker_payload(stdout, "record-failure")

    def test_async_overlap_profiler_accepts_torch_npu_list_trace(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_list_trace", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            trace = pathlib.Path(directory) / "trace.json"
            trace.write_text(json.dumps([
                {"ph": "X",
                 "name": "aclnnMatmul_MatMulV3Common_MatMulV3",
                 "ts": 100.0, "dur": 40.0, "tid": 61,
                 "args": {"Physic Stream Id": 61}},
                {"ph": "X", "name": "dispatch_kernel",
                 "ts": 120.0, "dur": 50.0, "tid": 26,
                 "args": {"Physic Stream Id": 26}},
            ]))
            overlap = module._find_npu_overlap_interval((trace,))

        self.assertEqual(overlap["overlap_us"], 20.0)
        self.assertEqual(
            overlap["compute_event"],
            "aclnnMatmul_MatMulV3Common_MatMulV3",
        )
        self.assertEqual(overlap["communication_event"], "dispatch_kernel")
        self.assertEqual(overlap["physical_compute_stream_id"], 61)
        self.assertEqual(overlap["physical_communication_stream_id"], 26)

    def test_async_overlap_profiler_rejects_ambiguous_event_family_streams(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_ambiguous_trace", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            trace = pathlib.Path(directory) / "trace.json"
            trace.write_text(json.dumps([
                {"ph": "X", "name": "MatMulV3", "ts": 100.0,
                 "dur": 40.0, "args": {"Physic Stream Id": 61}},
                {"ph": "X", "name": "MatMulV3", "ts": 110.0,
                 "dur": 40.0, "args": {"Physic Stream Id": 62}},
                {"ph": "X", "name": "dispatch_kernel", "ts": 120.0,
                 "dur": 50.0, "args": {"Physic Stream Id": 26}},
            ]))
            with self.assertRaisesRegex(
                    AssertionError,
                    "compute event family spans physical streams 61/62"):
                module._find_npu_overlap_interval((trace,))

    def test_async_overlap_timeout_output_is_normalized_to_text(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_timeout_text", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        command = [
            sys.executable,
            "-c",
            "import sys, time; "
            "print('PHASE3E_OVERLAP_SWEEP_RESULT {}', flush=True); "
            "print('partial diagnostic', file=sys.stderr, flush=True); "
            "time.sleep(5)",
        ]
        bounded = module._run_bounded(command, timeout_seconds=0.2)

        self.assertIsInstance(bounded["stdout"], str)
        self.assertIsInstance(bounded["stderr"], str)
        self.assertIn("PHASE3E_OVERLAP_SWEEP_RESULT {}", bounded["stdout"])
        self.assertIn("partial diagnostic", bounded["stderr"])
        self.assertEqual(
            module._marker_payload(
                bounded["stdout"], module.OVERLAP_SWEEP_RESULT_PREFIX),
            {},
        )

    def test_async_overlap_diagnostic_is_bounded_and_classifies_long_workload(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_diagnostic_contract", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        variants = (
            "pure-communication",
            "pure-compute",
            "combined-light",
            "post-dispatch-idle",
            "combined-heavy",
        )
        observations = []
        for variant_index, variant in enumerate(variants):
            ranks = []
            for rank in range(2):
                timed_out = variant in ("pure-compute", "combined-heavy")
                ranks.append({
                    "rank": rank,
                    "buffer_instance": f"rank-{rank}-buffer-{variant_index}",
                    "compute_stream_id": 7 + rank,
                    "communication_stream_id": 17 + rank,
                    "stage_seconds": {
                        "setup": 0.25,
                        "dispatch_return": 0.10 if variant != "pure-compute" else 0.0,
                        "compute_enqueue": 0.20 if variant != "pure-communication" else 0.0,
                        "event_wait": 5.01 if timed_out else 0.15,
                    },
                    "event_wait": {
                        "outcome": "timeout" if timed_out else "completed",
                        "bound_seconds": 5.0,
                    },
                    "queries": {
                        "comm_tail_immediate": False,
                        "comm_tail_before_wait": False,
                        "comm_tail_after_wait": not timed_out,
                        "comm_tail_after_grace": variant != "pure-compute",
                        "comm_tail_first_complete_seconds": (
                            6.4 if variant == "combined-heavy" else
                            0.2 if variant != "pure-compute" else None),
                        "compute_tail_after_grace": variant != "pure-communication",
                        "compute_tail_first_complete_seconds": (
                            6.2 if timed_out else
                            0.2 if variant != "pure-communication" else None),
                    },
                    "order": [
                        "dispatch-return",
                        "compute-enqueued",
                        "event-wait-start",
                        "event-wait-end",
                        "query-grace-end",
                    ],
                })
            observations.append({"variant": variant, "ranks": ranks})

        worker_payload = {"variants": observations}
        bounded_calls = []

        def run_bounded(command, timeout_seconds, env=None):
            bounded_calls.append((command, timeout_seconds, env))
            return {
                "status": "passed",
                "failure": None,
                "duration_seconds": 12.5,
                "exit_code": 0,
                "stdout": "PHASE3E_OVERLAP_DIAGNOSTIC_RESULT " +
                json.dumps(worker_payload) + "\n",
                "stderr": "",
            }

        run_diagnostic = getattr(module, "_run_overlap_diagnostic", None)
        self.assertIsNotNone(run_diagnostic)
        if run_diagnostic is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "diagnostic.json"
            trace_dir = pathlib.Path(directory) / "traces"
            with mock.patch.object(module, "_run_bounded", run_bounded):
                exit_code = run_diagnostic(output, trace_dir)
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(bounded_calls), 1)
        command, timeout_seconds, environment = bounded_calls[0]
        self.assertEqual(timeout_seconds, 180)
        self.assertIsNone(environment)
        self.assertIn("--overlap-diagnostic-worker", command)
        self.assertEqual(report["diagnostic"], "overlap-vs-serialized")
        self.assertEqual(report["diagnostic_timeout_seconds"], 180)
        self.assertEqual(report["classification"],
                         "workload-exceeds-event-deadline")
        self.assertEqual(
            tuple(row["variant"] for row in report["variants"]), variants)
        self.assertEqual(
            len({rank_row["buffer_instance"]
                 for row in report["variants"] for rank_row in row["ranks"]}),
            10)

    def test_async_overlap_diagnostic_classifies_long_communication_control(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_long_communication", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        outcomes = {
            "pure-communication": "timeout",
            "pure-compute": "completed",
            "combined-light": "timeout",
            "post-dispatch-idle": "completed",
            "combined-heavy": "timeout",
        }
        observations = []
        for variant in module.OVERLAP_DIAGNOSTIC_VARIANTS:
            ranks = []
            for rank in range(2):
                outcome = outcomes[variant]
                ranks.append({
                    "rank": rank,
                    "event_wait": {
                        "outcome": outcome,
                        "bound_seconds": 5.0,
                    },
                    "queries": {
                        "comm_tail_after_grace": True,
                        "comm_tail_first_complete_seconds": (
                            14.4 if variant != "pure-compute" else 0.07),
                        "compute_tail_after_grace": True,
                    },
                })
            observations.append({"variant": variant, "ranks": ranks})

        self.assertEqual(
            module._classify_overlap_diagnostic(observations),
            "workload-exceeds-event-deadline",
        )

    def test_async_overlap_threshold_failure_keeps_timing_and_profiler_evidence(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_threshold_evidence", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeTensor:
            shape = (256, 4096)

            def add(self, _value):
                return self

        class FakeBuffer:
            def dispatch(self, *_args, **_kwargs):
                return None, None, None, "handle"

        class FakeElasticBuffer:
            @staticmethod
            def get_buffer_size_hint(*_args, **_kwargs):
                return 4096

        class FakeDeepEp:
            ElasticBuffer = FakeElasticBuffer

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        tensor = FakeTensor()
        worker = module.AsyncOverlapWorker(
            None, None, FakeDist, FakeDeepEp, object(), object(), "/tmp")
        worker.new_buffer = mock.Mock(return_value=FakeBuffer())
        worker._make_overlap_communication_inputs = mock.Mock(
            return_value=(tensor, tensor))
        worker._make_formal_compute_inputs = mock.Mock(
            return_value=("conv-input", "conv-weight"))
        worker._make_overlap_inputs = mock.Mock(
            side_effect=AssertionError("formal path allocated matrix inputs"))
        worker._formal_overlap_iteration = mock.Mock(side_effect=(
            1.00, 0.99,
            1.02, 1.00,
            0.98, 0.97,
        ))
        profiler = {
            "overlap_us": 17.0,
            "compute_event": "aclnnConvolution_Conv3DV2",
            "communication_event": "dispatch",
            "compute_task_type": "KERNEL_AICORE",
            "communication_task_type": "KERNEL_AIVEC",
            "compute_interval_us": [100.0, 140.0],
            "communication_interval_us": [120.0, 170.0],
            "trace": "/tmp/overlap-rank0.json",
            "logical_compute_stream_id": 0,
            "logical_communication_stream_id": 128,
            "physical_compute_stream_id": 61,
            "physical_communication_stream_id": 26,
            "primary_compute_span_us": 1000.0,
            "auxiliary_aiv_events": [{
                "name": "Greater",
                "task_type": "KERNEL_AIVEC",
                "physical_stream_id": 61,
                "count": 1,
                "total_duration_us": 2.0,
                "first_start_us": 98.0,
                "last_end_us": 100.0,
            }],
            "auxiliary_aiv_total_duration_us": 2.0,
            "auxiliary_aiv_ratio": 0.002,
            "transdata_events": [],
            "compute_path_aic_only": False,
        }
        worker._profile_overlap = mock.Mock(return_value=profiler)

        with mock.patch.object(module, "OVERLAP_WARMUPS", 0), \
                mock.patch.object(module, "OVERLAP_REPETITIONS", 3):
            measurement = worker._run_overlap()

        worker._profile_overlap.assert_called_once()
        self.assertEqual(measurement["serialized"], {
            "median_seconds": 1.0,
            "p95_seconds": 1.018,
            "samples_seconds": [1.0, 1.02, 0.98],
        })
        self.assertEqual(measurement["overlapped"], {
            "median_seconds": 0.99,
            "p95_seconds": 0.999,
            "samples_seconds": [0.99, 1.0, 0.97],
        })
        self.assertAlmostEqual(measurement["median_improvement"], 0.01)
        self.assertEqual(measurement["profiler_overlap"], profiler)
        self.assertEqual(measurement["logical_compute_stream_id"], 0)
        self.assertEqual(measurement["logical_communication_stream_id"], 128)
        self.assertEqual(measurement["physical_compute_stream_id"], 61)
        self.assertEqual(measurement["physical_communication_stream_id"], 26)
        self.assertEqual(measurement["compute_variant"],
                         "fixed-bf16-conv3d")
        self.assertEqual(measurement["compute_input_shape"],
                         [1, 64, 24, 96, 96])
        self.assertEqual(measurement["compute_weight_shape"],
                         [64, 64, 3, 3, 3])
        self.assertEqual(measurement["compute_iterations"], 256)
        self.assertFalse(measurement["profiler_overlap"]
                         ["compute_path_aic_only"])
        self.assertIn("below 0.050000", measurement["acceptance_failure"])
        worker._profile_overlap.assert_called_once_with(
            worker.new_buffer.return_value, "handle", tensor,
            "conv-input", "conv-weight", formal=True)

    def test_async_overlap_missing_profiler_interval_keeps_stream_evidence(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_missing_profiler", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeProfile:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def export_chrome_trace(path):
                pathlib.Path(path).write_text('{"traceEvents": []}')

        class FakeProfiler:
            class ProfilerActivity:
                NPU = "npu"

            @staticmethod
            def profile(*_args, **_kwargs):
                return FakeProfile()

        class FakeStream:
            def __init__(self, stream_id):
                self.stream_id = stream_id

        class FakeNpu:
            @staticmethod
            def current_stream():
                return FakeStream(7)

        class FakeTorch:
            npu = FakeNpu()

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        class FakeBuffer:
            @staticmethod
            def get_comm_stream():
                return FakeStream(11)

        with tempfile.TemporaryDirectory() as directory:
            trace_dir = pathlib.Path(directory) / "tokens-256"
            worker = module.AsyncOverlapWorker(
                FakeTorch, type("FakeTorchNpu", (), {"profiler": FakeProfiler}),
                FakeDist, None, object(), object(), trace_dir)
            worker._overlap_iteration = mock.Mock(return_value=0.25)
            with mock.patch.object(
                    module, "_find_npu_overlap_interval",
                    side_effect=AssertionError("no positive profiler interval")):
                evidence = worker._profile_overlap(
                    FakeBuffer(), "handle", object(), object(), object())

        self.assertEqual(evidence["overlap_us"], 0.0)
        self.assertEqual(evidence["logical_compute_stream_id"], 7)
        self.assertEqual(evidence["logical_communication_stream_id"], 11)
        self.assertNotIn("physical_compute_stream_id", evidence)
        self.assertNotIn("physical_communication_stream_id", evidence)
        self.assertEqual(evidence["failure"], "no positive profiler interval")

    def test_async_overlap_sweep_keeps_timings_on_profiler_parse_failure(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_sweep_profiler_failure", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeTensor:
            shape = (256, 7168)

            def add(self, _value):
                return self

        class FakeStream:
            def __init__(self, stream_id):
                self.stream_id = stream_id

        class FakeBuffer:
            def dispatch(self, *_args, **_kwargs):
                return None, None, None, "handle", None

            @staticmethod
            def get_comm_stream():
                return FakeStream(11)

        class FakeElasticBuffer:
            @staticmethod
            def get_buffer_size_hint(*_args, **_kwargs):
                return 4096

        class FakeDeepEp:
            ElasticBuffer = FakeElasticBuffer

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        class FakeNpu:
            @staticmethod
            def current_stream():
                return FakeStream(7)

        class FakeTorch:
            npu = FakeNpu()

        class FakeProfile:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def export_chrome_trace(path):
                pathlib.Path(path).write_text(json.dumps([{
                    "ph": "X",
                    "name": "aclnnMatmul_MatMulV3Common_MatMulV3",
                    "ts": 100.0,
                    "dur": "not-a-duration",
                    "tid": 7,
                    "args": {"Physic Stream Id": 7},
                }]))

        class FakeProfiler:
            class ProfilerActivity:
                NPU = "npu"

            @staticmethod
            def profile(*_args, **_kwargs):
                return FakeProfile()

        tensor = FakeTensor()
        worker = module.AsyncOverlapWorker(
            FakeTorch, type("FakeTorchNpu", (), {"profiler": FakeProfiler}),
            FakeDist, FakeDeepEp, object(), object(), "/tmp")
        worker.new_buffer = mock.Mock(return_value=FakeBuffer())
        worker._make_overlap_inputs = mock.Mock(
            return_value=(tensor, tensor, tensor, tensor))
        worker._communication_iteration = mock.Mock(return_value=0.10)
        worker._compute_iteration = mock.Mock(return_value=0.20)
        worker._overlap_iteration = mock.Mock(
            side_effect=lambda *_args, overlap: 0.27 if overlap else 0.30)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(module, "OVERLAP_SWEEP_WARMUPS", 0), \
                mock.patch.object(module, "OVERLAP_SWEEP_REPETITIONS", 3):
            worker.trace_dir = pathlib.Path(directory)
            measurement = worker._run_overlap_sweep_point(256)

        self.assertEqual(measurement["communication"]["median_seconds"], 0.10)
        self.assertEqual(measurement["compute"]["median_seconds"], 0.20)
        self.assertEqual(measurement["serialized"]["median_seconds"], 0.30)
        self.assertEqual(measurement["overlapped"]["median_seconds"], 0.27)
        self.assertEqual(measurement["logical_compute_stream_id"], 7)
        self.assertEqual(measurement["logical_communication_stream_id"], 11)
        self.assertNotIn("physical_compute_stream_id", measurement)
        self.assertNotIn("physical_communication_stream_id", measurement)
        self.assertEqual(measurement["profiler_overlap"]["overlap_us"], 0.0)
        self.assertIn("ValueError", measurement["profiler_overlap"]["failure"])

    def test_async_overlap_sweep_checkpoints_each_measurement_phase(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_sweep_phases", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeTensor:
            shape = (1024, 7168)

            def add(self, _value):
                return self

        class FakeBuffer:
            def dispatch(self, *_args, **_kwargs):
                return None, None, None, "handle", None

        class FakeElasticBuffer:
            @staticmethod
            def get_buffer_size_hint(*_args, **_kwargs):
                return 4096

        class FakeDeepEp:
            ElasticBuffer = FakeElasticBuffer

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        tensor = FakeTensor()
        worker = module.AsyncOverlapWorker(
            None, None, FakeDist, FakeDeepEp, object(), object(), "/tmp")
        worker.new_buffer = mock.Mock(return_value=FakeBuffer())
        worker._make_overlap_inputs = mock.Mock(
            return_value=(tensor, tensor, tensor, tensor))
        worker._communication_iteration = mock.Mock(return_value=0.10)
        worker._compute_iteration = mock.Mock(return_value=0.20)
        worker._overlap_iteration = mock.Mock(
            side_effect=lambda *_args, overlap: 0.27 if overlap else 0.30)
        worker._profile_overlap = mock.Mock(return_value={
            "overlap_us": 17.0,
            "logical_compute_stream_id": 7,
            "logical_communication_stream_id": 11,
            "physical_compute_stream_id": 61,
            "physical_communication_stream_id": 26,
        })
        checkpoints = []

        def record_phase(phase, measurement):
            checkpoints.append((phase, json.loads(json.dumps(measurement))))

        with mock.patch.object(module, "OVERLAP_SWEEP_WARMUPS", 0), \
                mock.patch.object(module, "OVERLAP_SWEEP_REPETITIONS", 1):
            measurement = worker._run_overlap_sweep_point(
                1024, record_phase=record_phase)

        self.assertEqual(
            [(phase, measurement["phase_status"])
             for phase, measurement in checkpoints],
            [
                ("communication-only", "started"),
                ("communication-only", "completed"),
                ("compute-only", "started"),
                ("compute-only", "completed"),
                ("serialized", "started"),
                ("serialized", "completed"),
                ("overlapped", "started"),
                ("overlapped", "completed"),
                ("profiler", "started"),
                ("profiler", "completed"),
            ],
        )
        completed = {
            phase: checkpoint
            for phase, checkpoint in checkpoints
            if checkpoint["phase_status"] == "completed"
        }
        self.assertIn("communication", completed["communication-only"])
        self.assertNotIn("compute", completed["communication-only"])
        self.assertIn(
            "theoretical_maximum_improvement", completed["compute-only"])
        self.assertIn("serialized", completed["serialized"])
        self.assertIn("median_improvement", completed["overlapped"])
        self.assertEqual(completed["profiler"]["profiler_overlap"]["overlap_us"],
                         17.0)
        self.assertEqual(measurement, checkpoints[-1][1])

    def test_async_overlap_failed_result_keeps_both_rank_measurements(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_rank_evidence", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        reports = []
        for rank, improvement in ((0, 0.014373), (1, 0.012169)):
            reports.append({
                "rank": rank,
                "failure": None,
                "duration_seconds": 6.2,
                "measurements": {
                    "serialized": {
                        "median_seconds": 0.30 + rank * 0.01,
                        "p95_seconds": 0.32 + rank * 0.01,
                        "samples_seconds": [0.29, 0.30, 0.32],
                    },
                    "overlapped": {
                        "median_seconds": 0.30 * (1.0 - improvement),
                        "p95_seconds": 0.31,
                        "samples_seconds": [0.28, 0.30, 0.31],
                    },
                    "median_improvement": improvement,
                    "minimum_median_improvement": 0.05,
                    "profiler_overlap": {
                        "overlap_us": 9.0 + rank,
                        "logical_compute_stream_id": 7 + rank,
                        "logical_communication_stream_id": 17 + rank,
                        "physical_compute_stream_id": 61,
                        "physical_communication_stream_id": 26,
                    },
                    "acceptance_failure":
                        f"overlap median improvement {improvement:.6f} "
                        "is below 0.050000",
                },
            })

        measurements = module._case_measurements(
            "overlap-vs-serialized", reports)
        failures = module._measurement_failures(
            "overlap-vs-serialized", reports)

        self.assertEqual([row["rank"] for row in measurements["ranks"]],
                         [0, 1])
        self.assertEqual(
            measurements["ranks"][0]["profiler_overlap"]
            ["physical_communication_stream_id"],
            26,
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(any(
            "rank 0: overlap median improvement 0.014373" in failure
            for failure in failures))
        self.assertTrue(any(
            "rank 1: overlap median improvement 0.012169" in failure
            for failure in failures))

    def test_async_overlap_sweep_is_bounded_fresh_and_classifies_ratio(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_sweep_contract", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        point_inputs = ((1024, 0.12, 0.060, 18.0),)
        points = []
        for point_index, (tokens, theoretical, observed, overlap_us) in \
                enumerate(point_inputs):
            ranks = []
            for rank in range(2):
                profiler = {
                    "overlap_us": overlap_us,
                    "logical_compute_stream_id": 7 + rank,
                    "logical_communication_stream_id": 17 + rank,
                    "physical_compute_stream_id": 61,
                    "physical_communication_stream_id": 26,
                }
                if overlap_us == 0:
                    profiler["failure"] = "no positive profiler interval"
                ranks.append({
                    "rank": rank,
                    "tokens": tokens,
                    "buffer_instance":
                        f"rank-{rank}-sweep-buffer-{point_index}",
                    "event_wait_completed": True,
                    "communication": {
                        "median_seconds": 0.10,
                        "p95_seconds": 0.10,
                        "samples_seconds": [0.10],
                    },
                    "compute": {
                        "median_seconds": 0.20,
                        "p95_seconds": 0.20,
                        "samples_seconds": [0.20],
                    },
                    "serialized": {
                        "median_seconds": 0.30,
                        "p95_seconds": 0.30,
                        "samples_seconds": [0.30],
                    },
                    "overlapped": {
                        "median_seconds": 0.30 * (1.0 - observed),
                        "p95_seconds": 0.30 * (1.0 - observed),
                        "samples_seconds": [0.30 * (1.0 - observed)],
                    },
                    "theoretical_maximum_improvement": theoretical,
                    "median_improvement": observed,
                    "profiler_overlap": profiler,
                })
            points.append({"tokens": tokens, "ranks": ranks})

        bounded_calls = []

        def run_bounded(command, timeout_seconds, env=None):
            bounded_calls.append((command, timeout_seconds, env))
            return {
                "status": "passed",
                "failure": None,
                "duration_seconds": 19.0,
                "exit_code": 0,
                "stdout": "PHASE3E_OVERLAP_SWEEP_RESULT " +
                json.dumps({"points": points}) + "\n",
                "stderr": "",
            }

        run_sweep = getattr(module, "_run_overlap_sweep", None)
        self.assertIsNotNone(run_sweep)
        if run_sweep is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "sweep.json"
            trace_dir = pathlib.Path(directory) / "traces"
            with mock.patch.object(module, "_run_bounded", run_bounded):
                exit_code = run_sweep(output, trace_dir)
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(bounded_calls), 1)
        command, timeout_seconds, environment = bounded_calls[0]
        self.assertEqual(timeout_seconds, 150)
        self.assertIsNone(environment)
        self.assertIn("--overlap-sweep-worker", command)
        self.assertEqual(
            [point["classification"] for point in report["points"]],
            ["candidate"],
        )
        self.assertFalse(report["acceptance_eligible"])
        self.assertEqual(report["acceptance_ineligible_reason"],
                         "single-sample-diagnostic")
        self.assertEqual(report["recommended_tokens"], 1024)
        self.assertEqual(
            len({rank_row["buffer_instance"]
                 for point in report["points"]
                 for rank_row in point["ranks"]}),
            2,
        )

    def test_async_overlap_sweep_timeout_preserves_last_phase_checkpoint(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_sweep_timeout_checkpoint", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        active_point = {
            "tokens": 1024,
            "completed_phase": "compute-only",
            "active_phase": "serialized",
            "phase_status": "started",
            "ranks": [
                {
                    "rank": rank,
                    "tokens": 1024,
                    "buffer_instance": f"rank-{rank}-buffer",
                    "event_wait_completed": True,
                    "completed_phase": "compute-only",
                    "active_phase": "serialized",
                    "phase_status": "started",
                    "communication": {"median_seconds": 1.0},
                    "compute": {"median_seconds": 2.0},
                }
                for rank in range(2)
            ],
        }
        failure = {
            "status": "failed",
            "failure": "process timeout after 150s",
            "duration_seconds": 150.0,
            "exit_code": None,
            "stdout": "PHASE3E_OVERLAP_SWEEP_PHASE {}\n",
            "stderr": "",
        }

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "sweep.json"
            trace_dir = pathlib.Path(directory) / "traces"
            point_trace_dir = trace_dir / "tokens-1024"
            point_trace_dir.mkdir(parents=True)
            for row in active_point["ranks"]:
                (point_trace_dir / f"phase-rank{row['rank']}.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "tokens": 1024,
                        "completed_phase": "compute-only",
                        "active_phase": "serialized",
                        "phase_status": "started",
                        "measurement": row,
                    }))
            output.write_text(json.dumps({
                "schema_version": 1,
                "points": [],
            }))
            with mock.patch.object(
                    module, "_run_bounded", return_value=failure):
                exit_code = module._run_overlap_sweep(
                    output, trace_dir)
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["active_point"], active_point)
        self.assertEqual(report["launcher_failure"],
                         "process timeout after 150s")

    def test_async_overlap_component_diagnostic_classifies_measured_gaps(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_classification", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def rank_row(rank, serialized, overlapped):
            row = _valid_component_rank(rank)
            row["serialized_seconds"] = serialized
            row["overlapped_seconds"] = overlapped
            return row

        scenarios = (
            ("unserialized-baseline", 0.40, 0.38),
            ("resource-contention", 0.51, 0.50),
            ("effective-overlap", 0.50, 0.40),
        )
        for expected, serialized, overlapped in scenarios:
            with self.subTest(expected=expected):
                report = module._overlap_component_report([
                    rank_row(rank, serialized, overlapped)
                    for rank in range(2)
                ])
                self.assertEqual(report["classification"], expected)
                self.assertEqual(
                    [row["classification"] for row in report["ranks"]],
                    [expected, expected],
                )
                for row in report["ranks"]:
                    self.assertAlmostEqual(
                        row["measured_tolerance_seconds"]
                        ["component_sum"],
                        0.025,
                    )
                    self.assertAlmostEqual(
                        row["measured_tolerance_seconds"]
                        ["serialized"],
                        serialized * 0.05,
                    )

    def test_async_overlap_component_compute_uses_only_bf16_ncdhw_conv3d(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_compute", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Conv3dResult:
            device = type("Device", (), {"type": "npu"})()

            def mul(self, *_args, **_kwargs):
                raise AssertionError("Conv3D component path called mul")

            def mul_(self, *_args, **_kwargs):
                raise AssertionError("Conv3D component path called mul_")

        class FakeFunctional:
            def __init__(self):
                self.calls = []

            def conv3d(self, input_tensor, weight, **kwargs):
                self.calls.append((input_tensor, weight, kwargs))
                return Conv3dResult()

        class FakeTorch:
            bfloat16 = "bfloat16"

            def __init__(self):
                self.ones_calls = []
                self.nn = type("FakeNn", (), {
                    "functional": FakeFunctional(),
                })()

            def ones(self, shape, **kwargs):
                value = object()
                self.ones_calls.append((shape, kwargs, value))
                return value

            @staticmethod
            def matmul(*_args, **_kwargs):
                raise AssertionError("Conv3D component path called matmul")

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        torch = FakeTorch()
        worker = module.AsyncOverlapWorker(
            torch, None, FakeDist, None, object(), "npu:0", "/tmp")

        input_tensor, weight = worker._make_component_compute_inputs()
        result = worker._component_compute(
            input_tensor, weight, iterations=3)

        self.assertEqual(torch.ones_calls, [
            ((1, 64, 8, 32, 32), {
                "dtype": "bfloat16", "device": "npu:0"}, input_tensor),
            ((64, 64, 3, 3, 3), {
                "dtype": "bfloat16", "device": "npu:0"}, weight),
        ])
        self.assertEqual(len(torch.nn.functional.calls), 3)
        self.assertTrue(all(
            call_input is input_tensor and call_weight is weight and
            kwargs == {
                "bias": None,
                "stride": (1, 1, 1),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
                "groups": 1,
            }
            for call_input, call_weight, kwargs
            in torch.nn.functional.calls))
        self.assertIsInstance(result, Conv3dResult)

    def test_async_overlap_formal_compute_uses_fixed_bf16_ncdhw_conv3d(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_formal_compute", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Conv3dResult:
            device = type("Device", (), {"type": "npu"})()

            def mul(self, *_args, **_kwargs):
                raise AssertionError("formal Conv3D path called mul")

            def mul_(self, *_args, **_kwargs):
                raise AssertionError("formal Conv3D path called mul_")

        class FakeFunctional:
            def __init__(self):
                self.calls = []

            def conv3d(self, input_tensor, weight, **kwargs):
                self.calls.append((input_tensor, weight, kwargs))
                return Conv3dResult()

        class FakeTorch:
            bfloat16 = "bfloat16"

            def __init__(self):
                self.ones_calls = []
                self.nn = type("FakeNn", (), {
                    "functional": FakeFunctional(),
                })()

            def ones(self, shape, **kwargs):
                value = object()
                self.ones_calls.append((shape, kwargs, value))
                return value

            @staticmethod
            def matmul(*_args, **_kwargs):
                raise AssertionError("formal Conv3D path called matmul")

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        torch = FakeTorch()
        worker = module.AsyncOverlapWorker(
            torch, None, FakeDist, None, object(), "npu:0", "/tmp")

        input_tensor, weight = worker._make_formal_compute_inputs()
        result = worker._formal_compute(input_tensor, weight)

        self.assertEqual(torch.ones_calls, [
            ((1, 64, 24, 96, 96), {
                "dtype": "bfloat16", "device": "npu:0"}, input_tensor),
            ((64, 64, 3, 3, 3), {
                "dtype": "bfloat16", "device": "npu:0"}, weight),
        ])
        self.assertEqual(len(torch.nn.functional.calls), 256)
        self.assertTrue(all(
            call_input is input_tensor and call_weight is weight and
            kwargs == {
                "bias": None,
                "stride": (1, 1, 1),
                "padding": (0, 0, 0),
                "dilation": (1, 1, 1),
                "groups": 1,
            }
            for call_input, call_weight, kwargs
            in torch.nn.functional.calls))
        self.assertIsInstance(result, Conv3dResult)

    def test_async_overlap_component_calibrates_with_npu_event_completion(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_calibration", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        class FakeValue:
            device = type("Device", (), {"type": "npu"})()

        class FakeEvent:
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        worker = module.AsyncOverlapWorker(
            None, None, FakeDist, None, object(), object(), "/tmp")
        worker._component_compute = mock.Mock(return_value=FakeValue())
        warmup_event = FakeEvent()
        calibration_event = FakeEvent()
        worker._record_tail_event = mock.Mock(side_effect=(
            warmup_event, calibration_event))

        with mock.patch.object(
                module.time, "monotonic", side_effect=(10.0, 10.125)):
            calibration = worker._calibrate_component_compute(
                "input", "weight")

        self.assertEqual(worker._component_compute.call_args_list, [
            mock.call("input", "weight", iterations=1),
            mock.call("input", "weight", iterations=8),
        ])
        self.assertEqual(warmup_event.synchronize_calls, 1)
        self.assertEqual(calibration_event.synchronize_calls, 1)
        self.assertEqual(calibration, {
            "completion_wait": "npu-event-synchronize",
            "initial_iterations": 8,
            "iteration_bounds": [1, 2048],
            "measured_initial_seconds": 0.125,
            "selected_iterations": 16,
            "target_range_seconds": [0.20, 0.30],
            "target_seconds": 0.25,
            "warmup_iterations": 1,
        })

    def test_async_overlap_component_profiler_requires_task_types_and_inventories_auxiliary_events(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_trace", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        events = [
            {"ph": "X", "name": "aclnnConvolution_Conv3DV2",
             "ts": 100.0, "dur": 60.0,
             "args": {"Physic Stream Id": 61,
                      "Task Type": "KERNEL_AICORE"}},
            {"ph": "X", "name": "dispatch_kernel",
             "ts": 120.0, "dur": 50.0,
             "args": {"Physic Stream Id": 26,
                      "Task Type": "KERNEL_AIVEC"}},
            {"ph": "X", "name": "Cast",
             "ts": 99.0, "dur": 5.0,
             "args": {"Physic Stream Id": 61,
                      "Task Type": "KERNEL_AIVEC"}},
            {"ph": "X", "name": "TransData_5D_to_ND",
             "ts": 105.0, "dur": 7.0,
             "args": {"Physic Stream Id": 62,
                      "Task Type": "KERNEL_AIVEC"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = pathlib.Path(directory) / "trace.json"
            trace.write_text(json.dumps({"traceEvents": events}))
            overlap = module._find_component_npu_overlap_interval((trace,))

        self.assertEqual(overlap["overlap_us"], 40.0)
        self.assertEqual(overlap["compute_task_type"], "KERNEL_AICORE")
        self.assertEqual(overlap["communication_task_type"], "KERNEL_AIVEC")
        self.assertEqual(overlap["physical_compute_stream_id"], 61)
        self.assertEqual(overlap["physical_communication_stream_id"], 26)
        self.assertEqual(
            [row["name"] for row in overlap["auxiliary_aiv_events"]],
            ["Cast", "TransData_5D_to_ND"],
        )
        self.assertEqual(
            [row["name"] for row in overlap["transdata_events"]],
            ["TransData_5D_to_ND"],
        )
        self.assertFalse(overlap["compute_path_aic_only"])

        for family, index, wrong_type in (
                ("compute", 0, "KERNEL_MIX_AIC"),
                ("communication", 1, "KERNEL_AICORE")):
            with self.subTest(family=family):
                wrong_events = json.loads(json.dumps(events[:2]))
                wrong_events[index]["args"]["Task Type"] = wrong_type
                with tempfile.TemporaryDirectory() as directory:
                    trace = pathlib.Path(directory) / "wrong-trace.json"
                    trace.write_text(json.dumps(wrong_events))
                    with self.assertRaisesRegex(
                            AssertionError, f"{family} task type"):
                        module._find_component_npu_overlap_interval((trace,))

    def test_async_overlap_formal_profiler_reports_two_rank_task_stream_and_auxiliary_evidence(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_formal_trace", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        events = [
            {"ph": "X", "name": "aclnnConvolution_Conv3DV2",
             "ts": 100.0, "dur": 60.0,
             "args": {"Physic Stream Id": 61,
                      "Task Type": "KERNEL_AICORE"}},
            {"ph": "X", "name": "aclnnConvolution_Conv3DV2",
             "ts": 240.0, "dur": 60.0,
             "args": {"Physic Stream Id": 61,
                      "Task Type": "KERNEL_AICORE"}},
            {"ph": "X", "name": "dispatch_kernel",
             "ts": 120.0, "dur": 50.0,
             "args": {"Physic Stream Id": 26,
                      "Task Type": "KERNEL_AIVEC"}},
            {"ph": "X", "name": "Greater",
             "ts": 98.0, "dur": 0.5,
             "args": {"Physic Stream Id": 61,
                      "Task Type": "KERNEL_AIVEC"}},
            {"ph": "X", "name": "ZerosLike",
             "ts": 300.0, "dur": 1.0,
             "args": {"Physic Stream Id": 61,
                      "Task Type": "KERNEL_AIVEC"}},
        ]
        rank_measurements = []
        with tempfile.TemporaryDirectory() as directory:
            for rank in range(2):
                trace = pathlib.Path(directory) / f"rank-{rank}.json"
                trace.write_text(json.dumps({"traceEvents": events}))
                profiler = module._find_formal_npu_overlap_interval((trace,))
                profiler["logical_compute_stream_id"] = rank
                profiler["logical_communication_stream_id"] = 128 + rank
                rank_measurements.append({
                    "rank": rank,
                    "measurements": {"profiler_overlap": profiler},
                })

        report = module._case_measurements(
            "overlap-vs-serialized", rank_measurements)
        self.assertEqual([row["rank"] for row in report["ranks"]], [0, 1])
        for row in report["ranks"]:
            overlap = row["profiler_overlap"]
            self.assertEqual(overlap["overlap_us"], 40.0)
            self.assertEqual(overlap["compute_task_type"], "KERNEL_AICORE")
            self.assertEqual(overlap["communication_task_type"],
                             "KERNEL_AIVEC")
            self.assertEqual(overlap["physical_compute_stream_id"], 61)
            self.assertEqual(overlap["physical_communication_stream_id"], 26)
            self.assertEqual(overlap["primary_compute_span_us"], 200.0)
            self.assertEqual(overlap["auxiliary_aiv_total_duration_us"], 1.5)
            self.assertAlmostEqual(overlap["auxiliary_aiv_ratio"], 0.0075)
            self.assertEqual(
                [event["name"] for event in overlap["auxiliary_aiv_events"]],
                ["Greater", "ZerosLike"],
            )
            self.assertFalse(overlap["compute_path_aic_only"])

    def test_async_overlap_formal_profiler_rejects_one_percent_auxiliary_aiv(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_formal_auxiliary_gate", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        failures = module._formal_profiler_failures({
            "overlap_us": 10.0,
            "auxiliary_aiv_ratio": 0.01,
        })

        self.assertEqual(len(failures), 1)
        self.assertIn("auxiliary AIV ratio", failures[0])
        self.assertIn("0.010000", failures[0])

    def test_async_overlap_component_diagnostic_is_bounded_and_preserves_stages(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_contract", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        stage_seconds = _component_stage_seconds()
        ranks = [_valid_component_rank(rank) for rank in range(2)]

        bounded_calls = []

        def run_bounded(command, timeout_seconds, env=None):
            bounded_calls.append((command, timeout_seconds, env))
            return {
                "status": "passed",
                "failure": None,
                "duration_seconds": 17.0,
                "exit_code": 0,
                "stdout": "PHASE3E_OVERLAP_COMPONENT_RESULT " +
                json.dumps({"ranks": ranks}) + "\n",
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "components.json"
            trace_dir = pathlib.Path(directory) / "traces"
            with mock.patch.object(module, "_run_bounded", run_bounded):
                exit_code = module._run_overlap_component_diagnostic(
                    output, trace_dir)
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(bounded_calls), 1)
        command, timeout_seconds, environment = bounded_calls[0]
        self.assertEqual(timeout_seconds, 120)
        self.assertIsNone(environment)
        self.assertIn("--overlap-component-diagnostic-worker", command)
        self.assertEqual(report["classification"], "effective-overlap")
        self.assertTrue(report["diagnostic_success"])
        self.assertEqual(report["compute_variant"], "aic-only-conv3d")
        self.assertEqual(report["compute_input_shape"], [1, 64, 8, 32, 32])
        self.assertEqual(report["compute_weight_shape"], [64, 64, 3, 3, 3])
        self.assertFalse(report["acceptance_eligible"])
        self.assertEqual(report["warmups"], 0)
        self.assertEqual(report["repetitions"], 1)
        self.assertEqual(
            report["ranks"][0]["compute_variant"], "aic-only-conv3d")
        self.assertAlmostEqual(report["ranks"][0]["wall_gain_ratio"], 0.10)
        self.assertEqual(report["ranks"][0]["stage_seconds"], stage_seconds)

    def test_async_overlap_component_validation_enforces_capability_gates(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_validation", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        valid = [_valid_component_rank(rank) for rank in range(2)]
        module._validate_overlap_component(valid)

        invalid_cases = (
            ("serialized baseline", "serialized_seconds", 0.40),
            ("event wait", "event_wait_duration_seconds", 5.0),
            ("wall gain", "overlapped_seconds", 0.49),
        )
        for message, field, value in invalid_cases:
            with self.subTest(field=field):
                invalid = json.loads(json.dumps(valid))
                invalid[0][field] = value
                with self.assertRaisesRegex(AssertionError, message):
                    module._validate_overlap_component(invalid)

    def test_async_overlap_component_diagnostic_checkpoints_each_phase(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_component_phases", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeTensor:
            shape = (256, 4096)

            def add(self, _value):
                return self

        class FakeBuffer:
            def dispatch(self, *_args, **kwargs):
                if kwargs.get("do_cpu_sync"):
                    return None, None, None, "handle", None
                raise AssertionError("timed dispatch should use component helper")

        class FakeElasticBuffer:
            @staticmethod
            def get_buffer_size_hint(*_args, **_kwargs):
                return 4096

        class FakeDeepEp:
            ElasticBuffer = FakeElasticBuffer

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        tensor = FakeTensor()
        worker = module.AsyncOverlapWorker(
            None, None, FakeDist, FakeDeepEp, object(), object(), "/tmp")
        actions = []
        component_buffer = FakeBuffer()
        worker.new_buffer = mock.Mock(side_effect=lambda **_kwargs: (
            actions.append("buffer"), component_buffer)[1])
        worker._make_overlap_inputs = mock.Mock(
            return_value=(tensor, tensor, tensor, tensor))
        worker._make_component_compute_inputs = mock.Mock(
            return_value=("conv-input", "conv-weight"))
        calibration = {
            "completion_wait": "npu-event-synchronize",
            "initial_iterations": 8,
            "iteration_bounds": [1, 2048],
            "measured_initial_seconds": 0.125,
            "selected_iterations": 16,
            "target_range_seconds": [0.20, 0.30],
            "target_seconds": 0.25,
            "warmup_iterations": 1,
        }
        worker._calibrate_component_compute = mock.Mock(
            side_effect=lambda *_args: (
                actions.append("calibration"), calibration)[1])
        worker._component_communication_iteration = mock.Mock(return_value={
            "duration_seconds": 0.20,
            "stage_seconds": {
                "async_dispatch_return": 0.01,
                "event_wait": 0.19,
            },
        })
        worker._component_compute_iteration = mock.Mock(return_value={
            "duration_seconds": 0.30,
            "stage_seconds": {
                "compute_enqueue": 0.20,
                "compute_completion": 0.10,
            },
        })
        worker._component_combined_iteration = mock.Mock(side_effect=[
            {
                "duration_seconds": 0.50,
                "stage_seconds": {
                    "serialized_dispatch_return": 0.20,
                    "compute_enqueue": 0.20,
                    "compute_completion": 0.10,
                },
            },
            {
                "duration_seconds": 0.49,
                "stage_seconds": {
                    "async_dispatch_return": 0.01,
                    "compute_enqueue": 0.20,
                    "event_wait": 0.19,
                    "compute_completion": 0.10,
                },
            },
        ])
        worker._profile_overlap = mock.Mock(return_value={
            "overlap_us": 100000.0,
            "logical_compute_stream_id": 0,
            "logical_communication_stream_id": 143,
            "physical_compute_stream_id": 61,
            "physical_communication_stream_id": 11,
        })
        checkpoints = []

        measurement = worker._run_overlap_component_diagnostic(
            record_phase=lambda phase, row: checkpoints.append(
                (phase, json.loads(json.dumps(row)))),
        )

        self.assertEqual(
            [(phase, row["phase_status"]) for phase, row in checkpoints],
            [
                ("calibration", "started"),
                ("calibration", "completed"),
                ("communication-only", "started"),
                ("communication-only", "completed"),
                ("compute-only", "started"),
                ("compute-only", "completed"),
                ("serialized", "started"),
                ("serialized", "completed"),
                ("overlapped", "started"),
                ("overlapped", "completed"),
                ("profiler", "started"),
                ("profiler", "completed"),
            ],
        )
        self.assertEqual(actions[:2], ["calibration", "buffer"])
        self.assertEqual(measurement["communication_only_seconds"], 0.20)
        self.assertEqual(measurement["compute_variant"], "aic-only-conv3d")
        self.assertEqual(measurement["compute_input_shape"],
                         [1, 64, 8, 32, 32])
        self.assertEqual(measurement["compute_weight_shape"],
                         [64, 64, 3, 3, 3])
        self.assertEqual(measurement["compute_iterations"], 16)
        self.assertEqual(measurement["compute_calibration"], calibration)
        self.assertEqual(measurement["compute_only_seconds"], 0.30)
        self.assertEqual(measurement["serialized_seconds"], 0.50)
        self.assertEqual(measurement["overlapped_seconds"], 0.49)
        self.assertEqual(
            measurement["serialized_dispatch_return_duration_seconds"],
            0.20,
        )
        self.assertEqual(
            measurement["async_dispatch_return_duration_seconds"], 0.01)
        self.assertEqual(measurement["event_wait_duration_seconds"], 0.19)
        self.assertEqual(measurement["compute_completion_duration_seconds"],
                         0.10)
        self.assertEqual(measurement["profiler_overlap"]["overlap_us"],
                         100000.0)
        worker._component_compute_iteration.assert_called_once_with(
            "conv-input", "conv-weight", 16)
        self.assertEqual(
            worker._component_combined_iteration.call_args_list,
            [
                mock.call(
                    component_buffer, "handle", tensor,
                    "conv-input", "conv-weight", 16, overlap=False),
                mock.call(
                    component_buffer, "handle", tensor,
                    "conv-input", "conv-weight", 16, overlap=True),
            ],
        )
        worker._profile_overlap.assert_called_once_with(
            component_buffer, "handle", tensor,
            "conv-input", "conv-weight", component=True,
            component_iterations=16)
        self.assertEqual(measurement, checkpoints[-1][1])

    def test_async_overlap_suite_checkpoints_each_completed_case(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_checkpoint", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        selected = module.EVENT_CASES[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            observed_checkpoint = []

            def run_bounded(command):
                if output.exists():
                    observed_checkpoint.append(json.loads(output.read_text()))
                case = command[command.index("--worker") + 1]
                return {
                    "status": "passed",
                    "failure": None,
                    "duration_seconds": 0.25,
                    "exit_code": 0,
                    "stdout": _async_overlap_worker_stdout(case),
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

        def run_bounded(command):
            calls.append(len(calls))
            failed = len(calls) == 2
            case = command[command.index("--worker") + 1]
            return {
                "status": "failed" if failed else "passed",
                "failure": "process exited 17" if failed else None,
                "duration_seconds": 0.25,
                "exit_code": 17 if failed else 0,
                "stdout": "" if failed else
                    _async_overlap_worker_stdout(case),
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

    def test_async_overlap_sync_guard_exempts_only_buffer_construction(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_sync_scope", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        synchronizations = []

        class FakeNpu:
            @staticmethod
            def synchronize():
                synchronizations.append("buffer construction")

        class FakeTorch:
            npu = FakeNpu()

        class FakeBuffer:
            def __init__(self, *_args, **_kwargs):
                FakeTorch.npu.synchronize()

        class FakeDeepEp:
            ElasticBuffer = FakeBuffer

        class FakeDist:
            @staticmethod
            def get_rank(_group):
                return 0

        worker = module.AsyncOverlapWorker(
            FakeTorch, None, FakeDist, FakeDeepEp, object(), object(), "/tmp")
        with module._forbid_global_sync(FakeTorch):
            worker.new_buffer()
            with self.assertRaisesRegex(
                    AssertionError, "global NPU synchronization is forbidden"):
                FakeTorch.npu.synchronize()

        self.assertEqual(synchronizations, ["buffer construction"])

    def test_fp8_suite_cli_selects_exact_cases_and_requested_world_size(self):
        spec = importlib.util.spec_from_file_location(
            "run_fp8_async_cli", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for world_size in (2, 4, 8):
            with self.subTest(world_size=world_size), \
                    tempfile.TemporaryDirectory() as directory:
                calls = []

                def run_distributed_batch(
                        cases, trace_dir, record_result,
                        requested_world_size):
                    calls.append((
                        tuple(cases), pathlib.Path(trace_dir),
                        requested_world_size))
                    for case in cases:
                        record_result({
                            "case": case,
                            "status": "passed",
                            "duration_seconds": 0.25,
                            "exit_code": 0,
                            "failure": None,
                            "measurements": {},
                        })
                    return True

                output = pathlib.Path(directory) / "report.json"
                trace_dir = pathlib.Path(directory) / "traces"
                argv = [
                    str(ASYNC_OVERLAP),
                    "--suite", "fp8",
                    "--world-size", str(world_size),
                    "--output", str(output),
                    "--trace-dir", str(trace_dir),
                ]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(
                            module, "_run_distributed_batch",
                            run_distributed_batch), \
                        contextlib.redirect_stdout(io.StringIO()):
                    exit_code = module.main()

                self.assertEqual(exit_code, 0)
                self.assertEqual(calls, [(
                    EXPECTED_FP8_ASYNC_CASES, trace_dir, world_size)])

    def test_fp8_batch_command_launches_requested_world_size(self):
        spec = importlib.util.spec_from_file_location(
            "run_fp8_async_command", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        command = module._distributed_batch_command(
            EXPECTED_FP8_ASYNC_CASES, pathlib.Path("/tmp/fp8-traces"), 8)

        self.assertIn("--nproc-per-node=8", command)
        world_size_index = command.index("--world-size")
        self.assertEqual(command[world_size_index + 1], "8")
        batch_index = command.index("--batch-worker")
        self.assertEqual(
            tuple(command[batch_index + 1:]), EXPECTED_FP8_ASYNC_CASES)

    def test_fp8_declared_cases_route_to_their_behavior_handlers(self):
        spec = importlib.util.spec_from_file_location(
            "run_fp8_async_routing", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        matrix_cases = EXPECTED_FP8_ASYNC_CASES[:8]
        routes = {
            **{
                case: ("_run_fp8_matrix_case", (case,))
                for case in matrix_cases
            },
            "fp8-previous-event-allocate-true": (
                "_run_fp8_previous_event", ()),
            "fp8-empty-route": (
                "_run_fp8_route_case", ("empty-input",)),
            "fp8-asymmetric-route": (
                "_run_fp8_route_case", ("asymmetric-routing",)),
            "fp8-10-generations": ("_run_fp8_generations", ()),
            "fp8-cached-representation-changes": (
                "_run_fp8_representation_changes", ()),
            "fp8-completion-mismatch": (
                "_run_fp8_completion_mismatch", ()),
            "fp8-drop-event": ("_run_fp8_drop_event", ()),
            "fp8-destroy-pending-retry": (
                "_run_fp8_destroy_pending_retry", ()),
        }
        handler_names = {handler for handler, _args in routes.values()}

        class FakeNpu:
            @staticmethod
            def synchronize():
                return None

        class FakeTorch:
            npu = FakeNpu()

        self.assertEqual(tuple(routes), EXPECTED_FP8_ASYNC_CASES)
        for case, (expected_handler, expected_args) in routes.items():
            with self.subTest(case=case):
                worker = object.__new__(module.AsyncOverlapWorker)
                worker.torch = FakeTorch
                handlers = {}
                for handler_name in handler_names:
                    handler = mock.Mock(
                        name=handler_name, return_value=handler_name)
                    handlers[handler_name] = handler
                    setattr(worker, handler_name, handler)

                result = worker.run(case)

                self.assertEqual(result, expected_handler)
                handlers[expected_handler].assert_called_once_with(
                    *expected_args)
                for handler_name, handler in handlers.items():
                    if handler_name != expected_handler:
                        handler.assert_not_called()

    def test_async_overlap_full_suite_uses_one_distributed_launcher(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_batch_launch", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        distributed = module.CASE_NAMES[1:3]
        selected = (*distributed, "record-failure")
        batch_calls = []
        world_sizes = []
        bounded_calls = []

        def run_distributed_batch(
                cases, _trace_dir, record_result, world_size):
            batch_calls.append(tuple(cases))
            world_sizes.append(world_size)
            for case in cases:
                record_result({
                    "case": case,
                    "status": "passed",
                    "duration_seconds": 0.25,
                    "exit_code": 0,
                    "failure": None,
                    "measurements": {},
                })
            return True

        def run_bounded(command):
            bounded_calls.append(True)
            case = command[command.index("--worker") + 1]
            return {
                "status": "passed",
                "failure": None,
                "duration_seconds": 0.25,
                "exit_code": 0,
                "stdout": _async_overlap_worker_stdout(case),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            with mock.patch.object(module, "CASE_NAMES", selected), \
                    mock.patch.object(
                        module, "_run_distributed_batch",
                        run_distributed_batch, create=True), \
                    mock.patch.object(module, "_run_bounded", run_bounded), \
                    contextlib.redirect_stdout(io.StringIO()):
                exit_code = module._run_suite(
                    "full", output, pathlib.Path(directory) / "traces")

        self.assertEqual(exit_code, 0)
        self.assertEqual(batch_calls, [distributed])
        self.assertEqual(world_sizes, [2])
        self.assertEqual(len(bounded_calls), 1)

    def test_async_overlap_uncached_suite_uses_one_distributed_launcher(self):
        spec = importlib.util.spec_from_file_location(
            "run_uncached_async_batch_launch", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        selected = module.UNCACHED_CASE_NAMES[:3]
        batch_calls = []
        world_sizes = []

        def run_distributed_batch(
                cases, _trace_dir, record_result, world_size):
            batch_calls.append(tuple(cases))
            world_sizes.append(world_size)
            for case in cases:
                record_result({
                    "case": case,
                    "status": "passed",
                    "duration_seconds": 0.25,
                    "exit_code": 0,
                    "failure": None,
                    "measurements": {},
                })
            return True

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            with mock.patch.object(
                    module, "UNCACHED_CASE_NAMES", selected), \
                    mock.patch.object(
                        module, "_run_distributed_batch",
                        run_distributed_batch), \
                    mock.patch.object(
                        module, "_run_bounded",
                        side_effect=AssertionError(
                            "uncached case used standalone launcher")), \
                    contextlib.redirect_stdout(io.StringIO()):
                exit_code = module._run_suite(
                    "uncached", output, pathlib.Path(directory) / "traces")

        self.assertEqual(exit_code, 0)
        self.assertEqual(batch_calls, [selected])
        self.assertEqual(world_sizes, [2])

    def test_async_overlap_distributed_batch_checkpoints_each_case(self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_batch_checkpoint", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        selected = module.CASE_NAMES[1:3]
        observed_checkpoint = []

        def run_distributed_batch(
                cases, _trace_dir, record_result, world_size):
            self.assertEqual(world_size, 2)
            for index, case in enumerate(cases):
                record_result({
                    "case": case,
                    "status": "passed",
                    "duration_seconds": 0.25,
                    "exit_code": 0,
                    "failure": None,
                    "measurements": {},
                })
                if index == 0:
                    observed_checkpoint.append(
                        json.loads(output.read_text()))
            return True

        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            with mock.patch.object(module, "CASE_NAMES", selected), \
                    mock.patch.object(
                        module, "_run_distributed_batch",
                        run_distributed_batch, create=True), \
                    mock.patch.object(module, "_run_bounded"), \
                    contextlib.redirect_stdout(io.StringIO()):
                module._run_suite(
                    "full", output, pathlib.Path(directory) / "traces")

        self.assertEqual(
            [row["case"] for row in observed_checkpoint[0]["results"]],
            [selected[0]],
        )

    def test_async_overlap_distributed_case_watchdog_is_external_and_bounded(
            self):
        spec = importlib.util.spec_from_file_location(
            "run_async_overlap_batch_watchdog", ASYNC_OVERLAP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        run_streaming_batch = getattr(module, "_run_streaming_batch", None)
        self.assertIsNotNone(run_streaming_batch)
        if run_streaming_batch is None:
            return

        case = "cached-dispatch-sync-allocate-false"
        child = (
            "import json, time; "
            f"print('{module.CASE_START_PREFIX} ' + "
            f"json.dumps({{'case': '{case}'}}), flush=True); "
            "time.sleep(5)"
        )
        rows = []
        started = __import__("time").monotonic()
        passed = run_streaming_batch(
            [sys.executable, "-c", child], (case,), rows.append,
            timeout_seconds=0.2)
        elapsed = __import__("time").monotonic() - started

        self.assertFalse(passed)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(rows[0]["case"], case)
        self.assertEqual(
            rows[0]["failure"], "case watchdog timeout after 0.2s")

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
            "vector-hidden-256",
            "vector-tail-hidden-272-two-bias",
            "specialized-k8-hidden7168",
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
                    "vector-payload-and-tail",
                    "common-shape-specialization",
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
                    "common_shape": {
                        "allow_multiple_reduction": True,
                        "has_duplicate_contributor": True,
                        "has_inactive_lane": True,
                        "hidden": 7168,
                        "num_experts": 8,
                        "num_topk": 8,
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
                    "vector_payload": {
                        "pure_vector": {"bias_count": 0, "hidden": 256},
                        "vector_tail": {"bias_count": 2, "hidden": 272},
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

    def test_two_rank_combine_vector_payload_cases_contract(self):
        """Catches losing pure-vector and vector-plus-tail device coverage."""
        result = subprocess.run(
            ["python3", str(TWO_RANK_COMBINE), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = json.loads(result.stdout)
        self.assertIn("vector-hidden-256", contract["case_names"])
        self.assertIn("vector-tail-hidden-272-two-bias", contract["case_names"])
        self.assertIn(
            "vector-payload-and-tail",
            contract["contract_checks"])
        self.assertEqual(
            contract["behavior_fixtures"]["vector_payload"],
            {
                "pure_vector": {"bias_count": 0, "hidden": 256},
                "vector_tail": {"bias_count": 2, "hidden": 272},
            })

    def test_two_rank_combine_common_shape_case_contract(self):
        """Catches losing public K=8/H=7168 specialization coverage."""
        result = subprocess.run(
            ["python3", str(TWO_RANK_COMBINE), "--contract"],
            cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = json.loads(result.stdout)
        self.assertIn("specialized-k8-hidden7168", contract["case_names"])
        self.assertIn(
            "common-shape-specialization",
            contract["contract_checks"])
        self.assertEqual(
            contract["behavior_fixtures"]["common_shape"],
            {
                "allow_multiple_reduction": True,
                "has_duplicate_contributor": True,
                "has_inactive_lane": True,
                "hidden": 7168,
                "num_experts": 8,
                "num_topk": 8,
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
