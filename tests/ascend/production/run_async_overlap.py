import argparse
import ast
import copy
import gc
import json
import math
import os
import pathlib
import selectors
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import timedelta


ROOT = pathlib.Path(__file__).resolve().parents[3]
CASE_TIMEOUT_SECONDS = 30
WORLD_SIZE = 2
NUM_EXPERTS = 4
NUM_TOPK = 2
CAPACITY = 4
HIDDEN = 4

OVERLAP_WARMUPS = 3
OVERLAP_REPETITIONS = 7
OVERLAP_COMMUNICATION_TOKENS = 256
OVERLAP_COMMUNICATION_HIDDEN = 4096
OVERLAP_COMPUTE_VARIANT = "fixed-bf16-conv3d"
OVERLAP_COMPUTE_INPUT_SHAPE = (1, 64, 24, 96, 96)
OVERLAP_COMPUTE_WEIGHT_SHAPE = (64, 64, 3, 3, 3)
OVERLAP_COMPUTE_STRIDE = (1, 1, 1)
OVERLAP_COMPUTE_PADDING = (0, 0, 0)
OVERLAP_COMPUTE_DILATION = (1, 1, 1)
OVERLAP_COMPUTE_GROUPS = 1
OVERLAP_COMPUTE_ITERATIONS = 256
OVERLAP_EVENT_WAIT_LIMIT_SECONDS = 5.0
OVERLAP_MAX_AUXILIARY_AIV_RATIO = 0.01
OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT = 0.05
OVERLAP_DIAGNOSTIC_TIMEOUT_SECONDS = 180
OVERLAP_DIAGNOSTIC_GRACE_SECONDS = 10.0
OVERLAP_DIAGNOSTIC_IDLE_SECONDS = 6.0
OVERLAP_SWEEP_TIMEOUT_SECONDS = 150
OVERLAP_SWEEP_TOKENS = (1024,)
OVERLAP_SWEEP_WARMUPS = 0
OVERLAP_SWEEP_REPETITIONS = 1
OVERLAP_DIAGNOSTIC_COMPUTE_SHAPE = (4096, 4096)
OVERLAP_DIAGNOSTIC_COMPUTE_ITERATIONS = 256
OVERLAP_COMPONENT_TIMEOUT_SECONDS = 120
OVERLAP_COMPONENT_WARMUPS = 0
OVERLAP_COMPONENT_REPETITIONS = 1
OVERLAP_COMPONENT_COMPUTE_VARIANT = "aic-only-conv3d"
OVERLAP_COMPONENT_INPUT_SHAPE = (1, 64, 8, 32, 32)
OVERLAP_COMPONENT_WEIGHT_SHAPE = (64, 64, 3, 3, 3)
OVERLAP_COMPONENT_CALIBRATION_WARMUP_ITERATIONS = 1
OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS = 8
OVERLAP_COMPONENT_MIN_ITERATIONS = 1
OVERLAP_COMPONENT_MAX_ITERATIONS = 2048
OVERLAP_COMPONENT_TARGET_SECONDS = 0.25
OVERLAP_COMPONENT_TARGET_RANGE_SECONDS = (0.20, 0.30)
OVERLAP_COMPONENT_EVENT_WAIT_LIMIT_SECONDS = 5.0
OVERLAP_COMPONENT_CLASSIFICATIONS = (
    "unserialized-baseline",
    "resource-contention",
    "effective-overlap",
)
OVERLAP_DIAGNOSTIC_VARIANTS = (
    "pure-communication",
    "pure-compute",
    "combined-light",
    "post-dispatch-idle",
    "combined-heavy",
)

CASE_NAMES = (
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
)

EVENT_CASES = (
    "capture-current-stream",
    "record-failure",
    "event-timeout",
)

CASE_START_PREFIX = "PHASE3E_CASE_START"
CASE_RESULT_PREFIX = "PHASE3E_CASE_RESULT"
OVERLAP_DIAGNOSTIC_RESULT_PREFIX = "PHASE3E_OVERLAP_DIAGNOSTIC_RESULT"
OVERLAP_SWEEP_RESULT_PREFIX = "PHASE3E_OVERLAP_SWEEP_RESULT"
OVERLAP_COMPONENT_RESULT_PREFIX = "PHASE3E_OVERLAP_COMPONENT_RESULT"
WORKER_RESULT_PREFIX = "PHASE3E_WORKER_RESULT"

STANDALONE_MEASUREMENT_FIELDS = {
    "capture-current-stream": {
        "repeated_waits", "global_synchronizations"},
    "record-failure": {
        "injected_failure", "recovery_event_waited",
        "global_synchronizations"},
    "event-timeout": {
        "injected_failure", "same_event_recovered",
        "recovery_event_waited", "global_synchronizations"},
}

DISTRIBUTED_CASES = tuple(
    case for case in CASE_NAMES if case not in EVENT_CASES)

MATRIX_GROUPS = (
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
)

REGULAR_FIXTURE = {
    "payloads": (
        ((1, 2, 3, 4), (5, 6, 7, 8)),
        ((9, 10, 11, 12), (13, 14, 15, 16)),
    ),
    "routes": (
        ((2, -1), (0, -1)),
        ((1, -1), (3, -1)),
    ),
}

EMPTY_FIXTURE = {"payloads": ((), ()), "routes": ((), ())}

ASYMMETRIC_FIXTURE = {
    "payloads": (
        ((21, 22, 23, 24), (25, 26, 27, 28), (29, 30, 31, 32)),
        ((121, 122, 123, 124),),
    ),
    "routes": (
        ((2, -1), (0, -1), (3, -1)),
        ((1, -1),),
    ),
}


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _call_name(call):
    node = call.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _contract():
    with open(__file__, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=__file__)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = [_call_name(node) for node in ast.walk(tree)
             if isinstance(node, ast.Call)]
    bounded_calls = [_call_name(node) for node in ast.walk(functions["_run_bounded"])
                     if isinstance(node, ast.Call)]
    aggregate_calls = [
        _call_name(node) for node in ast.walk(
            functions["_run_distributed_batch_worker"])
        if isinstance(node, ast.Call)
    ]
    teardown = functions["_run_distributed_batch_worker"]
    teardown_finally = [node for node in ast.walk(teardown)
                        if isinstance(node, ast.Try) and node.finalbody]
    reference_calls = [
        _call_name(node) for node in ast.walk(functions["_literal_reference"])
        if isinstance(node, ast.Call)
    ]

    _check(set(EVENT_CASES).issubset(CASE_NAMES),
           "event suite contains an unregistered case")
    _check("subprocess.Popen" in bounded_calls,
           "per-case subprocess execution is missing")
    _check("dist.all_gather_object" in aggregate_calls,
           "rank failure aggregation is missing")
    _check(teardown_finally,
           "buffer/process-group teardown is not protected by finally")
    _check("worker.destroy_buffers" in calls and
           "dist.destroy_process_group" in calls,
           "buffer-before-process-group teardown is incomplete")
    _check("dist.all_gather_object" in reference_calls and
           "torch.tensor" in reference_calls,
           "literal Torch reference path is incomplete")
    _check("torch.npu.synchronize" not in calls,
           "global NPU synchronization is forbidden")
    _check("_find_npu_overlap_interval" in calls,
           "NPU profiler interval validation is missing")

    return {
        "case_names": list(CASE_NAMES),
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
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
        "event_cases": list(EVENT_CASES),
        "full_cases": list(CASE_NAMES),
        "matrix_groups": list(MATRIX_GROUPS),
        "overlap": {
            "component_diagnostic": {
                "classifications": list(OVERLAP_COMPONENT_CLASSIFICATIONS),
                "communication_tokens": OVERLAP_COMMUNICATION_TOKENS,
                "compute_calibration": {
                    "initial_iterations":
                        OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS,
                    "iteration_bounds": [
                        OVERLAP_COMPONENT_MIN_ITERATIONS,
                        OVERLAP_COMPONENT_MAX_ITERATIONS,
                    ],
                    "target_range_seconds":
                        list(OVERLAP_COMPONENT_TARGET_RANGE_SECONDS),
                    "target_seconds": OVERLAP_COMPONENT_TARGET_SECONDS,
                    "warmup_iterations":
                        OVERLAP_COMPONENT_CALIBRATION_WARMUP_ITERATIONS,
                },
                "compute_input_shape": list(OVERLAP_COMPONENT_INPUT_SHAPE),
                "compute_variant": OVERLAP_COMPONENT_COMPUTE_VARIANT,
                "compute_weight_shape":
                    list(OVERLAP_COMPONENT_WEIGHT_SHAPE),
                "event_wait_limit_seconds":
                    OVERLAP_COMPONENT_EVENT_WAIT_LIMIT_SECONDS,
                "minimum_wall_gain": OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
                "repetitions": OVERLAP_COMPONENT_REPETITIONS,
                "timeout_seconds": OVERLAP_COMPONENT_TIMEOUT_SECONDS,
                "warmups": OVERLAP_COMPONENT_WARMUPS,
            },
            "communication_hidden": OVERLAP_COMMUNICATION_HIDDEN,
            "communication_num_sms": 1,
            "communication_tokens": OVERLAP_COMMUNICATION_TOKENS,
            "compute_dilation": list(OVERLAP_COMPUTE_DILATION),
            "compute_dtype": "bfloat16",
            "compute_groups": OVERLAP_COMPUTE_GROUPS,
            "compute_input_shape": list(OVERLAP_COMPUTE_INPUT_SHAPE),
            "compute_iterations": OVERLAP_COMPUTE_ITERATIONS,
            "compute_invocations_per_rank":
                2 * (OVERLAP_WARMUPS + OVERLAP_REPETITIONS) + 1,
            "compute_layout": "NCDHW",
            "compute_padding": list(OVERLAP_COMPUTE_PADDING),
            "compute_stride": list(OVERLAP_COMPUTE_STRIDE),
            "compute_variant": OVERLAP_COMPUTE_VARIANT,
            "compute_weight_shape": list(OVERLAP_COMPUTE_WEIGHT_SHAPE),
            "conv3d_launches_per_rank": OVERLAP_COMPUTE_ITERATIONS * (
                2 * (OVERLAP_WARMUPS + OVERLAP_REPETITIONS) + 1),
            "diagnostic_compute_iterations":
                OVERLAP_DIAGNOSTIC_COMPUTE_ITERATIONS,
            "diagnostic_compute_shape":
                list(OVERLAP_DIAGNOSTIC_COMPUTE_SHAPE),
            "diagnostic_repetitions": OVERLAP_SWEEP_REPETITIONS,
            "diagnostic_timeout_seconds": OVERLAP_SWEEP_TIMEOUT_SECONDS,
            "diagnostic_tokens": list(OVERLAP_SWEEP_TOKENS),
            "diagnostic_warmups": OVERLAP_SWEEP_WARMUPS,
            "event_wait_limit_seconds": OVERLAP_EVENT_WAIT_LIMIT_SECONDS,
            "maximum_auxiliary_aiv_ratio":
                OVERLAP_MAX_AUXILIARY_AIV_RATIO,
            "minimum_median_improvement":
                OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
            "profiler_interval_required": True,
            "repetitions": OVERLAP_REPETITIONS,
            "report_percentiles": [50, 95],
            "warmups": OVERLAP_WARMUPS,
        },
        "reference": "rank-gathered-literal-inputs-and-torch-ops",
        "watchdog_seconds": CASE_TIMEOUT_SECONDS,
        "world_size": WORLD_SIZE,
    }


def _text_output(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _run_bounded(command, timeout_seconds=CASE_TIMEOUT_SECONDS, env=None):
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        return {
            "status": "failed",
            "failure": f"process timeout after {timeout_seconds}s",
            "duration_seconds": time.monotonic() - started,
            "exit_code": None,
            "stdout": _text_output(stdout or error.stdout),
            "stderr": _text_output(stderr or error.stderr),
        }
    return {
        "status": "passed" if process.returncode == 0 else "failed",
        "failure": None if process.returncode == 0 else
            f"process exited {process.returncode}",
        "duration_seconds": time.monotonic() - started,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _aggregate_rank_failures(reports):
    return "; ".join(
        f"rank {report['rank']}: {report['failure']}"
        for report in sorted(reports, key=lambda value: value["rank"])
        if report.get("failure")
    )


def _measurement_failures(case, reports):
    if case != "overlap-vs-serialized":
        return []
    return [
        f"rank {report['rank']}: "
        f"{report.get('measurements', {}).get('acceptance_failure')}"
        for report in sorted(reports, key=lambda value: value["rank"])
        if report.get("measurements", {}).get("acceptance_failure")
    ]


def _case_measurements(case, reports):
    ordered = sorted(reports, key=lambda value: value["rank"])
    if case in {
            "completion-mismatch", "drop-event", "destroy-pending-retry",
            "overlap-vs-serialized"}:
        return {
            "ranks": [
                {"rank": report["rank"], **(report.get("measurements") or {})}
                for report in ordered
            ],
        }
    return ordered[0].get("measurements") or {}


def _stream_id(event):
    for key, value in (event.get("args") or {}).items():
        normalized = "".join(character.lower() for character in str(key)
                             if character.isalnum())
        if normalized in ("physicstreamid", "streamid", "stream"):
            return value
    return event.get("tid")


def _trace_events(trace_paths):
    events = []
    for trace_path in trace_paths:
        try:
            payload = json.loads(pathlib.Path(trace_path).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        trace_events = payload if isinstance(payload, list) else \
            payload.get("traceEvents", ()) if isinstance(payload, dict) else ()
        events.extend(
            event for event in trace_events if isinstance(event, dict))
    return events


def _normalized_event_arg(event, expected):
    for key, value in (event.get("args") or {}).items():
        normalized = "".join(
            character.lower() for character in str(key)
            if character.isalnum())
        if normalized == expected:
            return value
    return None


def _has_explicit_stream_id(event):
    return any(
        "".join(character.lower() for character in str(key)
                if character.isalnum()) in
        ("physicstreamid", "streamid", "stream")
        for key in (event.get("args") or {}))


def _find_npu_overlap_interval(trace_paths):
    events = _trace_events(trace_paths)

    def event_family_intervals(name_fragment):
        selected = []
        for event in events:
            if event.get("ph") != "X" or "ts" not in event or \
                    "dur" not in event:
                continue
            name = str(event.get("name", ""))
            if name_fragment not in name:
                continue
            if not _has_explicit_stream_id(event):
                continue
            start = float(event["ts"])
            duration = float(event["dur"])
            if duration > 0:
                selected.append((start, start + duration, name,
                                 _stream_id(event)))
        return selected

    compute_intervals = event_family_intervals("MatMulV3")
    communication_intervals = event_family_intervals("dispatch_kernel")

    def unique_stream(intervals, family):
        streams = sorted({row[3] for row in intervals}, key=lambda value: str(value))
        _check(streams, f"NPU profiler did not contain {family} events")
        _check(
            len(streams) == 1,
            f"{family} event family spans physical streams " +
            "/".join(str(value) for value in streams),
        )
        return streams[0]

    physical_compute_stream_id = unique_stream(
        compute_intervals, "compute")
    physical_communication_stream_id = unique_stream(
        communication_intervals, "communication")
    _check(
        str(physical_compute_stream_id) !=
        str(physical_communication_stream_id),
        "profiler compute and communication event families alias physical "
        f"stream {physical_compute_stream_id}",
    )

    best = None
    for compute in compute_intervals:
        for communication in communication_intervals:
            overlap = min(compute[1], communication[1]) - \
                max(compute[0], communication[0])
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, compute, communication)
    if best is None:
        raise AssertionError(
            "NPU profiler did not contain an overlapping compute/communication "
            "interval for MatMulV3/dispatch_kernel on physical streams "
            f"{physical_compute_stream_id}/{physical_communication_stream_id}")
    return {
        "overlap_us": best[0],
        "compute_event": best[1][2],
        "communication_event": best[2][2],
        "compute_interval_us": [best[1][0], best[1][1]],
        "communication_interval_us": [best[2][0], best[2][1]],
        "physical_compute_stream_id": physical_compute_stream_id,
        "physical_communication_stream_id":
            physical_communication_stream_id,
    }


def _find_conv3d_npu_overlap_interval(trace_paths):
    events = _trace_events(trace_paths)

    intervals = []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or \
                "dur" not in event:
            continue
        duration = float(event["dur"])
        if not _has_explicit_stream_id(event) or duration <= 0:
            continue
        start = float(event["ts"])
        intervals.append({
            "start": start,
            "end": start + duration,
            "duration": duration,
            "name": str(event.get("name", "")),
            "stream_id": _stream_id(event),
            "task_type": _normalized_event_arg(event, "tasktype"),
        })

    def event_family(name_fragment, family, required_task_type):
        selected = [
            row for row in intervals if name_fragment in row["name"]]
        _check(selected,
               f"NPU profiler did not contain {family} events")
        task_types = sorted(
            {str(row["task_type"]) for row in selected})
        _check(
            all(row["task_type"] == required_task_type for row in selected),
            f"{family} task type is " + "/".join(task_types) +
            f", expected {required_task_type}",
        )
        streams = sorted(
            {row["stream_id"] for row in selected}, key=lambda value: str(value))
        _check(
            len(streams) == 1,
            f"{family} event family spans physical streams " +
            "/".join(str(value) for value in streams),
        )
        return selected, streams[0]

    compute_intervals, physical_compute_stream_id = event_family(
        "Conv3DV2", "compute", "KERNEL_AICORE")
    communication_intervals, physical_communication_stream_id = event_family(
        "dispatch_kernel", "communication", "KERNEL_AIVEC")
    _check(
        str(physical_compute_stream_id) !=
        str(physical_communication_stream_id),
        "profiler compute and communication event families alias physical "
        f"stream {physical_compute_stream_id}",
    )

    best = None
    for compute in compute_intervals:
        for communication in communication_intervals:
            overlap = min(compute["end"], communication["end"]) - \
                max(compute["start"], communication["start"])
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, compute, communication)
    if best is None:
        raise AssertionError(
            "NPU profiler did not contain an overlapping compute/communication "
            "interval for Conv3DV2/dispatch_kernel on physical streams "
            f"{physical_compute_stream_id}/{physical_communication_stream_id}")

    def inventory(selected):
        grouped = {}
        for row in selected:
            key = (row["name"], str(row["task_type"]), str(row["stream_id"]))
            entry = grouped.setdefault(key, {
                "name": row["name"],
                "task_type": row["task_type"],
                "physical_stream_id": row["stream_id"],
                "count": 0,
                "total_duration_us": 0.0,
                "first_start_us": row["start"],
                "last_end_us": row["end"],
            })
            entry["count"] += 1
            entry["total_duration_us"] += row["duration"]
            entry["first_start_us"] = min(
                entry["first_start_us"], row["start"])
            entry["last_end_us"] = max(entry["last_end_us"], row["end"])
        return [grouped[key] for key in sorted(grouped)]

    auxiliary_aiv = [
        row for row in intervals
        if row["task_type"] == "KERNEL_AIVEC" and
        "dispatch_kernel" not in row["name"]]
    transdata = [
        row for row in intervals if "TransData" in row["name"]]
    primary_compute_span_us = (
        max(row["end"] for row in compute_intervals) -
        min(row["start"] for row in compute_intervals)
    )
    auxiliary_aiv_total_duration_us = sum(
        row["duration"] for row in auxiliary_aiv)
    return {
        "overlap_us": best[0],
        "compute_event": best[1]["name"],
        "communication_event": best[2]["name"],
        "compute_interval_us": [best[1]["start"], best[1]["end"]],
        "communication_interval_us": [
            best[2]["start"], best[2]["end"]],
        "compute_task_type": "KERNEL_AICORE",
        "communication_task_type": "KERNEL_AIVEC",
        "physical_compute_stream_id": physical_compute_stream_id,
        "physical_communication_stream_id":
            physical_communication_stream_id,
        "primary_compute_span_us": primary_compute_span_us,
        "auxiliary_aiv_events": inventory(auxiliary_aiv),
        "auxiliary_aiv_total_duration_us":
            auxiliary_aiv_total_duration_us,
        "auxiliary_aiv_ratio":
            auxiliary_aiv_total_duration_us / primary_compute_span_us,
        "transdata_events": inventory(transdata),
        "compute_path_aic_only": not auxiliary_aiv,
    }


def _find_component_npu_overlap_interval(trace_paths):
    return _find_conv3d_npu_overlap_interval(trace_paths)


def _find_formal_npu_overlap_interval(trace_paths):
    evidence = _find_conv3d_npu_overlap_interval(trace_paths)
    evidence["compute_path_aic_only"] = False
    return evidence


def _formal_profiler_failures(interval):
    failures = []
    if interval.get("overlap_us", 0) <= 0:
        failures.append(
            interval.get("failure") or
            "NPU profiler did not contain a positive overlap interval")
    ratio = interval.get("auxiliary_aiv_ratio")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or \
            not math.isfinite(ratio):
        failures.append("NPU profiler auxiliary AIV ratio is missing")
    elif ratio >= OVERLAP_MAX_AUXILIARY_AIV_RATIO:
        failures.append(
            f"NPU profiler auxiliary AIV ratio {ratio:.6f} is not below "
            f"{OVERLAP_MAX_AUXILIARY_AIV_RATIO:.6f}")
    return failures


def _percentile(samples, percentile):
    ordered = sorted(float(value) for value in samples)
    _check(ordered, "cannot summarize empty samples")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(samples):
    return {
        "median_seconds": statistics.median(samples),
        "p95_seconds": _percentile(samples, 95),
        "samples_seconds": list(samples),
    }


def _worker_payload(stdout, expected_case):
    prefix = WORKER_RESULT_PREFIX + " "
    markers = [line[len(prefix):] for line in stdout.splitlines()
               if line.startswith(prefix)]
    if len(markers) != 1:
        raise ValueError(
            f"worker result marker count is {len(markers)}, expected 1")
    try:
        payload = json.loads(markers[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"worker result is malformed JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("worker result must be a JSON object")
    if payload.get("case") != expected_case:
        raise ValueError(
            f"worker result case is {payload.get('case')!r}, "
            f"expected {expected_case!r}")
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("worker result measurements must be a JSON object")
    required = STANDALONE_MEASUREMENT_FIELDS.get(expected_case)
    if required is None:
        raise ValueError(
            f"worker result case {expected_case!r} is not standalone")
    missing = sorted(required - measurements.keys())
    if missing:
        raise ValueError(
            "worker result measurements are missing " + ", ".join(missing))
    return payload


def _case_command(case, trace_dir):
    worker = [str(pathlib.Path(__file__).resolve()),
              "--worker", case, "--trace-dir", str(trace_dir)]
    if case in EVENT_CASES:
        return [sys.executable, *worker]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={WORLD_SIZE}", *worker]


def _distributed_batch_command(cases, trace_dir):
    worker = [str(pathlib.Path(__file__).resolve()),
              "--trace-dir", str(trace_dir), "--batch-worker", *cases]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={WORLD_SIZE}", *worker]


def _overlap_diagnostic_command(output, trace_dir):
    worker = [
        str(pathlib.Path(__file__).resolve()),
        "--overlap-diagnostic-worker",
        "--output", str(output),
        "--trace-dir", str(trace_dir),
    ]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={WORLD_SIZE}", *worker]


def _overlap_sweep_command(output, trace_dir):
    worker = [
        str(pathlib.Path(__file__).resolve()),
        "--overlap-sweep-worker",
        "--output", str(output),
        "--trace-dir", str(trace_dir),
    ]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={WORLD_SIZE}", *worker]


def _overlap_component_command(output, trace_dir):
    worker = [
        str(pathlib.Path(__file__).resolve()),
        "--overlap-component-diagnostic-worker",
        "--output", str(output),
        "--trace-dir", str(trace_dir),
    ]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={WORLD_SIZE}", *worker]


def _marker_payload(stdout, prefix):
    marker = prefix + " "
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return None


def _classify_overlap_diagnostic(variants):
    by_name = {row.get("variant"): row for row in variants}
    if tuple(by_name) != OVERLAP_DIAGNOSTIC_VARIANTS:
        return "incomplete"

    def rank_rows(variant):
        return by_name[variant].get("ranks") or []

    def all_wait(variant, outcome):
        rows = rank_rows(variant)
        return len(rows) == WORLD_SIZE and all(
            row.get("event_wait", {}).get("outcome") == outcome
            for row in rows)

    controls_pass = (
        all_wait("pure-communication", "completed") and
        all_wait("combined-light", "completed") and
        all_wait("post-dispatch-idle", "completed")
    )
    heavy_timed_out = all_wait("combined-heavy", "timeout")
    heavy_eventually_completed = all(
        row.get("queries", {}).get("comm_tail_after_grace") and
        (row.get("queries", {}).get("comm_tail_first_complete_seconds") or 0) >
        row.get("event_wait", {}).get("bound_seconds", 0)
        for row in rank_rows("combined-heavy")
    )
    pure_comm_rows = rank_rows("pure-communication")
    pure_comm_exceeded = len(pure_comm_rows) == WORLD_SIZE and all(
        row.get("event_wait", {}).get("outcome") == "timeout" and
        row.get("queries", {}).get("comm_tail_after_grace") and
        (row.get("queries", {}).get("comm_tail_first_complete_seconds") or 0) >
        row.get("event_wait", {}).get("bound_seconds", 0)
        for row in pure_comm_rows
    )
    pure_compute_exceeded = all(
        row.get("event_wait", {}).get("outcome") == "timeout" and
        row.get("queries", {}).get("compute_tail_after_grace")
        for row in rank_rows("pure-compute")
    )
    if (pure_comm_exceeded or
            (controls_pass and heavy_timed_out and
             (heavy_eventually_completed or pure_compute_exceeded))):
        return "workload-exceeds-event-deadline"

    pure_comm_stalled = any(
        row.get("event_wait", {}).get("outcome") == "timeout" and
        not row.get("queries", {}).get("comm_tail_after_grace")
        for row in rank_rows("pure-communication"))
    idle_event_stalled = any(
        row.get("queries", {}).get("comm_tail_after_grace") and
        row.get("event_wait", {}).get("outcome") == "timeout"
        for row in rank_rows("post-dispatch-idle"))
    if pure_comm_stalled or idle_event_stalled:
        return "completion-event-stalled"

    if (all_wait("pure-communication", "completed") and
            all_wait("post-dispatch-idle", "completed") and
            not all_wait("combined-light", "completed")):
        return "stream-dependency-order"
    return "inconclusive"


def _validate_overlap_diagnostic(variants):
    _check(tuple(row.get("variant") for row in variants) ==
           OVERLAP_DIAGNOSTIC_VARIANTS,
           "overlap diagnostic variants are missing or out of order")
    buffer_instances = []
    required_stages = {
        "setup", "dispatch_return", "compute_enqueue", "event_wait"}
    required_queries = {
        "comm_tail_immediate", "comm_tail_before_wait",
        "comm_tail_after_wait", "comm_tail_after_grace",
        "comm_tail_first_complete_seconds", "compute_tail_after_grace",
        "compute_tail_first_complete_seconds"}
    expected_order = [
        "dispatch-return", "compute-enqueued", "event-wait-start",
        "event-wait-end", "query-grace-end"]
    for variant in variants:
        ranks = variant.get("ranks") or []
        _check([row.get("rank") for row in ranks] == list(range(WORLD_SIZE)),
               f"{variant.get('variant')} omitted a rank diagnostic")
        for row in ranks:
            buffer_instances.append(row.get("buffer_instance"))
            _check(row.get("compute_stream_id") !=
                   row.get("communication_stream_id"),
                   "diagnostic compute and communication streams alias")
            _check(set(row.get("stage_seconds", {})) == required_stages,
                   "diagnostic stage timings are incomplete")
            _check(set(row.get("queries", {})) == required_queries,
                   "diagnostic event query boundaries are incomplete")
            wait = row.get("event_wait") or {}
            _check(wait.get("outcome") in ("completed", "timeout", "error"),
                   "diagnostic event wait outcome is invalid")
            _check(wait.get("bound_seconds") == 5.0,
                   "diagnostic event wait bound changed")
            _check(row.get("order") == expected_order,
                   "diagnostic stream-order evidence is incomplete")
    _check(None not in buffer_instances and
           len(set(buffer_instances)) == len(buffer_instances),
           "diagnostic variants did not use fresh per-rank buffers")


def _overlap_diagnostic_report(variants, *, duration_seconds=None,
                               launcher_failure=None):
    classification = _classify_overlap_diagnostic(variants)
    return {
        "schema_version": 1,
        "diagnostic": "overlap-vs-serialized",
        "diagnostic_timeout_seconds": OVERLAP_DIAGNOSTIC_TIMEOUT_SECONDS,
        "event_wait_bound_seconds": 5.0,
        "query_grace_seconds": OVERLAP_DIAGNOSTIC_GRACE_SECONDS,
        "classification": classification,
        "duration_seconds": duration_seconds,
        "launcher_failure": launcher_failure,
        "variants": variants,
    }


def _run_overlap_diagnostic(output, trace_dir):
    pathlib.Path(trace_dir).mkdir(parents=True, exist_ok=True)
    result = _run_bounded(
        _overlap_diagnostic_command(output, trace_dir),
        timeout_seconds=OVERLAP_DIAGNOSTIC_TIMEOUT_SECONDS)
    payload = _marker_payload(
        result.get("stdout", ""), OVERLAP_DIAGNOSTIC_RESULT_PREFIX)
    if payload is None:
        variants = []
        if pathlib.Path(output).exists():
            variants = json.loads(pathlib.Path(output).read_text()).get(
                "variants", [])
        report = _overlap_diagnostic_report(
            variants, duration_seconds=result.get("duration_seconds"),
            launcher_failure=result.get("failure") or
            "diagnostic worker omitted its result marker")
        _write_suite_report(output, report)
        print(OVERLAP_DIAGNOSTIC_RESULT_PREFIX + " " +
              json.dumps(report, sort_keys=True), flush=True)
        return 1
    variants = payload.get("variants") or []
    try:
        _validate_overlap_diagnostic(variants)
    except AssertionError as error:
        report = _overlap_diagnostic_report(
            variants, duration_seconds=result.get("duration_seconds"),
            launcher_failure=str(error))
        _write_suite_report(output, report)
        print(OVERLAP_DIAGNOSTIC_RESULT_PREFIX + " " +
              json.dumps(report, sort_keys=True), flush=True)
        return 1
    report = _overlap_diagnostic_report(
        variants, duration_seconds=result.get("duration_seconds"),
        launcher_failure=result.get("failure"))
    _write_suite_report(output, report)
    print(OVERLAP_DIAGNOSTIC_RESULT_PREFIX + " " +
          json.dumps(report, sort_keys=True), flush=True)
    return 0 if result.get("status") == "passed" else 1


def _classify_overlap_sweep_point(point):
    ranks = point.get("ranks") or []
    if len(ranks) != WORLD_SIZE:
        return "incomplete"
    if any(row.get("diagnostic_failure") or
           not row.get("event_wait_completed") for row in ranks):
        return "operation-failure"
    if any(row.get("theoretical_maximum_improvement", 0) <
           OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT for row in ranks):
        return "workload-ratio-cannot-meet-threshold"
    if any(row.get("profiler_overlap", {}).get("overlap_us", 0) <= 0
           for row in ranks):
        return "profiler-overlap-missing"
    if all(row.get("median_improvement", 0) >=
           OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT for row in ranks):
        return "candidate"
    return "overlap-efficiency-below-threshold"


def _validate_overlap_sweep(points):
    _check(tuple(point.get("tokens") for point in points) ==
           OVERLAP_SWEEP_TOKENS,
           "overlap sweep token points are missing or out of order")
    buffer_instances = []
    required_summaries = (
        "communication", "compute", "serialized", "overlapped")
    required_summary_fields = {
        "median_seconds", "p95_seconds", "samples_seconds"}
    for point in points:
        ranks = point.get("ranks") or []
        _check([row.get("rank") for row in ranks] == list(range(WORLD_SIZE)),
               f"overlap sweep tokens={point.get('tokens')} omitted a rank")
        for row in ranks:
            _check(row.get("tokens") == point.get("tokens"),
                   "overlap sweep rank token count mismatches its point")
            buffer_instances.append(row.get("buffer_instance"))
            _check(row.get("logical_compute_stream_id",
                           row.get("profiler_overlap", {}).get(
                               "logical_compute_stream_id")) !=
                   row.get("logical_communication_stream_id",
                           row.get("profiler_overlap", {}).get(
                               "logical_communication_stream_id")),
                   "overlap sweep compute and communication streams alias")
            profiler = row.get("profiler_overlap", {})
            if profiler.get("overlap_us", 0) > 0:
                physical_compute = row.get(
                    "physical_compute_stream_id",
                    profiler.get("physical_compute_stream_id"))
                physical_communication = row.get(
                    "physical_communication_stream_id",
                    profiler.get("physical_communication_stream_id"))
                _check(physical_compute is not None and
                       physical_communication is not None,
                       "positive overlap omitted physical stream IDs")
                _check(str(physical_compute) != str(physical_communication),
                       "profiler compute and communication streams alias")
            for name in required_summaries:
                _check(set(row.get(name, {})) == required_summary_fields,
                       f"overlap sweep {name} summary is incomplete")
            _check(isinstance(
                row.get("theoretical_maximum_improvement"), (int, float)),
                "overlap sweep theoretical improvement is missing")
            _check(isinstance(row.get("median_improvement"), (int, float)),
                   "overlap sweep observed improvement is missing")
    _check(None not in buffer_instances and
           len(set(buffer_instances)) == len(buffer_instances),
           "overlap sweep did not use fresh per-rank buffers")


def _overlap_sweep_report(points, *, duration_seconds=None,
                          launcher_failure=None):
    classified = []
    for point in points:
        row = dict(point)
        row["classification"] = _classify_overlap_sweep_point(row)
        classified.append(row)
    candidates = [
        point["tokens"] for point in classified
        if point["classification"] == "candidate"
    ]
    return {
        "schema_version": 1,
        "diagnostic": "overlap-load-ratio-sweep",
        "diagnostic_timeout_seconds": OVERLAP_SWEEP_TIMEOUT_SECONDS,
        "acceptance_eligible": False,
        "acceptance_ineligible_reason": "single-sample-diagnostic",
        "minimum_median_improvement":
            OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
        "compute_shape": list(OVERLAP_DIAGNOSTIC_COMPUTE_SHAPE),
        "compute_iterations": OVERLAP_DIAGNOSTIC_COMPUTE_ITERATIONS,
        "warmups": OVERLAP_SWEEP_WARMUPS,
        "repetitions": OVERLAP_SWEEP_REPETITIONS,
        "duration_seconds": duration_seconds,
        "launcher_failure": launcher_failure,
        "recommended_tokens": min(candidates) if candidates else None,
        "points": classified,
    }


def _load_overlap_sweep_phase_checkpoint(trace_dir):
    by_tokens = {}
    for path in pathlib.Path(trace_dir).glob("tokens-*/phase-rank*.json"):
        try:
            checkpoint = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        measurement = checkpoint.get("measurement")
        if not isinstance(measurement, dict):
            continue
        tokens = checkpoint.get("tokens")
        rank = measurement.get("rank")
        if tokens not in OVERLAP_SWEEP_TOKENS or not isinstance(rank, int):
            continue
        by_tokens.setdefault(tokens, {})[rank] = checkpoint
    for tokens in reversed(OVERLAP_SWEEP_TOKENS):
        checkpoints = by_tokens.get(tokens)
        if not checkpoints:
            continue
        ordered = [checkpoints[rank] for rank in sorted(checkpoints)]
        phases = {checkpoint.get("completed_phase") for checkpoint in ordered}
        active_phases = {
            checkpoint.get("active_phase") for checkpoint in ordered}
        statuses = {checkpoint.get("phase_status") for checkpoint in ordered}
        return {
            "tokens": tokens,
            "completed_phase": phases.pop() if len(phases) == 1 else None,
            "active_phase": active_phases.pop()
            if len(active_phases) == 1 else None,
            "phase_status": statuses.pop() if len(statuses) == 1 else None,
            "ranks": [checkpoint["measurement"] for checkpoint in ordered],
        }
    return None


def _run_overlap_sweep(output, trace_dir):
    pathlib.Path(trace_dir).mkdir(parents=True, exist_ok=True)
    result = _run_bounded(
        _overlap_sweep_command(output, trace_dir),
        timeout_seconds=OVERLAP_SWEEP_TIMEOUT_SECONDS)
    payload = _marker_payload(
        result.get("stdout", ""), OVERLAP_SWEEP_RESULT_PREFIX)
    if payload is None:
        points = []
        active_point = None
        if pathlib.Path(output).exists():
            checkpoint = json.loads(pathlib.Path(output).read_text())
            points = checkpoint.get("points", [])
            active_point = checkpoint.get("active_point")
        if active_point is None:
            active_point = _load_overlap_sweep_phase_checkpoint(trace_dir)
        report = _overlap_sweep_report(
            points, duration_seconds=result.get("duration_seconds"),
            launcher_failure=result.get("failure") or
            "overlap sweep worker omitted its result marker")
        if active_point is not None:
            report["active_point"] = active_point
        _write_suite_report(output, report)
        print(OVERLAP_SWEEP_RESULT_PREFIX + " " +
              json.dumps(report, sort_keys=True), flush=True)
        return 1
    points = payload.get("points") or []
    try:
        _validate_overlap_sweep(points)
    except AssertionError as error:
        report = _overlap_sweep_report(
            points, duration_seconds=result.get("duration_seconds"),
            launcher_failure=str(error))
        _write_suite_report(output, report)
        print(OVERLAP_SWEEP_RESULT_PREFIX + " " +
              json.dumps(report, sort_keys=True), flush=True)
        return 1
    report = _overlap_sweep_report(
        points, duration_seconds=result.get("duration_seconds"),
        launcher_failure=result.get("failure"))
    _write_suite_report(output, report)
    print(OVERLAP_SWEEP_RESULT_PREFIX + " " +
          json.dumps(report, sort_keys=True), flush=True)
    return 0 if result.get("status") == "passed" else 1


def _overlap_component_tolerances(row):
    component_sum = (
        float(row["communication_only_seconds"]) +
        float(row["compute_only_seconds"])
    )
    serialized = float(row["serialized_seconds"])
    return {
        "component_sum": (
            component_sum * OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT),
        "serialized": serialized * OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
    }


def _classify_overlap_component_rank(row):
    if row.get("diagnostic_failure"):
        return "operation-failure"
    required = (
        "communication_only_seconds",
        "compute_only_seconds",
        "serialized_seconds",
        "overlapped_seconds",
    )
    if any(not isinstance(row.get(name), (int, float)) for name in required):
        return "incomplete"

    communication = float(row["communication_only_seconds"])
    compute = float(row["compute_only_seconds"])
    serialized = float(row["serialized_seconds"])
    overlapped = float(row["overlapped_seconds"])
    component_sum = communication + compute
    tolerance = _overlap_component_tolerances(row)

    if serialized < component_sum - tolerance["component_sum"]:
        return "unserialized-baseline"
    if overlapped < serialized - tolerance["serialized"]:
        return "effective-overlap"
    if row.get("profiler_overlap", {}).get("overlap_us", 0) <= 0:
        return "profiler-overlap-missing"
    if (abs(serialized - component_sum) <= tolerance["component_sum"] and
            abs(overlapped - serialized) <= tolerance["serialized"]):
        return "resource-contention"
    return "inconclusive"


def _overlap_component_report(ranks, *, duration_seconds=None,
                              launcher_failure=None):
    classified = []
    for measurement in sorted(
            ranks, key=lambda value: value.get("rank", WORLD_SIZE)):
        row = dict(measurement)
        row["classification"] = _classify_overlap_component_rank(row)
        if all(isinstance(row.get(name), (int, float)) for name in (
                "communication_only_seconds", "compute_only_seconds",
                "serialized_seconds", "overlapped_seconds")):
            row["component_sum_seconds"] = (
                row["communication_only_seconds"] +
                row["compute_only_seconds"])
            row["serialized_component_gap_seconds"] = (
                row["component_sum_seconds"] - row["serialized_seconds"])
            row["overlap_gain_seconds"] = (
                row["serialized_seconds"] - row["overlapped_seconds"])
            row["wall_gain_ratio"] = (
                row["overlap_gain_seconds"] / row["serialized_seconds"]
                if row["serialized_seconds"] > 0 else None)
            row["measured_tolerance_seconds"] = \
                _overlap_component_tolerances(row)
            row["serialized_baseline_valid"] = (
                row["serialized_seconds"] >=
                row["component_sum_seconds"] -
                row["measured_tolerance_seconds"]["component_sum"])
        profiler = row.get("profiler_overlap") or {}
        row["event_wait_within_limit"] = (
            isinstance(row.get("event_wait_duration_seconds"), (int, float))
            and row["event_wait_duration_seconds"] <
            OVERLAP_COMPONENT_EVENT_WAIT_LIMIT_SECONDS)
        row["diagnostic_success"] = bool(
            row.get("serialized_baseline_valid") and
            isinstance(row.get("wall_gain_ratio"), (int, float)) and
            row["wall_gain_ratio"] >= OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT and
            row["event_wait_within_limit"] and
            profiler.get("overlap_us", 0) > 0 and
            profiler.get("compute_task_type") == "KERNEL_AICORE" and
            profiler.get("communication_task_type") == "KERNEL_AIVEC" and
            str(profiler.get("physical_compute_stream_id")) !=
            str(profiler.get("physical_communication_stream_id")))
        classified.append(row)

    classifications = {row["classification"] for row in classified}
    classification = classifications.pop() \
        if len(classifications) == 1 else "mixed"
    return {
        "schema_version": 1,
        "diagnostic": "overlap-component-timing",
        "diagnostic_timeout_seconds": OVERLAP_COMPONENT_TIMEOUT_SECONDS,
        "acceptance_eligible": False,
        "acceptance_ineligible_reason": "single-sample-diagnostic",
        "diagnostic_success": (
            launcher_failure is None and len(classified) == WORLD_SIZE and
            all(row["diagnostic_success"] for row in classified)),
        "classification": classification,
        "classification_materiality_ratio":
            OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
        "communication_tokens": OVERLAP_COMMUNICATION_TOKENS,
        "compute_calibration": {
            "initial_iterations":
                OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS,
            "iteration_bounds": [
                OVERLAP_COMPONENT_MIN_ITERATIONS,
                OVERLAP_COMPONENT_MAX_ITERATIONS,
            ],
            "target_range_seconds":
                list(OVERLAP_COMPONENT_TARGET_RANGE_SECONDS),
            "target_seconds": OVERLAP_COMPONENT_TARGET_SECONDS,
            "warmup_iterations":
                OVERLAP_COMPONENT_CALIBRATION_WARMUP_ITERATIONS,
        },
        "compute_input_shape": list(OVERLAP_COMPONENT_INPUT_SHAPE),
        "compute_variant": OVERLAP_COMPONENT_COMPUTE_VARIANT,
        "compute_weight_shape": list(OVERLAP_COMPONENT_WEIGHT_SHAPE),
        "event_wait_limit_seconds":
            OVERLAP_COMPONENT_EVENT_WAIT_LIMIT_SECONDS,
        "minimum_wall_gain": OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
        "warmups": OVERLAP_COMPONENT_WARMUPS,
        "repetitions": OVERLAP_COMPONENT_REPETITIONS,
        "duration_seconds": duration_seconds,
        "launcher_failure": launcher_failure,
        "ranks": classified,
    }


def _validate_overlap_component(ranks):
    _check([row.get("rank") for row in ranks] == list(range(WORLD_SIZE)),
           "overlap component diagnostic omitted a rank")
    required_durations = (
        "communication_only_seconds",
        "compute_only_seconds",
        "serialized_seconds",
        "overlapped_seconds",
        "serialized_dispatch_return_duration_seconds",
        "async_dispatch_return_duration_seconds",
        "event_wait_duration_seconds",
        "compute_completion_duration_seconds",
    )
    required_stages = {
        "communication-only": {"async_dispatch_return", "event_wait"},
        "compute-only": {"compute_enqueue", "compute_completion"},
        "serialized": {
            "serialized_dispatch_return", "compute_enqueue",
            "compute_completion",
        },
        "overlapped": {
            "async_dispatch_return", "compute_enqueue", "event_wait",
            "compute_completion",
        },
    }
    buffer_instances = []
    for row in ranks:
        _check(row.get("communication_tokens") ==
               OVERLAP_COMMUNICATION_TOKENS,
               "overlap component communication tokens changed")
        _check(row.get("compute_variant") ==
               OVERLAP_COMPONENT_COMPUTE_VARIANT,
               "overlap component compute variant changed")
        _check(row.get("compute_input_shape") ==
               list(OVERLAP_COMPONENT_INPUT_SHAPE) and
               row.get("compute_weight_shape") ==
               list(OVERLAP_COMPONENT_WEIGHT_SHAPE),
               "overlap component Conv3D shapes changed")
        calibration = row.get("compute_calibration") or {}
        iterations = row.get("compute_iterations")
        _check(
            calibration.get("completion_wait") ==
            "npu-event-synchronize" and
            calibration.get("initial_iterations") ==
            OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS and
            calibration.get("iteration_bounds") == [
                OVERLAP_COMPONENT_MIN_ITERATIONS,
                OVERLAP_COMPONENT_MAX_ITERATIONS] and
            calibration.get("target_range_seconds") ==
            list(OVERLAP_COMPONENT_TARGET_RANGE_SECONDS) and
            calibration.get("target_seconds") ==
            OVERLAP_COMPONENT_TARGET_SECONDS and
            calibration.get("warmup_iterations") ==
            OVERLAP_COMPONENT_CALIBRATION_WARMUP_ITERATIONS and
            isinstance(calibration.get("measured_initial_seconds"),
                       (int, float)) and
            calibration["measured_initial_seconds"] > 0 and
            isinstance(iterations, int) and
            OVERLAP_COMPONENT_MIN_ITERATIONS <= iterations <=
            OVERLAP_COMPONENT_MAX_ITERATIONS and
            calibration.get("selected_iterations") == iterations,
            "overlap component compute calibration is invalid",
        )
        _check(all(isinstance(row.get(name), (int, float)) and
                   row[name] >= 0 for name in required_durations),
               "overlap component durations are incomplete")
        stages = row.get("stage_seconds") or {}
        _check(set(stages) == set(required_stages),
               "overlap component phase timings are incomplete")
        for phase, names in required_stages.items():
            _check(set(stages.get(phase, {})) == names,
                   f"overlap component {phase} stage timings are incomplete")
        profiler = row.get("profiler_overlap") or {}
        _check(profiler.get("overlap_us", 0) > 0,
               "overlap component profiler interval is missing")
        _check(profiler.get("physical_compute_stream_id") is not None and
               profiler.get("physical_communication_stream_id") is not None,
               "overlap component physical stream IDs are missing")
        _check(str(profiler["physical_compute_stream_id"]) !=
               str(profiler["physical_communication_stream_id"]),
               "overlap component physical streams alias")
        _check("Conv3DV2" in str(profiler.get("compute_event", "")) and
               profiler.get("compute_task_type") == "KERNEL_AICORE",
               "overlap component compute task type is not KERNEL_AICORE")
        _check("dispatch_kernel" in
               str(profiler.get("communication_event", "")) and
               profiler.get("communication_task_type") == "KERNEL_AIVEC",
               "overlap component communication task type is not KERNEL_AIVEC")
        auxiliary_aiv = profiler.get("auxiliary_aiv_events")
        transdata = profiler.get("transdata_events")
        _check(isinstance(auxiliary_aiv, list) and
               isinstance(transdata, list),
               "overlap component auxiliary event inventory is missing")
        _check(profiler.get("compute_path_aic_only") ==
               (len(auxiliary_aiv) == 0),
               "overlap component pure-compute claim contradicts auxiliary AIV")
        component_sum = (
            float(row["communication_only_seconds"]) +
            float(row["compute_only_seconds"]))
        tolerance = _overlap_component_tolerances(row)
        serialized = float(row["serialized_seconds"])
        _check(serialized >= component_sum - tolerance["component_sum"],
               "overlap component serialized baseline is invalid")
        _check(row["event_wait_duration_seconds"] <
               OVERLAP_COMPONENT_EVENT_WAIT_LIMIT_SECONDS,
               "overlap component event wait reached the 5s limit")
        _check(serialized > 0, "overlap component serialized baseline is zero")
        wall_gain = 1.0 - float(row["overlapped_seconds"]) / serialized
        _check(wall_gain >= OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
               "overlap component wall gain is below 5%")
        buffer_instances.append(row.get("buffer_instance"))
    _check(None not in buffer_instances and
           len(set(buffer_instances)) == WORLD_SIZE,
           "overlap component diagnostic did not use fresh rank buffers")


def _load_overlap_component_phase_checkpoint(trace_dir):
    measurements = {}
    for path in pathlib.Path(trace_dir).glob("phase-rank*.json"):
        try:
            checkpoint = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        measurement = checkpoint.get("measurement")
        if not isinstance(measurement, dict):
            continue
        rank = measurement.get("rank")
        if isinstance(rank, int):
            measurements[rank] = measurement
    return [measurements[rank] for rank in sorted(measurements)]


def _run_overlap_component_diagnostic(output, trace_dir):
    pathlib.Path(trace_dir).mkdir(parents=True, exist_ok=True)
    result = _run_bounded(
        _overlap_component_command(output, trace_dir),
        timeout_seconds=OVERLAP_COMPONENT_TIMEOUT_SECONDS)
    payload = _marker_payload(
        result.get("stdout", ""), OVERLAP_COMPONENT_RESULT_PREFIX)
    if payload is None:
        ranks = []
        if pathlib.Path(output).exists():
            ranks = json.loads(pathlib.Path(output).read_text()).get("ranks", [])
        if not ranks:
            ranks = _load_overlap_component_phase_checkpoint(trace_dir)
        report = _overlap_component_report(
            ranks, duration_seconds=result.get("duration_seconds"),
            launcher_failure=result.get("failure") or
            "overlap component worker omitted its result marker")
        _write_suite_report(output, report)
        print(OVERLAP_COMPONENT_RESULT_PREFIX + " " +
              json.dumps(report, sort_keys=True), flush=True)
        return 1

    ranks = payload.get("ranks") or []
    try:
        _validate_overlap_component(ranks)
    except AssertionError as error:
        report = _overlap_component_report(
            ranks, duration_seconds=result.get("duration_seconds"),
            launcher_failure=str(error))
        _write_suite_report(output, report)
        print(OVERLAP_COMPONENT_RESULT_PREFIX + " " +
              json.dumps(report, sort_keys=True), flush=True)
        return 1
    report = _overlap_component_report(
        ranks, duration_seconds=result.get("duration_seconds"),
        launcher_failure=result.get("failure"))
    _write_suite_report(output, report)
    print(OVERLAP_COMPONENT_RESULT_PREFIX + " " +
          json.dumps(report, sort_keys=True), flush=True)
    return 0 if result.get("status") == "passed" else 1


def _suite_report(suite, selected, results):
    executed = len(results)
    return {
        "schema_version": 1,
        "suite": suite,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
        "results": results,
        "summary": {
            "total": executed,
            "selected": len(selected),
            "executed": executed,
            "passed": sum(row["status"] == "passed" for row in results),
            "failed": sum(row["status"] == "failed" for row in results),
            "not_run": len(selected) - executed,
        },
    }


def _write_suite_report(output_path, report):
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)


def _terminate_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def _run_streaming_batch(command, cases, record_result,
                         timeout_seconds=CASE_TIMEOUT_SECONDS):
    cases = tuple(cases)
    _check(cases, "distributed batch requires at least one case")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    pending = b""
    transcript = ""
    next_index = 0
    active_case = cases[0]
    active_started = time.monotonic()
    saw_start = False
    protocol_failed = False

    def diagnostic():
        return transcript

    def failed_row(case, failure, exit_code=None):
        row = {
            "case": case,
            "status": "failed",
            "duration_seconds": time.monotonic() - active_started,
            "exit_code": exit_code,
            "failure": failure,
            "measurements": {},
        }
        if diagnostic():
            row["diagnostic"] = diagnostic()
        return row

    def fail_protocol(message, case=None):
        nonlocal protocol_failed
        protocol_failed = True
        record_result(failed_row(case or active_case, message))

    def parse_marker(line, prefix):
        suffix = line[len(prefix):]
        if not suffix.startswith(" "):
            raise ValueError(f"malformed {prefix} marker")
        payload = json.loads(suffix[1:])
        if not isinstance(payload, dict):
            raise ValueError(f"{prefix} payload is not an object")
        return payload

    def handle_line(line):
        nonlocal active_case, active_started, next_index, protocol_failed
        nonlocal saw_start, transcript
        transcript = (transcript + line + "\n")[-4000:]
        print(line, flush=True)
        try:
            if line.startswith(CASE_START_PREFIX):
                payload = parse_marker(line, CASE_START_PREFIX)
                case = payload.get("case")
                if next_index >= len(cases):
                    fail_protocol(f"unexpected case start after batch completion: {case}",
                                  cases[-1])
                    return
                expected = cases[next_index]
                if saw_start:
                    fail_protocol(f"duplicate case start for {case}", expected)
                    return
                if case != expected:
                    fail_protocol(
                        f"out-of-order case start: expected {expected}, got {case}",
                        expected)
                    return
                active_case = case
                active_started = time.monotonic()
                saw_start = True
            elif line.startswith(CASE_RESULT_PREFIX):
                payload = parse_marker(line, CASE_RESULT_PREFIX)
                case = payload.get("case")
                if next_index >= len(cases):
                    fail_protocol(
                        f"unexpected case result after batch completion: {case}",
                        cases[-1])
                    return
                expected = cases[next_index]
                if not saw_start:
                    fail_protocol(
                        f"case result before start: expected {expected}, got {case}",
                        expected)
                    return
                if case != expected:
                    fail_protocol(
                        f"out-of-order case result: expected {expected}, got {case}",
                        expected)
                    return
                status = payload.get("status")
                if status not in ("passed", "failed"):
                    fail_protocol(f"invalid result status for {case}: {status}", case)
                    return
                duration = payload.get(
                    "duration_seconds", time.monotonic() - active_started)
                if not isinstance(duration, (int, float)) or duration < 0:
                    fail_protocol(f"invalid result duration for {case}: {duration}",
                                  case)
                    return
                failure = payload.get("failure")
                if status == "failed" and not failure:
                    fail_protocol(f"failed result omitted failure for {case}", case)
                    return
                row = {
                    "case": case,
                    "status": status,
                    "duration_seconds": duration,
                    "exit_code": payload.get(
                        "exit_code", 0 if status == "passed" else 1),
                    "failure": failure,
                    "measurements": payload.get("measurements") or {},
                }
                if status == "failed" and diagnostic():
                    row["diagnostic"] = diagnostic()
                record_result(row)
                if status == "failed":
                    protocol_failed = True
                    return
                next_index += 1
                saw_start = False
                active_case = cases[next_index] if next_index < len(cases) \
                    else cases[-1]
                active_started = time.monotonic()
        except (json.JSONDecodeError, ValueError) as error:
            fail_protocol(f"invalid batch protocol: {error}")

    try:
        while True:
            remaining = timeout_seconds - (time.monotonic() - active_started)
            if remaining <= 0:
                record_result(failed_row(
                    active_case,
                    f"case watchdog timeout after {timeout_seconds}s"))
                protocol_failed = True
                _terminate_process_group(process)
                break

            events = selector.select(timeout=min(remaining, 0.1))
            for key, _mask in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                pending += chunk
                while b"\n" in pending:
                    raw_line, pending = pending.split(b"\n", 1)
                    handle_line(raw_line.decode("utf-8", errors="replace"))
                    if protocol_failed:
                        break
                if protocol_failed:
                    break
            if protocol_failed:
                _terminate_process_group(process)
                break
            if process.poll() is not None and not selector.get_map():
                if pending:
                    handle_line(pending.decode("utf-8", errors="replace"))
                    pending = b""
                break

        if protocol_failed:
            return False

        return_code = process.wait()
        if next_index < len(cases):
            record_result(failed_row(
                cases[next_index],
                f"batch launcher exited {return_code} before case result",
                return_code))
            return False
        if return_code != 0:
            record_result(failed_row(
                cases[-1],
                f"batch launcher exited {return_code} after successful results",
                return_code))
            return False
        return True
    finally:
        selector.close()
        _terminate_process_group(process)


def _run_distributed_batch(cases, trace_dir, record_result):
    return _run_streaming_batch(
        _distributed_batch_command(cases, trace_dir), cases, record_result)


def _run_suite(suite, output, trace_dir):
    selected = EVENT_CASES if suite == "event" else CASE_NAMES
    results = []
    pathlib.Path(trace_dir).mkdir(parents=True, exist_ok=True)

    def record_result(row):
        existing = next(
            (index for index, result in enumerate(results)
             if result["case"] == row["case"]), None)
        if existing is None:
            results.append(row)
        else:
            results[existing] = row
        if row["status"] == "failed":
            print(f"FAIL {row['case']}: {row['failure']}", flush=True)
            if row.get("diagnostic"):
                print(f"DIAGNOSTIC {row['case']}:\n{row['diagnostic']}",
                      flush=True)
        else:
            print(f"PASS {row['case']} ({row['duration_seconds']:.3f}s)",
                  flush=True)
        _write_suite_report(output, _suite_report(suite, selected, results))

    def run_standalone(case):
        result = _run_bounded(_case_command(case, trace_dir))
        payload = None
        if result["status"] == "passed":
            try:
                payload = _worker_payload(result["stdout"], case)
            except ValueError as error:
                result = {
                    **result,
                    "status": "failed",
                    "failure": str(error),
                }
        row = {
            "case": case,
            "status": result["status"],
            "duration_seconds": result["duration_seconds"],
            "exit_code": result["exit_code"],
            "failure": result["failure"],
            "measurements": payload.get("measurements", {})
                if payload is not None else {},
        }
        if result["status"] == "failed":
            row["diagnostic"] = (
                result["stderr"] or result["stdout"])[-4000:]
        record_result(row)
        return result["status"] == "passed"

    distributed = tuple(case for case in selected
                        if case in DISTRIBUTED_CASES)
    batch_pending = bool(distributed)
    for case in selected:
        if case in DISTRIBUTED_CASES:
            if batch_pending:
                batch_pending = False
                if not _run_distributed_batch(
                        distributed, trace_dir, record_result):
                    break
            continue
        if not run_standalone(case):
            break
    report = _suite_report(suite, selected, results)
    print("PHASE3E_SUITE_RESULT " + json.dumps(report, sort_keys=True),
          flush=True)
    return 0 if report["summary"]["failed"] == 0 else 1


@contextmanager
def _forbid_global_sync(torch):
    original = torch.npu.synchronize

    def forbidden(*_args, **_kwargs):
        raise AssertionError("global NPU synchronization is forbidden")

    forbidden.phase3e_original = original
    torch.npu.synchronize = forbidden
    try:
        yield
    finally:
        torch.npu.synchronize = original


@contextmanager
def _allow_buffer_construction_sync(torch):
    guarded = torch.npu.synchronize
    original = getattr(guarded, "phase3e_original", guarded)
    torch.npu.synchronize = original
    try:
        yield
    finally:
        torch.npu.synchronize = guarded


def _run_capture_worker():
    import torch
    import torch_npu  # noqa: F401
    import deep_ep

    torch.npu.set_device(0)
    with _forbid_global_sync(torch):
        value = torch.zeros((16,), dtype=torch.float32, device="npu")
        value.fill_(7.0)
        event = deep_ep.ElasticBuffer.capture()
        copied = copy.copy(event)
        event.current_stream_wait()
        copied.current_stream_wait()
        observed = value.cpu()
        expected = torch.full((16,), 7.0, dtype=torch.float32)
        _check(torch.equal(observed, expected),
               "captured current-stream event did not publish the queued write")
    return {"repeated_waits": 2, "global_synchronizations": 0}


def _run_record_failure_worker():
    import torch
    import torch_npu  # noqa: F401
    import deep_ep

    torch.npu.set_device(0)
    with _forbid_global_sync(torch):
        failure = None
        os.environ["DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT"] = \
            "record_failure"
        try:
            try:
                deep_ep.ElasticBuffer.capture()
            except RuntimeError as error:
                failure = str(error)
        finally:
            os.environ.pop("DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT", None)
        _check(failure is not None and "record_event" in failure and
               "backend error -2" in failure,
               f"native record failure was not observed: {failure}")

        recovery = deep_ep.ElasticBuffer.capture()
        recovery.current_stream_wait()
    return {
        "injected_failure": failure,
        "recovery_event_waited": True,
        "global_synchronizations": 0,
    }


def _run_event_timeout_worker():
    import torch
    import torch_npu  # noqa: F401
    import deep_ep

    torch.npu.set_device(0)
    with _forbid_global_sync(torch):
        event = deep_ep.ElasticBuffer.capture()
        failure = None
        os.environ["DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT"] = \
            "query_not_ready"
        try:
            try:
                event.current_stream_wait()
            except RuntimeError as error:
                failure = str(error)
        finally:
            os.environ.pop("DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT", None)
        _check(failure is not None and "synchronize_event" in failure and
               "backend error -1" in failure,
               f"bounded native event timeout was not observed: {failure}")

        event.current_stream_wait()
        recovery = deep_ep.ElasticBuffer.capture()
        recovery.current_stream_wait()
    return {
        "injected_failure": failure,
        "same_event_recovered": True,
        "recovery_event_waited": True,
        "global_synchronizations": 0,
    }


def _offset_fixture(fixture, offset):
    return {
        "payloads": tuple(
            tuple(tuple(value + offset for value in row) for row in rank_rows)
            for rank_rows in fixture["payloads"]),
        "routes": fixture["routes"],
    }


def _literal_reference(torch, dist, group, rank, fixture):
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered,
        {"payloads": fixture["payloads"][rank],
         "routes": fixture["routes"][rank]},
        group=group)
    local_first = rank * (NUM_EXPERTS // WORLD_SIZE)
    local_last = local_first + NUM_EXPERTS // WORLD_SIZE
    payloads = []
    routes = []
    for source in gathered:
        for payload, route in zip(source["payloads"], source["routes"]):
            destinations = {
                expert // (NUM_EXPERTS // WORLD_SIZE)
                for expert in route if expert >= 0
            }
            if rank not in destinations:
                continue
            payloads.append(payload)
            routes.append(tuple(
                expert - local_first if local_first <= expert < local_last
                else -1 for expert in route))
    expected_x = torch.tensor(payloads, dtype=torch.bfloat16).reshape(
        len(payloads), HIDDEN)
    expected_routes = torch.tensor(routes, dtype=torch.int64).reshape(
        len(routes), NUM_TOPK)
    return expected_x, expected_routes


class AsyncOverlapWorker:
    def __init__(self, torch, torch_npu, dist, deep_ep, group, device,
                 trace_dir):
        self.torch = torch
        self.torch_npu = torch_npu
        self.dist = dist
        self.deep_ep = deep_ep
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.trace_dir = pathlib.Path(trace_dir)
        self.buffers = []

    def new_buffer(self, num_bytes=8 * 1024 * 1024):
        with _allow_buffer_construction_sync(self.torch):
            buffer = self.deep_ep.ElasticBuffer(
                self.group,
                num_bytes=num_bytes,
                num_gpu_timeout_secs=5,
                deterministic=False,
                allow_hybrid_mode=False,
                explicitly_destroy=True,
            )
        self.buffers.append(buffer)
        return buffer

    def destroy_buffers(self):
        failures = []
        for buffer in reversed(self.buffers):
            try:
                buffer.destroy()
            except BaseException as error:
                failures.append(str(error))
        self.buffers.clear()
        if failures:
            raise RuntimeError("; ".join(failures))

    def _materialize(self, fixture):
        rows = fixture["payloads"][self.rank]
        routes = fixture["routes"][self.rank]
        x = self.torch.tensor(
            rows, dtype=self.torch.bfloat16,
            device=self.device).reshape(len(rows), HIDDEN).contiguous()
        topk = self.torch.tensor(
            routes, dtype=self.torch.int64,
            device=self.device).reshape(len(routes), NUM_TOPK).contiguous()
        return x, topk

    def _assert_tensor(self, actual, expected, label):
        _check(actual is not None, f"{label} is None")
        _check(actual.device.type == "npu" and
               actual.device.index == self.device.index,
               f"{label} is on {actual.device}, expected {self.device}")
        _check(actual.dtype == expected.dtype,
               f"{label} dtype {actual.dtype} != {expected.dtype}")
        _check(tuple(actual.shape) == tuple(expected.shape),
               f"{label} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        observed = actual.detach().cpu()
        _check(self.torch.equal(observed, expected),
               f"{label} differs: {observed.tolist()} != {expected.tolist()}")

    def _seed(self, buffer, fixture=REGULAR_FIXTURE):
        expected_x, expected_routes = _literal_reference(
            self.torch, self.dist, self.group, self.rank, fixture)
        x, routes = self._materialize(fixture)
        result = buffer.dispatch(
            x,
            topk_idx=routes,
            num_experts=NUM_EXPERTS,
            num_max_tokens_per_rank=CAPACITY,
            expert_alignment=1,
            num_sms=1,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        )
        recv_x, recv_routes, recv_weights, handle, event = result
        self._assert_tensor(recv_x, expected_x, "seed recv_x")
        self._assert_tensor(recv_routes, expected_routes, "seed recv_topk_idx")
        _check(recv_weights is None, "seed unexpectedly returned weights")
        _check(event.event is None, "seed dispatch returned an async event")
        return handle

    def _cached_dispatch(self, buffer, handle, fixture, *, async_mode,
                         allocate, previous=None, wait=True):
        expected_x, expected_routes = _literal_reference(
            self.torch, self.dist, self.group, self.rank, fixture)
        x, _ = self._materialize(fixture)
        result = buffer.dispatch(
            x,
            handle=handle,
            num_sms=1,
            num_qps=0,
            previous_event=previous,
            async_with_compute_stream=async_mode,
            allocate_on_comm_stream=allocate,
        )
        recv_x, recv_routes, recv_weights, returned_handle, event = result
        _check(returned_handle is handle, "cached dispatch replaced its handle")
        if async_mode:
            _check(event.event is not None,
                   "async cached dispatch did not return a native event")
            if wait:
                event.current_stream_wait()
        else:
            _check(event.event is None,
                   "synchronous cached dispatch returned a native event")
        if wait:
            self._assert_tensor(recv_x, expected_x, "cached recv_x")
            self._assert_tensor(
                recv_routes, expected_routes, "cached recv_topk_idx")
            _check(recv_weights is None,
                   "weightless cached dispatch returned weights")
        return recv_x, event, expected_x, expected_routes

    def _run_cached_case(self, case):
        async_mode = "-async-" in case
        allocate = case.endswith("-true")
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        self._cached_dispatch(
            buffer, handle, fixture,
            async_mode=async_mode, allocate=allocate)
        return {"async": async_mode, "allocate_on_comm_stream": allocate}

    def _run_combine_case(self, case):
        async_mode = "-async-" in case
        allocate = case.endswith("-true")
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        recv_x, _, _, _ = self._cached_dispatch(
            buffer, handle, fixture, async_mode=False, allocate=False)
        combined_x, combined_weights, event = buffer.combine(
            recv_x,
            handle,
            num_sms=1,
            num_qps=0,
            async_with_compute_stream=async_mode,
            allocate_on_comm_stream=allocate,
        )
        if async_mode:
            _check(event.event is not None,
                   "async combine did not return a native event")
            event.current_stream_wait()
        else:
            _check(event.event is None,
                   "synchronous combine returned a native event")
        expected = self.torch.tensor(
            fixture["payloads"][self.rank], dtype=self.torch.bfloat16)
        self._assert_tensor(combined_x, expected, "combined_x")
        _check(combined_weights is None,
               "weightless combine returned weights")
        return {"async": async_mode, "allocate_on_comm_stream": allocate}

    def _run_previous_event(self):
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        x, _ = self._materialize(fixture)
        x.add_(1)
        previous = self.deep_ep.ElasticBuffer.capture()
        expected_fixture = _offset_fixture(REGULAR_FIXTURE, 101)
        expected_x, expected_routes = _literal_reference(
            self.torch, self.dist, self.group, self.rank, expected_fixture)
        recv_x, recv_routes, recv_weights, returned_handle, event = \
            buffer.dispatch(
                x,
                handle=handle,
                num_sms=1,
                num_qps=0,
                previous_event=previous,
                async_with_compute_stream=True,
                allocate_on_comm_stream=True,
            )
        event.current_stream_wait()
        _check(returned_handle is handle, "previous-event dispatch handle changed")
        self._assert_tensor(recv_x, expected_x, "previous-event recv_x")
        self._assert_tensor(
            recv_routes, expected_routes, "previous-event recv_topk_idx")
        _check(recv_weights is None, "previous-event dispatch returned weights")

        recv_x.add_(2)
        combine_previous = self.deep_ep.ElasticBuffer.capture()
        combined_x, combined_weights, combined_event = buffer.combine(
            recv_x,
            handle,
            num_sms=1,
            num_qps=0,
            previous_event=combine_previous,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
        combined_event.current_stream_wait()
        expected_combined = self.torch.tensor(
            expected_fixture["payloads"][self.rank],
            dtype=self.torch.bfloat16).add(2)
        self._assert_tensor(
            combined_x, expected_combined, "previous-event combined_x")
        _check(combined_weights is None,
               "previous-event combine returned weights")
        return {"dispatch_previous_event": True,
                "combine_previous_event": True}

    def _run_route_case(self, fixture):
        buffer = self.new_buffer()
        handle = self._seed(buffer, fixture)
        changed = _offset_fixture(fixture, 100)
        recv_x, _, _, _ = self._cached_dispatch(
            buffer, handle, changed, async_mode=True, allocate=True)
        combined_x, combined_weights, event = buffer.combine(
            recv_x, handle, num_sms=1, num_qps=0,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True)
        event.current_stream_wait()
        expected = self.torch.tensor(
            changed["payloads"][self.rank], dtype=self.torch.bfloat16
        ).reshape(len(changed["payloads"][self.rank]), HIDDEN)
        self._assert_tensor(combined_x, expected, "route combined_x")
        _check(combined_weights is None, "route combine returned weights")
        return {"local_tokens": len(changed["payloads"][self.rank])}

    def _run_generations(self):
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        for generation in range(1, 101):
            fixture = _offset_fixture(REGULAR_FIXTURE, generation)
            recv_x, _, _, _ = self._cached_dispatch(
                buffer, handle, fixture,
                async_mode=True, allocate=generation % 2 == 0)
            combined_x, combined_weights, event = buffer.combine(
                recv_x, handle, num_sms=1, num_qps=0,
                async_with_compute_stream=True,
                allocate_on_comm_stream=generation % 2 == 1)
            event.current_stream_wait()
            expected = self.torch.tensor(
                fixture["payloads"][self.rank], dtype=self.torch.bfloat16)
            self._assert_tensor(
                combined_x, expected, f"generation {generation} combined_x")
            _check(combined_weights is None,
                   f"generation {generation} returned weights")
        return {"generations": 100}

    def _run_independent_buffers(self):
        first = self.new_buffer()
        second = self.new_buffer()
        first_handle = self._seed(first)
        second_handle = self._seed(second)
        first_fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        second_fixture = _offset_fixture(REGULAR_FIXTURE, 200)
        first_result = self._cached_dispatch(
            first, first_handle, first_fixture,
            async_mode=True, allocate=True, wait=False)
        second_result = self._cached_dispatch(
            second, second_handle, second_fixture,
            async_mode=True, allocate=True, wait=False)
        first_result[1].current_stream_wait()
        second_result[1].current_stream_wait()
        self._assert_tensor(first_result[0], first_result[2], "buffer-a recv_x")
        self._assert_tensor(second_result[0], second_result[2], "buffer-b recv_x")
        return {"independent_buffers": 2}

    def _run_diagnostic_failure(self):
        buffer = self.new_buffer()
        local_failure = None
        if self.rank == 0:
            os.environ["DEEP_EP_ASCEND_TEST_DIAGNOSTIC"] = \
                "completion_timeout"
        try:
            try:
                buffer.barrier(use_comm_stream=True)
            except RuntimeError as error:
                local_failure = str(error)
        finally:
            os.environ.pop("DEEP_EP_ASCEND_TEST_DIAGNOSTIC", None)
        reports = [None] * WORLD_SIZE
        self.dist.all_gather_object(
            reports,
            {"rank": self.rank, "failure": local_failure},
            group=self.group,
        )
        aggregate = _aggregate_rank_failures(reports)
        _check("rank 0:" in aggregate and "completion_timeout" in aggregate,
               f"rank-qualified injected diagnostic missing: {aggregate}")
        try:
            buffer.destroy()
        except RuntimeError as error:
            _check("completion_timeout" in str(error),
                   f"destroy lost diagnostic failure: {error}")
        self.buffers.remove(buffer)
        return {"aggregated_failure": aggregate}

    def _run_completion_mismatch(self):
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        recv_x, event, expected_x, _ = self._cached_dispatch(
            buffer, handle, fixture,
            async_mode=True, allocate=True, wait=False)
        local_failure = None
        if self.rank == 0:
            os.environ["DEEP_EP_ASCEND_TEST_COMPLETION_FAULT"] = \
                "completion_mismatch"
        try:
            try:
                event.current_stream_wait()
            except RuntimeError as error:
                local_failure = str(error)
        finally:
            os.environ.pop("DEEP_EP_ASCEND_TEST_COMPLETION_FAULT", None)

        reports = [None] * WORLD_SIZE
        self.dist.all_gather_object(
            reports,
            {"rank": self.rank, "failure": local_failure},
            group=self.group,
        )
        aggregate = _aggregate_rank_failures(reports)
        expected_rank_zero = reports[0]["failure"]
        _check(expected_rank_zero is not None and
               "device completion generation mismatch" in expected_rank_zero,
               f"rank 0 completion mismatch is missing: {aggregate}")
        _check(reports[1]["failure"] is None,
               f"rank 1 unexpectedly failed: {aggregate}")
        self._assert_tensor(recv_x, expected_x, "completion mismatch recv_x")

        destroy_failure = None
        try:
            buffer.destroy()
        except RuntimeError as error:
            destroy_failure = str(error)
        _check(buffer.runtime is None,
               "completion mismatch buffer retained destroyed runtime")
        if self.rank == 0:
            _check(destroy_failure is not None and
                   "device completion generation mismatch" in destroy_failure,
                   f"destroy lost completion mismatch: {destroy_failure}")
        else:
            _check(destroy_failure is None,
                   f"healthy rank destroy failed: {destroy_failure}")
        self.buffers.remove(buffer)
        return {
            "local_failure": local_failure,
            "aggregated_failure": aggregate,
            "destroy_failure": destroy_failure,
            "runtime_released": True,
        }

    def _run_drop_event(self):
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        first_fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        first_x, dropped_event, first_expected, _ = self._cached_dispatch(
            buffer, handle, first_fixture,
            async_mode=True, allocate=True, wait=False)
        del dropped_event
        gc.collect()
        self._assert_tensor(first_x, first_expected, "dropped-event recv_x")

        handle = self._seed(buffer)
        second_fixture = _offset_fixture(REGULAR_FIXTURE, 200)
        second_x, _, second_expected, _ = self._cached_dispatch(
            buffer, handle, second_fixture,
            async_mode=True, allocate=True, wait=True)
        self._assert_tensor(second_x, second_expected, "post-drop recv_x")
        return {
            "event_dropped_without_wait": True,
            "garbage_collection_completed": True,
            "buffer_reused": True,
        }

    def _run_destroy_pending_retry(self):
        buffer = self.new_buffer()
        handle = self._seed(buffer)
        fixture = _offset_fixture(REGULAR_FIXTURE, 100)
        _, event, _, _ = self._cached_dispatch(
            buffer, handle, fixture,
            async_mode=True, allocate=True, wait=False)

        first_failure = None
        os.environ["DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT"] = \
            "destroy_failure"
        try:
            try:
                buffer.destroy()
            except RuntimeError as error:
                first_failure = str(error)
        finally:
            os.environ.pop("DEEP_EP_ASCEND_TEST_STREAM_EVENT_FAULT", None)
        _check(first_failure is not None and "destroy_event" in first_failure and
               "backend error -2" in first_failure,
               f"first destroy did not expose native failure: {first_failure}")
        _check(buffer.runtime is not None,
               "failed destroy released Python runtime ownership")

        retry_failure = None
        try:
            buffer.destroy()
        except RuntimeError as error:
            retry_failure = str(error)
        _check(retry_failure is not None and "destroy_event" in retry_failure,
               f"retry lost stable destroy failure: {retry_failure}")
        _check(buffer.runtime is None,
               "destroy retry did not release Python runtime ownership")
        self.buffers.remove(buffer)
        del event
        gc.collect()
        return {
            "first_destroy_failure": first_failure,
            "retry_failure": retry_failure,
            "runtime_retained_after_first_failure": True,
            "runtime_released_after_retry": True,
        }

    def _make_overlap_communication_inputs(
            self, tokens=OVERLAP_COMMUNICATION_TOKENS):
        hidden = OVERLAP_COMMUNICATION_HIDDEN
        cpu = self.torch.arange(
            tokens * hidden, dtype=self.torch.float32).reshape(
                tokens, hidden).remainder_(97).to(self.torch.bfloat16)
        x = cpu.to(self.device)
        peer_expert = 2 if self.rank == 0 else 0
        routes = self.torch.full(
            (tokens, 1), peer_expert, dtype=self.torch.int64,
            device=self.device)
        return x, routes

    def _make_overlap_inputs(self, tokens=OVERLAP_COMMUNICATION_TOKENS):
        x, routes = self._make_overlap_communication_inputs(tokens)
        hidden = OVERLAP_DIAGNOSTIC_COMPUTE_SHAPE[0]
        left = self.torch.arange(
            hidden * hidden, dtype=self.torch.float32).reshape(
                hidden, hidden).remainder_(31).div_(31).to(
                    self.torch.bfloat16).to(self.device)
        right = self.torch.arange(
            hidden * hidden, dtype=self.torch.float32).reshape(
                hidden, hidden).remainder_(29).div_(29).to(
                    self.torch.bfloat16).to(self.device)
        return x, routes, left, right

    def _wait_current_stream(self):
        self.deep_ep.ElasticBuffer.capture().current_stream_wait()

    def _compute(self, left, right):
        value = left
        for _ in range(OVERLAP_DIAGNOSTIC_COMPUTE_ITERATIONS):
            value = self.torch.matmul(value, right).mul_(0.001)
        return value

    def _make_formal_compute_inputs(self):
        input_tensor = self.torch.ones(
            OVERLAP_COMPUTE_INPUT_SHAPE,
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        weight = self.torch.ones(
            OVERLAP_COMPUTE_WEIGHT_SHAPE,
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        return input_tensor, weight

    def _formal_compute(self, input_tensor, weight):
        value = None
        for _ in range(OVERLAP_COMPUTE_ITERATIONS):
            value = self.torch.nn.functional.conv3d(
                input_tensor,
                weight,
                bias=None,
                stride=OVERLAP_COMPUTE_STRIDE,
                padding=OVERLAP_COMPUTE_PADDING,
                dilation=OVERLAP_COMPUTE_DILATION,
                groups=OVERLAP_COMPUTE_GROUPS,
            )
        return value

    def _make_component_compute_inputs(self):
        input_tensor = self.torch.ones(
            OVERLAP_COMPONENT_INPUT_SHAPE,
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        weight = self.torch.ones(
            OVERLAP_COMPONENT_WEIGHT_SHAPE,
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        return input_tensor, weight

    def _component_compute(self, input_tensor, weight, *, iterations):
        value = None
        for _ in range(iterations):
            value = self.torch.nn.functional.conv3d(
                input_tensor,
                weight,
                bias=None,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                groups=1,
            )
        return value

    def _calibrate_component_compute(self, input_tensor, weight):
        value = self._component_compute(
            input_tensor,
            weight,
            iterations=OVERLAP_COMPONENT_CALIBRATION_WARMUP_ITERATIONS,
        )
        self._record_tail_event().synchronize()

        started = time.monotonic()
        value = self._component_compute(
            input_tensor,
            weight,
            iterations=OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS,
        )
        self._record_tail_event().synchronize()
        measured_seconds = time.monotonic() - started
        _check(value.device.type == "npu", "compute result left the NPU")
        _check(math.isfinite(measured_seconds) and measured_seconds > 0,
               "component compute calibration did not produce a duration")
        scaled_iterations = round(
            OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS *
            OVERLAP_COMPONENT_TARGET_SECONDS / measured_seconds)
        selected_iterations = min(
            OVERLAP_COMPONENT_MAX_ITERATIONS,
            max(OVERLAP_COMPONENT_MIN_ITERATIONS, scaled_iterations),
        )
        return {
            "completion_wait": "npu-event-synchronize",
            "initial_iterations":
                OVERLAP_COMPONENT_CALIBRATION_INITIAL_ITERATIONS,
            "iteration_bounds": [
                OVERLAP_COMPONENT_MIN_ITERATIONS,
                OVERLAP_COMPONENT_MAX_ITERATIONS,
            ],
            "measured_initial_seconds": measured_seconds,
            "selected_iterations": selected_iterations,
            "target_range_seconds":
                list(OVERLAP_COMPONENT_TARGET_RANGE_SECONDS),
            "target_seconds": OVERLAP_COMPONENT_TARGET_SECONDS,
            "warmup_iterations":
                OVERLAP_COMPONENT_CALIBRATION_WARMUP_ITERATIONS,
        }

    def _light_compute(self, left, right):
        extent = 512
        return self.torch.matmul(
            left[:extent, :extent], right[:extent, :extent]).mul_(0.001)

    def _record_tail_event(self, stream=None):
        event = self.torch.npu.Event()
        if stream is None:
            event.record()
        else:
            with self.torch.npu.stream(stream):
                event.record()
        return event

    @staticmethod
    def _query_tail(event):
        return bool(event.query())

    def _run_overlap_diagnostic_variant(
            self, variant, x, routes, left, right, hint):
        variant_started = time.monotonic()
        setup_started = time.monotonic()
        buffer = self.new_buffer(num_bytes=hint)
        compute_stream = self.torch.npu.current_stream()
        comm_stream = buffer.get_comm_stream()
        _check(compute_stream.stream_id != comm_stream.stream_id,
               "diagnostic compute and communication streams alias")

        handle = None
        changed_x = None
        if variant != "pure-compute":
            seed = buffer.dispatch(
                x,
                topk_idx=routes,
                num_experts=NUM_EXPERTS,
                num_max_tokens_per_rank=x.shape[0],
                expert_alignment=1,
                num_sms=1,
                num_qps=0,
                do_handle_copy=True,
                do_cpu_sync=True,
            )
            handle = seed[3]
            changed_x = x.add(1)
        setup_seconds = time.monotonic() - setup_started

        dispatch_seconds = 0.0
        operation_event = None
        dispatch_started = time.monotonic()
        if variant != "pure-compute":
            _, _, _, _, operation_event = buffer.dispatch(
                changed_x,
                handle=handle,
                num_sms=1,
                num_qps=0,
                async_with_compute_stream=True,
                allocate_on_comm_stream=True,
            )
            _check(operation_event.event is not None,
                   "diagnostic dispatch omitted its native event")
            dispatch_seconds = time.monotonic() - dispatch_started
        comm_tail = self._record_tail_event(comm_stream)
        comm_tail_immediate = self._query_tail(comm_tail)

        compute_started = time.monotonic()
        value = left
        if variant in ("pure-compute", "combined-heavy"):
            value = self._compute(left, right)
        elif variant == "combined-light":
            value = self._light_compute(left, right)
        compute_seconds = time.monotonic() - compute_started
        compute_tail = self._record_tail_event()
        if variant == "pure-compute":
            operation_event = self.deep_ep.ElasticBuffer.capture()
        if variant == "post-dispatch-idle":
            time.sleep(OVERLAP_DIAGNOSTIC_IDLE_SECONDS)

        comm_tail_before_wait = self._query_tail(comm_tail)
        order = ["dispatch-return", "compute-enqueued", "event-wait-start"]
        wait_started = time.monotonic()
        wait_failure = None
        try:
            operation_event.current_stream_wait()
            wait_outcome = "completed"
        except RuntimeError as error:
            wait_failure = str(error)
            wait_outcome = "timeout" if (
                "synchronize_event" in wait_failure and
                "backend error -1" in wait_failure) else "error"
        wait_seconds = time.monotonic() - wait_started
        order.append("event-wait-end")

        comm_tail_after_wait = self._query_tail(comm_tail)
        comm_first_complete = (
            time.monotonic() - variant_started
            if comm_tail_after_wait or comm_tail_immediate else None)
        compute_complete = self._query_tail(compute_tail)
        compute_first_complete = (
            time.monotonic() - variant_started if compute_complete else None)
        grace_deadline = time.monotonic() + OVERLAP_DIAGNOSTIC_GRACE_SECONDS
        while time.monotonic() < grace_deadline and not (
                comm_tail_after_wait and compute_complete):
            if not comm_tail_after_wait:
                comm_tail_after_wait = self._query_tail(comm_tail)
                if comm_tail_after_wait and comm_first_complete is None:
                    comm_first_complete = time.monotonic() - variant_started
            if not compute_complete:
                compute_complete = self._query_tail(compute_tail)
                if compute_complete and compute_first_complete is None:
                    compute_first_complete = time.monotonic() - variant_started
            if not (comm_tail_after_wait and compute_complete):
                time.sleep(0.01)
        order.append("query-grace-end")
        _check(value.device.type == "npu", "diagnostic compute left the NPU")
        return {
            "rank": self.rank,
            "buffer_instance": f"rank-{self.rank}:{variant}:{id(buffer)}",
            "compute_stream_id": compute_stream.stream_id,
            "communication_stream_id": comm_stream.stream_id,
            "stage_seconds": {
                "setup": setup_seconds,
                "dispatch_return": dispatch_seconds,
                "compute_enqueue": compute_seconds,
                "event_wait": wait_seconds,
            },
            "event_wait": {
                "outcome": wait_outcome,
                "bound_seconds": 5.0,
                "failure": wait_failure,
            },
            "queries": {
                "comm_tail_immediate": comm_tail_immediate,
                "comm_tail_before_wait": comm_tail_before_wait,
                "comm_tail_after_wait": comm_tail_after_wait,
                "comm_tail_after_grace": comm_tail_after_wait,
                "comm_tail_first_complete_seconds": comm_first_complete,
                "compute_tail_after_grace": compute_complete,
                "compute_tail_first_complete_seconds": compute_first_complete,
            },
            "order": order,
        }

    def _overlap_iteration(self, buffer, handle, x, left, right, overlap):
        started = time.monotonic()
        _, _, _, _, event = buffer.dispatch(
            x,
            handle=handle,
            num_sms=1,
            num_qps=0,
            async_with_compute_stream=overlap,
            allocate_on_comm_stream=True,
        )
        value = self._compute(left, right)
        if overlap:
            event.current_stream_wait()
        else:
            _check(event.event is None,
                   "serialized dispatch returned a native event")
        self._wait_current_stream()
        _check(value.device.type == "npu", "compute result left the NPU")
        return time.monotonic() - started

    def _formal_overlap_iteration(
            self, buffer, handle, x, input_tensor, weight, overlap):
        started = time.monotonic()
        _, _, _, _, event = buffer.dispatch(
            x,
            handle=handle,
            num_sms=1,
            num_qps=0,
            async_with_compute_stream=overlap,
            allocate_on_comm_stream=True,
        )
        value = self._formal_compute(input_tensor, weight)
        if overlap:
            event.current_stream_wait()
        else:
            _check(event.event is None,
                   "serialized dispatch returned a native event")
        self._wait_current_stream()
        _check(value.device.type == "npu", "compute result left the NPU")
        return time.monotonic() - started

    def _communication_iteration(self, buffer, handle, x):
        started = time.monotonic()
        _, _, _, _, event = buffer.dispatch(
            x,
            handle=handle,
            num_sms=1,
            num_qps=0,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        self._wait_current_stream()
        return time.monotonic() - started

    def _compute_iteration(self, left, right):
        started = time.monotonic()
        value = self._compute(left, right)
        self._wait_current_stream()
        _check(value.device.type == "npu", "compute result left the NPU")
        return time.monotonic() - started

    def _component_communication_iteration(self, buffer, handle, x):
        started = time.monotonic()
        dispatch_started = time.monotonic()
        _, _, _, _, event = buffer.dispatch(
            x,
            handle=handle,
            num_sms=1,
            num_qps=0,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
        dispatch_seconds = time.monotonic() - dispatch_started
        _check(event.event is not None,
               "component async dispatch omitted its native event")
        wait_started = time.monotonic()
        event.current_stream_wait()
        wait_seconds = time.monotonic() - wait_started
        return {
            "duration_seconds": time.monotonic() - started,
            "stage_seconds": {
                "async_dispatch_return": dispatch_seconds,
                "event_wait": wait_seconds,
            },
        }

    def _component_compute_iteration(
            self, input_tensor, weight, iterations):
        started = time.monotonic()
        enqueue_started = time.monotonic()
        value = self._component_compute(
            input_tensor, weight, iterations=iterations)
        enqueue_seconds = time.monotonic() - enqueue_started
        completion_started = time.monotonic()
        self._wait_current_stream()
        completion_seconds = time.monotonic() - completion_started
        _check(value.device.type == "npu", "compute result left the NPU")
        return {
            "duration_seconds": time.monotonic() - started,
            "stage_seconds": {
                "compute_enqueue": enqueue_seconds,
                "compute_completion": completion_seconds,
            },
        }

    def _component_combined_iteration(
            self, buffer, handle, x, input_tensor, weight, iterations, *,
            overlap):
        started = time.monotonic()
        dispatch_started = time.monotonic()
        _, _, _, _, event = buffer.dispatch(
            x,
            handle=handle,
            num_sms=1,
            num_qps=0,
            async_with_compute_stream=overlap,
            allocate_on_comm_stream=True,
        )
        dispatch_seconds = time.monotonic() - dispatch_started
        stages = {
            "async_dispatch_return" if overlap else
            "serialized_dispatch_return": dispatch_seconds,
        }
        if overlap:
            _check(event.event is not None,
                   "component async dispatch omitted its native event")
        else:
            _check(event.event is None,
                   "component serialized dispatch returned a native event")

        enqueue_started = time.monotonic()
        value = self._component_compute(
            input_tensor, weight, iterations=iterations)
        stages["compute_enqueue"] = time.monotonic() - enqueue_started
        if overlap:
            wait_started = time.monotonic()
            event.current_stream_wait()
            stages["event_wait"] = time.monotonic() - wait_started
        completion_started = time.monotonic()
        self._wait_current_stream()
        stages["compute_completion"] = \
            time.monotonic() - completion_started
        _check(value.device.type == "npu", "compute result left the NPU")
        return {
            "duration_seconds": time.monotonic() - started,
            "stage_seconds": stages,
        }

    def _profile_overlap(
            self, buffer, handle, x, left, right, *, component=False,
            formal=False, component_iterations=None):
        _check(not (component and formal),
               "overlap profiler compute variants are mutually exclusive")
        profiler = self.torch_npu.profiler
        activity = profiler.ProfilerActivity.NPU
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace_dir / f"overlap-rank{self.rank}.json"
        with profiler.profile(activities=[activity]) as profile:
            if component:
                _check(isinstance(component_iterations, int),
                       "component profiler iterations are missing")
                self._component_combined_iteration(
                    buffer, handle, x, left, right, component_iterations,
                    overlap=True)
            elif formal:
                self._formal_overlap_iteration(
                    buffer, handle, x, left, right, overlap=True)
            else:
                self._overlap_iteration(
                    buffer, handle, x, left, right, overlap=True)
        profile.export_chrome_trace(str(trace_path))
        compute_stream = self.torch.npu.current_stream()
        comm_stream = buffer.get_comm_stream()
        try:
            if component:
                interval = _find_component_npu_overlap_interval((trace_path,))
            elif formal:
                interval = _find_formal_npu_overlap_interval((trace_path,))
            else:
                interval = _find_npu_overlap_interval((trace_path,))
        except Exception as error:
            failure = str(error) if isinstance(error, AssertionError) else \
                f"{type(error).__name__}: {error}"
            interval = {"overlap_us": 0.0, "failure": failure}
        interval["trace"] = str(trace_path)
        interval["logical_compute_stream_id"] = compute_stream.stream_id
        interval["logical_communication_stream_id"] = comm_stream.stream_id
        return interval

    def _run_overlap(self):
        x, routes = self._make_overlap_communication_inputs()
        input_tensor, weight = self._make_formal_compute_inputs()
        hint = self.deep_ep.ElasticBuffer.get_buffer_size_hint(
            self.group,
            num_max_tokens_per_rank=x.shape[0],
            hidden=x.shape[1],
            num_topk=1,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
        )
        buffer = self.new_buffer(num_bytes=hint)
        seed = buffer.dispatch(
            x,
            topk_idx=routes,
            num_experts=NUM_EXPERTS,
            num_max_tokens_per_rank=x.shape[0],
            expert_alignment=1,
            num_sms=1,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        )
        handle = seed[3]
        changed_x = x.add(1)
        for _ in range(OVERLAP_WARMUPS):
            self._formal_overlap_iteration(
                buffer, handle, changed_x, input_tensor, weight, overlap=False)
            self._formal_overlap_iteration(
                buffer, handle, changed_x, input_tensor, weight, overlap=True)

        serialized = []
        overlapped = []
        for _ in range(OVERLAP_REPETITIONS):
            serialized.append(self._formal_overlap_iteration(
                buffer, handle, changed_x, input_tensor, weight,
                overlap=False))
            overlapped.append(self._formal_overlap_iteration(
                buffer, handle, changed_x, input_tensor, weight,
                overlap=True))
        serialized_summary = _summary(serialized)
        overlapped_summary = _summary(overlapped)
        improvement = 1.0 - (
            overlapped_summary["median_seconds"] /
            serialized_summary["median_seconds"])
        interval = self._profile_overlap(
            buffer, handle, changed_x, input_tensor, weight, formal=True)
        measurement = {
            "warmups": OVERLAP_WARMUPS,
            "repetitions": OVERLAP_REPETITIONS,
            "communication_tokens": OVERLAP_COMMUNICATION_TOKENS,
            "communication_hidden": OVERLAP_COMMUNICATION_HIDDEN,
            "communication_num_sms": 1,
            "compute_variant": OVERLAP_COMPUTE_VARIANT,
            "compute_dtype": "bfloat16",
            "compute_layout": "NCDHW",
            "compute_input_shape": list(OVERLAP_COMPUTE_INPUT_SHAPE),
            "compute_weight_shape": list(OVERLAP_COMPUTE_WEIGHT_SHAPE),
            "compute_stride": list(OVERLAP_COMPUTE_STRIDE),
            "compute_padding": list(OVERLAP_COMPUTE_PADDING),
            "compute_dilation": list(OVERLAP_COMPUTE_DILATION),
            "compute_groups": OVERLAP_COMPUTE_GROUPS,
            "compute_iterations": OVERLAP_COMPUTE_ITERATIONS,
            "serialized": serialized_summary,
            "overlapped": overlapped_summary,
            "median_improvement": improvement,
            "minimum_median_improvement":
                OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
            "maximum_auxiliary_aiv_ratio":
                OVERLAP_MAX_AUXILIARY_AIV_RATIO,
            "event_wait_limit_seconds": OVERLAP_EVENT_WAIT_LIMIT_SECONDS,
            "profiler_overlap": interval,
            "global_synchronizations": 0,
        }
        for key in (
                "logical_compute_stream_id",
                "logical_communication_stream_id",
                "physical_compute_stream_id",
                "physical_communication_stream_id"):
            if key in interval:
                measurement[key] = interval[key]
        failures = []
        if improvement < OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT:
            failures.append(
                "overlap median improvement "
                f"{improvement:.6f} is below "
                f"{OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT:.6f}")
        failures.extend(_formal_profiler_failures(interval))
        if failures:
            measurement["acceptance_failure"] = "; ".join(failures)
        return measurement

    def _run_overlap_sweep_point(self, tokens, record_phase=None):
        x, routes, left, right = self._make_overlap_inputs(tokens)
        hint = self.deep_ep.ElasticBuffer.get_buffer_size_hint(
            self.group,
            num_max_tokens_per_rank=x.shape[0],
            hidden=x.shape[1],
            num_topk=1,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
        )
        buffer = self.new_buffer(num_bytes=hint)
        seed = buffer.dispatch(
            x,
            topk_idx=routes,
            num_experts=NUM_EXPERTS,
            num_max_tokens_per_rank=x.shape[0],
            expert_alignment=1,
            num_sms=1,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        )
        handle = seed[3]
        changed_x = x.add(1)
        measurement = {
            "rank": self.rank,
            "tokens": tokens,
            "buffer_instance":
                f"rank-{self.rank}:sweep-{tokens}:{id(buffer)}",
            "event_wait_completed": True,
        }

        def checkpoint_phase(phase, status):
            measurement["active_phase"] = phase if status == "started" else None
            measurement["phase_status"] = status
            if status == "completed":
                measurement["completed_phase"] = phase
            if record_phase is not None:
                record_phase(phase, measurement)

        for _ in range(OVERLAP_SWEEP_WARMUPS):
            self._communication_iteration(buffer, handle, changed_x)
            self._compute_iteration(left, right)
            self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=False)
            self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=True)

        checkpoint_phase("communication-only", "started")
        communication_summary = _summary([
            self._communication_iteration(buffer, handle, changed_x)
            for _ in range(OVERLAP_SWEEP_REPETITIONS)
        ])
        measurement["communication"] = communication_summary
        checkpoint_phase("communication-only", "completed")

        checkpoint_phase("compute-only", "started")
        compute_summary = _summary([
            self._compute_iteration(left, right)
            for _ in range(OVERLAP_SWEEP_REPETITIONS)
        ])
        measurement["compute"] = compute_summary
        component_total = (
            communication_summary["median_seconds"] +
            compute_summary["median_seconds"])
        _check(component_total > 0, "overlap sweep component time is zero")
        theoretical = min(
            communication_summary["median_seconds"],
            compute_summary["median_seconds"]) / component_total
        measurement["theoretical_maximum_improvement"] = theoretical
        checkpoint_phase("compute-only", "completed")

        checkpoint_phase("serialized", "started")
        serialized_summary = _summary([
            self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=False)
            for _ in range(OVERLAP_SWEEP_REPETITIONS)
        ])
        measurement["serialized"] = serialized_summary
        checkpoint_phase("serialized", "completed")

        checkpoint_phase("overlapped", "started")
        overlapped_summary = _summary([
            self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=True)
            for _ in range(OVERLAP_SWEEP_REPETITIONS)
        ])
        measurement["overlapped"] = overlapped_summary
        observed = 1.0 - (
            overlapped_summary["median_seconds"] /
            serialized_summary["median_seconds"])
        measurement["median_improvement"] = observed
        checkpoint_phase("overlapped", "completed")

        checkpoint_phase("profiler", "started")
        profiler = self._profile_overlap(
            buffer, handle, changed_x, left, right)
        for key in (
                "logical_compute_stream_id",
                "logical_communication_stream_id",
                "physical_compute_stream_id",
                "physical_communication_stream_id"):
            if key in profiler:
                measurement[key] = profiler[key]
        measurement["profiler_overlap"] = profiler
        checkpoint_phase("profiler", "completed")
        return measurement

    def _run_overlap_component_diagnostic(self, record_phase=None):
        _check(OVERLAP_COMPONENT_WARMUPS == 0,
               "overlap component diagnostic warmups changed")
        _check(OVERLAP_COMPONENT_REPETITIONS == 1,
               "overlap component diagnostic repetitions changed")
        x, routes, _, _ = self._make_overlap_inputs(
            OVERLAP_COMMUNICATION_TOKENS)
        input_tensor, weight = self._make_component_compute_inputs()
        measurement = {
            "rank": self.rank,
            "communication_tokens": OVERLAP_COMMUNICATION_TOKENS,
            "compute_input_shape": list(OVERLAP_COMPONENT_INPUT_SHAPE),
            "compute_variant": OVERLAP_COMPONENT_COMPUTE_VARIANT,
            "compute_weight_shape": list(OVERLAP_COMPONENT_WEIGHT_SHAPE),
            "warmups": OVERLAP_COMPONENT_WARMUPS,
            "repetitions": OVERLAP_COMPONENT_REPETITIONS,
            "stage_seconds": {},
        }

        def checkpoint_phase(phase, status):
            measurement["active_phase"] = \
                phase if status == "started" else None
            measurement["phase_status"] = status
            if status == "completed":
                measurement["completed_phase"] = phase
            if record_phase is not None:
                record_phase(phase, measurement)

        checkpoint_phase("calibration", "started")
        calibration = self._calibrate_component_compute(input_tensor, weight)
        iterations = calibration["selected_iterations"]
        measurement["compute_calibration"] = calibration
        measurement["compute_iterations"] = iterations
        checkpoint_phase("calibration", "completed")

        hint = self.deep_ep.ElasticBuffer.get_buffer_size_hint(
            self.group,
            num_max_tokens_per_rank=x.shape[0],
            hidden=x.shape[1],
            num_topk=1,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
        )
        buffer = self.new_buffer(num_bytes=hint)
        measurement["buffer_instance"] = \
            f"rank-{self.rank}:component:{id(buffer)}"
        seed = buffer.dispatch(
            x,
            topk_idx=routes,
            num_experts=NUM_EXPERTS,
            num_max_tokens_per_rank=x.shape[0],
            expert_alignment=1,
            num_sms=1,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        )
        handle = seed[3]
        changed_x = x.add(1)

        checkpoint_phase("communication-only", "started")
        communication = self._component_communication_iteration(
            buffer, handle, changed_x)
        measurement["communication_only_seconds"] = \
            communication["duration_seconds"]
        measurement["stage_seconds"]["communication-only"] = \
            communication["stage_seconds"]
        checkpoint_phase("communication-only", "completed")

        checkpoint_phase("compute-only", "started")
        compute = self._component_compute_iteration(
            input_tensor, weight, iterations)
        measurement["compute_only_seconds"] = compute["duration_seconds"]
        measurement["stage_seconds"]["compute-only"] = \
            compute["stage_seconds"]
        checkpoint_phase("compute-only", "completed")

        checkpoint_phase("serialized", "started")
        serialized = self._component_combined_iteration(
            buffer, handle, changed_x, input_tensor, weight, iterations,
            overlap=False)
        measurement["serialized_seconds"] = serialized["duration_seconds"]
        measurement["stage_seconds"]["serialized"] = \
            serialized["stage_seconds"]
        measurement["serialized_dispatch_return_duration_seconds"] = \
            serialized["stage_seconds"]["serialized_dispatch_return"]
        checkpoint_phase("serialized", "completed")

        checkpoint_phase("overlapped", "started")
        overlapped = self._component_combined_iteration(
            buffer, handle, changed_x, input_tensor, weight, iterations,
            overlap=True)
        measurement["overlapped_seconds"] = overlapped["duration_seconds"]
        measurement["stage_seconds"]["overlapped"] = \
            overlapped["stage_seconds"]
        measurement["async_dispatch_return_duration_seconds"] = \
            overlapped["stage_seconds"]["async_dispatch_return"]
        measurement["event_wait_duration_seconds"] = \
            overlapped["stage_seconds"]["event_wait"]
        measurement["compute_completion_duration_seconds"] = \
            overlapped["stage_seconds"]["compute_completion"]
        checkpoint_phase("overlapped", "completed")

        checkpoint_phase("profiler", "started")
        profiler = self._profile_overlap(
            buffer, handle, changed_x, input_tensor, weight, component=True,
            component_iterations=iterations)
        measurement["profiler_overlap"] = profiler
        for key in (
                "logical_compute_stream_id",
                "logical_communication_stream_id",
                "physical_compute_stream_id",
                "physical_communication_stream_id"):
            if key in profiler:
                measurement[key] = profiler[key]
        checkpoint_phase("profiler", "completed")
        return measurement

    def run(self, case):
        with _forbid_global_sync(self.torch):
            if case.startswith("cached-dispatch-"):
                return self._run_cached_case(case)
            if case.startswith("combine-"):
                return self._run_combine_case(case)
            if case == "previous-event-allocate-true":
                return self._run_previous_event()
            if case == "empty-route":
                return self._run_route_case(EMPTY_FIXTURE)
            if case == "asymmetric-route":
                return self._run_route_case(ASYMMETRIC_FIXTURE)
            if case == "100-generations":
                return self._run_generations()
            if case == "two-independent-buffers":
                return self._run_independent_buffers()
            if case == "diagnostic-failure":
                return self._run_diagnostic_failure()
            if case == "completion-mismatch":
                return self._run_completion_mismatch()
            if case == "drop-event":
                return self._run_drop_event()
            if case == "destroy-pending-retry":
                return self._run_destroy_pending_retry()
            if case == "overlap-vs-serialized":
                return self._run_overlap()
        raise AssertionError(f"distributed case is not implemented: {case}")


def _run_distributed_batch_worker(cases, trace_dir, batch_protocol=True):
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(seconds=25))
        initialized = True
        group = dist.group.WORLD
        _check(dist.get_world_size(group) == WORLD_SIZE,
               f"async overlap requires {WORLD_SIZE} ranks")
        rank = dist.get_rank(group)
        for case in cases:
            dist.barrier(group=group)
            if rank == 0 and batch_protocol:
                print(CASE_START_PREFIX + " " + json.dumps({"case": case}),
                      flush=True)
            started = time.monotonic()
            worker = AsyncOverlapWorker(
                torch, torch_npu, dist, deep_ep, group,
                torch.device("npu", local_rank), trace_dir)
            local_failure = None
            measurements = None
            try:
                measurements = worker.run(case)
            except BaseException as error:
                local_failure = f"{type(error).__name__}: {error}"
            finally:
                try:
                    worker.destroy_buffers()
                except BaseException as error:
                    cleanup_failure = \
                        f"cleanup {type(error).__name__}: {error}"
                    local_failure = "; ".join(
                        value for value in (local_failure, cleanup_failure)
                        if value)

            reports = [None] * WORLD_SIZE
            dist.all_gather_object(
                reports,
                {
                    "rank": rank,
                    "failure": local_failure,
                    "duration_seconds": time.monotonic() - started,
                    "measurements": measurements,
                },
                group=group,
            )
            aggregate = "; ".join(value for value in (
                [_aggregate_rank_failures(reports)] +
                _measurement_failures(case, reports)
            ) if value)
            if rank == 0:
                if batch_protocol:
                    print(CASE_RESULT_PREFIX + " " + json.dumps({
                        "case": case,
                        "status": "failed" if aggregate else "passed",
                        "duration_seconds": max(
                            report["duration_seconds"] for report in reports),
                        "exit_code": 1 if aggregate else 0,
                        "failure": aggregate or None,
                        "measurements": _case_measurements(case, reports),
                    }, sort_keys=True), flush=True)
                elif not aggregate:
                    print("PHASE3E_WORKER_RESULT " + json.dumps({
                        "case": case,
                        "measurements": _case_measurements(case, reports),
                    }, sort_keys=True), flush=True)
            if aggregate:
                return 1
        return 0
    finally:
        if initialized:
            dist.destroy_process_group()


def _run_distributed_worker(case, trace_dir):
    return _run_distributed_batch_worker(
        (case,), trace_dir, batch_protocol=False)


def _run_overlap_diagnostic_worker(output, trace_dir):
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(seconds=25))
        initialized = True
        group = dist.group.WORLD
        _check(dist.get_world_size(group) == WORLD_SIZE,
               f"overlap diagnostic requires {WORLD_SIZE} ranks")
        rank = dist.get_rank(group)
        input_worker = AsyncOverlapWorker(
            torch, torch_npu, dist, deep_ep, group,
            torch.device("npu", local_rank), trace_dir)
        with _forbid_global_sync(torch):
            x, routes, left, right = input_worker._make_overlap_inputs()
            hint = deep_ep.ElasticBuffer.get_buffer_size_hint(
                group,
                num_max_tokens_per_rank=x.shape[0],
                hidden=x.shape[1],
                num_topk=1,
                use_fp8_dispatch=False,
                allow_hybrid_mode=False,
                allow_multiple_reduction=True,
            )

        variants = []
        for variant in OVERLAP_DIAGNOSTIC_VARIANTS:
            dist.barrier(group=group)
            worker = AsyncOverlapWorker(
                torch, torch_npu, dist, deep_ep, group,
                torch.device("npu", local_rank), trace_dir)
            local_measurement = None
            local_failure = None
            with _forbid_global_sync(torch):
                try:
                    local_measurement = worker._run_overlap_diagnostic_variant(
                        variant, x, routes, left, right, hint)
                except BaseException as error:
                    local_failure = f"{type(error).__name__}: {error}"
                try:
                    worker.destroy_buffers()
                except BaseException as error:
                    cleanup = f"{type(error).__name__}: {error}"
                    if local_measurement is not None:
                        local_measurement["cleanup_failure"] = cleanup
                    else:
                        local_failure = "; ".join(
                            value for value in (local_failure, cleanup) if value)
            if local_measurement is None:
                local_measurement = {
                    "rank": rank,
                    "diagnostic_failure": local_failure or
                    "diagnostic variant produced no measurement",
                }
            elif local_failure:
                local_measurement["diagnostic_failure"] = local_failure

            reports = [None] * WORLD_SIZE
            dist.all_gather_object(reports, local_measurement, group=group)
            if rank == 0:
                row = {
                    "variant": variant,
                    "ranks": sorted(reports, key=lambda value: value["rank"]),
                }
                variants.append(row)
                checkpoint = _overlap_diagnostic_report(variants)
                _write_suite_report(output, checkpoint)
                print("PHASE3E_OVERLAP_DIAGNOSTIC_VARIANT " +
                      json.dumps(row, sort_keys=True), flush=True)

        if rank == 0:
            payload = {"variants": variants}
            print(OVERLAP_DIAGNOSTIC_RESULT_PREFIX + " " +
                  json.dumps(payload, sort_keys=True), flush=True)
        return 0
    finally:
        if initialized:
            dist.destroy_process_group()


def _run_overlap_sweep_worker(output, trace_dir):
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(seconds=25))
        initialized = True
        group = dist.group.WORLD
        _check(dist.get_world_size(group) == WORLD_SIZE,
               f"overlap sweep requires {WORLD_SIZE} ranks")
        rank = dist.get_rank(group)
        points = []
        for tokens in OVERLAP_SWEEP_TOKENS:
            dist.barrier(group=group)
            point_trace_dir = pathlib.Path(trace_dir) / f"tokens-{tokens}"
            worker = AsyncOverlapWorker(
                torch, torch_npu, dist, deep_ep, group,
                torch.device("npu", local_rank), point_trace_dir)
            local_measurement = None
            local_failure = None

            def record_phase(phase, measurement):
                nonlocal local_measurement
                local_measurement = copy.deepcopy(measurement)
                checkpoint = {
                    "schema_version": 1,
                    "tokens": tokens,
                    "active_phase": local_measurement.get("active_phase"),
                    "completed_phase": local_measurement.get(
                        "completed_phase"),
                    "phase_status": local_measurement["phase_status"],
                    "measurement": local_measurement,
                }
                phase_path = point_trace_dir / f"phase-rank{rank}.json"
                _write_suite_report(phase_path, checkpoint)
                print("PHASE3E_OVERLAP_SWEEP_PHASE " +
                      json.dumps(checkpoint, sort_keys=True), flush=True)

            with _forbid_global_sync(torch):
                try:
                    local_measurement = worker._run_overlap_sweep_point(
                        tokens, record_phase=record_phase)
                except BaseException as error:
                    local_failure = f"{type(error).__name__}: {error}"
                try:
                    worker.destroy_buffers()
                except BaseException as error:
                    cleanup = f"cleanup {type(error).__name__}: {error}"
                    local_failure = "; ".join(
                        value for value in (local_failure, cleanup) if value)
            if local_measurement is None:
                local_measurement = {
                    "rank": rank,
                    "tokens": tokens,
                    "diagnostic_failure": local_failure or
                    "overlap sweep point produced no measurement",
                }
            elif local_failure:
                local_measurement["diagnostic_failure"] = local_failure

            reports = [None] * WORLD_SIZE
            dist.all_gather_object(reports, local_measurement, group=group)
            if rank == 0:
                point = {
                    "tokens": tokens,
                    "ranks": sorted(
                        reports, key=lambda value: value["rank"]),
                }
                points.append(point)
                checkpoint = _overlap_sweep_report(points)
                _write_suite_report(output, checkpoint)
                print("PHASE3E_OVERLAP_SWEEP_POINT " +
                      json.dumps(point, sort_keys=True), flush=True)

        if rank == 0:
            print(OVERLAP_SWEEP_RESULT_PREFIX + " " + json.dumps({
                "points": points,
            }, sort_keys=True), flush=True)
        return 0
    finally:
        if initialized:
            dist.destroy_process_group()


def _run_overlap_component_diagnostic_worker(output, trace_dir):
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    initialized = False
    try:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(seconds=25))
        initialized = True
        group = dist.group.WORLD
        _check(dist.get_world_size(group) == WORLD_SIZE,
               f"overlap component diagnostic requires {WORLD_SIZE} ranks")
        rank = dist.get_rank(group)
        dist.barrier(group=group)
        worker = AsyncOverlapWorker(
            torch, torch_npu, dist, deep_ep, group,
            torch.device("npu", local_rank), trace_dir)
        local_measurement = None
        local_failure = None

        def record_phase(phase, measurement):
            nonlocal local_measurement
            local_measurement = copy.deepcopy(measurement)
            checkpoint = {
                "schema_version": 1,
                "active_phase": local_measurement.get("active_phase"),
                "completed_phase": local_measurement.get("completed_phase"),
                "phase_status": local_measurement["phase_status"],
                "measurement": local_measurement,
            }
            phase_path = pathlib.Path(trace_dir) / f"phase-rank{rank}.json"
            _write_suite_report(phase_path, checkpoint)
            print("PHASE3E_OVERLAP_COMPONENT_PHASE " +
                  json.dumps(checkpoint, sort_keys=True), flush=True)

        with _forbid_global_sync(torch):
            try:
                local_measurement = worker._run_overlap_component_diagnostic(
                    record_phase=record_phase)
            except BaseException as error:
                local_failure = f"{type(error).__name__}: {error}"
            try:
                worker.destroy_buffers()
            except BaseException as error:
                cleanup = f"cleanup {type(error).__name__}: {error}"
                local_failure = "; ".join(
                    value for value in (local_failure, cleanup) if value)
        if local_measurement is None:
            local_measurement = {
                "rank": rank,
                "diagnostic_failure": local_failure or
                "overlap component diagnostic produced no measurement",
            }
        elif local_failure:
            local_measurement["diagnostic_failure"] = local_failure

        reports = [None] * WORLD_SIZE
        dist.all_gather_object(reports, local_measurement, group=group)
        ranks = sorted(reports, key=lambda value: value["rank"])
        if rank == 0:
            checkpoint = _overlap_component_report(ranks)
            _write_suite_report(output, checkpoint)
            print(OVERLAP_COMPONENT_RESULT_PREFIX + " " + json.dumps({
                "ranks": ranks,
            }, sort_keys=True), flush=True)
        return 0
    finally:
        if initialized:
            dist.destroy_process_group()


def _run_worker(case, trace_dir):
    if case == "capture-current-stream":
        measurements = _run_capture_worker()
        print(WORKER_RESULT_PREFIX + " " + json.dumps({
            "case": case,
            "measurements": measurements,
        }, sort_keys=True), flush=True)
        return 0
    if case == "record-failure":
        measurements = _run_record_failure_worker()
        print(WORKER_RESULT_PREFIX + " " + json.dumps({
            "case": case,
            "measurements": measurements,
        }, sort_keys=True), flush=True)
        return 0
    if case == "event-timeout":
        measurements = _run_event_timeout_worker()
        print(WORKER_RESULT_PREFIX + " " + json.dumps({
            "case": case,
            "measurements": measurements,
        }, sort_keys=True), flush=True)
        return 0
    return _run_distributed_worker(case, trace_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--suite", choices=("event", "full"))
    parser.add_argument("--worker", choices=CASE_NAMES)
    parser.add_argument(
        "--batch-worker", nargs="+", choices=DISTRIBUTED_CASES,
        metavar="CASE")
    parser.add_argument("--overlap-diagnostic", action="store_true")
    parser.add_argument("--overlap-diagnostic-worker", action="store_true")
    parser.add_argument("--overlap-sweep", action="store_true")
    parser.add_argument("--overlap-sweep-worker", action="store_true")
    parser.add_argument(
        "--overlap-component-diagnostic", action="store_true")
    parser.add_argument(
        "--overlap-component-diagnostic-worker", action="store_true")
    parser.add_argument("--output", default="/tmp/phase3e-async-overlap.json")
    parser.add_argument("--trace-dir", default="/tmp/phase3e-async-traces")
    args = parser.parse_args()
    if args.contract:
        print(json.dumps(_contract(), sort_keys=True))
        return 0
    if args.worker:
        return _run_worker(args.worker, args.trace_dir)
    if args.batch_worker:
        return _run_distributed_batch_worker(args.batch_worker, args.trace_dir)
    if args.overlap_diagnostic_worker:
        return _run_overlap_diagnostic_worker(args.output, args.trace_dir)
    if args.overlap_diagnostic:
        return _run_overlap_diagnostic(args.output, args.trace_dir)
    if args.overlap_sweep_worker:
        return _run_overlap_sweep_worker(args.output, args.trace_dir)
    if args.overlap_sweep:
        return _run_overlap_sweep(args.output, args.trace_dir)
    if args.overlap_component_diagnostic_worker:
        return _run_overlap_component_diagnostic_worker(
            args.output, args.trace_dir)
    if args.overlap_component_diagnostic:
        return _run_overlap_component_diagnostic(args.output, args.trace_dir)
    if args.suite:
        return _run_suite(args.suite, args.output, args.trace_dir)
    parser.error(
        "one of --contract, --suite, --worker, --batch-worker, "
        "--overlap-diagnostic, --overlap-diagnostic-worker, "
        "--overlap-sweep, --overlap-sweep-worker, "
        "--overlap-component-diagnostic, or "
        "--overlap-component-diagnostic-worker is required")


if __name__ == "__main__":
    raise SystemExit(main())
