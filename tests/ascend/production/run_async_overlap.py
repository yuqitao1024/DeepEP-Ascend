import argparse
import ast
import copy
import json
import math
import os
import pathlib
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
OVERLAP_COMPUTE_SHAPE = (4096, 4096)
OVERLAP_COMPUTE_ITERATIONS = 8
OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT = 0.05

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
    "drop-event",
    "destroy-pending-retry",
)

HOST_CONTRACT_CASES = {
    "record-failure": (
        "tests/ascend/test_core_operator_contract.py::"
        "AscendCoreOperatorContractTest::test_public_combine_async_probe_executes"
    ),
    "event-timeout": (
        "tests/ascend/test_core_operator_contract.py::"
        "AscendCoreOperatorContractTest::test_async_runtime_stream_event_contract"
    ),
    "completion-mismatch": (
        "tests/ascend/test_core_operator_contract.py::"
        "AscendCoreOperatorContractTest::test_async_runtime_stream_event_contract"
    ),
    "drop-event": (
        "tests/ascend/test_core_operator_contract.py::"
        "AscendCoreOperatorContractTest::test_async_runtime_stream_event_contract"
    ),
    "destroy-pending-retry": (
        "tests/ascend/test_core_operator_contract.py::"
        "AscendCoreOperatorContractTest::"
        "test_barrier_buffer_lifecycle_resource_concurrency"
    ),
}

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
            functions["_run_distributed_worker"])
        if isinstance(node, ast.Call)
    ]
    teardown = functions["_run_distributed_worker"]
    teardown_finally = [node for node in ast.walk(teardown)
                        if isinstance(node, ast.Try) and node.finalbody]
    reference_calls = [
        _call_name(node) for node in ast.walk(functions["_literal_reference"])
        if isinstance(node, ast.Call)
    ]

    _check(set(EVENT_CASES).issubset(CASE_NAMES),
           "event suite contains an unregistered case")
    _check(set(HOST_CONTRACT_CASES).issubset(CASE_NAMES),
           "host lifecycle suite contains an unregistered case")
    _check("subprocess.run" in bounded_calls,
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
            "finally-buffer-before-process-group-teardown",
            "zero-global-synchronization",
            "npu-profiler-overlap-interval",
        ],
        "event_cases": list(EVENT_CASES),
        "full_cases": list(CASE_NAMES),
        "matrix_groups": list(MATRIX_GROUPS),
        "overlap": {
            "compute_iterations": OVERLAP_COMPUTE_ITERATIONS,
            "compute_shape": list(OVERLAP_COMPUTE_SHAPE),
            "minimum_median_improvement":
                OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
            "profiler_interval_required": True,
            "repetitions": OVERLAP_REPETITIONS,
            "report_percentiles": [50, 95],
            "warmups": OVERLAP_WARMUPS,
        },
        "reference": "rank-gathered-literal-inputs-and-torch-ops",
        "world_size": WORLD_SIZE,
    }


def _run_bounded(command, timeout_seconds=CASE_TIMEOUT_SECONDS, env=None):
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        return {
            "status": "failed",
            "failure": f"process timeout after {timeout_seconds}s",
            "duration_seconds": time.monotonic() - started,
            "exit_code": None,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
        }
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "failure": None if completed.returncode == 0 else
            f"process exited {completed.returncode}",
        "duration_seconds": time.monotonic() - started,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _aggregate_rank_failures(reports):
    return "; ".join(
        f"rank {report['rank']}: {report['failure']}"
        for report in sorted(reports, key=lambda value: value["rank"])
        if report.get("failure")
    )


def _stream_id(event):
    for key, value in (event.get("args") or {}).items():
        normalized = "".join(character.lower() for character in str(key)
                             if character.isalnum())
        if normalized in ("streamid", "stream"):
            return value
    return event.get("tid")


def _find_npu_overlap_interval(
        trace_paths, compute_stream_id, comm_stream_id):
    events = []
    for trace_path in trace_paths:
        try:
            payload = json.loads(pathlib.Path(trace_path).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        events.extend(payload.get("traceEvents", ()))

    def intervals(stream_id):
        selected = []
        for event in events:
            if event.get("ph") != "X" or "ts" not in event or \
                    "dur" not in event:
                continue
            if str(_stream_id(event)) != str(stream_id):
                continue
            category = str(event.get("cat", "")).lower()
            if "npu" not in category and "kernel" not in category and \
                    "ascend" not in category:
                continue
            start = float(event["ts"])
            duration = float(event["dur"])
            if duration > 0:
                selected.append((start, start + duration,
                                 str(event.get("name", "unnamed"))))
        return selected

    best = None
    for compute in intervals(compute_stream_id):
        for communication in intervals(comm_stream_id):
            overlap = min(compute[1], communication[1]) - \
                max(compute[0], communication[0])
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, compute, communication)
    if best is None:
        raise AssertionError(
            "NPU profiler did not contain an overlapping compute/communication "
            f"interval for streams {compute_stream_id}/{comm_stream_id}")
    return {
        "overlap_us": best[0],
        "compute_event": best[1][2],
        "communication_event": best[2][2],
        "compute_interval_us": [best[1][0], best[1][1]],
        "communication_interval_us": [best[2][0], best[2][1]],
    }


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


def _worker_payload(stdout):
    prefix = "PHASE3E_WORKER_RESULT "
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    return {}


def _case_command(case, trace_dir):
    if case in HOST_CONTRACT_CASES:
        return [sys.executable, "-m", "pytest", "-q",
                HOST_CONTRACT_CASES[case]]
    worker = [str(pathlib.Path(__file__).resolve()),
              "--worker", case, "--trace-dir", str(trace_dir)]
    if case == "capture-current-stream":
        return [sys.executable, *worker]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={WORLD_SIZE}", *worker]


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


def _run_suite(suite, output, trace_dir):
    selected = EVENT_CASES if suite == "event" else CASE_NAMES
    results = []
    pathlib.Path(trace_dir).mkdir(parents=True, exist_ok=True)
    for case in selected:
        result = _run_bounded(_case_command(case, trace_dir))
        payload = _worker_payload(result["stdout"])
        row = {
            "case": case,
            "status": result["status"],
            "duration_seconds": result["duration_seconds"],
            "exit_code": result["exit_code"],
            "failure": result["failure"],
            "measurements": payload.get("measurements", {}),
        }
        if result["status"] == "failed":
            row["diagnostic"] = (
                result["stderr"] or result["stdout"])[-4000:]
            print(f"FAIL {case}: {row['failure']}", flush=True)
            if row["diagnostic"]:
                print(f"DIAGNOSTIC {case}:\n{row['diagnostic']}", flush=True)
        else:
            print(f"PASS {case} ({result['duration_seconds']:.3f}s)",
                  flush=True)
        results.append(row)
        report = _suite_report(suite, selected, results)
        _write_suite_report(output, report)
        if result["status"] == "failed":
            break
    print("PHASE3E_SUITE_RESULT " + json.dumps(report, sort_keys=True),
          flush=True)
    return 0 if report["summary"]["failed"] == 0 else 1


@contextmanager
def _forbid_global_sync(torch):
    original = torch.npu.synchronize

    def forbidden(*_args, **_kwargs):
        raise AssertionError("global NPU synchronization is forbidden")

    torch.npu.synchronize = forbidden
    try:
        yield
    finally:
        torch.npu.synchronize = original


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

    def _make_overlap_inputs(self):
        tokens = 4096
        hidden = OVERLAP_COMPUTE_SHAPE[0]
        cpu = self.torch.arange(
            tokens * hidden, dtype=self.torch.float32).reshape(
                tokens, hidden).remainder_(97).to(self.torch.bfloat16)
        x = cpu.to(self.device)
        peer_expert = 2 if self.rank == 0 else 0
        routes = self.torch.full(
            (tokens, 1), peer_expert, dtype=self.torch.int64,
            device=self.device)
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
        for _ in range(OVERLAP_COMPUTE_ITERATIONS):
            value = self.torch.matmul(value, right).mul_(0.001)
        return value

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

    def _profile_overlap(self, buffer, handle, x, left, right):
        profiler = self.torch_npu.profiler
        activity = profiler.ProfilerActivity.NPU
        trace_path = self.trace_dir / f"overlap-rank{self.rank}.json"
        with profiler.profile(activities=[activity]) as profile:
            self._overlap_iteration(
                buffer, handle, x, left, right, overlap=True)
        profile.export_chrome_trace(str(trace_path))
        compute_stream = self.torch.npu.current_stream()
        comm_stream = buffer.get_comm_stream()
        interval = _find_npu_overlap_interval(
            tuple(self.trace_dir.rglob("*.json")),
            compute_stream_id=compute_stream.stream_id,
            comm_stream_id=comm_stream.stream_id,
        )
        interval["trace"] = str(trace_path)
        interval["compute_stream_id"] = compute_stream.stream_id
        interval["communication_stream_id"] = comm_stream.stream_id
        return interval

    def _run_overlap(self):
        x, routes, left, right = self._make_overlap_inputs()
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
            self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=False)
            self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=True)

        serialized = []
        overlapped = []
        for _ in range(OVERLAP_REPETITIONS):
            serialized.append(self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=False))
            overlapped.append(self._overlap_iteration(
                buffer, handle, changed_x, left, right, overlap=True))
        serialized_summary = _summary(serialized)
        overlapped_summary = _summary(overlapped)
        improvement = 1.0 - (
            overlapped_summary["median_seconds"] /
            serialized_summary["median_seconds"])
        _check(improvement >= OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
               "overlap median improvement "
               f"{improvement:.6f} is below "
               f"{OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT:.6f}")
        interval = self._profile_overlap(
            buffer, handle, changed_x, left, right)
        return {
            "warmups": OVERLAP_WARMUPS,
            "repetitions": OVERLAP_REPETITIONS,
            "compute_shape": list(OVERLAP_COMPUTE_SHAPE),
            "compute_iterations": OVERLAP_COMPUTE_ITERATIONS,
            "serialized": serialized_summary,
            "overlapped": overlapped_summary,
            "median_improvement": improvement,
            "minimum_median_improvement":
                OVERLAP_MINIMUM_MEDIAN_IMPROVEMENT,
            "profiler_overlap": interval,
            "global_synchronizations": 0,
        }

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
            if case == "overlap-vs-serialized":
                return self._run_overlap()
        raise AssertionError(f"distributed case is not implemented: {case}")


def _run_distributed_worker(case, trace_dir):
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    worker = None
    initialized = False
    primary_error = None
    try:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(seconds=25))
        initialized = True
        group = dist.group.WORLD
        _check(dist.get_world_size(group) == WORLD_SIZE,
               f"async overlap requires {WORLD_SIZE} ranks")
        worker = AsyncOverlapWorker(
            torch, torch_npu, dist, deep_ep, group,
            torch.device("npu", local_rank), trace_dir)
        local_failure = None
        measurements = None
        try:
            measurements = worker.run(case)
        except BaseException as error:
            local_failure = f"{type(error).__name__}: {error}"
        reports = [None] * WORLD_SIZE
        dist.all_gather_object(
            reports,
            {"rank": dist.get_rank(group), "failure": local_failure},
            group=group,
        )
        aggregate = _aggregate_rank_failures(reports)
        if aggregate:
            raise RuntimeError(f"{case} failed: {aggregate}")
        if dist.get_rank(group) == 0:
            print("PHASE3E_WORKER_RESULT " + json.dumps({
                "case": case,
                "measurements": measurements,
            }, sort_keys=True), flush=True)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error = None
        if worker is not None:
            try:
                worker.destroy_buffers()
            except BaseException as error:
                cleanup_error = error
        if initialized:
            dist.destroy_process_group()
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


def _run_worker(case, trace_dir):
    if case == "capture-current-stream":
        measurements = _run_capture_worker()
        print("PHASE3E_WORKER_RESULT " + json.dumps({
            "case": case,
            "measurements": measurements,
        }, sort_keys=True), flush=True)
        return 0
    _run_distributed_worker(case, trace_dir)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--suite", choices=("event", "full"))
    parser.add_argument("--worker", choices=CASE_NAMES)
    parser.add_argument("--output", default="/tmp/phase3e-async-overlap.json")
    parser.add_argument("--trace-dir", default="/tmp/phase3e-async-traces")
    args = parser.parse_args()
    if args.contract:
        print(json.dumps(_contract(), sort_keys=True))
        return 0
    if args.worker:
        return _run_worker(args.worker, args.trace_dir)
    if args.suite:
        return _run_suite(args.suite, args.output, args.trace_dir)
    parser.error("one of --contract, --suite, or --worker is required")


if __name__ == "__main__":
    raise SystemExit(main())
