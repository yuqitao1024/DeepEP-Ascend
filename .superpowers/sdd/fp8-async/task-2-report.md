# Task 2 Report: Python And C++ Dispatch Admission

## Implementation

- Extended `_scenario_ascend_fp8_dispatch` with FP32 and packed int32 scale
  factor dispatches for cached and CPU-sync non-cached pure-scale-up stream
  modes. The scenario covers `async_with_compute_stream`,
  `allocate_on_comm_stream`, cached-handle reuse, and `previous_event`.
- Kept `previous_event_before_epilogue`, hybrid, scale-out, and non-cached
  `do_cpu_sync=False` calls rejected before the fake runtime boundary.
- Relaxed the Python stream preflight only by removing the blanket FP8 term;
  existing hybrid, scale-out, predecessor-allocation, and non-cached CPU-sync
  checks remain.
- Made C++ `split_dispatch` independent of scale-factor presence and admitted
  stream mode only for cached or split pure-scale-up dispatches.
- Did not change `recv_sf` allocation, exact output sizing, or return behavior.

## Files Changed

- `tests/ascend/test_python_api.py`
- `deep_ep/buffers/elastic.py`
- `csrc/backends/ascend/elastic_buffer.hpp`

## TDD Evidence

The brief named a stale pytest class. The actual singular class is
`PythonApiIsolationTest`, confirmed with `pytest --collect-only`.

RED command:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py::PythonApiIsolationTest::test_fp8_dispatch_collectively_validates_scale_factor_contract
```

RED output:

```text
FAILED ... RuntimeError: DeepEP Ascend backend: dispatch preflight failed on rank 0 (unsupported_dispatch_mode)
1 failed in 0.24s
```

Focused GREEN command:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py::PythonApiIsolationTest::test_fp8_dispatch_collectively_validates_scale_factor_contract
```

Focused GREEN output:

```text
1 passed in 0.18s
```

Required full Python API GREEN command:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py
```

Full-suite output:

```text
29 passed, 2 skipped, 19 subtests passed in 1.55s
```

Additional diff check:

```bash
git diff --check
```

Output: exit code 0 with no output.

## Self-Review

- Python and C++ preserve pure-scale-up and CPU-sync restrictions.
- `previous_event` still requires communication-stream allocation.
- The regression asserts observable runtime forwarding and pre-runtime
  rejection rather than source text or private branch details.
- Existing synchronous FP8 pairing and scale-factor validation stay covered by
  the scenario's existing valid and invalid inputs.
- No specs, plans, runtime.cpp, production matrix, benchmark inventory, or
  roadmap files were changed.

## Concerns

This task deliberately stops at admission and mode selection. Exact FP8 split
`recv_sf` allocation and return behavior remains Task 3 work and is not
qualified by this change.

## Review Fix Round 1

Added observable FP8-specific rejection characterization in
`_scenario_ascend_fp8_dispatch`:

- E5M2 payloads paired with valid scale factors fail preflight with
  `invalid_fp8_pairing` and do not reach `runtime.dispatch_calls`.
- Non-contiguous E4M3 payloads paired with valid scale factors fail preflight
  with `invalid_x_tensor` and do not reach `runtime.dispatch_calls`.
- Cached and CPU-sync non-cached FP8 dispatches during NPU stream capture
  report `unsupported_graph_capture` and do not reach the runtime.
- FP8 dispatch with `previous_event` but without
  `allocate_on_comm_stream=True` reports `unsupported_dispatch_mode` and does
  not reach the runtime.

The fake Torch fixture now supplies `torch.float8_e5m2` solely to characterize
the unsupported payload case. No production code changed. Per controller
ruling, pending-operation rejection and direct native C++ execution remain
deferred to Task 3's compiled public dispatch probe.

Focused characterization command:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py::PythonApiIsolationTest::test_fp8_dispatch_collectively_validates_scale_factor_contract
```

Focused result:

```text
1 passed in 0.20s
```

Full Python API command:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py
```

Full-suite result:

```text
29 passed, 2 skipped, 19 subtests passed in 1.71s
```

Review self-check:

- The added cases assert preflight errors and unchanged runtime call counts.
- Capture coverage includes both cached and non-cached FP8 paths.
- The only implementation-support change is the test fake's E5M2 dtype.
- No Task 3 lifetime/allocation behavior or native C++ execution was added.
