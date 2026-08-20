# Task 3 Report: Exact SF Allocation, Return, And Lifetime

## Implementation

- Added one shared local `recv_sf` allocator in the Ascend binding. BF16 still
  produces `std::nullopt`; FP8 uses `torch::empty_strided` with the input SF
  dtype and the exact logical output-token count.
- Cached dispatch allocates SF from cached exact normal or expanded counts,
  binds its pointer and element strides before launch, retains it with the
  other communication-stream tensors, and returns it instead of dropping it.
- Split dispatch allocates SF only after count synchronization, binds the exact
  tensor into the epilogue arguments, retains it before async publication, and
  returns the same storage. The old post-publication column-major replacement
  remains only for the non-split path.
- Extended the compiled public probe to execute native FP8 split and cached
  gates for FP32 and packed int32 SF. It covers normal and expanded counts,
  row-major and column-major layouts, empty column-major strides `(1, 0)`,
  exact storage, pointer binding, cached non-null return, async event wait,
  retained input/output storage, and same-buffer busy rejection before event
  completion.
- Extended the host Tensor stub only to preserve explicit strides, calculate
  strided storage span, and expose a weak storage observer for lifetime checks.

## Files Changed

- `csrc/backends/ascend/elastic_buffer.hpp`
- `tests/ascend/production_dispatch_probe.cpp`
- `tests/ascend/test_stub_source.py`
- `.superpowers/sdd/fp8-async/task-3-report.md`

## TDD Evidence

The brief's source-text contract assertions were superseded by the controller
ruling. The actual pytest class is singular: `AscendCoreOperatorContractTest`.

RED command:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py::AscendCoreOperatorContractTest::test_public_dispatch_probe_executes
```

RED output against `d3dacbf` after adding the behavioral probe/stub support:

```text
F                                                                        [100%]
E           AssertionError: -6 != 0 : terminate called after throwing an instance of 'std::runtime_error'
E             what():  DeepEP Ascend backend: dispatch dispatch epilogue scale factors do not match element kind
1 failed in 3.53s
```

This is the expected native failure: the current split branch passed null SF
output arguments to the FP8 epilogue.

Focused GREEN command:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py::AscendCoreOperatorContractTest::test_public_dispatch_probe_executes
```

Focused GREEN output:

```text
.                                                                        [100%]
1 passed in 3.23s
```

Required broad GREEN command:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py tests/ascend/test_python_api.py
```

Broad-suite output:

```text
.............................................. [ 35%]
............................................................ [ 89%]
...........ss                                         [100%]
126 passed, 2 skipped, 48 subtests passed in 21.16s
```

Diff check:

```bash
git diff --check
```

Output: exit code 0 with no output.

## Observed Public And Native Behavior

- Public result SF has exact `[output_tokens, sf_packs]` logical shape and
  preserves either FP32 or packed int32 dtype.
- Row-major output strides are `[sf_packs, 1]`. Column-major strides are
  `[1, align_up(output_tokens, 4)]`; the empty split result is `[1, 0]`.
- The native cached launch and split epilogue receive the same pointer and
  element strides exposed by the returned SF tensor, and the probe verifies
  values written through those arguments at the returned tensor coordinates.
- Cached dispatch returns SF rather than `std::nullopt` for both normal and
  expanded output counts.
- Both native cached and non-cached FP8 async paths return an event. A second
  dispatch on the same buffer is rejected as busy before event completion.
- Weak storage observations show input X, input SF, and exact output SF remain
  owned after public results leave scope while the event is pending, then are
  released after `current_stream_wait()` completes the operation.
- Existing BF16 probe cases and all deferred stream/mode admission tests remain
  green.

## Self-Review

- Mutation check: null return, wrong dtype/shape/stride/storage, wrong bound
  pointer, missing SF writes, missing async event, missing retention, or missing
  same-buffer exclusion each fails at least one behavioral probe assertion.
- Cached exact allocation happens after cached count/shape validation and
  before launch; split exact allocation happens after synchronized count
  validation and before epilogue publication.
- The returned split SF is the retained tensor. It is not replaced after
  async publication.
- BF16 continues through the shared allocator as `std::nullopt`, leaving its
  arguments and public return unchanged.
- No Python preflight, `runtime.cpp`, specs, plans, roadmap, benchmark
  inventory, or NPU production matrix files were changed.

## Concerns

The required host sanitizer-backed native probe and Python suites passed. Per
task scope, no physical-NPU production matrix was run, so device allocator and
kernel behavior beyond the exercised public/native host contract remains for
the later NPU qualification task.

## Review Fix Round 1

### Changes

- Added event-ID-aware observations to distinguish the split operation's
  completion event from compute-dependency and predecessor-retirement events.
  Async split FP8 now asserts the epilogue launch precedes that completion
  event's record.
- Added an FP8 split epilogue-failure case which asserts the completion event
  is not recorded when the epilogue launch fails.
- Added a host-test-only Tensor callback at the existing
  `DEEP_EP_ASCEND_ASYNC_STATE_HOST_TEST_TENSOR` boundary. The stub exposes
  storage identity, and the public probe observes storage identity, stream
  identity, and a monotonic sequence without changing production APIs.
- Cached and split async FP8 cases assert input SF and exact output SF are
  recorded on the communication stream before the cached launch or the split
  count/epilogue launch, respectively.

Files changed in this round:

- `csrc/backends/ascend/elastic_buffer.hpp`
- `tests/ascend/production_dispatch_probe.cpp`
- `tests/ascend/test_stub_source.py`
- `.superpowers/sdd/fp8-async/task-3-report.md`

### TDD RED

With the test callback and assertions present but production host recording
still compiled out:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py::AscendCoreOperatorContractTest::test_public_dispatch_probe_executes
```

```text
F                                                                        [100%]
E           AssertionError: 1 != 0 : failed: FP8 uncached async scale-factor lifetime and busy rejection
E           failed: FP8 cached async scale-factor lifetime and busy rejection
1 failed in 27.63s
```

This RED isolates the missing host-test stream-record behavior in both native
async gates.

### Mutation Evidence

All mutations below were temporary and restored before GREEN.

Moving the split completion record before the epilogue and removing its normal
post-publication record:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py::AscendCoreOperatorContractTest::test_public_dispatch_probe_executes
```

```text
E           AssertionError: 1 != 0 : failed: FP8 uncached async scale-factor lifetime and busy rejection
E           failed: FP8 epilogue failure does not record completion event
1 failed in 14.12s
```

Removing `retain(sf)` so input SF is neither retained nor stream-recorded:

```text
E           AssertionError: 1 != 0 : failed: FP8 uncached async scale-factor lifetime and busy rejection
E           failed: FP8 cached async scale-factor lifetime and busy rejection
1 failed in 27.29s
```

Removing both cached/non-split and split `retain(recv_sf)` calls so exact output
SF is neither retained nor stream-recorded:

```text
E           AssertionError: 1 != 0 : failed: FP8 uncached async scale-factor lifetime and busy rejection
E           failed: FP8 cached async scale-factor lifetime and busy rejection
1 failed in 25.01s
```

### GREEN

Focused command after restoring all mutations:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py::AscendCoreOperatorContractTest::test_public_dispatch_probe_executes
```

```text
.                                                                        [100%]
1 passed in 3.45s
```

Full core operator contract:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py
```

```text
.............................................. [ 47%]
...................................................                   [100%]
97 passed, 29 subtests passed in 19.49s
```

Python API suite:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py
```

```text
.............................ss                       [100%]
29 passed, 2 skipped, 19 subtests passed in 1.59s
```

### Self-Review

- The callback exists only under the existing host-test macro. Non-test builds
  retain the original `resources_->record_tensor_stream` implementation.
- Event ordering uses native event IDs, so predecessor-retirement records
  cannot satisfy the completion-event assertion.
- Tensor ordering uses a separate monotonic sequence, preserving existing
  exact string-order assertions in the probe.
- Input and output SF assertions compare storage identity rather than nullable
  data pointers, so empty storage remains distinguishable if later covered.
- The epilogue-failure test checks the specific newly created completion event,
  not the total record count, allowing legitimate predecessor records.
- Cached/row-major empty variants remain outside this fix round as ledgered.
- No Task 4 behavior, Python preflight, `runtime.cpp`, specs, plans, roadmap,
  benchmark inventory, or NPU production matrix files changed.
