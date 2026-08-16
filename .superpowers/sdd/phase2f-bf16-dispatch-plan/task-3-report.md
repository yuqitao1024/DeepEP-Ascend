# Task 3 Report: Public Ascend C++ Dispatch Runtime

## Implementation

- Replaced the public Ascend `ElasticBuffer::dispatch` stub with synchronous
  BF16 dispatch over the existing `build_core_tiling` and
  `launch_internal_dispatch` path.  The public 16-field result tuple and
  `combine()` ABI/gate remain unchanged.
- Dispatch accepts only contiguous PrivateUse1/NPU rank-2 BF16 activations,
  rank-2 int64 top-k indices, and matching contiguous float32 weights.  It
  rejects scale factors, cumulative statistics, events, async allocation,
  hybrid mode, FP8, TMA mode, channel handles, invalid expansion flags, and
  non-`num_sms == 1` / non-`num_qps == 0` calls before allocation or launch.
- Uses the fixed-shard tiling already owned by the kernel, allocates maximum
  output capacity, launches on the current stream, synchronizes it, checks the
  transport diagnostic, copies counts to host, and returns exact narrow views.
- Added a process-monotonic `ElasticBuffer` dispatch family plus
  `DispatchSequence`.  The descriptor is serialized into the existing
  `token_metadata_at_forward` optional tensor; cached calls copy it to host and
  validate family, topology, shape, experts, alignment, capacity, and mode
  before launch.
- Added `CannRuntimeResources::copy_from_host`, backed by
  `aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)`, and extended the lifecycle
  mock to prove a host-to-device copy failure does not discard runtime
  ownership and a retry succeeds.

## Files

- `csrc/backends/ascend/elastic_buffer.hpp`
- `csrc/backends/ascend/runtime/cann_runtime.hpp`
- `csrc/backends/ascend/runtime/cann_runtime.cpp`
- `tests/ascend/production_lifecycle_probe.cpp`
- `tests/ascend/test_core_operator_contract.py`
- `tests/ascend/test_stub_source.py`

## TDD Evidence

RED command:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_production_api_does_not_bypass_transport_gate tests.ascend.test_stub_source
```

RED output: 5 tests ran; `test_production_api_does_not_bypass_transport_gate`
failed as intended because `launch_internal_dispatch` was absent from
`csrc/backends/ascend/elastic_buffer.hpp`.  The four stub-source checks passed.

Focused GREEN command:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_production_api_does_not_bypass_transport_gate tests.ascend.test_stub_source tests.ascend.test_simt_urma_transport.AscendSimtUrmaTransportTest.test_production_runtime_resource_lifecycle
```

Focused GREEN output: 6 tests ran, all passed.

Full host matrix command:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract tests.ascend.test_stub_source tests.ascend.test_simt_urma_transport
```

Full host matrix output: 36 tests ran, all passed; 3 CANN-dependent tests were
skipped because `ASCEND_HOME_PATH` is not configured.

Formatting check:

```bash
git diff --check
```

Output: exit 0 with no whitespace errors.

## Self-Review

- The public runtime does not inspect routing values or duplicate kernel
  routing/protocol logic; it builds the existing tiling and forwards existing
  `DispatchArguments`.
- The dispatch capability set removes direct peer pointers and remote atomic
  add, while `combine()` remains unsupported and unchanged.
- Expanded capacity reserves all top-k lanes, including duplicate expert IDs,
  before alignment padding.
- Cache descriptor validation precedes output allocation and kernel launch.

## Concerns

- CANN/Torch-NPU integration probes could not run in this environment because
  `ASCEND_HOME_PATH` is absent.  Host tests compile the production header with
  a Tensor API double and exercise runtime resource lifecycle behavior.

## Fix Round 1

### Root Cause and Fixes

- Cached reuse compared the transient `kCached` mode bit even though an
  uncached forward necessarily serializes a descriptor without it.  Descriptor
  serialization now removes only that transient bit; expanded and zero-padding
  bits remain part of the comparison.
- Cached metadata was sized from scalar handles without checking the device
  prefix tails used by the kernel.  Cached rank/expert/unaligned tensors are
  copied and checked before launch for tails, monotonicity, and local expert
  count agreement with the host list.
- Empty token inputs can legally have null data pointers.  Internal dispatch
  now permits null `x`, top-k, and destination-slot pointers only when tiling
  has zero tokens; all non-empty/output/storage requirements remain enforced.
- Every input and cached tensor is now checked against the full input
  `torch::Device`, not just the PrivateUse1 device type.  `do_cpu_sync=false`
  is rejected for uncached calls and retained only for validated cached calls.
- Dispatch diagnostics are validated immediately after `read_diagnostic`,
  before any host count copy can mask the device error.
- The runtime lifecycle probe now covers null destination/source, zero-byte,
  and post-destroy H2D calls and proves none invoke the backend callback.

### Fix Round TDD Evidence

RED command:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_public_dispatch_probe_executes tests.ascend.test_simt_urma_transport.AscendSimtUrmaTransportTest.test_production_runtime_resource_lifecycle
```

RED output: the new executable probe test failed at compile time because the
test-only public-dispatch probe hook and its injected test environment were
absent.  The lifecycle probe passed.

GREEN command:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_public_dispatch_probe_executes tests.ascend.test_stub_source tests.ascend.test_simt_urma_transport.AscendSimtUrmaTransportTest.test_production_runtime_resource_lifecycle
```

GREEN output: 6 tests passed.

Final host matrix:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract tests.ascend.test_stub_source tests.ascend.test_simt_urma_transport
git diff --check
```

Final output: 37 tests passed, with 3 expected skips because
`ASCEND_HOME_PATH` is unset; `git diff --check` exited 0.

### Round 2

`CannRuntimeApi` now captures the allocation-time NPU through `aclrtGetDevice` and dispatch requires the owner, input, cached tensors, and active device to agree before stream acquisition. The public probe checks active and cached device mismatch, all 16 tuple fields, and consecutive successful generations. Focused and full matrix evidence: 37 passing tests, 3 expected skips, clean diff.

### Probe Completion

The executable probe now uses the test-only `make_testing_buffer` seam to
inject real initialized `CannRuntimeResources` plus host-backed Tensor, kernel,
and transport doubles.  It calls the production public dispatch implementation
and checks exact tuple counts/narrowing, weights, zero-token null pointers,
uncached-to-cached descriptor reuse, cached tail mismatch rejection before the
launch counter increments, and full device mismatch rejection before launch.

Focused RED/GREEN sequence for the executable probe was recorded with:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_public_dispatch_probe_executes
```

The initial probe failed because the injection seam and Tensor/runtime doubles
were absent; the final run passed.  The full matrix after this probe expansion
reported 37 passing tests and 3 expected CANN skips.

### Final Probe Cases and Self-Review

- The launch double can fail after `DispatchAttempt` construction.  The probe
  confirms that this poisons the real per-buffer sequence and a follow-up call
  fails before another launch; the initial successful calls on the same buffer
  prove normal completion advances the generation.
- The transport double emits a fully populated device diagnostic while its
  subsequent count-copy path is configured to fail.  The probe asserts the
  public diagnostic message wins and that only the diagnostic read callback was
  invoked, proving count copies were not attempted.

Final focused command:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_public_dispatch_probe_executes
```

Output: 1 test passed.  Final full matrix and `git diff --check` are recorded
above: 37 tests passed with 3 expected CANN skips and no whitespace errors.

## Fix Round 3

### Attempt Boundary

- Root cause: `DispatchAttempt` began before output allocation, descriptor H2D,
  and current-stream acquisition. Its failure-on-destruction behavior therefore
  poisoned the buffer for setup failures which had not launched device work.
- The attempt now begins immediately before `launch_internal_dispatch`, after
  validation, cached D2H preflight, all output and descriptor allocations,
  descriptor H2D, argument construction, current-device validation, stream
  acquisition, and launch-storage construction. The generation is assigned to
  the already-built arguments only after the attempt begins.
- Launch, stream synchronization, diagnostic reads, and post-launch result D2H
  remain inside the attempt, so any failure after launch begins still poisons
  the sequence. The generation-independent descriptor and public ABI are
  unchanged.

### Exact Public Probe

- The public kernel double now writes literal BF16 payload bits, top-k indices,
  weights, counts, rank/expert prefixes, unaligned counts, source metadata, and
  destination slots. The Tensor double preserves device options and implements
  a real deep clone so cached-device and copied-index checks exercise the public
  behavior rather than placeholder values.
- The probe asserts the exact shape and literal contents of all 16 result tuple
  positions, including absent positions 1, 14, and 15, decoded descriptor
  fields, copied top-k indices, and a successful call without weights.
- Cached preflight coverage mutates a cloned rank-prefix tail, verifies no
  launch, then retries the corrected handles successfully at generation 2. A
  rank-prefix tensor on device 1 proves cached tensor device mismatch rejection.
- Separate buffers inject descriptor-copy and null-stream failures, then retry
  successfully with generation 1. The launch failure case asserts one launch at
  generation 1, disables the injected failure, and proves the next call is
  rejected without another launch.

### TDD Evidence

Focused RED against `476d974`:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_public_dispatch_probe_executes
```

The probe compiled and executed. It failed only the independent `descriptor
copy retry` and `stream acquisition retry` cases; exact output, corrected cached
retry, cached-device mismatch, launch poisoning, diagnostic ordering, and empty
dispatch cases executed without failure.

Focused GREEN after moving the attempt boundary:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract.AscendCoreOperatorContractTest.test_public_dispatch_probe_executes
```

Output: 1 test passed.

Full Task 3 host matrix:

```bash
python3 -m unittest -v tests.ascend.test_core_operator_contract tests.ascend.test_stub_source tests.ascend.test_simt_urma_transport
```

Output: 37 tests passed; 3 CANN-dependent tests skipped because
`ASCEND_HOME_PATH` is not configured.

### Self-Review

- A descriptor-copy, allocation, current-device, or stream-acquisition failure
  cannot create an attempt or consume a generation. Executable descriptor-copy
  and stream retries both launch at generation 1.
- The launch-poison test cannot pass because the injected launch failure remains
  active: it disables that injection before the second call, checks the precise
  sequence error, and asserts the launch count and generation remain unchanged.
- Literal expectations are independent of descriptor builders and kernel
  helpers. The descriptor is decoded and compared with the hand-derived 96-byte
  BF16 dispatch handle for family 7, topology `(0,2,0,2,0,1)`, shape
  `(1,8,2,1)`, alignment 1, capacity 4, and mode flags 0.

### Concerns

- CANN/Torch-NPU integration remains unavailable in this environment because
  `ASCEND_HOME_PATH` is unset; the host probe compiles the production dispatch
  implementation with runtime, transport, kernel, and Tensor doubles.
