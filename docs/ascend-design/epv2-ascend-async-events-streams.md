# EPv2 Ascend Async Events, Streams, And Overlap

## Status And Scope

This document defines Phase 3E for the Ascend 950 EPv2 `ElasticBuffer` path.
It builds on the rank-parameterized, one-operation-per-buffer contract from
Phase 3B and the synchronous BF16 dispatch/combine implementation. DeepEP V1,
GPU execution, FP8, and performance parity with CUDA are outside this phase.

Phase 3E is split into independently accepted vertical slices. The accepted
3E.1 slice provides native event and communication-stream ownership plus real
communication/computation overlap for cached BF16 pure-scale-up dispatch and
combine. Later slices extend the same ownership model to dynamic-shape and
hybrid paths; an interface or compile probe alone is not runtime support.

The 3E.1 production gate is
`tests/ascend/production/run_async_overlap.py`. It reports cached BF16
sub-operation coverage independently from the 144-row parity benchmark; no
full functional benchmark row is credited by partial sub-operation coverage.

### Acceptance status (2026-08-20)

Phase 3E.1 is accepted. The complete production-backed evidence is recorded
after the qualification history below. Phase 3E overall remains incomplete;
3E.2-3E.4 and FP8 Phase 3F retain their separate acceptance boundaries.

#### Qualification history

Before final qualification, Phase 3E.1 was not accepted. Exact-archive
build/import task
`task_20260819_203518_127255030913` passed, and one-NPU event/lifecycle task
`task_20260819_203858_128526415799` passed all five selected cases with two
repeated waits and zero global synchronizations. The required two-NPU matrix
task `task_20260819_204026_128929210009` did not pass: only
`capture-current-stream` passed before the following cases exited 1:

- `cached-dispatch-sync-allocate-false`;
- `cached-dispatch-sync-allocate-true`;
- `cached-dispatch-async-allocate-false`;
- `cached-dispatch-async-allocate-true`;
- `previous-event-allocate-true`;
- `combine-sync-allocate-false`;
- `combine-sync-allocate-true`;
- `combine-async-allocate-false`;
- `combine-async-allocate-true`;
- `empty-route`; and
- `asymmetric-route`.

The 300-second hard limit interrupted the task before these rows ran:

- `100-generations`;
- `two-independent-buffers`;
- `record-failure`;
- `event-timeout`;
- `diagnostic-failure`;
- `completion-mismatch`;
- `drop-event`;
- `destroy-pending-retry`; and
- `overlap-vs-serialized`.

The failed runner captured child stderr but did not persist it before suite
completion; the corresponding targeted TorchElastic directories contain no
rank log or error JSON. That job therefore did not establish the common
distributed failure. No serialized/overlapped median or p95 measurement and no
NPU profiler overlap interval were produced. The runner was then changed to
write an atomic checkpoint after every case, stream failed-child diagnostics,
and stop at the first failure.

A follow-up fail-fast matrix, task
`task_20260819_213704_141211314907`, reran the corrected synchronization scope
and persistent two-rank launcher. It completed in 1m42s and passed capture,
all cached dispatch/combine allocation combinations, previous-event ordering,
empty and asymmetric routes, 100 generations, and two independent buffers.
It then stopped at `diagnostic-failure`: the injected
`kCompletionTimeout` reached async finalization, but the returned stable
`TransportStatus` retained only `device diagnostic reported failure` and
dropped the error name, command, rank/peer, backend, and generation fields.
The five standalone lifecycle rows and `overlap-vs-serialized` were therefore
not run.

A deterministic host RED reproduced that loss in `AsyncBufferState`.
Finalization now uses the same qualified diagnostic-to-status mapping as the
established barrier and dispatch paths, preserving the complete device
diagnostic while retaining exactly-once failure storage. The focused host
probe and all sanitizer-backed production probes are green.

Exact-source rebuild task `task_20260819_221458_163301111405` then passed on
devices `6,7` in 2m54s for commit `9854a4e24fd918f30f0360ee45870dcb8f3a7bc9`
and tree `016a76c6561d503d8293261986c787497c226674`. The first build submission,
`task_20260819_221134_162617630198`, had exited before compilation because its
staging script omitted the qualified CMake directory from `PATH`; the retry
changed only that environment entry and used a fresh extraction directory.

The subsequent fail-fast matrix task `task_20260819_222014_164400825732`
completed in 1m0s with 15 passed, one failed, and five not run. It validated
the complete qualified `completion_timeout` diagnostic, then failed at
`overlap-vs-serialized` on both ranks because the production event-specific
5-second completion deadline expired (`synchronize_event`, backend `-1`). The
same terminal error recurred during buffer cleanup. The profiler stage was not
reached, its trace directory remained empty, and the later `record-failure`,
`event-timeout`, `completion-mismatch`, `drop-event`, and
`destroy-pending-retry` cases did not run. No serialized/overlapped median or
p95 values, stream IDs, or positive profiler interval were produced. Phase
3E.1 therefore remained not accepted after that task.

Subsequent bounded diagnostics refined that result without changing the
production five-second deadline or the fixed 5% overlap threshold. Task
`task_20260820_015907_282885225738` showed that the 4096-token communication
control itself completes only near 14.4 seconds, so the original event timeout
was workload-bound rather than evidence of a stalled completion event. A
256-token full-matrix retry reached the overlap measurement but its median
improvements, `0.014373` and `0.012169`, were below 5%.

Sweep task `task_20260820_022734_29035825747` persisted profiler traces for
256 and 512 before its 180-second controller bound expired during 1024. The
runner initially failed to parse them because this Torch-NPU version exports a
top-level JSON array rather than a `{traceEvents: [...]}` object. Offline
structured parsing of the exact hashed traces proves positive overlap on both
ranks: 256 uses compute/communication streams `61/26` with `269777.0us` and
`268351.5us`; 512 uses streams `61/25` with `587813.0us` and `594118.25us`.
No 1024 trace was created. Artifact mtimes place that timeout before profiling,
inside the token-scaled 1024 warmup/timing work, not trace export, parsing, or
teardown.

The acceptance runner now accepts both trace containers, recognizes
Torch-NPU's `Physic Stream Id`, normalizes byte-valued timeout output before
marker recovery, and retains completed timing summaries plus stream/profiler
failure evidence. At that point, Phase 3E.1 remained not accepted: 256 did not
meet the fixed wall-time threshold, 512 timing summaries were lost by the old
runner, and 1024 had not completed a bounded diagnostic. The next proposed
target action was a diagnostic-only 1024 point with zero warmups, one
repetition, phase checkpoints, a 150-second controller bound, and the required
300-second TaskQueue bound. One sample could guide workload selection but
could not satisfy the warmed-median acceptance requirement.

That diagnostic, `task_20260820_025915_29863145111`, exited 0 after the runner
completed in 116.488 seconds. Both ranks completed every event wait and phase.
Communication took approximately 1.388 seconds, isolated compute 11-12ms,
serialized execution approximately 1.397 seconds, and overlapped execution
approximately 1.394-1.395 seconds. The theoretical maximum improvement was
only `0.008096`/`0.008735`; observed improvement was
`0.002188`/`0.001010`. This 1024 workload therefore cannot meet the fixed 5%
gate even with ideal scheduling.

Torch runtime stream objects reported logical compute/communication IDs
`0/128`, while the exported profiler trace labels the physical lanes `61/26`.
Offline structured parsing of the exact hashed traces shows positive
MatMulV3/DeepEP-dispatch overlap of `1381206.75us` and `1381206.0us`. Thus the
implementation produces real hardware overlap, but that diagnostic's
resource/workload balance does not produce a 5% wall-time gain. The result
remains diagnostic-only and does not accept Phase 3E.1.

The final acceptance workload keeps 256 communication tokens, the
`4096x4096` BF16 matmul shape, three warmups, seven measured repetitions, the
fixed 5% median-improvement threshold, and the 30-second external per-case
watchdog. It changes only the runner's compute workload from 8 to 256 matmul
iterations; production C++ and the five-second event deadline are unchanged.
The 256-token traces place communication between approximately 0.27 and 0.59
seconds. Scaling the measured 1024-token diagnostic's 8-iteration compute
time of 0.011-0.012 seconds by 32 estimates 256 iterations at approximately
0.36-0.39 seconds, putting compute and communication in the same range.

At the high ends of those evidence ranges, the complete overlap case's three
warmup serialized/overlapped pairs, seven measured pairs, and one profiler
iteration consume approximately 14.3 seconds. Buffer creation and seeding
therefore retain substantial margin under the unchanged 30-second watchdog;
warmups and repetitions are not reduced. The profiler parser now resolves
physical streams only from the unique `MatMulV3` compute event family and
unique DeepEP `dispatch_kernel` communication event family, rejects aliased
or ambiguous physical lanes, and reports runtime logical IDs separately from
trace physical IDs. At that point, a final 21-case matrix remained required
for acceptance.

Final full-matrix task `task_20260820_031911_304861030703` used those exact
parameters on devices `6,7` and reached terminal `completed (exit=1)` after
1m15s. Its fail-fast summary is 21 selected, 16 executed, 15 passed, one
failed, and five not run. The overlap case completed in 29.315 seconds and
retained seven raw samples per mode and rank. Rank 0 measured serialized
median/p95 `0.360265130`/`0.362707628` seconds and overlapped
`0.360237910`/`0.362412710`, an improvement of `0.000075555`. Rank 1 measured
`0.361493950`/`0.362399942` versus `0.358961650`/`0.360509258`, an improvement
of `0.007005097`. Both remained below the unchanged 5% gate.

The same report proves positive `MatMulV3`/DeepEP-dispatch profiler overlap:
rank 0 `268335.5us` and rank 1 `270927.0us`. Runtime logical
compute/communication IDs were `0/143`; trace physical IDs were `61/11`.
Report SHA-256 is
`db667b7e4a2afd490b3f5b9ae59a176432cae69cbc22b8b0bc8ce1a907236189`.
Because the overlap row failed, fail-fast did not run `record-failure`,
`event-timeout`, `completion-mismatch`, `drop-event`, or
`destroy-pending-retry`. Phase 3E.1 remained unaccepted while a bounded
component-timing diagnostic determined why positive physical overlap did not
improve the wall-time control.

Component diagnostic task `task_20260820_033904_310487927692` used a fresh
buffer per rank with 256 communication tokens, 256 compute iterations, zero
warmups, one repetition, and a 120-second controller bound. It ran on devices
`6,7` with the required 300-second outer bound and reached terminal
`completed (exit=0)` after 59 seconds. The runner completed in 50.007 seconds
and classified both ranks as `resource-contention`.

Rank 0 measured communication-only/compute-only/serialized/overlapped walls
of `0.279502830`/`0.095801200`/`0.359756550`/`0.357143020` seconds. Its
component sum exceeded serialized by `0.015547480` seconds, within the measured
`0.018765202` tolerance; overlap saved only `0.002613530` seconds, within its
`0.017987828` tolerance. Rank 1 measured
`0.279471440`/`0.095291040`/`0.359411640`/`0.357857410`; its corresponding
gaps were `0.015350840` versus tolerance `0.018738124` and `0.001554230`
versus `0.017970582`.

The synchronous dispatch returns took `0.278580530` and `0.278420550`
seconds, matching communication-only wall and ruling out an unserialized
control. Asynchronous returns took only `0.007083720` and `0.007498330`, but
event waits stretched to `0.342516940` and `0.342942600` seconds while compute
completion after those waits was already below `0.000071` seconds. The exact
traces still prove `269032.25us` and `268594.75us` of positive
`MatMulV3`/DeepEP-dispatch overlap on physical streams `61/26` (logical
`0/128`). The report SHA-256 is
`13a5d76521841f9bfbf55aaa74e9b85828567417c078b0ce530cd031255ebb3e`.
This evidence supports resource contention, not a synchronous baseline defect;
production remained unchanged and rerunning the same full matrix was not
justified. Phase 3E.1 remained unaccepted under the fixed 5% wall-time gate.

Static investigation does not support changing the Torch-NPU pool selector as
a contention fix. The qualified `torch_npu 2.10.0.post2` wheel records revision
`8751b36d5d6959e499e6bf6530c1928060ced030`. In that source,
`getStreamFromPool(true)` selects the logical high-priority pool with priority
`-1`, while `false` selects logical priority `0`; both pools are created by
`AclrtCreateStreamWithConfig` with physical priority `0` and flags
`ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC`. Changing the boolean would
therefore change logical pool identity but not CANN stream priority in this
qualified revision.

The dispatch launch is already constrained to one outer AIV/vector block:
`CoreLaunchShape` is `num_blocks=1`, `num_threads=512`, the public dispatch
contract requires `num_sms=1`, `dispatch_kernel` is declared
`__global__ __vector__`, and its outer launch uses `tiling.launch.num_blocks`.
The 512-thread value is passed to the nested `asc_vf_call` stages. Inspection
of the exact qualified object also found the embedded device ELF
`.AIV_Kernel_Type` dispatch symbol and vector-SIMT entries. This proves one
outer AIV/vector block, but the available public CANN headers do not establish
how that block maps to a physical vector core; that mapping remains a vendor
documentation gap rather than evidence for a launch-shape change.

A final component-only diagnostic isolated the remaining elementwise-work
hypothesis. Commit `8e812bd` makes the component path submit exactly 256
independent `torch.matmul(left, right)` calls and no `mul` or `mul_`; the
formal `overlap-vs-serialized` case retains the mixed MatMulV3-plus-`mul_`
workload and its fixed acceptance gate. Task
`task_20260820_040533_315212229000` ran only this `pure-matmul` diagnostic on
devices `6,7` with the 120-second controller and 300-second outer bounds. It
reached terminal `completed (exit=0)` after 52 seconds, with a runner duration
of `50.108964270` seconds, and classified both ranks as
`resource-contention`.

Rank 0 measured communication-only/compute-only/component-sum/serialized/
overlapped walls of `0.278048730`/`0.087509200`/`0.365557930`/
`0.353972660`/`0.353502300` seconds. Rank 1 measured
`0.278018550`/`0.086884680`/`0.364903230`/`0.354368630`/
`0.353292460`. The overlap gains were only `0.000470360` and `0.001076170`
seconds, or `0.1329%` and `0.3037%`, far below 5%. Synchronous dispatch
returns remained `0.278711710`/`0.279201640` seconds and overlapped event waits
remained stretched at `0.341599990`/`0.340491320` seconds. The profiler still
proved `268224.5us`/`269994.5us` of positive overlap on physical streams
`61/26` (logical `0/128`). Report SHA-256 is
`4eef5df836b6dcfc09def67cd5bf58b6fd517e70faa93ca1d2b6e5accc247369`.

Removing `mul_` reduced compute-only time from approximately 95ms to 87ms but
did not remove the communication stretch or produce material wall-time gain.
The elementwise-chain contention hypothesis is therefore rejected. No
production change, formal-workload substitution, full-matrix retry, or
additional NPU task followed from this result; the mixed formal acceptance case
remained unchanged and Phase 3E.1 remained unaccepted.

A bounded AIC-only primary-kernel capability diagnostic then replaced only
the component compute path with BF16 NCDHW `torch.nn.functional.conv3d`.
Runner commit `8aaa624` used input shape `(1,64,8,32,32)`, weight shape
`(64,64,3,3,3)`, and NPU-event-specific calibration; production and the
formal acceptance workload remained unchanged. The only capability task,
`task_20260820_045758_323960612146`, completed on devices `6,7` with exit 0
and a runner duration of `44.573933470` seconds.

Rank 0 measured communication-only/compute-only/component-sum/serialized/
overlapped walls of `0.279388210`/`0.056249980`/`0.335638190`/
`0.332164400`/`0.275529020` seconds, a `17.0504%` gain. Rank 1 measured
`0.282109590`/`0.056503370`/`0.338612960`/`0.332342290`/
`0.275268150`, a `17.1733%` gain. Event waits were only
`0.220873860`/`0.221500330` seconds. Both profiler traces identify primary
`Conv3DV2` as exactly `KERNEL_AICORE`, DeepEP dispatch as exactly
`KERNEL_AIVEC`, distinct physical streams `61/26`, and positive intervals of
`28.5/28.75us`.

Calibration selected `2039/2048` iterations at its upper clamp, but
compute-only wall remained about 56ms rather than the intended 0.20-0.30s.
Each trace also contains tiny `Greater` and `ZerosLike` AIV families totaling
`3.401us` and `2.738us`; neither contains TransData. The evidence therefore
proves that an AIC-only primary kernel can overlap AIV dispatch and produce a
material wall reduction, but it does not prove a wholly pure-AIC compute path.
The report correctly records `compute_path_aic_only=false` and remains
acceptance-ineligible as a single-sample capability diagnostic. Report
SHA-256 is
`88a4fa784c660ffa078b1a443e6b0b9573230ceed458a3932eb2c7944c31a813`.
Phase 3E.1 remained unaccepted pending the complete formal matrix.

The capability result authorizes one formal-workload substitution. The final
`overlap-vs-serialized` row keeps 256 communication tokens, hidden size 4096,
`num_sms=1`, the production five-second event deadline, the 5% median gate,
and the 30-second case watchdog. It replaces only formal runner compute with
fixed BF16 NCDHW Conv3D: input `(1,64,24,96,96)`, weight
`(64,64,3,3,3)`, stride `(1,1,1)`, padding `(0,0,0)`, dilation `(1,1,1)`,
groups 1, and 256 iterations. There is no formal dynamic calibration, MatMul,
`mul`, or `mul_`; the component capability diagnostic remains unchanged.

Three warmups and seven repetitions are retained. Scaling the slower observed
capability compute wall (`0.056503370` seconds for 2039 iterations) by output
volume and iteration count estimates `0.168339307` seconds per fixed compute
call. Conservatively charging each of the 21 warmup, measured, and profiler
calls for both that compute and the slower observed `0.282109590`-second
communication wall estimates `9.459426836` seconds before setup and trace
export, below the 30-second watchdog. Repetitions therefore remain at seven.

Formal profiling requires `Conv3DV2` exactly `KERNEL_AICORE`, DeepEP
`dispatch_kernel` exactly `KERNEL_AIVEC`, distinct physical lanes, and a
positive interval; logical and physical IDs are reported separately. Every
other AIV family, including the known tiny `Greater` and `ZerosLike` work, is
inventoried with total duration. That total must be below 1% of the primary
Conv3D span. The report always records `compute_path_aic_only=false` and never
promotes primary-kernel task type into a whole-path purity claim. The declared
gate required the complete matrix to pass 21/21, both ranks to clear 5%, and
every profiler check to pass.

#### Production-backed final acceptance

Independent review rejected the earlier acceptance decision based on task
`task_20260820_052826_33515907092`: five negative and lifecycle rows were
routed through host contract tests rather than the production workers. That
result remains qualification history, but its fake-runtime rows cannot support
an NPU acceptance claim. The corrected runner removed that routing. Its
one-NPU event suite contains exactly `capture-current-stream`,
`record-failure`, and `event-timeout`; `completion-mismatch`, `drop-event`, and
`destroy-pending-retry` execute inside the persistent two-rank production
worker.

Exact-source build task `task_20260820_062500_37256458511` completed in
`2m55s`, compiled with `DEEP_EP_ASCEND_TESTING=1` against the pinned HCOMM
package, and imported the Ascend extension. The local and remote source
archives both had SHA-256
`573d6970812fe4f8cf6b60b06d2d4e0bb9547ddfc35cd8177d69d677922955de`;
the build log printed the actual hash and matched the expected value. The
accepted production commit/tree are
`28f3deaed1b950e782ba57a0d0256cd2f35687f9`/
`47ed60717b538b88436fa2602628f28b9a76f234`; the extension SHA-256 is
`dd9e91a69b8526ba9b828387fdc1b655a46f4a4ae10ce4af3d36fec7fd41bb67`.
Runner commit `b185d95c28864c22faf99dc67f5492028e6383fb` has file SHA-256
`e3154dd502d1f9986d16d7b131520d6ab3d292d191e4cb949ecb7a686491300f`.

The corrected one-NPU event task `task_20260820_063113_375337613415`
passed 3/3 on device 6 in `52s`. The first corrected full task,
`task_20260820_063252_376006031537`, passed 16 rows before `drop-event`
failed on a stale generation-bound dispatch handle and the remaining four rows
did not run. A deterministic host RED reproduced that sequence: dropping the
first event finalized its generation, so reusing the original seed handle for
the second dispatch was invalid even though the buffer itself was reusable.
Runner commit `b185d95c28864c22faf99dc67f5492028e6383fb` reseeds the same
buffer before the second dispatch; its full host GREEN was 199 passed, 7
skipped, and 48 subtests.

Final two-NPU task
`task_20260820_064309_380109210223` ran on devices `6,7` with
`--max-time 300`, reached authoritative `completed (exit=0)` in `1m49s`, and
passed all 21 selected and executed cases with zero failures and zero not-run
rows. Both ranks passed `drop-event` with the same buffer reused after event
drop and garbage collection. Both retained runtime ownership after the
injected first destroy failure and released it after retry. Record failure,
event timeout, qualified diagnostic failure, and post-bookkeeping completion
mismatch also passed through production code.

| Rank | Serialized median/p95 | Overlapped median/p95 | Median gain |
|---:|---:|---:|---:|
| 0 | `0.304671370/0.305792395s` | `0.277607860/0.278358744s` | `8.8829%` |
| 1 | `0.304223720/0.306510999s` | `0.277838260/0.278617760s` | `8.6730%` |

Both ranks retained three warmups and seven measured samples, reported zero
global synchronizations, and exceeded the unchanged 5% gate. The profiler
identified `Conv3DV2=KERNEL_AICORE` and `dispatch_kernel=KERNEL_AIVEC` with
logical compute/communication stream IDs `0/146`. Rank 0 used distinct
physical lanes `56/6` with `110.5us` overlap; rank 1 used `61/8` with
`112.0us` overlap. Auxiliary AIV ratios were `0.012324%` and `0.010512%`,
below the 1% gate, and neither trace contains TransData.
`compute_path_aic_only=false` remains explicit, so no whole-path pure-AIC
claim is made.

The final report SHA-256 is
`0e6d7674f03967285ead97a75d7c99cb7537a8157ca9794c2a2bb644221c82ee`.
Rank 0 and rank 1 trace SHA-256 values are
`0c079bbdc9bc5c722394079fb5bd14dba9e23ec0b369ce5479cc486e04a8b5f9`
and `e9dc5c3c487a63ed3d0f8f70c75884171aa0a15e435f14bf80fcef5ef4882001`.
Every fixed Phase 3E.1 gate passed, so Phase 3E.1 is accepted. This does not
promote any of the 120 deferred full functional benchmark rows or change the
3E.2-3E.4 and FP8 Phase 3F boundaries.

Post-rebase qualification keeps every functional and performance gate above,
but raises the external per-case process watchdog from 30 to 45 seconds. A
clean exact-source run completed both CANN profiler exports in about 19.7
seconds yet crossed the old watchdog at `30.0009s` before result publication;
the earlier accepted overlap row had already consumed `26.7121s`. The extra
controller margin does not change the five-second native event deadline, the
5% median-improvement threshold, the three warmups, seven measured samples, or
the 300-second TaskQueue bound.

## Decisions

### Native resource model

- Each `ElasticBuffer` owns one Torch-NPU pool stream for communication. The
  stream is retained as a `c10_npu::NPUStream` value and is never destroyed by
  DeepEP.
- DeepEP owns raw CANN events through a small RAII adapter. Raw events are used
  instead of `c10_npu::NPUEvent` because Phase 3E needs explicit timeout,
  backend-code, and retryable destruction behavior.
- The CANN/Torch-NPU calls are behind an injectable runtime seam. Host tests
  exercise every ownership transition without importing Torch NPU.
- The selected pool-stream, event, timeout, Python stream-wrapper, and allocator
  APIs must pass a targeted NPU8P capability probe before production code is
  enabled. A declared header without a link and runtime probe is insufficient.

### Initial wait semantics

Ascend `EventHandle.current_stream_wait()` performs a bounded, event-specific
host completion wait and terminal DeepEP validation. It does not synchronize
the whole device and does not call `torch.npu.synchronize()`.

This is intentionally stronger than CUDA's enqueue-only wait. CANN requires an
event to outlive all stream-wait tasks, while `EventOverlap` permits
`release_handle=True`. A host-completed event can be released safely without a
background reaper. A later enqueue-only slice requires a reaper or callback
protocol that owns the event until every dependent stream task retires and has
a defined path for delayed DeepEP errors.

### Operation and failure model

One buffer continues to admit exactly one barrier, dispatch, or combine at a
time. An async operation remains active until its returned event finalizes it.
A second same-buffer entry is a deterministic busy error and follows the
existing deferred-poison rule. Independent buffers own independent streams,
coordinators, windows, workspaces, diagnostics, and events and may run
concurrently.

The operation lease moves into a shared pending operation after launch. It is
not completed when the C++ method returns. Finalization is exactly once and
includes event completion, diagnostic validation, generation/completion
validation, lease retirement, stable error storage, and release of retained
tensors and predecessor events.

Pre-launch failures cancel the reservation without consuming a generation.
Every post-activation launch, record, timeout, diagnostic, completion, or
finalizer failure poisons the buffer. Repeated waits return the same terminal
result and never re-read a diagnostic slot that a later operation could have
overwritten.

Destroy drains the one pending operation with the same finite timeout. If
completion and event lifetime cannot be proven, destroy returns a retryable
error and retains the stream, event, workspace, window, transport, and device
allocations. Phase 3E does not use `aclrtStreamAbort`, force stream destruction,
or fabricated collective cancellation.

### Stream and tensor lifetime

The communication stream waits on the caller's `previous_event`, or on an
event captured from the current compute stream when no predecessor is supplied.
The predecessor remains strongly retained through communication completion.
`previous_event` requires `allocate_on_comm_stream=True`, matching the shared
EPv2 contract.

When `allocate_on_comm_stream=True`, a Torch-NPU stream guard selects the
buffer's communication stream before the first output or handle allocation and
restores the compute stream before returning. Every input, output, metadata,
handle tensor, and predecessor needed by a pending operation is strongly
retained. The selected generic Torch-NPU allocator record-stream API is also
used after the capability probe, but strong retention remains the correctness
fallback for custom allocators.

The synchronous and asynchronous paths use the same communication stream,
completion event, and terminal validator:

- synchronous calls finalize before returning and return no underlying event;
- asynchronous calls return a real event immediately after the operation and
  completion event are enqueued;
- no accepted asynchronous path contains a hidden device-global synchronize.

## 3E.1 Supported Matrix

The first production slice supports:

- `ElasticBuffer.capture()` recording a real event on the current NPU stream;
- `ElasticBuffer.get_comm_stream()` returning the owning Torch-NPU stream;
- synchronous `barrier(use_comm_stream=True)` with compute-to-communication
  ordering; barrier remains return-void and host-synchronous;
- synchronous and asynchronous cached BF16 pure-scale-up dispatch;
- synchronous and asynchronous BF16 pure-scale-up combine;
- `previous_event` absent or present, with the allocation precondition;
- `allocate_on_comm_stream` false or true;
- exact event-specific waits, repeated waits, event drop, and retryable destroy;
- empty and asymmetric routes, cached-handle reuse, repeated generations, and
  independent buffers.

The first slice explicitly rejects before launch:

- non-cached async dispatch, because exact output extents and handle
  attestation currently require post-kernel host work;
- `previous_event_before_epilogue`, because dispatch and combine each expose one
  whole-kernel launch and no safe host-visible epilogue boundary;
- logical-hybrid and physical scale-out async, because Phase 3D route-table and
  binding attestation are post-launch host work;
- more than one in-flight operation per buffer, because the command queue,
  workspace, symmetric controls, diagnostic, and completion records are
  single-slot;
- async barrier, graph capture, mapped CPU communication, FP8, pipeline,
  Engram, AGRS, and device-transport async request flags.

`TransportCapability::kAsyncCompletion` and `CoreMode::kAsyncEvent` remain
disabled in 3E.1. A host stream returning before a whole AICore kernel completes
does not implement the device request lifecycle represented by those flags.

## Components

### Stream and event runtime seam

`csrc/backends/ascend/runtime/stream_event.hpp/.cpp` owns the injectable
stream/event API and native RAII event state. It provides:

- current and pool stream acquisition with device identity;
- event create, record, query-poll bounded wait, stream wait, and destroy;
- generic allocator stream recording;
- `TransportStatus` errors with the exact operation and backend code; and
- retryable state transitions that never clear ownership after a failed
  destroy.

`CannRuntimeResources` owns the pool-stream handle and exposes checked access.
The resource teardown order is pending operation, event, transport, workspace,
and window. Torch-NPU retains ownership of the pool stream.

### Shared async state

`AsyncBufferState` is shared by the Python-visible buffer and every event that
can outlive it. It owns runtime resources, the operation coordinator, the
pending-operation slot, timeout configuration, and the first stable terminal
failure. No event stores a raw pointer to `ElasticBuffer`.

`PendingOperation` owns its moved operation lease, generation, completion
event, retained tensors, predecessor events, and operation-specific terminal
validator. A mutex and terminal state implement:

```text
Launched -> Finalizing -> Succeeded
                      \-> Failed
```

All callers observe the same terminal state. Only the first finalizer performs
native waits and diagnostic reads.

`EventHandle` is copyable through shared event state. A captured event contains
only native event/device state; an operation event also references its pending
operation. `current_stream_wait()` validates current device identity and invokes
bounded finalization.

## Data Flow

```text
compute stream
    |
    | record dependency event or use previous_event
    v
buffer comm stream -> wait dependency -> allocate/record tensors -> EP kernel
                                                                  |
                                                                  v
                                                         completion event
                                                          /             \
                                                sync finalize       async return
                                                validate/retire       EventOverlap
                                                                        |
                                                             independent compute
                                                                        |
                                                             current_stream_wait
                                                             validate/retire
```

All collective preflight that can fail locally remains before lease activation.
All post-launch checks remain generation-qualified and finite.

## Capability Gate

Before implementation is enabled, a serialized NPU8P probe must prove against
the pinned CANN, Torch-NPU, and HCOMM environment:

1. `c10_npu::getStreamFromPool`, current-stream lookup, and an NPU stream guard;
2. reconstruction of the retained stream as a usable Python `torch.npu.Stream`;
3. event create, record, query-poll bounded wait, stream wait, and destroy;
4. predecessor-event lifetime across a stream wait;
5. the selected generic allocator record-stream call; and
6. ordering and completion without a device-global synchronize.

NPU8P CANN 9.2 exports `aclrtSynchronizeEventWithTimeout` from
`libascendcl.so`, but the pinned public CANN and Torch-NPU headers do not
declare it or expose recoverable signature metadata. Phase 3E.1 therefore uses
the supported fallback: `aclrtQueryEventStatus` polling with a
`std::chrono::steady_clock` five-second deadline. The exact non-ready result is
`aclrtEventRecordedStatus` value `ACL_EVENT_RECORDED_STATUS_NOT_READY`; only
`ACL_EVENT_RECORDED_STATUS_COMPLETE` completes the wait. If Python stream
reconstruction is not supported, the extension returns a Torch-NPU stream
object directly; it must not expose a raw integer pointer as the public stream.

## Verification

Host and contract tests cover ownership rollback, exactly-once finalization,
copied event handles, same-device checks, predecessor retention, allocator
recording, pending destroy/retry, stored poisoning, timeout, and proof that no
global synchronize callback is used.

Production acceptance is serialized through TaskQueue and uses the pinned
`hcomm-deepep-current` package:

- one-NPU event/stream ordering, repeated wait/drop, timeout, allocator
  lifetime, and pending-destroy tests;
- two-NPU synchronous BF16 regression plus async cached dispatch/combine against
  an independent reference for empty and asymmetric routes and at least 100
  generations;
- bounded record, launch, diagnostic, completion, and timeout failures with
  rank-set retirement and retryable teardown;
- an overlap workload whose NPU timeline shows compute and EP communication
  intervals overlapping and whose warmed median wall time is lower than the
  serialized sum by a threshold fixed in the runner; and
- four-rank and eight-rank pure-scale-up async smoke only when the authorized
  device policy permits those topologies.

GPU tests and DeepEP V1 tests are not part of any acceptance layer.

## Later Phase 3E Slices

- 3E.2 is the next planned slice. It splits or redesigns dynamic-shape BF16
  normal and expanded non-cached dispatch so event return, communication-stream
  allocation, asynchronous output extents, handle construction, and descriptor
  publication remain honest. This direction does not expand the accepted 3E.1
  production boundary above.
- 3E.3 remains a deferred design direction for introducing a real
  pre-epilogue dependency boundary and then enabling
  `previous_event_before_epilogue`.
- 3E.4 remains a deferred design direction for adding per-flight queue,
  workspace, diagnostic, completion, and generation slots before permitting
  same-buffer multiflight.
- Hybrid/physical scale-out async is enabled only after its route attestation
  and mapped-memory lifetime move into the generation-bound finalizer.

Each later slice requires its own design update, TDD plan, production build,
multi-NPU reference run, bounded failure tests, and teardown acceptance.
