# EPv2 Ascend Roadmap

## Status

This document is the forward roadmap for the Ascend 950 port of DeepEP V2.
It starts from the validated Phase 2G baseline and covers only the EPv2
`ElasticBuffer` path.

The roadmap is directional. Each phase requires its own reviewed design and
implementation plan before development begins. A phase is complete only after
its stated Ascend build and runtime acceptance passes; the presence of an API,
stub, capability bit, or compile probe is not runtime support.

## Permanent Scope Decisions

The following work is explicitly excluded from this roadmap:

- DeepEP V1, including `deep_ep.buffers.legacy.Buffer`, legacy kernels, and
  the NVSHMEM-based V1 communication path;
- GPU builds, GPU runtime tests, GPU performance runs, and CUDA regression
  verification as part of Ascend acceptance; and
- changing the upstream CUDA implementation to make the Ascend backend work.

The shared Python API and build boundary must continue to isolate the Ascend
backend from CUDA, NCCL, and NVSHMEM. Host-side platform-isolation tests remain
required, but they are not a substitute for, or a request to run, GPU
regression tests.

## Completed Baseline

The following milestones are complete on `main`:

- Phase 1: compile-time CUDA/Ascend backend separation around EPv2
  `ElasticBuffer`;
- Phase 2A: backend-neutral transport contract and explicit capability model;
- Phase 2B: SIMT-callable transport facade and complete stub surface;
- Phase 2C: Ascend core operator schedules developed against the stub;
- Phase 2D: minimal staged SIMT-to-AICore URMA transport primitives;
- Phase 2E: production CANN/HCOMM runtime, ownership, teardown, and barrier;
- Phase 2F: synchronous two-rank BF16 dispatch; and
- Phase 2G: synchronous two-rank BF16 combine.

The accepted production baseline is deliberately narrow: CANN 9.2.0,
`dav-3510`, one host, two Ascend 950 devices, one HCCL scale-up team, BF16,
device-only symmetric communication memory, non-hybrid routing, synchronous
completion, and one operation in flight per `ElasticBuffer`.

## Roadmap Principles

1. Preserve the public EPv2 Python and C++ signatures unless an upstream EPv2
   change requires a versioned contract update.
2. Add transport primitives only when a selected EPv2 consumer requires them.
3. Keep capability bits disabled until the exact end-to-end behavior passes
   multi-NPU acceptance.
4. Land one vertical production slice at a time: resource ownership, transport,
   operator, public API, diagnostics, teardown, and runtime acceptance.
5. Treat bounded failure, generation safety, and retryable teardown as part of
   correctness rather than follow-up hardening.
6. Use Ascend-native performance targets. CUDA results may be an informational
   reference, but no phase depends on executing GPU tests or benchmarks.

## Phase Summary

| Phase | Objective | Primary dependencies |
| --- | --- | --- |
| 3A | Production dependency and deployment closure | Phase 2G |
| 3B | Generalized scale-up topology and operation concurrency | 3A |
| 3C | Cross-host scale-out and RoCE data path | 3A, topology contracts from 3B |
| 3D | Hybrid scale-up/scale-out routing and mapped CPU memory | 3C |
| 3E | Async events, communication streams, and overlap | 3B |
| 3F | FP8 dispatch and combine runtime | 3B; may proceed beside 3C-3E |
| 3G | Pipeline, Engram, AGRS, all-gather, and CP-related modules | Required transport phases below |
| 3H | Performance maturity, long-duration stability, and recovery | All selected production features |

The phase letters define dependency order, not a requirement to implement
every later feature in one release. Each phase may be split into independently
accepted subphases.

## Phase 3A: Production Dependency And Deployment Closure

### Current decision

Status: closed for the current NPU8P development baseline.

Freeze the validated downstream package built from HCOMM baseline `8c5d5ad`
plus `third_party/patches/hcomm/hcomm-team-window-deregister.patch`. NPU8P
tasks consume it through `hcomm-deepep-current`. The MR 4655 backport is
abandoned and further Phase 3A patch development is closed; neither resumes
without a new explicit project decision. Revisit the dependency only when a
compatible vendor package provides the same lifecycle behavior.
An official vendor-distributed package and path-independent deployment remain
desirable, but are deferred and do not block Phase 3B.

### Deliverables

- move the archived HCOMM team/window lifecycle fix into an official package
  or a reproducible, versioned downstream build;
- eliminate dependence on an ad hoc user-local binary overlay;
- pin compatible CANN, HCOMM headers, `libhcomm.so`, compiler, and ABI hashes;
- provide a reproducible Ascend production-extension build and installation
  path; and
- retain lifecycle retry, idempotent destruction, and dependency-order tests.

### Acceptance for the current NPU8P baseline

- a clean environment can build and import the production extension without
  CUDA, NCCL, or NVSHMEM dependencies;
- repeated communicator, team, window, and channel creation/destruction passes;
- the complete Phase 2E-2G two-NPU regression passes with the packaged
  dependency set; and
- NPU8P deployment uses the stable `hcomm-deepep-current` entry point without
  embedding credentials. Path-independent deployment is deferred as recorded
  in the current decision above.

## Phase 3B: Scale-Up Topology And Concurrency

### Deliverables

- generalize the fixed two-rank scale-up contracts to selected 4-rank and
  8-rank single-host topologies;
- remove rank-count assumptions from layout sizing, shard addressing,
  generation records, diagnostics, and test references;
- support multiple independent `ElasticBuffer` instances safely;
- define and implement the selected number of concurrent operations per
  buffer, including generation slots, queue ownership, and failure poisoning;
- validate asymmetric token counts, empty ranks, near-capacity routing, and
  partial failures at every supported rank count; and
- preserve deterministic contributor and lane ordering.

### Acceptance

- the two-rank topology passes the complete public BF16 barrier, dispatch, and
  combine reference matrices;
- the selected four-rank and eight-rank topologies pass public BF16 barrier and
  dispatch/combine round-trip smoke tests;
- concurrent operations cannot consume stale control or payload records;
- timeout diagnostics identify rank, peer, operation, command, channel, and
  generation; and
- repeated teardown is safe after success, timeout, launch failure, and
  partial initialization.

The implementation is rank-parameterized and is not gated to a selected device
count. NPU8P has passed the complete two-rank matrix and public four-rank and
eight-rank BF16 barrier/dispatch/combine smoke tests. Full four-rank and
eight-rank asymmetric, near-capacity, repeated-generation, failure, and
teardown matrices remain open for topology qualification; that gap is not a
runtime rank gate.

## Phase 3C: Cross-Host Scale-Out And RoCE

### Current status

Status: logical 2x2 vertical slice implemented and qualified; physical
multi-host RoCE qualification remains open.

The versioned row-major topology, logical team-to-world peer translation,
route-aware synchronous BF16 barrier/dispatch/combine path, bounded command
validation, collective topology preflight, handle attestation, and teardown
are implemented. NPU8P passed a four-rank `logical-single-host` run using
devices `0,1` as logical host A and `6,7` as logical host B, including 100
barrier generations and an independent BF16 dispatch/combine round trip.

This single-host evidence does not qualify physical RoCE addressing, NIC
selection, cross-host registration, link/process failure behavior, or
performance. The physical scale-out capability remains disabled until the
real multi-host acceptance below passes.

### Deliverables

- define the Ascend scale-out team and topology contract;
- establish the supported RoCE resource, addressing, registration, and
  completion model;
- implement cross-host symmetric-memory publication and peer discovery;
- add the transport operations required by scale-out dispatch and combine;
- implement bounded cross-host barrier, dispatch, and combine schedules;
- qualify failure behavior for unreachable peers, link errors, stale remote
  registrations, and asymmetric preflight rejection; and
- keep scale-up-only calls free of scale-out resource requirements.

### Transport Capability Track

`device get` and direct peer pointer access are not standalone goals. They must
be implemented only if a selected scale-out, hybrid, Engram, pipeline, AGRS,
or CP algorithm requires their semantics and CANN exposes a supportable model.
Each one receives a separate capability bit, compile contract, bounds model,
ordering proof, timeout behavior, and multi-NPU or multi-host acceptance.
Until then, both remain explicitly unsupported.

### Acceptance

- public BF16 barrier, dispatch, and combine match an independent reference on
  the selected multi-host topology;
- at least 100 sequential generations complete without stale state;
- all remote waits are finite and return rank-qualified diagnostics; and
- scale-out resources tear down before the owning process group and memory.

## Phase 3D: Hybrid Routing And Mapped CPU Memory

### Current status

Status: logical 2x2 device-only hybrid routing implemented and qualified;
physical multi-host/RoCE hybrid routing and mapped CPU memory remain open.

The accepted snapshot is commit `53fec72`. NPU8P production build/import task
`task_20260819_121747_7237407472` passed on devices `0,1`. The later serialized
four-rank task `task_20260819_122044_73205324377` used devices `2,3` as logical
host A and `4,5` as logical host B and printed
`PASS logical-single-host device-only hybrid smoke`. It covered local,
scale-up, scale-out-rail, and diagonal dispatch/combine; independent exact BF16
payload and shape checks; fresh and cached handles; empty and asymmetric
routes; both multiple-reduction policies; repeated generations; rank-qualified
bounded failure; poisoned teardown; and repeated destruction. The tracked
source archive had SHA-256
`c4a62f050bee82adb7bf99d49815fce09930bd5d69b67634735a42c45bf6aaa6`.

This evidence qualifies only the explicit `logical-single-host` device path.
It does not qualify physical RoCE addressing, NIC selection, cross-host memory
registration, link/process failure behavior, mapped CPU visibility, or
performance.

The transactional mapped-memory provider, descriptor, bounds, publication,
and retryable reverse-teardown contracts are implemented and covered with a
fake provider. Production mapped CPU capability remains disabled because the
pinned public CANN/HCOMM surface does not expose both a supported host-
allocation device map/unmap pair and documented CPU/NPU release-acquire cache
visibility semantics. Nonzero CPU communication bytes therefore continue to
fail before resource publication.

### Deliverables

- implement scale-up/scale-out hybrid routing through the existing EPv2 handle
  contract;
- define the NPU-device and CPU physical-memory mapping behind one logical
  elastic address space;
- implement mapped CPU buffer allocation, registration, ownership, and
  teardown;
- implement the CPU communication path and host-visible synchronization needed
  by hybrid routing;
- define cross-rank preflight error aggregation before collective publication;
- validate imbalanced EP routing without silently falling back to a local-only
  path; and
- keep device-only configurations independent of CPU communication resources.

### Acceptance

- direct scale-up, direct scale-out, and hybrid routes produce the same public
  reference result for the same routing graph;
- mapped CPU and device segments obey one checked address and capacity model;
- failures before publication are aggregated without stranding another rank;
- CPU-visible completion and teardown are bounded and repeatable; and
- no unsupported route returns fabricated empty tensors or metadata.

## Phase 3E: Async Events, Streams, And Overlap

### Current decision

Phase 3E lands as separate runtime-accepted slices. Phase 3E.1 is limited to
native event/stream ownership, cached BF16 pure-scale-up dispatch, BF16
combine, previous-event ordering, communication-stream allocation, and one
operation per buffer. Its coverage is recorded in the dedicated acceptance
runner and does not promote the 120 full functional benchmark rows that also
require normal and expanded non-cached dispatch.

Phase 3E.2 is the next planned slice for BF16 normal/expanded non-cached
dispatch event return, communication-stream allocation, and asynchronous
publication. Phase 3E.3 (a real pre-epilogue dependency boundary) and Phase
3E.4 (same-buffer multiflight resource slots) remain deferred design
directions. FP8 remains Phase 3F and is not credited to any Phase 3E slice.

Phase 3E.1 is accepted as of 2026-08-20. The production-backed final evidence
appears after the qualification history below. Phase 3E overall remains
incomplete; 3E.2 is the next planned development slice, while 3E.3, 3E.4, and
FP8 Phase 3F remain outside this acceptance boundary.

#### Qualification history

Before final qualification, Phase 3E.1 was not accepted. The exact-archive
build/import task `task_20260819_203518_127255030913` passed, and one-NPU
event/lifecycle task `task_20260819_203858_128526415799` passed its five cases.
Two-NPU matrix
task `task_20260819_204026_128929210009` passed only
`capture-current-stream`; cached dispatch sync/async with allocation
false/true, previous-event allocation, combine sync/async with allocation
false/true, empty-route, and asymmetric-route then exited 1. The job reached
its 300-second hard limit before `100-generations`, `two-independent-buffers`,
`record-failure`, `event-timeout`, `diagnostic-failure`,
`completion-mismatch`, `drop-event`, `destroy-pending-retry`, or
`overlap-vs-serialized` ran. It produced no overlap median/p95 or NPU profiler
interval. Captured child diagnostics were lost at outer timeout and the
targeted TorchElastic directories were empty, so the distributed root cause
was not established by that job. The runner subsequently gained checkpoint,
diagnostic, fail-fast, and persistent-launch hardening. At that time, Phase
3E.2 was the next planned development slice, but it did not waive the
incomplete 3E.1 acceptance gate.

Follow-up task `task_20260819_213704_141211314907` validated the persistent
two-rank launcher and passed capture plus 13 distributed correctness rows,
including 100 generations and two independent buffers, in 1m42s before
failing fast at the injected diagnostic row. The failure identified a
production finalizer regression: async completion converted the structured
device diagnostic to a generic status and lost its error name, command,
peer/rank, backend, and generation detail. A host RED/GREEN now routes async
diagnostics through the established qualified status mapping.

Exact-source rebuild task `task_20260819_221458_163301111405` passed on
devices `6,7` in 2m54s. Fail-fast matrix task
`task_20260819_222014_164400825732` then passed 15 cases, including the
qualified diagnostic row, before `overlap-vs-serialized` exceeded the
production event-specific 5-second completion deadline on both ranks. It
exited 1 after 1m0s with one failed and five not run. Profiling was never
reached, so no serialized/overlapped median or p95, stream IDs, or positive
NPU profiler interval existed. The complete matrix and overlap evidence still
had to pass before 3E.1 could be accepted.

Bounded follow-up evidence on 2026-08-20 shows that the 4096-token control was
too long for the five-second event deadline, while a 256-token matrix completed
and measured only `0.014373`/`0.012169` median improvement, below the fixed 5%
gate. Sweep task `task_20260820_022734_29035825747` then completed 256 and 512
before timing out during pre-profiler work at 1024. Its Torch-NPU traces use a
top-level JSON array; after a host-tested runner parser correction, offline
analysis proves positive NPU overlap on both ranks at both completed points:
about 268-270ms at 256 and 588-594ms at 512. The old runner lost the associated
wall-time summaries, and no 1024 trace exists, so this is hardware-overlap
evidence but not Phase 3E.1 acceptance.

The next proposed NPU action was not another three-point sweep. It was one
diagnostic-only 1024 point with zero warmups, one repetition, explicit phase
checkpoints, a 150-second controller bound, and the required 300-second
TaskQueue bound. The fixed 5% threshold remained unchanged, and one sample
could not be promoted to the warmed-median acceptance result. No retry was
submitted until that bounded shape and the runner evidence were reviewed.

The approved diagnostic, `task_20260820_025915_29863145111`, subsequently
exited 0. Both ranks completed all event waits and profiler phases. At 1024
tokens, communication was approximately 1.388 seconds while isolated compute
was only 11-12ms, limiting ideal improvement to 0.81-0.87%; observed
improvement was 0.10-0.22%. Exact traces also show approximately 1.381 seconds
of positive MatMulV3/DeepEP-dispatch overlap on physical streams `61/26` on
both ranks. Runtime objects expose those streams as logical IDs `0/128`.

This closes the bounded diagnostic question: hardware overlap is present, but
the tested load ratio cannot satisfy the fixed 5% wall-time criterion. Because
the run used one sample and is explicitly acceptance-ineligible, Phase 3E.1
remained not accepted after that task.

The final acceptance matrix therefore retains 256 communication tokens,
`4096x4096` BF16 compute, three warmups, seven repetitions, the fixed 5% gate,
and the 30-second per-case watchdog, while increasing only the runner compute
iterations from 8 to 256. The 256-token trace evidence puts communication at
approximately 0.27-0.59 seconds; scaling the measured 8-iteration compute
component estimates 256 iterations at 0.36-0.39 seconds. Even the high-end
static estimate for all warmup, measured, and profiler iterations is about
14.3 seconds, so the existing watchdog retains margin without reducing the
meaningful seven-sample median. This is runner-only load balancing, not a
production change, event-deadline increase, or acceptance-threshold decrease.

The runner also separates logical runtime stream IDs from physical profiler
lane IDs. Physical lanes must be uniquely identified by the `MatMulV3` and
DeepEP `dispatch_kernel` event families, must differ, and must have a positive
overlap interval. At that point, one final complete 21-case two-rank matrix
remained required before Phase 3E.1 could be accepted.

Final full-matrix task `task_20260820_031911_304861030703` reached the overlap
row but did not pass it. The fail-fast report contains 21 selected, 16
executed, 15 passed, one failed, and five not run. Rank 0 serialized/overlapped
medians were `0.360265130`/`0.360237910` seconds for only `0.000075555`
improvement; rank 1 medians were `0.361493950`/`0.358961650` for
`0.007005097`. Both are below 5%. Nevertheless the event-family parser proves
positive physical overlap of `268335.5us` and `270927.0us`, with logical IDs
`0/143` and physical IDs `61/11` on both ranks.

That result kept Phase 3E.1 unaccepted and narrowed the next action to one
bounded component-timing diagnostic. It had to distinguish an unserialized
control from true compute/communication resource contention before any
production change or further full-matrix run was considered.

The single authorized component diagnostic,
`task_20260820_033904_310487927692`, then completed successfully on devices
`6,7`. Both ranks classified as `resource-contention`. Communication-only was
`0.279502830`/`0.279471440` seconds and compute-only was
`0.095801200`/`0.095291040`; serialized was
`0.359756550`/`0.359411640`, within the measured 5%-of-component-sum
tolerances of `0.018765202`/`0.018738124`. Synchronous dispatch return itself
took `0.278580530`/`0.278420550`, proving that the baseline does wait for
communication.

Overlapped wall was `0.357143020`/`0.357857410`, saving only
`0.002613530`/`0.001554230`, below measured serialized-wall tolerances of
`0.017987828`/`0.017970582`. During overlap, event wait expanded to
`0.342516940`/`0.342942600` seconds even though the profiler proved
`269032.25us`/`268594.75us` of physical overlap on streams `61/26`. This rules
out an unserialized control and supports compute/communication resource
contention. No production change or unchanged full-matrix retry followed from
this result; Phase 3E.1 remained unaccepted under its fixed 5% gate while the
specific contention source is investigated separately.

Static investigation also rules out two immediate tuning assumptions. In the
qualified Torch-NPU revision, `getStreamFromPool(true)` selects logical
priority `-1`, but both logical high and normal pools create CANN streams at
physical priority `0` with the same fast-launch/fast-sync flags; changing the
pool selector would not change physical stream priority. The qualified
dispatch has `num_blocks=1`, `num_threads=512`, a public `num_sms=1` contract,
an outer `__global__ __vector__` kernel launch by `num_blocks`, and matching
AIV/vector symbols in the compiled object. This establishes one outer AIV
block, but available vendor documentation does not establish its exact
physical-core mapping.

Component-only commit `8e812bd` then replaced its mixed compute chain with
exactly 256 independent matmuls and no `mul`/`mul_`, without changing the
formal acceptance workload. The only authorized pure-matmul diagnostic,
`task_20260820_040533_315212229000`, completed on devices `6,7` and again
classified both ranks as `resource-contention`. Communication-only was
`0.278048730`/`0.278018550` seconds, pure-matmul compute-only was
`0.087509200`/`0.086884680`, serialized was
`0.353972660`/`0.354368630`, and overlapped was
`0.353502300`/`0.353292460`. The gains were only `0.1329%` and `0.3037%`,
while event waits remained stretched to `0.341599990`/`0.340491320` seconds
despite `268224.5us`/`269994.5us` of physical overlap on streams `61/26`.
Removing the elementwise chain reduced compute-only time but did not remove
the communication stretch, so the `mul_` contention hypothesis is rejected.
The formal mixed workload and fixed 5% gate remained unchanged, no further full
matrix was justified, and Phase 3E.1 remained unaccepted.

The next component-only capability probe used BF16 NCDHW
`torch.nn.functional.conv3d` with primary `Conv3DV2` compute. Runner commit
`8aaa624` and task `task_20260820_045758_323960612146` changed no production
code or formal acceptance workload. Both ranks established valid serialized
baselines, event waits near 0.221 seconds, exact primary
`Conv3DV2=KERNEL_AICORE` and dispatch `KERNEL_AIVEC` task types, distinct
physical streams `61/26`, and positive `28.5/28.75us` intervals. Serialized
walls of `0.332164400/0.332342290` seconds fell to overlapped walls of
`0.275529020/0.275268150`, gains of `17.0504%/17.1733%`.

The capability remains qualified rather than accepted. Calibration reached
its `2039/2048` iteration clamp while compute-only remained about 56ms, below
the intended 0.20-0.30s range. Each trace also inventories tiny `Greater` and
`ZerosLike` auxiliary AIV families and no TransData, so the primary kernel is
AIC-only but the complete compute path is not. The report SHA-256 is
`88a4fa784c660ffa078b1a443e6b0b9573230ceed458a3932eb2c7944c31a813`.
This single-sample result established capability and motivated a formal
fixed-workload matrix; Phase 3E.1 remained unaccepted pending that matrix's
unchanged 5% and profiler gates.

The resulting final matrix keeps communication fixed at 256 tokens, hidden
size 4096, and `num_sms=1`, and replaces only its formal compute workload with
256 iterations of BF16 NCDHW Conv3D over input `(1,64,24,96,96)` and weight
`(64,64,3,3,3)`, using unit stride/dilation, zero padding, and groups 1. It
retains three warmups, seven measured samples, the production five-second event
deadline, the 5% per-rank median gate, and the 30-second case watchdog; there is
no formal calibration, MatMul, `mul`, or `mul_`. The static timed/profile
workload estimate is `9.459426836` seconds before setup and trace export, so no
warmup or repetition reduction is justified.

Formal profiler acceptance requires exact `Conv3DV2=KERNEL_AICORE` and
`dispatch_kernel=KERNEL_AIVEC` task types, distinct physical streams, positive
overlap, retained logical/physical IDs, and an inventory of every auxiliary AIV
family. Auxiliary AIV total duration must stay below 1% of the primary Conv3D
span. Known tiny `Greater` and `ZerosLike` families are allowed within that
bound, but `compute_path_aic_only` remained false and no whole-path pure-AIC
claim was made. The declared gate required one complete 21/21 matrix with both
ranks above 5% and every profiler check passing.

#### Production-backed final acceptance

Independent review rejected task `task_20260820_052826_33515907092` as final
acceptance evidence because five failure and lifecycle rows used host
fake-runtime contracts rather than production workers. Fix round 1 removed
that routing, reduced the one-NPU suite to its three true event cases, moved
the remaining lifecycle rows into the persistent two-rank worker, and added
testing-only exact-value native failure controls. The earlier result remains
diagnostic history and is superseded for the acceptance decision.

Exact-source build task `task_20260820_062500_37256458511` completed in
`2m55s`. Its local and remote source archives both had SHA-256
`573d6970812fe4f8cf6b60b06d2d4e0bb9547ddfc35cd8177d69d677922955de`;
the build log printed the actual hash and matched the expected value. The task
produced extension SHA-256
`dd9e91a69b8526ba9b828387fdc1b655a46f4a4ae10ce4af3d36fec7fd41bb67`
from production commit/tree
`28f3deaed1b950e782ba57a0d0256cd2f35687f9`/
`47ed60717b538b88436fa2602628f28b9a76f234`. Runner commit
`b185d95c28864c22faf99dc67f5492028e6383fb` has file SHA-256
`e3154dd502d1f9986d16d7b131520d6ab3d292d191e4cb949ecb7a686491300f`.
The corrected one-NPU event task `task_20260820_063113_375337613415` passed
3/3 on device 6 in `52s`. The first corrected full task,
`task_20260820_063252_376006031537`, passed 16 rows before `drop-event`
failed on a stale generation-bound dispatch handle, leaving four rows not run.
A deterministic host RED reproduced the invalid reuse: event drop finalized
the first generation, so the buffer remained reusable but its seed handle did
not. Runner commit `b185d95c28864c22faf99dc67f5492028e6383fb` reseeds the
buffer before the second dispatch and passed the full host suite with 199
passed, 7 skipped, and 48 subtests.

Final task `task_20260820_064309_380109210223` ran on devices `6,7` with
`--max-time 300`, reached authoritative `completed (exit=0)` in `1m49s`, and
passed 21/21 selected and executed cases. Rank 0 serialized/overlapped medians
were `0.304671370`/`0.277607860` seconds for an `8.8829%` gain; rank 1
measured `0.304223720`/`0.277838260` for `8.6730%`. Both cleared the unchanged
5% gate with three warmups, seven measured samples, and zero global
synchronizations.

Profiler evidence identifies `Conv3DV2=KERNEL_AICORE` and
`dispatch_kernel=KERNEL_AIVEC`, logical streams `0/146`, and positive overlap
of `110.5us`/`112.0us` on distinct rank-local physical lane pairs `56/6` and
`61/8`. Auxiliary AIV ratios were `0.012324%`/`0.010512%`, below 1%, and
neither trace contains TransData. Both ranks also passed real event drop and
buffer reuse, completion mismatch, and retryable pending destroy. Record
failure and event timeout passed through the one-NPU production workers.

Report SHA-256 is
`0e6d7674f03967285ead97a75d7c99cb7537a8157ca9794c2a2bb644221c82ee`;
rank trace hashes are
`0c079bbdc9bc5c722394079fb5bd14dba9e23ec0b369ce5479cc486e04a8b5f9`
and `e9dc5c3c487a63ed3d0f8f70c75884171aa0a15e435f14bf80fcef5ef4882001`.
Every Phase 3E.1 gate passed. All 120 full functional benchmark rows remain
deferred to 3E.2, and 3E.3, 3E.4, and FP8 Phase 3F remain incomplete.

Post-rebase integration qualification used exact source commit
`88829ba02a2b8a3f964fca8a27de2fa01c70a93a`. NPU8P build task
`task_20260820_115216_277302926350` and final devices-`6,7` task
`task_20260820_115539_278562813437` both reached `completed (exit=0)`; the
final task passed 21/21 selected and executed cases with no failures or not-run
rows. Rank 0 and rank 1 median overlap gains were
`8.6122%` and `8.1679%`; profiler evidence retained logical streams `0/146`,
distinct physical streams `61/8`, and positive overlap of `111.5us` and
`110.25us`. The 45-second external case watchdog only accommodates bounded
CANN profiler export; the native five-second event deadline, 5% gate, three
warmups, seven samples, and 300-second TaskQueue bound remain unchanged.

The final archive and production extension SHA-256 values are
`44bc33e8e2639cf7ebf6b6af4f22e507f4b439d1e2ac7bc3c2f2860114e1b87a`
and `869b0a1e8ce4fac816a182c411f8fef15a4d2619dbd5eaecfd68336e99e1b606`.
The report SHA-256 is
`fd795707cae564a4bc73415e96d75410ebc6b0a682a21034fcc72fb97cf3385a`;
rank trace hashes are
`fb0b6660198601e19e13ec0507d71fff455086d1db272f1993c982e364241519`
and `c573782f9de130389b24784ee6d9f35831f046b4c0a03842234db87c2371ff28`.
Final review regressions additionally require graph capture to fail closed when
Torch-NPU cannot report capture state, release the Python GIL around bounded
Ascend buffer destruction, and preserve Torch-NPU ownership of the underlying
pool stream while DeepEP retains its checked identity.
This closes Phase 3E.1 only and does not credit any deferred 3E.2-3E.4 work.

### Deliverables

- implement native Ascend event ownership behind `EventOverlap`;
- expose the supported communication stream through the existing public API;
- implement `async_with_compute_stream`, `allocate_on_comm_stream`, previous
  event chaining, and stream-safe output lifetime;
- support the selected number of in-flight operations without generation or
  queue aliasing;
- define cancellation, failure poisoning, and teardown while work is pending;
  and
- add communication/computation overlap without changing synchronous results.

### Acceptance

- synchronous and asynchronous calls produce identical public results;
- event waits establish correct visibility without a hidden global device
  synchronization;
- stream ownership remains valid until every returned tensor is safe to use;
- overlap is demonstrated on Ascend with a reproducible workload; and
- failure and destruction with pending work complete within finite bounds.

## Phase 3F: FP8 Runtime

Status: synchronous E4M3 dispatch with FP32 or packed UE8M0x4 scale-factor
transport is implemented; BF16 combine remains the selected upstream-compatible
combine path. The synchronous Phase 3F scope is complete for single-host 2-,
4-, and 8-rank execution.

TaskQueue run `task_20260819_171408_412650329832` established the clean
`DEEP_EP_ASCEND_TESTING=0` CANN 9.2.0 production build/import and dependency
audit: no CUDA, NCCL, or NVSHMEM dependency was present. Before a benchmark
verifier failure, 176 host tests passed with 2 skipped across 23 subtests, the
two-rank FP8 matrix passed 12 cases including empty column-major output, the
barrier passed 100 generations, BF16 dispatch passed 14 cases, and BF16
combine passed 24 cases. The verifier incorrectly indexed an NPU E4M3 tensor
directly; commit `fa30444` changed correctness indexing to its byte view.

The next multi-rank run `task_20260819_173647_53443028121` passed the repaired
two-rank BF16/FP8 benchmark and reached four-rank combine. It exposed that the
test oracle summed integer values before BF16 conversion, while the runtime
correctly receives individually quantized BF16 contributions and accumulates
them in FP32. Commit `72de60f` fixed the oracle and added a host regression for
the quantization order.

Final serialized run `task_20260819_184826_108102819968` used devices 0-7 at
commit `72de60f` and exited 0. It passed 56 focused host tests plus 20 subtests,
the two-rank BF16/FP8 benchmark smoke with 2 cases, and the complete 12-case
FP8 runtime matrix at both four and eight ranks. Together with the clean-build,
dependency, and broad two-rank evidence above, this closes synchronous Phase
3F. It is not performance qualification, physical multi-host qualification,
FP8 combine support, or evidence for the Phase 3E async/stream-overlap path.

### Deliverables

- implement the selected E4M3 dispatch contract with FP32 or packed UE8M0x4
  scale-factor transport, while retaining BF16 combine;
- validate supported Ascend FP8 representations and conversion rules;
- preserve BF16 combine behavior where the public contract requires it;
- cover normal, expanded, padded, cached, weighted, empty, and asymmetric
  routing; and
- define numerical tolerances independently of the production kernels.

### Acceptance

- public FP8 dispatch and the selected combine path match an independent
  reference across supported ranks and modes;
- scale factors, padding, cached handles, and `-1` lanes are validated exactly;
- malformed dtype, shape, scale, and handle inputs fail before launch; and
- BF16 regressions remain green on Ascend.

## Phase 3G: Auxiliary EPv2 Modules

This phase is a collection of separate vertical slices, not one combined
implementation task.

### Pipeline Send/Receive

- implement `pp_set_config`, `pp_send`, and `pp_recv` on the selected scale-up
  and scale-out domains;
- provide finite completion, ordering, size bounds, and peer diagnostics; and
- validate repeated and bidirectional transfers through the public API.

### Engram

- implement `engram_write` and `engram_fetch` using the mapped-memory and
  remote-access capabilities selected in Phases 3C and 3D;
- support the required BF16 and FP8 storage contracts; and
- validate ownership, cache visibility, missing entries, bounds, and teardown.

### AGRS

- implement session lifecycle, configuration, in-place tensor ownership, and
  the selected communication schedule;
- isolate concurrent sessions and make session destruction retryable; and
- validate public session context-manager behavior and failure cleanup.

### All-Gather And CP-Related Operations

- implement the EPv2 `all_gather` surface and selected context-parallel
  operations only after their transport semantics are specified;
- keep them separate from legacy V1 collectives; and
- define independent correctness and topology acceptance for each operation.

Each auxiliary slice requires its own design, plan, Ascend build, public API
tests, multi-NPU runtime reference, diagnostics, and teardown acceptance.

## Phase 3H: Performance And Reliability Maturity

### Deliverables

- tune AICore block count, queue depth, channel use, packing, and pipeline
  boundaries for supported Ascend topologies;
- reduce synchronization, intermediate memory, and host round trips;
- add long-duration generation, lifecycle, and failure-injection runs;
- define behavior under transient transport and teardown failures;
- track bandwidth, latency, AICore occupancy, memory use, and overlap efficiency;
  and
- publish reproducible Ascend benchmark configurations and results.

### CUDA Comparison Policy

Functional compatibility with the shared EPv2 API remains required. Competitive
performance relative to upstream CUDA results is a long-term reference goal,
but measured CUDA parity is not an acceptance criterion because this project
does not run GPU regression or GPU benchmark jobs. Published upstream results
may be cited for context; completion claims must be based on reproducible
Ascend measurements and Ascend hardware limits.

### Acceptance

- correctness matrices remain green after every optimization;
- performance results are reproducible across clean builds and repeated runs;
- no optimization weakens finite timeout, generation, diagnostic, or teardown
  guarantees; and
- the selected production workload meets explicit Ascend throughput, latency,
  memory, and stability targets defined before implementation.

## Validation Policy

Every implementation phase uses the following evidence layers:

1. host contract and platform-isolation tests;
2. clean CANN/Bisheng builds for every changed AOT and production extension;
3. dependency audits excluding CUDA, NCCL, and NVSHMEM from Ascend artifacts;
4. public PyTorch NPU reference tests on the selected topology;
5. bounded negative-path diagnostics and lifecycle failure injection;
6. repeated generation and construct/destroy coverage; and
7. one final serialized TaskQueue acceptance after focused debugging is green.

NPU work must follow the active TaskQueue policy and device authorization. A
roadmap phase that needs a larger topology cannot be declared complete using a
smaller topology, and the device policy must not be bypassed to obtain the
result.

GPU regression verification and legacy V1 support are excluded at every layer.
They must not be added to a phase's acceptance matrix, release gate, or progress
percentage.

## Progress Reporting

Progress is reported against this EPv2 Ascend roadmap, not against all files or
APIs in the upstream repository:

- `complete`: the full Ascend acceptance boundary passed;
- `in progress`: implementation or focused validation is active;
- `interface only`: the public shape exists but runtime execution is gated;
- `deferred`: included in this roadmap but not assigned to an active phase; and
- `excluded`: legacy V1 or GPU regression work that this project will not do.

An interface-only or compile-only feature does not contribute to completed
runtime migration progress.
