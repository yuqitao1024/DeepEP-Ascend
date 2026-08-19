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

Phase 3E.1 is not accepted as of 2026-08-19. The exact-archive build/import
task `task_20260819_203518_127255030913` passed, and one-NPU event/lifecycle
task `task_20260819_203858_128526415799` passed its five cases. Two-NPU matrix
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
diagnostic, fail-fast, and persistent-launch hardening. Phase 3E.2 remains the
next planned development slice, but it does not waive the incomplete 3E.1
acceptance gate.

Follow-up task `task_20260819_213704_141211314907` validated the persistent
two-rank launcher and passed capture plus 13 distributed correctness rows,
including 100 generations and two independent buffers, in 1m42s before
failing fast at the injected diagnostic row. The failure identified a
production finalizer regression: async completion converted the structured
device diagnostic to a generic status and lost its error name, command,
peer/rank, backend, and generation detail. A host RED/GREEN now routes async
diagnostics through the established qualified status mapping. The extension
must be rebuilt and the complete matrix, including overlap median/p95 and the
profiler interval, must pass before 3E.1 is accepted.

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
