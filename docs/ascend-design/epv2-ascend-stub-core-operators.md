# EPv2 Ascend Stub-Based Core Operators

## Status

This document defines Phase 2C of the EPv2 Ascend 950 migration. It builds on
the backend split and device transport facade completed in Phases 1, 2A, and
2B. Phase 2C ports the operator structure against the Phase 2B communication
stubs. It does not enable production communication.

The target is Ascend 950 with CANN 9.2 and `dav-3510`. CUDA sources and CUDA
runtime behavior remain unchanged.

## Decision Summary

Phase 2C will:

1. add Ascend implementations for a single-host scale-up barrier,
   non-hybrid dispatch, and non-hybrid combine;
2. preserve the upstream division between layout/counting, data movement,
   communication scheduling, and combine reduction;
3. route every device communication operation through the Phase 2B
   `DeviceTransportFacade` and its zero-capability stubs;
4. precompile `.asc` kernels ahead of time and pass runtime tiling and
   workspace descriptors to them;
5. expose an internal test launcher that permits only one rank and validates
   communication-independent local behavior on real Ascend 950 hardware;
6. keep public `ElasticBuffer.barrier`, `dispatch`, and `combine` calls behind
   the existing host transport capability gates;
7. compile the FP8 dispatch surface, but use BF16 as the Phase 2C runtime
   correctness baseline; and
8. keep async events, CPU synchronization, hybrid, pipeline, and engram paths
   at interface and compile coverage only.

No successful Phase 2C test is evidence of multi-rank communication.

## Goals

Phase 2C will provide:

- backend-local, vendor-neutral token and workspace layouts for Ascend;
- explicit host-side tiling and workspace size calculations;
- AOT Ascend kernel entry points for barrier, dispatch, dispatch copy
  epilogue, combine, and combine reduction epilogue;
- structurally visible transport calls at the same scheduling points as the
  corresponding upstream Gin calls;
- a local-rank fast path that performs real copy, metadata, prefix-sum,
  expansion, zero-padding, and reduction work without relying on a stubbed
  remote operation;
- host launch plumbing with checked arguments and deterministic error
  reporting;
- compile probes for communication and deferred feature branches; and
- device tests on node 20002 for the approved single-rank BF16 matrix.

## Non-goals

Phase 2C will not:

- advertise any host transport capability;
- let a public multi-rank API reach a no-op communication stub;
- emulate remote writes by copying local memory;
- report multi-rank barrier, dispatch, or combine correctness;
- initialize HCCL teams, symmetric windows, AIN channels, or Hcomm contexts;
- bind public AIN calls or CANN internal communication APIs;
- use `HcclAlltoAllV` or another collective fallback;
- implement cross-host scale-out communication;
- complete hybrid, pipeline, engram, CPU-sync, or async-event behavior;
- require FP8 runtime numerical validation;
- reproduce CUDA JIT compilation on Ascend; or
- tune operator performance.

## Considered Build Strategies

### AOT Ascend kernels with runtime tiling

This is the selected approach. `.asc` sources are compiled for `dav-3510`
when the Ascend extension or its operator test target is built. Shapes and
mode flags remain runtime data in small POD tiling descriptors. This matches
the CANN toolchain, gives stable kernel symbols and ABI, and keeps the Phase 2C
work focused on operator semantics.

### Runtime Ascend JIT

A CUDA-like JIT would preserve the upstream template specialization model,
but it would require compiler discovery, source generation, cache invalidation,
binary loading, error handling, and process-lifetime management. None of that
validates the Phase 2B transport boundary. It is rejected for Phase 2C.

### Standalone compile probes only

Standalone object builds are useful for compiler qualification, but they do
not validate tiling, kernel registration, argument marshalling, device launch,
or data correctness. They remain part of testing but are not the operator
integration strategy.

## Architecture

```text
Python EPv2 interface
        |
Ascend ElasticBuffer capability gate  (production stays closed)
        |
Ascend host operator API and launcher
        |
runtime tiling + workspace descriptors
        |
AOT .asc kernels
        |
local-rank data path ---- DeviceTransportFacade ---- Phase 2B stubs
```

The internal test launcher enters below the production capability gate. It is
compiled only for the Ascend backend and rejects any topology whose world,
scale-up, or scale-out size is not one. It is not part of the Python EPv2
surface.

## Source Ownership

Ascend operator code is owned below `csrc/backends/ascend` and does not modify
the CUDA implementations in `deep_ep/include/deep_ep/impls` or
`csrc/kernels/elastic`.

The intended ownership is:

```text
csrc/backends/ascend/elastic/
  layout.hpp            backend-local layout and size arithmetic
  tiling.hpp            POD tiling and launch descriptors
  kernels.hpp           host-visible kernel declarations
  runtime.hpp/.cpp      validation, workspace calculation, and launch API
  barrier.asc           barrier kernel
  dispatch.asc          dispatch and copy-epilogue kernels
  combine.asc           combine and reduction-epilogue kernels
```

Tests may add an Ascend-only executable or extension bridge below
`tests/ascend`. Production code must not depend on test code.

## Backend-Local Layout Contract

CUDA `layout.cuh` is not directly reusable because it exposes CUDA qualifiers,
PTX alignment constants, and an `mbarrier` field. The Ascend layout keeps the
same logical regions and token field order where they affect EPv2 semantics,
but owns its alignment and temporary-memory rules.

### Workspace regions

The Phase 2C workspace contains, in order:

1. barrier epoch and arrival storage;
2. per-block rank and expert counters;
3. reduced rank and expert counters;
4. rank and expert prefix sums;
5. dispatch slot indices and source metadata; and
6. kernel-local scratch offsets required by the selected tiling.

Every region is represented by an offset and byte count. Size arithmetic is
checked for overflow on the host. Device code receives the descriptor by
value and never reconstructs the layout using CUDA constants.

### Token regions

A communication token keeps these logical fields:

1. hidden payload;
2. optional FP8 scale-factor packs;
3. top-k indices;
4. optional top-k weights;
5. source token metadata; and
6. linked-list or slot metadata required by a deferred hybrid interface.

Fields that are unused by a mode have zero bytes. The layout uses an explicit
Ascend alignment constant and exposes only offset arithmetic; it contains no
backend runtime handle.

## Tiling Contract

Host tiling is runtime-driven and uses trivially copyable POD structures.
Each descriptor contains:

- version and operation kind;
- rank and topology fields;
- token, hidden, expert, and top-k dimensions;
- element size and scale-factor strides;
- expert alignment and maximum tokens per rank;
- mode flags for cached, expanded, zero-padding, and multiple reduction;
- block, workgroup, and participant counts;
- buffer and workspace offsets; and
- the `DeviceTransportContext` plus channel assignment needed by facade call
  sites.

Validation occurs before launch. Invalid dimensions, unsupported modes,
misaligned storage, insufficient workspace, and non-single-rank use of the
internal launcher return an error without launching a kernel.

Phase 2C does not select performance-optimal tilings. Initial tilings favor
simple, deterministic work partitioning and correctness on `dav-3510`.

## Device Execution Model

The operator kernels use an explicit Ascend SIMT VF seam. A kernel entry owns
the AICore wrapper and calls one or more SIMT VFs using the compiler-supported
mixed AICore/SIMT mode. The build must use the normal ASC CMake language flags;
it must not add `--enable-simt`, which selects an incompatible pure-SIMT mode
on the validated CANN 9.2 toolchain.

Phase 2C does not transliterate PTX. It replaces:

- CUDA warp/lane queries with Ascend SIMT thread and workgroup indices;
- TMA transactions with explicit vectorized global-memory copies;
- CUDA named barriers with supported Ascend block/workgroup synchronization;
- CUDA grid dependencies with separate ordered kernel launches; and
- CUDA block reductions and scans with deterministic Ascend-local algorithms.

All device helpers used by a SIMT VF must carry the validated SIMT callee
qualifier through a backend-local macro.

## Barrier

The barrier kernel preserves separate prologue/final barrier call sites and
team selection in its arguments. For an internal single-rank launch it executes
only local ordering and completion steps and finishes successfully.

For a remote peer, the kernel constructs `DeviceTransportFacade` and issues the
required barrier/signal operations through it. Those calls are compileable but
have no effect under the Phase 2B stub. The production host gate prevents this
path from being presented as a successful operation.

The internal launcher rejects rank counts greater than one before allocation
or launch, so the local success case cannot hide a missing remote barrier.

## Dispatch

Dispatch is split into two ordered kernels so that Phase 2C does not require a
device-wide grid dependency:

1. dispatch/count-and-place;
2. dispatch copy epilogue.

### Count-and-place

The first kernel:

- counts valid expert selections;
- deduplicates rank selections;
- computes unaligned and aligned per-expert counts;
- builds rank and expert prefix sums;
- assigns deterministic destination slots;
- writes source metadata and the cached slot map; and
- schedules remote put, put-value, signal, and barrier facade calls at the
  corresponding semantic points.

For a single rank, peer accessibility resolves to the original local address.
Payload and metadata are copied directly to their local destination. The code
does not depend on `put`, `signal`, or `device_barrier` stubs to move data or
make it visible.

### Copy epilogue

The second kernel materializes the public receive layout. It supports normal
and expanded layouts, optional top-k weights, cached slot reuse, and zeroing of
expert-alignment padding. It also copies FP8 scale-factor fields when present;
the FP8 path is compile-tested in Phase 2C.

Cached dispatch consumes an existing validated slot map and metadata layout.
It does not silently recompute routing.

## Combine

Combine is split into two ordered kernels:

1. combine/place-or-local-reduce;
2. combine reduction epilogue.

The first kernel reads dispatch source metadata and writes the master token
buffer. It preserves the upstream distinctions between normal layout,
expanded layout, and multiple local reduction. Remote destination selection
and completion remain explicit facade call sites.

For a single rank, hidden payloads and optional top-k weights are placed in the
local receive buffer without using a communication stub. The epilogue then:

- selects valid top-k sources;
- reduces BF16 payloads using a sufficiently precise accumulator;
- applies zero, one, or two optional BF16 bias tensors;
- writes combined top-k weights; and
- produces one combined token per original token.

Separate kernel launches provide the Phase 2C ordering between placement and
reduction. Phase 2D may optimize this once real completion semantics exist.

## Deferred Interfaces

The tiling flags, metadata offsets, and function declarations needed by these
paths remain representable and compileable:

- async event completion;
- CPU synchronization and mapped host workspace;
- hybrid scale-up/scale-out routing;
- pipeline send/receive;
- engram read/write; and
- FP8 dispatch execution.

Their host entry points return an explicit unsupported status or remain behind
the existing capability gate. No deferred branch may return fabricated output.

## Host Runtime and Error Handling

The Ascend host runtime owns:

- tiling construction and validation;
- workspace and communication-buffer size calculation;
- kernel binary/symbol lookup;
- launch argument marshalling;
- stream-ordered launches; and
- conversion of ACL/CANN failures into existing DeepEP error conventions.

The host operator API returns a status before touching output memory when an
operation is unsupported. Device-side argument failures are avoided through
host validation rather than reported through a fake communication result.

The production `ElasticBuffer` calls `require_transport` before allocation,
tiling, or launch. Phase 2C leaves the stub capability mask at zero, so public
barrier, dispatch, and combine behavior remains unchanged.

## Build Integration

The Ascend build gains an ASC-language operator target only when
`DEEP_EP_PLATFORM=ascend`. It targets `dav-3510`, includes only public CANN
headers needed for kernel and ACL launch support, and links the resulting
kernel artifact and host runtime into the Ascend test or extension target.

The CUDA build graph, source list, compile flags, and link libraries are not
changed. Ascend headers must not leak into CUDA targets or backend-neutral
Python code.

A clean build must not depend on NCCL, NVSHMEM, CUDA, AIN, Hcomm, or CANN
internal headers.

## Test Strategy

### Host contract tests

Tests running without an NPU verify:

- layout offsets, alignment, byte counts, and overflow rejection;
- deterministic tiling for supported shapes and modes;
- rejection of invalid dimensions and rank counts;
- zero advertised transport capabilities;
- production gate-before-allocation and gate-before-launch behavior;
- internal launcher visibility only in Ascend test builds; and
- compile-time isolation from CUDA, NCCL, NVSHMEM, AIN, and Hcomm.

### Ascend 950 compile tests

On node 20002, clean builds for `dav-3510` must compile:

- all three operator families and both epilogues;
- every `DeviceTransportFacade` call used by them;
- BF16 runtime variants;
- FP8 dispatch layout and copy variants; and
- deferred async, CPU-sync, hybrid, pipeline, and engram interface branches.

Compiler warnings about invalid SIMT/AICore call direction or qualifier loss
are failures.

### Ascend 950 single-rank execution tests

The internal launcher runs with world, scale-up, and scale-out size one. BF16
tests compare device output against a CPU or Torch reference for:

- local barrier completion;
- normal dispatch and combine round trip;
- expanded dispatch and combine;
- expanded zero-padding;
- cached dispatch reuse;
- `allow_multiple_reduction` enabled and disabled;
- top-k weights; and
- combine with zero, one, and two biases.

Cases include empty input, invalid expert ids, repeated rank selections,
multiple valid experts per token, expert alignment gaps, and nontrivial hidden
sizes. Tests verify payloads, counts, prefix sums, slot maps, source metadata,
top-k indices, top-k weights, and final combined values.

No test uses a rank count greater than one. No expected value depends on a
stubbed put, get, signal, wait, flush, or barrier side effect.

## Acceptance Criteria

Phase 2C is complete when:

1. CUDA production files are unchanged and CUDA runtime validation remains
   deferred by project decision;
2. Ascend host-only contract tests pass;
3. clean `dav-3510` AOT builds pass on node 20002 without forbidden
   dependencies or SIMT qualifier warnings;
4. the approved BF16 single-rank runtime matrix passes on Ascend 950;
5. the FP8 dispatch and deferred feature surfaces compile;
6. production Ascend barrier, dispatch, and combine still fail at the host
   capability gate before allocation or launch; and
7. documentation states clearly that real single-host multi-NPU correctness
   remains a Phase 2D requirement.

## Phase 2D Boundary

Phase 2D replaces only the lowest device transport and host resource layers.
It provisions teams, windows, channels, device contexts, ordering, completion,
timeouts, and teardown, then validates real single-host multi-NPU behavior.

Operator call sites, tiling descriptors, local layouts, and host launch APIs
from Phase 2C should remain stable unless real communication evidence exposes
a missing semantic distinction. Production capability bits are enabled only
after end-to-end multi-NPU correctness and ordering tests pass.
