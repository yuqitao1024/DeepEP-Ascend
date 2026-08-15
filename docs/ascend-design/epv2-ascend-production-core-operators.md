# EPv2 Ascend Production Core Operators

## Status

This document defines the next production-enablement round of the Ascend 950
EPv2 port. The round is split into three sequential phases:

- Phase 2E: production runtime integration and public scale-up barrier;
- Phase 2F: two-rank BF16 dispatch; and
- Phase 2G: two-rank BF16 combine.

Each phase has an independent acceptance boundary. Work does not proceed to
the next phase until the current phase passes its complete local, build, and
two-NPU validation matrix.

The design builds on:

- `epv2-ascend-transport-contract.md`;
- `epv2-ascend-simt-transport-stub.md`;
- `epv2-ascend-stub-core-operators.md`; and
- `epv2-ascend-simt-urma-transport.md`.

Phase 2D proved the required staged transport primitives and enabled the exact
transport capability mask `0x775`. It did not connect those primitives to the
public `ElasticBuffer` runtime or prove the multi-rank operator schedules.

## Goal

Make the existing Python EPv2 barrier, dispatch, and combine interfaces usable
on one host with two Ascend 950 devices while preserving the CUDA
implementation and API behavior.

The first production path is intentionally narrow:

- CANN 9.2.0 and `dav-3510`;
- one host and exactly two participating NPUs;
- one HCCL scale-up team;
- BF16 payloads;
- synchronous completion before returning to Python;
- device-only symmetric communication memory;
- non-hybrid routing; and
- one operation in flight per `ElasticBuffer` instance.

## Non-Goals

This round does not implement:

- cross-host scale-out or RoCE;
- hybrid scale-up/scale-out routing;
- direct remote pointer dereference;
- device `get`;
- asynchronous events or communication/computation overlap;
- public communication-stream access;
- mapped CPU buffers or a CPU communication data path;
- pipeline send/receive;
- Engram;
- AGRS;
- FP8 runtime execution; or
- performance parity with the CUDA implementation.

The corresponding interfaces remain present. Unsupported combinations fail
before output allocation or kernel launch and do not return fabricated data.

## Current Gaps

The transport layer is functional, but the production call path still stops
above it:

1. `deep_ep.platform.get_comm_handle` returns no communicator for Ascend.
2. The Ascend `ElasticBuffer` constructs `StubHostTransport` instead of
   `CannHostTransport`.
3. The production Ascend extension does not build and link the `.asc` core
   kernels, host runtime, CANN transport, ACL, and HCOMM together.
4. The core host runtime accepts only a single-rank topology.
5. The Phase 2C kernels contain placeholder remote calls rather than complete
   rank routing and symmetric-buffer offsets.
6. The kernels enqueue staged commands but do not invoke the Phase 2D AICore
   service at the required phase boundaries.
7. The public dispatch and combine gates still require
   `kDirectPeerPointer`, although the selected Ascend algorithm uses one-sided
   puts and never directly dereferences a remote pointer.
8. Python barrier, dispatch, domain-query, stream, and tensor checks still
   route non-CUDA platforms to unconditional unsupported errors.

This round closes only the gaps needed by the three core operations.

## Selected Approach

Use vertical production slices rather than completing all kernels before any
public integration.

Phase 2E first establishes the real extension, communicator, memory, stream,
transport, kernel-service, and teardown path and proves it through public
barrier. Phase 2F adds the complete dispatch protocol on that path. Phase 2G
adds the reverse combine protocol and reduction.

Two alternatives are rejected:

1. Completing dispatch and combine against another test-only runner would
   delay discovery of Python, PyTorch NPU, stream, allocation, and extension
   lifecycle issues.
2. Implementing `get`, direct peer pointers, async requests, or scale-out
   first would add transport surface that the selected core algorithms do not
   require.

## Architecture

```text
Python ElasticBuffer API
        |
platform adapter: HCCL handle, NPU tensor checks, synchronous stream policy
        |
Ascend ElasticBuffer host runtime
        |-- persistent symmetric communication buffer
        |-- local workspace and output tensors
        |-- CannHostTransport and DeviceTransportContext
        |-- current/communication stream ordering
        |
Ascend AOT operator wrapper
        |
SIMT producer VF
        |
AICore staged transport service
        |
optional continuation VF or stream-ordered epilogue kernel
        |
BF16 output tensors returned through the existing Python interface
```

The CUDA build graph and CUDA classes remain unchanged. Shared Python code
selects a platform adapter; it does not import CUDA objects on Ascend or NPU
objects on CUDA.

## Production Runtime

### HCCL communicator adapter

The Ascend platform adapter obtains the HCCL communicator owned by the supplied
PyTorch `ProcessGroupHCCL`. It uses the current local NPU rank and the backend's
publicly exposed communicator accessor, normalizes a one-element tuple or list
to one integer handle, and rejects a null or ambiguous result.

The adapter is non-owning. The Python process group must outlive the C++
`ElasticBuffer`. Destroying the buffer destroys its DeepEP team, window, and
channels before the process group is destroyed; it never destroys the
PyTorch-owned HCCL communicator.

### Extension build

The Ascend production extension adds ASC language support and compiles:

- `elastic/barrier.asc`;
- `elastic/dispatch.asc`;
- `elastic/combine.asc`;
- `elastic/runtime.cpp`; and
- `transport/cann_transport.cpp`.

It targets `dav-3510`, defines the staged URMA path for production operator
sources, and links the matching ACL and weekly HCOMM libraries. Weekly HCOMM
architecture-specific headers precede system CANN headers. CUDA, NCCL,
NVSHMEM, and CUDA kernel sources remain absent from the Ascend target.

### Buffer ownership

Each Ascend `ElasticBuffer` owns these resources in order:

1. a non-owning HCCL communicator handle;
2. a persistent, aligned NPU communication allocation;
3. one host-created CANN world team;
4. one registered symmetric window over the communication allocation;
5. one AIV UBC_CTP channel per remote peer;
6. the exported `DeviceTransportContext`;
7. local operator workspace; and
8. the stream state required by synchronous launches.

Initialization publishes no partially usable object. Failure and explicit
destroy release resources in exact reverse order while retaining the first
error. Destruction is idempotent. The CANN transport is destroyed before the
HCCL process group can be released.

`num_cpu_buffer_bytes` must be zero. `allow_hybrid_mode` must be false for this
round. The CUDA QP argument is not reused as an Ascend channel count; the
Ascend runtime owns exactly one facade channel and validates that policy
internally.

### Stream and completion policy

The first production implementation is synchronous. It uses the public
Torch NPU stream adapter to order input readiness, Ascend kernel launches,
transport service execution, and output visibility. The call synchronizes its
selected NPU stream before returning to Python.

`async_with_compute_stream`, `allocate_on_comm_stream`, previous events, and
public `get_comm_stream` remain unsupported. The returned optional event is
empty. This is deliberately slower than the future asynchronous path but gives
an unambiguous correctness baseline.

Exact receive counts may be copied from NPU control storage after stream
completion so output tensor views can have exact shapes. This does not enable
the deferred mapped-host CPU communication path.

### Capability gates

Barrier requires the Phase 2D signal, system-ordering, device-barrier, and
scale-up-team capabilities.

Dispatch requires symmetric window, put, put-value64, signal, system ordering,
device barrier, and scale-up team. Combine requires symmetric window, put,
FAA, signal, system ordering, device barrier, and scale-up team.

Neither operation requires `kDirectPeerPointer`. Removing that bit from the
Ascend production gate is allowed only after source and runtime tests prove
that every remote operand remains a logical symmetric address consumed by a
transport operation. CUDA requirements are unchanged.

## Symmetric Window Layout

The persistent communication window is divided into versioned, aligned
regions:

1. operation control and generation records;
2. dispatch counts and per-source dispatch inboxes;
3. combine counts and per-contributor combine inboxes; and
4. reserved growth space required by the selected alignment.

Every destination owns a fixed inbound shard for each source rank. A sender
therefore computes a remote offset from `(region, source_rank, slot)` without a
remote allocation or direct pointer.

For two ranks, each dispatch source shard can hold at least
`num_max_tokens_per_rank` records. Each combine contributor shard can hold at
least one aggregate record for every original token, plus the selected top-k
weight fields. Size arithmetic is checked for overflow and aligned to the
public elastic-buffer alignment.

Each operation increments a 64-bit generation. Count and completion records
carry that generation so a later call cannot consume stale state from a
previous call. Because the API is synchronous and permits only one in-flight
operation, one generation-tagged slot set per operation family is sufficient;
double buffering is deferred.

## AICore Service Boundaries

Every production operator that communicates follows the Phase 2D staged
contract:

```text
service::reset(context, generation)
  -> asc_vf_call<producer>
  -> service::execute(context)
  -> optional asc_vf_call<consumer>
  -> return or launch a stream-ordered epilogue
```

A producer VF never consumes data after enqueueing a flush or barrier in the
same VF. Data that depends on remote completion is consumed by a later VF or a
later kernel launch. Service diagnostics are copied into an operation status
and checked before outputs are returned.

One block, SIMT thread zero, facade channel zero, and one queue producer remain
the Phase 2D launch contract. Relaxing those limits is a later performance
phase.

## Phase 2E: Runtime Integration And Barrier

### Scope

Phase 2E implements:

- the non-owning HCCL communicator adapter;
- the production Ascend extension build and link graph;
- persistent NPU communication-buffer ownership;
- CANN transport creation, window registration, channel acquisition, context
  export, and teardown in `ElasticBuffer`;
- platform-aware NPU tensor and synchronization helpers;
- Ascend buffer-size calculation for the fixed two-rank layout;
- multi-rank host tiling validation for barrier;
- the staged AICore service boundary in the barrier kernel; and
- the public synchronous `ElasticBuffer.barrier` path.

The Python method keeps its existing signature. For Ascend,
`use_comm_stream` selects the supported synchronous NPU stream policy,
`with_cpu_sync` may request an additional host-visible synchronization, and
`sequential` must remain true. Unsupported flag combinations fail explicitly.

### Barrier protocol

The barrier producer enqueues the Phase 2D scale-up device barrier for the
current generation. The enclosing AICore wrapper executes the command batch,
waits for every peer generation with a finite timeout, records diagnostics,
and returns only after acquire ordering is established.

The public call checks the device diagnostic after stream completion. It does
not add a second HCCL collective as the data-plane barrier.

### Phase 2E acceptance

Phase 2E is complete when:

1. a production Ascend extension imports with no CUDA, NCCL, or NVSHMEM
   dependency;
2. an `ElasticBuffer` can create and destroy its CANN resources repeatedly on
   two ranks;
3. all partial-initialization failures clean up without double destruction;
4. public domain queries report `(scale_out_size=1, scale_up_size=2)`;
5. public barrier completes at least 100 repeated generations on two NPUs;
6. writes before the barrier are visible to a continuation after the barrier;
7. timeout and injected device diagnostics become rank-qualified Python
   errors; and
8. dispatch and combine remain gated.

## Phase 2F: BF16 Dispatch

### Initial routing constraints

The global expert count must be positive and evenly divisible by two. Experts
are assigned contiguously to ranks. `topk_idx` is contiguous int64, uses `-1`
for an unused choice, and every other value must lie in the global expert
range. Inputs are contiguous BF16 NPU tensors. Optional top-k weights are
contiguous float32 NPU tensors.

### Dispatch protocol

The producer derives the destination rank of each valid expert. In normal
layout, a source token is sent at most once to each selected rank even when
multiple selected experts reside on that rank. Its record retains all expert
ids, weights, source rank, source token index, and lane metadata.

Records for destination rank `r` are packed into the remote dispatch inbox
shard owned by the current source rank. The source enqueues:

1. payload and metadata puts for each packed record;
2. a generation-tagged record count using put-value64;
3. a signal publication after the payload; and
4. the device barrier that closes the batch.

After the AICore service completes, the receiver scans source shards in source
rank order. It validates generations and counts, computes per-rank and
per-expert prefixes, compacts records into output tensors, and writes the
dispatch handle metadata used by combine.

This fixed-shard protocol is deterministic and requires no direct peer pointer
or remote slot allocator.

### Dispatch feature order

Phase 2F is implemented and validated in this order:

1. normal, non-cached BF16 dispatch without top-k weights;
2. optional top-k weights;
3. expanded expert layout;
4. expert-alignment zero padding; and
5. cached dispatch using a validated prior handle.

Cached dispatch reuses destination slots and source metadata from the handle.
It rejects topology, shape, expert count, alignment, or generation-family
mismatches instead of silently recomputing the route.

### Python result contract

The existing Python tuple shape is preserved. Phase 2F returns:

- exact received BF16 tokens;
- received top-k indices;
- optional received top-k weights;
- exact per-rank and per-expert counts and prefix tensors;
- unaligned expert counts;
- destination slot mapping;
- source metadata sufficient for combine;
- a reusable cached handle; and
- no asynchronous event.

Modes outside the phase scope fail before launch. FP8 inputs remain compile
covered but not executable.

### Phase 2F acceptance

Phase 2F is complete when:

1. public two-rank BF16 dispatch passes against a PyTorch reference;
2. asymmetric token counts and cross-rank expert selections are covered;
3. empty inputs, unused `-1` choices, duplicate destination ranks, and
   multiple valid experts per token are covered;
4. payloads, expert ids, weights, counts, prefixes, slot maps, and source
   metadata all match the reference;
5. normal, expanded, zero-padded, and cached cases pass;
6. at least 100 sequential dispatch generations pass without stale counts or
   queue corruption;
7. invalid input or device diagnostic returns a bounded error instead of a
   hang; and
8. combine remains gated.

## Phase 2G: BF16 Combine

### Combine protocol

Combine consumes only metadata produced by a compatible Phase 2F dispatch.
Each expert rank uses source rank and source token metadata to return expert
outputs to the originating rank.

For each original token, contributions from experts on the same contributor
rank are either retained separately or reduced locally according to
`allow_multiple_reduction`. A contributor writes its record into the origin's
combine inbox shard indexed by contributor rank and original token. It then
publishes a generation-tagged count/signal and enters the service barrier.

After remote completion, the origin scans contributor shards, validates the
generation and metadata, accumulates BF16 contributions in float32, combines
optional top-k weights, applies each optional BF16 bias exactly once, and
writes one BF16 output token.

The layout prevents two remote ranks from writing the same record. The final
reduction is local to the origin rank, so no remote atomic floating-point
reduction is required. Phase 2D FAA remains available for control publication,
not payload reduction.

### Combine feature order

Phase 2G is implemented and validated in this order:

1. normal BF16 combine without weights or bias;
2. expanded dispatch inputs;
3. optional top-k weights;
4. zero, one, and two BF16 biases; and
5. `allow_multiple_reduction` disabled and enabled.

### Phase 2G acceptance

Phase 2G is complete when:

1. dispatch followed by a deterministic synthetic expert transform and
   combine matches a PyTorch reference on both ranks;
2. normal and expanded layouts pass;
3. optional top-k weights and zero, one, and two biases pass;
4. both multiple-reduction modes pass within an explicitly documented BF16
   tolerance;
5. empty and asymmetric routing cases pass;
6. at least 100 dispatch/combine generations pass without stale metadata or
   queue corruption;
7. failures tear down the transport before ProcessGroupHCCL destruction; and
8. the existing public Python dispatch/combine signatures remain unchanged.

## Error Handling

Host validation checks platform, device, dtype, contiguity, dimensionality,
shape relationships, expert partition, supported flags, buffer capacity,
topology, handle compatibility, and transport capabilities before launch.

ACL, CANN, HCOMM, launch, stream, and device-diagnostic failures preserve:

- operation name;
- rank;
- backend error code;
- device diagnostic and command index when present; and
- a concise stable message.

All device waits have finite limits. If one rank rejects an invalid local
argument while another rank has already entered a device operation, the remote
rank must time out with diagnostics rather than wait indefinitely. Cross-rank
preflight error aggregation is deferred with the mapped CPU synchronization
path, so callers remain responsible for invoking collective operations
consistently on all ranks.

## Validation Strategy

### Local tests

Host-only tests cover:

- platform routing without CUDA imports;
- HCCL handle normalization and lifetime;
- build-source and link ownership;
- buffer-layout size, alignment, and overflow;
- capability masks for each phase;
- lifecycle success and every injected failure boundary;
- tiling and argument validation for two ranks;
- exact dispatch and combine layout models; and
- preservation of unsupported deferred interfaces.

### Ascend compile tests

Clean `dav-3510` builds compile the production extension and every AOT kernel
with staged URMA enabled. Production transport sources must not include CANN
internal HCOMM implementation headers. Weekly HCOMM architecture-specific
headers and the matching `libhcomm.so` must be selected together.

### Two-NPU tests

All NPU8P work uses TaskQueue, one task at a time, and only an allowed pair
`0,1` or `6,7`. Each phase runs its complete matrix in one final serialized
task after focused debug tasks finish.

Tests compare public Python outputs with a PyTorch reference and verify both
ranks' diagnostics. They use nontrivial and asymmetric routing so a local-only
implementation cannot pass. GPU runtime validation remains deferred by
project decision.

## Delivery Boundaries

One design document governs the round, but each phase receives its own
implementation plan and completion commit sequence.

- Phase 2E may merge only when its acceptance matrix is green and dispatch and
  combine remain gated.
- Phase 2F may merge only when dispatch is green and combine remains gated.
- Phase 2G may merge only when the full synchronous BF16 round trip is green.

The final Phase 2G state is a correctness baseline, not a performance claim.
Async execution, broader topology, FP8, and other EPv2 modules are planned only
after this baseline is stable.
