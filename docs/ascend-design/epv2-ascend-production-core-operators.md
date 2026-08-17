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

Phases 2E, 2F, and 2G are implemented and validated on NPU8P for the selected
two-rank synchronous BF16 scope.

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

## Remaining Gaps

Phases 2E, 2F, and 2G close the communicator, production extension, two-rank
topology, buffer ownership, staged service, synchronous barrier, teardown,
BF16 dispatch, and BF16 combine gaps. Async events, public communication
streams, hybrid routing, pipeline, Engram, AGRS, FP8 runtime, and scale-out
remain interface-only or unsupported as listed in the non-goals.

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

### HCOMM memory and channel generations

Repeated team/window lifecycles require an HCOMM source fix archived at
`third_party/patches/hcomm/hcomm-team-window-deregister.patch`. The patch is
based on HCOMM commit `8c5d5ad081e763f981c237d8dfdb15faea292d6e` and has
SHA256 `7394982ec1c5432b3fe15898974e64441ba5a24a2bf5c110e49bb758174a9329`.

Team and window destruction unbind their `CommMems` tags, while endpoint
registrations and old physical channels remain domain-owned until `MyRank`
teardown. Channel lookup separates the logical index from the physical index
and keys physical generations by `(engine, CommMems version, logical index)`.
The matching socket tag also includes the memory version. This is required
because AIV URMA channel `UpdateMemInfo` is not implemented and otherwise
silently retains the first team's synchronization buffer.

Window deregistration uses a retryable two-phase commit. HCOMM teardown is
recorded before unregistering the corresponding `CommMems`; owner and alias
records are removed only after both operations succeed. A retry skips an
already completed HCOMM step, while a window in teardown is unavailable for
reuse or pending remote-memory exchange.

Team destruction propagates any window-deregistration failure and preserves
the team/window state for retry. Communicator destruction first closes a
shared lifecycle gate, waits for in-flight team operations, and only then
unloads binaries and clears teams. World-team teardown immediately makes its
subteams unavailable, and handles already removed by communicator cleanup are
idempotent no-ops instead of triggering a second raw HCOMM destruction.

The validated isolated library was built with the CANN GCC 7.3 toolchain and
has SHA256 `afb65298169b7810269322a32576429bcd67798a3336718a2642d2fb97332e77`.
No HCOMM binary is committed to this repository.

The added lifecycle mutex and state change the concrete C++ `CollComm` layout.
The patched HCOMM library and all C++ consumers must be rebuilt together; the
final validation performs that coordinated rebuild.

DeepEP teardown preserves dependency order on failure. Once teardown starts,
the runtime rejects further communication. A failed window deregistration
retains the HCOMM window, team, and owning NPU allocation; a failed team
destruction retains the team and owning allocation. A later `destroy()` call
retries from that state, and the allocation is released only after HCOMM no
longer owns either handle.

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
put-value64, signal, system ordering, device barrier, and scale-up team.

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

All eight items passed on NPU8P devices 6 and 7. The final evidence includes
three explicit create/destroy cycles, descriptor proof that a later channel
uses the new team synchronization memory, 100 public barrier generations,
Phase 2D `barrier-repeat` and `teardown`, and injected rank-qualified timeout
diagnostics. Task identifiers and artifact hashes are recorded in
`third_party/patches/hcomm/README.md`.

The final clean production task used `DEEP_EP_ASCEND_TESTING=0`, passed 53
Ascend tests, 15 platform tests, 10 build tests, and a fresh 100-generation
two-rank run. The same task also passed 100 injected-diagnostic generations.
Its extension had no CUDA, NCCL, or NVSHMEM runtime dependency. The final task
is `task_20260816_173926_27925920094`; artifact hashes and the HCOMM regression
record are in `third_party/patches/hcomm/README.md`.

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

All eight items passed on NPU8P devices 6 and 7 with system CANN 9.2.0 and the
weekly HCOMM package from `hcomm-weekly-20260814` overlaid after the system
environment. The focused build task
`task_20260816_234947_110751921358` clean-built the testing extension and
passed 61 Ascend tests, 15 platform tests, and 11 build tests. The final
serialized acceptance task `task_20260816_235509_112303119438` repeated those
test counts, clean-built both testing and production extensions, passed all 12
two-rank reference cases, and printed `PHASE2F_ACCEPTANCE=PASS`. The matrix
included cached-handle reuse, 100 sequential generations per rank, bounded
invalid-expert diagnostics, and the Phase 2G combine gate.

The testing extension SHA256 is
`b3f88d1f6353f604555c0d5f47168b2cd7b4f5cad357bda65ad62f4fb1916c02`; the
production extension SHA256 is
`ab1843adf4b6f1c8dd8a81c7172ea19775bc099161b36f7901735b4fcbfef755`. The
production dependency audit found no CUDA, NCCL, or NVSHMEM dependency. The
source archive used for final acceptance has SHA256
`94e894fc728c68051ad14329362a2b6ca2ab1b80cf50c8c452717dda74aacbbb`.

A final-review correction made aligned expanded output capacity a checked
`CoreTiling` ABI field shared by host allocation and AICore prefix/destination
bounds. Final serialized task `task_20260817_004736_25120404459` on NPU8P
devices 6 and 7 clean-built testing and production `dav-3510` extensions from
source commit `5855b606c0da0e4f225d18e0729782beb7087dda`, passed 61 Ascend tests,
15 platform tests, 11 build tests, and all 14 two-rank reference cases, and
printed `PHASE2F_ACCEPTANCE=PASS`. The added `aligned-near-capacity` and
`cached-aligned-near-capacity` cases exercise capacity 4, top-k 1, two local
experts, alignment 4, and aligned output beyond the eight-row raw lane count.

The final-review testing extension SHA256 is
`a2c77dcfc36f9f297ba59551ab4f49842584f87e18beac7764d14898fca765a0`;
the production extension SHA256 is
`edddedf364abfe50f85c932d65495b0dd298ad084805fa9bb9628dc5174dcf07`;
and the source archive SHA256 is
`2cb18d1a58efbdd5a7af858643851023e2e901832ca33370d00f40553b66afb0`.
The production dependency audit again found no CUDA, NCCL, or NVSHMEM
dependency. The same task passed cached handle reuse, 100 sequential
generations per rank, bounded invalid-expert diagnostics, and the Phase 2G
combine gate.

The accepted descriptor protocol uses fixed source-owned inbox shards plus a
generation-tagged count, signal, and barrier. Cached dispatch carries
`DispatchHandleDescriptor` ABI version 1, including the generation family,
topology, tensor shapes, alignment, capacity, and mode flags. Phase 2F records
but does not consume that descriptor in combine. The accepted Phase 2G path
now validates and consumes it before every combine launch.

## Phase 2G: BF16 Combine

### Combine protocol

Combine consumes only metadata produced by a compatible Phase 2F dispatch.
Each expert rank uses source rank and source token metadata to return expert
outputs to the originating rank.

The Ascend public path decodes and validates the Phase 2F
`DispatchHandleDescriptor` before allocating outputs or publishing transport
commands. The descriptor must belong to the current buffer family and match
the two-rank topology, original token count, hidden size, global expert count,
top-k width, expert alignment, capacity, and expanded mode. Source metadata,
rank prefixes, expanded slots, and all encoded source ranks, token indices,
and lanes are validated against those descriptor bounds. A handle from a
different buffer, topology, shape, or mode is rejected before launch.
Expanded handles may retain the dispatch-only zero-padding mode when their
complete family attestation matches; zero padding is not forwarded into the
combine tiling.

The symmetric combine region contains two kinds of fixed shards and a control
array:

- one receive shard per contributor rank, owned by the origin rank;
- one registered staging shard per destination origin rank, owned by the
  contributor; and
- one `CombineControlSlot {generation, count}` per contributor rank.

A packed combine record contains a BF16 hidden payload, optional float32
weights for all top-k lanes, and a versioned header containing the origin
token, contributor rank, master lane, and contribution lane. Normal layout
uses at most one record per origin token and contributor rank. Expanded layout
with `allow_multiple_reduction=true` reduces valid local expert slots for one
origin token in ascending top-k lane order using float32 and publishes one
record. Expanded layout with `allow_multiple_reduction=false` retains one
record per valid lane; its shard capacity is therefore
`num_max_tokens_per_rank * num_topk`. Alignment padding is never published or
read. Expanded top-k weights remain unsupported when multiple reduction is
disabled, matching the upstream CUDA contract.

For each original token, contributions from experts on the same contributor
rank are either retained separately or reduced locally according to
`allow_multiple_reduction`. A contributor writes its record into its local
staging shard indexed by destination origin rank, then publishes one
contiguous payload put into the origin's receive shard indexed by contributor
rank. The payload is followed by
a generation-tagged count put, a signal, and the finite-timeout scale-up
barrier. Self-bound records use the same receive-shard and control semantics
without a remote put. The AICore service executes the batch before the origin
reads any receive record.

After remote completion, the origin scans contributor shards, validates the
generation, record count, contributor identity, token, and lane metadata, then
accumulates BF16 contributions in contributor-rank and lane order using
float32. Optional top-k weights are routing values returned to their original
top-k lanes; they are not multiplied into the expert output. Each optional
BF16 bias is applied exactly once after all contributor payloads have been
accumulated, and the result is cast once to one BF16 output token.

The layout prevents two remote ranks from writing the same record. The final
reduction is local to the origin rank, so no remote atomic floating-point
reduction is required. Combine publishes generation-tagged counts with
put-value64 and does not require FAA for control or payload publication.

Combine owns a monotonic generation sequence independent of dispatch and
barrier. Every attempted device launch consumes one generation; an incomplete
launch, synchronization, copy, or diagnostic boundary poisons the sequence so
later calls cannot accept stale records. All retryable tensor, handle, mode,
capacity, and device-ownership validation happens before the attempt begins.
Diagnostics retain operation, rank, command index, opcode, peer, channel, and
generation.

The public Python and C++ signatures remain unchanged. Ascend accepts exactly
one AICore block and zero QPs, synchronizes before returning, returns no native
event, and rejects previous events, asynchronous compute-stream overlap,
communication-stream allocation, hybrid mode, scale-out, FP8, and every other
deferred mode before launch. CUDA defaults and behavior remain unchanged.

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

The Phase 2G reference accumulates in the deterministic order above. Combined
BF16 payloads use `rtol=1/128` and `atol=1/128`; returned float32 top-k weights
must match exactly, including zeroes for `-1` lanes. Both reduction modes must
select the same contributors and satisfy the same reference tolerance even
when their BF16 staging changes rounding.

Acceptance includes normal and expanded inputs, padded extents strictly larger
than the raw lane bound in both reduction modes, expanded weights, odd hidden
widths with and without weights, cached dispatch handles with changed expert
outputs, zero/one/two biases, duplicate same-rank experts, `-1` lanes, empty
and asymmetric ranks, malformed and cross-buffer handles, bounded peer-failure
diagnostics, repeated construct/destroy, and at least 100 dispatch/combine
generations. The terminal NPU8P task clean-builds testing and production
extensions, audits out CUDA/NCCL/NVSHMEM dependencies, and prints
`PHASE2G_ACCEPTANCE=PASS` only after every case passes on both ranks.

Final serialized task `task_20260817_111959_419358541` on NPU8P devices 6 and
7 clean-built testing and production `dav-3510` extensions from source commit
`eb46612e24bbedde1d9b60ca1931519b6117a351`. It passed 75 Ascend tests, 15
platform tests, 11 build tests, the production dependency audit, and all 23
public two-rank cases for 46 rank-case executions. It also passed 100
dispatch/combine generations per rank, exact float32 routing weights, bounded
invalid-peer diagnostics, sequence poisoning, and repeated teardown, then
printed `PHASE2G_ACCEPTANCE=PASS` and exited zero.

The accepted source archive SHA256 is
`52efaa4fed3fc8a7e84a86f444ca36111e49c323a613484dd3cd246b8205b6c7`;
the testing extension SHA256 is
`2ff7e80b4e6ab5ee14280bedb95f5a9e43fb4cb37352c80c2f14ffd2c63d25ed`;
and the production extension SHA256 is
`42e430fd2aec0174b1ba7fdbde9059d617da4790b535a020f905baa56f76dd55`.
The production audit found no CUDA, NCCL, or NVSHMEM dependency.

The task used patched HCOMM library SHA256
`afb65298169b7810269322a32576429bcd67798a3336718a2642d2fb97332e77`,
weekly HCOMM library SHA256
`5be221e4a6c0e04af029f77f5b106c8f385eefcf8a2e5b0e9e5dcc8bf3c82118`,
weekly-compatible `hccl_team.h` SHA256
`b843960291727653cebd1f4453cb71c4ab3255743c5216e2ef17baaaf1be312b`,
and archived lifecycle patch SHA256
`7394982ec1c5432b3fe15898974e64441ba5a24a2bf5c110e49bb758174a9329`.
The system CANN `set_env.sh` SHA256 was
`ff907f2bfb0e94346489498bce054e79d591a2a9c6801544d69e88f5cf22116b`.
The accepted protocol uses contributor-owned staging and receive shards,
generation-tagged `{generation, count}` control slots, and an independent
monotonic combine generation. Combined BF16 values use `rtol=1/128` and
`atol=1/128`; routing weights match exactly. Async events, mapped-CPU
synchronization, hybrid, pipeline, Engram, AGRS, FP8, and scale-out execution
remain deferred.

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

Follow-on scope, dependency order, permanent exclusions, and acceptance policy
are defined in `epv2-ascend-roadmap.md`.
