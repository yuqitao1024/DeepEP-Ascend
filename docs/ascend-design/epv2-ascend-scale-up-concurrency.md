# EPv2 Ascend Scale-Up Topology And Concurrency

## Status

This document defines Phase 3B of the Ascend 950 EPv2 port. It extends the
validated synchronous BF16 barrier, dispatch, and combine implementation from
a fixed two-rank shape to a rank-parameterized single-host scale-up design. It
also defines safe ownership and concurrency for multiple `ElasticBuffer`
instances.

The implementation is rank-parameterized and must not contain a production
`world_size == 2` gate or fixed two-entry rank storage. The current acceptance
environment provides two NPUs, so this phase records full runtime acceptance
for two ranks only. Four-rank and eight-rank acceptance use the same code path
and will be added when those environments are available.

## Scope

Phase 3B includes:

- rank-parameterized single-host, scale-up-only topology validation;
- dynamic symmetric-window control and shard layouts;
- dynamic per-rank device scratch layouts for barrier, dispatch, and combine;
- rank-parameterized public buffer sizing and production runtime admission;
- one synchronous operation at a time within each `ElasticBuffer`;
- concurrent operation of independent `ElasticBuffer` instances;
- buffer-scoped queue, workspace, window, generation, diagnostic, poison, and
  destruction state; and
- host/model coverage beyond two ranks plus full two-NPU runtime regression.

Phase 3B does not include:

- scale-out, RoCE, or hybrid topology;
- mapped CPU communication memory;
- asynchronous events, communication streams, or compute overlap;
- multiple simultaneous operations within one `ElasticBuffer`;
- FP8 runtime support;
- GPU validation; or
- legacy DeepEP V1.

## Decisions

### Rank cardinality is a runtime parameter

A supported Phase 3B topology is a pure scale-up topology:

```text
world_size >= 2
0 <= world_rank < world_size
scale_up_size == world_size
scale_up_rank == world_rank
scale_out_size == 1
scale_out_rank == 0
```

There is no production allowlist such as `{2, 4, 8}`. Integer range, layout
overflow, command-capacity, and backend communicator checks bound the actual
cardinality. Two, four, and eight ranks are the selected acceptance sizes, not
separate implementation modes.

HCOMM communicator rank/size remains the source of truth. Initialization must
reject a mismatch between the requested topology and the communicator rather
than silently resizing either one.

### Persistent rank state uses dynamic layout regions

`SymmetricControlHeader` remains a fixed scalar header. Rank-indexed barrier
generation and completion records move to layout-described regions:

```text
control header
barrier generation slots [world_size]
barrier completion slots [world_size]
dispatch control slots [world_size]
```

`SymmetricWindowLayout` carries offset, byte-count, and entry-count metadata
for every dynamic region. Device code computes a checked address from the
region base and `world_rank`; it never indexes a fixed `[2]` or `[8]` persistent
array. The symmetric-window ABI and core tiling ABI are bumped together.

Dispatch receive/staging shards and combine receive/staging/control shards
remain ordered by ascending rank and sized from `world_size`. This preserves
deterministic source and contributor order for every rank cardinality.

### Device-local rank state uses dynamic workspace

SIMT variable-length stack arrays are not assumed to be supported. All
temporary arrays whose size depends on rank count are placed in the
buffer-owned device workspace and described by tiling offsets and sizes.

The workspace builder uses checked arithmetic and reserves operation-specific
rank scratch for the maximum simultaneously live arrays. Dispatch and combine
may reuse rank scratch across producer and epilogue phases because the kernel
schedule is sequential. Typed views must remain within the declared scratch
region and use ascending rank order.

No fixed maximum rank constant is used as a substitute for layout sizing. The
host must reject a topology before launch when its required transport command
count or any persistent/workspace layout exceeds the representable or
allocated capacity.

### One operation per buffer, multiple buffers concurrently

One `ElasticBuffer` owns one transport command queue, diagnostic record,
workspace, symmetric window, HCOMM team, and generation namespace. These
resources cannot safely serve simultaneous barrier, dispatch, or combine
operations.

Each buffer therefore has one operation coordinator shared by all three
operation types. Its state machine is:

```text
Idle -> Reserved(operation) -> Active(operation, generation) -> Idle
                                                        \----> Poisoned
Idle/Poisoned --------------------------> Destroyed
```

The coordinator mutex is the linearization boundary. An operation becomes
reserved while holding it before any queue, workspace, window, or sequence
state is touched. Local validation and resource preparation run while the
reservation pins buffer lifetime. Activation publishes the operation kind and
nonzero generation only after preflight succeeds. Destruction claims the
buffer while holding the same mutex before releasing resources. Therefore
reservation and destruction cannot both observe `Idle` and proceed.

The detailed rules are:

- reservation rejects a second operation deterministically before launch; if
  the first reservation activates, the rejection poisons the local buffer when
  that active operation retires;
- the lease covers launch, stream synchronization, diagnostic validation, and
  completion validation;
- a validation error while reserved cancels back to `Idle`, consumes no
  generation, and does not poison the buffer;
- a launch, synchronization, transport, protocol, diagnostic, or completion
  failure after admission poisons the entire buffer;
- a poisoned buffer rejects every later barrier, dispatch, and combine call;
- destroying a reserved or active buffer is rejected deterministically without
  changing the operation state; the caller may retry after it retires, and
  destruction never frees resources beneath active work;
- successful destruction is idempotent; and
- failure or poisoning of one buffer does not affect another buffer.

Independent buffers use independent coordinators and retain their existing
per-instance runtime/transport ownership, so they may execute concurrently.
The process-global dispatch-family counter remains only a handle provenance
allocator and is not an operation lock.

The C++ runtime ownership model permits independent buffers to run
concurrently. Existing Python bindings retain the GIL and therefore serialize
Python-thread entry in this phase; releasing the GIL around asynchronous or
overlapped calls belongs to the Phase 3E stream/event work.

Collective order remains a caller contract: every rank must invoke the same
operation generations in the same order. A local validation or concurrent-entry
error can prevent that rank from launching while peers have already launched.
Those peers must exit through the existing finite device timeout and poison
their local buffers. The runtime cannot directly mutate an `ElasticBuffer`
object in another process. Consequently, any operation error reported by any
rank invalidates the logical buffer on every participating rank; the caller
must coordinate destruction and recreation of the complete rank set before
issuing another operation. Tests must verify bounded peer timeout and complete
rank-set retirement rather than claiming automatic remote host-state mutation.

True same-buffer multi-flight execution requires per-operation queue,
diagnostic, workspace, window slots, events, and stream lifetime. It belongs
to Phase 3E rather than this synchronous phase.

## Public And Runtime Contract

The Python and C++ EPv2 signatures remain unchanged. The group rank and size
flow into `ElasticBuffer`, transport configuration, tiling, and layout without
replacement by a constant.

`calculate_elastic_buffer_size` currently receives a communicator handle but
not an explicit rank count. Its Ascend implementation must query the
communicator topology through the backend contract, or the shared Python layer
must pass the group size through an existing compatible boundary. It must not
continue assuming two ranks. Buffer construction independently recomputes and
validates the required layout against the supplied allocation.

Production runtime initialization accepts any valid pure scale-up topology
that passes communicator matching, checked allocation sizing, and transport
command-capacity validation. It continues to reject hybrid mode, CPU buffers,
extra participant channels, scale-out topology, and unsupported modes.

## Failure And Diagnostic Contract

Every admitted operation has a nonzero, buffer-local generation. Rank-indexed
control records are accepted only when their generation matches the current
operation. Payload and count records from older generations are ignored or
reported as stale; they are never consumed as current data.

Failures after admission poison the owning local buffer and invalidate the
logical buffer across the participating rank set as described above. Error text and device
diagnostics preserve operation, local rank, peer when known, opcode, channel,
generation, backend code, and protocol error. Concurrency rejection and
destroy-while-busy rejection occur before launch. Concurrent operation entry
poisons after the first lease activates and retires; if the first lease cancels
during preflight, no generation was published and the buffer returns to
`Idle`. Destroy-while-busy never poisons.

## Determinism

Dispatch receive shards are consumed in ascending source-rank order. Within a
source rank, token and top-k lane order remains stable. Combine contributors
are reduced in ascending contributor-rank order and then stable record/lane
order. Empty ranks contribute zero records but retain their rank position in
control and prefix metadata.

Rank parameterization must preserve cached-handle provenance and exact layout
validation. A handle produced by one buffer is rejected by every other buffer,
including buffers with the same topology and shape.

## Implementation Slices

1. Add RED host probes for dynamic 2/4/8-rank layout, topology, overflow,
   deterministic shard ordering, and absence of two-entry rank storage.
2. Introduce dynamic barrier control regions and dynamic rank workspace,
   update ABI equality/validation, and remove fixed-rank kernel arrays.
3. Parameterize public sizing, buffer validation, operator capacities,
   production runtime admission, and internal launch validation.
4. Add a buffer operation coordinator and concurrency/lifecycle probes.
5. Run local platform and host-model coverage, build all Ascend kernels, then
   run the complete two-NPU barrier/dispatch/combine/lifecycle matrix.

## Acceptance

### Required for the two-rank implementation closeout

- host layout and tiling probes pass for 2, 4, and 8 ranks, including
  asymmetric counts, empty ranks, near-capacity sizing, overflow, and malformed
  topology;
- the production kernel compile probe contains no fixed two-rank rank storage
  or topology gate;
- fake-runtime tests prove queue/window/workspace/diagnostic isolation between
  two live buffers and safe interleaving, failure isolation, and destruction;
- same-buffer concurrent entry and destroy-while-busy fail deterministically,
  obey the coordinator linearization rules, and do not race resource teardown;
- local platform-isolation and Python API tests pass without CUDA, NCCL, or
  NVSHMEM dependencies;
- the Ascend production extension and all selected kernels build on NPU8P;
- two-NPU public BF16 barrier, dispatch, and combine reference matrices pass,
  including asymmetric routing, empty input, cached handles, expanded modes,
  near-capacity cases, repeated generations, and cross-buffer handle rejection;
  and
- two independent buffers pass a two-NPU interleaved operation matrix.

### Full Phase 3B topology qualification

Four-NPU and eight-NPU public BF16 barrier, dispatch, combine, repeated
generation, failure, and teardown matrices remain required for full Phase 3B
topology qualification. They are deferred until suitable environments are
available. The implementation may be integrated after the two-rank closeout,
but Phase 3B topology qualification remains explicitly open. This acceptance
gap is not a production rank gate and is not evidence that a separate code
path is needed.
