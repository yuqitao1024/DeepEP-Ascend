# Ascend Dispatch and Direct Combine SIMT Parallelization

**Status:** Implemented; representative 8P correctness passed, final benchmark
profiles pending post-push execution

## Purpose

Remove the single-thread bottleneck from the Ascend EPv2 direct dispatch and
combine paths so the canonical eight-rank benchmark can process its production
payload size. The original kernels launched a 512-thread SIMT VF but returned
immediately for every thread except thread zero on the main data paths. That
one thread performed route construction, record validation, expert scans,
destination assignment, padding, BF16 or FP8 copies, combine staging, and
hidden reduction.

The first canonical FP8 case has 4,096 input tokens per rank, hidden width
7,168, top-k 6, 256 experts, and eight ranks. Running all producer and epilogue
work on one thread does not complete before the existing five-second event
wait. A focused run with correctness checks disabled fails in the same place,
so Python reference generation and result checking are not the cause.

## Scope

### Included

- Direct, single-host scale-up dispatch used by the 144 benchmark cases.
- Producer route planning and record construction.
- Epilogue validation, count and prefix generation, compact output, expanded
  output, zero padding, and cached-handle reuse.
- BF16 and FP8 payloads, including FP8 scale-factor copies.
- Synchronous dispatch and the asynchronous copy kernel.
- Direct combine record construction, local staging, contributor validation,
  deterministic receive-slot indexing, hidden reduction, and routing-weight
  output.
- Existing protocol diagnostics and completion-generation ordering.

### Excluded

- Changing the fixed five-second host wait.
- Changing public Python or C++ APIs.
- Changing output shapes, record order, logical-byte accounting, or benchmark
  workloads.
- Parallelizing hybrid or cross-host routing. Those paths retain their current
  implementation and behavior.

The implementation must not hide a slow kernel by increasing the timeout. The
first success criterion is that the existing canonical case completes under
the existing wait with correct outputs.

## Considered Approaches

### Staged SIMT VFs in the Existing Kernel

Split producer and epilogue work into consecutive `asc_vf_call` stages. Each
stage has a single ownership rule: thread zero for control publication, one
thread per rank or local expert for deterministic scans, or a grid-stride loop
across records for data copies. Consecutive VF calls provide the phase boundary
needed before later stages consume workspace written by earlier stages.

This is the selected approach. It preserves the current launch API and stream
ordering while exposing the expensive data-plane work to the 512 configured
threads.

### Multiple Global Kernels

Separate control, planning, copy, and completion into independent global
kernels. This gives explicit stream-ordered synchronization but expands the
runtime launch API and adds several kernel launches to every dispatch. The
extra interface and launch overhead are unnecessary for a one-block staged
pipeline.

### Parallel Payload Copy Only

Keep the current serial metadata algorithm and parallelize only hidden and
scale-factor bytes. This is smaller, but leaves the serial expert rescans and
quadratic destination and cached-metadata validation in the critical path. It
does not adequately address the canonical workload.

## Architecture

The direct dispatch path remains one global producer kernel and, for async
operation, one global copy kernel. Direct combine also keeps its existing
global launch API. Each global kernel invokes several SIMT VFs in sequence. A
stage reads only state completed by earlier stages and publishes only state
owned by that stage.

Control-plane work stays on thread zero. Rank-partitioned work uses one owner
thread per rank. Expert-partitioned work uses one owner thread per local
expert. Record construction and output copies use all 512 threads with
grid-stride loops over logical records or record lanes.

The hybrid dispatch/combine paths and expanded combine without multiple
reduction continue to call the existing serial VFs. This isolates the
benchmark fix from hybrid forwarding and scale-out protocol behavior.

## Workspace

The dispatch workspace gains explicit temporary regions for:

- per-rank or per-expert error candidates;
- per-rank cached-slot validation bitmaps; and
- per-local-expert cached-destination validation bitmaps.

Bitmap ranges are padded and owned by one rank or expert thread, so they do not
require atomic operations. Their capacity is derived with checked arithmetic
from rank count, shard capacity, local expert count, and dispatch output
capacity. Layout construction rejects overflow. The workspace ABI version and
layout equality tests are updated with the new fields.

Temporary regions are reusable between producer and epilogue stages because
the stages do not overlap. No output tensor is used as scratch storage.

Direct combine additionally reserves one aligned `int32` record-slot entry per
expanded output record. Record builders publish deterministic receive slots
into this region before the parallel reduction stage consumes them.

## Producer Pipeline

### Control Setup

Thread zero clears the operation status and local count, validates topology and
layout, and initializes rank-level control state. If this stage fails, all
later stages observe the nonzero status and return without modifying protocol
or output data.

### Input Validation and Route Planning

Thread zero performs the bounded top-k range validation. One owner thread per
destination rank then scans tokens in source order. It determines whether each
token targets that rank, computes the rank-local shard slot, validates cached
slots, and writes uncached `destination_slots` in deterministic token order.

For cached dispatch, each rank owner uses its private bitmap to verify that
slots are unique, contiguous, and below shard capacity. This preserves every
valid handle produced by the existing backend without retaining the current
quadratic previous-token scan.

### Record Construction

All threads iterate over logical `(token, destination_rank)` records. A thread
skips absent routes; for a present route it owns the entire destination record
and writes hidden payload, FP8 scale factors when present, top-k indices,
weights, source row, master lane, and initialized destination metadata. Record
ownership prevents duplicate writes while distributing the large payload over
the full VF.

### Release

After record construction completes, thread zero publishes local or remote
counts and release controls using the existing transport protocol. No release
is visible before its records are complete.

## Epilogue Pipeline

### Acquire and Rank Prefix

Thread zero waits for source release controls, validates generations and
counts, stores per-source counts in workspace, computes the compact rank
prefix, and validates cached rank prefixes. This work is bounded by world size.

### Record Validation

One owner thread per source rank validates its shard in source-slot order. It
checks all top-k values, local-lane presence, master-lane metadata, source row,
and cached source metadata. Each owner writes only its rank error candidate.
Thread zero reduces candidates in rank order and publishes at most one protocol
failure before any output tensor is mutated.

### Expert Counts and Prefixes

One owner thread per local expert scans received records and counts matching
lanes. Nonlocal expert counts are zero. Thread zero then computes the aligned
prefix array, validates cached count and prefix tensors, and checks output
capacity. This replaces the existing single-thread scan over every expert.

### Expanded Destination Assignment

For uncached expanded dispatch, each local-expert owner scans records in
source-rank, source-slot, and top-k-lane order. It assigns destinations from
that expert's prefix and writes them into source metadata. This preserves the
current deterministic ordering without the current per-lane scan of every
prior record.

For cached expanded dispatch, each local-expert owner validates destination
range and uniqueness with its private bitmap. Valid destination permutations
remain accepted; validation does not require cached slots to appear in prefix
order.

### Padding and Output Copy

All threads clear expanded padding with grid-stride loops. A later stage
iterates over compact records or expanded record lanes and copies hidden data,
scale factors, top-k indices, weights, and source metadata. Each compact row or
expanded destination has exactly one owner.

### Completion

The final VF first checks operation status. Only thread zero then performs the
release store to `dispatch_generation`. This stage runs after all output-copy
VFs have completed, so the generation can never advertise partially written
outputs.

When `copy_outputs` is false, the pipeline stops after validated count and
prefix metadata and does not publish copy completion prematurely. The existing
async copy kernel runs the remaining epilogue stages and publishes generation.

## Direct Combine Pipeline

Thread zero owns producer control and transport publication. Destination-rank
owners plan source records and write private error candidates; a thread-zero
reduction stage stops the pipeline on the first deterministic error. All
threads then construct records and copy local staging rows with grid-stride
ownership.

After transport acquire, all threads clear output indices and validate
contributors. A deterministic token/lane mapping selects each receive-slot
index without atomics. Grid-stride workers reduce hidden rows and write routing
weights, and thread zero publishes completion only after those stages finish.

## Error Handling

Parallel workers never race to write the shared protocol status. They write
private candidate slots. A thread-zero reduction stage selects the lowest
logical owner index, records the corresponding existing
`DispatchProtocolError`, and prevents subsequent mutation stages from running.

Capacity, layout, generation, and transport failures keep their current error
types and stages. The implementation does not add retries or timeout changes.
Cached metadata is completely validated before expanded output tensors are
modified.

## Compatibility Requirements

- Compact rows remain ordered by source rank and source slot.
- Expanded rows remain grouped by global expert with existing alignment.
- Uncached destination slots remain deterministic across repeated runs.
- A cached handle produced by a successful uncached dispatch remains valid and
  produces identical metadata and tensor placement.
- FP8 scale-factor addressing continues to use token and pack strides.
- `dispatch_generation` remains the sole successful completion publication.
- CPU-sync and async-copy paths produce the same tensors and metadata.
- Hybrid routing retains its current code path and protocol behavior.

## Testing

### Host Contract Tests

- Workspace layout size, alignment, offsets, overflow rejection, ABI equality,
  and direct-versus-hybrid behavior.
- Pure helper tests for bitmap sizing, owner partitioning, compact-index
  mapping, and deterministic destination assignment.
- Source-contract checks that direct producer and epilogue data stages use
  `threadIdx.x` grid-stride ownership and that completion remains a final
  thread-zero stage.
- Existing production dispatch, runtime contract, and automation tests.

### Device Build Tests

- Compile all modified ASC kernels with the benchmark NPU environment.
- Keep hybrid and standalone epilogue launch entry points build-compatible.

### NPU Validation

NPU work is submitted only through TaskQueue, one task at a time. Validation
starts with the first canonical FP8 case on eight ranks, first with correctness
enabled. Representative cases then cover:

| Dimension | Required coverage |
| --- | --- |
| Payload | BF16 and FP8 |
| Layout | Compact and expanded |
| Handle | Uncached and cached reuse |
| Padding | Alignment 1 and 128 |
| Execution | Synchronous and async copy/event |

After representative correctness passes, run the 144-case smoke profile. A
full canonical performance run remains a separate benchmark step after the
single-case regression is proven.

### Verified Results

All tasks used CANN 9.2.0, the pinned `hcomm-deepep-current` package, the
qualified Python 3.10 environment, and serialized TaskQueue submission.

| Evidence | Task | Result |
| --- | --- | --- |
| ASC/C++ extension build at `81da5ee` | `task_20260821_035026_270464421601` | Passed, `exit=0` |
| FP8 alignment 128, sync, handle copy | `task_20260821_035216_270878415029` | 1 case passed, `exit=0` |
| BF16 alignment 1, sync, handle copy | `task_20260821_035424_27140337624` | 1 case passed, `exit=0` |
| BF16 alignment 1, async, previous event, comm-stream allocation | `task_20260821_035649_272065721223` | 1 case passed, `exit=0` |
| FP8 alignment 128, async, previous event, comm-stream allocation | `task_20260821_035912_272675920526` | 1 case passed, `exit=0` |

Each device case used the canonical `4096 x 7168`, top-k 6, 256-expert,
eight-rank workload with correctness enabled, one warmup, and one measured
iteration. These runs establish functionality only; their timings are not the
formal performance sample.

The first post-implementation run exposed a separate capacity-contract bug:
the buffer-size hint used the configured multiple-reduction layout while
dispatch tiling omitted that mode and requested the larger single-reduction
window. Dispatch tiling now carries the buffer's internal reduction-layout
flag, runtime validation accepts it, and dispatch handle mode flags continue
to exclude it. A production dispatch probe covers uncached and cached expanded
launches. Host verification after this fix passed `102` tests plus `48`
subtests in the core suite and `134` benchmark contract/automation tests.

The 144-case smoke profile and full canonical performance profile remain
pending and must run only after the branch is rebased and pushed.

## Success Criteria

- No direct benchmark dispatch data path is restricted to thread zero.
- The first eight-rank canonical FP8 case completes under the unchanged
  five-second wait and passes correctness checks.
- Representative BF16, FP8, compact, expanded, cached, and async cases pass.
- The 144-case smoke run completes with 720 operation records after push.
- Existing host contract tests and ASC compilation pass.
- No timeout increase, public API change, or benchmark workload reduction is
  used to obtain success.
