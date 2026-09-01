# EPv2 Ascend Source Pipeline Progress And Multi-Chunk Design

**Status:** Design and diagnosis baseline. The source-token pipeline is not
accepted for production until the multi-chunk correctness gates below pass.

**Scope:** Normal, direct, uncached Dispatch on Ascend 950/NPU8P. This document
addresses two related failures in the P7 source-token pipeline:

1. producer block progress, including the previously observed producer-35
   progress visibility failure; and
2. the structural stall when the source pipeline has more work than the
   two-slot ring can hold, especially `chunk_count > 2`.

The document is intentionally limited to Dispatch. Combine, Cached Dispatch,
Expanded Dispatch, and scale-out transport remain regression gates, not part of
this repair.

## 1. Problem Statement

The representative workload is:

```text
world size       2 for focused diagnosis, 8 for final acceptance
tokens/rank      8192
hidden           7168
top-k            8
experts         256
alignment       128
data blocks      72 public blocks
source pipeline  DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES=1024
```

For 8192 tokens, a smaller `CHUNK_TILES` produces multiple source chunks. The
normal path completes when the source pipeline is disabled or reduced to one
chunk, but the source-token path has shown either no progress until timeout or
an invalid prefix after an experimental event barrier.

The observed symptoms are different manifestations of one protocol gap:

```text
producer VF publishes progress
          |
          v
AICore manager must observe progress and publish slot completion
          |
          v
producer VF may reuse the ring slot for chunk n + 2
```

If the VF blocks waiting for the manager while the manager is the caller of
that VF, the manager cannot run. When chunk `n + 2` reuses slot `n`, this is a
direct self-deadlock rather than a slow HCCS operation.

## 2. Current Implementation Facts

### 2.1 State and ring size

The state is defined in `csrc/backends/ascend/elastic/layout.hpp`:

- `kDispatchPipelineSlotCount == 2`;
- each `DispatchPipelineSlot` is 128 bytes and cache-line aligned;
- each source chunk maps to `chunk_index % 2`;
- a slot is reusable for chunk `n` only when it is empty (initial chunks) or
  `kCompleted` for chunk `n - 2`;
- `scalar_progress[block][chunk]` is a 64-byte aligned record; and
- `hidden_progress[block]` is a separate 64-byte aligned record.

The host-side helper in `kernels.hpp` makes the reuse dependency explicit:

```text
slot(n) = n mod 2

reusable(n) =
    slot(n).state == Empty                         , n < 2
    slot(n).state == Completed and
    slot(n).published_chunk + 2 == n               , n >= 2
```

The two slots are not a limit on the number of chunks. They are a bounded
in-flight window. Any number of chunks is valid only if the owner that marks a
slot `Completed` can run while the producer is waiting to reuse that slot.

### 2.2 Launch and execution order

The source pipeline currently performs these logical stages:

```text
producer stream:
  ProducerControl -> ProducerGroup -> ProducerPrefix

communication stream:
  ProducerReleasePipeline

producer stream:
  ProducerRecordPipeline -> Dispatch epilogue
```

The persistent producer and release paths each use an AICore wrapper and one or
more `asc_vf_call` operations. On this CANN path an `asc_vf_call` blocks its
enclosing AICore function until the VF returns. It is therefore not equivalent
to enqueueing an independent runtime task.

The important scheduling rule is:

```text
one persistent AICore stage -> one persistent VF invocation
VF                         -> pack/publish work, then return
AICore manager             -> observe progress, run service, publish done
```

Calling a blocking VF from a manager and making the VF wait for a value that
only the manager can publish creates a cycle.

### 2.3 Producer block occupancy

The source pipeline reserves one schedulable block for the release service. At
the public 72-block setting, the source producer therefore uses 71 blocks:

```text
public data blocks     = 72
release service block  = 1
producer blocks        = 71
```

This reservation is local to the persistent source pipeline. It does not
change the public 72-block launch shape for ordinary non-persistent Dispatch
or Combine.

## 3. Failure A: Producer Progress Visibility

### 3.1 The old publication pattern

The problematic scalar publication wrote two independent fields in one aligned
record:

```text
progress.completed_chunk = n
fence
progress.generation = g
fence
```

The manager then read both fields and required both to match. Because the two
words share a cache line, a cache-line writeback or observation could expose a
mixed state: the manager saw the new generation with the old chunk, or the new
chunk with the old generation. Producer block 35 was the clearest observed
case, but the protocol made any block vulnerable.

This is a visibility protocol problem, not evidence that block 35 owns a
different address or that HCCS drops one producer's data.

### 3.2 Publication rule

Scalar progress is indexed by source chunk:

```text
scalar_progress[producer_block][source_chunk]
```

Therefore the chunk index is already encoded by the array index. The protocol
must publish one value only:

```text
system fence
store scalar_progress[block][chunk].generation = g
system fence / cache visibility operation required by the target API
```

The manager accepts a scalar contribution when:

\[
P_{b,n}.generation = G
\]

for the current operation generation `G`. It must not require a second
`completed_chunk` word for scalar readiness.

Hidden progress is different because it uses one record per producer block
across chunks. It must publish a pair atomically from the protocol's point of
view:

```text
hidden_progress[block].completed_chunk = n
fence
hidden_progress[block].generation = g
fence
cacheline visibility operation
```

The AICore manager accepts hidden completion only when both values match the
same `(g, n)`. Every block record is independently 64-byte aligned, so block
35 cannot overwrite block 34 or block 36 through cache-line writeback.

### 3.3 Required memory API contract

The implementation must document and test the exact primitive used by each
side:

| Operation | Producer SIMT/VF | AICore manager | Requirement |
| --- | --- | --- | --- |
| scalar generation publish | `store_published` | `load_published` | generation is the only scalar readiness value |
| hidden completion publish | device store + fence | `load_device` after cache visibility handling | pair `(generation, completed_chunk)` must match |
| slot state publish | device store + fence | `load_device` | state transition is release/acquire ordered |
| transport completion | HCOMM/service protocol | service completion read | request cannot be reused early |

`DCCI`, `DataSyncBarrier<DSB_DDR>`, `asc_sync_data_barrier`, and `load_dev`/
`store_device` are not interchangeable by name alone. The target compiler and
runtime contract must be confirmed with a minimal probe before selecting one as
the production visibility primitive. A probe that merely reads a value once is
insufficient; it must cover repeated generations and cache-line neighbors.

## 4. Failure B: Why More Than Two Chunks Stalls

### 4.1 Intended two-slot protocol

For source chunks `0, 1, 2, 3`, the intended ownership is:

```text
chunk       0           1           2           3
slot        0           1           0           1
producer    fill        fill        wait 0      wait 1
release     send 0      send 1      complete 0  complete 1
reuse       -           -           slot 0       slot 1
```

Chunk 2 must not overwrite slot 0 until release has consumed its payload and
published `kCompleted` for chunk 0. Chunk 3 has the corresponding dependency on
chunk 1. This is correct bounded-buffer behavior.

### 4.2 Actual persistent execution cycle

The current persistent design can create this cycle:

```text
Producer VF (inside asc_vf_call)
  1. packs chunk 0 and publishes scalar/hidden progress
  2. starts chunk 1
  3. reaches chunk 2 and waits for slot 0 == Completed
                                  ^
                                  |
Release/AICore manager must run service and publish Completed
                                  ^
                                  |
The manager is the caller blocked inside asc_vf_call
```

No additional HCCS bandwidth or number of AIV blocks can break this cycle.
Adding NOPs can make a polling loop less aggressive, but it cannot provide the
missing scheduling opportunity. An event barrier can alter launch ordering or
cache timing, but it does not remove this device-side dependency cycle.

The same issue repeats for any `chunk_count > 2`; after every two chunks the
producer needs a completion that its blocked caller is responsible for
publishing. This explains why one-chunk runs pass while multi-chunk source
pipeline runs do not establish forward progress.

### 4.3 Distinguishing a true deadlock from a visibility delay

The diagnosis must classify the stop point using monotonic markers, not a
single timeout:

```text
ProducerStarted
  -> ScalarReady(n)
  -> PayloadReady(n)
  -> ReleaseSawPayloadReady(n)
  -> ReleaseBatchTargetPublished(n)
  -> ReleaseBatchConsumed(n)
  -> RequestCompleted(n)
  -> SlotCompleted(n)
```

Interpretation:

| Last marker | Likely class |
| --- | --- |
| no `ScalarReady(n)` | producer progress publication, block count, or scalar visibility |
| `ScalarReady` but no `PayloadReady` | hidden progress publication or AICore copy stage |
| `PayloadReady` but no `ReleaseSawPayloadReady` | release polling or stream scheduling |
| batch target published but not consumed | service execution/SQ-CQ progress |
| request completed but slot not completed | release continuation/state publication |
| chunk 0/1 complete, chunk 2 never starts | two-slot reuse cycle |
| all chunks ready, prefix is invalid | count/prefix input visibility or stale tile offsets |

Each marker must include `{generation, chunk_index, block_count}` and be written
to a cache-line-isolated diagnostic record. Host timeout handling reads the
records after device synchronization; it must not print from a device polling
loop.

## 5. Design To Resolve Both Failures

### 5.1 Separate producer packing from manager progress

The persistent producer stage is changed to a single VF invocation with no
blocking wait for manager-owned state:

```text
Producer VF:
  for n in [0, chunk_count):
    write a distinct slot for source chunk n (no manager-owned wait)
    pack scalar/metadata for source chunk n
    publish scalar_progress[block][n]
    return after all chunk records are published

Producer AICore manager:
  concurrently observe scalar progress
  copy hidden payload for ready chunks
  publish hidden progress and PayloadReady
```

The critical constraint is that the producer VF cannot wait on a value whose
publisher is code after the same `asc_vf_call`. In particular, it must not wait
for `slot(n)` to become `Completed` before writing chunk `n` when the enclosing
AICore manager is the code that publishes `Completed`.

The recommended persistent layout gives every logical source chunk a distinct
slot for the bounded maximum (`kMaximumDispatchPipelineChunks`, currently 8).
The VF can then publish all chunks without a reuse wait; the manager consumes
the published chunks after the VF returns. If a future configuration exceeds
that bound, the launcher uses the host-driven fallback below instead of
entering a ring-wrap wait inside the VF.

### 5.2 Use explicit producer and release continuations

The release path follows the same rule:

```text
Release VF:
  observe PayloadReady(n)
  append payload puts and flush
  publish exact queue target
  return

Release AICore manager:
  execute service for each new queue target
  publish consumed target
  complete request n
  publish SlotCompleted(n)
```

No VF may wait for `release_batch_consumed` while its enclosing AICore is the
only code that can call `service::execute()`.

### 5.3 Make both slot modes generation-safe

For the persistent no-reuse batch, slot `s = n` stores these fields:

```text
slot[s].chunk_index       = n
slot[s].generation        = G
slot[s].state             = Producing | ScalarReady | Ready | InFlight | Completed
slot[s].published_chunk   = n
```

The persistent no-reuse readiness predicate is simply that the slot for the
same chunk is ready and carries the current generation. The producer does not
wait for a prior slot completion before publishing that chunk.

For the two-slot fallback ring, the reuse predicate is:

where `s = n mod 2`:

\[
Reuse(n) \iff state(s)=Completed
\land published\_chunk(s)=n-2
\land generation(s)=G
\]

The generation check prevents a previous operation from satisfying a new
operation's reuse wait. Chunk indices and counters remain 32-bit; operation
generation remains 64-bit.

### 5.4 Choose one of two execution modes explicitly

The runtime must not silently mix the two models:

| Mode | Scheduling model | Use |
| --- | --- | --- |
| Persistent | one producer VF + one release VF; producer writes unique per-chunk slots and returns, managers execute after VF return | target optimized mode after multi-chunk probe |
| Fallback | host enqueues one record/release pair per chunk with stream dependencies | correctness reference and temporary production fallback |

The selector must be visible in the benchmark report. A source pipeline request
that cannot satisfy persistent-runtime preconditions must fall back or fail
explicitly; it must not enter a path known to self-wait.

## 6. Event Barrier Decision

The experimental `aclrtCreateEventWithFlag` / `aclrtRecordEvent` /
`aclrtStreamWaitEvent` change is **not ready to merge**.

It records an event on the communication stream and makes the producer stream
wait for that event. This can prove that the release kernel was enqueued first,
but it does not prove that:

- the release manager has consumed a payload;
- the producer can reuse slot `n mod 2`;
- scalar or hidden progress is cache-visible; or
- the source pipeline has no manager/VF self-wait.

The event-barrier experiment changed one timeout into an invalid prefix in a
clean reproduction. That is useful diagnostic evidence, but it is not a
correctness result. It also adds a host/runtime synchronization edge to the
critical path and conflicts with the event-free persistent design target.

Retention rule:

```text
keep event barrier only in a probe branch;
merge only if a clean build passes all multi-chunk correctness cases,
the marker sequence reaches SlotCompleted for every chunk, and
same-binary performance shows no material regression.
```

Until then, the formal design remains event-free at the host level and uses
workspace generations plus explicit stream ownership for device ordering.

## 7. Implementation Plan

### Phase A: Contract and probe (no performance claim)

Files:

- `csrc/backends/ascend/elastic/layout.hpp`
- `csrc/backends/ascend/elastic/kernels.hpp`
- `tests/ascend/core_operator_contract_probe.cpp`
- `tests/ascend/test_core_operator_contract.py`

Actions:

1. Assert cache-line alignment and standard-layout size for every progress
   record.
2. Test scalar readiness using generation only.
3. Test hidden readiness using the `(generation, completed_chunk)` pair.
4. Test ring reuse for chunk sequence `0,1,2,3,4`.
5. Test stale generation, skipped chunk, duplicate publication, and producer
   block counts 1, 71, and 72.
6. Add a bounded marker reader that reports the last completed protocol stage.

### Phase B: Isolated multi-chunk runtime probe

Use a clean source directory and one two-rank TaskQueue job. Keep the workload
fixed and vary exactly one dimension:

| Probe | `CHUNK_TILES` | Expected purpose |
| --- | ---: | --- |
| one chunk | 2048 or larger | establish the non-wrapping baseline |
| two chunks | 1024 | identify the first slot handoff |
| three chunks | 512 or smaller | force slot 0 reuse after chunk 0 |
| four chunks | 256 or smaller | force both slot reuse paths twice |

For each probe collect the marker sequence, final slot states, scalar/hidden
completion counts, release queue target/consumed values, and prefix buffers.
The probe is diagnostic only and must not be used as a performance result.

### Phase C: Persistent protocol repair

Files:

- `csrc/backends/ascend/elastic/dispatch.asc`
- `csrc/backends/ascend/elastic/layout.hpp`
- `csrc/backends/ascend/elastic/kernels.hpp`

Actions:

1. Keep exactly one `asc_vf_call` per persistent stage.
2. Remove waits inside a VF for manager-owned completion values.
3. Change the persistent producer workspace from a two-slot reuse ring to a
   bounded no-reuse batch with one slot per logical source chunk (up to the
   configured maximum of 8).
4. Move service execution and slot completion into the enclosing AICore
   continuation.
5. Publish scalar generation once per `(block, chunk)`.
6. Store hidden progress as one cache-line-isolated record per
   `(producer_block, chunk)` and require matching generation/chunk pairs.
7. Use `producer_blocks = 71` whenever the release service occupies one
   block; never let a completion loop wait for 72 producer contributions when
   only 71 producer blocks were launched.
8. Preserve timeout, first-error, request ownership, and final barrier rules.

The host-driven fallback remains available while this change is qualified. Its
required schedule is: enqueue chunk `n+2` only after the completion event for
chunk `n` is observed on the producer stream. This is a real scheduling
boundary, not a polling loop inside a blocked VF; the schedule still needs a
clean multi-chunk qualification before it can be called the production
fallback.

### Phase D: Fallback and selector

The host launcher keeps the existing per-chunk schedule as a correctness
fallback. The source-pipeline selector records:

```text
requested mode
selected mode: persistent or fallback
chunk_tiles
chunk_count
producer_blocks
release blocks
```

No automatic fallback is allowed after a device timeout; a timeout is a test
failure and must be visible. Fallback is selected before launch when runtime
capabilities or the probe contract are not satisfied.

## 8. Validation Matrix

### 8.1 Functional gates

All rows must pass for hidden widths 7168 and 7184:

- two-rank normal Dispatch, one through four chunks;
- all-local and all-remote routes;
- duplicate experts mapping to one destination rank;
- inactive `-1` top-k lanes;
- two consecutive operation generations;
- producer blocks 1, 71, and the ordinary 72-block non-persistent path;
- five-operation shared-state regression matrix; and
- no stale prefix, count, generation, or request state.

### 8.2 Structural gates

- exactly one persistent producer VF invocation;
- exactly one persistent release VF invocation;
- no `asc_vf_call` inside a runtime chunk loop;
- no producer wait on manager-owned completion from inside that VF;
- all polling loops include the approved NOP cadence, without relying on NOP
  as a synchronization primitive;
- each producer block contributes exactly once per chunk; and
- the host persistent schedule has no per-chunk ACL event or synchronization.

### 8.3 Performance gates

Only after correctness passes:

1. same-binary baseline with source pipeline disabled;
2. persistent candidate at `CHUNK_TILES=1024`;
3. persistent candidate at the best verified chunk size;
4. 30 warmups and 30 measured samples on the fixed EP8 representative case;
5. report mean, p50, p95, logical bytes, logical bandwidth, per-rank envelope,
   D3/D4/D8 spans, and SQ/CQ high-watermarks.

No candidate is retained based on a single faster sample. Correctness and
forward progress are mandatory before any bandwidth comparison.

## 9. Acceptance Criteria

The repair is accepted only when all of the following are true:

1. producer block 35 and every other producer block publish and are observed
   for every source chunk;
2. two-slot reuse completes for chunks 0 through at least 4;
3. no chunk-count-dependent timeout occurs for the configured maximum;
4. every chunk reaches `SlotCompleted` before its slot is reused;
5. prefix/count validation passes with sentinel-initialized buffers;
6. no event barrier is required for correctness;
7. the five-operation regression matrix remains green; and
8. only then is the candidate compared with the fixed P6/P7 baseline.

The target of this document is therefore first **provable forward progress**,
then performance. A source pipeline that reports high bandwidth but cannot
complete a three- or four-chunk run is rejected.

## 10. Code Map

| Responsibility | Current location |
| --- | --- |
| pipeline state and slot layout | `csrc/backends/ascend/elastic/layout.hpp` |
| ring predicates and block-count policy | `csrc/backends/ascend/elastic/kernels.hpp` |
| producer VF and scalar/hidden publication | `csrc/backends/ascend/elastic/dispatch.asc` |
| release VF and HCOMM service continuation | `csrc/backends/ascend/elastic/dispatch.asc` |
| source chunk plan and host launch | `csrc/backends/ascend/elastic/dispatch.asc` |
| tiling/workspace allocation | `csrc/backends/ascend/elastic/tiling.hpp` |
| host/runtime argument validation | `csrc/backends/ascend/elastic/runtime.cpp` |
| source/protocol contract tests | `tests/ascend/test_core_operator_contract.py` |
| compile-time/runtime probe | `tests/ascend/core_operator_contract_probe.cpp` |
