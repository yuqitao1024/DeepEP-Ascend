# Ascend EPv2 Performance Optimization

**Status:** P0/P1/P2 implementation complete; formal EP8 performance acceptance pending

## Purpose

Bring the Ascend EPv2 direct dispatch and combine implementation from its
current functionality-oriented baseline to a performance-oriented design. The
first target is the canonical eight-rank benchmark workload: 8,192 input
tokens per rank, hidden width 7,168, top-k 8, 256 experts, and the same 144
cases and workload manifest used by the H800 comparison.

This design preserves benchmark shapes, routing data, operation semantics,
the five-second operation watchdog, deterministic output ordering, and public
APIs. Performance work must not hide slow execution by reducing the workload,
dropping cases, changing logical-byte formulas, or increasing timeouts.

## Current Evidence

The complete eight-rank inventory contains 144 cases and 720 operation
records. Current formal performance values remain pending until H800 and
Ascend both generate reports from the 8K/top-k 8 workload and the same manifest.
No public table or earlier benchmark artifact substitutes for those reports.

An `asys info -r hardware` query on NPU8P reported:

| Resource | Per device | Eight devices |
| --- | ---: | ---: |
| AI Core | 36 | 288 |
| AI Vector | 72 | 576 |

The device is `Ascend 950PR_9599 V100`. The query completed in TaskQueue task
`task_20260821_063351_32625357843`.

The direct Ascend path now separates one-block control stages from data stages.
Data stages accept 1 through 72 `__vector__` blocks with 512 SIMT threads per
block; the canonical maximum-performance setting is 72. Hybrid, scale-out, and
expanded combine without multiple reduction keep their qualified one-block
fallbacks.

## Scope

### Included

- Direct, single-host scale-up dispatch and combine used by the 144 benchmark
  cases.
- BF16 and FP8 dispatch, compact and expanded layouts, cached dispatch,
  multiple reduction, synchronous and asynchronous operation, previous-event
  dependencies, and communication-stream allocation.
- Data-plane use of up to all 72 AI Vector resources per device.
- Top-k lane grouping, routing metadata construction, payload movement,
  combine reduction, and HCOMM command publication.
- Stage-level profiling and repeatable comparison against the current
  canonical baseline.

### Excluded

- Hybrid and scale-out algorithms. They retain their current serial fallback
  until the direct path is qualified.
- Changing benchmark cases, workload sizes, warmup or measurement counts,
  logical-byte formulas, or correctness rules.
- Increasing GPU, CPU, event, or transport timeouts.
- Claiming H800 parity without running the current automation on H800.

## Prioritized Findings

The eight findings are listed in implementation priority. Some share a root
cause, but each has a distinct code change and acceptance signal.

| Priority | Finding | Current behavior | Performance consequence | Acceptance evidence |
| --- | --- | --- | --- | --- |
| P0.1 | Only one AI Vector is used | Each rank launches one `__vector__` block although the device has 72 AI Vector resources | Caps device-level bandwidth and compute parallelism at roughly 1/72 of the available execution domains | Data stages launch 72 blocks; profiler confirms work on more than one AI Vector |
| P0.2 | No `match_any_sync`-equivalent top-k grouping | Equal destination or contributor ranks are found through repeated rank/top-k scans | Prevents lane cooperation and repeats routing tests in dispatch and for every hidden element in combine | A subgroup grouping primitive elects one owner per unique rank and combine no longer scans `world_size * topk` per hidden element |
| P0.3 | Payload movement is scalar | One SIMT thread owns a record and copies hidden bytes or BF16 elements in scalar loops | Produces inefficient global-memory accesses and does not use vector/UB copy facilities | Aligned vector or DataCopy path handles the main body; scalar code is restricted to tails |
| P1.4 | HCOMM publication is serialized on thread zero | One thread loops over peers, issues puts, flushes, publishes controls, acquires releases, and executes the device barrier | Limits peer concurrency and places communication progress behind a single control thread | Stage traces separate submission and wait time; independent peer payload commands are batched or issued concurrently without changing release ordering |
| P1.5 | Expert counting and expanded destination assignment use only 32 threads | One owner per local expert rescans every received record and every top-k lane | Repeats record reads for all local experts and leaves most SIMT threads idle | A record-oriented histogram/prefix/scatter pipeline reads each top-k entry a bounded number of times |
| P1.6 | Combine planning and record validation use only eight threads | One owner per destination or contributor rank serially scans its rows or received records | Leaves 504 of 512 threads idle during metadata-heavy stages | Row- or record-partitioned validation uses the full data grid and reduces deterministic error candidates afterward |
| P2.7 | Dispatch route planning uses only eight threads | One owner per destination rank scans all 8,192 tokens and all eight top-k lanes | Repeats the top-k scan once per rank and serializes slot assignment within each owner | Top-k grouping and a count/prefix/scatter pipeline replace per-rank full scans |
| P2.8 | Dispatch receive validation uses only eight threads | One owner per source rank validates its entire receive shard | Large shards are processed serially per rank | Records are partitioned over the full data grid while deterministic first-error selection is preserved |

## Implementation Status

| Finding | Status | Implementation evidence |
| --- | --- | --- |
| P0.1 | Implemented | Direct control and data stages use separate launch shapes; data launches accept up to 72 blocks |
| P0.2 | Implemented | Ballot and shuffle based top-k subgroup grouping, owner election, and owner broadcast are shared by dispatch and combine |
| P0.3 | Implemented | Dispatch and combine use aligned vector/DataCopy payload paths with scalar tails |
| P1.4 | Implemented | Commit `20fa915` submits all peer payload puts, flushes once, then publishes controls in deterministic peer order |
| P1.5 | Implemented | Commit `c06074e` uses receive-record tiles, per-tile expert histograms, ordered expert/tile prefixes, and deterministic scatter |
| P1.6 | Implemented | Producer planning and receive validation are tiled; build task `task_20260824_031243_78079520953` and two-rank correctness task `task_20260824_031918_79868621126` passed |
| P2.7 | Implemented and revalidated | Commit `b6c5d0d` supplies grouping, prefix, and record stages; current host contracts and the P1.6 Bisheng build revalidated the path |
| P2.8 | Implemented | Commit `c06074e` partitions rank-major receive records and reduces private tile errors in order |

### P1.4: two-pass HCOMM publication

The producer release stage keeps payload-before-control ordering without one
flush per peer. It first walks peers in world-rank order, copies the local
payload directly, and submits every non-empty remote payload put. It then
performs one payload flush. A second peer walk publishes each remote count and
generation, followed by the existing device barrier. Empty peers still publish
their control state, so a receiver cannot confuse an empty generation with a
missing generation.

This change reduces serialization in the HCOMM command stream. The control
stage remains one block because publication order is part of the protocol; it
is not a candidate for unordered cross-block atomics.

### P1.5 and P2.8: dispatch receive metadata

Dispatch validation maps each logical record to `(source_rank, source_slot)`
in rank-major order. Data-stage tiles validate their own bounded records and
write one private error candidate. The following control stage scans candidates
in logical order and publishes the first error. No data block writes the shared
status word.

Expanded dispatch reuses those tiles for expert processing:

1. Each tile scans its records once and writes a histogram for every local
   expert.
2. A control stage scans experts, then tiles, producing aligned expert prefixes
   and converting histogram counts to exclusive tile offsets.
3. Each tile rescans only its own records and assigns
   `expert_prefix + tile_prefix + local_occurrence`.

The order remains source rank, source slot, then top-k lane. `-1` lanes do not
enter the histogram. Cached mode validates the existing destination slot and
keeps the bitmap fallback instead of replacing its public handle format.

### P1.6: combine producer planning

The producer is four stream-ordered stages:

1. `ProducerControl` validates the complete `prefix_per_rank` array once and
   stores each rank's row begin. Empty rank ranges are valid; decreasing,
   negative, out-of-range, or incomplete prefixes fail before data work.
2. `ProducerPlan` divides source rows into fixed 128-row tiles. A tile validates
   row metadata, records a rank-local occurrence for every owned row, writes a
   tile-by-rank count, and emits at most one error candidate.
3. `ProducerPlanPrefix` scans tile errors in row order, converts tile counts to
   exclusive rank-local prefixes, and checks receive and staging byte capacity.
4. `ProducerRecord` computes the final record slot as
   `tile_rank_prefix + row_local_occurrence` and writes the record.

Producer workspace is sized from the maximum legal source extent, not the
final output-token count:

```text
producer_capacity = num_max_tokens_per_rank * world_size
producer_tile_count = ceil(producer_capacity / 128)
```

For the 8,192-token EP8 shape this is 65,536 source rows and 512 tiles. Sizing
from the 8,192 final tokens would allocate only 64 tiles and leave legal source
rows unplanned.

### P1.6: combine receive validation

The receive extent is also represented in rank-major fixed-capacity space:

```text
logical_record = contributor_rank * shard_capacity + receive_slot
```

The data stage partitions that extent into 128-record tiles. Each tile validates
the record header and top-k lane contract, stores the derived
`origin_token * topk + contribution_lane` index, and writes one private error
candidate. It deliberately does not write the public `slots` array because two
records can derive the same index.

The next one-block control stage scans contributor ranks and receive slots in
protocol order. It constructs `slots`, detects the first duplicate, and compares
that position with the earliest tiled validation error. This preserves the old
rank/slot first-error rule without making atomic completion order observable.
Only after this stage succeeds does the vector path compact contributor slots.

The focused two-rank matrix passed normal combine, expanded multiple reduction,
duplicate experts within one rank, and the top-k 8 / hidden 7,168 specialization.
These runs establish compile and functional correctness, not EP8 performance.

### P2.7: direct dispatch grouping reuse

Normal direct dispatch now reads each token's top-k lanes once in
`ProducerGroup`. The data stage stores token/rank ownership and tile counts,
the one-block prefix stage converts them to deterministic rank-local offsets,
and `ProducerRecord` writes records from those offsets. Cached mode and subgroup
shapes outside the qualified fast path keep their validated fallback. Current
stage-order tests, the full host suite, and the Bisheng extension build all
revalidated this split after the P1/P2 metadata changes.

### Interaction Between P0.2 and the Metadata Findings

The missing top-k grouping primitive does not change how many threads or
blocks are launched. It reduces effective parallelism by forcing repeated
work. CUDA maps the top-k choices for a token to warp lanes, uses
`__match_any_sync` to group equal rank keys, and elects one lane for slot
allocation or reduction. The current Ascend direct combine instead performs
up to:

```text
8192 tokens * 7168 hidden * 8 ranks * 8 top-k
  = 3,758,096,384 rank/lane checks per rank
```

The new Ascend subgroup adapter must provide these operations:

- group active lanes by an integer key;
- elect one deterministic owner for each group;
- broadcast the elected lane's slot or rank;
- produce active and unique-owner masks; and
- preserve behavior for invalid `-1` lanes.

Use a native Ascend SIMT match primitive when the qualified compiler exposes
one. Otherwise implement the adapter from qualified subgroup ballot and
shuffle operations. The rest of dispatch and combine must depend on this
adapter rather than compiler-specific intrinsics directly.

## Architecture

### Multi-Kernel Stage Pipeline

The selected design separates control and data stages into stream-ordered
global kernels. This is required because the existing consecutive
`asc_vf_call` stages synchronize only the threads in one block. Merely changing
the current launch from one to 72 blocks would duplicate transport publication
and completion stores and would allow later blocks to observe incomplete work.

Control kernels use one block. Data kernels use a configurable block count,
with 72 selected for the first maximum-performance trial. All data work uses:

```text
global_thread = blockIdx.x * blockDim.x + threadIdx.x
global_stride = gridDim.x * blockDim.x
```

Kernel launch order on the same stream is the global stage boundary. No
cross-block spin barrier is introduced. This avoids deadlock assumptions about
block residency and gives each stage an independently measurable event range.

The tiling and runtime contract will distinguish:

- `control_num_blocks = 1`;
- `data_num_blocks`, initially 72 for the benchmark;
- the existing 512 SIMT threads per block; and
- the workspace needed by deterministic count, prefix, scatter, and error
  reduction stages.

The public `num_sms` argument maps to the Ascend data block count. Value 1
remains a supported fallback. The backend validates the configured count
against the qualified device limit. The canonical Ascend benchmark explicitly
uses 72; production callers may select fewer resources when communication must
overlap with compute.

### Dispatch Pipeline

Direct dispatch is split into these ordered stages:

1. A one-block control kernel clears status and counters and validates layout
   and topology.
2. A data-grid grouping/count kernel reads each token's top-k choices once,
   groups equal destination ranks, and accumulates rank and expert counts.
3. A prefix kernel assigns deterministic rank-local slots and expert ranges.
4. A data-grid scatter kernel writes destination slots and constructs records.
5. A one-block release kernel publishes payload and control state only after
   all record writers have completed.
6. A one-block acquire kernel observes source controls and creates receive
   ranges.
7. Data-grid validation, metadata, expanded destination, padding, and output
   copy kernels process received records.
8. A one-block completion kernel publishes the final generation after all
   output writes complete.

Cached handles keep their existing slot and ordering contracts. Parallel
validation writes private error candidates, and a deterministic reduction
selects the first error in the existing rank, record, and lane order.

### Combine Pipeline

Direct combine is split into these ordered stages:

1. One-block setup validates layout and clears status.
2. Data-grid planning validates source rows and builds destination counts.
3. Prefix and scatter stages construct combine records with vectorized hidden
   copies.
4. One-block transport publication releases completed records.
5. One-block acquire observes contributor controls.
6. Data-grid validation constructs a compact contributor list per token. Top-k
   grouping ensures that multiple lanes targeting one contributor select one
   record when multiple reduction is enabled.
7. The reduction grid partitions `(token, hidden-vector)` work. It iterates the
   compact contributor list rather than all ranks and all top-k lanes.
8. A separate data stage writes routing weights, followed by one-block
   completion publication.

### Vectorized Data Movement

Dispatch and combine records are aligned by the existing symmetric-window
layout. The new copy helpers use the widest qualified aligned transaction for
hidden payloads and record-local metadata. FP8 scale-factor layouts retain
their existing token and pack strides. A scalar tail handles unaligned byte
counts, but the canonical hidden widths must remain entirely on the aligned
path.

Where the Ascend programming model requires UB staging, each data block owns a
disjoint tile and uses double-buffered GM-to-UB and UB-to-GM movement. Output
ownership remains disjoint, so the copy path does not require atomics.

### Communication Publication

Correct release ordering is non-negotiable: remote control may become visible
only after its payload. The first communication optimization batches payload
commands for all nonlocal peers, performs the minimum required flush, then
publishes controls in deterministic rank order. Further peer/channel
parallelism is retained only when stage traces demonstrate that command
submission or service time is material.

The design does not infer performance from host submission time. Profiling
must distinguish command construction, device transport service, payload
completion, remote release observation, and the final device barrier.

## Profiling Protocol

Before retaining an optimization, run one representative canonical FP8 case
with correctness enabled, one warmup, and at least three measured repetitions.
Record:

- total dispatch, expanded dispatch, cached dispatch, combine, and reduced
  combine device time;
- every control and data stage device time;
- AI Vector utilization and active block count;
- bytes copied by local data stages;
- HCOMM submission, service, wait, and barrier time; and
- the maximum latency rank.

Use the unchanged workload manifest and event timing boundaries. Profiling
instrumentation must be compile-time or command-line gated and must not remain
active in formal canonical results.

Each optimization is evaluated independently. There is no fixed percentage
gate. Retain it when correctness and protocol checks pass and repeated
measurements show a defensible gain or a justified neutral result without a
material regression in unaffected public operations. This development
decision is not a final H800 parity claim.

## Implementation Order

1. Add stage timing and AI Vector utilization evidence without changing
   behavior.
2. Introduce stream-ordered control/data kernels and run payload and reduction
   stages with 72 AI Vector blocks.
3. Add the Ascend top-k subgroup grouping adapter and eliminate the combine
   `world_size * topk` hidden-element scan.
4. Replace scalar hidden and scale-factor copies with aligned vector or
   DataCopy paths.
5. Replace expert-owner rescans with record-oriented histogram, prefix, and
   scatter stages.
6. Partition rank-owned dispatch and combine validation across the data grid.
7. Batch or parallelize HCOMM peer command publication based on measured stage
   time.
8. Tune data block counts across 1, 8, 16, 32, 64, and 72 after the full-72
   trial. Keep 72 as the canonical maximum-performance setting unless measured
   contention makes a lower count faster.

This order addresses the device-wide utilization cap first, then removes the
largest repeated-work term, then improves memory transactions. Lower-volume
metadata and transport-control work follows measured impact.

## Correctness And Compatibility

- Direct compact and expanded output ordering remains byte-for-byte compatible
  with the current backend.
- Cached destination slots, rank prefixes, expert prefixes, source metadata,
  routing weights, and error precedence remain unchanged.
- Release and acquire generation ordering remains unchanged.
- Async operation and previous-event dependencies retain the same public event
  boundary.
- Hybrid and scale-out execution continue to select their current fallback and
  must not observe the direct-path workspace.
- BF16 and FP8 use the same stage structure; only payload and scale-factor copy
  helpers differ.

## Verification

Verification proceeds from narrow contracts to formal comparison:

1. Host tests cover tiling ABI, checked workspace arithmetic, block ownership,
   deterministic prefix/scatter logic, subgroup grouping, first-error
   selection, and fallback selection.
2. ASC compile probes instantiate every direct BF16/FP8, compact/expanded,
   cached/uncached, and sync/async stage.
3. One- and two-rank device tests verify record ordering and transport release
   behavior within the TaskQueue device policy.
4. A representative canonical FP8 case verifies stage timings and output
   correctness.
5. The eight-rank smoke profile must pass all 144 cases and 720 operations.
6. The full canonical profile runs 30 warmups and 30 measured iterations for
   all 144 cases and generates the standard JSON and Markdown artifacts.
7. A fresh H800 canonical report with the same workload fingerprint is used by
   the offline comparison tool.

At every stage, keep the five-second operation watchdog unchanged. A timeout,
incorrect output, missing case, malformed report, or incomplete device event
is a failed optimization regardless of measured speedup.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Seventy-two blocks duplicate control work or race completion | Put control and data work in different global kernels and use stream order as the global barrier |
| Full AI Vector use prevents communication-compute overlap | Treat 72 as the canonical maximum-performance mode and retain configurable smaller counts for production overlap |
| Parallel counts change deterministic ordering | Use count, deterministic prefix, and scatter stages; never use an atomic result as final order |
| Subgroup behavior differs from CUDA warp behavior | Hide intrinsics behind a tested Ascend adapter with explicit active-mask and invalid-lane semantics |
| Vector copies violate FP8 scale layout | Preserve existing token/pack strides and test row-major and column-major scale-factor layouts independently |
| Batched HCOMM publication exposes control before payload | Keep payload completion and release publication as separate ordered stages and retain generation validation |
| Extra kernel launches dominate after large speedups | Measure launch overhead, then fuse only stages with compatible ownership and no global dependency |

## Completion Criteria

The optimization program is complete only when:

- all eight findings have either an implemented optimization or measured
  evidence that the proposed change is not beneficial;
- direct data stages demonstrably execute on multiple AI Vector resources and
  the canonical maximum-performance profile evaluates 72;
- the representative canonical case improves without timeout or correctness
  changes;
- all 144 smoke and canonical cases pass with 720 complete operation records;
- formal reports retain the same workload fingerprint, operation set, timing
  protocol, and logical-byte formulas on H800 and Ascend; and
- the final design records residual bottlenecks and the measured reason for any
  retained single-block or single-thread stage.
