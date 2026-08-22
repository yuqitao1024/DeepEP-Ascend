# Ascend Top-k Subgroup Grouping Design

## Status

This specification defines P0.2 from
`epv2-ascend-performance-optimization.md`. It follows the completed 72-AIV
direct pipeline work. The first implementation slice optimizes direct combine;
the second reuses the same grouping adapter in direct dispatch.

The repository calls the completed multi-AIV work P0.1. Earlier discussion
called the 72-core slice P0.0. Direct-combine CUDA-alignment items 1 through 5
and the four-token direct dispatch grouping path are now qualified. This
document uses the repository numbering:

- P0.1: stream-ordered control/data kernels and a configurable 1-to-72 data
  grid, completed at commit `993ba59`;
- P0.2: vote-ballot grouping plus token-subgroup execution and register-local
  routing metadata, specified here; and
- P0.3: vector or DataCopy payload movement and common-shape specialization,
  tracked here as the final CUDA-alignment steps but implemented only after
  the P0.2 control-path gates pass.

## Problem

P0.1 removed the single-AI-Vector launch ceiling. It did not remove repeated
routing work inside the data kernels. The direct combine reduction still
derives the contributor and receive slot for every output hidden element even
though those values depend only on the token and top-k lane.

Define:

- `T`: number of tokens;
- `H`: hidden elements per token;
- `R`: world size;
- `K`: top-k width;
- `E[t, k]`: expert selected by lane `k` for token `t`;
- `rank(E[t, k])`: contributor rank that owns that expert;
- `S[t, k]`: receive slot recorded for that contribution; and
- `P[r, s, h]`: hidden element `h` in contributor `r`, receive slot `s`.

The current reduction is equivalent to:

```text
Y[t, h] = sum over r in [0, R), k in [0, K):
              valid(E[t, k])
            * (rank(E[t, k]) == r)
            * (S[t, k] >= 0)
            * P[r, S[t, k], h]
```

The validity, rank, and slot predicates do not depend on `h`, but the current
loop evaluates them under the `(token, hidden)` loop. Its routing-check count
is therefore:

```text
T * H * R * K
```

For the qualified representative workload:

```text
T = 4096, H = 7168, R = 8, K = 6

4096 * 7168 * 8 * 6 = 1,409,286,144 rank/lane checks
```

This is not a second execution of the model router. `combined_topk_indices`
already contains the selected experts. The repeated work is decoding each
expert's contributor, testing it against every rank, finding the receive slot,
and reconstructing the record address for each hidden element.

### Why a top-k lane can be `-1`

A normal unconstrained router can produce `K` valid experts. The DeepEP
interface is broader: `topk_idx` has a fixed dense shape `[T, K]` and uses
expert `-1` for an unused choice or masked route. This permits partially used
top-k rows and tokens with no active route without introducing a ragged tensor
or a second per-token length array.

The protocol meaning is exact:

```text
expert == -1   unused lane; do not dispatch, count, allocate, or combine
expert < -1    invalid expert; preserve the current protocol error
expert >= E    invalid expert; preserve the current protocol error
```

P0.2 converts every unused lane to a false logical-active predicate before
`vote_ballot`. It never creates an invalid-key group, elects an owner for that
lane, or reads a receive slot for it. The route's weight is ignored by the
communication operation regardless of its stored value.

### What the repeated-scan elimination simplifies

The optimization is loop-invariant code motion. Contributor rank and receive
slot are invariant over hidden element `h`, so P0.2 moves their derivation out
of the hidden loop and stores the result once per token.

Consider one token with `R = 8`, `K = 6`, and these derived contributor keys:

```text
top-k lane:                 0   1   2   3   4   5
contributor rank:           0   0   1   1   5  -1
validated contribution slot: 7  -1   3  -1   9  -1
```

In a collapsed multiple-reduction mode, lanes with the same contributor share
one returned record. Before P0.2, every hidden element scans all `8 * 6 = 48`
rank/lane combinations to rediscover that only three records contribute.

Grouping runs once for the token and produces:

```text
compact contributor entries:
    [(rank=0, owner_lane=0, slot=7),
     (rank=1, owner_lane=2, slot=3),
     (rank=5, owner_lane=4, slot=9)]

resolved slot by top-k lane:
    [7, 7, 3, 3, 9, -1]
```

The hidden loop then becomes the direct expression:

```text
Y[t, h] = P[0, 7, h] + P[1, 3, h] + P[5, 9, h]
```

For this token, the old route bookkeeping performs:

```text
H * R * K = 7168 * 8 * 6 = 344,064 rank/lane checks
```

The new path groups six lanes once, then performs three necessary payload
loads and FP32 additions for each hidden element. It does not remove expert
execution, payload transfer, or numerical reduction. It removes the repeated
expert-to-rank comparison, slot validity test, and record-address search that
surrounded those necessary additions.

## Goals

- Expose a backend-local top-k grouping adapter with `match_any`-equivalent
  semantics built from `vote_ballot`, shuffle, first-set-bit, and
  population-count operations.
- Resolve direct-combine contributor records once per token in deterministic
  compact order, first through a checked workspace ABI and finally in
  subgroup registers after the fused path is qualified.
- Make the hidden reduction iterate compact contributor entries without a
  `world_size * topk` routing scan.
- Reuse the grouping adapter in a later direct-dispatch count, prefix, and
  scatter pipeline.
- Preserve public outputs, BF16 accumulation order, routing weights, cached
  handles, protocol errors, and release/acquire ordering.
- Retain the explicit one-block, hybrid, scale-out, and expanded
  single-reduction fallbacks until their separate paths are qualified.

## Non-goals

- Vectorizing hidden, FP8 scale-factor, or metadata copies inside the P0.2
  control-path changes. Vector/DataCopy work starts only at alignment item 4,
  the separately gated P0.3 payload slice.
- Changing HCOMM command submission, service, flush, or release publication.
- Implementing the P1.5 expert histogram or P1.6 validation parallelization.
- Changing public Python arguments, benchmark cases, logical-byte formulas,
  workload fingerprints, watchdogs, or timing boundaries.
- Claiming H800 parity from logical bandwidth or from this optimization alone.

## Grouping Semantics

Each subgroup lane supplies an integer key and an active predicate. For top-k
grouping, the key is the contributor or destination rank. A route with expert
`-1` is invalid and excluded from all groups.

For example:

```text
top-k lane:       0  1  2   3  4  5
rank key:         2  2  5  -1  5  2

rank 2 group:     lanes 0, 1, 5; owner lane 0
rank 5 group:     lanes 2, 4;    owner lane 2
invalid group:    lane 3 is inactive
```

For every participating lane, the adapter returns:

```text
active_mask         all valid lanes in the subgroup
group_mask          valid lanes with the caller's key
owner_lane          least-numbered lane in group_mask
unique_owner_mask   one owner bit for every unique key
is_owner            caller lane equals owner_lane
```

The owner rule is deterministic. Invalid lanes have an empty group, no owner,
and never appear in `unique_owner_mask`. The adapter also broadcasts an owner
value, such as the receive slot, to all lanes in the group.

The subgroup width and mask representation are properties of the qualified
Ascend compiler and target. Consumers must use adapter constants and types;
they must not hard-code CUDA's warp width or mask type.

## Vote-ballot Implementation

The Ascend backend does not assume or require a native `match_any` intrinsic.
It constructs equivalent grouping semantics from `vote_ballot` and shuffle.
All physical lanes in the active execution subgroup participate in the same
loop; logical invalid lanes only contribute a false ballot predicate.

```text
valid_mask = vote_ballot(logically_active && key != -1)
remaining = valid_mask
my_group = 0

while remaining != 0:
    leader = first_set_bit(remaining)
    leader_key = shuffle(key, leader)
    peers = vote_ballot(logically_active && key == leader_key)
    if peers contains my_lane:
        my_group = peers
    remaining = remaining & ~peers

owner_lane = first_set_bit(my_group)
unique_owner_mask = vote_ballot(my_group != 0 && my_lane == owner_lane)
owner_value = my_group == 0 ? invalid : shuffle(value, owner_lane)
```

This algorithm groups equal keys without assuming that equal-key lanes are
contiguous. The loop executes once per unique key, bounded by `K` for the
DeepEP top-k use case.

A compile and device probe compares the ballot implementation with the host
reference model and must return byte-identical masks, owners, broadcasts, and
compact entry order. The probe also establishes the exact ballot and shuffle
API spelling, subgroup width, and mask type for the pinned CANN toolchain.
These compiler-specific details stay inside the adapter.

The Phase A probe qualified this contract on CANN 9.2.0 for `dav-3510`:

```text
subgroup width:       32 lanes
mask type:            uint32_t
ballot:               asc_ballot(predicate ? 1 : 0)
shuffle:              asc_shfl(value, source_lane, 32)
first set lane:       __ffs(static_cast<int32_t>(mask)) - 1
lower-lane mask:      lanemask_lt()
population count:     __popc(mask)
```

TaskQueue task `task_20260821_160951_6423513389` compiled, linked, and ran the
five device fixtures on NPU device 6. The fixtures covered all-equal keys,
all-distinct keys, noncontiguous duplicates, inactive `-1` lanes, and a
partial logical mask. The executable reported `topk-grouping PASSED`. This is
a ballot/shuffle construction; the backend neither found nor uses a native
`match_any` intrinsic.

For `K` greater than the qualified subgroup width, the public API remains
valid and selects a deterministic per-token bounded-scan implementation. The
canonical `K = 6` path must use subgroup grouping.

## Phase A: Combine-first Optimization

### Current data flow

```text
acquire contributor controls
  -> validate records and write slots[token, contribution_lane]
  -> reduce validation errors
  -> for every (token, hidden), scan every rank and top-k lane
  -> scan top-k again for each routing-weight output
  -> publish completion
```

### First compact-table data flow

```text
acquire contributor controls
  -> validate records and write slots[token, contribution_lane]
  -> reduce validation errors
  -> group top-k once per token and build compact contributor entries
  -> reduce every (token, hidden) over compact entries
  -> write routing weights through resolved owner slots
  -> publish completion
```

The grouping kernel is a data-grid stage submitted on the same stream after
validation error reduction and before hidden reduction. Stream order is the
global boundary. No cross-block spin barrier is added.

Each subgroup owns one token at a time. Lanes below `num_topk` load one expert
each, map valid experts to contributor-rank keys, group equal ranks, and elect
the canonical lowest lane. Remaining subgroup lanes participate in the
vote-ballot primitive with a false logical-active predicate.

### Workspace representation

The direct-combine workspace adds checked storage for:

```cpp
struct CombineContributorEntry {
    std::int32_t contributor_rank;
    std::int32_t contribution_lane;
    std::int32_t receive_slot;
};

std::uint32_t contributor_count[num_tokens];
CombineContributorEntry contributors[num_tokens][num_topk];
```

Every size and offset uses the existing checked layout builder. The entry ABI
has an explicit size and alignment assertion. Unused entries are not read.

The existing epilogue table at `WorkspaceLayout::slot_offset` is reused in
two phases:

1. validation writes slots only at observed `contribution_lane` positions and
   detects duplicates;
2. after validation succeeds, grouping broadcasts the canonical owner slot
   and fills a resolved slot for every valid top-k lane.

This reuse avoids a second `num_tokens * num_topk` slot table. It is safe only
because error reduction and grouping are separate stream-ordered kernels.
`WorkspaceLayout::combine_record_slots_offset` is separate producer scratch
indexed by source row and is not reused by P0.2.

### Compact order and numerical compatibility

The existing numerical contract accumulates by contributor rank, then by
logical top-k lane. Compact entries therefore use this exact ordering:

```text
(contributor_rank ascending, contribution_lane ascending)
```

A unique owner computes its compact ordinal by counting unique owners with a
smaller contributor key, with lane as the deterministic tie-breaker. Entry
construction never uses an atomic arrival order.

For normal compact combine and expanded combine with
`allow_multiple_reduction = true`, one contributor record represents all
same-rank lanes and produces one compact entry at the canonical master lane.
The resolved lane-slot table still broadcasts that record slot to all
same-rank lanes so routing weights can be restored without a candidate-lane
scan.

Expanded combine with `allow_multiple_reduction = false` can contain one
record per contribution lane. It retains the existing one-block algorithm in
the first P0.2 slice. Extending the compact table to that mode is a separate
qualification step and must retain contributor-rank then lane order.

### Reduced equation and complexity

Let `G[t]` be the compact list for token `t`:

```text
G[t] = {(rank_i, lane_i, slot_i) | the contribution record is valid}
```

The hidden reduction becomes:

```text
Y[t, h] = sum over entry i in G[t]:
              P[rank_i, slot_i, h]
```

Let `U[t]` be the number of unique valid contributor keys and `C[t]` the
compact entry count. For the collapsed multiple-reduction modes:

```text
U[t] <= min(R, K)
C[t] = U[t]
```

The ballot algorithm executes one collective round per unique key. Its work
and the payload reduction become:

```text
key formation:                   O(T * K)
vote-ballot collective rounds:   O(sum_t U[t])
scalar-equivalent comparisons:   O(sum_t K * U[t])
payload reduction:               O(sum_t H * C[t])
```

For `T = 4096`, `K = 6`, and all six keys distinct, grouping requires at most
`4096 * 6 = 24,576` unique-key ballot rounds. A scalar equivalent would make
at most `4096 * 6 * 6 = 147,456` key comparisons. Both are performed once
before hidden reduction, instead of the current 1,409,286,144 rank/lane checks
under the hidden loop.

The payload loads and FP32 additions remain necessary. P0.2 removes repeated
expert validity checks, contributor comparisons, slot-index calculations, and
record-address selection from the hidden loop. Relative to the current code,
the routing relationship is derived once per token instead of `H = 7168`
times for the representative workload.

### CUDA alignment gaps and closure order

The first compact-table implementation proved the grouping semantics and
removed the explicit `R * K` scan, but it did not reproduce the execution and
storage hierarchy of the CUDA DeepEP v2 epilogue. CUDA assigns one warp to a
token, keeps `topk_slot_idx[]` and rank metadata in registers, reduces vector
chunks cooperatively, and writes through shared memory and TMA in the same
epilogue kernel. Materializing the compact relation in global memory is not
part of that CUDA path.

The remaining gaps are split into five ordered items. Each item gets its own
host RED/GREEN, ASC compile, two-rank correctness, and unchanged representative
72-block measurement. A later item cannot hide a failed gate from an earlier
one.

| Order | Gap | Current Ascend behavior | CUDA DeepEP v2 behavior | Ascend closure |
| ---: | --- | --- | --- | --- |
| 1 | Token execution granularity | The reduction data grid assigns one SIMT thread to one scalar `(token, hidden)` element. | One warp owns one token and its lanes cover hidden chunks cooperatively. | Assign one qualified 32-lane subgroup to one token. Lane `l` reduces `h = l, l + 32, ...`; preserve contributor order and scalar BF16 output exactly. |
| 2 | Routing metadata load ownership | Item 1 leaves all 32 lanes loading the same count and 12-byte entries once per token. | The warp computes slots once and exchanges register values between lanes. | Make one lane load each field and broadcast it with `asc_shfl`, reducing 32 identical per-token loads to one; all physical lanes execute the same collectives. |
| 3 | Stage and GM round trip | A group stage writes count, entries, and resolved slots to GM; a later reduce stage reads count and entries. | Grouping and payload reduction share one epilogue kernel and register state. | Fuse grouping into the token-subgroup reduction after item 2 is qualified. Keep resolved slots only when the separate weight output still needs them; then remove unused compact workspace and the extra stage through an ABI change. |
| 4 | Payload width and local storage | Each lane performs scalar BF16 load, FP32 add, and BF16 store operations. | Lanes reduce vector chunks, use shared memory, and issue TMA output movement. | Introduce the qualified Ascend vector/DataCopy/UB path with a scalar tail. This is the P0.3 payload slice and must not change logical bytes or numerical order. |
| 5 | Compile-time specialization | Runtime `num_topk`, hidden bounds, and address arithmetic remain in inner loops. | `K`, `H`, rank layout, vector width, and unroll factors are template constants. | Add explicit AOT fast paths for qualified common shapes, starting with `K=6` and aligned `H=7168`; retain the dynamic path for all other valid shapes. |

The first correction implements item 1 only. Each lane may load the same
compact metadata once per token in that step; the 32-to-1 metadata-load
reduction belongs to item 2. This deliberately separates the benefit of
token-subgroup scheduling from the benefit of owner-lane broadcast.

Item 1 changes the work mapping from:

```text
thread q -> one scalar output (token = q / H, hidden = q % H)
```

to:

```text
subgroup w -> one token t
lane l -> hidden indices l, l + 32, l + 64, ... < H
```

For the representative `H = 7168`, each lane processes exactly
`7168 / 32 = 224` hidden elements. Contributor metadata is loop-invariant for
those 224 elements and remains in lane registers after its once-per-token
load. Item 2 then changes 32 identical lane loads into one owner-lane load plus
subgroup broadcasts.

The five-item sequence is not a claim that CUDA TMA maps directly to one
Ascend instruction. It aligns ownership, lifetime, and movement level first;
the concrete Ascend implementation uses only operations qualified by the
pinned CANN toolchain.

## Phase B: Dispatch Grouping

After Phase A passes its correctness and performance gates, direct dispatch
reuses the adapter:

```text
load one token's top-k experts
  -> map expert to destination-rank key
  -> group equal destination ranks
  -> elect one owner per unique destination
  -> count records per destination
  -> deterministic prefix by destination and token tile
  -> scatter destination slots and broadcast each slot to same-rank lanes
  -> construct records
```

### Four-token tile implementation

The direct-dispatch candidate assigns one 32-lane subgroup to a four-token
tile. For each token, lanes `0..K-1` load one expert and derive its destination
rank. The vote-ballot adapter groups equal ranks, and the lowest top-k lane in
each group becomes its owner. The group stage writes:

```text
owner[token, destination rank] = lowest matching top-k lane
tile_count[tile, destination rank] = selected tokens in this tile
tile_error[tile] = deterministic invalid-top-k candidate
```

Absent owner entries are `-1`. Expert `-1` never enters a group and leaves its
uncached destination slot at `-1`. A non-`-1` invalid expert contributes only
an error candidate, not a destination count.

The workspace sizes are checked on the host before launch:

```text
owners:      T * R * sizeof(int32_t)
tile counts: ceil(T / 4) * R * sizeof(uint64_t)
tile errors: ceil(T / 4) * sizeof(uint64_t)
```

One control block scans `tile_count[:, r]` in tile order for each rank `r`. It
replaces every count with its exclusive prefix and writes the final rank count.
The record stage then starts a register-local cursor from that tile base and
advances it once for every selected token. Therefore the uncached local slot is
equivalent to:

```text
slot(t, r) = sum(tile_count[j, r] for j < tile(t))
           + sum(selected(u, r) for u in tile(t), u < t)
```

This preserves token-ascending slots without using atomic arrival order. Each
top-k lane shuffles the cursor and owner from its destination-rank lane. Only
the owner writes the record; duplicate lanes receive the same encoded slot.

The split direct pipeline now has eight stream-ordered stages:

```text
producer control -> group -> prefix -> record -> release
                 -> epilogue prepare -> copy -> complete
```

Cached dispatch deliberately retains the existing serialized planner and
bitmap validation. After validation, the group stage creates only the owner
metadata needed by record construction; its tile count and error outputs are
not consumed by the cached path. Record construction reuses the already
validated encoded slots.
Hybrid dispatch, `K > 32`, `R > 32`, and the operational `num_sms = 1`
rollback retain the previous paths.

The first CANN 9.2.0 Bisheng resource build for this candidate completed with:

| VF | Registers | Stack |
| --- | ---: | ---: |
| `direct_dispatch_producer_group_vf` | 57 | 0 B |
| `direct_dispatch_producer_prefix_vf` | 26 | 0 B |
| `direct_dispatch_producer_record_vf` | 64 | 152 B |

The two-rank production matrix passed all 14 cases in TaskQueue task
`task_20260823_044925_111036914431`. Coverage included duplicate destination
ranks, expert `-1`, cached reuse, 100 sequential generations, round trip, and
invalid-expert diagnostics.

The unchanged representative case used `T=4096`, `H=7168`, `K=6`, `E=256`,
world size 2, BF16, `num_sms=72`, one warmup, and three measured samples. The
workload fingerprint remained
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`.
The baseline is the two-run `__launch_bounds__(512)` result at commit
`589ecce`; the candidate is the mean of TaskQueue tasks
`task_20260823_045607_114494525479` and
`task_20260823_045705_114794218227`:

| Operation | Baseline latency | Candidate latency | Latency change | Baseline bandwidth | Candidate bandwidth | Bandwidth change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 143.977 ms | 100.620 ms | -30.11% | 4.844 GB/s | 6.931 GB/s | +43.09% |
| Expanded dispatch | 144.348 ms | 100.582 ms | -30.32% | 9.752 GB/s | 13.995 GB/s | +43.51% |
| Cached dispatch | 133.245 ms | 105.302 ms | -20.97% | 5.234 GB/s | 6.623 GB/s | +26.54% |
| Combine control | 88.094 ms | 87.808 ms | -0.32% | 6.595 GB/s | 6.617 GB/s | +0.33% |
| Reduced combine control | 110.519 ms | 110.098 ms | -0.38% | 5.257 GB/s | 5.277 GB/s | +0.38% |

The two candidate dispatch runs were `6.933/6.929 GB/s`, expanded dispatch
was `13.978/14.012 GB/s`, and cached dispatch was `6.664/6.583 GB/s`.
Unrelated combine operations stayed within 0.4% of their launch-bounds
baseline, which is consistent with the candidate changing only dispatch.
Raw candidate reports are retained as
`/tmp/direct-dispatch-grouping-run1.json` and
`/tmp/direct-dispatch-grouping-run2.json` on the development node.

Uncached slot order remains token ascending within each destination rank.
Cached mode validates that all lanes in one destination group carry the same
encoded slot before broadcasting it. A mismatch remains
`kInvalidCachedSlot` with the current first-error precedence.

Final destination slots cannot be assigned by atomic arrival order. The
count, deterministic prefix, and scatter stages preserve handle compatibility
and make block scheduling irrelevant to output order. Phase B therefore
shares primitives with P2.7 but does not include expert histogram or receive
validation work from later priorities.

## Error And Fallback Behavior

- Expert `-1` is inactive and leaves its resolved lane slot at `-1`.
- Other negative experts and experts greater than or equal to `num_experts`
  retain the current protocol error.
- Empty groups produce no contributor entry.
- Missing owner slots, duplicate records, contributor mismatches, and invalid
  headers retain the existing error code and deterministic rank/record order.
- A nonzero status prevents grouping, reduction, weight output, and completion
  publication.
- `num_sms = 1` remains the operational rollback.
- Hybrid, physical/logical scale-out, and expanded single-reduction continue
  selecting their existing fallback and do not consume the new workspace.
- No watchdog is increased and no malformed result is accepted for a faster
  measurement.

## Verification

### Host semantic tests

The backend-local reference model covers:

- all-distinct and all-equal keys;
- repeated noncontiguous keys such as `[2, 2, 5, -1, 5, 2]`;
- partial active masks at the start and end of a token batch;
- invalid `-1` lanes;
- top-k values from 1 through the qualified subgroup width;
- subgroup-width overflow selecting the bounded-scan path;
- deterministic lowest-lane ownership and rank/lane compact ordering;
- checked workspace arithmetic and forged descriptor rejection; and
- mutations that reverse rank or lane accumulation order.

### ASC compile and device probes

Compile instantiations exercise the vote-ballot implementation for complete
and partial subgroups. A device probe compares its exact output with the host
semantic fixtures. It also proves that the selected subgroup width, mask type,
shuffle, first-set-bit, and population-count operations compile for the pinned
CANN toolchain.

NPU work uses only TaskQueue, one active task at a time, and permitted devices
`0`, `1`, `6`, or `7`; two-rank tasks use only `0,1` or `6,7`. Device selection
is explicit and never uses `auto`.

### Operation correctness

The two-rank suite covers BF16 compact, FP8 dispatch followed by BF16 combine,
cached dispatch handles, expanded multiple reduction, weighted duplicate
same-contributor lanes, invalid lanes, previous-event dependencies, and async
compute-stream mode. The order-sensitive case must retain its exact BF16
result, not merely a loose tolerance.

The eight-rank smoke remains the final compatibility gate for all 144 cases
and 720 operation records.

## Performance Protocol

The P0.1 72-block representative result at commit `993ba59` is the P0.2
baseline. These are logical benchmark bandwidths, not HBM or HCCL counter
measurements:

| Operation | 72-block logical bandwidth | Speedup over 1 block |
| --- | ---: | ---: |
| Dispatch | 4.638705 GB/s | 9.43x |
| Expanded dispatch | 9.257460 GB/s | 16.81x |
| Cached dispatch | 4.874099 GB/s | 9.85x |
| Combine | 5.939782 GB/s | 10.25x |
| Reduced combine | 4.800069 GB/s | 10.56x |

The matching one-block and 72-block tasks were
`task_20260821_132309_271487832248` and
`task_20260821_132456_271989713619`, using workload fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`.

P0.2 development qualification uses the unchanged representative workload,
one warmup, and at least three measured iterations. It records the qualified
subgroup width and mask type, grouping-stage time, reduction-stage time,
weight-stage time, public combine latency, logical bandwidth, and maximum
latency rank.

Qualification does not impose a minimum percentage performance threshold.
Every slice reports its measured change from the immediately preceding
selected implementation and from P0.1. Correctness under the existing
five-second watchdog and unchanged report schema, workload fingerprint, case
IDs, logical bytes, timing protocol, and execution protocol remain hard gates.
Small changes within run-to-run variation are reported as performance-neutral,
not promoted to gains or used as an automatic rollback trigger.

The grouping microbenchmark explains adapter cost but does not replace the
public-operation gate. Profiler evidence must distinguish grouping, payload
reduction, routing-weight restoration, launch overhead, and transport wait.
The earlier P0.1 run did not capture profiler-level simultaneous active-core
evidence, so its `device.num_sms = 72` field is configuration evidence, not an
active-residency claim.

### First compact-table measurement

Task `task_20260821_170909_151408525395` built the production extension on
devices `6,7` and passed eight focused two-rank cases: compact combine,
expanded multiple reduction, weights, duplicate same-contributor lanes,
`-1` routes, empty input, and one- and two-bias output. Task
`task_20260821_171407_18450785056` then passed the FP8/previous-event/async
smoke and the unchanged representative BF16 case.

The representative report kept schema version 2, the exact case
`ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc0`, fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`,
`num_sms = 72`, one warmup, three measured iterations, maximum-latency rank
aggregation, and the original logical-byte counts.

| Operation | P0.1 mean | Compact-table mean | Bandwidth change | Latency change |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | 4.638705 GB/s, 150.35 ms | 4.084343 GB/s, 170.75 ms | -11.95% | +13.57% |
| Expanded dispatch | 9.257460 GB/s, 152.06 ms | 8.291660 GB/s, 169.77 ms | -10.43% | +11.65% |
| Cached dispatch | 4.874099 GB/s, 143.09 ms | 4.657750 GB/s, 149.73 ms | -4.44% | +4.64% |
| Combine | 5.939782 GB/s, 97.82 ms | 5.062780 GB/s, 114.76 ms | -14.76% | +17.32% |
| Reduced combine | 4.800069 GB/s, 121.04 ms | 4.697684 GB/s, 123.68 ms | -2.13% | +2.18% |

The compact-table implementation therefore regressed public combine by 14.76
percent in this run. The simultaneous dispatch regressions and wide
three-sample spread indicate system-level noise, but do not convert the result
into a gain. The measurement motivated an exact repeat and stage-level
diagnosis. The earlier minimum-percentage rule has since been superseded by the
threshold-free qualification policy above.

Repeat task `task_20260821_171559_200754410601` produced no benchmark
measurement. It failed while constructing `ElasticBuffer`, before an EP kernel
launch, because HCCL `all_gather_object` timed out establishing its socket.
That task is environment-failure evidence only and is excluded from all
bandwidth and latency comparisons.

### CUDA-alignment item 1: one subgroup per token

Commit `8e68b8a` changed only the compact direct-combine reduction. One
32-lane subgroup owned a token, lane `l` reduced hidden elements
`l, l + 32, ...`, and each lane loaded the token's compact contributor ranks
and receive slots once before its hidden loop. It did not add the item 2
owner-lane broadcast: all 32 lanes still loaded the same metadata. Counts above
32 kept a correctness fallback that read entries directly rather than
overflowing the fixed lane-local arrays.

Task `task_20260821_184344_11774205132` compiled the complete ASC target on
device 6, linked the runner, and passed `combine-state-probe`. Task
`task_20260821_184701_126929126773` rebuilt the production extension with
`DEEP_EP_ASCEND_TESTING=0` on devices `6,7` and passed the eight focused cases:
normal, expanded multiple reduction, weights, duplicate same-rank experts,
`-1` route, empty input, one bias, and two biases. Its following FP8 launcher
failed before `ElasticBuffer` construction because the driver could not
allocate another HCCL stream after the first distributed launcher exited. That
failure contains no EP-kernel result. The same FP8 previous-event, async, and
communication-stream-allocation case passed by itself in task
`task_20260821_185045_13946435099`.

Representative task `task_20260821_185356_147346114207` ran on devices `6,7`
and exited successfully. The report SHA-256 is
`3ae007cf427178c8186e768c9072852d090a1983ff3458e954123c946034a7c2`.
It retained schema version 2, formula version 1, the exact case
`ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc0`, fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`,
`T=4096`, `H=7168`, `K=6`, `E=256`, world size 2, `num_sms=72`, one warmup,
three measured iterations, NPU-event timing, maximum-rank latency aggregation,
and `allow_multiple_reduction=1`. All five logical-byte objects exactly matched
the P0.1 report.

| Operation | Device samples | Mean latency | Logical bandwidth | Change from P0.1 | Change from compact table |
| --- | --- | ---: | ---: | ---: | ---: |
| Dispatch | 149.974, 149.869, 151.092 ms | 150.312 ms | 4.639822 GB/s | +0.02% | +13.60% |
| Expanded dispatch | 152.328, 151.244, 151.363 ms | 151.645 ms | 9.282500 GB/s | +0.27% | +11.95% |
| Cached dispatch | 143.888, 139.906, 142.190 ms | 141.995 ms | 4.911590 GB/s | +0.77% | +5.45% |
| Combine | 90.939, 90.153, 91.153 ms | 90.748 ms | 6.402551 GB/s | +7.79% | +26.46% |
| Reduced combine | 114.057, 114.276, 113.001 ms | 113.778 ms | 5.106618 GB/s | +6.39% | +8.71% |

Item 1 removes a real bottleneck: public combine improves 26.46 percent over
the compact-table experiment and 7.79 percent over P0.1, while unaffected
operations stay within the 10 percent regression limit. The five
CUDA-alignment items are not an all-or-nothing performance switch. Item 1
removed part of the bottleneck, and later items address the next exposed limit.
Item 1 is
**correctness-qualified and retained as the item 2 baseline**. Item 2 reports
its incremental change from `6.402551 GB/s`, in addition to the cumulative
change from P0.1.

### CUDA-alignment item 2: owner-lane metadata broadcast

Commit `41862b3` retained item 1's one-subgroup-per-token mapping and changed
only compact metadata ownership in direct combine. Lane 0 loads each token's
contributor count and each compact entry's contributor rank and receive slot;
all 32 physical lanes execute `asc_shfl(..., 0, 32)` in identical loop order.
The payload loads, FP32 accumulation order, BF16 output, workspace ABI, group
stage, weights stage, and contributor-count-above-32 fallback are unchanged.

Task `task_20260821_190342_174895030919` compiled and linked the complete ASC
target on device 6 and passed `combine-state-probe`. Task
`task_20260821_190618_18311782219` rebuilt the production extension on devices
`6,7` and passed normal, expanded multiple reduction, weights, duplicate
same-rank experts, `-1` route, empty input, one bias, and two biases. Independent
task `task_20260821_190827_188696829593` passed the FP8 previous-event, async,
and communication-stream-allocation smoke.

Representative tasks `task_20260821_191248_199948811884` and
`task_20260821_191359_204177716994` both exited successfully on devices `6,7`.
Their report SHA-256 values are
`71cf360b889bf4d5d1990d5195af75c3b7bb730489a014eac9b1b98d1b740af5` and
`68938fe5e9bb8c12bad7812c9620bd855dcff8069fd8dd42778e44cd47aef394`.
Both reports retained the item 1 case, fingerprint, workload, world size,
`num_sms=72`, one warmup, three measured iterations, NPU-event timing,
maximum-rank latency aggregation, execution protocol, and all five logical-byte
objects. The table aggregates all six samples from the two confirmation runs.

| Operation | Six device samples (ms) | Mean latency | Logical bandwidth | Change from item 1 | Change from P0.1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dispatch | 149.802, 149.779, 149.291, 149.790, 149.972, 149.800 | 149.739 ms | 4.657565 GB/s | +0.38% | +0.41% |
| Expanded dispatch | 151.351, 151.214, 151.564, 151.297, 151.546, 150.361 | 151.222 ms | 9.308448 GB/s | +0.28% | +0.55% |
| Cached dispatch | 142.806, 142.459, 142.414, 141.759, 141.853, 141.609 | 142.150 ms | 4.906213 GB/s | -0.11% | +0.66% |
| Combine | 91.473, 89.837, 91.155, 90.845, 91.102, 91.157 | 90.928 ms | 6.389870 GB/s | -0.20% | +7.58% |
| Reduced combine | 112.546, 113.792, 113.187, 113.328, 112.510, 113.751 | 113.186 ms | 5.133340 GB/s | +0.52% | +6.94% |

Item 2 is **correctness-qualified and performance-neutral** for this workload.
The 0.20 percent combine decrease is smaller than the variation inside the six
samples and is not treated as a regression or as a gain. The result shows that
the once-per-token compact metadata loads are not the current public-combine
bottleneck; shuffle cost can offset the removed duplicate GM loads. Item 2 is
retained as the ownership basis for item 3, where grouping and reduction can
share register state and remove the separate GM round trip. Item 3 must compare
against the immediate item 2 state, the best item 1 result, and P0.1.

### CUDA-alignment item 3: fused grouping and reduction

Commit `9bff96b` removed the standalone direct-combine Group stage. The direct
pipeline now has seven stages rather than eight: error reduction is followed
directly by the fused Reduce stage, then weights and completion. The checked
core tiling ABI moved from 16 to 17. The workspace no longer contains the
per-token contributor count or `T * K` compact entry table; the two-rank host
fixture's total combine workspace fell from 928 to 736 bytes, and its aligned
scratch region fell from 544 to 352 bytes. The record-slot workspace remains
because the following weights stage reads the resolved slot for every logical
top-k lane.

For `K <= 32`, one subgroup still owns one token. Every lane loads its expert
and observed slot once, calls the vote-ballot grouping adapter, broadcasts the
owner slot, and writes the resolved slot for the weights stage. Each valid
owner computes its own rank-order ordinal from its unbroadcast contributor
key. The subgroup then walks the common owner mask and broadcasts the owner's
ordinal, contributor rank, and receive slot into lane-local arrays. Indexing
those arrays by ordinal preserves the prior ascending-contributor-rank FP32
addition order. Passing a rank value already broadcast from one owner into the
ordinal helper would make every owner appear to have ordinal zero; the
implementation and source contract specifically prevent that mistake.

For `K > 32`, the fused kernel takes a correctness fallback before any
subgroup collective. One scalar grid worker owns a token, scans contributor
ranks in ascending order, propagates each first valid owner's slot to duplicate
lanes, and reduces hidden elements in the same rank order. This avoids a fixed
array larger than 32 and avoids divergent ballot or shuffle calls. It is not a
performance target, but it retains the earlier fallback's route and numerical
semantics.

The host RED/GREEN cycle finished with all 106 contracts passing. Task
`task_20260821_195147_27124303804` compiled and linked the complete ASC target
on device 6 and passed `combine-state-probe`. Task
`task_20260821_195400_280557827737` rebuilt the production extension on devices
`6,7` and passed normal, expanded multiple reduction, weights, duplicate
same-rank experts, `-1` route, empty input, one bias, and two biases. Task
`task_20260821_195723_320695327050` independently passed the FP8 previous-event,
async, and communication-stream-allocation smoke.

Representative tasks `task_20260821_195820_330724814006` and
`task_20260821_195929_349642329352` both exited successfully on devices `6,7`.
Their report SHA-256 values are
`e843b0001f8e90870832a65c3a0b4dd53e8210dd999cf83f108e25761877b80b` and
`a224fb22f25bd3d9cb60fa9da23f03e65a4202f0b63330684b610efc86d3dd78`.
Both retained schema 2, formula 1, case
`ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc0`, fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`,
world size 2, `num_sms=72`, one warmup, three measured iterations, NPU-event
timing, maximum-rank latency aggregation, execution protocol, and all five
logical-byte objects. The table aggregates all six device samples.

| Operation | Six device samples (ms) | Mean latency | Logical bandwidth | Change from item 2 | Change from item 1 | Change from P0.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dispatch | 148.939, 149.307, 148.022, 148.058, 148.565, 149.074 | 148.661 ms | 4.691347 GB/s | +0.73% | +1.11% | +1.13% |
| Expanded dispatch | 150.755, 149.796, 150.613, 198.230, 188.469, 148.257 | 164.353 ms | 8.564745 GB/s | -7.99% | -7.73% | -7.48% |
| Cached dispatch | 143.532, 142.002, 143.516, 135.090, 136.746, 139.630 | 140.086 ms | 4.978506 GB/s | +1.47% | +1.36% | +2.14% |
| Combine | 89.969, 88.864, 90.926, 89.788, 89.611, 92.479 | 90.273 ms | 6.436262 GB/s | +0.73% | +0.53% | +8.36% |
| Reduced combine | 114.813, 113.017, 114.359, 113.924, 111.844, 115.561 | 113.920 ms | 5.100253 GB/s | -0.64% | -0.12% | +6.25% |

Item 3 is **correctness-qualified and retained with a small positive public
combine result**. Its six-sample combine bandwidth is 0.73 percent above the
immediate item 2 state and 8.36 percent above P0.1. Expanded dispatch is noisy:
the first run measured 150.388 ms, while the second included 198.230 and
188.469 ms samples around a 148.257 ms sample. That operation does not consume
the changed combine code, and the six-sample aggregate remains within the
existing 10 percent unaffected-operation regression bound. We record the
outliers as system-level variation and do not attribute them to the fused
kernel. This qualification has no fixed minimum improvement threshold and does
not trigger an automatic rollback. Item 3 becomes the baseline for item 4.

### CUDA-alignment item 4: vectorized payload reduction

Commit `cbd88a2` moves the aligned direct-combine payload through AscendC
DataCopy and UB vector operations. The change applies to the staged direct
path when `K <= 32`, `H % 16 == 0`, and the launch uses more than one data
block. `kFull`, one-block execution, `K > 32`, and hidden sizes that fail the
16-element DataCopy alignment keep the item 3 SIMT reducer.

The seven-stage pipeline and tiling ABI 17 are unchanged. The work is split
across three existing stages:

1. `kEpiloguePrepare` runs ballot grouping and writes each duplicate lane's
   resolved slot to the existing slot workspace.
2. `kEpilogueReduce` assigns tokens to AIV blocks. Each block reduces full
   256-element BF16 tiles in UB. It copies one contributor tile from GM,
   casts it to FP32, adds contributors in ascending rank order, then adds bias
   0 and bias 1 in that order. A single round-to-nearest cast produces the
   BF16 output tile before DataCopy writes it to GM.
3. `kEpilogueWeights` runs the SIMT tail only when `H % 256 != 0`. The tail
   starts at `H - H % 256` and uses the same contributor and bias order.

This layout avoids padding and out-of-bounds reads. It also keeps the scalar
tail out of the outer `__aicore__` body. The pinned `dav-3510` compiler rejects
scalar BF16 conversion there, even though the same conversion is valid inside
a SIMT VF.

The compiler probes found two concrete toolchain restrictions. Task
`task_20260821_202248_28322867425` rejected a const pointer passed directly to
`GlobalTensor::SetGlobalBuffer`. The probe passed in task
`task_20260821_202441_303257421578` after a narrow `const_cast` at that API
boundary. Task `task_20260821_203127_380216632533` then rejected the outer-AIV
scalar BF16 tail. Moving the tail to the next stage's SIMT VF produced a full
compile, link, and `combine-state-probe` pass in task
`task_20260821_203520_417968711622`.

Task `task_20260821_203738_31306819256` passed the original eight focused
two-rank cases, and task `task_20260821_204021_5799062390` passed the independent
FP8 previous-event, async, and communication-stream-allocation smoke. The final
alignment gate compiled in task `task_20260821_205810_32507671319`. Task
`task_20260821_210655_11174930146` rebuilt the production extension and passed
ten cases on devices `6,7`. The two added cases use `H=256` for a pure vector
tile and `H=272` with two biases for one vector tile plus a 16-element SIMT
tail.

Representative tasks `task_20260821_211002_3775754950` and
`task_20260821_211104_45653122871` both completed on devices `6,7`. Their report
SHA-256 values are
`1bd821d7f9a49eba6373f56897c602272264795053f21214d4d9073ad476a808` and
`a09f63e5a4221b3215196ca8681030fb1387d743b408af345870584addd24c9b`.
Both reports retained schema 2, formula 1, case
`ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc0`, fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`,
world size 2, `num_sms=72`, one warmup, three measured iterations, maximum-rank
latency aggregation, and all five logical-byte objects. The table aggregates
the six samples from both runs.

| Operation | Six device samples (ms) | Mean latency | Logical bandwidth | Change from item 3 | Change from P0.1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dispatch | 148.928, 149.332, 149.388, 150.491, 149.016, 152.588 | 149.957 ms | 4.650786 GB/s | -0.86% | +0.26% |
| Expanded dispatch | 151.143, 149.830, 150.123, 151.280, 149.936, 149.701 | 150.336 ms | 9.363349 GB/s | +9.32% | +1.14% |
| Cached dispatch | 142.727, 141.639, 142.006, 138.151, 139.772, 137.760 | 140.343 ms | 4.969408 GB/s | -0.18% | +1.96% |
| Combine | 90.141, 90.804, 91.577, 87.689, 86.578, 89.429 | 89.370 ms | 6.501314 GB/s | +1.01% | +9.45% |
| Reduced combine | 114.342, 114.387, 114.068, 114.416, 111.941, 111.471 | 113.437 ms | 5.121944 GB/s | +0.43% | +6.71% |

Public combine is 1.01 percent faster than item 3, 1.74 percent faster than
item 2, 1.54 percent faster than item 1, and 9.45 percent faster than P0.1.
The two runs also show normal short-run variation: their separate combine
means are 6.396054 and 6.610096 GB/s. The six-sample aggregate is used for the
comparison. Item 4 is correctness-qualified and retained. Item 5 uses this
result as its immediate baseline and specializes only the common
`K=6, H=7168` shape.

### CUDA-alignment item 5: common-shape AOT specialization

Commit `dc35be3` adds one explicit template instance for the representative
`K=6, H=7168` shape. The item 4 vector reducer now takes `StaticNumTopk` and
`StaticHiddenElements` as non-type template parameters. `<6, 7168>` supplies
fixed bounds for the top-k scan, hidden-vector extent, token stride, and output
address calculation. The template exposes those values as constants when the
compiler generates the common-shape kernel. Token count, expert count, world
size, contributor count, bias count, and receive-shard layout stay dynamic.

The staged reduce branch selects `<6, 7168>` only when both runtime fields
match. Every other aligned `K <= 32` shape uses the `<0, 0>` dynamic instance.
The item 3 SIMT fallback still handles `kFull`, `K > 32`, one-block launches,
and hidden sizes that do not satisfy the 16-element DataCopy alignment. Item 5
does not change the seven stages, tiling ABI 17, workspace, HCOMM order, or
ascending-rank FP32 accumulation order.

The new public two-rank case uses `K=6`, `H=7168`, and `E=8`. Six top-k lanes
cannot be validated against the harness's old `E=4` default, because the public
dispatch preflight requires `K <= E`. The route fixture includes duplicate
contributors and an inactive `-1` lane, so it checks the specialized path
without weakening preflight. Task `task_20260821_212042_160760221632` compiled
and linked the complete ASC target and passed `combine-state-probe`. After the
fixture correction, task `task_20260821_213939_210129024589` rebuilt the
production extension on devices `6,7` and passed all ten item 4 cases plus the
new common-shape case.

Representative tasks `task_20260821_214231_224930514348` and
`task_20260821_214327_228271326996` both completed on devices `6,7`. Their
report SHA-256 values are
`ffc3418b8c6eec7774009f2b4f0db70b547d82cf5029a3687df71d1f5ec7e0f2` and
`b7131cc571562ac5bbc2813166b4ccf9c9c18d1f0236d5e8b906d5e50d442df5`.
Both reports retained schema 2, formula 1, case
`ep-bf16-align128-bias0-hcopy1-prev0-async0-alloc0`, fingerprint
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`,
world size 2, `num_sms=72`, one warmup, three measured iterations, maximum-rank
latency aggregation, execution protocol, and all five logical-byte objects.

| Operation | Six device samples (ms) | Mean latency | Logical bandwidth | Change from item 4 | Change from P0.1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dispatch | 150.222, 148.588, 150.373, 149.108, 150.731, 149.427 | 149.741 ms | 4.657491 GB/s | +0.14% | +0.40% |
| Expanded dispatch | 151.149, 149.355, 150.862, 150.218, 151.527, 151.343 | 150.742 ms | 9.338080 GB/s | -0.27% | +0.87% |
| Cached dispatch | 139.150, 142.330, 142.879, 143.084, 141.137, 142.032 | 141.769 ms | 4.919414 GB/s | -1.01% | +0.93% |
| Combine | 91.186, 93.721, 90.904, 89.947, 89.502, 91.764 | 91.171 ms | 6.372882 GB/s | -1.98% | +7.29% |
| Reduced combine | 112.540, 111.205, 109.153, 113.094, 114.036, 113.919 | 112.324 ms | 5.172692 GB/s | +0.99% | +7.76% |

The AOT instance did not improve public combine in this measurement. Its
six-sample bandwidth is 1.98 percent below item 4, 0.98 percent below item 3,
0.27 percent below item 2, and 0.46 percent below item 1. It remains 7.29
percent above P0.1. The result does not show that the compiler ignored the
constants. Confirming constant folding requires generated-code evidence. This
end-to-end test only shows that fixed `K/H` bounds did not produce a measurable
gain over item 4; any saving is smaller than the sample variation and the
remaining public-operation cost. Item 5 is correctness-qualified and retained
under the threshold-free policy. The dynamic instance remains the fallback for
every other valid shape.

## Documentation Updates

When Phase A is qualified, update the teaching guide to:

- mark P0.1 complete rather than saying the direct path still uses one of 72
  AI Vector resources;
- retain the P0.1 logical-bandwidth table with its measurement caveat;
- explain the `T * H * R * K` repeated-check formula and compact-table form;
- explain that Ascend constructs match-any grouping from `vote_ballot`, and
  record the qualified subgroup width and mask type; and
- list Phase B dispatch grouping as the next P0.2 slice until it passes its
  own gate.

## Delivery Order

1. Add host grouping semantics and checked workspace-layout tests.
2. Add the vote-ballot ASC compile and device probe.
3. Add the backend-local match-any-equivalent grouping adapter.
4. Measure the compact-table implementation and retain its correctness and
   failed-performance evidence.
5. Close CUDA-alignment item 1, token-subgroup execution, and run its complete
   gate without item 2 changes. Completed.
6. Close item 2, owner-lane metadata load and broadcast, and repeat the same
   gate. Completed.
7. Close item 3, fused grouping and reduction, remove unused workspace through
   a checked ABI change, and repeat the same gate. Completed.
8. Close item 4, vector/DataCopy payload movement, with aligned and scalar-tail
   coverage. Completed.
9. Close item 5, common-shape AOT specialization, while preserving the dynamic
   fallback. Completed.
10. Record qualified evidence in the teaching guide, then design the
    deterministic dispatch count/prefix/scatter slice using the qualified
    adapter.

Each implementation slice is independently revertible. A slice must pass the
correctness and report-identity gates; its measured end-to-end change is
recorded without a fixed minimum percentage threshold or automatic rollback.
