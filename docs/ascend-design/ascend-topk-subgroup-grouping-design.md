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
- P0.3: vector or DataCopy payload movement. The combine and direct-dispatch
  slices are qualified; the formal top-k-8 AOT specialization remains a
  separate optimization, and scalar tails and dynamic fallbacks stay in place.

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

For the formal DeepEP V2 comparison workload:

```text
T = 8192, H = 7168, R = 8, K = 8

8192 * 7168 * 8 * 8 = 3,758,096,384 rank/lane checks
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

Consider one token with `R = 8`, `K = 8`, and these derived contributor keys:

```text
top-k lane:                  0   1   2   3   4   5   6   7
contributor rank:            0   0   1   1   5  -1   3   3
validated contribution slot: 7  -1   3  -1   9  -1   4  -1
```

In a collapsed multiple-reduction mode, lanes with the same contributor share
one returned record. Before P0.2, every hidden element scans all `8 * 8 = 64`
rank/lane combinations to rediscover that only four records contribute.

Grouping runs once for the token and produces:

```text
compact contributor entries:
    [(rank=0, owner_lane=0, slot=7),
     (rank=1, owner_lane=2, slot=3),
     (rank=3, owner_lane=6, slot=4),
     (rank=5, owner_lane=4, slot=9)]

resolved slot by top-k lane:
    [7, 7, 3, 3, 9, -1, 4, 4]
```

The hidden loop then becomes the direct expression:

```text
Y[t, h] = P[0, 7, h] + P[1, 3, h] + P[3, 4, h] + P[5, 9, h]
```

For this token, the old route bookkeeping performs:

```text
H * R * K = 7168 * 8 * 8 = 458,752 rank/lane checks
```

The new path groups eight lanes once, then performs four necessary payload
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
top-k lane:       0  1  2   3  4  5  6  7
rank key:         2  2  5  -1  5  2  3  3

rank 2 group:     lanes 0, 1, 5; owner lane 0
rank 5 group:     lanes 2, 4;    owner lane 2
rank 3 group:     lanes 6, 7;    owner lane 6
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
formal `K = 8` path must use subgroup grouping.

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

For `T = 8192`, `K = 8`, and all eight keys distinct, grouping requires at
most `8192 * 8 = 65,536` unique-key ballot rounds. A scalar equivalent would
make at most `8192 * 8 * 8 = 524,288` key comparisons. Both are performed
once before hidden reduction, instead of 3,758,096,384 rank/lane checks under
the hidden loop.

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
| 5 | Compile-time specialization | Runtime `num_topk`, hidden bounds, and address arithmetic remain in inner loops. | `K`, `H`, rank layout, vector width, and unroll factors are template constants. | Add an explicit AOT fast path for the formal `K=8`, aligned `H=7168` shape; retain the dynamic path for all other valid shapes. |

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


The focused two-rank tasks above establish functional correctness only. Formal
dispatch performance is accepted only from the eight-rank canonical profile,
which measures all 144 cases and all 720 operation records under the current
DeepEP V2 workload and timing protocol.

Uncached slot order remains token ascending within each destination rank.
Cached mode validates that all lanes in one destination group carry the same
encoded slot before broadcasting it. A mismatch remains
`kInvalidCachedSlot` with the current first-error precedence.

Final destination slots cannot be assigned by atomic arrival order. The
count, deterministic prefix, and scatter stages preserve handle compatibility
and make block scheduling irrelevant to output order. Phase B therefore
shares primitives with P2.7 but does not include expert histogram or receive
validation work from later priorities.

### P0.3 direct-dispatch payload movement

The record stage now leaves the aligned main body of each hidden payload to a
512-byte AscendC DataCopy loop. The SIMT writer still handles record metadata,
top-k indices and weights, packed scale factors, and any hidden-byte tail. The
epilogue uses the same split in reverse: DataCopy moves aligned hidden bytes,
while the SIMT path handles metadata, scale factors, and tails. Hybrid,
one-block, and grouping-ineligible paths keep their previous implementation.

The first producer implementation divided `(token, destination rank)` pairs
over AIV blocks. That partition covered the right records but did not match
the record VF's grouping-tile partition. One AIV could therefore read a
destination slot while another AIV was still writing it. A local vector sync
and DDR barrier could not close a cross-AIV RAW race. A diagnostic global
`AscendC::SyncAll()` made the representative case pass, confirming the
ownership mismatch, but it is not part of the production solution.

Production payload movement reuses the record VF's physical-AIV mapping. For
launch thread count `N`, subgroup width `W=32`, block index `b`, subgroup
index `s`, and block count `B`:

```text
subgroups_per_block = N / W
first_tile(b, s)    = b * subgroups_per_block + s
tile_stride         = B * subgroups_per_block
token_begin(tile)   = tile * 4
```

With `N=512`, the AIV context for each launch block consumes the same 16
subgroup tile streams that the block's record VF produced. The boundary is:

```text
record VF writes slots and record metadata
  -> asc_sync_vec()
  -> asc_sync_data_barrier(DSB_DDR)
  -> same AIV copies the aligned hidden payload
```

This is a local producer-consumer boundary, not a global core barrier. The
contract test forbids `AscendC::SyncAll()` and checks both the tile mapping and
the sync order.


The BF16 and FP8 focused matrices passed after this ownership correction.
Those tasks establish correctness of the staged DataCopy path; their timing
is not used in the formal H800/Ascend comparison.

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

Performance qualification uses two profiles with different authority:

- `representative` runs one FP8 case on eight ranks as a precheck. It verifies
  the device identity, workload fingerprint, case ID, logical-byte formulas,
  operation order, and timing protocol. It is not a formal performance result.
- `canonical` runs the same 8,192-token, hidden-7,168, top-k-8, 256-expert
  workload for all 144 cases. It records 720 operation rows after 30 warmups
  and 30 measured iterations per operation. Only this profile can support a
  formal H800/Ascend performance statement.

The H800 run creates the workload manifest. The Ascend run consumes that file
byte-for-byte. Comparison fails if the schema, workload fingerprint, ordered
case IDs, operation IDs, logical bytes, rank aggregation, timing counts, or
execution protocol differ. Ascend must report 72 AI Vector blocks and zero
CUDA QPs; CUDA must identify an H800 and retain its own SM/QP selection.

Every canonical row reports H800 and Ascend mean, p50, and p95 device latency,
logical GB/s, `Ascend latency / H800 latency`, and
`Ascend bandwidth / H800 bandwidth`. The comparison is complete only when
all 144 case IDs and 720 operation records are present. A one-case precheck,
smoke run, focused two-rank task, or archived measurement cannot be promoted
to the formal table.

Logical bandwidth is derived from the shared byte formula:

```text
logical_GBps = aggregate_logical_bytes / maximum_rank_latency / 1e9
```

It is not an HBM, HCCL, NVLink, or RDMA hardware-counter measurement. Stage
profiling should still separate grouping, payload movement, routing-weight
restoration, HCOMM submission/service/wait, and launch overhead, but stage
times do not replace the public-operation comparison.

There is no fixed minimum percentage gain for retaining an optimization.
Correctness, unchanged protocol fields, and complete case coverage are hard
gates. Changes within run-to-run variation are reported as neutral. No case
may be dropped, resized, or given a longer watchdog to improve the result.

## Current Qualification State

The subgroup grouping, fused reduction, vector/DataCopy movement, and direct
dispatch count/prefix/scatter path have focused host, compiler, and device
coverage. The formal top-k-8 shape currently uses the dynamic aligned combine
instance; an explicit top-k-8 AOT specialization remains a separate
optimization and must keep the dynamic fallback.

A fresh H800 canonical report and a matching Ascend canonical report are still
required before stating a cross-platform bandwidth or latency ratio. Until
both reports pass the comparison tool, focused task IDs are functionality
evidence only.

## Delivery Order

1. Keep host grouping semantics and checked workspace-layout tests.
2. Keep the vote-ballot ASC compile and device probe.
3. Keep the backend-local match-any-equivalent grouping adapter.
4. Keep token-subgroup execution and owner-lane metadata broadcast.
5. Keep fused grouping/reduction and the reduced workspace ABI.
6. Keep vector/DataCopy payload movement with aligned and scalar-tail coverage.
7. Add and qualify the formal top-k-8, hidden-7,168 AOT path if profiler data
   shows the dynamic instance is material.
8. Run the eight-rank representative precheck on H800 and Ascend.
9. Run canonical on both platforms with the H800 manifest, then compare all
   144 cases and 720 operation records.

Each implementation slice must remain independently diagnosable. A focused
correctness result can qualify code behavior, but only the complete canonical
comparison can qualify performance.
