# Four-operation performance parity plan

## Status

- Date: 2026-09-29
- Scope: Expanded Dispatch, Cached Dispatch, Normal Combine, and Reduced
  Combine on the canonical eight-rank Ascend workload.
- Goal: move the other four operations toward the retained Normal Dispatch
  performance level. Normal Dispatch itself is closed for this phase.
- Measurement authority: no stage profile, no launch-deadline rendezvous, and
  the retained benchmark process identity (one NUMA0 physical core per rank,
  one Torch thread, disabled Python cyclic GC, and at most 64 AIV blocks).

## Baseline and interpretation

The retained no-profile benchmark is the acceptance record:
`task_20260929_143324_94671119821`. All five operations passed. Wall time is
the primary end-to-end metric; Event time is diagnostic only because affinity
changed what the Event interval covers.

| Operation | Wall mean | Wall effective bandwidth |
| --- | ---: | ---: |
| Normal Dispatch | 4.096 ms | 1.901 TB/s |
| Expanded Dispatch | 15.155 ms | 0.610 TB/s |
| Cached Dispatch | 63.010 ms | 0.124 TB/s |
| Normal Combine | 13.407 ms | 0.813 TB/s |
| Reduced Combine | 13.868 ms | 0.786 TB/s |

The old profile report
`.scratch/epilogue-profile/results/epilogue-8192-canonical-profile-20260928-r1.json`
identifies the current device-side centers. Its values are per-rank maxima or
overlapping phase buckets and are not additive:

| Operation | Dominant stage or phase | Secondary center |
| --- | --- | --- |
| Expanded Dispatch | `producer_record` 0.947 ms; `epilogue_copy` 2.101 ms | release payload/CQ wait 0.804 ms |
| Cached Dispatch | about 62 ms wall anomaly not explained by the 1.35 ms service envelope | producer 0.975 ms; epilogue copy 0.319 ms |
| Normal Combine | `producer_record` 3.370 ms; `epilogue_reduce` 2.437 ms | release/CQ wait 1.604 ms |
| Reduced Combine | `producer_record` 3.800 ms; `epilogue_reduce` 2.440 ms | release/CQ wait 1.637 ms |

This is why Normal Dispatch fixes do not transfer as one flat recipe. The host
environment and the transport service are shared; producer and epilogue work
differ by operation and layout.

## Execution model

Take larger, grouped steps, but keep commits separable by optimization item.
A change is allowed into a grouped device validation batch when it has a
specific mechanism, a correctness argument, and a local contract. It is not
necessary to spend three ABBA rounds on every individual item.

1. Implement a small set of high-confidence changes in one source batch.
2. Keep each optimization item in its own commit with a focused contract.
3. Run local contracts after each commit.
4. Build once remotely and run the canonical five-operation correctness and
   performance gate.
5. If a grouped result regresses, use operation-level profile data and
   revert-by-commit rather than debugging a mixed diff.

## Work items

### C1. Explain and remove the Cached Dispatch wall anomaly

Priority: P0. This is the largest absolute gap and is not explained by the
transport service profile.

The profiled service envelope is only about 1.35 ms, while wall time is about
63 ms. Cached Dispatch is the only operation that always forces the comm
stream, creates a predecessor dependency, publishes an async operation, and
waits with `operation->finish(5000)`. The likely center is therefore host or
stream/event retirement rather than the transport kernel itself.

Actions:

1. Add a Cached-Dispatch-only host timeline: descriptor staging, event
   creation, dependency wait, launch, event record, publish, and
   `operation->finish`.
2. Add per-sample wall and Event samples for the operation, plus one
   synchronized profile launch outside timing.
3. If `finish` dominates, test a bounded event-wait/completion path or reuse
   an existing async event without changing public semantics.
4. Preserve descriptor attestation, generation checks, two-generation reuse,
   and error paths.

Acceptance: reduce Cached Dispatch wall mean from roughly 63 ms toward the
single-digit-ms range before comparing with the other operations. Correctness
must pass the canonical case and cached-handle reuse.

#### C1 diagnosis update (2026-09-29)

The dedicated four-rank host/stage profile
task_20260929_162712_290697414768 found the missing attribution. Cached
Dispatch wall time was about 58-61 ms, and cached_dispatch_completion_wait
was 54.5-56.2 ms per rank. Descriptor staging, event creation, publication,
record, and submit were each only about 2-160 microseconds, and the prelaunch
path was 0.5-1.2 ms. The device timeline had about 52.9 ms idle inside the
envelope.

The gap was between the profiled producer-control marker and producer-group:
about 48.5 ms on every rank. That interval belongs to
direct_dispatch_producer_plan, not to host completion or CANN event querying.
The legacy cached planner assigns one thread per destination rank; with 8192
tokens and top-k 8, each thread serially rescans all 65,536 route entries and
repeatedly tests and sets a slot bitmap. This explains the earlier apparent
operation->finish cost: completion was merely waiting for the planner to
finish.

The retained C1 implementation parallelizes only the qualified cached layout
(top-k <= 32 and world size <= 32):

1. Keep the existing serial implementation as an explicit fallback for
   uncached, oversized, and rollback shapes.
2. Use one 32-lane subgroup over grouping tiles, as the uncached grouping path
   already does.
3. Keep invalid-expert, encoded-rank, slot-range, inactive-lane, and
   destination-group slot-consistency checks. An invalid lane reports its own
   producer error even when it cannot participate in a valid destination
   subgroup.
4. Replace the serial bitmap test-and-set with an atomic OR on the same
   per-destination bitmap word, then inspect the returned prior word. A set
   target bit is the existing duplicate-slot protocol error; other bits in the
   same 64-bit word may be set concurrently.
5. Clear each destination's bitmap, count, and maximum-slot accumulator
   before grouping, then aggregate successful groups with global atomic add
   and atomic max. The epilogue's existing maximum_slot == count and
   cached-prefix checks remain unchanged.

This change does not weaken handle validity, generation checks, or
two-generation reuse semantics.

#### C1 development-gate result (2026-09-29)

The first real remote build exposed a SIMT calling-convention constraint:
the `simt_vf` planner entry could not call the old serial fallback marked as
another `simt_vf`. The fallback is now an explicit `simt_callee`, and the
contract test asserts that hierarchy. Task
`task_20260929_230307_176006632758` rebuilt successfully; the extension
SHA-256 is
`9338245e86515949262365c4717672331eea7d08d1eefd4d82a03444e63b4793`.

Task `task_20260929_230610_177603421237` then ran the four-rank development
gate on devices 0-3 with the representative FP8 case, 30 warmups and 30
iterations, 64 AIV blocks, retained host affinity, no stage profile, no
launch-skew profile, and no launch deadline. The case passed correctness.

| Operation | Wall mean | Wall p95 | Interpretation |
| --- | ---: | ---: | --- |
| Cached Dispatch | 26.973 ms | 27.628 ms | improved, but far below the sub-10-ms checkpoint |
| Expanded Dispatch | 11.195 ms | 11.586 ms | gate context only |
| Normal Combine | 10.885 ms | 11.118 ms | gate context only |
| Reduced Combine | 12.362 ms | 12.654 ms | gate context only |

The four-rank Cached Dispatch result improved from the earlier diagnostic
envelope of roughly 58-61 ms to 26.973 ms, confirming that the planner was a
major contributor, but the mechanism did not disappear. The next C1 action is
an operation-level Cached Dispatch profile of the new build to determine
whether the remaining time is still planner work, bitmap atomics, producer
grouping, or the completion path. Do not commit C1 until that attribution and
the final eight-rank gate are complete.

The post-v2 profile task (task_20260929_233143_201896428583) attributed the
remaining rank-0 kernel span almost entirely to the planner: 16.55-16.58 ms in
`direct_dispatch_producer_plan_kernel`, followed by 2.67 ms producer record
and about 2.03 ms transport service. Increasing the producer-plan block count
was then tested as a bounded launch-shape probe. Two data blocks regressed
Cached Dispatch to 128.861 ms mean / 129.921 ms p95, and four blocks regressed
it to 74.051 ms mean / 76.515 ms p95. The launch-shape change is therefore
rejected and must remain reverted; the remaining planner cost needs a
different mechanism that reduces per-token work or atomic traffic without
changing the control-stage topology.

After reverting the launch-shape probe, the four-rank development gate on
devices 4-7 passed correctness and returned 27.525 ms mean / 28.944 ms p95
for Cached Dispatch, matching the retained 26.973 ms / 27.628 ms result within
normal variation. The C1 retained delta is therefore limited to the qualified
cached planner subgroup parallelization and the `simt_callee` fallback fix.
The remaining planner bottleneck is accepted for this commit; a future C1
follow-up must first reduce per-token subgroup work or atomic traffic before
changing launch topology.

#### C7 candidate: cached planner single-round grouping (2026-09-30)

The retained profile still attributes 10.97-16.56 ms per planner launch to
the plan kernel. Most of that per-token work is not the bitmap atomics: the
generic top-k grouping helper iterates once per unique destination and
computes owner ordinals that cached validation never uses.

The first probe incorrectly used one globally active owner and was rejected
by the development gate; the corrected candidate scans only lower active
lanes to find this destination's first owner. For the common first-owner
case the loop exits immediately. It preserves the first-lane owner,
slot-broadcast consistency check, all cached-slot checks, the bitmap duplicate
check, count/max accumulation, and the fallback path. It changes no launch
shape or protocol semantics.

Result: rejected. The first simplified owner selection failed the four-rank
correctness gate. The corrected lower-lane owner scan passed correctness but
returned Cached Dispatch at 26.690 ms mean / 27.166 ms p95, effectively
unchanged from the retained 26.681 / 27.298 ms result; Reduced Combine also
returned to 10.453 / 10.870 ms in that run. Validation records:
task_20260930_015137_281824910211 and
task_20260930_020318_288846122891. The source and focused contract were
reverted; the planner bottleneck is not the generic owner-grouping loop.

#### C10 candidate: duplicate-free cached route fast path (2026-09-30)

The retained four-rank trace shows the cached planner at 10.95-16.55 ms while
the following producer-record stage is only about 2.65 ms. The planner already
performs one bitmap atomic per destination; the additional generic grouping
work—ballots, shuffles, owner selection, and the duplicate-lane mask—is not
needed by a route that individually validates and wins its bitmap slot.

The candidate adds a duplicate-free fast path after per-lane expert/rank/slot
validation. Such a lane immediately performs the same bitmap test-and-set and
the same count/max atomics as the grouped owner, then skips subgroup grouping.
Invalid lanes and lanes that lose the bitmap race fall through to the original
grouped path, preserving slot-broadcast consistency, duplicate detection,
error encoding, and all capacity checks.

Acceptance: reduce Cached Dispatch wall mean/p95 materially without changing
cached-handle reuse semantics or regressing the other four operations. A
correctness failure rejects the candidate immediately.

Result: rejected for correctness. The first fast-path probe let each valid
lane perform its bitmap test-and-set immediately and then skip subgroup
grouping. Remote CANN 9.3.0 compilation initially failed only because a new
use of the destination key was misspelled; after that mechanical fix, build
task task_20260930_022852_29980896977 succeeded. The four-rank development
gate task task_20260930_022941_300480230260 then failed during preparation
with a device invalid-protocol diagnostic on generation 2.

The failure is expected from the mechanism, not from the benchmark: a losing
duplicate lane must remain visible to the grouped slot-broadcast consistency
mask, while the first probe skipped it as soon as it won the bitmap race. The
candidate is therefore rejected and reverted. Any follow-up must preserve a
lane-visibility bit even when the lane wins its bitmap slot; it cannot simply
skip the subgroup path.

#### C10 follow-up direction (2026-09-30)

Cached Dispatch cannot reach the sub-10-ms checkpoint by another owner-grouping
micro-optimization. The retained path is:

1. The cached planner, at 10.95-16.55 ms in the four-rank trace.
2. The producer-record kernel, at about 2.64-2.67 ms.
3. The epilogue-acquire kernel, at 0.04-6.81 ms depending on rank arrival
   overlap.

The next accepted mechanism must remove work from the planner itself. Two
bounded directions remain:

- Preserve lane visibility while removing redundant subgroup operations for
  duplicate-free tokens. A winning lane may publish a single route-presence bit
  and skip later grouping work, but the subgroup mask must still contain every
  active lane so losing duplicate lanes remain visible to consistency checks.
- Reuse the cached handle as stronger evidence: when the attested descriptor,
  input shape, and generation checks pass, the planner's role is validation of
  immutable route/slot relationships. Explore a compact per-token signature or
  summary that proves the full route set without scanning every top-k lane in
  the expensive generic path.

Do not retry a plain launch-topology change. The retained two- and four-block
data probes regressed Cached Dispatch, and the planner is the bottleneck.

#### C11 rejected probe: cached peer-mask specialization (2026-09-30)

A second candidate preserved full lane visibility in the invalid-mask ballot
and selected each destination owner with one active-peer ballot plus one
broadcast. It retained the original owner bitmap test-and-set, duplicate
detection, count/max atomics, and all error encodings; it removed only the
generic helper's unique-owner loop and unused owner-ordinal computation.

The build succeeded after correcting the local lane identifier, but the
four-rank development gate task task_20260930_023505_30318798480 still failed
with the same generation-2 invalid-protocol diagnostic as C10. This narrows
the failure to the two-ballot peer/owner construction itself, most likely
because the active-peer ballot is evaluated with a key-dependent predicate and
produces a different convergence/participation set than the generic helper's
key-independent active ballot on this SIMT dialect. The candidate is rejected
and reverted; do not retry key-dependent ballot masks without first proving
their participation semantics in an isolated probe.

### C2. Expanded Dispatch producer-record acceleration

Priority: P1. Expanded Dispatch has the same shared transport service cost as
Normal Dispatch but performs about 0.95 ms of producer-record work versus
about 0.17 ms for Normal.

Actions:

1. Profile `direct_dispatch_producer_record_vf` by destination rank, expert
   group, and record count.
2. Reuse the retained large-consumer tile strategy only where it is the
   bottleneck; do not blindly increase producer parallelism if record count
   and metadata dependencies differ.
3. Candidate mechanisms are vectorized record packing, fewer per-record
   metadata reads, and expert-prefix-driven range partitioning.
4. Keep the expanded layout and slot semantics unchanged.

Acceptance: reduce `producer_record` materially without regressing Normal
Dispatch. Target the first 1-2 ms of the 11.1 ms wall gap.

### C3. Expanded Dispatch epilogue-copy lane optimization

Priority: P1. The vector epilogue copy is about 2.1 ms after the 8192-byte
tile selector. MTE2 wait is already near zero, so this is a bandwidth or
software-loop bottleneck rather than an arrival gap.

Actions:

1. Add per-tile or per-record timing for lookup, local copy, remote copy, GM
   load, and UB store in Expanded mode.
2. Increase in-flight MTE requests while preserving the retained double-buffer
   safety.
3. Evaluate a wider vector copy or multiple records per MTE request.
4. Do not reopen arrival-driven copy; the prior data ruled out an arrival gap.

Acceptance: reduce Expanded Dispatch epilogue-copy span while keeping output
and metadata correctness. Target another 1-2 ms of the wall gap.

#### C3 rejected experiment (2026-09-30)

The first bounded candidate increased the epilogue-copy in-flight tile count
from two to four. It used four independent MTE free-buffer events and four
equal dynamic-UB slots, while preserving the per-slot event order and all
expanded destination/metadata checks.

The four-rank development gate passed correctness but did not improve the
target operation. Compared with the retained C1+C4 result, Expanded Dispatch
mean changed from 11.167 to 11.189 ms and p95 from 11.358 to 11.406 ms; both
are within normal variation. Cached Dispatch and Normal Combine regressed in
that single gate, so the candidate cannot be attributed a target-specific
gain. The four-buffer version is therefore rejected and reverted.
Validation record: task_20260930_012406_270542915522.

#### C12 candidate: block-contiguous expanded copy scheduling (2026-09-30)

The retained Expanded Dispatch stage profile shows the epilogue-copy stage at
about 3.38M cycles in the tile-8192 probe. Lookup is only about 128k cycles;
local and remote copy issue are about 0.76M and 2.20M cycles. This is a
memory-locality problem, not a coordinate-lookup problem.

The current scalar epilogue copy uses a data-wide grid stride: neighboring
logical records are assigned round-robin across all 64 blocks. The candidate
uses the existing block-distributed grid mapping so each block owns a
contiguous logical interval. This preserves the exact record count, source and
destination mappings, output layout, validation, scale-factor packing, and
weight copy; it changes only which block executes a logical copy.

Acceptance: improve Expanded Dispatch wall mean/p95 materially without
regressing Normal Dispatch, Cached Dispatch, or either Combine variant in the
four-rank development gate.

Result: rejected. The candidate built and passed the four-rank development
gate, but Expanded Dispatch changed from the retained C5 result of 11.204 /
11.467 ms to 11.220 / 11.607 ms; Normal Dispatch, Cached Dispatch, and Reduced
Combine also moved slightly worse. Contiguous block ownership does not improve
the scalar epilogue copy in this case, so the source and contract change are
reverted. Validation task: task_20260930_024005_305465044123.

#### C9 candidate: expanded epilogue source-rank binary search (2026-09-30)

The retained Expanded Dispatch epilogue profile attributes about 321,790
cycles to vector lookup, versus 76,420 cycles for metadata and about 604,655
cycles combined for local and remote payload copy. MTE waits are only 691 and
710 cycles, so the lookup is software work rather than an arrival or copy
queue stall.

The candidate keeps compact-record ownership and all destination/metadata
validation unchanged, but replaces the source-rank linear scan with a binary
search over the monotonic source-bases prefix. With world size four, the
asymptotic loop count falls from up to four to two; the more important
mechanism is removing repeated rank-base/count loads from the common lookup
path. The epilogue output layout, copy scheduling, and buffer/event semantics
are unchanged.

Acceptance: improve Expanded Dispatch wall mean/p95 without regressing the
other four operations in the four-rank development gate. If the lookup saving
does not move end-to-end wall time, reject the candidate rather than claiming
the isolated cycle reduction as a performance result.

Result: rejected. The candidate compiled remotely with CANN 9.3.0 and passed
the four-rank development gate in task
task_20260930_022202_29690912501. Expanded Dispatch changed from the retained
C5 result of 11.204 / 11.467 ms to 11.130 / 11.268 ms, only about
0.074 / 0.199 ms and within normal variation; the other four operations also
remained within normal variation. The isolated lookup cycle saving does not
produce a material end-to-end gain, so the source change is not retained.

### C4. Combine producer-record and release-wait reduction

Priority: P1. Normal and Reduced Combine spend about 3.4-3.8 ms in producer
record and about 1.6-1.7 ms in CQ completion. They share this structure.

Actions:

1. Extend the release/CQ attribution already retained from Normal Dispatch to
   Combine; do not duplicate a second profile mechanism.
2. Profile `direct_combine_producer_record_vf` by source rank and reduction
   kind.
3. Reuse the safe non-inline control-slot strategy and any proven completion
   consolidation where the payload-before-control invariant is identical.
4. Candidate mechanisms are fewer per-record loads, batched producer commands,
   and shared completion boundaries between Normal and Reduced Combine.

Acceptance: improve both Combine variants together. Do not accept a Normal
Combine gain that regresses Reduced Combine, or vice versa.

#### C4 implementation start (2026-09-30)

The retained profile shows Normal Combine spends about 1.13 ms in
`direct_combine_producer_plan_kernel` and 0.76 ms in
`direct_combine_producer_plan_prefix_kernel`. The plan assigned one thread
to each 128-row tile and that thread scanned the tile once per destination
rank; the prefix kernel used only thread zero and scanned every tile once per
rank.

The first C4 implementation changes only planning parallelism:

1. Use the existing 32-lane subgroup decomposition for producer planning. One
   lane handles one destination rank for each tile, so the metadata scan and
   validation work is partitioned by destination instead of repeated by one
   thread.
2. Keep source identity, encoded source rank, master lane, expanded input-row
   validation, tile counts, and tile error encoding unchanged.
3. Clear tile errors once by lane zero, synchronize the block, and let each
   destination lane publish its own tile count or error.
4. Parallelize the prefix scan by rank: thread `rank` computes that rank's
   tile-prefix scan and final count. Error discovery and capacity checks
   remain unchanged.

This intentionally does not change producer records, transport commands,
release order, epilogue reduction, or output layouts.

The grouped C1+C4 four-rank development gate on devices 4-7 passed
correctness with the representative FP8 case, 30 warmups and 30 iterations,
64 AIV blocks, retained host affinity, no stage profile, no launch-skew
profile, and no launch deadline. Compared with the C1-only revert run:

| Operation | Before mean / p95 | After mean / p95 | Change |
| --- | ---: | ---: | ---: |
| Cached Dispatch | 27.525 / 28.944 ms | 26.681 / 27.298 ms | -0.844 / -1.646 ms |
| Normal Combine | 11.389 / 12.207 ms | 10.293 / 10.893 ms | -1.096 / -1.314 ms |
| Reduced Combine | 12.402 / 12.728 ms | 11.761 / 12.051 ms | -0.641 / -0.677 ms |

Normal Dispatch and Expanded Dispatch remain within normal variation. The
planning/prefix change therefore has a clear four-rank benefit for both
Combine variants and does not regress the other three operations.

#### C4 follow-up candidate: parallel combine error scan (2026-09-30)

The retained producer-prefix kernel still scans every tile error on one
thread before the rank-specific prefix pass begins. The candidate distributes
tile-error inspection by rank while preserving the first-error peer mapping,
error encoding, capacity checks, and per-rank prefix construction. This is a
bounded software change with no launch-shape, workspace, or protocol change.

Acceptance: improve Normal and Reduced Combine without regressing Dispatch.

Result: rejected. The candidate compiled and passed the four-rank development
gate, but Normal Combine regressed from 10.293 / 10.893 ms to 10.407 /
11.574 ms while Reduced Combine improved only slightly to 9.921 / 10.185 ms.
The single-thread error scan is not the bottleneck in the normal path, so the
source and focused contract were reverted. Validation record:
task_20260930_021006_29189185911.

#### C4 rejected follow-up (2026-09-30)

A second candidate made the producer-prefix kernel's deterministic tile/rank
transform single-owner: block zero performed the scan while other blocks
retired immediately. It passed correctness, but the four-rank development
gate regressed Normal Combine mean from 10.293 to 11.473 ms and p95 from
10.893 to 16.017 ms. Reduced Combine was unchanged. The likely cause is that
retiring the additional AIV blocks does not compensate for losing the other
blocks' overlap with the subsequent stage. This follow-up is rejected and
reverted together with the grouped C3 probe.
Validation record: task_20260930_012406_270542915522.

### C5. Combine epilogue-reduce acceleration

Priority: P2. The reduce stage is about 2.44 ms in both Combine variants.

Actions:

1. Profile reduce by input row, local/remote source, top-k kind, and bias
   path.
2. Evaluate wider vector accumulation and multiple-row MTE requests.
3. Keep bias and multi-reduction semantics unchanged.

Acceptance: reduce the common 2.44 ms reduce span by at least one millisecond
without changing numerical-order guarantees beyond the existing explicit
allow-multiple-reduction setting.

#### C5 candidate (2026-09-30)

The retained Reduced/Expanded Combine vector producer already uses the
vector pipeline, but its VECIN queue holds only one tile. Each input tile is
therefore copied, consumed, and freed before the next GM load is issued,
leaving the MTE pipeline idle during Cast and Add.

The bounded candidate increases the VECIN queue depth from one to two equal
tiles and changes no addresses, layouts, reduction lane order, or numerical
accumulation order. It only lets the next GM load overlap with the current
Cast and Add. This affects the qualified Reduced/Expanded Combine producer
vector path; Normal Combine's plain payload copy is unchanged.

Acceptance: improve Reduced Combine materially without regressing Normal
Combine or the three Dispatch operations.

#### C5 result (2026-09-30)

The candidate compiled remotely with CANN 9.3.0 and the extension SHA-256 is
`edd8b9d2bf7c12da35523bee8affeda3804c619a211e06fabe761efaadd1e8d9`. The
four-rank development gate task
`task_20260930_014401_278552721876` passed correctness with the retained
benchmark identity.

| Operation | Before mean / p95 | Candidate mean / p95 | Change |
| --- | ---: | ---: | ---: |
| Reduced Combine | 11.761 / 12.051 ms | 10.007 / 10.311 ms | -1.754 / -1.740 ms |
| Normal Combine | 10.293 / 10.893 ms | 10.315 / 10.876 ms | +0.022 / -0.017 ms |
| Expanded Dispatch | 11.167 / 11.358 ms | 11.204 / 11.467 ms | +0.037 / +0.109 ms |
| Cached Dispatch | 26.681 / 27.298 ms | 26.810 / 27.289 ms | +0.129 / -0.009 ms |
| Normal Dispatch | 3.551 / 3.905 ms | 3.469 / 3.577 ms | -0.082 / -0.328 ms |

Reduced Combine improved materially while the other four operations stayed
within normal variation. The change is retained.

#### C13 rejected probe: normal producer queue-based payload copy (2026-09-30)

A follow-up considered converting Normal Combine's plain GM-to-UB-to-GM
payload copy from a `VECCALC` buffer to the depth-two `VECIN` queue used by
the retained Reduced Combine producer. The intent was to overlap the next GM
load with the current GM store.

This reuse is not safe without a new hardware lifetime proof. In Reduced
Combine, the queued input is consumed by Cast/Add and can be freed immediately
after the vector reads. In Normal Combine, the same UB tensor is also the read
source for MTE3; returning it through `FreeTensor` before an explicit MTE3
completion boundary can recycle the tile while the GM store is still in
flight. The original explicit `MTE2_MTE3` and `MTE3_MTE2` flags are the
conservative lifetime mechanism.

The half-finished queue conversion was therefore reverted without a remote
run. A future candidate must first establish a queue or event lifetime that
covers the MTE3 read, rather than weakening that boundary for pipeline form.

#### C16 rejected probe: skip inactive top-k grouping reductions (2026-09-30)

A bounded candidate returned immediately when the subgroup active ballot was
empty. It preserved the zero group mask and no-owner result that inactive
lanes previously obtained, while avoiding shuffles and ballots for fully
inactive token subgroups.

The remote build task task_20260930_025906_313107120391 succeeded with
extension SHA-256
2593bd7f463e107db1c04911dd3adf56151cb95d367481f8696622af65d0f105. The
expanded top-k grouping probe passed its all-inactive fixture, and the
four-rank gate task task_20260930_030032_31442517993 passed correctness:

| Operation | Retained mean / p95 | Candidate mean / p95 | Change |
| --- | ---: | ---: | ---: |
| Cached Dispatch | 26.810 / 27.289 ms | 26.981 / 27.530 ms | +0.171 / +0.241 ms |
| Normal Dispatch | 3.469 / 3.577 ms | 3.680 / 4.005 ms | +0.211 / +0.428 ms |
| Expanded Dispatch | 11.204 / 11.467 ms | 11.123 / 11.258 ms | -0.081 / -0.209 ms |
| Normal Combine | 10.315 / 10.876 ms | 10.409 / 10.918 ms | +0.094 / +0.042 ms |
| Reduced Combine | 10.007 / 10.311 ms | 10.054 / 10.524 ms | +0.047 / +0.213 ms |

The representative route has essentially no fully inactive token subgroup, so
the early return does not reduce active-lane grouping work. The source and
probe fixture were reverted in 6622835; future Cached Dispatch work must
reduce active-token planner work or replace validation with stronger cached
handle evidence.

#### C17 candidate: cached route plan digest (2026-09-30)

Priority: P0. Cached Dispatch remains about 27 ms while the post-C1 profile
still attributes 11-16 ms to direct_dispatch_producer_plan_kernel. The
handle currently proves descriptor identity, but every cached launch still
rebuilds and validates the entire outbound route plan from topk_indices and
destination_slots.

The candidate adds a device-resident cached route plan:

1. Store outbound per-destination counts and a 64-bit route digest beside the
   cached handle. This is internal handle state; it does not change the
   descriptor ABI, Python public EPHandle shape, or descriptor fingerprint.
2. On the initial non-cached dispatch, the planner computes the same
   destination counts and a commutative digest over the observed top-k and cached slot
   values.
3. On cached dispatch, a bounded device preflight recomputes that digest and
   restores the stored counts. A digest match is not treated as mathematical
   proof; it is an additional strong identity check layered over the existing
   descriptor, generation, topology, shape, and mode checks.
4. Only after a digest match may the launch skip the full planner. A mismatch
   reports InvalidCachedMetadata and falls back to the existing planner.
5. The preflight must also validate stored count bounds. maximum_slots is
   restored from the existing invariant that, for a valid cached route, the
   maximum slot equals the outbound destination count.

Acceptance: reduce Cached Dispatch wall mean and p95 materially without
regressing the other four operations in the four-rank development gate.
Correctness must exercise handle reuse and mutation rejection. If the digest
preflight is not cheaper than the remaining planner, revert it and record the
result here.

Implementation status (2026-09-30): the route plan is represented by a compact
`[num_ranks + 2]` uint64 tensor: per-destination counts, a 64-bit commutative
digest, and one reserved word. A dedicated small VF originally populated it
after Normal Dispatch; that probe exposed the launcher ABI rule that this CANN
build silently drops a non-struct kernel argument list. The VF now uses the
same packed `KernelArguments` ABI as the production kernels. Cached Dispatch
passes the tensor into the producer-plan VF. After the digest, per-rank counts,
and source-slot checks pass, it restores the cached destination counts and
maximum-slot values while the full planner is skipped.

The first fast-path validation used top-k lane counts, which over-counted a
token routed to multiple local experts on the same destination. Both producer
and cached paths now count a token once per destination. The host also validates
counts against `capacity * num_ranks`; the earlier `capacity`-only check
incorrectly rejected valid balanced four-rank traffic.

Correctness and performance were measured on devices 0-3 with the
representative FP8 8,192-token, hidden 7,168, top-k 8, 256-expert case. The
final build task task_20260930_160914_30880696115 produced extension SHA-256
64b09e5a82c1bda6cd27311048e78d5ebd2ee2d373fe849feaf6a36a5809baaa, and gate
task task_20260930_161020_309766612030 passed. Compared with clean head
aebb8ff, Cached Dispatch improved from 26.16 / 27.30 ms wall mean / p95 to
10.70 / 11.21 ms (about 2.45x, 103.65 to 260.10 GB/s logical). Normal Combine
and Reduced Combine stayed within noise. A fused Normal Dispatch probe was
also tried; it left host wall time at 43-52 ms, so it was rejected and the
dedicated small VF remains the implementation.

Known regression and follow-up: the dedicated route-plan VF adds an extra scan
to Normal Dispatch. In the final four-rank gate, Normal Dispatch moved from
3.523 / 3.903 ms to 4.237 / 5.435 ms wall mean / p95, and Expanded Dispatch was
also affected. A bounded probe that disabled the artifact for Expanded Dispatch
failed the cached-expanded correctness path because that handle also requires
the route plan. The accepted follow-up is lazy generation: uncached Normal and
Expanded Dispatch no longer launch the route-plan VF. The first cached call
that sees a zero digest launches the VF before the producer-plan fast path,
synchronizes once, and verifies the digest; subsequent cached calls consume the
stored artifact. This removes the extra scan from the uncached operations while
preserving cached correctness.

Lazy-build validation update (2026-10-08): the first build, task
task_20261008_081038_160423029781, and gate, task_20261008_081916_164526314061,
reproduced the expected initialization hazard. The uncached route-plan tensor
was allocated with `torch::empty`, so a nonzero garbage count could be rejected
before the lazy builder ran. The retained correction treats a zero digest as
the sole "not built" sentinel, skips count validation in that state, and
allocates the small route-plan artifact with `torch::zeros`. Build task
task_20261008_082440_168118231872 produced extension SHA-256
cb3c6bc188f00d3cf69eb31f7ff01d18d28dafd2592ac97a5ef0ee8bc099a43e.

The corrected binary has not completed the four-rank gate yet. Task
task_20261008_103823_142641017894 stopped at the standalone topk-grouping probe
with exit 139; rerunning that one-card probe as task
task_20261008_105106_6718223360 passed, so the crash was treated as transient
host/device instability rather than a route-plan failure. Gate task
task_20261008_105304_9824125158 was then queued, but the host became
SSH-unreachable before its result could be observed. Local route-plan and
Python API tests passed. Re-run the four-rank gate on this exact build when the
environment is healthy; do not claim the Normal/Expanded Dispatch recovery or
the preserved Cached Dispatch gain until that gate reports metrics.

Lazy-build acceptance (2026-10-09): the reinstalled NPU8P host now runs CANN
9.3.0. The original corrected build reached the gate, but CANN rejected the
small route-plan artifact at initialization because this runtime does not
implement aclnnInplaceZero for uint64. The retained compatibility fix allocates
equally-sized int64 storage, zeroes it, and exposes a uint64 view before the
artifact enters the existing handle ABI. Build task
task_20261009_090056_18422638577 succeeded and produced extension SHA-256
ef2ad99d99d76a034067438bde8b0d938314370a99031c897e994c2f0e247231.

The first retried gate failed for an environment reason unrelated to the
route-plan change: with ASCEND_RT_VISIBLE_DEVICES=0,1,2,3, HCOMM topology
queries for physical devices 4-7 returned CANN 107001 (device-id mapping). A
four-process HCCL all-reduce on devices 0-3 passed, confirming that the devices
and communicator were healthy. The accepted retry let task-submit lock devices
0-3 without exporting ASCEND_RT_VISIBLE_DEVICES; gate task
task_20261009_090353_184977523962 then passed the representative four-rank
case:

| Operation | Before lazy-build mean / p95 | Lazy-build mean / p95 |
| --- | ---: | ---: |
| Normal Dispatch | 4.237 / 5.435 ms | 4.028 / 4.331 ms |
| Expanded Dispatch | affected by route-plan scan | 11.448 / 11.839 ms |
| Cached Dispatch | 10.700 / 11.210 ms | 10.657 / 11.083 ms |
| Normal Combine | within noise | 10.328 / 10.861 ms |
| Reduced Combine | within noise | 9.982 / 10.331 ms |

The gate preserves the roughly 10.7 ms Cached Dispatch gain, recovers Normal
Dispatch to near its pre-C17 result, and leaves the other three operations in
their expected ranges. A separate two-rank correctness matrix, task
task_20261009_090524_185416223382, passed all 14 cases, including cached reuse,
near-capacity cached traffic, 100 sequential generations, and round-trip
behavior. C17 is accepted; the next work item should be selected from the
remaining operations using a fresh stage profile rather than re-opening Normal
Dispatch micro-optimization.

#### C18 prerequisite: shared pinned handle readback (2026-10-09)

Before opening the dispatch-stage-gap and Combine producer work items, the
retained Normal Dispatch host-transfer optimization was audited. Its reusable
4096-byte pinned staging path was present in the runtime, but Cached Dispatch
and Combine still used synchronous `copy_to_host` for descriptor, route-plan,
prefix, unaligned-count, and bounded hybrid-route readbacks.

The prerequisite change routes those bounded reads through the existing
`SmallHostTransfer` stream path. No validation, generation, topology, shape,
or handle-attestation check is removed. The first lazy-build call still
synchronizes through the same stream-ordered copy after building its route
plan; only the redundant separate stream synchronize is removed.

The candidate built as task task_20261009_095354_204461512397 with extension
SHA-256 2a816b333dc0688d578cca762c3541f53997820a0f19af150bd4925396c2aa5e.
The four-rank gate task task_20261009_095445_204812330959 passed:

| Operation | Before mean / p95 | Candidate mean / p95 | Change |
| --- | ---: | ---: | ---: |
| Normal Dispatch | 4.028 / 4.331 ms | 3.475 / 3.660 ms | -0.552 / -0.671 ms |
| Expanded Dispatch | 11.448 / 11.839 ms | 11.166 / 11.284 ms | -0.283 / -0.556 ms |
| Cached Dispatch | 10.657 / 11.083 ms | 9.651 / 10.388 ms | -1.006 / -0.696 ms |
| Normal Combine | 10.328 / 10.861 ms | 9.911 / 10.198 ms | -0.417 / -0.664 ms |
| Reduced Combine | 9.982 / 10.331 ms | 9.751 / 10.189 ms | -0.231 / -0.142 ms |

The follow-up stage profile task task_20261009_095623_205841315789 passed and
confirmed the mechanism: Cached Dispatch prelaunch setup fell from about 2.455
to 0.680 ms, while Combine handle readback fell from about 0.255-0.270 to
0.020 ms. The two-rank dispatch matrix task
task_20261009_095552_205494510247 passed all 14 cases. This prerequisite is
accepted because it applies an already proven host mechanism to the four
remaining operations without weakening any handle proof.

#### C14 candidate: single-pass combine contributor lookup (2026-09-30)

The retained vector epilogue resolves each output token by scanning all
top-k lanes once for every contributor rank. The candidate instead scans the
top-k lanes once per token into a small rank-to-receive-slot map, then emits
contributors in ascending rank order. This changes only lookup scheduling.

The first lane that maps to a source rank is retained, invalid experts are
ignored, and the emitted contributor order remains ascending rank order. The
reduction order, bias order, tile order, queue depths, and output layout are
unchanged.

Acceptance: improve Normal and Reduced Combine wall mean/p95 materially
without regressing the three Dispatch operations in the four-rank development
gate. An isolated cycle reduction without end-to-end improvement is not
sufficient.

Result: no end-to-end benefit in the current build. The remote build task
task_20260930_025327_31016547476 succeeded with extension SHA-256
89742936667d09dcb67dbb906c078051cd9b2b62c44e586b8c34b44332ea145a. The
four-rank gate task task_20260930_025438_311345818326 passed correctness for
both runs. Relative to the retained C5 baseline:

| Operation | Retained mean / p95 | C14 default mean / p95 | 1024 tile mean / p95 |
| --- | ---: | ---: | ---: |
| Normal Combine | 10.315 / 10.876 ms | 10.322 / 10.710 ms | 11.820 / 12.437 ms |
| Reduced Combine | 10.007 / 10.311 ms | 10.078 / 10.496 ms | 11.371 / 11.688 ms |
| Normal Dispatch | 3.469 / 3.577 ms | 3.513 / 3.729 ms | 3.639 / 3.966 ms |
| Expanded Dispatch | 11.204 / 11.467 ms | 11.208 / 11.407 ms | 11.161 / 11.374 ms |
| Cached Dispatch | 26.810 / 27.289 ms | 27.006 / 27.410 ms | 26.923 / 27.840 ms |

C14 leaves the default path within normal variation but does not produce a
material end-to-end gain. C15's explicit 1024 tile regresses both Combine
variants, so the larger reduction tile is rejected as a default or recommended
setting. The safe selector extension can remain available for diagnosis, but
the production default stays 512 and performance work should continue with a
different mechanism.

#### C15 candidate: 1024-element combine reduction tile (2026-09-30)

The retained epilogue reduction uses a 512-element BF16 tile. The producer
already uses a separate 1024-element tile and has a much lower payload-copy
span. The bounded candidate extends the existing selector and template
fallback to accept an explicit 1024-element reduction tile, reducing the
representative 7168-element hidden dimension from fourteen to seven vector
iterations.

The default remains 512. `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE=0`
still selects the conservative fallback, and no launch shape, protocol,
buffer depth, or numerical order changes. The tile must remain aligned with
the existing DataCopy alignment qualification.

Acceptance: with the explicit 1024 setting, improve Normal and Reduced
Combine materially without regressing the three Dispatch operations. If the
gain is absent, keep the selector change only when it does not affect the
default run and continue with the next mechanism.

### C6. Grouped validation gate

Priority: P0, required after every grouped source batch.

Validation has two levels during this phase.

#### Development gate: four ranks

During implementation and quick iteration, run the same representative FP8
case with four ranks on devices 0-3. Use the retained benchmark defaults, no
stage profile, no launch deadline, and at least 30 warmups and 30 iterations.
This gate checks correctness and large, obvious performance changes while
avoiding long eight-card queue waits. Its latency and bandwidth are not the
final target and must not replace the eight-rank record.

A four-rank result is sufficient to advance a work item when the mechanism is
local to the operation path (for example, a host completion wait or a producer
or epilogue kernel) and the direction is large and unambiguous. If the
mechanism depends on cross-rank arrival, CQ completion scaling, or HCCS
topology, use an eight-rank diagnostic instead.

#### Final gate: eight ranks

Build and run on NPU8P through TaskQueue only. Use devices 0-7, the canonical
FP8 case, 30 warmups and 30 iterations, and the retained benchmark defaults.

1. Correctness gate: all five operations pass.
2. Primary performance gate: wall mean and p95 for each operation.
3. Diagnostic gate: Event mean is reported but not used as end-to-end
   throughput.
4. Regression rule: Normal Dispatch wall must not regress materially; any
   regression must be assigned to a commit and reverted or reworked.
5. Teardown-only SIGSEGV is acceptable only after a complete report and a
   passing case summary, as established in prior records.

## Explicitly non-goals

- Do not re-open Normal Dispatch micro-optimization.
- Do not use the 2-ms launch deadline as a production performance mechanism.
- Do not use affinity-enabled Event bandwidth as end-to-end throughput.
- Do not implement a speculative protocol change without operation-level
  attribution.
- Do not optimize Cached Dispatch by weakening handle validity or generation
  checks.

## Target checkpoint

The immediate engineering target is not an equal bandwidth number across all
operations. Their logical byte formulas differ. The first checkpoint is:

1. Cached Dispatch wall mean below 10 ms.
2. Expanded Dispatch wall mean below 12 ms.
3. Normal and Reduced Combine wall means below 12 ms.
4. Normal Dispatch remains near the retained 4.1 ms wall result.

If C1 confirms that most of Cached Dispatch is host/event overhead, reaching
the first checkpoint should be substantially easier than optimizing its
transport path. After that checkpoint, targets should be recalculated from
each operation's logical bytes and its required wall time at a 3 TB/s-class
effective rate.
