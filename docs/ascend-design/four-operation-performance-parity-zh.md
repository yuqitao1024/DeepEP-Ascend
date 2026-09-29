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

#### C4 rejected follow-up (2026-09-30)

A second candidate made the producer-prefix kernel's deterministic tile/rank
transform single-owner: block zero performed the scan while other blocks
retired immediately. It passed correctness, but the four-rank development
gate regressed Normal Combine mean from 10.293 to 11.473 ms and p95 from
10.893 to 16.017 ms. Reduced Combine was unchanged. The likely cause is that
retiring the additional AIV blocks does not compensate for losing the other
blocks' overlap with the subsequent stage. This follow-up is rejected and
reverted together with the grouped C3 probe.

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
