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
