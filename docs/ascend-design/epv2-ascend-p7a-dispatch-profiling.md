# EPv2 Ascend P7A Normal Dispatch Profiling And Barrier Optimization

**Status:** tracking document. P7 is frozen as historical work; all new
Normal Dispatch profiling and optimization work is tracked here.

## 1. Purpose

P7A narrows the next optimization loop using device-stage evidence from the
retained P7 tree. The immediate objective remains improving the normal,
uncached, direct scale-up Dispatch path toward `2000 logical GB/s` on the
fixed eight-rank representative case. This document is separate from the
older P7 plan so that new candidates, measurements, and regressions remain
attributable.

P7A does not reopen rejected P7 barrier designs, does not enable the
source-token pipeline, and does not change the logical-byte formula or public
output contract.

## 2. Fixed Measurement Contract

All comparisons use the same binary configuration and workload:

```text
world_size=8
num_tokens=8192 per rank
hidden=7168
top-k=8
experts=256
expert_alignment=128
dispatch=FP8
data_blocks=72
source pipeline=disabled
warmups=30
measured samples=30
seed=0
```

For each iteration, the operation sample is the maximum device-event duration
across ranks:

\[
T_i = \max_{r=0}^{7} T_{i,r}, \qquad
T_{mean}=\frac{1}{N}\sum_i T_i
\]

Logical bandwidth uses the unchanged eight-rank logical-byte count and
`T_mean`; it is not a physical HCCS counter. Runs with `--profile-stages` add
instrumentation and are for attribution only, not end-to-end regression
comparison with a non-profiled timing run.

## 3. Latest Baseline

Artifact: `/tmp/d3-profile-837c456.json` (remote copy:
`/home/pyptouser/yuqitao/p8-d3-profile-837c456/profile.json`). The task was
`task_20260902_123138_234621518356`, an eight-card Ascend 950 NPU8P run with
30 warmups and 30 measured samples. Build and representative case completed.

### 3.1 End-to-end results

| Operation | Mean (ms) | P50 (ms) | P95 (ms) | Logical bandwidth (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| Normal Dispatch | 22.559 | 22.413 | 23.950 | 345.14 |
| Normal Combine | 88.676 | 88.549 | 90.892 | 122.93 |
| Expanded Dispatch | 35.066 | 34.737 | 36.319 | 263.83 |
| Cached Dispatch | 84.053 | 83.970 | 85.950 | 92.63 |
| Reduced Combine | 152.126 | 151.978 | 153.571 | 71.66 |

Normal Dispatch is the P7A target. The other operations are correctness and
broad-regression gates, not first-slice optimization targets.

### 3.2 Normal Dispatch stage envelope

Stage values are device-cycle spans. A multi-block stage is latest block end
minus earliest block start, and the eight-rank summary takes the maximum per
stage.

| Stage | Cycles | Share of 7,580,641-cycle envelope |
| --- | ---: | ---: |
| `release_barrier` | 3,649,900 | 48.15% |
| `release_payload` | 961,637 | 12.69% |
| `epilogue_copy` | 665,577 | 8.78% |
| `epilogue_validate` | 504,507 | 6.66% |
| `producer_record` | 417,381 | 5.51% |

The largest `release_barrier` span is rank 2. Per-rank spans are:

```text
rank 0: 3,507,548 cycles
rank 1:   109,623 cycles
rank 2: 3,649,900 cycles
rank 3:   700,730 cycles
rank 4: 1,995,177 cycles
rank 5: 2,733,894 cycles
rank 6: 3,031,521 cycles
rank 7: 2,116,117 cycles
```

The operation completes at the slowest rank, so this skew is more important
than the average of the eight rank spans.

### 3.3 Normal Combine context

Normal Combine is recorded to prevent cross-operation conclusions. Its largest
stage is `producer_local_copy` (`51,385,631` cycles, `66.91%` of a
`76,802,553`-cycle envelope), followed by `release_barrier` and
`producer_record`. That bottleneck belongs to a later Combine track and is not
mixed into the first P7A Dispatch change.

## 4. Root-Cause Reading

The retained D3 load-once change reduced `producer_record` to about `0.42M`
cycles in this profile. It is no longer the largest Normal Dispatch owner.
The exposed critical path is:

```text
producer_record -> release_payload -> release_control
                                      -> release_barrier (rank-skewed)
                                      -> epilogue_validate -> epilogue_copy
```

`release_barrier` is a protocol/service stage, not the HCCS payload link by
itself. The implementation issues remote FAA markers, drains their completion,
then polls remote generation counters through the AICore transport service.
`pending_peers` already avoids the old strictly serial peer scan. P7A must
measure a new source of skew rather than repeat historical rejected designs.

## 5. P7A Work Breakdown And Priority

Priority follows measured critical-path share and expected ability to reduce
the slowest-rank tail. Each item is opt-in until it passes the gates.

### P7A.0: Barrier service decomposition (P0)

Add or retain counters around remote FAA issue, CQ drain, first and last
generation-counter observation, polling-loop iterations/`poll_nop` count, and
barrier completion publication. Capture values per rank and peer. This is
measurement-only and must not alter protocol ordering. It must identify whether
rank 2 is delayed by submission, CQ drain, remote visibility, or scheduling.

### P7A.1: Reduce barrier tail without changing generations (P0)

Select one narrowly scoped candidate from P7A.0 evidence. Candidates may
include removing redundant local drains, batching only proven-safe marker work,
or moving a completion observation earlier while preserving generation and
acquire/release rules. No required FAA, visibility operation, or reuse barrier
may be removed without proof and a two-rank test.

Acceptance requires a lower maximum-rank `release_barrier` span and a lower or
neutral end-to-end Normal Dispatch mean in same-binary ABBA runs. A stage-only
improvement with a flat or worse end-to-end result is rejected.

### P7A.2: Payload publication and barrier overlap (P1)

After P7A.1, test whether `release_payload` can be issued in bounded batches
while the service prepares the next control operation. Preserve command order,
generation tagging, and final CQ/error handling. The candidate must expose
actual overlap in stage timestamps; launch reordering alone is insufficient.

### P7A.3: D8 consumer copy (P1)

Measure `epilogue_validate` and `epilogue_copy` separately for local and remote
records. Evaluate larger contiguous copy tiles or a double-buffered schedule,
keeping output layout and validation semantics unchanged. This is secondary
because its current envelope is smaller than the barrier tail.

### P7A.4: Representative acceptance (P0 gate)

After every retained candidate, run the five-operation representative case and
then the final 30-warmup/30-sample Normal Dispatch profile. Report mean, P50,
P95, logical bytes, logical bandwidth, stage envelope, rank skew, and operation
regressions. The stretch target remains `>=2000 GB/s`; no fixed percentage
threshold is used for intermediate retention.

## 6. Required Correctness Gates

- host contract tests and clean Ascend extension build;
- two-rank and eight-rank Normal Dispatch correctness;
- hidden widths `7168` and aligned-tail `7184`;
- duplicate same-rank experts and inactive `-1` top-k lanes;
- all-local, all-remote, asymmetric-count, and two consecutive generations;
- five-operation smoke run for shared buffers and teardown; and
- no timeout, stale generation, CQ error, or output mismatch.

NPU8P execution must use TaskQueue. At most one pending/running task for this
user may exist at a time.

## 7. Decision Log

| Item | Decision | Reason |
| --- | --- | --- |
| Old P7 document | retain unchanged as history | preserves prior candidates and measurements |
| Source-token pipeline | disabled for P7A | cached/source path has unresolved lifecycle behavior and is not required for the current normal path |
| Extra HCCS channels | deferred | transport-only evidence is already much faster than production Dispatch |
| P7-rejected barrier designs | do not repeat | no new evidence; prior candidates regressed or failed protocol gates |
| Normal Combine optimization | separate track | `producer_local_copy` dominates Combine but does not explain Dispatch barrier skew |

## 8. Evidence Map

- `csrc/backends/ascend/elastic/dispatch.asc`: producer release and D8 copy;
- `csrc/backends/ascend/elastic/kernels.hpp`: release segments and stage IDs;
- `csrc/backends/ascend/transport/device_transport_commands.hpp`: barrier,
  flush, and wait commands;
- `csrc/backends/ascend/transport/aicore_transport_service.hpp`: FAA issue,
  CQ drain, generation polling, and barrier completion;
- `tests/ascend/benchmark/runtime.py`: max-rank event aggregation and logical
  bandwidth; and
- `/tmp/d3-profile-837c456.json`: latest stage-profile artifact.

## 9. Completion Criteria

P7A is complete only when the retained Normal Dispatch path has a documented
stage profile, all correctness gates pass, no rejected candidate remains in the
tree, and the final five-operation representative report is reproducible. The
`2000 logical GB/s` target is a performance goal, not a reason to weaken
generation safety, visibility ordering, timeout behavior, or output semantics.
