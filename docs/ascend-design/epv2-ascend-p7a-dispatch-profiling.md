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

### Rejected P7A.0 decomposition candidate

The barrier decomposition probe from `bb2d59e` was removed after qualification.
It added FAA/CQ/poll telemetry and moved invariant peer-mask and offset work out
of the hot loops. The two-rank build and correctness run passed all `144/144`
cases, but the fixed eight-rank workload did not produce a benchmark profile and
hit the TaskQueue timeout during the first case. Restoring only the peer scan or
only the offset evaluation order did not recover the eight-rank run. This rules
out both changes as isolated causes and indicates that the added profile fields
or the resulting AICore service compilation shape changed multi-rank barrier
progress. The complete candidate and its follow-up probes are therefore not
retained; future barrier work must start from the current baseline and change
one execution detail at a time.

### Rejected P7A.3 source-range hint

The Vector epilogue copy was changed to carry its current source-rank range
across monotonically increasing compact records, avoiding repeated GM reads of
the rank prefix table in the common case. The change preserved output and
protocol semantics, built cleanly, passed the host operator contracts, and
passed the fixed two-rank five-operation benchmark
(`task_20260903_154614_196126929443`). However, Normal Dispatch regressed from
`9.667822 ms` to `11.708268 ms` (`21.1%`), while Expanded Dispatch regressed
from `17.968080 ms` to `21.139809 ms` (`17.6%`). The extra loop-carried state
changed the Vector kernel cost more than the saved prefix-table loads, so the
candidate was removed without an eight-rank run.

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
| Immediate first generation poll | rejected | NPU8P 30/30 regressed Normal Dispatch by 7.8%; the fixed poll cadence remains in the retained tree |

### 7.1 Immediate first generation poll experiment

The candidate moved the first generation-counter read ahead of the fixed
`poll_nop`, while retaining the existing FAA submission, CQ drain, generation
comparison, timeout, and completion protocol. The code was evaluated only in
the uncommitted `d4-optimize` worktree and then reverted after qualification.

Both runs used the same NPU8P device pair (`0,1`), CANN 9.2.0, world size 2,
the fixed FP8 representative case (`8192 x 7168`, top-k 8, 256 experts), and
30 warmups plus 30 measured iterations. Each run passed the selected case and
all five operation records were valid.

| Tree | TaskQueue task | Normal Dispatch mean | P50 | P95 | Logical bandwidth |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `task_20260903_061543_38428091754` | `12.139629 ms` | `12.097425 ms` | `13.037351 ms` | `60.452955 GB/s` |
| Candidate | `task_20260903_061254_383508617461` | `13.089253 ms` | `13.104343 ms` | `13.834845 ms` | `56.067095 GB/s` |

The candidate was therefore `7.82%` slower on the end-to-end operation and
was removed. The result supports keeping the established `poll_nop` before
each generation observation; reducing the apparent first-iteration wait does
not reduce the service critical path on this implementation.

The requested eight-rank screening was then run with the same workload on
devices `0-7` and with output redirected to the user's home filesystem (the
shared `/tmp` filesystem was full during the first attempt). Both five-warmup,
five-sample runs passed all five operation records:

| Tree | TaskQueue task | Normal Dispatch mean | P50 | P95 | Logical bandwidth |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `task_20260903_062705_38773525491` | `19.924784 ms` | `19.612476 ms` | `21.406521 ms` | `390.774 GB/s` |
| Candidate | `task_20260903_063058_389576629596` | `22.407781 ms` | `21.961489 ms` | `23.645286 ms` | `347.473 GB/s` |

The eight-rank candidate is `12.46%` slower than its control, so the rejection
is not an artifact of the earlier two-rank screening. No eight-rank performance
claim is made for this candidate, and the retained implementation remains
unchanged.

### 7.2 D4 barrier diagnostics experiment

The D4 barrier profile experiment added lightweight diagnostics in the
existing stage-profile ABI space. With `--profile-stages` enabled, each rank
reported local issue and CQ-drain spans, poll elapsed cycles and iteration
count, participating peer count, first-observation latency, and completion
publication span. The counters were sampled in local registers and written
once per barrier phase, so they did not add a global-memory update to each
poll. The legacy `barrier_poll_cycles` field and validation semantics remained
unchanged; the values were exposed under `service.barrier_diagnostics`.

The first 2-rank smoke with the instrumentation passed all five operations
(`task_20260903_123334_126590311352`). For Normal Dispatch it reported
approximately `2.3k` issue cycles, `4.5k` drain cycles, `1.4k` poll elapsed
cycles, and one peer. This was diagnostic instrumentation only and was not
included in unprofiled performance comparisons.

### 7.3 Rejected poll address linearization

The eight-rank profile showed that the poll tail dominates the barrier span,
so a narrow candidate replaced the per-peer `aicore_barrier_offset()` call in
the hot loop with one phase row base plus a linear peer offset. FAA issue,
CQ drain, peer order, `poll_nop`, generation checks, and timeout behavior were
unchanged. The candidate passed all five operations in the profile run
(`task_20260903_132500_199815527764`), but its unprofiled eight-rank `5/5`
result regressed Normal Dispatch:

| Tree | TaskQueue task | Normal Dispatch mean | P50 | P95 | Logical bandwidth |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `task_20260903_062705_38773525491` | `19.924784 ms` | `19.612476 ms` | `21.406521 ms` | `390.774 GB/s` |
| Row-base candidate | `task_20260903_132929_20220996027` | `21.749820 ms` | `21.528963 ms` | `22.994246 ms` | `357.984 GB/s` |

The candidate was removed. Its AICore objects were also larger
(`barrier.asc.o` grew from `480,152` to `491,080` bytes), consistent with a
code-generation or resource-layout cost outweighing the saved address
arithmetic. The retained poll loop therefore keeps the original offset
evaluation order.

### 7.4 Rejected deferred payload flush

To test payload/barrier overlap, the normal final `release_all` path was
changed so payload puts, control publication, and barrier markers were queued
before the service drained completions. Chunked releases and the profiled
payload-only stage retained their existing flush. The candidate built cleanly
and passed the fixed two-rank FP8 correctness case
(`task_20260903_144232_359127823471`). The eight-rank five-warmup,
five-sample run also passed all five operation records
(`task_20260903_150024_28175731347`), but Normal Dispatch regressed:

| Tree | Normal Dispatch mean | P50 | P95 | Logical bandwidth |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `19.924784 ms` | `19.612476 ms` | `21.406521 ms` | `390.774 GB/s` |
| Deferred flush | `22.120521 ms` | `22.132566 ms` | `22.820036 ms` | `351.985 GB/s` |

The `11.0%` mean regression and `6.6%` P95 regression show that the single
AICore service queue did not overlap payload progress with barrier processing;
it merely moved payload/control completion work into the barrier tail. The
candidate was reverted and no change to generation, signal, CQ, or visibility
ordering was retained.

### 7.5 Control-before-flush screening

A narrower follow-up kept the payload CQ drain before the barrier, but moved
remote control publication ahead of that drain. A serialized 2-rank ABBA
screening (`task_20260903_155647_20293331037`) showed a promising but noisy
signal: the two baseline samples averaged `12.139516 ms`, while the two
candidate samples averaged `10.883337 ms` (`10.35%` faster). Because this is
only two samples per tree and the rank-level variance is material, the change
was not accepted from this result alone.

### 7.6 Control-before-flush eight-rank follow-up

A narrower follow-up kept the payload CQ drain before the barrier, but moved
remote control publication ahead of that drain. The earlier 2-rank ABBA screen
was promising but noisy. The fixed eight-rank follow-up
(`task_20260903_161008_24723424169`) passed all five operation records but
rejected the candidate on end-to-end timing: Normal Dispatch was
`22.017331 ms` (`353.635 GB/s`), versus the retained baseline
`19.924784 ms` (`390.774 GB/s`), a `10.5%` regression. The candidate was
removed; no control, signal, CQ, or generation ordering change is retained.

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

## 10. Rejected Candidate: Barrier Poll Row-Base Hoist

The candidate hoisted the invariant barrier signal-row base address out of the
polling loop in `execute_barrier`, leaving FAA submission, CQ drain, peer scan,
generation checks, timeout behavior, and protocol ordering unchanged. Host
contract tests passed (`170 passed, 48 subtests passed`).

The two-rank smoke result looked favorable, but the representative eight-rank
run is authoritative. TaskQueue tasks
`task_20260903_172051_77202918706`,
`task_20260903_172326_82450820530`,
`task_20260903_172450_83546827469`, and
`task_20260903_172558_84495317230` ran independent 30-warmup/30-sample
benchmarks in the order baseline A, candidate A, candidate B, baseline B. All
four runs passed the representative case correctness gate:

| Run | Result artifact | Dispatch mean (ms) | Dispatch P50 (ms) | Dispatch P95 (ms) | Expanded Dispatch mean (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline A | `p7a-pollbase-ep8-abba/baseline-a.json` | 21.412 | 21.282 | 23.263 | 34.146 |
| candidate A | `p7a-pollbase-ep8-abba/candidate-a.json` | 21.864 | 22.095 | 23.364 | 34.284 |
| candidate B | `p7a-pollbase-ep8-abba/candidate-b.json` | 22.957 | 23.070 | 24.623 | 34.723 |
| baseline B | `p7a-pollbase-ep8-abba/baseline-b.json` | 20.392 | 20.578 | 22.128 | 33.795 |

Pair means are 20.902 ms for baseline and 22.410 ms for the candidate, a
`+7.218%` Normal Dispatch regression. Expanded Dispatch also regressed by
`+1.568%` on pair means. The source change was reverted and is not eligible
for retention based on the two-rank signal.
