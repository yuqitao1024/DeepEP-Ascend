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

The candidate was removed. Its AICore objects were also larger (`barrier.asc.o`
grew from `480,152` to `491,080` bytes), consistent with a code-generation or
resource-layout cost outweighing the saved address arithmetic. The retained
poll loop therefore keeps the original offset evaluation order.

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
is not accepted from this result alone. The candidate remains uncommitted and
requires the fixed eight-rank workload before any retention decision.

### 7.5 Control-before-flush eight-rank follow-up

A narrower follow-up kept the payload CQ drain before the barrier, but moved
remote control publication ahead of that drain. The earlier 2-rank ABBA screen
was promising but noisy. The fixed eight-rank follow-up
(`task_20260903_161008_24723424169`) passed all five operation records but
rejected the candidate on end-to-end timing: Normal Dispatch was
`22.017331 ms` (`353.635 GB/s`), versus the retained baseline
`19.924784 ms` (`390.774 GB/s`), a `10.5%` regression. The candidate was
removed; no control, signal, CQ, or generation ordering change is retained.

### 7.6 Eight-rank barrier decomposition evidence

The next diagnostic run used the retained D4 tree with `--profile-stages`, 30
warmups, and 30 measured iterations on devices `0-7`. The artifact is
`/home/pyptouser/yuqitao/deepep-results/d4-barrier-diag-8r-30x.json` from
TaskQueue task `task_20260903_175035_195767814868`. Profile mode stores one
diagnostic snapshot per operation, so these values are attribution evidence
and must not be compared directly with unprofiled ABBA operation times.

The implementation was built successfully on NPU8P with CANN 9.2.0 in TaskQueue
task `task_20260903_201945_119174932319`. A single explicit representative case
was then run on two ranks with one warmup and one profiled iteration in task
`task_20260903_202814_154052132579`; correctness passed and the artifact is
`/home/pyptouser/yuqitao/deepep-results/d4-peer-diag-2r-explicit-1x.json`.
For Normal Combine, rank 0's peer 1 became ready about `0.33k` cycles after
its first observation, while rank 1's peer 0 became ready about `5,353,477`
cycles later. The aggregate Combine poll elapsed value was `5,353,477` cycles
(`13,960` iterations), with issue `4,774` and CQ drain `7,666` cycles. This
is the direct per-peer form of the previously observed Normal Combine tail.

Two earlier attempts were intentionally terminated at the hard cap because
the benchmark command omitted `--cases` and selected the entire supported
matrix: task `task_20260903_194150_62090525532` (8 ranks, 30/30) reached 30
minutes with no artifact, and task `task_20260903_201331_99051722089` (2 ranks,
1/1) was stopped after the same all-case mistake. They are not communication
timeouts and must not be used as performance evidence.

The per-rank Normal Dispatch release-barrier values were:

| Rank | Issue | CQ drain | First observation | Poll elapsed | Poll iterations | Barrier span |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53,386 | 42,898 | 1,860 | 5,606,381 | 8,110 | 5,718,635 |
| 1 | 50,897 | 42,324 | 2,145 | 855,901 | 2,052 | 965,510 |
| 2 | 49,932 | 41,527 | 2,137 | 4,062,049 | 6,973 | 4,169,764 |
| 3 | 52,659 | 42,381 | 1,919 | 3,387 | 0 | 114,672 |
| 4 | 48,670 | 40,873 | 2,115 | 2,358,974 | 4,828 | 2,464,123 |
| 5 | 49,583 | 41,554 | 2,087 | 850,487 | 2,070 | 957,075 |
| 6 | 50,018 | 41,334 | 2,184 | 2,069,659 | 4,281 | 2,176,809 |
| 7 | 49,475 | 40,607 | 1,887 | 2,564,248 | 5,005 | 2,670,077 |

The rank maxima/means were `50,578`/`41,687` issue and drain cycles,
`2,042` first-observation cycles, `2,296,386` poll-elapsed cycles, `4,165`
poll iterations, and `2,404,583` barrier-span cycles. The profile stage
maximum was `release_barrier` (`5,718,635` cycles), ahead of
`release_payload` (`1,040,737`) and `release_control` (`274,514`). Thus the
tail is almost entirely generation-counter polling; FAA issue, CQ completion,
and first observation are comparatively small. The slow rank rotates across
runs, so this is not evidence of one permanently bad card.

Normal Combine shows the same release-barrier shape when profiled, although
its current end-to-end envelope is still dominated by `producer_local_copy`.
This makes a shared transport/service-control cause more plausible than an
operation-specific payload-copy cause. The evidence is not sufficient to
exclude producer arrival skew: a peer can be late because its SIMT producer
or AICore service has not published its generation yet. The retained
implementation therefore records per-phase, per-peer FAA issue, CQ drain,
first-observation, ready, and pending-bit clear order in the profile ABI for
the next run.

### 7.7 CUDA comparison and ownership hypothesis

The DeepEP CUDA implementation performs the barrier in GPU execution: after
TMA store commit/wait and the required grid synchronization, a notify warp
polls the GIN signal table directly. Dispatch uses a final barrier with store
flush and start synchronization, while scale-up and scale-out barriers can be
assigned to separate SMs in hybrid mode. There is no equivalent centralized
AICore service queue between the producer and the peer signal poll.

The Ascend path is currently closer to `SIMT producer -> DeviceTransportFacade
command queue -> AICore service -> FAA/CQ drain -> GM generation poll`.
Consequently, a serialized AICore service/control path is a credible source
of the long tail, especially if control commands from multiple producers wait
behind one service queue. The per-peer timestamps are intended to separate
the cases: a late issue points to producer/service scheduling, a fast issue
with late ready points to FAA visibility or control-plane progress, and all
peers becoming late together points to service serialization. No data-plane
or management-plane optimization is retained until this distinction is
measured.

### 7.8 Eight-rank per-peer barrier profile

The v3 per-peer diagnostic ABI was exercised on the fixed eight-rank case in
TaskQueue task `task_20260903_210945_247814231713`. The run used commit
`38a25fb`, one explicit case, 30 warmups, and 30 measured iterations, and
passed all correctness checks. The artifact is
`/home/pyptouser/yuqitao/deepep-results/d4-peer-diag-8r-explicit-30x.json`.
It contains eight observing ranks, two barrier phases, and seven world peers
per rank. Only phase 0 is active for this topology; phase 1 has no records.

The operation-level profile is:

| Operation | Device mean | Issue | CQ drain | First observation | Poll elapsed | Poll iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal Dispatch | 22.298 ms | 54,201 cycles | 46,062 | 2,614 | 3,807,060 | 5,983 |
| Normal Combine | 87.831 ms | 54,047 cycles | 45,086 | 3,062 | 6,936,309 | 7,911 |

The per-peer values below are `ready - first_observation` cycles for Normal
Dispatch; `-` denotes the self peer. They are one diagnostic snapshot, not an
end-to-end timing sample.

| observing rank \\ world peer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | - | 1,813,548 | 273,755 | 475 | 440,578 | 98 | 103 | 93 |
| 1 | 244 | - | 116 | 120 | 110 | 110 | 110 | 111 |
| 2 | 337 | 1,456,738 | - | 98 | 91,406 | 104 | 109 | 100 |
| 3 | 1,901,529 | 3,804,291 | 2,263,460 | - | 2,438,598 | 1,052,249 | 1,858,528 | 1,859,910 |
| 4 | 280 | 1,301,468 | 96 | 99 | - | 95 | 90 | 90 |
| 5 | 792,548 | 2,696,260 | 1,154,685 | 117 | 1,322,948 | - | 749,168 | 752,279 |
| 6 | 573 | 1,897,690 | 355,655 | 104 | 524,189 | 104 | - | 104 |
| 7 | 713 | 1,901,890 | 359,852 | 110 | 528,203 | 110 | 114 | - |

Normal Combine has the same shape with a larger tail: rank 7 observes world
peers 1, 3, 4, and 5 at `6,932,687`, `6,590,512`, `6,467,603`, and
`6,698,919` cycles, while rank 1 observes every peer within 238 cycles. Rank 2
also waits `5,060,147` cycles for peer 1. This reproduces the Combine symptom
with the same barrier implementation.

Issue and CQ-drain components stay around 6--10k cycles per peer; the
multi-million-cycle component begins after first observation. The slow
observer differs by operation and several peers are late together, so this is
not evidence of one permanently bad physical link or a payload data-plane
transfer. It does establish a shared generation/barrier control-path tail.
Producer arrival skew and serialization in the centralized AICore service
queue remain indistinguishable with this snapshot because device cycle
counters are not a cross-rank arrival trace. Therefore no FAA, CQ-drain, or
generation-ordering change is retained yet; the next probe must isolate
producer arrival from service scheduling before modifying the communication
library.

### 7.9 Release-signal readiness cross-check

The follow-up signal-flags smoke used the same fixed eight-rank topology and
one explicit representative case, with one warmup and one measured iteration.
TaskQueue task `task_20260903_215923_258149918648` passed build, correctness,
and profile checks; the artifact is
`/home/pyptouser/yuqitao/deepep-results/d4-peer-diag-8r-signalflags-smoke.json`.
All 560 recorded peer-phase entries (eight ranks, seven remote peers, five
profiled operations, and two profile phases) reported `release_signal_flags=3`.
Bit 0 means the Dispatch or Combine release signal was already at the target
generation at the first barrier poll; bit 1 means it was still ready when the
barrier counter became ready.

This cross-check strengthens the attribution: for the observed generations,
the release signal and its control publication were visible before the
multi-million-cycle barrier-counter tail. The signal result is diagnostic only
and does not prove that every producer arrived at the same instant, so it does
not by itself justify removing the generation barrier. A repeated multi-
iteration profile and a two-generation buffer-reuse correctness run are still
required before testing a barrier-elision candidate.

### 7.10 Signal-only release-barrier candidate

The source now exposes the diagnostic CMake option
`DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY` (default `ON` for Ascend builds). When
disabled, the retained tree re-enables the final direct Dispatch/Combine
producer-release barrier. When enabled, only the
final direct Dispatch/Combine producer-release barrier is skipped. Payload and
control puts, their CQ drains, release-signal publication, the preceding
route-plan barrier, and the service completion check remain unchanged. This
keeps the experiment focused on the suspected duplicate FAA/generation path
instead of changing the data-transfer or route-publication protocol.
The same switch can be passed through the extension build as
`DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=0 python setup.py build_ext --inplace`
to restore the barrier for comparison.

The option is enabled in the retained Ascend build after passing the 8-rank
A/B comparison, consecutive-generation and buffer-reuse checks, and the
five-operation correctness matrix. The build-time switch remains available to
restore the barrier for controlled comparisons.

### 7.11 Post-fix eight-rank stage profile

The signal-only binary was rebuilt with the compile-time definition verified
as `DEEP_EP_ASCEND_RELEASE_SIGNAL_ONLY=1`, then profiled on NPU8P with only
`ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`, five warmups, five measured
iterations, and 72 data blocks. Task `task_20260904_150723_119905128621`
passed and wrote `d4-release-signal-only-profile-8r-v3.json`. An earlier v2
artifact from task `task_20260904_145621_112733215607` is invalid as post-fix
evidence because that binary had no release-signal-only compile definition.

Normal Dispatch measured `19.848 ms` mean device time (`20.823 ms` maximum).
D3 `producer_record` was `0.175M` cycles at the rank maximum, while the D4
`release_barrier` stage collapsed from the earlier multi-million-cycle tail to
only `8.5K` cycles. `release_payload` and `release_control` were `0.486M` and
`0.272M` cycles. All ranks reported the expected diagnostic reason
`completed_service_has_outstanding_requests`, because signal-only completion
does not require the service queue to be fully drained before capture.

| Rank | Device envelope | D3 record | D4 barrier stage |
| ---: | ---: | ---: | ---: |
| 0 | 5,164,737 | 162,687 | 8,445 |
| 1 | 3,434,056 | 162,018 | 8,517 |
| 2 | 2,783,392 | 164,352 | 8,213 |
| 3 | 2,798,282 | 164,592 | 8,484 |
| 4 | 2,595,895 | 172,515 | 8,123 |
| 5 | 5,871,984 | 159,138 | 8,281 |
| 6 | 2,509,514 | 175,321 | 8,232 |
| 7 | 2,566,539 | 171,557 | 8,138 |

The signal-only experiment removes D4 from the visible producer timeline, but
it is not a complete fix: all ranks exit the service with three outstanding SQ
and CQ requests, so the wait is transferred into consumer acquire and the
following generation can reset management state before transport completion.

A follow-up terminal-flush candidate drains the ordered count/generation/signal
WQEs after all peers have been submitted, without restoring the FAA barrier.
Task `task_20260904_154001_1721798688` passed the profiled representative case
with zero SQ/CQ depth on every rank.  A 30-warmup/30-sample run in task
`task_20260904_154752_176897311521` measured `18.743 ms / 415.41 GB/s`, versus
the signal-only run's `21.946 ms / 354.78 GB/s`.  This fixes the false
generation-completion semantic and materially improves Dispatch, but does not
remove the rank tail: rank-mean spread was `1.069 ms`, essentially unchanged
from signal-only's `1.019 ms`.  The remaining investigation is therefore the
single service queue's per-peer submission/progress fairness, not another
barrier-elision change.

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
