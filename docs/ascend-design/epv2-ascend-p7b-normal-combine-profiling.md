# EPv2 Ascend P7B Normal Combine Profiling And Optimization

**Status:** tracking document. P7B is the dedicated Normal Combine track and
is independent of the P7A Normal Dispatch barrier track.

## 1. Objective And Scope

P7B uses stage profiling to reduce the latency of the normal, uncached,
direct scale-up Combine path. Combine returns expert-produced records to their
source ranks and assembles the public output tensor. The immediate objective
is to remove avoidable local-copy and reduction work while preserving exact
record ordering, top-k metadata, weights, generation safety, and output
semantics.

The first target is `producer_local_copy` (C3), which dominates the current
Normal Combine device envelope. The P7B path does not change the public Python
or C++ API, logical-byte accounting, HCOMM protocol, or Normal Dispatch. The
source-token pipeline remains disabled.

## 2. Fixed Measurement Contract

All comparisons use the same representative configuration:

```text
world_size=8
num_tokens=8192 per rank
hidden=7168
top-k=8
experts=256
expert_alignment=128
dispatch=FP8
combine=BF16
data_blocks=72
source pipeline=disabled
warmups=30
measured samples=30
seed=0
```

For each iteration, the eight-rank operation duration is the slowest rank:

\[
T_i = \max_{r=0}^{7} T_{i,r}, \qquad
T_{mean}=\frac{1}{N}\sum_i T_i
\]

Logical bandwidth uses the existing logical-byte definition divided by
`T_mean`; it is not an HCCS physical-link counter. A `--profile-stages` run
adds device instrumentation and is used for attribution only. End-to-end
comparisons use a non-profiled binary with the same warmup/sample contract.

## 3. Latest Baseline

The latest shared five-operation profile is:

```text
artifact: /tmp/d3-profile-837c456.json
remote:   /home/pyptouser/yuqitao/p8-d3-profile-837c456/profile.json
task:     task_20260902_123138_234621518356
platform: eight-card Ascend 950 NPU8P
```

### 3.1 End-to-end operation results

| Operation | Mean (ms) | P50 (ms) | P95 (ms) | Logical bandwidth (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| Normal Dispatch | 22.559 | 22.413 | 23.950 | 345.14 |
| **Normal Combine** | **88.676** | **88.549** | **90.892** | **122.93** |
| Expanded Dispatch | 35.066 | 34.737 | 36.319 | 263.83 |
| Cached Dispatch | 84.053 | 83.970 | 85.950 | 92.63 |
| Reduced Combine | 152.126 | 151.978 | 153.571 | 71.66 |

Normal Combine is the P7B target. The other operations are correctness and
shared-resource regression gates.

### 3.2 Normal Combine stage envelope

The profile envelope is `76,802,553` cycles. For a multi-block stage, the span
is the latest block end minus the earliest block start; the report takes the
maximum span across ranks for each stage.

| Stage | Code phase | Cycles | Envelope share |
| --- | --- | ---: | ---: |
| `producer_local_copy` | C3 | 51,385,631 | 66.91% |
| `release_barrier` | C4 | 9,762,167 | 12.71% |
| `producer_record` | C2 | 9,579,636 | 12.47% |
| `epilogue_reduce` | C6 | 4,105,707 | 5.35% |

The largest C3 spans were measured on rank 5 (`51,385,631` cycles), rank 3
(`51,153,187`), and rank 0 (`50,669,273`). This is a broad workload-dependent
copy cost, not only a single-rank barrier outlier.

## 4. Combine Data Flow And Stage Meaning

```text
expert output records
        |
        v
C1 producer control / C2 plan and prefix
        |
        v
C2 producer_record: write return records to per-destination staging shards
        |
        v
C3 producer_local_copy: local staging shard -> local receive shard
        |
        +--> C4 release_payload/control/barrier for remote ranks
        |
        v
C5 acquire + validate contributor slots
        |
        v
C6 epilogue_reduce: combine records by source token
        |
        v
C7 weights -> F0 completion -> public output
```

### 4.1 C3 `producer_local_copy`

`direct_combine_producer_local_copy_vf` handles the local rank's staging
records. It reads the rank count, computes a byte range
`count[world_rank] * combine_record_bytes`, and copies the valid aligned body
from the local staging shard to the local receive shard. The current SIMT body
assigns scalar-tail bytes by `threadIdx.x`; AICore fallback specializations
use `DataCopy` tiles and MTE2/MTE3 event pairs.

This copy exists because the producer writes records into a staging layout,
while the consumer expects a receive-shard layout. It is not the remote HCOMM
payload transfer and it is not the final mathematical weighted reduction.
For the representative case, a large number of local records makes the same
hidden payload traverse GM -> local staging -> local receive before C6 reads
it again.

### 4.2 C4 release

`direct_combine_producer_release_vf` publishes local count/generation metadata,
puts remote staging data into the destination rank's receive shard, flushes
payload, publishes control, and executes the generation barrier. C4 is a
secondary bottleneck in this profile; P7B does not remove its visibility or
reuse guarantees while optimizing C3.

### 4.3 C5/C6 consumer path

C5 acquires and validates contributor slots, including top-k/source metadata.
C6 groups records belonging to an output token and performs the BF16 reduction.
The profile currently assigns `4,105,707` cycles to C6, so reducing C3 must not
merely move the same copy cost into the reduction kernel.

## 5. Root-Cause Hypotheses And Priority

The ranking combines measured stage share, expected critical-path impact, and
the ability to validate one mechanism at a time.

### P7B.0: C3 copy decomposition (P0, measurement)

Split C3 counters into: count/status observation, copy-plan construction,
aligned vector bytes, scalar tail bytes, per-tile MTE2->MTE3 wait, and final
completion. Record bytes, tile size, block count, and local-copy span per rank.
The instrumentation must not change copy order or synchronization.

Required conclusion: determine whether the 66.91% span is dominated by bytes,
tile/event frequency, block imbalance, or waiting for a producer publication.

### P7B.1: Contiguous tile and block-balance candidate (P0)

Using P7B.0 evidence, select one tile size and work partition. Prefer a
contiguous aligned body per block, with a separate bounded tail path. Keep the
existing `simt_combine_local_copy_plan` eligibility checks and avoid reading
past `count[world_rank] * combine_record_bytes`.

Retain only when C3 maximum-rank span and non-profiled Normal Combine mean both
improve in same-binary ABBA runs, with no output mismatch or timeout.

### P7B.2: Remove redundant local staging copy (P0, conditional)

Evaluate whether local records can be written directly to the receive-shard
interval that C5 consumes. This is allowed only when layout, capacity,
generation, and producer/consumer ordering prove identical. Remote records must
retain their existing HCOMM source and destination addresses. If direct local
placement changes an address exposed to a cached handle or violates reuse
barriers, reject it rather than changing the ABI.

The implementation candidate is selected with
`DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT=1` for direct, non-hybrid
Combine. The producer writes records for its own destination rank directly to
the corresponding receive shard, while records for every other destination
continue to use staging. The local-copy stage is skipped only for this
candidate; the producer fence, local control publication, remote puts, flush,
generation tags, and world barrier are unchanged. Unset or `0` retains the
staging copy, and hybrid paths always disable the candidate.

### P7B.3: C3/C4 overlap (P1)

Pipeline independent local-copy tiles with remote payload publication only
after the corresponding records are visible. Use bounded double buffering or
tile groups, retain command generation tags, and preserve the final barrier.
Evidence must show overlapping stage windows, not just reordered launches.

### P7B.4: C6 reduction and metadata locality (P1)

After C3 is reduced, profile C5/C6 by local versus remote records and by output
token occupancy. Consider coalesced metadata reads and contiguous reduction
groups. Preserve deterministic contributor order, BF16 accumulation behavior,
top-k weights, inactive `-1` lanes, and scalar tails.

### P7B.5: Representative acceptance (P0 gate)

After every retained candidate, run the five-operation representative case and
the final 30/30 Normal Combine profile. Report mean, P50, P95, logical bytes,
logical bandwidth, C3/C4/C5/C6 spans, rank skew, and regressions in all four
other operations. No fixed percentage gate is used; correctness and stable
same-binary movement decide retention.

## 6. Correctness And Safety Gates

- clean host contract suite and Ascend extension build;
- two-rank and eight-rank Normal Combine correctness;
- hidden width `7168` and aligned-tail width `7184`;
- duplicate same-rank contributors, inactive `-1` lanes, and zero-count ranks;
- all-local, all-remote, asymmetric-count, and near-capacity records;
- two consecutive generations and repeated buffer reuse;
- no stale generation, CQ failure, timeout, output mismatch, or data race; and
- five-operation smoke run after each retained change.

All NPU8P runs use TaskQueue, with no more than one pending/running task for
this user at a time.

## 7. Implementation Map

- `csrc/backends/ascend/elastic/combine.asc`: C3 local copy, C4 release, C5
  acquire/validation, C6 reduction and AOT/SIMT launch selection;
- `csrc/backends/ascend/elastic/kernels.hpp`: `DirectCombineStage`, profile
  pipeline, and stage-to-profile mapping;
- `tests/ascend/benchmark/timeline.py`: C-stage semantic names and source
  function mapping;
- `tests/ascend/benchmark/runtime.py`: maximum-rank event aggregation and
  logical bandwidth; and
- `csrc/backends/ascend/transport/aicore_transport_service.hpp`: C4 command
  service, CQ drain, generation polling, and barrier accounting.

## 8. Decision Log

| Item | Decision | Reason |
| --- | --- | --- |
| Normal Dispatch | separate P7A track | its current dominant stage is D4 barrier, not C3 |
| C3 direct placement | conditional, P7B.2 | may remove a full GM copy but risks layout/ABI and reuse contracts |
| Extra HCOMM channels | deferred | C3 dominates before transport saturation is established |
| C4 barrier redesign | not duplicated from P7A | keep one owner for barrier experiments and avoid mixed attribution |
| C6 reduction | after C3 | current C6 span is much smaller than C3 |

## 9. Completion Criteria

P7B is complete when the retained Normal Combine path has reproducible C-stage
profiles, all correctness gates pass, rejected candidates are removed, and the
five-operation report is published. Any bandwidth improvement must come from
less critical-path work or measured overlap; changing logical-byte accounting,
weakening synchronization, or hiding a regression does not count.

### P7B.5.1: 8-rank NPU8P acceptance

Commit `3c55a44` was built on the Ascend 950 NPU8P host with system CANN
9.2.0 and run on devices 0-7 through TaskQueue. The representative case was
`ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`, with 8192 tokens per rank,
hidden 7168, top-k 8, 256 experts, 72 data blocks, 30 warmups, and 30 measured
samples. Both baseline and candidate completed the correctness preflight.

The baseline used `DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768` with
direct placement disabled. The candidate enabled
`DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT=1` in the same built binary.

| Operation | Baseline mean (ms) | Candidate mean (ms) | Delta | Candidate logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Normal Dispatch | 23.737 | 22.714 | -4.31% | 353.46 |
| Expanded Dispatch | 35.965 | 36.132 | +0.46% | 261.38 |
| Cached Dispatch | 86.231 | 84.913 | -1.53% | 92.58 |
| **Normal Combine** | **42.238** | **39.066** | **-7.51%** | **284.91** |
| Reduced Combine | 105.897 | 103.175 | -2.57% | 106.51 |

Normal Combine P50/P95 moved from `42.340/43.553 ms` to `39.240/40.000 ms`.
The candidate stage profile reported C3 `1,703` cycles, C4 release payload
`1,698,266` cycles, C4 release barrier `823,407` cycles, and C6 reduction
`4,103,255` cycles. The candidate profile case also passed correctness.

Artifacts are retained on the NPU8P host at
`/tmp/p7b-3c55a44-baseline.json`, `/tmp/p7b-3c55a44-candidate.json`, and
`/tmp/p7b-3c55a44-profile.json`. TaskQueue task IDs are
`task_20260902_173000_40669146384` for the baseline/candidate ABBA and
`task_20260902_220929_247467032012` for the stage profile.

### P7B.4.1: C6 metadata-cache candidate rejected

Commit `c29d002` added an opt-in
`DEEP_EP_ASCEND_COMBINE_METADATA_CACHE` switch that cached each token's
top-k indices and receive slots before contributor-rank selection. The
candidate was tested in the same binary on the same eight-rank case with
`DEEP_EP_ASCEND_COMBINE_DIRECT_LOCAL_PLACEMENT=1`,
`DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768`, 10 warmups, and 10
measured samples. The ABBA order was cache `0 -> 1 -> 1 -> 0`; all four runs
passed correctness. TaskQueue task: `task_20260902_223539_257650028114`.

| Operation | Cache 0 mean (ms) | Cache 1 mean (ms) | Delta |
| --- | ---: | ---: | ---: |
| Normal Dispatch | 22.616 | 21.974 | -2.84% |
| Expanded Dispatch | 34.880 | 33.811 | -3.06% |
| Cached Dispatch | 83.156 | 83.609 | +0.54% |
| **Normal Combine** | **37.701** | **38.452** | **+1.99%** |
| Reduced Combine | 101.046 | 101.541 | +0.49% |

The candidate was rejected because Normal Combine regressed by `1.99%` and
Reduced Combine was slightly slower. No stage-profile follow-up was run after
the negative end-to-end gate. The implementation and selector were reverted
in `bc9c33a`; C6 remains the next optimization target, with reduction
data-movement or scheduling changes requiring a fresh profile first.

### P7B.4.2: C6 vector input double buffering accepted

Commit `1531101` changed the direct vector reduction path to use a two-slot
VECIN queue. While consuming one contributor tile, the next contributor tile
is copied from GM into the other slot; contributor ordering, BF16-to-FP32
accumulation, bias handling, and output conversion are unchanged. The change
was tested on NPU8P with the representative eight-rank case, direct local
placement, and `DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768`.

The stage profile reduced C6 `epilogue_reduce` from `4,103,255` to
`2,144,801 cycles` for Normal Combine and from `4,100,455` to `2,148,412
cycles` for Reduced Combine. Both the 10-warmup/10-iteration smoke run and
the formal 30-warmup/30-iteration run passed correctness. Formal results:

| Operation | 3c55a44 candidate (ms) | 1531101 candidate (ms) | Delta |
| --- | ---: | ---: | ---: |
| Normal Dispatch | 22.028 | 21.227 | -3.64% |
| Expanded Dispatch | 35.395 | 33.973 | -4.02% |
| Cached Dispatch | 84.103 | 83.237 | -1.03% |
| **Normal Combine** | **38.262** | **34.974** | **-8.59%** |
| Reduced Combine | 102.347 | 99.103 | -3.17% |

The formal run used TaskQueue task `task_20260903_065134_396515530654` and
artifact `/home/pyptouser/yuqitao/p7b-1531101-formal.json` on NPU8P. The
double-buffered C6 path is retained as the default implementation for the
existing direct vector-reduction specialization.

Commit `2fdffaa` makes the three previously opt-in direct-combine candidates
default-on when their mode constraints are satisfied: direct local placement,
32768-byte local `DataCopy`, and expanded vector reduction. Explicit `=0`
values still select the conservative fallback paths. The 512-element vector
reduction tile was already default-on.

### P7B.4.3: C2 producer payload double buffering candidate rejected

Commit `1b787b3` pipelines the normal Combine producer's aligned BF16 payload
copies with a two-slot VECIN queue. The next 256-element GM-to-local tile is
issued before the preceding local-to-record tile is written back, while record
addresses, tile order, scalar-tail handling, and the producer/consumer fence
remain unchanged. The host contract suite passes (`160 passed`, `48 subtests`).

An 8-rank NPU8P acceptance task using the representative 30-warmup/30-iteration
workload was submitted as `task_20260903_111039_41193855060`. Compilation
succeeded, but correctness preflight failed before timing: rank 6/7 reported
`58,453,456 / 58,670,080` mismatched elements (`99.6%`). The task exited with
code 1, so no performance or stage-profile result is attributable to this
candidate. The implementation was reverted; the original explicit MTE2/MTE3
event synchronization remains required.

### P7B.4.4: C6 rank-bucket traversal candidate rejected

Commits `91eb509` and `ffdaea9` added the opt-in
`DEEP_EP_ASCEND_COMBINE_METADATA_BUCKETS` path. It reads each token's top-k
metadata once and inserts contributors in rank order, preserving the original
first-seen slot and accumulation ordering. The candidate was tested on the
same NPU8P eight-rank workload with direct local placement and
`DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768`; the ABBA order was
`0 -> 1 -> 1 -> 0`, with 10 warmups and 10 measured samples. All four runs
passed correctness. TaskQueue task: `task_20260902_231624_2909934315`.

| Operation | Buckets 0 mean (ms) | Buckets 1 mean (ms) | Delta |
| --- | ---: | ---: | ---: |
| Normal Dispatch | 22.594 | 23.114 | +2.30% |
| Expanded Dispatch | 34.656 | 34.647 | -0.03% |
| Cached Dispatch | 83.849 | 83.849 | +0.00% |
| **Normal Combine** | **37.889** | **38.143** | **+0.67%** |
| Reduced Combine | 101.386 | 100.867 | -0.51% |

The Normal Combine gate did not improve and the Reduced Combine change was
within run-to-run variation. The candidate was therefore reverted in
`3b9c4d2` and `0fbeb26`; C6 optimization continues with a data-movement
pipeline candidate rather than metadata traversal.

### P7B.4.5: C2 producer payload tile accepted

The retained C2 producer path now uses a producer-specific 1024-element
(`2 KiB`) BF16 tile. The C6 reduction tile remains independently configurable;
the producer tile is applied to both normal payload copies and expanded
producer reduction so that the 7168-element representative hidden dimension
uses seven aligned transfers instead of fourteen. The existing explicit
MTE2/MTE3 event pair is preserved for every transfer.

The candidate was built with system CANN 9.2.0 on Ascend 950 and passed the
two-rank preflight (`task_20260903_123514_12951618963`) and the eight-rank
preflight (`task_20260903_123721_130344931108`). The formal eight-rank run used
the representative case, 8192 tokens per rank, hidden 7168, top-k 8, 256
experts, 72 data blocks, 30 warmups, and 30 measured samples. TaskQueue task:
`task_20260903_123912_132113529816`; artifact:
`/tmp/p7b-1024tile-formal.json`.

| Operation | 1531101 baseline (ms) | 1024-element tile (ms) | Delta | Candidate logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Normal Dispatch | 21.227 | 21.649 | +1.99% | 359.65 |
| Expanded Dispatch | 33.973 | 34.198 | +0.66% | 270.53 |
| Cached Dispatch | 83.237 | 83.479 | +0.29% | 93.27 |
| **Normal Combine** | **34.974** | **28.616** | **-18.18%** | **380.94** |
| Reduced Combine | 99.103 | 30.218 | -69.51% | 360.75 |

All five operations and the case correctness preflight passed. The large
Reduced Combine movement is attributable to the same producer tile removing
most of the fixed per-tile event overhead. The one-sample eight-rank stage
profile (`task_20260903_124419_137228024837`) reports the following critical
segments:

| Operation | Largest stage | Cycles | C2 producer record (cycles) |
| --- | --- | ---: | ---: |
| Normal Combine | `release_barrier` | 7,863,877 | 3,057,646 |
| Reduced Combine | `release_barrier` | 8,500,936 | 3,483,057 |

Thus C2 is no longer the largest end-to-end segment, and C3 local staging is
negligible (`2,364` / `2,176` cycles in the same profile). Further work should
target release-barrier/network overlap rather than increasing the producer
tile again.

A second formal 30-warmup/30-iteration run (`task_20260903_125112_144546914660`)
also passed the case correctness check and measured `29.490 ms` Normal Combine
and `31.189 ms` Reduced Combine (`369.65` / `349.52` logical GB/s). This
recheck confirms the first run is not a one-sample timing artifact.

### P7B.3.1: Deferred final-release flush candidate

The retained 1024-element producer tile moved the Normal Combine critical path
to `release_barrier`. In the unsplit direct release, the producer previously
queued payload puts, explicitly drained them with `flush_payload`, then queued
control publications and the barrier. The barrier service already processes
commands in order, posts its FAA markers after those earlier WQEs, and drains
all peer queues before polling generation counters. This makes the explicit
queue-wide flush a candidate for removal from the final `release_all` path.

The opt-in selector
`DEEP_EP_ASCEND_COMBINE_RELEASE_FLUSH_BARRIER=1` now defers that flush while
leaving payload-only and split profiling stages unchanged. The command order,
control publication, generation tags, final barrier, and CQ drain semantics are
unchanged; hybrid paths and non-direct Combine disable the selector. Host
contract coverage is in place. NPU8P correctness and timing evidence are still
required before considering a default-on change.
