# EPv2 Ascend P3 Communication-Compute Overlap Optimization

## 1. Goal And Scope

P3 turns the current synchronous staged-transport execution into a measured,
request-driven pipeline. The work is deliberately ordered:

1. P3.0 measures producer, command publication, AICore service, completion
   wait, consumer, and epilogue work without changing the transport protocol.
2. P3.1 gives `DeviceRequest`, `flush_async()`, and `wait()` a real single-
   channel lifecycle.
3. P3.2 uses two request and workspace slots to overlap adjacent chunks.
4. P3.3 is considered only when P3.0 proves that one SQ/CQ or one service
   drain is saturated. P3.4 and P3.5 remain outside this implementation.

The first production target is the direct scale-up path on one Ascend 950
server. Hybrid and physical scale-out use the same fixed ABI, but they are not
claimed as qualified until their own hardware tests pass.

## 2. Decision

### 2.1 Alternatives considered

Host events around every direct stage would provide accurate elapsed time, but
would add event lifecycle and synchronization to each launch and would not
explain SQ/CQ waits inside the AICore service. Reusing
`DeviceTransportDiagnostic` would keep allocation count small, but would mix
failure state with high-volume measurements and make error handling depend on
profiling. Opening multiple channels first could increase concurrency, but it
has no evidence-based target and can amplify queue and HBM contention.

P3 therefore uses a separately allocated device profile buffer. A null profile
pointer is the disabled representation. Enabled direct kernels record one
start/end pair per launched block, while the AICore transport service records
command, byte, queue-depth, submission, and completion-wait counters. The host
copies the snapshot only after the operation's NPU event has completed.

### 2.2 Why per-block records

A data stage can launch 72 blocks. Timing block 0 alone would under-report a
stage whenever another block starts earlier or finishes later. For stage `s`,
the host derives the device span as

```text
T_s = max_b(end[s,b]) - min_b(start[s,b])
```

over the recorded blocks. The stage sum is diagnostic work accounting, not a
wall-time identity: adjacent resources can overlap and service time is nested
inside producer release. Reports therefore include both raw stage spans and
non-double-counted phase totals.

## 3. P3.0 Profile ABI

`TransportStageProfile` is a fixed, trivially copyable, 64-byte-aligned ABI.
It contains:

- ABI version and exact structure size;
- operation kind and operation generation;
- a valid-stage mask and launched-block count per stage;
- start/end system cycles for up to 72 blocks and 16 direct stages;
- command count, payload bytes, SQ/CQ observed depth and high-water marks;
- AICore service start/end cycles and accumulated CQ wait cycles;
- a completion generation written after the service and consumer finish.

`StagedTransportContext` gains explicit `stage_profile` and
`stage_profile_bytes` fields. Its existing `reserved` field remains the
registration cookie. Changing the layout increments the staged command ABI;
host, SIMT, and AICore validators continue to reject mixed versions.

The profile buffer is allocated only when
`DEEP_EP_ASCEND_PROFILE_STAGES=1`. With profiling disabled, the staged context
publishes a null pointer and device helpers execute one predictable null check.
No profile copy occurs on the operator timing path.

The host-side reset is the only operation that clears the profile. A direct
kernel cannot safely let block 0 clear shared records because Ascend blocks do
not have an ordered start: another block may already have written its start
cycle. Device blocks therefore publish the same operation/generation
idempotently and write only their own block slot. The completion generation is
consumed only after the benchmark synchronizes the NPU stream, so it is a
snapshot generation tag rather than a device-wide block barrier.

## 4. Stage And Phase Accounting

Direct dispatch and combine already launch named producer and epilogue stages.
P3.0 records those exact stage identifiers. The report also groups them into
the following stable phases:

| Phase | Dispatch | Combine |
| --- | --- | --- |
| producer | control/group/prefix/record | control/plan/prefix/record |
| publication | producer release excluding nested service | producer release excluding nested service |
| service submit | validated command execution excluding CQ waits | same |
| CQ wait | service polling/drain cycles | same |
| consumer wait | epilogue acquire/validate | epilogue acquire/validate |
| consumer compute | expert counts, metadata, output copy | reduction and weight output |
| epilogue | completion publication | completion publication |

For nested work, the non-double-counted values are calculated as:

```text
publication = max(0, producer_release - service)
service_submit = max(0, service - cq_wait)
```

The snapshot is accepted only when its operation generation matches the
completed buffer generation. A stale, partial, wrong-ABI, or wrong-size
snapshot is reported as unavailable rather than silently attached to another
operation.

## 5. Host And Benchmark Interfaces

`HostTransport` provides reset/read operations for the profile buffer.
`ElasticBuffer.reset_stage_profile()` establishes a fresh capture boundary;
`ElasticBuffer.get_stage_profile()` returns a plain Python dictionary so the
benchmark does not need to understand the C++ layout. Disabled, stale,
partial, and malformed snapshots return `available=false` with a reason, while
an actual device-copy error remains a runtime failure.

`tests/ascend/benchmark/bench_ep.py --profile-stages` enables allocation before
constructing the buffer. Timed warmups and samples retain their existing NPU
event boundaries. After each operation's timing is complete, the benchmark
runs one synchronized profiling launch and reads the snapshot outside the
timed interval. Benchmark schema version 3 stores a generation-qualified
`stage_profile` object on each operation and records whether profiling was
enabled in `execution_protocol`.

Comparison rejects reports whose profiling protocol differs. Performance
comparison continues to use device event samples; internal system-cycle values
are Ascend-only diagnostic data and are never compared with CUDA kernel-stage
times.

## 6. P3.1 Request Lifecycle

P3.1 keeps one channel and at most one pending request. `DeviceRequest` becomes
a typed 32-byte ABI containing version/state, command range, queue generation,
target consumed count, and terminal error. Its states are empty, pending,
completed, and failed.

`flush_async()` appends the flush boundary, publishes the command range, and
returns pending without draining the service. `wait()` validates ABI and
generation, waits only for the recorded consumed target, propagates the
generation-qualified diagnostic, and makes repeated waits idempotent. Reusing
a pending request, waiting an empty request, stale generation, queue reset, and
timeout all fail closed. `kAsyncCompletion` is advertised only after the device
and production lifecycle probes pass.

## 7. P3.2 Two-Slot Pipeline

P3.2 divides an operation into token/byte chunks and alternates slots 0 and 1:

```text
time --->
slot 0: produce chunk 0 | network chunk 0 | consume chunk 0 | produce chunk 2
slot 1:                   produce chunk 1 | network chunk 1 | consume chunk 1
```

Each slot owns its command range, request, staging region, generation, and
completion signal. Before reusing slot `i`, the producer waits for the previous
request in slot `i`. Tail chunks and zero-token peers follow the same state
machine; they do not select a separate protocol.

The first implementation targets the operation selected by P3.0's measured
overlap ceiling. Dispatch is not automatically first: if combine's reduction
dominates and communication is small, chunking dispatch cannot address the
critical path. Chunk size is swept from a small fixed set and selected by
median representative-case time, subject to correctness and p95 guards.

## 8. Acceptance Gates

P3.0 is accepted when host probes, SIMT/AICore compile probes, two-rank
correctness, and the representative EP8 case pass; disabled profiling changes
each operation by no more than run-to-run noise, and enabled output is complete
and generation-qualified.

P3.1 is accepted when request state, stale/reuse/error/timeout cases and the
production lifecycle pass. It is an enabling change and is not required to
improve isolated latency.

P3.2 is retained only when the representative EP8 case improves without a
correctness or p95 regression. The report must show positive overlap between
different chunks. The complete 144-case matrix remains the functional gate
before integration, even though the authorized eight-device run is limited to
the representative performance case.

The representative case is
`ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`: EP8, 8192 tokens per rank,
hidden 7168, top-k 8, 256 experts, FP8 dispatch, BF16 combine, 72 data blocks,
30 warmups, and 30 measured iterations. All NPU work is submitted serially
through TaskQueue with CANN 9.2, the pinned `hcomm-deepep-current` package, and
the qualified Python 3.10 environment.

## 9. Decision Gate After P3.0

Let `P`, `N`, and `C` be measured producer, transport, and consumer critical
work. The optimistic steady-state ceiling is

```text
speedup_max = (P + N + C) / max(P, N, C)
```

before fill, drain, launch, and contention costs. P3.1/P3.2 proceed when at
least two phases are material and their overlap ceiling exceeds measurement
noise. P3.3 proceeds only when SQ/CQ high-water marks repeatedly approach
capacity or service submission/drain dominates transport time. If reduction
or payload copy dominates alone, the next optimization stays in that compute
stage instead of increasing communication concurrency.

## 10. P3.0 Hardware Qualification (2026-08-24)

P3.0 is accepted for the direct, single-host Ascend 950 scale-up path. This is
an instrumentation qualification only: it does not change the one-channel
transport protocol and does not qualify hybrid or physical scale-out.

### 10.1 Reproducible inputs and execution

The candidate was commit
`f73da24b954be6575eee6be877e5f3845dc1f48c`, archived from a clean tree as
`deepep-p3-f73da24-c28dda33.tar.gz` with SHA-256
`c28dda33c8019e4d40805728a1e1fb7bebdba9f27fa1a8f8ef1dc695a86d3286`.
The remote staging directory was
`/home/pyptouser/yuqitao/deepep-staging/p3-c28dda33c8019e4d`.

All NPU work ran serially through TaskQueue with CANN 9.2,
`hcomm-deepep-current`, and the qualified Python 3.10 virtual environment:

| TaskQueue task | Scope | Terminal result |
| --- | --- | --- |
| `task_20260824_163111_52195411081` | Archive extraction and production/SIMT/AICore builds | exit 0 |
| `task_20260824_163521_5505779125` | Two-rank SIMT/AICore and production correctness on devices 0,1 | exit 0 |
| `task_20260824_164240_58668910244` | EP8 disabled-profile ABBA control | exit 0 |
| `task_20260824_164801_6088375783` | EP8 enabled-profile representative case | exit 0 |

The two-rank transport suite passed `put`, `put-value64`, `faa64`, `signal`,
`signal-set`, `flush`, `payload-signal-order`, `barrier-repeat`, `queue-wrap`,
`phase-boundary`, and `teardown`, all with `diagnostics=kNone`. The production
benchmark passed all five operations. Its result is
`/home/pyptouser/yuqitao/deepep-results/p3-c28dda33-r2/production-correctness.json`
(SHA-256 `2c08868fa799ba306d9244bc1302518ec70b0c10e4e6f053175daff29ad1686a`).

The EP8 command used `torch.distributed.run --standalone --nproc-per-node=8`
with `bench_ep.py --cases ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0
--num-sms 72 --warmups 30 --iterations 30`; the enabled run additionally used
`--profile-stages`. It retained the byte-identical workload manifest
`/home/pyptouser/yuqitao/deepep-results/ascend950-representative-60e3d08/workload.json`
(SHA-256 `98d9dc5ff7b8f31afbc9589b037fd658d99b85b739b8572de1751b9e979eb623`,
fingerprint `d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00`).

The enabled schema-v3 result is
`/home/pyptouser/yuqitao/deepep-results/p3-c28dda33-ep8-profile/benchmark.json`
(SHA-256 `e5b79ad753624523c3433481656096721c3b1f37ac08c5c6e591009a221b6a80`).
The disabled ABBA result directory is
`/home/pyptouser/yuqitao/deepep-results/p3-c28dda33-ep8-ab`; its four JSON
SHA-256 values are recorded in the task report under
`/tmp/deepep-ascend-plans/p3-overlap-sdd/task-4-report.md`.

### 10.2 Disabled-profile control

The control order was baseline A, candidate A, candidate B, baseline B, using
the same representative case and 30 warmups plus 30 samples for every run.
The baseline was commit
`2bd268fe0dd939af3f3668de557cf8df0ca36a83`, archive SHA-256
`5fc0940a373391424243188400e876ffbd0bf56840c1fe4769384b4707b6e983`.
That archive predates schema v3, so its schema-2 output has no `stage_profile`;
this is a disabled-overhead control, not a formal cross-schema comparator.

| Operation | Control mean ms | Candidate mean ms | Delta | Control pair variation | Candidate pair variation |
| --- | ---: | ---: | ---: | ---: | ---: |
| dispatch | 37.727 | 37.665 | -0.164% | 1.328% | 2.507% |
| expanded_dispatch | 38.265 | 38.428 | +0.425% | 0.068% | 0.828% |
| cached_dispatch | 84.198 | 84.683 | +0.576% | 1.798% | 2.005% |
| combine | 140.878 | 140.261 | -0.438% | 0.686% | 0.987% |
| reduced_combine | 168.187 | 167.801 | -0.229% | 0.087% | 0.468% |

Each candidate disabled-profile mean is within its observed candidate
run-to-run variation. P3.0 profiling is retained; no arbitrary overhead
threshold was used.

### 10.3 Enabled EP8 latency and pipeline evidence

The following device-event figures are max-rank operation values. Logical
bandwidth uses the unchanged benchmark formulas.

| Operation | Mean ms | p50 ms | p95 ms | Logical GB/s | Producer cycles/ms | Network cycles/ms | Consumer cycles/ms | Ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 38.178 | 38.076 | 40.124 | 203.939 | 4,472,437 / 4.472 | 6,432,550 / 6.433 | 8,776,951 / 8.777 | 2.242x |
| expanded_dispatch | 38.736 | 38.545 | 40.472 | 238.832 | 4,479,470 / 4.479 | 7,825,052 / 7.825 | 10,296,814 / 10.297 | 2.195x |
| cached_dispatch | 85.021 | 84.915 | 86.774 | 91.579 | 51,029,833 / 51.030 | 10,513,472 / 10.513 | 8,805,904 / 8.806 | 1.379x |
| combine | 140.305 | 140.461 | 141.480 | 77.697 | 47,210,962 / 47.211 | 61,404,108 / 61.404 | 18,000,774 / 18.001 | 2.062x |
| reduced_combine | 167.866 | 167.933 | 170.405 | 64.940 | 73,551,009 / 73.551 | 60,538,982 / 60.539 | 18,001,954 / 18.002 | 2.068x |

The source of cycle conversion is the checked-in Ascend DevKit reference
`/root/aiagent/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md`:
Ascend 950PR/950DT `GetSystemCycle()` is 1 GHz. Thus one cycle is 0.001 us,
`time_us = cycles / 1000`, and `time_ms = cycles / 1000000`. The phase tables
below show cycles with the converted milliseconds; the detailed task report
also retains microseconds for every value.

| Operation | Producer | Publication | Service submit | CQ wait | Consumer wait | Consumer compute | Epilogue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 4,472,437 / 4.472 ms | 162,862 / 0.163 ms | 5,395,936 / 5.396 ms | 873,752 / 0.874 ms | 607,816 / 0.608 ms | 8,165,797 / 8.166 ms | 3,338 / 0.003 ms |
| expanded_dispatch | 4,479,470 / 4.479 ms | 162,620 / 0.163 ms | 6,851,373 / 6.851 ms | 811,059 / 0.811 ms | 607,699 / 0.608 ms | 9,685,972 / 9.686 ms | 3,143 / 0.003 ms |
| cached_dispatch | 51,029,833 / 51.030 ms | 163,211 / 0.163 ms | 9,477,790 / 9.478 ms | 872,471 / 0.872 ms | 658,724 / 0.659 ms | 8,144,130 / 8.144 ms | 3,050 / 0.003 ms |
| combine | 47,210,962 / 47.211 ms | 51,433,595 / 51.434 ms | 8,441,917 / 8.442 ms | 1,528,596 / 1.529 ms | 10,363,481 / 10.363 ms | 7,635,281 / 7.635 ms | 2,012 / 0.002 ms |
| reduced_combine | 73,551,009 / 73.551 ms | 51,671,572 / 51.672 ms | 7,258,696 / 7.259 ms | 1,608,714 / 1.609 ms | 10,359,249 / 10.359 ms | 7,640,691 / 7.641 ms | 2,014 / 0.002 ms |

Raw named-stage spans are retained below as `cycles / us / ms`:

| Operation | Raw stage spans |
| --- | --- |
| dispatch | `producer_control` 7,081 / 7.081 / 0.007; `producer_group` 28,693 / 28.693 / 0.029; `producer_prefix` 569,641 / 569.641 / 0.570; `producer_record` 3,871,441 / 3871.441 / 3.871; `producer_release` 6,341,637 / 6341.637 / 6.342; `epilogue_acquire` 52,152 / 52.152 / 0.052; `epilogue_validate` 503,655 / 503.655 / 0.504; `epilogue_validate_reduce` 52,361 / 52.361 / 0.052; `epilogue_expert_count` 339,414 / 339.414 / 0.339; `epilogue_expert_prefix` 2,773,827 / 2773.827 / 2.774; `epilogue_metadata` 204,786 / 204.786 / 0.205; `epilogue_copy` 4,855,414 / 4855.414 / 4.855; `epilogue_complete` 3,338 / 3.338 / 0.003 |
| expanded_dispatch | `producer_control` 7,703 / 7.703 / 0.008; `producer_group` 28,711 / 28.711 / 0.029; `producer_prefix` 567,933 / 567.933 / 0.568; `producer_record` 3,879,221 / 3879.221 / 3.879; `producer_release` 7,824,497 / 7824.497 / 7.824; `epilogue_acquire` 52,450 / 52.450 / 0.052; `epilogue_validate` 503,650 / 503.650 / 0.504; `epilogue_validate_reduce` 51,848 / 51.848 / 0.052; `epilogue_expert_count` 338,973 / 338.973 / 0.339; `epilogue_expert_prefix` 2,774,644 / 2774.644 / 2.775; `epilogue_metadata` 528,254 / 528.254 / 0.528; `epilogue_copy` 6,051,809 / 6051.809 / 6.052; `epilogue_complete` 3,143 / 3.143 / 0.003 |
| cached_dispatch | `producer_control` 47,160,172 / 47160.172 / 47.160; `producer_group` 28,113 / 28.113 / 0.028; `producer_prefix` 580 / 0.580 / 0.001; `producer_record` 3,858,685 / 3858.685 / 3.859; `producer_release` 10,417,428 / 10417.428 / 10.417; `epilogue_acquire` 52,702 / 52.702 / 0.053; `epilogue_validate` 556,060 / 556.060 / 0.556; `epilogue_validate_reduce` 52,172 / 52.172 / 0.052; `epilogue_expert_count` 338,933 / 338.933 / 0.339; `epilogue_expert_prefix` 2,792,368 / 2792.368 / 2.792; `epilogue_metadata` 167,907 / 167.907 / 0.168; `epilogue_copy` 4,851,407 / 4851.407 / 4.851; `epilogue_complete` 3,050 / 3.050 / 0.003 |
| combine | `producer_control` 9,052 / 9.052 / 0.009; `producer_plan` 976,268 / 976.268 / 0.976; `producer_plan_prefix` 773,675 / 773.675 / 0.774; `producer_record` 45,458,283 / 45458.283 / 45.458; `producer_release` 60,124,965 / 60124.965 / 60.125; `epilogue_acquire` 55,971 / 55.971 / 0.056; `epilogue_validate` 553,488 / 553.488 / 0.553; `epilogue_validate_reduce` 9,754,147 / 9754.147 / 9.754; `epilogue_reduce` 7,612,421 / 7612.421 / 7.612; `epilogue_weights` 24,902 / 24.902 / 0.025; `epilogue_complete` 2,012 / 2.012 / 0.002 |
| reduced_combine | `producer_control` 8,924 / 8.924 / 0.009; `producer_plan` 1,020,054 / 1020.054 / 1.020; `producer_plan_prefix` 773,671 / 773.671 / 0.774; `producer_record` 71,748,449 / 71748.449 / 71.748; `producer_release` 60,163,662 / 60163.662 / 60.164; `epilogue_acquire` 56,137 / 56.137 / 0.056; `epilogue_validate` 550,957 / 550.957 / 0.551; `epilogue_validate_reduce` 9,753,681 / 9753.681 / 9.754; `epilogue_reduce` 7,617,315 / 7617.315 / 7.617; `epilogue_weights` 25,521 / 25.521 / 0.026; `epilogue_complete` 2,014 / 2.014 / 0.002 |

Each operation and rank emitted 30 commands with seven payload puts. The
observed SQ/CQ depths were 0/0 at completion and their high-water marks were
2/2. Dispatch payload bytes ranged from 285,118,320 to 287,399,024 per rank;
combine payload bytes ranged from 543,110,512 to 548,352,112 per rank.

### 10.4 Decision

- P3.1 may proceed after review as the bounded single-channel request
  lifecycle described above; it was not implemented in P3.0.
- P3.2 may proceed after P3.1 review, first for dispatch and expanded dispatch:
  their 2.242x and 2.195x ceilings leave material producer/network/consumer
  overlap opportunity. Cached dispatch is producer-dominated (1.379x ceiling)
  and is not the initial chunking target.
- P3.3 is rejected/deferred. With final SQ/CQ depths 0/0 and high-water marks
  only 2/2, this qualification shows no queue saturation evidence for another
  channel or an additional service drain.
