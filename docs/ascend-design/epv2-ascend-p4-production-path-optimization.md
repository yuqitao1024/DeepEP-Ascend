# EPv2 Ascend P4 Production-Path Optimization

## 1. Goal

P4 targets the software work around HCCS transport after the P0-P3 kernel and
pipeline changes. The Ascend 950 EP8 transport probe reaches `2535.046 GB/s`
for the representative remote-byte matrix, while full dispatch reaches
`208.715 logical GB/s`. These bandwidth values use different byte formulas and
cannot be divided to obtain hardware efficiency. They do show that large
contiguous HCOMM puts and eight-rank concurrency are not the first bottleneck
to optimize.

P4 keeps the existing transport protocol and correctness contract until
profiling identifies a specific cost. Each retained optimization must improve
the representative case in repeated measurements without a correctness or p95
regression. There is no fixed percentage threshold.

## 2. Scope

P4 contains five workstreams:

| Workstream | Purpose |
| --- | --- |
| P4.0 | Split production transport service time into payload, control, flush, CQ drain, barrier submission, and barrier polling. |
| P4.1 | Reduce dispatch and combine producer record staging cost. |
| P4.2 | Optimize the SIMT transport control and barrier path selected by P4.0 evidence. |
| P4.3 | Reduce dispatch expert-prefix and final output-copy cost. |
| P4.4 | Attribute kernel-launch, event, stream-wait, and inter-stage idle time, then remove confirmed scheduling overhead. |

HCOMM or CANN changes are outside this repository's first response. P4 only
escalates to those layers when a minimal DeepEP-Ascend probe shows that the
remaining time is inside an external call rather than in command preparation,
protocol work, memory movement, or scheduling owned here.

## 3. Baseline Evidence

The retained P3 EP8 profile reports these independently aggregated max-rank
times:

| Operation | Mean | Producer | Network | Consumer |
| --- | ---: | ---: | ---: | ---: |
| dispatch | 36.176 ms | 4.470 ms | 6.748 ms | 8.787 ms |
| expanded dispatch | 38.347 ms | 4.484 ms | 6.642 ms | 10.300 ms |
| cached dispatch | 85.689 ms | 50.759 ms | 10.935 ms | 8.813 ms |
| combine | 141.098 ms | 47.147 ms | 60.302 ms | 18.424 ms |
| reduced combine | 168.954 ms | 74.430 ms | 62.896 ms | 18.419 ms |

Important subphases are:

- dispatch producer record: `3.860 ms`;
- dispatch expert prefix: `2.788 ms`;
- dispatch output copy: `4.857 ms`;
- combine producer record: `45.336 ms`;
- reduced-combine producer record: `72.613 ms`; and
- dispatch service submit: `5.734 ms`, derived as total service minus CQ drain.

The last value is not a standalone SQE-construction measurement. Device
barrier generation polling is not included in the existing CQ wait counter and
is therefore folded into service submit.

## 4. P4.0 Transport-Service Attribution

### 4.1 Production command sequence

For non-pipelined EP8 dispatch, the 30 logical commands are:

```text
7 payload puts
1 payload flush
7 * (count put-value + generation put-value + generation signal)
1 device barrier
```

The payload schedule already matches the transport-only probe at a high level:
all nonzero peer puts are issued before one flush. The difference is the
control and barrier tail. The optional dispatch chunk pipeline can add payload
batches, but it does not justify assuming that payload flush placement is the
root cause.

### 4.2 Profile ABI

`TransportStageProfile` advances to ABI version 2 and records:

```text
payload_command_cycles
control_command_cycles
flush_command_cycles
barrier_command_cycles
barrier_poll_cycles
```

The first four counters include command cache refresh, validation, route and
buffer resolution, request construction, SQ copy, doorbell, and any drain made
inside that command. `barrier_poll_cycles` is a subset of
`barrier_command_cycles` and records only polling of remote generation
counters. Existing `wait_cycles` remains the CQ-drain counter.

The derived phase report changes to:

```text
cq_wait = wait_cycles
barrier_wait = barrier_poll_cycles
service_submit = service_cycles - cq_wait - barrier_wait
```

The derivation is valid only when the two wait counters fit within total
service cycles. Raw command-category counters remain available even when a
consumer wants a different grouping.

### 4.3 Decision gate

Run the unchanged representative case with profiling and compare:

```text
payload_command_cycles
control_command_cycles
flush_command_cycles
barrier_command_cycles - barrier_poll_cycles
barrier_poll_cycles
```

P4.2 starts with the largest stable release component. If no component explains
the network span, add a benchmark-only ablation that stops after payload flush,
then after control publication, and finally after the barrier. Production
protocol behavior must not be weakened for measurement.

### 4.4 Profiling code-generation boundary

The production dispatch kernel is already large enough that adding command
timestamps directly to its inlined transport service can exceed the AIV scalar
internal-buffer limit. This is a code-generation limit, not a profile-buffer
addressing error. The two-rank investigation established the boundary with the
same `normal-fp32` case and environment:

| Task | Variant | Result |
| --- | --- | --- |
| `task_20260825_231543_203150823010` | retained P3 profile | pass |
| `task_20260825_230106_18609987745` | ABI v2 with five inlined category timers | AIV scalar address fault |
| `task_20260825_231310_199551832562` | removed the long-lived per-command timestamp only | AIV scalar internal-buffer fault |
| `task_20260825_232308_215071211106` | ABI v2 with all new device timers removed | pass |

The profile allocation, enlarged ABI-v2 layout, and three-cache-line header
flush are therefore valid. The failing boundary is the extra timer state and
branches after they are inlined into production dispatch.

Follow-up experiments ruled out moving the timestamps to another AICore
function. They also showed that the limit is reached by one additional live
timer, not only by five timers or by an inlined implementation:

| Task | Variant | Result |
| --- | --- | --- |
| `task_20260826_001454_29288729485` | no-inline callee with a small direct context | pass |
| `task_20260826_001740_297749022436` | the same no-inline callee in production | invalid expert counts |
| `task_20260826_002405_304396818030` | outer `CoreTiling`, reference context | category counters remain zero |
| `task_20260826_003123_310837724009` | outer `CoreTiling`, context passed by value | category counters remain zero |
| `task_20260826_004405_44412830526` | one compile-time-selected flush timer | AIV scalar internal-buffer fault |

The no-inline design is therefore rejected. ABI v2, the three-cache-line
profile header, host parsing, and aggregation remain valid, but the five new
device counters stay zero in production kernels. P4.0 continues with a
benchmark-only staged ablation that runs separately after payload flush,
control publication, and barrier completion. It does not add timestamps to the
aggregate service kernel and does not expose a weakened protocol through the
production API.

### 4.5 Staged-ablation result

The staged release implementation used archive SHA-256
`a6c7d2647f0e5b77afeb170d6fa451d27f3deeff368971278c114606c38b49b8`.
It compiled in `task_20260826_013408_83673925337`; the resulting extension
SHA-256 was
`86f4b85a8b707571f1606418ded93ade7a1346b1d7ff995bfb82c7f2a40b1540`.
The two-rank five-operation correctness and profile check passed in
`task_20260826_013909_87589324108`. Both ranks reported matching generation
and completion generation, final SQ/CQ depth `0/0`, and all three release
stages.

The EP8 representative capture ran twice in
`task_20260826_014139_88647725079`. Both runs used workload fingerprint
`d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00`,
72 blocks, 30 warmups, and 30 measured samples. Values below are max-rank
stage spans in millions of device cycles. A stage span includes its separate
kernel and context setup, so it is an attribution upper bound.

| Operation | Mean A/B | Payload A/B | Control A/B | Barrier A/B |
| --- | ---: | ---: | ---: | ---: |
| dispatch | 36.767 / 37.834 ms | 0.968 / 0.988 | 0.377 / 0.373 | 5.680 / 7.922 |
| expanded dispatch | 38.055 / 39.520 ms | 0.957 / 0.982 | 0.377 / 0.372 | 7.328 / 6.500 |
| cached dispatch | 85.182 / 85.899 ms | 1.017 / 0.967 | 0.377 / 0.373 | 7.549 / 8.621 |
| combine | 104.403 / 105.602 ms | 53.256 / 53.301 | 0.378 / 0.373 | 8.852 / 12.784 |
| reduced combine | 168.293 / 168.600 ms | 53.063 / 53.028 | 0.375 / 0.371 | 8.439 / 9.362 |

Control changes by about `1%` between runs but is small. Barrier is larger but
changes by `11-44%`, and the max frequently moves between ranks; it primarily
shows arrival skew rather than a stable barrier implementation cost. Combine
payload is both largest and stable: normal combine changes by `0.09%` and
reduced combine by `0.07%`.

Each combine rank sends seven payload puts, about `543-548 MB` in total, and
then one flush. The max-rank release-payload span is about `53 ms`, while the
profiled AICore service interval is only about `2-15 ms` depending on barrier
arrival. The release SIMT function has one other large operation: an
unconditional `system_fence()` after the producer has written the staging
buffer. In the 72-block staged path, producer record and release are already
separate kernels on the same stream. The kernel boundary completes the GM
writes before release starts, so this fence is a measured duplicate candidate.
The `kFull` single-kernel path still requires the explicit fence.

P4.2 therefore starts with one narrow hypothesis: keep the fence for `kFull`
and remove it only from separately launched combine release stages. Put,
flush, control, signal, barrier, timeout, generation, and first-error behavior
remain unchanged. The candidate is retained only after two-rank correctness
and EP8 ABBA qualification.

### 4.6 P4.2 fence-ablation result

The candidate archive SHA-256 was
`edfeb5d014e07997511d376ea1d834d0299984b9008c8931ca967d163d5d2b92`.
It compiled in `task_20260826_022403_115187917853`; the resulting extension
SHA-256 was
`7812a98f541ad5d24b851e3019115c3627e0b25ccb6ffd9babffe3ed1e32fb86`.
The two-rank five-operation check passed in
`task_20260826_022928_116817319530`. Both ranks reported matching generation
and completion generation and final SQ/CQ depth `0/0` for every operation.

The EP8 qualification ran in `task_20260826_023306_117662012202`. It used the
same representative workload fingerprint, 72 blocks, 30 warmups, and 30
samples per run. The normal-timing order was baseline A, candidate A,
candidate B, baseline B. Values below pool the two 30-sample runs for each
build.

| Operation | Baseline mean | Candidate mean | Mean delta | Baseline p95 | Candidate p95 | p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 40.249 ms | 38.792 ms | -3.621% | 45.416 ms | 41.135 ms | -9.427% |
| expanded dispatch | 39.753 ms | 39.828 ms | +0.190% | 42.128 ms | 41.201 ms | -2.201% |
| cached dispatch | 85.833 ms | 86.264 ms | +0.501% | 87.917 ms | 87.787 ms | -0.148% |
| combine | 105.586 ms | 105.810 ms | +0.213% | 107.265 ms | 107.332 ms | +0.063% |
| reduced combine | 168.766 ms | 168.830 ms | +0.038% | 170.765 ms | 170.280 ms | -0.284% |

The apparent dispatch movement is not attributable to this combine-only
change: the two baseline dispatch means differ by `9.433%`, while the two
candidate means differ by `0.366%`. Normal combine is effectively unchanged.
The profile runs independently confirm that the fence is not the stable
payload cost:

| Operation | Baseline payload A/B | Candidate payload A/B |
| --- | ---: | ---: |
| dispatch | 0.968 / 0.988 Mcycles | 0.973 / 1.023 Mcycles |
| expanded dispatch | 0.957 / 0.982 Mcycles | 0.988 / 0.951 Mcycles |
| cached dispatch | 1.017 / 0.967 Mcycles | 0.965 / 0.969 Mcycles |
| combine | 53.256 / 53.301 Mcycles | 53.131 / 53.508 Mcycles |
| reduced combine | 53.063 / 53.028 Mcycles | 53.080 / 53.204 Mcycles |

The candidate is rejected and the unconditional fence is retained. Kernel
ordering may make the extra fence logically redundant in the staged path, but
removing it neither reduces the stable 53-million-cycle payload span nor
improves end-to-end combine latency. The dominant work in that span remains
the seven payload puts and flush. P4 proceeds to dispatch record/output-copy
work instead of changing release ordering further.

## 5. P4.1 Producer Record

### 5.1 Combine

The normal combine record path currently assigns a row to a SIMT lane and then
copies all `7168` hidden elements with a scalar loop. The first candidate uses
aligned vector or `DataCopy` movement for normal BF16 records while retaining a
scalar tail. Expanded or reduced combine is a separate candidate because it
must perform reduction rather than a direct copy.

The normal-copy candidate and reduction candidate are measured separately.
Neither may change record headers, routing weights, contributor metadata, or
the symmetric-window layout.

The first normal-copy candidate uses 256-element GM-to-UB-to-GM `DataCopy`
tiles and leaves only the tail to the SIMT scalar loop. Expanded combine stays
on the scalar reduction path. It passed the three-case two-rank matrix in
`task_20260826_005213_5276784839` after the full build in
`task_20260826_004835_5107941741`.

The production-disabled EP8 ABBA qualification ran in
`task_20260826_010024_61308724012`. The baseline was clean commit
`ef7de4b0bc8b132f52d807af0e3e5a9479df6f19`, archive SHA-256
`69d422c7c3cb7602691bcb885a52b4fbe1794b40696605e0b704f485b0a5b812`;
the candidate archive SHA-256 was
`e66c01c5e4f3b096e6cc96d98ce763204b0467036548ca28ff2bd2328fd31254`.
All four runs used the representative manifest, 72 blocks, 30 warmups, 30
samples, and the order baseline A, candidate A, candidate B, baseline B.

| Operation | Baseline mean | Candidate mean | Mean delta | Baseline p95 | Candidate p95 | p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 38.276 ms | 38.656 ms | +0.991% | 39.721 ms | 40.102 ms | +0.958% |
| expanded dispatch | 39.786 ms | 39.853 ms | +0.168% | 41.340 ms | 41.675 ms | +0.810% |
| cached dispatch | 85.874 ms | 86.003 ms | +0.150% | 87.690 ms | 87.587 ms | -0.118% |
| combine | 141.040 ms | 105.482 ms | -25.211% | 142.966 ms | 107.310 ms | -24.940% |
| reduced combine | 168.445 ms | 169.233 ms | +0.468% | 170.285 ms | 171.074 ms | +0.463% |

The two candidate combine means differ by only `0.082%`, compared with
`0.278%` between the two baseline means. The candidate is retained because it
produces a repeatable normal-combine improvement and leaves the reduction path
unchanged. The sub-percent movement in the other operations is treated as run
variation; it does not select another implementation change.

### 5.2 Dispatch

Dispatch already has a vector payload path for eligible grouping shapes. P4.1
measures vector-body and scalar-tail cycles separately before changing it. The
first candidate removes duplicated staging work or fuses metadata publication;
it does not replace a vector copy with another equivalent copy.

## 6. P4.2 SIMT Control And Barrier

P4.2 is selected by P4.0 rather than assumed in advance.

If control commands dominate, candidates are evaluated one at a time:

1. cache resolved per-peer channel, SQ/CQ, and registered-buffer metadata for
   the duration of one service execution;
2. publish adjacent count and generation fields with one registered 16-byte
   put instead of two inline 8-byte writes; and
3. use a payload or control remote action only if HCOMM ordering and error
   propagation match the existing signal contract.

If barrier polling dominates, first document the invariant protected by the
barrier: request completion, remote visibility, generation lifetime, or buffer
reuse. P3.4 showed that removing a host-side tail synchronization regresses
dispatch, so P4 does not delete synchronization based on apparent redundancy.
Candidates may move a proven wait off the critical path or replace it with an
equivalent generation lifecycle, but must retain timeout and first-error
behavior.

## 7. P4.3 Dispatch Consumer

P4.3 splits and then optimizes:

```text
expert histogram aggregation
expert alignment and prefix
destination-slot assignment
hidden output copy
scale copy
top-k metadata copy
```

The first prefix candidate reuses an existing count only when its ownership,
alignment, and capacity semantics exactly match the consumer result. The first
copy candidate changes the received or staging layout only when it removes a
full data movement; reshuffling the same bytes between new temporary buffers is
not an optimization.

### 7.1 Compact receive traversal

The retained copy candidate does not change the receive record layout. It
changes the logical domain traversed by the output-copy workers. The previous
domain covered every reserved source slot:

```text
logical_count = world_size * shard_capacity * copies_per_record
```

For the representative shape this is up to `8 * 8192 = 65536` record slots,
although the eight sources together normally contribute about `8192` records.
Most worker iterations therefore decoded a source rank and slot only to reject
an unused slot.

The compact path uses the already produced per-source bases and counts:

```text
total_records = source_bases[last_rank] + source_counts[last_rank]
logical_count = total_records * copies_per_record
```

A compact record index is mapped back to its source rank and source-local slot
by finding the containing prefix interval. The SIMT scalar tail and
scale/weight copy and the AICore hidden-payload `DataCopy` use the same compact
mapping. Destination-slot assignment, hidden and metadata addresses, expanded
dispatch behavior, generation handling, and the transport protocol are
unchanged.

The candidate archive SHA-256 was
`a0924e13b91fbbf5c35af687241c496ce3627a17f342b024396115ea233f820a`.
It compiled in `task_20260826_025158_161776222434`; the resulting extension
SHA-256 was
`ccec003f420c5c649267dbc1128b130ad239a52fc5cb0e4de889ee912c516fb9`.
The two-rank five-operation validation passed in
`task_20260826_025628_163209823333`. Every operation and rank reported a valid
profile, matching generation and completion generation, and final SQ/CQ depth
`0/0`.

The EP8 ABBA and profile qualification ran in
`task_20260826_025829_163760620301`. It used workload fingerprint
`98d9dc5ff7b8f31afbc9589b037fd658d99b85b739b8572de1751b9e979eb623`,
72 blocks, 30 warmups, and 30 samples per run. The normal-timing order was
baseline A, candidate A, candidate B, baseline B. Values below pool the two
30-sample runs for each build.

| Operation | Baseline mean | Candidate mean | Mean delta | Baseline p95 | Candidate p95 | p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 38.133 ms | 37.734 ms | -1.044% | 40.293 ms | 39.623 ms | -1.663% |
| expanded dispatch | 39.222 ms | 39.388 ms | +0.422% | 40.841 ms | 40.974 ms | +0.325% |
| cached dispatch | 85.183 ms | 85.149 ms | -0.039% | 86.629 ms | 86.943 ms | +0.362% |
| combine | 105.084 ms | 105.102 ms | +0.016% | 107.059 ms | 106.624 ms | -0.406% |
| reduced combine | 169.135 ms | 168.965 ms | -0.101% | 174.368 ms | 170.954 ms | -1.958% |

The independent profile A/B averages support the normal-dispatch result:
`epilogue_copy` falls from `4.859` to `4.712` million cycles (`-3.013%`) and
consumer compute falls from `8.182` to `8.026` million cycles (`-1.908%`).
Cached dispatch shows nearly the same copy-stage reduction (`-2.972%`) but its
producer dominates the end-to-end time. Expanded dispatch copy changes only
`-0.635%`, so its `+0.422%` end-to-end movement is not evidence of a copy-path
regression. The candidate is retained: it removes empty-slot traversal and
improves the intended normal-dispatch stage without changing non-dispatch
operations.

## 8. P4.4 Launch And Pipeline Gaps

P4.4 adds a host/device timeline for:

- every direct-stage kernel submission and device interval;
- producer-stream and communication-stream event record and wait operations;
- communication completion to epilogue submission;
- host blocking in event completion and async-state retirement; and
- the idle interval between producer, transport service, and consumer work.

The report must distinguish host submission overhead from device idle time.
Kernel fusion is considered only when launch and dependency gaps are a stable
part of the operation. Event removal is considered only when the remaining
stream order proves the same tensor lifetime, error propagation, and
communication-completion contract.

### 8.1 Host and device timeline result

The optional host timeline reuses the stage-profile enable switch and records
the blocking host intervals around uncached dispatch. It splits stream
synchronization, count copies, CPU prefix calculation, output allocation,
epilogue setup and submission, completion publication, and completion wait.
The device timeline derives the enclosing interval, active-stage union, idle
time, and overlap from the existing absolute stage timestamps. Profiling stays
disabled by default.

The first EP8 capture used archive SHA-256
`2f89efdb222fb366fd22564c43de66ab6eef0c6c59843725bbc846607d2a132a`.
It compiled in `task_20260826_032200_171520122960`; the extension SHA-256 was
`efdf68c1078547624cc1a2f9f1ab32d30cae95a254da96d612a419b24e8b55ca`.
The two-rank five-operation check passed in
`task_20260826_032659_173028923176`, and the EP8 profile passed in
`task_20260826_032906_17358638238`.

The two EP8 profile runs show that required waits dominate the host envelope.
For normal dispatch, stream synchronization averages `15.490 ms` and
completion wait averages `6.108 ms`. For expanded dispatch, they average
`17.152 ms` and `7.912 ms`. These waits protect count visibility, output
allocation, completion, and buffer lifetime, so P4.4 does not remove them.
The repeatable repository-owned cost is the five small synchronous count
copies:

```text
device -> host: rank prefix, all-expert prefix, all-expert unaligned counts
host -> device: local-expert prefix, local-expert unaligned counts
```

Normal dispatch averages `1.119 ms` for the three D2H copies and `0.906 ms`
for the two H2D copies. Expanded dispatch averages `1.711 ms` and `0.978 ms`.
CPU prefix calculation itself is only about `0.007 ms`, so changing the prefix
algorithm is not the first host-path optimization.

`host_timeline_ns.total` is the maximum per-rank sum of host phases. It is not
the sum of the independently aggregated maximum for each phase. Per-phase
values can therefore be compared individually, but must not be added to
reconstruct the reported total.

### 8.2 Retained count-bridge batching

The retained candidate places the three kernel count outputs in one contiguous
int32 tensor and exposes narrow tensor views to the existing kernel arguments.
For the representative EP8 shape, the 64-byte-aligned layout is:

```text
rank prefix:          [0, 8)
padding:              [8, 16)
all-expert prefix:    [16, 273)
padding:              [273, 288)
all-expert unaligned: [288, 544)
```

One D2H copy transfers all 544 elements. The host then copies the three logical
ranges into the existing validation and prefix vectors. The two public
local-expert outputs use a second aligned tensor:

```text
local-expert prefix:    [0, 32)
local-expert unaligned: [32, 64)
```

After CPU prefix calculation, the host packs both ranges and performs one H2D
copy. The public tensors remain narrow views with the same shapes and values as
before. Cached dispatch keeps its existing validation and copy path. The
change therefore reduces uncached dispatch from three synchronous D2H calls
and two synchronous H2D calls to one call in each direction without changing
kernel addresses, count validation, output ownership, or the dispatch-handle
contract.

The host test tensor originally modeled `narrow()` by changing only the shape
and ignoring its start offset. That made all bridge views alias element zero,
allowed unaligned counts to overwrite the rank-prefix tail, and reduced the
test's output token count to zero. The test tensor now preserves a storage
offset and returns an offset `data_ptr()`, matching the PyTorch view behavior
used by production. Both public dispatch and combine probes reproduce the
failure under ASan before the test fix and pass after it.

The candidate archive SHA-256 was
`b47ce50424fb8f8f0af4d84f5745dd1c45c3d2083482231a98da6b2b71378d12`.
It compiled in `task_20260826_034330_17765825066`; the extension SHA-256 was
`f1d3cfba9ee3c5093b0404014c427b75dfef64039cd34bcb38a33fc5aa8d2f42`.
The two-rank five-operation profile passed in
`task_20260826_034807_17962687110`. Every operation and rank reported a valid
profile, matching generation and completion generation, and final SQ/CQ depth
`0/0`. Its JSON SHA-256 was
`f2660f2069f641a4d728b10488d4cb11cb7dfe4687c703a5eea267e4a24a3098`.

The EP8 ABBA qualification ran in
`task_20260826_035057_180287110171`. It used workload fingerprint
`98d9dc5ff7b8f31afbc9589b037fd658d99b85b739b8572de1751b9e979eb623`,
72 blocks, 30 warmups, and 30 samples per run. The order was baseline A,
candidate A, candidate B, baseline B. Values below pool the two 30-sample runs
for each build.

| Operation | Baseline mean | Candidate mean | Mean delta | Baseline p95 | Candidate p95 | p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 36.815 ms | 35.868 ms | -2.572% | 38.373 ms | 37.545 ms | -2.156% |
| expanded dispatch | 37.796 ms | 37.286 ms | -1.350% | 39.998 ms | 38.739 ms | -3.148% |
| cached dispatch | 84.592 ms | 84.667 ms | +0.088% | 86.717 ms | 86.272 ms | -0.514% |
| combine | 105.032 ms | 104.675 ms | -0.339% | 106.780 ms | 106.901 ms | +0.113% |
| reduced combine | 168.736 ms | 168.137 ms | -0.355% | 170.365 ms | 169.780 ms | -0.343% |

The targeted host phases move in the expected direction:

| Operation | Phase | Baseline | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| dispatch | counts D2H | 1.423 ms | 0.351 ms | -75.326% |
| dispatch | prefix H2D | 1.019 ms | 0.442 ms | -56.577% |
| expanded dispatch | counts D2H | 1.206 ms | 0.408 ms | -66.179% |
| expanded dispatch | prefix H2D | 0.899 ms | 0.397 ms | -55.892% |

The change is retained. Both target-operation mean and p95 improve, the two
copy phases explain the movement, and cached dispatch and combine do not show
a material regression. The four ABBA JSON SHA-256 values are:

| Run | SHA-256 |
| --- | --- |
| baseline A | `cb6c5e51e86c383be9fb826b884a038bf783b5af3f5dbbf444f96af0cfc91f3d` |
| candidate A | `8780dc96158354c62663f56feb371952907088cd9faa2b35e34a2fe40cc13a29` |
| candidate B | `4475b9e567d1617dfd5223e9c3bb014addfc8c6606d2eef4f9cddd5dc480b300` |
| baseline B | `506bfec7e9e2d6820f54761ff7ca9fcd99cce509393094020d9f0595ea59f9e0` |

## 9. Validation

Every implementation task follows this sequence:

1. focused host contract test with a demonstrated red failure;
2. Ascend source compile;
3. two-rank lifecycle and correctness tests;
4. one unchanged EP8 representative run when multi-card evidence is required;
5. ABBA performance comparison with 30 warmups and 30 samples for retained
   optimizations; and
6. the 144-case, 720-operation gate before P4 is declared complete.

Profiling-only code must produce zero generated-code or representative
performance change when profiling is disabled. Performance changes are kept
only when repeated means improve outside observed run-order variation and p95
does not show a material regression. Raw JSON, source hashes, workload
fingerprints, commands, task identifiers, and environment versions are retained
with every hardware conclusion.

### 9.1 Final P4 gate

The final local regression passed `188` tests and `48` subtests, including the
two ASan public probes used to diagnose count-view aliasing. The production
extension build, two-rank five-operation profile, and EP8 ABBA evidence are
recorded in section 8.2.

The complete functional matrix passed in
`task_20260826_040301_183578618748`. It used two ranks, 72 blocks, one warmup,
and one measured iteration per case. The report contains `144` passed cases,
no failed or pending cases, and `720` operation records covering dispatch,
expanded dispatch, cached dispatch, combine, and reduced combine. The workload
fingerprint is
`da3db00de02d1c93088c159430363bfb8659a8ed2397b6768f41145bfad14522`.

| Artifact | SHA-256 |
| --- | --- |
| `benchmark.json` | `66715ec0efcdbea064876cf9061cf040a05cd43c3e305c9f4b25d81eb4ef90a8` |
| `workload.json` | `617248230a3733fec854f21f650c3efb3b649286507b74592b6c5f5ff5b55cab` |

The preceding task `task_20260826_040149_183221022990` stopped before running
an operator because it was given an EP8 manifest with `world_size=8` while the
functional gate used two ranks. The successful retry generated and retained a
separate two-rank manifest. This is an input-identity correction, not a failed
operator case.

## 10. Completion Summary

P4.0 adds the production stage-profile contract, service/barrier accounting,
and staged release attribution. Production in-kernel command-category timers
remain disabled because even one additional live timer exceeds the dispatch
AIV scalar-buffer limit. The staged profile shows that combine payload is the
largest stable release component, while barrier time mostly reflects rank
arrival skew.

P4.1 retains the 256-element BF16 combine producer `DataCopy` path. It reduces
representative normal combine mean from `141.040` to `105.482 ms`
(`-25.211%`) and leaves reduced combine on its scalar reduction path.

P4.2 rejects removal of the separately launched combine release fence. It did
not reduce the stable payload span or improve combine latency, so the existing
ordering and synchronization contract remains intact.

P4.3 retains compact dispatch receive traversal. It removes empty reserved-slot
scans, improves normal dispatch mean by `1.044%`, and reduces its epilogue-copy
stage by `3.013%`.

P4.4 retains host/device timeline attribution and count-bridge batching. The
batch reduces uncached dispatch's five small synchronous count copies to two,
improves normal dispatch mean/p95 by `2.572%`/`2.156%`, and improves expanded
dispatch mean/p95 by `1.350%`/`3.148%`.

The remaining high-cost paths are not unmeasured launch bookkeeping. Dispatch
still blocks mainly in the required count-stage stream synchronization and
completion wait. Combine remains dominated by producer payload construction,
seven peer payload puts, one flush, and consumer reduction. Further work should
start with a new measured design rather than removing waits or fences already
rejected by P4 evidence.
