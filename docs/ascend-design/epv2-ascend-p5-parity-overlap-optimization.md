# EPv2 Ascend P5 Timeline, CUDA Parity, and Overlap Optimization

## 1. Purpose

P5 targets the remaining gap between the Ascend 950 production EP path and
the available HCCS payload transport capability. The representative
transport-only probe reaches `2535.046 GB/s`, while the retained P4 normal
Dispatch result reaches about `217.1 logical GB/s`. These values use different
byte formulas and cannot be divided to obtain a hardware-efficiency number.
They do establish that large contiguous HCOMM payload puts are not the first
layer to optimize.

P5 first builds a common Ascend/CUDA timeline and parity matrix, then uses that
evidence to remove serialization in packing, transport submission, count and
prefix handling, and consumer copy or reduction. The work does not start by
changing HCOMM or the SIMT transport facade. Those layers are changed only
when a production-aligned measurement attributes a remaining critical-path
cost to them.

This document is both the P5 design specification and the assignment boundary
for later implementation tasks. It defines:

- the primary performance target and measurement protocol;
- the existing P3/P4 timeline evidence and its limits;
- the complete Dispatch and Combine data flows;
- the relationship among the three timed Dispatch operations and two timed
  Combine operations;
- a stage-by-stage Ascend/CUDA parity matrix;
- the P5.0 through P5.5 optimization order and decision gates; and
- the prompt used to generate the companion timeline image.

## 2. Primary Goal and Measurement Contract

### 2.1 Primary target

The P5 primary acceptance target is:

```text
Ascend 950, single-host EP8, normal FP8 Dispatch
logical bandwidth > 2000 GB/s
```

The target applies to the full public normal Dispatch operation, not to the
transport-only probe, an internal kernel stage, or the sum of per-rank
bandwidth values computed with a different byte formula.

The selected representative case is:

| Setting | Value |
| --- | --- |
| Device | Ascend 950 |
| Topology | single-host EP8 |
| Tokens | 8,192 maximum per rank |
| Hidden width | 7,168 |
| Top-k | 8 |
| Experts | 256 |
| Dispatch representation | FP8 |
| Combine representation | BF16 |
| Expert alignment | 128 |
| Ascend data blocks | 72 |
| Warmup / measured samples | 30 / 30 |
| Timing | device event, rank-max per sample |
| Case ID | `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` |

The synchronous `async0/alloc0` case is intentional. It exposes the complete
production dependency chain and prevents an external workload from hiding a
mandatory Dispatch stage. Async/event variants remain secondary overlap
qualification cases.

The representative normal Dispatch logical byte count is about `7.786 GB`.
Therefore the target implies an end-to-end mean latency below approximately:

\[
T_{2000} = \frac{7.786\times 10^9}{2000\times 10^9}
         \approx 3.893\ \text{ms}
\]

The retained P4 result is about `35.868 ms`, so the target requires a major
pipeline change rather than another isolated scalar-loop improvement.

### 2.2 Secondary gates

Every P5 performance run also reports all five timed operations:

1. normal Dispatch;
2. expanded Dispatch;
3. cached normal Dispatch;
4. normal Combine; and
5. reduced Combine.

Normal Dispatch is the primary bandwidth target. The other four operations
are regression and attribution gates. A candidate is not retained if it
changes routing, output layout, contributor ordering, generation handling,
timeout behavior, or public tensor ownership.

Each retained optimization must pass:

- focused host contract tests;
- Ascend source compilation;
- two-rank production and SIMT/AICore correctness checks;
- an unchanged EP8 representative ABBA comparison;
- mean, p50, p95, and logical-bandwidth reporting; and
- the complete 144-case, 720-operation correctness and performance gate before
  P5 is declared complete.

There is no per-item percentage threshold. Stage improvement must explain an
end-to-end movement, and the final P5 goal remains `>2000 GB/s` for normal
Dispatch.

## 3. Baseline Evidence

### 3.1 Retained P4 end-to-end results

The current P4 retained version uses the same EP8 representative workload,
72 blocks, 30 warmups, and 30 measured samples:

| Operation | Mean | P95 | Approximate logical bandwidth |
| --- | ---: | ---: | ---: |
| Dispatch | 35.868 ms | 37.545 ms | 217.1 GB/s |
| Expanded Dispatch | 37.286 ms | 38.739 ms | 248.1 GB/s |
| Cached Dispatch | 84.667 ms | 86.272 ms | 92.0 GB/s |
| Combine | 104.675 ms | 106.901 ms | 104.1 GB/s |
| Reduced Combine | 168.137 ms | 169.780 ms | 64.8 GB/s |

Logical bandwidth is the benchmark's logical-byte throughput. It is not HCCS
physical link bandwidth.

### 3.2 HCCS and transport-only evidence

The Ascend HCCS benchmark separates physical and software layers:

| Probe | Result | Interpretation |
| --- | ---: | --- |
| One-way 64 MiB P2P | about 52.282 GB/s | large-message per-sender reference |
| EP8 32 MiB/peer all-to-all | 2507.178 aggregate GB/s | measured concurrent HCCS/HCOMM result |
| Independent-link extrapolation | 2927.775 aggregate GB/s | topology reference, not measured production bandwidth |
| Representative transport-only | 0.903 ms, 2535.046 aggregate GB/s | production facade, command queue, service, put, flush, and CQ |
| Full representative Dispatch | about 35.9-37.3 ms | includes all production preparation and consumer work |

The transport-only probe sends the representative remote payload matrix but
excludes grouping, offsets, production record packing, control publication,
consumer copy, and host count handling. It proves that payload data-plane
capacity is available; it does not reproduce the full Dispatch protocol.

## 4. Existing Timeline Capability

### 4.1 Device-stage timeline

P3.0 and P4.0 provide absolute device-stage timestamps and derived phase
accounting:

```text
producer
publication
service_submit
cq_wait
barrier_wait
consumer_wait
consumer_compute
epilogue
```

The named producer and consumer stages further identify control, grouping,
prefix, record construction, validation, expert counting, metadata,
destination assignment, copy, reduction, weights, and completion.

The P3 profile baseline for normal Dispatch and normal Combine is:

| Phase | Dispatch | Combine |
| --- | ---: | ---: |
| Producer | 4.470 ms | 47.147 ms |
| Publication | 0.163 ms | 51.361 ms |
| Service submit | 5.734 ms | 7.384 ms |
| CQ wait | 0.851 ms | 1.557 ms |
| Consumer wait | 0.605 ms | 10.782 ms |
| Consumer compute | 8.179 ms | 7.639 ms |
| Completion publication | 0.003 ms | 0.002 ms |

These values are initial attribution references. P5.0 refreshes the same
fields on the retained P4 binary before selecting a P5.1 candidate.

### 4.2 Host/device scheduling timeline

P4.4 adds an optional host timeline around uncached Dispatch. It records:

```text
stream synchronization
count copies from device to host
CPU prefix calculation
prefix copies from host to device
output allocation
epilogue setup
epilogue submission
completion event publication
completion wait and async-state retirement
```

The P4.4 representative measurements include:

| Host phase | Normal Dispatch | Expanded Dispatch |
| --- | ---: | ---: |
| Stream synchronization | 15.490 ms | 17.152 ms |
| Completion wait | 6.108 ms | 7.912 ms |
| Three D2H count copies before batching | 1.119 ms | 1.711 ms |
| Two H2D prefix copies before batching | 0.906 ms | 0.978 ms |
| CPU prefix calculation | about 0.007 ms | about 0.007 ms |

P4.4 retained count-bridge batching, reducing three D2H calls plus two H2D
calls to one call in each direction. Required visibility, allocation,
completion, and buffer-lifetime waits remain.

### 4.3 Interpretation constraints

For sample `i`, the formal operation latency is:

\[
T_i = \max_{0\le r<8} T_{i,r}
\]

The mean, p50, and p95 are computed from these rank-max samples. Stage
profiling independently selects the maximum for each phase, so one reported
phase vector may contain components from different ranks. Consequently:

- stage values must not be summed to reconstruct end-to-end latency;
- stage spans may include separate launch and context setup;
- barrier spans can include rank-arrival skew;
- an overlap ceiling is a diagnostic bound, not a measured speedup; and
- the companion image must be labeled `not to scale`.

## 5. Timeline Gaps P5.0 Must Close

The existing timeline identifies long stages but does not yet provide a
CUDA-parity critical-path trace. P5.0 adds or derives the following fields:

| Field | Required interpretation |
| --- | --- |
| `stage_id` | Stable ID shared by the figure, JSON, tables, and code mapping; `D0-D8`, `C0-C7`, `T0-T1`, `H0-H2`, `S0-S3`, and `F0` are reserved by this spec |
| `operation` | Dispatch, expanded Dispatch, cached Dispatch, Combine, or reduced Combine |
| `rank` | Rank that produced the interval; do not discard it before critical-path analysis |
| `lane` | Host, SIMT producer, AICore service, remote receive, or consumer epilogue |
| `start_cycle/end_cycle` | Absolute device timestamps for overlap calculation |
| `host_start_ns/host_end_ns` | Host submission and blocking intervals |
| `implementation` | Ascend function or runtime method responsible for the stage |
| `cuda_counterpart` | CUDA kernel or warp role that performs equivalent semantic work |
| `parallel_unit` | Thread, subgroup, warp, block, SM, rank owner, expert owner, or service channel |
| `work_items` | Tokens, records, contributors, experts, hidden elements, commands, or bytes |
| `bytes` | Payload, scale, metadata, D2H, H2D, SQ, and CQ bytes where defined |
| `dependency` | Stream order, event, generation, barrier, CQ completion, or host allocation |
| `active/idle/overlap` | Device active union, inter-stage idle span, and measured overlap |

P5.0 must split broad stages enough to distinguish:

- hidden record packing from scale and top-k metadata packing;
- command reserve/write/publication from WQE/SQE construction and SQ post;
- payload flush from control publication;
- barrier submission from remote-generation polling;
- expert count from expert prefix and destination assignment;
- vector main-body copy from scalar tail; and
- contributor discovery from hidden reduction and final output write.

If production in-kernel timestamps exceed the AIV scalar internal-buffer
limit, use separately compiled profile kernels or repeated stage ablations.
Profiling must remain disabled by default and must not change disabled-path
generated code or performance.

## 6. Dispatch and Combine Mode Relationships

### 6.1 Dispatch modes

The three timed Dispatch operations are not three independent layouts.
Normal and expanded select the output layout; cached selects whether a valid
previous handle is reused.

| Timed operation | Layout | Behavior | Downstream operation |
| --- | --- | --- | --- |
| Normal Dispatch | compact expert input | One source token record is sent once per destination rank; the receiver resolves local expert placement | Normal Combine |
| Expanded Dispatch | expanded expert-major input | Each valid top-k expert route receives an independent output slot | Reduced Combine |
| Cached Dispatch | cached compact layout | Reuses validated destination slots and handle metadata from a previous normal Dispatch | Normal Combine |

A cached-expanded variant also exists. It reuses an expanded Dispatch handle
and validates zero padding and output placement. The canonical suite prepares
and checks it, but does not report it as a sixth timed operation.

The relationship is:

```text
router output: x + scale + top-k index/weight
    |
    +-- normal Dispatch ----------------> compact expert input
    |       `-- cached normal replay ----> same compact layout
    |                                      `-- normal Combine
    |
    `-- expanded Dispatch --------------> expanded expert-major input
            `-- cached expanded replay --> same expanded layout
                                           `-- reduced Combine
```

### 6.2 Combine modes

| Timed operation | Input layout | Behavior |
| --- | --- | --- |
| Normal Combine | compact layout from normal Dispatch | Returns expert output to the source rank and performs direct copy or bounded contributor merging |
| Reduced Combine | expanded layout from expanded Dispatch | Resolves multiple expert or rank contributors for a token, reduces hidden values, and writes the final BF16 output |

`allow_multiple_reduction` controls whether compatible local contributions can
be reduced before they are represented as remote records. It does not turn
normal Combine and reduced Combine into identical operations. Their input
layouts and contributor work remain different.

### 6.3 Timeline selection

The primary timeline and P5 bandwidth gate use:

```text
normal FP8 Dispatch -> expert compute boundary -> normal BF16 Combine
```

This path contains grouping, prefix, record packing, transport, consumer
layout, and reverse communication without cached-handle reuse or expanded
reduction amplification.

The companion figure also contains a secondary branch:

```text
expanded FP8 Dispatch -> expert compute boundary -> reduced BF16 Combine
```

This branch is the P5.4 reduction target. Cached normal and cached-expanded
operations are shown as dashed replay branches instead of separate main
timelines.

## 7. End-to-End Stage Model

### 7.1 Dispatch

```text
D0 input and control
 -> D1 rank grouping and count
 -> D2 rank prefix and staging-slot assignment
 -> D3 hidden/scale/top-k/metadata record packing
 -> D4 logical command publication
 -> T0 AICore HCOMM put/flush and request submission
 -> T1 CQ, generation, and remote visibility wait
 -> D5 acquire, record validation, and local-expert count
 -> D6 expert prefix and alignment
 -> D7 metadata and destination assignment
 -> D8 hidden/scale/top-k output copy
 -> F0 completion publication
```

### 7.2 Combine

```text
C0 input control and validation
 -> C1 contributor plan and rank prefix
 -> C2 contributor record packing
 -> C3 local staging copy
 -> C4 logical command publication
 -> T0 AICore HCOMM put/flush and request submission
 -> T1 CQ, generation, and remote visibility wait
 -> C5 contributor validation and slot resolution
 -> C6 direct copy or hidden reduction
 -> C7 routing-weight output
 -> F0 completion publication
```

The expert GEMM lies between Dispatch and Combine. It is a model dependency,
not part of either EP operator's measured latency. The image must show this
boundary without adding GEMM time to the five-operation benchmark.

## 8. Ascend/CUDA Parity Matrix

### 8.1 Function mapping

| Stage | Ascend implementation | CUDA implementation |
| --- | --- | --- |
| D0 control | `direct_dispatch_producer_control_vf` | `dispatch_impl` prologue and notify-warps setup |
| D1 grouping | `direct_dispatch_producer_group_vf` | notify warps in `dispatch_impl` |
| D2 prefix/slot | `direct_dispatch_producer_prefix_vf` | warp prefix and atomic destination-slot assignment in `dispatch_impl` |
| D3 record packing | `direct_dispatch_producer_record_vf` plus vector payload path | dispatch warps, TMA load/store, scale copy, and metadata writes in `dispatch_impl` |
| D4 publication | `direct_dispatch_producer_release_vf` | Gin put, put-value, flush, and GPU barrier in `dispatch_impl` |
| D5 acquire/count | `direct_dispatch_epilogue_acquire_vf`, `direct_dispatch_epilogue_validate_records_vf`, `direct_dispatch_epilogue_count_experts_vf` | notify results, acquire, validation, and token traversal in `dispatch_copy_epilogue_impl` |
| D6 expert prefix | `direct_dispatch_epilogue_prefix_vf` | warp prefix and exchange in `dispatch_copy_epilogue_impl` |
| D7 metadata/destination | `direct_dispatch_epilogue_metadata_vf`, `direct_dispatch_epilogue_assign_destinations_vf` | destination assignment and metadata traversal in `dispatch_copy_epilogue_impl` |
| D8 output copy | `direct_dispatch_epilogue_copy_outputs_vf` plus AICore DataCopy path | per-warp TMA load/store in `dispatch_copy_epilogue_impl` |
| C0 control | `direct_combine_producer_control_vf` | combine kernel prologue and workspace setup in `combine_impl` |
| C1 plan/prefix | `direct_combine_producer_plan_vf`, `direct_combine_producer_plan_prefix_vf` | contributor metadata traversal and prefix in `combine_impl` |
| C2 record | `direct_combine_producer_record_vf` and vector payload implementation | TMA copy or warp-cooperative local reduction in `combine_impl` |
| C3 local staging | `direct_combine_producer_local_copy_vf` | local-copy and send-buffer preparation in `combine_impl` |
| C4 publication | `direct_combine_producer_release_vf` | Gin put and final GPU barrier in `combine_impl` |
| C5 contributor slots | `direct_combine_epilogue_validate_vf`, prepare-vector-slots, and slot workspace stages | warp ballot, exchange, and `compute_topk_slots` |
| C6 reduction/copy | `direct_combine_epilogue_reduce_vf` and vector implementation | `combine_reduce` in `combine_reduce_epilogue_impl` |
| C7 weights | `direct_combine_epilogue_weights_vf` | lane-level top-k weight copy in `combine_reduce_epilogue_impl` |
| T0 transport service | staged command queue and `aicore_transport_service` | NCCL Device API Gin executed by kernel warps |
| T1 completion | CQ drain, generation checks, barrier polling | Gin barrier and CUDA programmatic dependency |
| F0 completion | Dispatch/Combine complete VFs | final store, barrier, or programmatic launch completion |

CUDA source references:

- `deep_ep/include/deep_ep/impls/dispatch.cuh`;
- `deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh`;
- `deep_ep/include/deep_ep/impls/combine.cuh`; and
- `deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh`.

Ascend source references:

- `csrc/backends/ascend/elastic/dispatch.asc`;
- `csrc/backends/ascend/elastic/combine.asc`;
- `csrc/backends/ascend/transport/device_transport_commands.hpp`;
- `csrc/backends/ascend/transport/aicore_transport_service.hpp`; and
- `csrc/backends/ascend/runtime/host_timeline.hpp`.

### 8.2 Parallelism gaps

| Dimension | CUDA DeepEP | Current Ascend | P5 consequence |
| --- | --- | --- | --- |
| Main-kernel roles | Notify warps and dispatch warps advance inside one kernel | Control, group, prefix, record, release, and service are separate ordered stages | Measure and then overlap record chunks with publication/service |
| Payload work unit | Warp owns a token; lanes cooperate over top-k and hidden chunks | 72-block record grid gives record ownership, but release waits for the complete stage | Add bounded chunk lifecycle and independent staging ownership |
| Data movement | TMA load/store, `cp.async`, and metadata work are pipelined per token | Vector/DataCopy main bodies exist, but metadata and service are separated by stage barriers | Trace vector body, scalar tail, and service overlap separately |
| Communication issue | Gin calls are issued from the executing CUDA warp/channel | SIMT records commands; a later AICore service constructs and posts URMA requests | Preserve staged ABI but pipeline producer and service generations |
| Count and prefix | Device-side notify reduction and warp prefix are available | Synchronous path crosses D2H, CPU prefix, and H2D boundaries | Move prefix and output sizing off the host critical path where contracts allow |
| Consumer start | CUDA PDL and device dependency start the epilogue without a host round trip | Ascend submits multiple epilogue stages after host-visible counts and allocation | Create a device-visible epilogue boundary and separate exact allocation from critical progress |
| Expert metadata | Warp/subgroup primitives keep small count and prefix state near the kernel | Rank/expert owner scans and one-block control stages serialize parts of the path | Increase subgroup and block participation without changing deterministic order |
| Combine reduction | Warp-cooperative contributor selection, vector reduction, and TMA output | Normal Combine has a retained DataCopy path; expanded/reduced paths use more explicit contributor and hidden loops | Align reduced Combine with subgroup/vector execution before transport tuning |
| Queue/channel use | Warps can map to multiple Gin/QP channels | Production uses a staged service and bounded queues; prior HWM was only 4/4 | Do not add channels until the new timeline proves service saturation |
| Launch and wait | Fewer fused kernels and device-side dependency | More `asc_vf_call` stages, stream waits, count copies, and completion waits | Attribute launch gaps and fuse only stable critical-path boundaries |

## 9. P5 Workstreams

### 9.1 P5.0: Timeline and CUDA parity instrumentation

P5.0 produces one machine-readable critical-path report for each of the five
operations and a Markdown table using the stable stage IDs in this spec.

Required outputs:

- per-rank absolute stage timestamps;
- host submission and blocking intervals;
- stage work counts and byte counts;
- active-stage union, idle gaps, and measured overlap;
- Ascend function and CUDA counterpart fields;
- normal Dispatch/normal Combine primary timeline;
- expanded Dispatch/reduced Combine secondary timeline; and
- a refreshed P4-final baseline before an implementation candidate is chosen.

The CUDA side first uses existing kernel boundaries and profiler/NVTX ranges.
It does not require invasive per-warp timing in production code. Nsight or
equivalent evidence may refine a stage, but the formal cross-platform table
compares semantic stages rather than claiming that differently decomposed
kernels have identical timer overhead.

P5.0 is complete when the report can answer:

1. how much device idle time exists between packing, service, and consumer;
2. whether producer work and HCOMM service overlap at all;
3. which host dependency blocks the consumer submission;
4. how much work is executed by one block, one owner, or the 72-block grid;
5. which Ascend GM round trips have no CUDA counterpart; and
6. which long interval is computation, arrival skew, or communication wait.

### 9.2 P5.1: Producer/transport chunk overlap

P5.1 is the first performance implementation after P5.0. It replaces the
all-records-then-release schedule with a bounded chunk lifecycle:

```text
chunk n packing
    -> publish chunk n
    -> service chunk n
       || chunk n+1 packing
    -> remote visibility and chunk completion
```

The candidate uses double-buffered or bounded ring staging only when every
chunk owns independent records, commands, request state, generation, and
completion state. A producer may not reuse a chunk slot until its request and
remote visibility contract are complete.

The first slice applies to normal non-cached Dispatch. Expanded, cached,
hybrid, and scale-out paths keep their existing schedule until the lifecycle
is qualified. Chunk count is a measured tiling choice, not an unrestricted
runtime knob.

### 9.3 P5.2: Remove the host count bridge from the critical path

P5.2 selects one of two approaches from P5.0 evidence:

1. device-side count/prefix and conservative preallocation; or
2. asynchronous count copy and prefix publication that overlaps independent
   work without violating output-allocation ownership.

The design must preserve exact public tensor shapes, zero-token behavior,
cached-handle validation, stream ownership, and error propagation. CPU prefix
arithmetic is not the target; the target is the dependency and copy chain
around it.

### 9.4 P5.3: Consumer metadata and copy parallelism

P5.3 addresses the longest retained consumer substage in this order:

1. expert count;
2. expert prefix and alignment;
3. destination assignment;
4. metadata copy;
5. hidden and scale output copy.

Candidates use record-oriented or subgroup-oriented work distribution and
reuse already valid compact source bases and counts. They must not restore a
scan over all reserved source slots or introduce a new full GM temporary.

### 9.5 P5.4: Reduced Combine parity

P5.4 aligns the expanded/reduced path with CUDA's cooperative structure:

- resolve contributor ranks and slots once per token;
- retain contributor state across hidden chunks where the compiler permits;
- assign one qualified subgroup or data unit to a token or hidden tile;
- use vector/DataCopy movement for aligned bodies;
- restrict scalar loops to tails and unsupported shapes; and
- avoid a separate GM round trip when grouping and reduction can share a
  qualified kernel boundary.

Normal Combine remains the control because P4.1 already retained its
256-element BF16 DataCopy path.

### 9.6 P5.5: SIMT/HCOMM control and service

P5.5 starts only if P5.0-P5.4 leave a stable service-side critical path. Its
candidate order is:

1. batch command resolution and request construction;
2. cache per-peer channel and registered-buffer metadata for one execution;
3. combine adjacent compatible control writes while preserving ordering;
4. overlap barrier polling with independent consumer preparation; and
5. add channels only after SQ/CQ saturation evidence.

Direct SIMT URMA doorbell submission remains outside the first response
because CANN 9.2 does not expose the required operation to the current SIMT VF
environment. Replacing the staged architecture requires a separately
qualified vendor interface or compiler/runtime capability.

## 10. Decision and Retention Rules

For every workstream:

1. preserve the workload manifest and logical-byte formula;
2. compare the same binary configuration and case in ABBA order;
3. retain raw per-rank and rank-max samples;
4. record mean, p50, p95, logical bandwidth, stage time, and work counts;
5. explain the end-to-end movement with a targeted stage movement;
6. reject a candidate that merely moves time into an unmeasured stage;
7. reject a candidate with correctness, generation, timeout, or buffer-lifetime
   regressions; and
8. do not add the percentages of independent workstream experiments.

The final P5 report compares the complete retained tree with the retained P4
baseline. It does not construct a projected result by summing isolated gains.

## 11. Companion Timeline Figure Contract

The generated image is tracked under the design assets directory with an ASCII
filename so Markdown renderers and review tools can load it consistently. It
was generated from the GPT Image prompt in section 11.1 through the
authenticated Codex image-generation workflow.

![EPv2 P5 Dispatch and Combine timeline](assets/epv2-p5-dispatch-combine-timeline.png)

Asset path:

```text
docs/ascend-design/assets/epv2-p5-dispatch-combine-timeline.png
```

The image is an explanatory reference. Numeric latency fields are intentionally
blank in the generated asset; P5.0 fills them from the machine-readable
timeline report and the corresponding Markdown table.

The figure must:

- use a left-to-right time direction;
- show Host, Ascend SIMT producer, AICore HCOMM service, remote receive,
  Ascend consumer, and CUDA reference swimlanes;
- contain both the normal Dispatch/normal Combine main path and the
  expanded Dispatch/reduced Combine secondary path;
- show cached modes as dashed replay branches;
- label every block with the stage ID used by the timeline table;
- show explicit serialization barriers and possible overlap regions;
- map each Ascend stage to its CUDA counterpart;
- link stage IDs to the exact implementation table in section 8.1 rather than
  rendering long source symbols in the bitmap;
- include a latency placeholder for P5.0 data; and
- state that stage timings are independently aggregated and not drawn to
  scale.

### 11.1 Self-contained GPT Image prompt

Use this prompt in a new image-generation context. Do not provide any previous
image as input; the prompt defines the complete composition and is the single
source of truth.

```text
Create a new high-resolution technical engineering diagram titled:

"DeepEP v2 on Ascend 950 vs CUDA DeepEP - Complete MoE Dispatch and Combine Timeline"

Start from a blank canvas. Do not imitate or edit a previous image. The result is for a desktop engineering design specification, so use a very wide landscape canvas of at least 3840 x 2160 pixels, a clean white background, thin straight arrows, square or slightly rounded rectangular phase blocks, and highly readable text. Use monospace text for stage IDs. Do not use gradients, 3D effects, illustrations, decorative shapes, or marketing styling.

The figure has three vertically stacked sections:

1. A. DISPATCH PIPELINE
2. a gray EXPERT GEMM boundary
3. B. COMBINE PIPELINE

Dispatch and Combine each use exactly these six horizontal swimlanes:

1. Host runtime and stream scheduling
2. Ascend SIMT producer
3. Ascend AICore HCOMM/URMA transport service
4. Remote rank and receive buffers
5. Ascend consumer epilogue
6. CUDA reference implementation

Time moves from left to right.

CRITICAL LAYOUT RULE:
Each stage ID must appear exactly once in each pipeline. Never duplicate a stage in another swimlane. Arrows may cross swimlanes, but the stage block remains only in its actual execution lane.

Place Dispatch stages as follows:

Host runtime lane:
- S0: submit producer stages
- H0: device-to-host count bridge
- H1: CPU prefix and output sizing
- H2: host-to-device prefix bridge
- S1: submit consumer stages D5-D8/F0

Ascend SIMT producer lane:
- D0: control
- D1: rank grouping
- D2: rank prefix and staging-slot assignment
- D3: producer record packing
- D4: logical command publication

Ascend AICore HCOMM/URMA transport service lane:
- T0: HCOMM put, flush, WQE/SQE construction, and SQ post
- T1: CQ drain, generation wait, and barrier polling

Remote rank and receive-buffer lane:
- R0: payload arrival in remote receive/staging shards
- R1: generation and payload visibility

Ascend consumer epilogue lane:
- D5: acquire, record validation, and local-expert count
- D6: expert prefix and alignment
- D7: metadata and destination assignment
- D8: hidden, scale, and top-k output copy
- F0: Dispatch completion publication

CUDA reference lane:
- one dashed CUDA producer/transport semantic-reference block spanning the work corresponding to D0-D4 and T0-T1
- one dashed CUDA consumer-epilogue semantic-reference block spanning the work corresponding to D5-D8/F0
- show notify warps and dispatch warps executing inside one main kernel
- show warp grid-stride work, TMA load/store, asynchronous payload movement, and device-side dependency

The Dispatch arrows must show this order:

S0 -> D0 -> D1 -> D2 -> D3 -> D4 -> T0 -> T1 -> R0/R1 -> H0 -> H1 -> H2 -> S1 -> D5 -> D6 -> D7 -> D8 -> F0

Divide the Dispatch timeline into four visible time bands from left to right:
1. producer submission and D0-D4;
2. T0/T1 transport and R0/R1 remote visibility;
3. only after T1 and R1 complete, H0/H1/H2 host count bridge;
4. S1 and D5-D8/F0 consumer work.

Draw an explicit dependency arrow from T1/R1 to H0. H0 must be positioned to the right of T1 and R1. Do not draw H0, H1, or H2 concurrently with T0 or T1 in the current synchronous path.

Show a blue possible-overlap arrow between D3 packing and T0 transport, labeled:
"P5.1 producer/transport overlap target"

Show the host count bridge H0 -> H1 -> H2 as a dashed purple path, labeled:
"P5.2 host count bridge target"

Place a gray horizontal boundary after Dispatch F0 and before Combine C0. Label it exactly:

"Expert GEMM / model compute"
"outside EP Dispatch/Combine timing"

Show the logical model flow:

Dispatch F0 -> Expert GEMM / model compute -> Combine C0

Draw the complete arrow through all three items. Do not terminate the arrow at the gray boundary, and do not start Combine as an unrelated flow.

Place Combine stages as follows:

Host runtime lane:
- S2: submit producer stages C0-C4
- S3: submit consumer stages C5-C7/F0

Ascend SIMT producer lane:
- C0: control
- C1: contributor plan and rank prefix
- C2: contributor record packing
- C3: local staging copy
- C4: logical command publication

Ascend AICore HCOMM/URMA transport service lane:
- T0: HCOMM put, flush, WQE/SQE construction, and SQ post
- T1: CQ drain, generation wait, and barrier polling

Remote rank and receive-buffer lane:
- R0: payload arrival in remote receive shards
- R1: generation and payload visibility

Ascend consumer epilogue lane:
- C5: contributor validation and slot resolution
- C6: direct copy or hidden reduction
- C7: routing-weight copy
- F0: Combine completion publication

CUDA reference lane:
- one dashed CUDA producer/transport semantic-reference block spanning the work corresponding to C0-C4 and T0-T1
- one dashed CUDA reduction-epilogue semantic-reference block spanning the work corresponding to C5-C7/F0
- show warp grid-stride work, TMA load/store, asynchronous payload movement, warp-cooperative reduction, and device-side dependency

The Combine arrows must show this order:

S2 -> C0 -> C1 -> C2 -> C3 -> C4 -> T0 -> T1 -> R0/R1 -> S3 -> C5 -> C6 -> C7 -> F0

Do not label any unmeasured hardware interval as "Idle". Empty lane space means only that no stage block is assigned there; it is not measured idle time.

Add a right-side relationship panel. It must show:

Dispatch variants:
- normal Dispatch: compact expert-major layout -> normal Combine
- cached Dispatch: reuse previous handle and destination slots -> normal Combine
- expanded Dispatch: one output slot per valid top-k expert route -> reduced Combine
- cached expanded Dispatch: correctness-only cached variant -> reduced Combine; use a gray dashed border

Combine variants:
- normal Combine: compact input, direct copy or limited contributor merge
- reduced Combine: expanded input, multi-contributor hidden reduction

The timeline blocks must contain only:
- exact stage ID
- short stage name
- a blank field written exactly as "mean ms: ___"

Do not render a parity table, source-code function names, file paths, or long implementation symbols anywhere in the image. The Markdown specification contains the authoritative implementation mapping. The image and that table correspond through the stage IDs only.

Add this exact clarification below the timelines:

"CUDA counterpart = semantic mapping; internal kernel decomposition differs. Multiple Ascend stages may correspond to work fused inside one CUDA kernel. Mapping does not imply one-to-one function or timing equivalence."

Visually emphasize these differences without duplicating stages:
- CUDA fuses notify, payload movement, communication issue, and dependency handling into fewer kernels.
- Ascend uses multiple ordered asc_vf_call stages plus a separate AICore transport service.
- Ascend producer packing and transport currently have a stage boundary.
- Ascend Dispatch has a host-assisted count and prefix bridge.
- Ascend reduced Combine has explicit contributor discovery and reduction stages.

Use this restrained legend:
- green = data movement
- purple = control and metadata
- blue = communication
- yellow = synchronization or wait
- orange outline = optimization target
- gray dashed block = CUDA semantic reference or cached variant
- blue double-headed arrow = possible overlap
- red barrier = required serialization

Add this exact note at the bottom:

"Stage timings are independently aggregated max-rank spans and are not drawn to scale. End-to-end latency must be measured with device events. Logical bandwidth target for P5 normal Dispatch: >2000 GB/s."

Before generating the final image, perform these consistency checks:
1. Every Dispatch and Combine stage appears exactly once in its assigned Ascend swimlane.
2. Dispatch host IDs are S0, H0, H1, H2, S1; Combine host IDs are S2 and S3.
3. There is no D9 and no HB0, HB1, or HB2.
4. Expert GEMM is shown between Dispatch and Combine and marked outside EP timing.
5. No parity table, source-code function name, or file path appears in the image.
6. The CUDA mapping is explicitly semantic, not one-to-one.
7. T1/R1 visibly precede H0/H1/H2 in Dispatch.
8. No empty region is labeled as measured idle time.
9. All text is legible at desktop viewing size and no text overlaps a border, arrow, or adjacent label.
```

## 12. Deliverables

P5 produces:

1. this specification;
2. the generated Dispatch/Combine companion timeline image;
3. a machine-readable Ascend/CUDA parity timeline report;
4. refreshed P4-final stage and end-to-end baseline data;
5. independently qualified P5.1-P5.5 candidates;
6. five-operation EP8 performance tables after every retained workstream;
7. the final 144-case, 720-operation gate; and
8. a final P4-to-P5 comparison stating whether normal Dispatch exceeds
   `2000 logical GB/s`.

P5 is not complete merely because every planned candidate has been tried. It
is complete only when the correctness gates pass, the final retained tree is
reported, and the primary target result is explicitly accepted or the
remaining hardware/runtime boundary is demonstrated with a minimal aligned
probe.
