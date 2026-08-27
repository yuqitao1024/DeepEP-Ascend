# EPv2 Ascend P6 Critical-Path Optimization

## 1. Goal and baseline

P6 targets the largest exposed production-path costs left after P5. The goal
is to raise the representative EP8 logical bandwidth quickly by shortening the
actual critical path, rather than by accumulating small control-path wins that
do not change end-to-end latency.

The fixed representative workload is:

```text
world_size=8, num_tokens=8192, hidden=7168
num_topk=8, num_experts=256, seed=0
workload fingerprint:
d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00
```

The retained P5.4 profile reports about `28.4 ms` for Dispatch and about
`105 ms` for both normal and Reduced Combine. Representative logical
bandwidth is therefore about `274 GB/s` for Dispatch and `102-104 GB/s` for
Combine, far below the roughly `3 TB/s` aggregate payload result of the HCCS
transport-only probe. The gap is in the production pipeline around transport,
not evidence that the HCCS payload link itself is ten times too slow.

## 2. Evidence carried from P5.5

P5.5 isolated three service/control hypotheses with same-binary ABBA and stage
profiles:

| Candidate | Measured mechanism | End-to-end decision | P6 treatment |
| --- | --- | --- | --- |
| Staged release fence elision | Release-payload span changed by about `+0.7%`; CQ wait regressed | Rejected | Do not retry without a different ordering model |
| Adjacent control-request doorbell batching | Control-release span fell `17-19%`, but it is only about `0.22M cycles`; Reduced Combine regressed | Rejected | Revisit only if WQE count is reduced, not merely doorbells |
| Single-entry execution metadata cache | Lookup reuse executed, but Combine did not improve and Reduced Combine regressed | Rejected | Do not optimize lookup before payload critical path |

These experiments confirmed an important priority rule: service micro-work is
measurable, but reducing it alone does not shorten the current Combine critical
path. P6 keeps these results as negative evidence instead of reintroducing the
discarded code.

## 3. Current critical path

The retained Reduced Combine profile contains:

| Span | Mean cycles | Meaning |
| --- | ---: | --- |
| C4 release-payload span | `53,093,252` | Local publication preparation plus transport service |
| Service submit | about `11.60M` | WQE/SQ submission and payload progress |
| CQ wait | about `1.53M` | Completion wait |
| Residual inside release-payload | about `39.96M` | Work before/around service not explained by submit and CQ |

The largest concrete operation in that residual is
`direct_combine_producer_local_copy_vf`: it copies the local rank's complete
staging shard to its receive shard one byte per SIMT iteration before payload
release. For `B = count[rank] * combine_record_bytes` and `T` SIMT threads, the
current loop performs

```text
thread t copies byte indices t, t + T, t + 2T, ... < B
```

That means every byte is a separate scalar global load/store operation. CUDA's
corresponding path uses cooperative bulk movement. P6.0 tests whether replacing
the aligned body with AICore GM-to-UB-to-GM `DataCopy` removes a material part
of the roughly `40M-cycle` residual.

## 4. Priority order

| Priority | Item | Target | Reason and expected signal |
| --- | --- | --- | --- |
| P6.0 | AICore local-rank Combine copy | C3/C4, about `40M` unexplained cycles | Largest bounded scalar byte loop; low protocol risk; expect a visible Combine and Reduced Combine reduction if attribution is correct |
| P6.1 | Chunked producer-to-transport pipeline | C2/C3/C4 serialization | Publish completed destination chunks while later chunks are packed; first true packing/communication overlap candidate |
| P6.2 | Parallel Combine slot-map construction | C5 serial validation reduction | Reuse the slot index already calculated by parallel validation; replace the second full record scan with atomic insertion and a small tile-error reduction |
| P6.3 | Release-payload path decomposition and acceleration | C4, about `16M cycles` after P6.2 | `release_payload` remains about `14.5M cycles`; split local DataCopy, remote command construction, service submit, and barrier tail before changing concurrency |
| P6.4 | Producer record packing | C2, about `9.5M cycles` | Compare record packing with the CUDA warp/TMA path and reduce repeated metadata or payload traversal without increasing command count |
| P6.5 | Consumer vector reduction | C6, about `7.4M cycles` | Improve GM read, BF16-to-FP32 accumulation, and output-store tiling after the serial C5 scan is gone |
| P6.6 | Persistent or fused scheduling | Cross-stage boundaries | Reconsider only when a profile exposes a material idle or overlap opportunity; P6.0 showed only about `28K cycles` idle |
| P6.7 | Remote payload concurrency and WQE reduction | Service and control tail | Require SQ/CQ and topology evidence; P5.5 proved doorbell-only control batching is too small |

The order is adaptive. After each retained item, a fresh profile is the source
of truth. P6.0 removed the scalar local copy, P6.1 proved that host-event
chunking adds more work than it hides, and P6.2 removed the next largest
measured serial span. P6.3 therefore starts from the new C4 breakdown rather
than assuming that service or launch overhead is dominant.

## 5. P6.0 design

### 5.1 Selection contract

`DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=1` enables the candidate only for
direct, non-hybrid Combine and selects the qualified `32768`-byte tile. Unset or
`0` selects the retained SIMT copy. Explicit `512`, `1024`, `2048`, `4096`,
`8192`, `16384`, and `32768` values are accepted for screening; other values
fail buffer construction. The selector travels in `CombineArguments` so
baseline and candidate use the same compiled binary.

The copy plan divides `B` bytes into:

```text
aligned_body = floor(B / tile_bytes) * tile_bytes
tail_bytes   = B - aligned_body
```

The AICore path copies `aligned_body` with fixed-size GM-to-UB-to-GM tiles.
The existing SIMT function copies only `[aligned_body, B)` for the candidate,
and `[0, B)` for the baseline. Zero bytes and invalid/unaligned plans remain
well-defined. Source and destination ranges must not overlap.

### 5.2 Ordering and protocol invariants

P6.0 does not change:

- staging and receive addresses or record layout;
- per-rank record counts and control-slot values;
- remote HCOMM put/flush commands;
- generation, barrier, CQ, or consumer protocol;
- hybrid Combine and non-direct fallback behavior.

The AICore copy must complete and reach DDR visibility before
`direct_combine_producer_release_vf` publishes local count/generation and
before the transport service executes. The existing release ordering remains
the authority; P6.0 adds only the execution-domain synchronization required
between DataCopy and the following SIMT release.

### 5.3 Validation and retention

Validation order is:

1. host copy-plan and selector tests, including invalid values and tails;
2. full local Ascend contract suite;
3. NPU extension compilation;
4. two-rank correctness at hidden `7168` and `7184`, with unset, explicit-zero,
   and two enabled generations across all five operations;
5. EP8 representative ABBA with 72 blocks, 30 warmups, and 30 samples;
6. an enabled stage profile to verify the mechanism.

The candidate is retained only when normal Combine and Reduced Combine improve
beyond pair noise, Dispatch operations do not materially regress, correctness
is unchanged, and the C4/release-payload profile moves in the expected
direction. There is no fixed percentage threshold. A flat profile rejects the
attribution even if one noisy end-to-end mean appears faster.

## 6. P6.1-P6.7 entry gates

P6.1 requires per-destination or per-chunk readiness metadata so transport can
consume completed chunks without observing partially packed records. P6.2
requires a cleanly initialized slot array, a native atomic CAS, and preserved
duplicate-record detection. P6.3 must first time local copy, remote request
construction, service execution, and barrier tail independently. P6.4 and
P6.5 require a fresh stage profile proving their span is still on the critical
path. P6.6 requires a material exposed gap or a persistent design that reduces
work as well as launches. P6.7 requires SQ occupancy, CQ latency, and
route/channel evidence; aggregate HCCS bandwidth alone is insufficient, and it
must not repeat the P5.5 doorbell-only experiment.

Every item uses an opt-in selector until the same-binary ABBA and profile gate
accept it. Rejected code is removed, while its measurements and conclusion are
kept in this document.

## 7. P6.0 measured decision

### 7.1 Tile screening

The first implementation used a `256`-byte tile. Correctness passed, but the
EP8 ABBA run (`task_20260827_055751_49745719724`) regressed normal Combine by
`92.04%` and Reduced Combine by `90.43%`. The enabled profile
(`task_20260827_060335_5125989534`) showed C4 growing from the retained roughly
`53M cycles` to `155-159M cycles`. C2, C5, and C6 stayed near their previous
values, so the root cause was the per-tile MTE event and synchronization cost,
not HCOMM or changed payload work.

The extension was rebuilt with four explicit tile instances in
`task_20260827_060816_52050625137`. The EP8 screening run
`task_20260827_061139_528455595` used 10 warmups and 10 samples per tile:

| Tile | Combine mean | Combine GB/s | Reduced Combine mean | Reduced Combine GB/s |
| ---: | ---: | ---: | ---: | ---: |
| Baseline SIMT | `104.430 ms` | `104.388` | `106.105 ms` | `102.740` |
| `512 B` | `134.068 ms` | `81.311` | `134.377 ms` | `81.124` |
| `1024 B` | `100.414 ms` | `108.562` | `103.066 ms` | `105.769` |
| `2048 B` | `78.627 ms` | `138.644` | `80.244 ms` | `135.850` |
| `4096 B` | `65.103 ms` | `167.445` | `67.884 ms` | `160.587` |

The monotonic improvement from `512 B` through `4096 B` confirms that fixed
per-DataCopy event cost dominates at small tiles. The later `8192-32768 B`
screening extended the same trend; `32768 B` is now the qualified default for
selector value `1`. All explicit values remain available as screening controls,
while `4096 B` is retained as the formal ABBA baseline.

### 7.2 Correctness and formal ABBA

The explicit `4096`-byte two-rank gate
`task_20260827_061957_5461893425` passed hidden sizes `7168` and `7184`, with
unset, explicit-zero, and two candidate generations. Dispatch, Expanded
Dispatch, Cached Dispatch, Combine, and Reduced Combine all passed.

The formal EP8 ABBA run `task_20260827_062329_55721432505` used 30 warmups and
30 samples for each A/B generation:

| Operation | Baseline mean | 4096B mean | Delta | 4096B logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | `29.721 ms` | `29.599 ms` | `-0.41%` | `263.055` |
| Expanded Dispatch | `37.217 ms` | `37.009 ms` | `-0.56%` | `250.078` |
| Cached Dispatch | `85.603 ms` | `85.078 ms` | `-0.61%` | `91.518` |
| Combine | `105.236 ms` | `66.646 ms` | `-36.67%` | `163.573` |
| Reduced Combine | `106.371 ms` | `67.785 ms` | `-36.27%` | `160.820` |

Baseline and candidate pair variation was below `1%` for both Combine
operations. Mean and p95 moved together, and no Dispatch operation regressed.

### 7.3 Stage attribution and retention

The explicit `4096`-byte profile `task_20260827_062745_57032732072` measured
rank-0 C4 at `15.30M cycles` for Combine and `15.45M cycles` for Reduced
Combine. Relative to the retained P5 profile's approximately `53.09M-cycle`
C4, this is about a `71%` reduction. C2 remained about `9.49-9.62M cycles`, C5
about `10.31M cycles`, and C6 about `7.38-7.39M cycles`, so the profile moved
only the intended local-copy/publication span.

P6.0 is retained. After this change, C4 is no longer the single overwhelming
span: C4, C5, C2, and C6 are now of the same order. P6.1 must therefore target
serialization across these spans, not another isolated scalar loop inside the
old C4 residual.

## 8. P6.1 source-tile producer/transport pipeline

### 8.1 Chosen decomposition

P6.1 chunks the existing `kCombineRecordsPerTile` producer tiles, not the
final output-slot space. After `direct_combine_producer_plan_prefix_vf`, entry
`prefix[tile][rank]` is the first output slot contributed by that source tile
to `rank`. For a source-tile interval `[a, b)`, the payload interval for rank
`r` is therefore:

```text
begin(r) = prefix[a][r]
end(r)   = b == tile_count ? total_count[r] : prefix[b][r]
bytes(r) = (end(r) - begin(r)) * combine_record_bytes
```

The interval is contiguous because tile prefixes are cumulative. The producer
record kernel processes only source rows covered by `[a, b)`. The transport
kernel copies exactly `[begin(r), end(r))` from each staging shard to the same
offset in the remote receive shard. This avoids the rejected alternative of
partitioning by output slot, which would rescan every source row once per
chunk merely to discover whether its final slot belongs to the chunk.

### 8.2 Phase-A two-stream schedule

The implemented phase-A schedule is deliberately simpler than the proposed
two-slot request design. The producer stream first runs control, plan, and
plan-prefix once. For each chunk it then runs record packing and records
`ready[c]`. The communication stream waits for `ready[c]`, performs the
local-rank DataCopy plus remote puts, synchronously executes the transport
service for that chunk, and records `done[c]`. The host enqueues all chunks;
it does not wait between chunks. There is no two-slot `DeviceRequest` reuse in
phase A.

After all chunks, the producer stream waits for the final `done`, publishes
count/generation control, executes the world barrier, and runs the unchanged
Combine epilogue. Counts and generation are never visible before every payload
chunk is complete. A chunk may arrive out of order because chunks write
disjoint receive intervals, but the final control publication remains the
single consumer-visible readiness point. An ACL event pair per chunk orders
the two streams. The launcher synchronizes both streams before destroying the
events because ACL event lifetime cannot end while queued waits still refer to
them.

### 8.3 Selector and scope

`DEEP_EP_ASCEND_COMBINE_PIPELINE_CHUNK_ROWS` controls the candidate. Unset or
`0` retains the single-stream pipeline. Value `1` selects the initial
`16384`-row default; explicit screening values must be positive multiples of
`kCombineRecordsPerTile`, produce at least two chunks, and stay within the
bounded host event array. The first implementation is limited to direct,
non-hybrid, non-stream-mode Combine with a distinct runtime communication
stream. Expanded and normal direct Combine share the same source-tile prefix
contract; unsupported modes disable the candidate rather than changing their
path.

### 8.4 Retention gate

The tests must prove chunk interval continuity, exact coverage without overlap,
bounded request-slot reuse, unchanged final control ordering, and selector
fallback. NPU validation repeats the two-rank hidden `7168/7184` gate, then
screens row sizes before a 30/30 EP8 ABBA. The candidate is retained only if
both Combine operations improve outside pair noise and a profile shows actual
C2/C4 overlap or a shorter combined producer/publication critical span. More
kernel launches without measured overlap reject the candidate.

### 8.5 Correctness incident and fix

The first two-rank run, `task_20260827_065632_61259628040`, failed with:

```text
command_overflow command_index=6
```

At world size two the fixed command capacity is `(2 - 1) * 5 + 1 = 6`.
Every chunk appended at least one put and one flush, while service execution
did not clear `queue->count`; later chunks therefore appended beyond the
array even though earlier commands had completed. Phase A resets the service
queue before every non-empty release segment. This is ordered after the
previous communication-stream release and service execution, so the reset
does not discard live commands.

The extension rebuilt successfully in `task_20260827_070018_61992129475`.
The corrected two-rank run `task_20260827_070243_62559319740` passed all five
operations for hidden sizes `7168` and `7184`, with the selector unset, `0`,
and `128` twice. The 512-token input forced more than one chunk, so this was
not a disabled-path-only correctness result.

### 8.6 EP8 screening and timeline decision

The representative EP8 screening task
`task_20260827_070732_63744411957` retained the P6.0 4096-byte local copy and
used 10 warmups plus 10 measured iterations:

| Chunk rows | Combine | Logical bandwidth | Reduced Combine | Logical bandwidth |
| ---: | ---: | ---: | ---: | ---: |
| `0` baseline | `66.663 ms` | `163.526 GB/s` | `69.420 ms` | `157.032 GB/s` |
| `8192` | `67.835 ms` | `160.702 GB/s` | `69.618 ms` | `156.586 GB/s` |
| `16384` | `68.714 ms` | `158.645 GB/s` | `71.550 ms` | `152.358 GB/s` |
| `32768` | `68.973 ms` | `158.050 GB/s` | `70.891 ms` | `153.775 GB/s` |

The best candidate, `8192`, regressed Combine by `1.76%` and Reduced Combine
by `0.28%`, so it did not enter the 30/30 ABBA gate. The enabled profile
`task_20260827_071151_64902729768` explains why:

| Rank-0 span | P6.0 Combine | P6.1 Combine | Delta | P6.0 Reduced | P6.1 Reduced | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 record envelope | `9.485M` | `15.029M` | `+5.543M` | `9.622M` | `14.757M` | `+5.134M` |
| C4 publication envelope | `15.306M` | `22.395M` | `+7.089M` | `15.452M` | `21.944M` | `+6.492M` |
| C2/C4 overlap | `0` | `11.472M` | `+11.472M` | `0` | `11.143M` | `+11.143M` |
| C2/C4 union | `24.793M` | `25.952M` | `+1.159M` | `25.076M` | `25.558M` | `+0.482M` |
| Whole operation envelope | `44.302M` | `45.476M` | `+1.174M` | `44.620M` | `45.121M` | `+0.501M` |

The pipeline did create real overlap. It lost because the work added to C2
and C4 was larger than the hidden interval. For rank 0, command count grew
from `30` to `40`, put count from `7` to `12`, service wait from about
`1.63M` to `7.06M cycles`, and total service span from about `2.41M` to
`9.35M cycles`. Chunked record launches also contend with simultaneous
release/service work, stretching the C2 envelope. The synchronous per-chunk
flush/service, repeated queue reset, kernel launches, and ACL event boundaries
are therefore the measured problem; lack of overlap is not.

P6.1 phase A is rejected and its opt-in production path is removed. Its useful
result is architectural evidence for later persistent scheduling: retain
source-tile readiness, but move chunk progress into a device schedule that
avoids a service completion and host-launched stage boundary per chunk. A
second two-stream host-event implementation is out of scope unless it first
proves that it reduces command/service count as well as creating overlap.

## 9. P6.2 parallel Combine slot-map construction

### 9.1 Why this moved ahead of stage fusion

The retained P6.0 profile contains only about `28K cycles` of device-timeline
idle, so removing launch gaps alone cannot recover a meaningful fraction of a
roughly `44M-cycle` Combine envelope. C5, however, is about `10.32M cycles`.
Its `epilogue_validate_reduce` substage accounts for about `9.71M cycles` and
runs only on `threadIdx.x == 0`.

Parallel `direct_combine_epilogue_validate_vf` already validates each received
record and calculates the final slot index. The old reduce stage discarded
that parallelism and scanned all received records again. In the representative
case this is roughly `43K` live records, followed by serial duplicate checks
and slot writes. CUDA's fixed rank layout and warp-level top-k mapping do not
need this second full scan.

### 9.2 Slot mapping and atomic rule

For token `t`, top-k contribution lane `l`, and `K = num_topk`, the flat slot
index is

```text
i(t, l) = t * K + l,       0 <= t < N, 0 <= l < K
```

`slots[i]` stores the receive-shard slot `s` for that contribution. The
contributor rank is still derived from the routed expert; the slot array does
not change layout. `direct_combine_epilogue_clear_index_vf` initializes every
entry to `-1`. After header, token, contributor, lane, and bounds validation,
the parallel validator executes

```text
old = CAS(&slots[i(t, l)], -1, s)

old == -1    first valid record; insertion succeeds
old != -1    another record already owns (t, l); report duplicate record
```

The `-1` value is only the empty-map sentinel. A valid receive slot is always
non-negative. If two records race for the same `(t, l)`, exactly one CAS can
observe `-1`; the other records `kDuplicateRecord` in its tile error. The
partially populated map is never consumed after an error because the following
reduce stage publishes the protocol failure first.

The old complexity was a parallel validation plus a serial `O(R)` second pass,
where `R` is the number of received records. P6.2 performs the insert during
the existing parallel pass, approximately `O(R/P)` work per active worker,
then reduces only `ceil(capacity / 128)` tile summaries. The representative
capacity has about `512` summaries, versus roughly `43K` live records.

### 9.3 Scope and correctness invariants

P6.2 includes `simt_api/device_atomic_functions.h` and uses the CANN 9.2 native
`asc_atomic_cas` on GM `int32_t` entries. It removes the record-index scratch
read and the contributor/record loops from
`direct_combine_epilogue_reduce_errors_vf`. It does not change record layout,
top-k semantics, contributor ordering, payload values, weights, generation,
release/acquire ordering, or the downstream reduction layout.

The two-rank gate covers normal Combine, multiple lanes from one contributor,
the top-k 8 common shape with repeated contributors and inactive `-1` lanes,
malformed-handle recovery, and five-operation round trips at hidden sizes
`7168` and `7184`. These cases prove that legitimate same-contributor lanes are
not mistaken for duplicate `(token, lane)` records.

### 9.4 Build and functional evidence

| Task | Result |
| --- | --- |
| `task_20260827_073513_68179622211` | CANN 9.2 `dav-3510` ASC compile and extension link passed |
| `task_20260827_073908_69037032178` | Four focused Combine cases and both hidden-size five-operation gates passed |
| `task_20260827_074134_696020228` | EP8 10-sample screening and stage profile passed |
| `task_20260827_074705_70623418640` | Two independent EP8 30-warmup/30-sample repeats passed |

The local Ascend source and contract suites also passed all `140` tests.

### 9.5 Timeline attribution

Rank-0 profile cycles compare as follows:

| Span | P6.0 Combine | P6.2 Combine | P6.0 Reduced | P6.2 Reduced |
| --- | ---: | ---: | ---: | ---: |
| Parallel validate | `0.549M` | `0.588M` | `0.549M` | `0.584M` |
| Validate reduce | `9.710M` | `0.600M` | `9.709M` | `0.595M` |
| Whole C5 | `10.319M` | `1.248M` | `10.317M` | `1.240M` |
| Whole operation envelope | `44.302M` | `35.985M` | `44.620M` | `36.809M` |

The atomic insert adds only about `35-39K cycles` to parallel validation while
removing about `9.11M cycles` from the serial reduction. C2 and C6 remain
effectively unchanged. C4 varies by less than about `1.3M cycles` between the
two captures, so the end-to-end movement is correctly attributed to C5.

### 9.6 End-to-end decision

The first 10-sample screening measured:

| Operation | P6.0 mean | P6.2 mean | Delta | P6.2 logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Combine | `66.663 ms` | `57.996 ms` | `-13.00%` | `187.966` |
| Reduced Combine | `69.420 ms` | `59.755 ms` | `-13.92%` | `182.431` |

Two 30-sample P6.2 repeats then measured:

| Operation | Repeat A mean / p95 | Repeat B mean / p95 | Logical GB/s A / B |
| --- | ---: | ---: | ---: |
| Combine | `58.529 / 60.223 ms` | `58.379 / 59.417 ms` | `186.252 / 186.731` |
| Reduced Combine | `60.018 / 61.435 ms` | `60.051 / 61.224 ms` | `181.633 / 181.531` |

The two Combine means differ by about `0.26%`, and the two Reduced Combine
means by about `0.06%`. Relative to the P6.0 formal means of `66.646 ms` and
`67.785 ms`, P6.2 improves Combine by about `12.2-12.4%` and Reduced Combine
by about `11.4-11.5%`. P6.2 is retained.

After P6.2, the largest rank-0 spans are C4 at about `16M cycles`, C2 at about
`9.5M`, and C6 at about `7.4M`. P6.3 therefore starts by decomposing the
roughly `14.5M-cycle` `release_payload` span; it does not assume that raw HCCS
or kernel-launch idle is the bottleneck.

## 10. P6.3 local-copy tile qualification

P6.3 kept the same AICore GM-to-UB-to-GM implementation and screened larger
tiles after the initial `4096 B` result. The independent profile task
`task_20260827_081617_7694133954` used the representative EP8 case, 72 blocks,
and `DEEP_EP_ASCEND_COMBINE_LOCAL_COPY_DATACOPY=32768`. It measured rank-0
`producer_local_copy` at about `2.1-2.3M cycles`; `release_payload` remained
about `1.7M cycles`, while C2 record stayed near `9.7-9.9M` and C6 reduction
near `7.6M`. This confirms that the larger tile does not move work into the
transport or epilogue stages.

The formal same-binary ABBA task `task_20260827_082157_79555523701` ran the
order `4096, 32768, 32768, 4096`, with 30 warmups and 30 samples per run:

| Operation | 4096 B mean / p95 | 32768 B mean / p95 | Mean delta | 32768 B logical GB/s |
| --- | ---: | ---: | ---: | ---: |
| Dispatch | `29.705 / 31.364 ms` | `30.996 / 32.254 ms` | `+4.35%` | `251.253` |
| Expanded Dispatch | `37.035 / 38.317 ms` | `37.838 / 40.772 ms` | `+2.17%` | `244.691` |
| Cached Dispatch | `85.544 / 86.978 ms` | `86.467 / 88.122 ms` | `+1.08%` | `90.051` |
| Combine | `57.792 / 59.222 ms` | `47.751 / 49.646 ms` | `-17.37%` | `228.297` |
| Reduced Combine | `58.832 / 60.174 ms` | `50.262 / 52.216 ms` | `-14.56%` | `216.970` |

The default-selector correctness gate
`task_20260827_093140_161200320447` passed the four two-rank Combine cases
(`normal`, duplicate same-rank experts, specialized top-k 8, and malformed
handle), and all four ABBA reports passed the five-operation correctness
contract. The Dispatch
differences are cross-run drift rather than a selector-dependent path change:
the tile is read only by `combine_kernel`, and the three Dispatch operations
execute before it. They are recorded here because this ABBA did not isolate a
Dispatch-only pair; a future tuning change must rerun a Dispatch-only ABBA if
those operations become a concern. Combine p95 follows the mean reduction, and
the two 32768 B runs agree within `0.8%` for Combine and `3.9%` for Reduced
Combine, so the tile-size signal is substantially larger than pair noise.

P6.3 therefore changes the `=1` recommendation from `4096 B` to `32768 B`.
The environment variable remains opt-in, hybrid and non-direct paths remain
unchanged, and explicit tile values remain available for shape-specific
screening. This is a local staging optimization, not an HCCS bandwidth claim:
the payload transport protocol, command count, and release ordering are
unchanged.

## 11. P6.4 producer-rank decode screening

The C2 profile still contains a per-row scan of `workspace_rank_values` to
recover the destination rank. Because `source_metadata[row][1]` also encodes
the validated master rank, P6.4 briefly tested replacing that scan with
`decode_dispatch_source_rank`. The implementation compiled on `dav-3510` as
part of task `task_20260827_093815_166260119861`, and its 10-warmup/10-sample
representative run passed all five operations:

| Operation | P6.3 32768 B ABBA mean | P6.4 screening mean | P6.4 logical GB/s |
| --- | ---: | ---: | ---: |
| Dispatch | `30.996 ms` | `28.925 ms` | `269.180` |
| Expanded Dispatch | `37.838 ms` | `35.671 ms` | `259.355` |
| Cached Dispatch | `86.467 ms` | `85.285 ms` | `91.295` |
| Combine | `47.751 ms` | `47.967 ms` | `227.266` |
| Reduced Combine | `50.262 ms` | `48.510 ms` | `224.719` |

Combine is effectively unchanged and the run has no stage profile proving a C2
reduction. A follow-up profile task could not allocate streams because other
8-rank jobs occupied the host (`Resource_Error: Resources are exhausted`), so
there is no valid attribution or formal ABBA for this candidate. The change is
therefore rejected and removed. The original rank-begin scan remains the
defensive path for malformed metadata; P6.4 should only be revisited together
with a producer layout that carries a trusted destination-rank side array and
reduces a measured C2 span.

## 12. P6.5 consumer vector-reduction tile candidate

After P6.2 and P6.3, C6 `direct_combine_epilogue_vector_reduce_impl` is the
largest remaining compute span on the consumer side (about `7.4M cycles`).
The implementation currently processes `256` BF16 elements per AICore tile.
For each tile it allocates and drains a VECIN tensor for every contributor and
for each optional bias, then casts, accumulates, casts back, and stores. The
per-tile queue and MTE/VEC instruction overhead is therefore paid
`ceil(hidden / 256)` times per token.

P6.5 adds an opt-in `DEEP_EP_ASCEND_COMBINE_VECTOR_REDUCE_TILE` selector:

| Value | Meaning |
| --- | --- |
| unset or `0` | Retained `256`-element implementation |
| `1` or `512` | Candidate `512`-element common-shape AOT instance |

The candidate is selected only for direct, non-hybrid K=8/H=7168. Dynamic
shapes and hybrid paths retain the existing implementation. For the
representative shape, reduction iterations change from
`7168 / 256 = 28` to `7168 / 512 = 14`; contributor traversal and BF16-to-FP32
accumulation order do not change. The larger tile uses twice the UB for
input/output and accumulation scratch, so compilation must confirm no
register or local-memory spill before any performance decision.

The candidate does not change record layout, slot mapping, transport commands,
release ordering, or output rounding. `256` remains the baseline in the same
binary so ABBA measurements isolate tile size. Acceptance requires host
selector and AOT call-site tests, NPU8P compile/link, two-rank correctness for
hidden `7168` and `7184` (including no-bias, one-bias, two-bias, duplicate
same-rank experts, malformed handles, and two generations), EP8 10/10
screening, and a stage profile showing C6 reduction work decreases. Only then
will a 30-warmup/30-sample ABBA run be considered. A flat C6 profile or a
noisy/negative end-to-end result rejects P6.5 and the selector is removed.

### 12.1 Build and correctness evidence

The NPU extension build `task_20260827_103200_28675128835` completed
successfully for `dav-3510`, including compilation of `combine.asc` and final
extension linking. The two-rank correctness task
`task_20260827_103542_292883530876` passed the normal, duplicate-same-rank
expert, specialized K=8/H=7168, and malformed-handle cases. The selector did
not change output values or protocol validation.

### 12.2 Same-shape screening

The two-rank 10-warmup/10-sample screening task
`task_20260827_103806_29422868167` compared the two instances from the same
binary:

| Operation | 256 baseline | 512 candidate | Delta |
| --- | ---: | ---: | ---: |
| Dispatch | `19.174 ms` | `19.845 ms` | `+3.50%` |
| Expanded Dispatch | `22.548 ms` | `22.634 ms` | `+0.38%` |
| Cached Dispatch | `68.184 ms` | `68.737 ms` | `+0.81%` |
| Combine | `23.924 ms` | `22.675 ms` | `-5.22%` |
| Reduced Combine | `82.401 ms` | `80.912 ms` | `-1.81%` |

Normal Combine logical bandwidth increased from `49.08` to `51.78 GB/s`;
Reduced Combine increased from `14.25` to `14.51 GB/s`. The selector is not
read by any Dispatch path, so the Dispatch deltas are cross-process pair drift
rather than candidate work.

### 12.3 Stage attribution and retention

The baseline and candidate profiles ran successfully on devices 0 and 1 in
`task_20260827_104304_296772130303` and
`task_20260827_104427_297509419459`. An earlier candidate submission on
devices 6 and 7, `task_20260827_104043_29542263948`, failed before kernel
entry with stream resource allocation failure and is excluded from the
comparison.

| Profile span | 256 baseline | 512 candidate | Delta |
| --- | ---: | ---: | ---: |
| Combine C6 `epilogue_reduce` | `2,768,992` | `1,514,865` cycles | `-45.29%` |
| Combine `consumer_compute` | `2,788,878` | `1,534,644` cycles | `-44.97%` |
| Combine device envelope | `13,792,256` | `13,152,227` cycles | `-4.64%` |
| Reduced C6 `epilogue_reduce` | `2,765,884` | `1,515,967` cycles | `-45.19%` |
| Reduced `consumer_compute` | `2,786,537` | `1,536,329` cycles | `-44.87%` |
| Reduced device envelope | `71,725,462` | `70,617,104` cycles | `-1.55%` |

The C6 span falls by about half, matching the reduction in tile iterations
from 28 to 14. Other Combine compute spans remain effectively unchanged. The
smaller end-to-end effect on Reduced Combine is expected because its producer
record span is still about `60.5M cycles`; C6 is only a small fraction of that
operation's critical path.

P6.5 is retained as an opt-in qualified candidate. It is not made the default
yet because the current evidence is two-rank, while the fixed acceptance
target is EP8. A later EP8 ABBA can promote value `512` to the default without
changing the AOT implementation.

## 13. P6.6 persistent or fused scheduling gate

P6.5 does not create evidence for launch fusion. Its candidate profile reports
only `34,813 cycles` of idle device time for Combine and `34,197 cycles` for
Reduced Combine. That is about `0.26%` and `0.05%` of their device envelopes,
respectively. Removing every exposed launch gap would therefore be much
smaller than the remaining producer, payload-release, and consumer spans.

P6.6 is deferred at its entry gate. Persistent scheduling should be revisited
only with a design that also removes repeated work or enables overlap without
the per-chunk service inflation observed in P6.1; launch fusion alone is not a
current high-priority candidate. P6 proceeds to P6.7 remote payload
concurrency and WQE reduction.
