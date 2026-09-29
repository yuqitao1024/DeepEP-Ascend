+# Dispatch release send-path optimization

## Status

- Date: 2026-09-29
- Scope: native AICore URMA transport service used by Normal Dispatch.
- Baseline: commit `1cc2655`, diagnostic build with profile ABI 7.
- Current priority: P2, safe non-inline 16-byte control-slot publication.
- P1 status: complete and validated remotely; it is measurement-only and has
  no performance claim.
- Current protocol changes: `kPutControlSlot`, symmetric-window reserve
  geometry/ABI v9, and per-peer two-generation control-source slots.

## Baseline evidence

The 8-rank canonical profile used 8192 tokens, hidden 7168, top-k 8, 256
experts, FP8, 64 AIV, 30 warmups and 30 iterations. It is attribution data,
not a formal no-profile performance number.

| Metric | Cycles | Approximate time |
| --- | ---: | ---: |
| `release_payload` stage | 930,259 | 930 µs |
| `epilogue_copy` stage | 319,053 | 319 µs |
| `release_control` stage | 278,089 | 278 µs |
| `producer_record` stage | 172,615 | 173 µs |
| service active | 1,527,900 | 1,528 µs |
| flush command | 899,979 | 900 µs |
| CQ drain | 808,773 | 809 µs |
| explicit payload-flush drain | 798,129 | 798 µs |
| control command | 201,992 | 202 µs |
| payload command | 73,710 | 74 µs |
| release VF payload command construction | 107,592 | 108 µs |
| release VF peer-control command construction | 212 | 0.2 µs |

The stage values are per-rank maxima and are not additive across rows. The
dominant attributable stage is the release payload flush/CQ completion path.

## Send path

The path crosses two execution domains:

1. The producer release VF is a SIMT vector function.
2. The native URMA executor is AICore code and uses MTE for SQ writes.

### Phase A: SIMT release VF

`direct_dispatch_persistent_release_vf` lets lane zero wait for a ready
pipeline slot and invokes `direct_dispatch_producer_release_body`.

The release body:

1. Walks destination ranks.
2. Computes the staging source and receive-window destination.
3. Calls `release_protocol::put_staged_records_striped`.
4. That helper splits records over available channels.
5. Each `put_on_channel` appends a high-level `TransportCommand` to the GM
   command queue.
6. `flush_payload` appends a flush command.
7. Control publication appends count, generation, and signal commands.

This phase constructs transport commands. It does not construct URMA WQEs.

Measured release-VF command construction is already small for control:
local control is 66 cycles, all peer control command construction is 212
cycles. Therefore preparing the existing transport commands earlier has no
remaining hiding space.

### Phase B: AICore transport service

`execute_body` validates commands and executes them in queue order.

For a payload `Put` command:

1. Resolve peer, channel, local buffer, and remote buffer.
2. Snapshot the SQ and remote buffer descriptors.
3. Build a URMA write request with `urma::make_write`.
4. `post_request` checks queue capacity.
5. `copy_request` stages the request into UB and uses `AscendC::DataCopy` to
   write 64-byte blocks into the SQ ring.
6. `post_request` updates SQ head and writes the SQ doorbell.

For a `Flush` command:

1. `drain_all` visits every peer and channel.
2. `drain_channel` reads SQ head and CQ tail.
3. It polls the CQE owner/status word with a cache-bypassing scalar load.
4. It validates status and advances its local CQ tail.
5. After all expected completions are observed, it writes CQ tail, CQ
   doorbell, and SQ tail.

Control commands use the same WQE/SQ/doorbell path with inline-write or
signal requests.

## Decision: four ordered work items

### P1. Sub-stage attribution for WQE/SQ and CQ drain

Add measurement-only fields for:

- SQ snapshot and WQE construction.
- Queue-capacity check.
- UB staging and SQ `DataCopy`.
- SQ head and doorbell publication.
- SQ head and CQ tail reads.
- CQE owner polling.
- CQE status validation.
- CQ/SQ tail and CQ doorbell stores.

No protocol, command order, queue depth, or reuse rule changes.

Exit criterion: quantify how much of the approximately 798 µs explicit
flush drain is true completion wait versus software bookkeeping.

### P2. Evaluate control WQE consolidation

Only start after P1 shows that WQE construction or SQ publication is a
material part of the approximately 202 µs control command time.

Candidate directions:

- Combine count and generation into one control-slot write.
- Pre-pack the control slot and issue one write per peer.
- Reduce the current three control commands per peer.

Constraints: preserve payload-before-control ordering, remote layout
compatibility, error reporting, and two-generation buffer reuse.

### P3. Prepare control WQEs during payload completion wait

Only start if P1 shows a stable software interval that can be hidden without
moving control visibility.

Candidate protocol:

1. Build control WQEs and write their SQ entries while payload completion is
   pending.
2. Do not publish their SQ head/doorbell.
3. After payload completion succeeds, publish control SQ entries.
4. Preserve payload-before-control visibility.

Required validation: two-generation reuse, error injection, failed-payload
rollback, SQ capacity safety, and 8-rank no-profile ABBA.

### P4. Evaluate doorbell batching

Only start if P1 shows that per-request SQ head/doorbell publication is
material. Candidate: write several SQ entries and publish one doorbell.

This is lower priority because there are only about seven payload requests per
rank and current payload command time is approximately 74 µs.

## Measurement contract

- Profile fields are diagnostic only and compiled out in normal builds.
- Cycle buckets must state whether they are disjoint, nested, or sampled.
- Per-rank maxima are not additive.
- Profile-on device timings must not be mixed with formal no-profile
  benchmark results.
- Any retained optimization requires repeated no-profile A/B or ABBA runs and
  full functional validation.

## P1 remote result (2026-09-28)

The 8-rank diagnostic build was task `task_20260928_194350_15735403956` on
devices 0-7. Correctness passed one canonical case. The remote artifact
SHA-256 was `93604f704db13a8d211f8305e3f63cbd7e0934116b2044aa41fd6f72d288abed`.
Profile-on dispatch mean was 3.500319 ms. This timing is attribution-only and
is not comparable with the formal no-profile baseline.

| Metric | Cycles |
| --- | ---: |
| `release_payload` | 605,865 |
| `epilogue_copy` | 198,799 |
| `release_control` | 294,620 |
| `producer_record` | 137,666 |
| service active | 1,191,007 |
| flush command | 574,932 |
| CQ drain | 482,607 |
| explicit payload-flush drain | 471,136 |
| control command | 214,756 |
| payload command | 75,675 |

Send-path per-rank maxima:

| Field | Cycles |
| --- | ---: |
| `post_sq_snapshot_cycles` | 173,307 |
| `post_queue_check_cycles` | 1,869 |
| `post_sq_copy_cycles` | 19,224 |
| `post_sq_publish_cycles` | 157,636 |
| `drain_head_tail_load_cycles` | 2,494 |
| `drain_cqe_poll_cycles` | 475,736 |
| `drain_cqe_status_cycles` | 643 |
| `drain_tail_doorbell_cycles` | 578 |

On rank 0, CQE owner polling was about 98.4% of CQ drain, while SQ snapshot
and SQ publication were about 59.5% and 54.5% of payload-plus-control command
time, respectively. The queue check, CQ bookkeeping, CQE status parsing, and
tail/doorbell stores were negligible. There were 31 commands, seven payload
writes, and about 174 MB of profiled payload bytes; SQ/CQ high watermark was
three and both final depths were zero.

Conclusions:

1. CQ drain is dominated by real completion wait, not software bookkeeping.
2. SQ snapshot and publication have material software overhead, so P2 is
   justified.
3. P3 remains plausible but must be re-evaluated after P2.
4. P4 remains low priority: request count is small and queue checking is tiny.

Reporting anomaly: the remote report contained send-path attribution in every
per-rank service record, but the top-level dispatch aggregation omitted it.
The aggregator code is contract-tested locally. The likely cause is that the
captured profile came from the report slot named `expanded_dispatch` rather
than the direct `dispatch` slot. This is a reporting-path issue, not an
instrumentation failure, and will be checked on the next profile run.

## P1 implementation record (2026-09-28)

P1 is implemented as a measurement-only change:

- Profile ABI is upgraded from 7 to 8.
- Eight send-path counters are appended to the profile structure:
  - `post_sq_snapshot_cycles`
  - `post_queue_check_cycles`
  - `post_sq_copy_cycles`
  - `post_sq_publish_cycles`
  - `drain_head_tail_load_cycles`
  - `drain_cqe_poll_cycles`
  - `drain_cqe_status_cycles`
  - `drain_tail_doorbell_cycles`
- Host export adds `service.release_attribution.send_path_attribution`.
- Benchmark aggregation uses per-rank max and marks the result as
  `max_per_rank_not_additive`.
- No command order, WQE format, SQ/CQ ownership, doorbell update, error check,
  or buffer reuse rule is changed.

Buckets are nested subsets:

- Post buckets are subsets of payload or control command execution time.
- Drain buckets are subsets of CQ drain time.
- The post buckets must not be summed with their parent command buckets, and
  drain buckets must not be summed with CQ drain.

Local validation:

- `PYTHONPATH=. pytest -q tests/ascend/test_benchmark_contract.py
  -k 'send_path or release_phase or cq_drain or epilogue_copy'`
  passed 5 tests.
- `pytest -q tests/ascend/test_transport_contract.py` passed 3 tests.
- `git diff --check` passed.

## P2 rejected candidate: 16-byte inline WQE (2026-09-28)

The first P2 candidate changed Normal Dispatch control publication from two
peer writes to one 16-byte inline WQE:

1. `kPutControlSlot` is appended to the opcode enum to preserve existing
   numeric opcode values.
2. The command carried generation and count in two adjacent inline words and
   validated `value_bytes == 16`.
3. The native service posted an `InlineWrite128Request`; the WQE remained a
   64-byte SQ entry and wrote 16 inline bytes.
4. `release_protocol::publish_control_slot_and_release` issues the combined
   write followed by the unchanged release signal.
5. Direct Normal Dispatch uses the new helper. Combine, hybrid, expanded, and
   cached paths keep the original two-write helper.
6. The stub backend accepted the call as a no-op.

This candidate is rejected. The known-good inline encoding covers only the
existing 8-byte request. The candidate extrapolated the inline-length field in
`word1` from 8 to 16 bytes, but that value has no official semantic contract
in the available URMA ABI material. A malformed or unsupported value can make
the device reject the WQE before the remote write is observable.

The two retries failed deterministically in the same first dispatch execution,
with an error CQE and NPU device error type 4 / UB LINK ERROR 507056 on rank 1.
The peer changed from device 3 to device 4, which is inconsistent with a fixed
link failure and consistent with malformed or unsupported WQE encoding. No
result JSON was produced, so neither run is a correctness or performance pass.

The current code contains no 16-byte inline request type. `make_sqe` is
restored to the known-good 8-byte inline encoding, and Normal Dispatch does
not extrapolate that field. Do not retry this candidate through the full
DeepEP dispatch path.

## P2 safe candidate: non-inline 16-byte control-slot write

### Design

The retained design still combines the two 8-byte control words into one
16-byte remote write, but uses the known-good non-inline URMA write request:

1. For each destination, the release VF writes `generation` and `count` into
   a 16-byte source slot in the symmetric-window reserve region.
2. It executes `system_fence()` so both source words are globally visible
   before the transport command is published.
3. It appends one `kPutControlSlot` command carrying the remote control-slot
   destination and local source-slot address.
4. The AICore service resolves both addresses through the registered local and
   remote buffer tables.
5. It builds `urma::make_write(..., bytes = 16)`, which is the same WQE family
   already used by payload writes.
6. The separate release signal remains after the control write.

The official SIMT backend maps the command to one 16-byte `WriteNbi`, and the
stub backend remains a no-op for host-only probes.

### Expected cost model

The candidate removes one of the two per-peer control WQEs. Its main expected
benefit is therefore lower SQ snapshot and SQ publication work, along with one
fewer command and CQE. It adds one local-buffer lookup and a 16-byte source
read. That added work is small, but it is not assumed to be free.

The candidate is useful only if the no-profile A/B shows an end-to-end Normal
Dispatch improvement while correctness passes. If control command time remains
dominated by completion or the added source lookup erases the WQE reduction,
the candidate should be rejected.

### Source-slot ownership

The local rank's own `DispatchControlSlot` is not a valid source. Remote
control slots are indexed by sender rank, while the count being published is
the number of records sent to each destination rank. One local slot cannot
represent the different counts for all peers.

A shared staging slot is also unsafe. A non-inline WQE stores a source address
and may read the source after command publication; writing the next peer's
control words into the same location could change bytes that a pending WQE has
not yet read.

The safe layout is therefore:

```text
slot_index = (generation & 1) * world_size + destination_rank
slot_bytes = 16
reserve_bytes = align(2 * world_size * 16, 32)
```

Separate destination slots prevent same-generation publication from overwriting
a pending source. The generation parity matches the existing two-generation
reuse contract: generation `g+2` must not begin while generation `g` is still
outstanding. This invariant is part of the correctness validation.

### Layout and ABI

`kSymmetricWindowAbiVersion` is upgraded from 8 to 9 because the reserve
region changes from a fixed 32 bytes to the two-generation per-peer source-slot
area. The public struct layout and field offsets do not change.

For world size 4, reserve bytes become 128. The direct reserve offset remains
45408 because preceding regions are unchanged. In the hybrid probe layout, all
regions after reserve shift by the 96-byte growth over the old 32-byte reserve.

`TransportCommand` remains ABI 3: the existing `source` field is unused by
`kPut` and is now interpreted only by `kPutControlSlot`. The opcode is
appended after all existing values. The P1 profile ABI remains 8.

### Validation plan

1. Build the production-shape candidate with diagnostics off.
2. Run the 8-rank canonical correctness case.
3. Run the weight-boundary regression: 8 ranks, 17 tokens, hidden 4865, BF16,
   top-k 2, 16 experts, masked ratio 0.5, and consumer tile 512.
4. Run the 8-rank no-profile ABBA benchmark.
5. Retain the change only if correctness passes and the end-to-end A/B shows a
   reproducible improvement. Do not use profile-on timing as the performance
   result.

### 16-byte inline probe trigger

The inline variant may be reconsidered only if all of the following hold:

1. The non-inline candidate passes correctness and shows an end-to-end gain.
2. Control command time remains material in the no-profile A/B.
3. The added local source lookup/read is shown to limit the gain.

If triggered, the next step is a standalone minimal URMA probe, not a DeepEP
dispatch run. It must establish the official 16-byte inline `word1` encoding,
verify CQE behavior across two devices, and only then propose a main-path
change.

### Invariants retained by the safe candidate

- Payload commands remain before control commands.
- The release signal remains separate.
- Remote control-slot layout and generation/count semantics are unchanged.
- Two-generation reuse rules are unchanged.
- Profile payload accounting counts a control-slot write as 16 bytes.

### Remote validation of the rejected inline candidate (2026-09-28)

The diagnostic candidate built successfully in task
`task_20260928_201213_40677695296`. The candidate extension SHA-256 is
`876563db202efea09250deb568efff14e3745ee7b05e9cb8cf7b120b0e026270`.
Build A used diagnostics ON. The source snapshot and production extension
were restored after the build.

The first 8-rank profile validation, task
`task_20260928_201307_408079722809`, reached the profiled canonical case
after queueing for eight devices. Rank 1 then reported an NPU device error
type 4, UB LINK ERROR 507056, with an error CQE between devices 1 and 3. No
result JSON was produced. The task wrapper reported exit 0 because the
launcher's terminal state was mishandled; the run itself is failed and must
not be interpreted as a pass.

Interpretation of the rejected inline candidate:

1. This is a device-level CQE failure, not a Python assertion or host-side
   protocol validation failure.
2. The 16-byte inline WQE remains the primary suspect because the failure
   appeared during the first profiled dispatch execution after the candidate
   was loaded.
3. A clean retry is useful to distinguish a transient host/link fault from a
   deterministic WQE encoding problem. If the same error reproduces, P2 should
   be rejected or redesigned rather than measured for performance.

### Inline-candidate rejection (2026-09-28)

A second clean 8-rank run, task `task_20260928_202304_19833421052`, reproduced
the same failure in the same profiled dispatch phase. Rank 1 again reported
NPU device error type 4, UB LINK ERROR 507056 and an error CQE; the peer
changed from device 3 to device 4, while the first run used peer device 3.
The corrected wrapper propagated `RUN_RC=1`, and no result JSON was produced.
The production extension was restored to SHA-256
`d08352909619bc12e95ec2fc2d03ff4dfaef78ccb7b5b08dfae7350593bd1cad`.

Decision: reject the 16-byte inline-write P2 candidate. The error is
deterministic at the same execution point across retries, with the peer rank
changing, which points to malformed or unsupported inline WQE encoding rather
than a stable 1-3 or 1-4 link fault. Do not run performance measurements with
this candidate.

### Safe P2 remote validation (2026-09-29)

The retained candidate is the non-inline 16-byte control-slot write. The
candidate extension SHA-256 is
12983dadae4e83dd0bed9b696e0252f13772685049fb123a68e29786f921fba8; the
baseline extension SHA-256 is
d08352909619bc12e95ec2fc2d03ff4dfaef78ccb7b5b08dfae7350593bd1cad.

Correctness results:

| Case | Result | Exit behavior |
| --- | --- | --- |
| 2-rank small | failed=0, passed=1, pending=0 | Rank 1 SIGSEGV after the complete JSON and passing summary; accepted under the agreed teardown-only policy |
| 4-rank tail | failed=0, passed=1, pending=0 | Exit 0 |
| 8-rank canonical | failed=0, passed=1, pending=0 | Exit 0 |
| 8-rank weight boundary (17 tokens, hidden 4865, BF16, top-k 2, 16 experts, masked ratio 0.5, consumer tile 512) | failed=0, passed=1, pending=0 | Exit 0 |

The installed extension was restored to the baseline SHA after every run,
including the final ABBA run.

No-profile canonical ABBA used 8 ranks, 8192 tokens, hidden 7168, top-k 8,
256 experts, FP8, 30 warmups, 30 iterations, and a 2000-us rank-launch
deadline. A is the baseline extension and B is the safe P2 candidate.

| Group | A mean (ms) | B mean (ms) | B gain |
| --- | ---: | ---: | ---: |
| 1 | 3.199198 | 3.183852 | 0.480% |
| 2 | 3.254363 | 3.174189 | 2.464% |
| 3 | 3.227211 | 3.177750 | 1.533% |
| Overall step mean | 3.226924 | 3.178597 | 1.498% |

All 12 ABBA steps passed correctness. The overall dispatch mean improves by
48.327 us. Logical bandwidth rises from 2412.983 GB/s to 2451.098 GB/s. The
remaining gap to the 3 TB/s target time of 2.595363 ms is 0.583234 ms.

Decision: retain safe P2. The benefit is positive in all three interleaved
groups and is not measured under profile-on timing. The magnitude is modest,
so it should not be described as closing the remaining 3 TB/s gap; it reduces
that gap by about 1.5% of the current Normal Dispatch mean.

### D3.3 re-profile after safe P2 (2026-09-29)

The D3.3 diagnostic build was repeated on the retained safe-P2 tree
2516f6b in a fresh source snapshot. The diagnostic extension SHA-256 is
3f921344f748d83ba58d7e562e5453e679a38d0141ea7a4d31cccc871c215f91. The
8-rank profile task was task_20260929_024653_362689812552; its correctness
summary was failed=0, passed=1, pending=0. The result SHA-256 is
fc9922813bfb4f5dd900d55723c4ebf06c5521d6e173030d84a0cfed88bf24cd.

Compared with the pre-P2 D3.3 run, the per-rank means changed as follows:

| Metric | Pre-P2 | Safe P2 | Change |
| --- | ---: | ---: | ---: |
| control_command_cycles | 191,610 | 138,595 | -53,015 |
| flush_command_cycles | 893,924 | 894,276 | +352 |
| payload_command_cycles | 70,376 | 69,901 | -475 |
| barrier_command_cycles | 218,524 | 199,278 | -19,246 |
| service_active_cycles | 1,460,976 | 1,388,600 | -72,376 |

The profile-on dispatch mean in this diagnostic run was 4.245499 ms; this is
attribution-only and is not a no-profile performance result. Explicit
payload-flush CQ drain was about 803,565 cycles and queue-full drain remained
zero, so the available hiding window remains large. Send-path per-rank maxima
were:

| Field | Cycles |
| --- | ---: |
| post_sq_snapshot_cycles | 138,682 |
| post_queue_check_cycles | 1,518 |
| post_sq_copy_cycles | 11,704 |
| post_sq_publish_cycles | 124,371 |
| drain_head_tail_load_cycles | 2,360 |
| drain_cqe_poll_cycles | 806,291 |
| drain_cqe_status_cycles | 396 |
| drain_tail_doorbell_cycles | 914 |

Interpretation:

1. Safe P2 reduced control command work by about 53k cycles, but SQ snapshot
   and SQ publication remain material software costs.
2. CQ drain is still dominated by true completion polling; queue bookkeeping
   and queue-full drains are negligible.
3. The remaining P3 direction is therefore still open, but it must be a
   prebuild experiment rather than an early control publication.

The minimal P3 candidate is:

1. While the explicit payload flush is waiting, pre-resolve the peer, buffers,
   SQ context, and SQ slot for control requests, and pre-stage WQE bytes.
2. Do not update SQ head or doorbell until the payload flush has completed
   successfully.
3. On payload completion, publish only the already prepared SQ entries.
4. Preserve payload-before-control order, two-generation reuse, error
   handling, and the release signal.

Acceptance gates for P3 remain: source contracts, two-generation/error
injection, 2/4/8-rank correctness, and at least three no-profile ABBA groups.
If the prebuilt SQ state is invalidated by completion or a peer update, the
candidate must fall back to the normal post path rather than publish stale
ownership bits.

## P3a experiment: conservative prebuild, not retained (2026-09-29)

P3a was implemented and measured, then removed from production after the
performance gate failed. The experiment was confined to the native AICore
service and only to a command sequence that exactly matches Normal Dispatch's
retained release shape:

1. payload writes;
2. explicit flush;
3. alternating control-slot write and signal-set commands, optionally ending
   at a barrier.

Any other sequence disabled prebuilding. The implementation did not alter the
SIMT command producer, command ABI, WQE encoding, control-slot layout, or
two-generation source-slot ownership.

Before the explicit flush waited for completion, the service resolved each peer,
registered buffer, remote control slot, and remote signal destination; took
one SQ head snapshot per peer; checked CQ capacity for the pair of control
requests; constructed the known-good non-inline 16-byte control write WQE and
the known-good inline 8-byte signal-set WQE; and wrote both WQE bytes into
their SQ slots through the existing MTE3 path.

It did not update SQ head or SQ doorbell before the flush succeeded. If flush
failed, the batch was marked unusable and no prebuilt request was published.

When the original command was reached, the service re-resolved the peer and
compared channel pointers, CQ pointers, SQ pointers, world peer, channel index,
and the full SQ head value against the prebuild snapshot. On any mismatch it
invalidated that request and executed the original post path. This prevents
publishing stale slot or owner bits after a queue-full drain or peer
reconfiguration.

The experiment still published the two prebuilt entries one at a time:
each publication updated SQ head and wrote one doorbell. Batch doorbell
publication remains P3b and is intentionally not enabled without an
independent URMA semantic probe.

The optimization moved resolution, one SQ snapshot, WQE construction, and SQ
copy into the payload-completion wait. It did not move remote control
visibility: a remote peer cannot observe the control write or release signal
until the service published the corresponding doorbell after the payload flush
has succeeded.

For validation, the experiment added a dedicated two-rank
`prebuild-control-order` case. It drove the exact P3a command shape: payload
put, explicit flush,
visible 16-byte control source, control-slot write, signal-set, and barrier,
then verified payload, control generation, control count, and signal for each
generation. The default sequence repeated 1,000 generations, covering
two-generation source reuse and queue-slot wrap. The case is
synchronization-sensitive and used the same stream barrier protocol as the
existing route-order cases. Failed-payload/error-injection coverage remained
through the native invalid-queue/address checks and the prebuild invalidation
checks; no production error path was bypassed.

Remote two-rank validation on devices 0-1 used the rebuilt runner SHA-256
`2de8bdadaead33a9d483415d6462c27ce863e25b33400710890b0f182cdfbbc2`.
Both `prebuild-control-order` and teardown passed for 1,000 generations
with no transport diagnostic. The task was
`task_20260929_034305_182775418013`. The installed extension was restored to
the baseline SHA `d08352909619bc12e95ec2fc2d03ff4dfaef78ccb7b5b08dfae7350593bd1cad`
after the run.

Additional correctness used the P3a candidate extension SHA-256
`d0bab90f33f8cd25b30d81f748a14f20e64b1c1d235d2a9e8c909ab81c245047`:

| Gate | Task | Result |
| --- | --- | --- |
| 2-rank small | `task_20260929_031441_137873116833` | passed |
| 4-rank tail | `task_20260929_031522_138286916118` | passed |
| 8-rank canonical | `task_20260929_031733_14172872969` | passed |
| 8-rank weight boundary (17 tokens, hidden 4865, BF16) | `task_20260929_034427_183425826075` | passed |
| 2-rank `prebuild-control-order`, 1,000 generations | `task_20260929_034305_182775418013` | passed |

No-profile 8-rank ABBA used 8192 tokens, hidden 7168, top-k 8, 256 experts,
FP8 canonical, 30 warmups / 30 iterations,
`DEEP_EP_ASCEND_DISPATCH_CONSUMER_TILE_BYTES=8192`, and
`rank-launch-deadline-us=2000`. A was the baseline extension SHA
`d08352909619bc12e95ec2fc2d03ff4dfaef78ccb7b5b08dfae7350593bd1cad`; B was the
P3a candidate SHA above.

| Group | A mean (ms) | B mean (ms) | B gain |
| --- | ---: | ---: | ---: |
| 1 | 3.198414 | 3.149501 | +1.529% |
| 2 | 3.171345 | 3.277469 | -3.346% |
| 3 | 3.198997 | 3.190938 | +0.252% |
| Pooled | 3.189585 | 3.205969 | -0.514% |

All 12 ABBA JSON reports passed their single benchmark case. Two runs ended
with the pre-existing teardown-only SIGSEGV after the complete JSON and
`1 cases passed` line; the established acceptance policy allows this. The
queue restored the baseline extension SHA after each group.

Decision: reject P3a and retain the safe P2 baseline. The measured benefit is
within run-to-run noise, the pooled result is negative, and group 2 shows a
material regression. The source changes and the experiment-only probe were
removed; the design record and remote artifacts preserve the validation path.
P3 should not be retried in the same conservative form. A future version needs
a materially different publication mechanism, such as an independently proven
batch doorbell, before it justifies re-opening this path.

## Launch-skew attribution plan (2026-09-29)

The aligned Event mode intentionally moves rendezvous outside the measured
interval. It answers kernel-implementation A/B questions, but it is not an
application end-to-end result. A one-run no-deadline sample on the retained
safe-P2 extension measured Dispatch at 3.697083 ms / 2106.009 GB/s, while the
three-group aligned baseline pooled at 3.189585 ms / 2441 GB/s-class. The
difference is measurement exposure, not evidence of a device-speedup feature.

The follow-up is a diagnostic-only benchmark switch, `--profile-launch-skew`,
that does not alter launch order or add synchronization:

1. after each NPU start-event record, capture the host timestamp;
2. after operation submission, capture a launch-complete timestamp;
3. gather the existing per-rank Event samples;
4. report cross-rank start-record spread, launch-complete spread, and a
   conservative Event exposure bound not explained by start-record spread;
5. mark the report with `timing_protocol.profile_launch_skew: true`.

The target question is how the roughly 0.51 ms aligned-versus-unaligned delta
splits among host entry skew, Python submission duration, stream queuing, and
device-side arrival differences. Only after that attribution should the next
production candidate be selected. Host-entry alignment or busy waiting will not
make the latest rank launch earlier; persistent or launch-ahead service must be
justified by the measured component it actually removes.

### First launch-skew attribution result

The diagnostic benchmark ran as task
`task_20260929_073055_58585620454` on devices 0-3 using the retained safe-P2
extension SHA `d083529...`. It completed 30 warmups and 30 iterations with
`failed=0, passed=1, pending=0`. The report is
`results/p2/p2-r8-launch-skew-20260929.json`.

The observed Dispatch mean was 3.878437 ms / 2007.533 GB/s. Cross-rank
start-record spread averaged 0.817509 ms (p95 1.069919 ms, max 2.252970 ms).
Launch-complete spread averaged only 0.213777 ms. More importantly, each rank's
host operation submission took roughly 2.9-5.3 ms; the slowest-rank submission
correlated with the max-rank Event time at 0.985, while start-record spread
alone correlated at 0.957.

This identifies the main issue as the Python/PyBind/C++ launch path, not merely
the moment at which ranks enter the benchmark. The benchmark's host-side
deadline rendezvous removes this cost from the Event interval but does not make
the application faster. It must remain diagnostic-only.

The preferred follow-up is therefore a C++ persistent or launch-ahead dispatch
service:

1. pre-stage stable descriptors, buffers, tiling, and launch arguments;
2. enqueue the dispatch/epilogue work before all Python-side per-call work is
   finished;
3. let device-side generation/ready flags gate peer-visible work;
4. retain the existing completion, error, generation, and two-generation
   ownership checks;
5. keep dynamic count-dependent output allocation and CPU count publication
   outside the initial launch-ahead window.

This changes when host work occurs rather than hiding it from the timer. Host
alignment, busy waiting, or a measurement-only barrier is not an acceptable
production solution.

## P3 first slice: stable Dispatch tiling cache, rejected (2026-09-29)

The first launch-ahead slice tested the safest reusable part of the proposed
preparation set: caching the Dispatch CoreTiling inside one ElasticBuffer
generation. The launch key covered the full stable launch shape:

1. token, hidden, expert, top-k, alignment, and capacity dimensions;
2. mode flags, element kind, scale-factor packs, and data block count;
3. the complete runtime topology: world rank/size, scale-up rank/size,
   scale-out rank/size, topology kind, and topology epoch.

A hit copied the cached tiling and refreshed the transport context from the
current runtime context. A miss rebuilt the tiling through the existing
core-tiling builder and stored it under the buffer lifecycle mutex. Buffer
destroy invalidated the key, cache, and valid bit.

The topology fields were part of the acceptance boundary before any device
run: without them, a topology or epoch change inside the same object could
reuse a stale tiling topology even though the transport context was refreshed.
The probe also added a topology-epoch mismatch contract.

### Validation and result

Local contracts passed:

| Gate | Result |
| --- | --- |
| focused tiling/cache contracts after the candidate edit (2 tests) | passed |
| benchmark / transport / SIMT URM A contract suite | 144 passed, 3 skipped |
| git diff whitespace check | passed |

The rebuilt extension SHA was
994480fb904fd13617d8a655712a566aeec043fcc8b67052995c651578a2a7b5.
Two-rank production Dispatch correctness ran as task
task_20260929_091346_31056359377: all 14 dispatch matrix cases passed,
including expanded, aligned, cached reuse, 100 sequential generations,
round-trip, and invalid-expert diagnostics. The queue script restored the
baseline extension SHA
d08352909619bc12e95ec2fc2d03ff4dfaef78ccb7b5b08dfae7350593bd1cad
after the run.

No-profile 8-rank ABBA used the same canonical 8192-token FP8 workload,
30 warmups / 30 iterations, the retained benchmark environment, and no launch
deadline. A was the baseline SHA above; B was the tiling-cache candidate.

| Group | A mean (ms) | B mean (ms) | B gain | Task |
| --- | ---: | ---: | ---: | --- |
| 1 | 3.888685 | 3.907258 | -0.478% | task_20260929_092148_330615432627 |
| 2 | 3.903342 | 3.679716 | +5.729% | task_20260929_093643_351880525823 |
| 3 | 3.732774 | 3.833333 | -2.694% | task_20260929_095848_1205897569 |

All 12 JSON reports passed their single case. Two complete runs again ended
with the pre-existing teardown-only SIGSEGV after writing the report and
printing 1 cases passed; the established acceptance policy applies. The
installed extension was restored to the baseline SHA after each group.

Decision: reject and remove this slice. The three-group direction is
inconsistent (-0.478%, +5.729%, -2.694%), and the pooled mean is approximately
neutral. This is consistent with the earlier profile bound: the complete
Dispatch prelaunch setup was only about 0.089-0.149 ms, so saving only the
tiling subset is too small to overcome launch-path variance. The source changes
were removed and only this design record remains. A future launch-ahead
proposal should move a materially larger, independently attributable part of
the host preparation path, not retry this cache alone.

## Existing stream-overlap modes are not a launch-ahead upper bound
(2026-09-29)

Before designing a new persistent service, the zero-code mode probe asked
whether the existing comm-stream and async-with-compute paths already provide a
useful overlap upper bound. The runs used the canonical 8192-token FP8 Normal
Dispatch workload, 30 warmups / 30 iterations, devices 0-7, the retained
benchmark environment, and the baseline extension SHA
`d083529...`. Profile stages were disabled.

The first run was task
`task_20260929_104400_45421131946`, and the confirmation run was
`task_20260929_104858_5581328775`. The confirmation script was invoked with
run number 2, but a path-suffix mistake made it read the completed first-run
files as `r1` and then fail its summary step after all benchmark cases had
finished. The two available outputs are therefore the first run's normal-mode
result and the confirmation run's two stream-mode results. Both runs completed
their used outputs with `failed=0, passed=1, pending=0`. One normal-mode
process ended with the pre-existing teardown-only SIGSEGV only after writing
the complete JSON and printing `1 cases passed`; the established acceptance
policy applies. The installed extension SHA remained unchanged.

| Dispatch mode | First run (ms) | Confirmation run (ms) |
| --- | ---: | ---: |
| Normal (current production path) | 3.862063 | 3.959054 |
| `allocate_on_comm_stream` | 15.689018 | 15.176279 |
| `allocate_on_comm_stream` + `async_with_compute_stream` | 15.655756 | 15.472624 |

Cached Dispatch stayed around 63.5-64.0 ms in every mode. The stream modes are
therefore about four times slower for this Normal Dispatch workload and do not
provide a useful overlap upper bound. Reusing their current allocation/stream
structure is not a candidate for launch-ahead; a future service must retain the
Normal Dispatch stream and move only the independently attributed host
preparation work.

## Python launch-phase attribution (2026-09-29)

The launch-skew diagnostic was extended to split the public Python
`ElasticBuffer.dispatch` wrapper into four disjoint wall-clock phases:
`preflight`, `preparation`, `runtime`, and `result`.
The buckets are diagnostic-only, preserve execution order, and are exported
only when `--profile-launch-skew` is explicitly requested.

The 8-rank run was task
`task_20260929_113730_416585724716` on devices 0-7, using the canonical
8192-token FP8 Normal Dispatch workload, 30 warmups / 30 iterations, and the
retained CANN/HCOMM environment. The Python source snapshot used the new
diagnostic wrapper and the baseline extension SHA `d083529...`. It
completed with `failed=0, passed=1, pending=0` and exit 0. The result is
attribution-only and is not a no-profile performance result.

| Python dispatch phase | Mean (ms) | p50 (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| Preflight and contract construction | 0.099246 | 0.099015 | 0.113900 |
| Handle unpacking and defaults | 0.002848 | 0.002790 | 0.003520 |
| C++ runtime call | 3.105823 | 2.983215 | 3.766820 |
| EPHandle/result construction | 0.038889 | 0.037885 | 0.046970 |
| Total public Python wrapper | 3.346053 | 3.215840 | 3.986890 |

The profile-on Dispatch mean was 3.781115 ms. The total Python wrapper is not
pure host work: the runtime bucket contains C++ launch plus stream/device
synchronization and therefore waits for the multi-rank critical path. The
python-only portion is bounded by approximately 0.14 ms per call. This is
consistent with the C++ prelaunch profile (roughly 0.1 ms) and rules out
removing only Python preflight/argument assembly as a material launch-ahead
optimization. The next attribution boundary is inside the C++ runtime call:
separate C++ prelaunch, kernel submission, `synchronize_stream`, count
readback, and descriptor publication before proposing another service design.

## C++ Dispatch host-timeline attribution (2026-09-29)

The C++ runtime boundary was profiled with the same canonical 8192-token FP8
Normal Dispatch workload, 30 warmups / 30 iterations, and eight ranks. The
diagnostic build was task `task_20260929_121313_2573261675`. It passed its
single canonical case with `failed=0, passed=1, pending=0`. The extension
SHA-256 was
`e61b469ce74849299b9321d160859ec25038686e711bb922254c2c7d98eb0f98`, and the
remote artifact was retained under
`/home/pyptouser/yuqitao/deepep-cpp-host-timeline-20260929/.scratch/launchahead/results/r8-cpp-host-timeline-20260929-r2.json`.

This run had `stage_profile=1`. Its profile-on Dispatch mean of 4.291659 ms
and 1814.238 GB/s is attribution-only and must not be compared with a
no-profile result. Stage profiling disables compact stage boundaries and the
fused consumed barrier, so the slower profile-on envelope is expected.

The report contains eight per-rank records. The per-rank maxima were:

| C++ host phase | Observed |
| --- | ---: |
| `dispatch_prelaunch_setup` | 0.151240 ms |
| `dispatch_submit` | 0.200190 ms |
| `dispatch_synchronize` | 3.883320 ms |
| `dispatch_diagnostic_read` | 0.025920 ms |
| `dispatch_counts_to_host` | 0.017140 ms |
| `dispatch_host_prefix` | 0 |
| `dispatch_prefix_to_device` | 0 |
| `dispatch_descriptor_publication` | 0.021960 ms |

These are per-rank maxima, not disjoint phases on one timeline, so they must
not be added. The reported total for the maximum-total rank was 4.145560 ms.
In the captured build `dispatch_synchronize` was recorded after the nested
diagnostic readback. The local instrumentation has since been corrected to
record synchronize immediately after `synchronize_stream` and to time the
diagnostic readback separately. The nested readback was only about 0.026 ms,
so the conclusion is unchanged.

The per-rank synchronize distribution is more informative than the maxima:

| Rank | Synchronize (ms) | Host entry skew (us) | Host timeline total (ms) |
| ---: | ---: | ---: | ---: |
| 0 | 2.963010 | +983.070 | 3.336110 |
| 1 | 3.089270 | +846.870 | 3.395550 |
| 2 | 2.959450 | +896.010 | 3.263380 |
| 3 | 3.087130 | +690.900 | 3.406690 |
| 4 | 3.883320 | 0.000000 | 4.145560 |
| 5 | 3.007880 | +887.100 | 3.358530 |
| 6 | 3.314240 | +508.170 | 3.581380 |
| 7 | 3.130030 | +872.650 | 3.427320 |

Rank 4 entered Dispatch earliest and waited longest; the other ranks waited
approximately 2.96-3.31 ms. This pattern is consistent with waiting for a
global device-side critical path rather than one locally slow rank. Device
envelopes also show roughly 1.0-1.9 ms idle per rank, with the largest idle
interval on rank 4. The observed host-entry spread is therefore one input to
the critical path, but the stream-synchronize wait is not explained by local
Python or C++ preparation alone.

Conclusion: apart from synchronization, the attributable host work is small.
Python-side work was already bounded at approximately 0.14 ms, C++ prelaunch
setup is approximately 0.08-0.15 ms, submission is approximately 0.14-0.20 ms,
and diagnostic/readback/publication are each approximately 0.02 ms. A generic
launch-ahead or host-preparation service has no sufficiently large independent
interval to hide on this workload. The remaining large component is device-side
or cross-rank arrival/completion structure, including launch skew and the
release/CQ completion path. The next optimization proposal must first identify a
specific change to that critical path; merely moving Python/C++ preparation
earlier is not a candidate.

## Performance-metric boundaries (2026-09-29)

Four numbers from the recent Normal Dispatch records are useful but use four
different timing boundaries. They must not be pooled or compared directly.

| Number | Boundary | Meaning and limitation |
| ---: | --- | --- |
| `2.1 TB/s` | No-deadline NPU Event | A one-run no-deadline sample reported `3.697083 ms / 2106.009 GB/s`. It excludes most host work before and after the Event interval, while still exposing some cross-rank launch skew. It is an Event diagnostic, not end-to-end latency. |
| `2.4 TB/s` | 2-ms-aligned NPU Event | The three-group `--rank-launch-deadline-us 2000` record pooled at `3.216025 ms / 2421.029 GB/s`. The rendezvous and wait are outside the Event interval, so it is a device-path attribution number. It must not be called application throughput or a production speedup. |
| `1.9 TB/s` | No-deadline wall time | The retained configuration measured `4.095875 ms`; dividing the unchanged logical bytes by wall time gives `1900.959 GB/s`. Wall includes Python/PyBind launch, stream synchronization, and teardown-adjacent host work, so it is the conservative end-to-end comparison for the 3 TB/s target. |
| `11.76 TB/s` | Affinity-enabled NPU Event | With per-rank CPU affinity, the Event interval was only `0.662098 ms`, while wall time remained `4.095875 ms`. The Event no longer covers the host launch/synchronization path, so this bandwidth is not physically meaningful as end-to-end throughput. |

The workload identity did not change: all records are the canonical eight-rank
FP8 case `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0`, 8192 tokens,
hidden 7168, top-k 8, 256 experts, seed 0, 30 warmups, and 30 iterations.
The retained benchmark environment is now part of the production default:
one NUMA0 physical core per rank, one Torch compute thread, one inter-op
thread, disabled Python cyclic GC, and `min(64, device AIV count)` data
blocks. The 2-ms launch deadline remains diagnostic-only and disabled by
default.
