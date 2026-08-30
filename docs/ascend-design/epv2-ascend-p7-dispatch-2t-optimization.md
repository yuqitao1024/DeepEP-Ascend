# EPv2 Ascend P7 Normal Dispatch 2T Optimization

**Status:** P7 implementation in progress; the retained tree is correctness-qualified but remains below the 2T target

## 1. Goal

P7 focuses only on normal, uncached, direct scale-up Dispatch. The primary
acceptance target is at least `2000 logical GB/s` on the fixed eight-rank
representative workload. Expanded Dispatch, Cached Dispatch, normal Combine,
and Reduced Combine remain correctness regressions gates, but they are not P7
performance targets.

The fixed workload is:

```text
world_size=8
num_tokens=8192 per rank
hidden=7168
top-k=8
experts=256
expert_alignment=128
dispatch=FP8
data_blocks=72
warmups=30
samples=30
seed=0
workload fingerprint:
d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00
```

The retained P6 result is:

| Metric | Value |
| --- | ---: |
| Mean | `28.574865 ms` |
| P50 | `28.553505 ms` |
| P95 | `30.383396 ms` |
| Logical bytes | `7,786,089,792` |
| Logical bandwidth | `272.480370 GB/s` |

The unchanged bandwidth formula gives the target latency:

```text
7,786,089,792 bytes / 2,000,000,000,000 bytes/s = 3.893045 ms
```

P7 succeeds only when the final retained tree measures a mean no greater than
`3.893045 ms` and at least `2000 logical GB/s`, with correct output and no
material p95 instability. Changing the logical-byte formula, workload, rank
aggregation, warmup count, or sample count does not satisfy the target.

## 2. Scope

### 2.1 Included

- CUDA-parity producer data flow: load one input token once and fan it out to
  all unique destination ranks.
- Larger producer payload movement and reduced MTE synchronization frequency.
- Early rank and expert count publication before payload completion.
- Source-token chunking that preserves contiguous per-destination payload
  intervals and avoids rescanning all tokens for every chunk.
- Producer, HCOMM, and consumer-copy overlap required to reach the final
  latency budget.
- Direct, uncached, synchronous, non-expanded, pure scale-up Dispatch first.
- FP8 hidden width `7168` as the performance specialization and width `7184`
  as the required aligned-body plus scalar-tail correctness case.
- Stage profiling and same-binary performance selection for every candidate.

### 2.2 Excluded

- Returning a tensor that aliases the symmetric communication window.
- Removing the final Dispatch copy epilogue.
- Changing the public Python or C++ API, output layout, handle semantics,
  record order, or logical-byte accounting.
- Adding HCOMM channels without new queue or service-saturation evidence.
- Optimizing Expanded Dispatch, Cached Dispatch, Combine, Reduced Combine,
  hybrid routing, or scale-out transport in the same performance change.
- Weakening generation, visibility, timeout, first-error, or buffer-reuse
  guarantees.

## 3. Root-Cause Evidence

### 3.1 Current stage profile

The final P6 profile has a mean per-rank device envelope of about `11.560M`
cycles and a maximum-rank envelope of `14.367M` cycles. Mean idle is negligible,
so launch fusion without reducing work cannot provide the required speedup.

| Stage | Mean cycles | Share of active path |
| --- | ---: | ---: |
| D3 producer record | `3.835M` | `33.99%` |
| D4 release barrier | `2.902M` | `23.32%` |
| D8 output copy | `1.665M` | `14.74%` |
| D4 release payload | `0.915M` | `8.09%` |
| D2 producer prefix | `0.567M` | `5.02%` |
| D5 validation | `0.503M` | `4.45%` |
| D5 expert count | `0.339M` | `3.00%` |
| D4 release control | `0.261M` | `2.31%` |

D3, D4, and D8 are ordered rather than overlapped. Optimizing any one of them
in isolation cannot reduce `28.575 ms` to `3.893 ms`.

### 3.2 CUDA producer data flow

CUDA `dispatch_impl` assigns one dispatch warp iteration to a token. It issues
one TMA load for the complete hidden row:

```text
x[token, 0:hidden_bytes] -> shared token buffer
```

Scale factors are loaded separately with `cp.async`, top-k indices and weights
are loaded cooperatively by lanes, and source metadata is added to the same
token buffer. The completed token buffer is then stored or put to every unique
destination rank. For an NVLink-accessible peer, the store targets the peer's
symmetric receive buffer directly. For a non-direct peer, the token is staged
once in the send buffer and used by Gin puts.

For token `t`, let `R_t` be its number of unique destination ranks. CUDA's
hidden-source read volume is:

```text
B_cuda_source = T * H
```

and its destination write volume is:

```text
B_cuda_fanout = sum_t(R_t) * H
```

This is not end-to-end zero-copy. CUDA still runs
`dispatch_copy_epilogue_impl`, which loads records from the communication
buffer and TMA-stores the final public `recv_x` tensor. The parity target is
load-once fan-out and asynchronous bulk movement, not removal of D8.

### 3.3 Current Ascend producer amplification

The eligible Ascend path first creates one record per unique `(token,
destination_rank)`. Its AICore vector body then iterates over those records.
For each destination record, it reloads the token's hidden row from input GM
in `512`-byte pieces and writes those pieces through UB to the staging or
local receive shard.

For the representative workload:

```text
total records across eight ranks = 346,603
mean records per rank             = 43,325.375
input tokens across eight ranks   = 65,536
mean unique ranks per token       = 346,603 / 65,536 = 5.288742
512-byte tiles per 7168-byte row  = 14
```

The current hidden-source read volume is:

```text
B_ascend_source = sum_t(R_t) * H
```

so it reads the same input hidden row about `5.29` times on average. It also
performs about:

```text
43,325.375 records/rank * 14 tiles/record = 606,555 tile iterations/rank
```

Each iteration carries GM-to-UB, event ordering, and UB-to-GM work. The first
P7 optimization removes this source-read amplification before changing the
transport protocol.

### 3.4 HCCS is not the first root cause

The representative transport-only benchmark uses the production
`DeviceTransportFacade`, command queue, AICore service, one facade channel,
HCOMM put/flush, and CQ completion. It moves `2.290 GB` of remote payload in
`0.903 ms`, reporting `2535.046 aggregate GB/s` under its transport-only byte
formula.

This does not predict full Dispatch latency because its byte formula and
protocol tail differ. It does prove that large contiguous puts, representative
rank imbalance, and a single facade channel can sustain much more throughput
than the current production operator. P7 therefore does not begin with link
tuning or extra channels.

## 4. Selected Architecture

P7 uses a measured sequence of four workstreams. Each workstream starts from
the last retained tree. A rejected candidate is removed before the next
candidate is built. The workstreams execute in order unless an earlier
retained tree already reaches the final `2000 logical GB/s` target. The final
target is architectural, but each measurement changes one mechanism at a
time.

```text
P7.0 token-resident producer
   -> P7.1 early count/expert plan
      -> P7.2 source-token chunk pipeline
         -> P7.3 arrival-driven consumer tail
            -> P7.4 final 2T acceptance
```

### 4.1 P7.0 token-resident load-once fan-out

#### Data flow

The P7.0 AICore producer is token-oriented rather than destination-record
oriented:

```text
for each owned token t:
    resolve unique destination ranks and destination slots
    load x[t] aligned body from GM to UB once
    for each unique destination rank r:
        store UB hidden body to record(t, r)
    leave only the sub-32-byte hidden suffix to the SIMT path
```

For a vector body `H_v` and unique destination count `R_t`, the source read
count changes from:

```text
current: R_t * H_v bytes
P7.0:    1   * H_v bytes
```

The destination write count remains `R_t * H_v`; those bytes are required by
the Dispatch routing contract.

#### Existing metadata reuse

P7.0 consumes the existing grouping results:

- `dispatch_group_owner[token][rank]` identifies the master top-k lane for a
  unique destination rank;
- `dispatch_group_tile[tile][rank]` contains the destination slot prefix for
  that grouping tile; and
- `destination_slots[token][lane]` contains the encoded source rank and local
  slot used by the handle and Combine.

The optimization must not recompute a rank scan per destination. One token
owner reads the eight owner entries, resolves only active destinations, and
fans out the resident payload.

#### Tile and UB policy

The first implementation uses one token-resident buffer for the largest
32-byte-aligned body that fits the qualified UB allocation. For the target
shape this is the complete `7168`-byte hidden row. Width `7184` uses a
`7168`-byte vector body plus a `16`-byte SIMT tail.

If a future hidden row exceeds the qualified buffer, it is divided into large
row tiles. Every row tile is still loaded once per token and stored to all
destinations before the next source load. Falling back to one input load per
destination is not allowed in the P7.0 eligible path.

The first candidate keeps conservative MTE ordering: the source load is known
complete before any destination store, and the UB buffer is not overwritten
until its stores are complete. Store batching or double buffering is a later
P7.0 candidate and is retained only if a D3 profile proves additional benefit.

#### Metadata boundary

The first P7.0 slice moves only the hidden vector body. Existing SIMT code
continues to write scale factors, top-k indices, top-k weights, source
metadata, destination metadata initialization, and the scalar hidden suffix.
This isolates the dominant byte movement.

A second P7.0 slice may load scale factors once per token and fan them out if
the retained hidden candidate leaves scale movement visible in D3. It must not
be bundled into the first performance result. Top-k and source metadata are
small and move only when profiling shows they are material.

#### Eligibility and selector

The same binary must contain baseline and candidate paths.
`DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT=1` enables the candidate, while unset or
`0` selects the retained P6 path. Any other value fails at the host boundary.
The selector enables the candidate only for:

- direct, non-hybrid, pure scale-up Dispatch;
- uncached and non-expanded mode;
- grouping-eligible `top-k <= 8`, `world_size <= 8` shapes;
- no existing destination-slot chunk pipeline; and
- a valid aligned-body and UB plan.

Pipeline composition has one important distinction. The early-route selector
receives `pipeline_config.enabled`, which represents only the legacy
destination-slot pipeline; it remains mutually exclusive with P7.1. The
source-token selector is independent, so P7.1 can be composed with
`DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_TILES`. P7.0 token fan-out is kept out
of either pipeline until the persistent producer path has separate parity
evidence: the host passes
`pipeline_config.enabled || source_pipeline_config.enabled` to the fan-out
selector. This prevents a source-token chunk from silently selecting the
untested fan-out body while preserving the intended P7.1 + P7.2 composition.

Ineligible shapes use the retained P6 implementation. Invalid selector values
fail at the host boundary. The selector remains opt-in until correctness,
same-binary ABBA, and profile gates all pass.

#### P7.0B EP8 result

TaskQueue task `task_20260827_155240_35696777216` completed with exit code
zero. It used the fixed representative fingerprint
`d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00`,
72 data blocks, 30 warmups, and 30 measured samples for each same-binary ABBA
run. The only ABBA variable was
`DEEP_EP_ASCEND_DISPATCH_TOKEN_FANOUT=0|1`.

| Run | Selector | Mean | P50 | P95 | Logical bandwidth | Rank mean envelope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline A | `0` | `29.752022 ms` | `29.804678 ms` | `31.164570 ms` | `261.700 GB/s` | `27.844267-29.368730 ms` |
| candidate A | `1` | `24.284135 ms` | `25.007016 ms` | `27.022045 ms` | `320.625 GB/s` | `22.805175-23.725172 ms` |
| candidate B | `1` | `25.971721 ms` | `25.888251 ms` | `26.903730 ms` | `299.791 GB/s` | `24.492869-25.637063 ms` |
| baseline B | `0` | `29.669349 ms` | `29.568598 ms` | `31.192362 ms` | `262.429 GB/s` | `28.188482-29.296955 ms` |

The two baseline and two candidate run means are:

| Metric | P6 baseline | P7.0B candidate | Change |
| --- | ---: | ---: | ---: |
| Mean | `29.710685 ms` | `25.127928 ms` | `-15.425%` |
| P95 | `31.178466 ms` | `26.962887 ms` | `-13.521%` |
| Logical bandwidth | `262.064 GB/s` | `310.208 GB/s` | `+18.371%` |

The representative EP8 run contains `65,536` source tokens and `346,603`
unique destination records. The retained candidate therefore performs:

```text
source DataCopy operations      = 65,536
destination DataCopy operations = 346,603
source vector bytes              = 65,536 * 7,168
                                 = 469,762,048 bytes
destination vector bytes         = 346,603 * 7,168
                                 = 2,484,450,304 bytes
```

The source-load count is now proportional to input tokens rather than
destination records. The source-load operation count is `5.29x` smaller than
the destination-store count for this route matrix. Source contracts and the
two-rank mixed-route correctness task prove that the input `DataCopy` is
outside the destination loop; the EP8 work counts prove the loop cardinalities
for the measured route matrix.

The candidate profile used 10 warmups and 10 measured samples. Its Dispatch
event mean was `25.874619 ms` and its logical bandwidth was `300.916 GB/s`.
The independently aggregated stage spans were:

| Stage | Mean span | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| D3 record packing | `407,606.5 cycles` | `389,252` | `438,186` |
| D4 publication | `2,908,067.1 cycles` | `1,283,714` | `6,235,860` |
| D8 output copy | `1,663,595.9 cycles` | `1,640,485` | `1,685,712` |

D3 is no longer the largest exposed stage. The D4 payload substage is
`909,729-1,034,134 cycles`, which agrees with the existing representative
transport-only result of `903,444` cycles for the same seven-put, one-flush
payload pattern. The additional D4 tail is the production control and global
barrier protocol. Rank 6 spent `5,008,911 cycles` in `release_barrier`; its
complete D4 span was `6,235,860 cycles`. Command queues drained to zero, and
SQ/CQ high-watermarks were both four, so the evidence does not indicate queue
capacity pressure or a slow HCCS payload link.

P7.0B is retained. P7.0C store-order screening is deferred because its entire
measured D3 budget is smaller than D8 and far smaller than the D4 tail. The
next implementation step remains P7.1, but the measured purpose is now
explicit: publish the route plan early enough that later P7.2/P7.3 scheduling
can overlap receiver planning and copy work with the D4 service/barrier tail.
The final barrier is not deleted; changing or hiding it requires separate
correctness evidence for payload visibility, staging reuse, and generation
lifecycle.

The raw artifacts are retained on NPU8P under
`/home/pyptouser/yuqitao/deepep-results/p7-token-fanout-vWazzF/ep8-abba-profile`.
Key SHA-256 values are:

```text
baseline-a.json         acd5f655e0d934321e70a17d10ee28e2d46c0620fb93a1b3005e38e07b518fc0
candidate-a.json        33b14789da5af0002086bf3b77ee744207851b871da8103610bdb4410c6a0de5
candidate-b.json        a76b556849ddd4e6eeb558b74c842af162b8a46e32bc7e7cc7906b3117e4b38a
baseline-b.json         b618a79bc7b8b74d55366519eaf37e8bc31779b01c33f2ba165cb2d2f03603a7
candidate-profile.json  4bea183579c726725e2288caac033ac273b27c8e01d55455da1c483947cdaaa3
summary.json            d98f65b7dcddd02bb7ad97110dbfb009776f127127360ecc2e1b268043277b4b
```

#### Latest P7 qualification checkpoint

The retained P7 tree now also uses a single polling loop for all outstanding
peer barrier counters.  Each iteration keeps a `pending_peers` bitmask and
reads only peers that have not reached the current generation.  This removes
the old serial pattern in which a delayed peer blocked observation of every
later peer.  The representative eight-rank profile (one warmup, one sample,
stage profiling enabled) measured:

| Tree | Mean | Logical bandwidth | Result |
| --- | ---: | ---: | --- |
| P7 before peer-scan change | `24.312164 ms` | `320.254902 GB/s` | retained baseline |
| P7 peer-scan candidate | `22.302765 ms` | `349.108724 GB/s` | retained |

The candidate's rank-local `release_barrier` spans were still variable
(`0.081M` to `5.394M` cycles), so the barrier remains the largest exposed
owner.  The profile script exits with status 1 after a successful benchmark
because its optional summary formatter still references a nonexistent
`operation.status`; acceptance uses the generated `profile.json` and the
`cases passed` line instead.

Two isolated candidates were rejected.  Enabling `DEEP_EP_ASCEND_DISPATCH_EARLY_ROUTE_PLAN=1`
added route-plan puts, signals, and a barrier and measured `333.884014 GB/s`,
below the retained tree.  The source-token pipeline (`chunk_tiles=256`) still
stalls in its persistent producer/release protocol before writing a profile;
it is not enabled for Normal Dispatch.  A D8 source-rank lookup experiment
and a packed 16-byte control-slot put likewise showed no qualified gain (the
latter was terminated after exceeding the normal runtime), so neither is
part of the retained implementation.

#### P7 barrier candidate qualification

The retained 4096-byte consumer-tile tree was used as the control. Its
interleaved eight-rank ABBA result was `21.312 ms / 365.64 GB/s` (four runs,
`8192 x 7168`, top-k 8, 256 experts, FP8 dispatch, 72 data blocks). Three
barrier-only candidates were tested against the same case and removed after
qualification:

| Candidate | Result | Decision | Evidence |
| --- | ---: | --- | --- |
| Sequential three-round dissemination marker | `22.134 ms / 351.97 GB/s` ABBA average | reject | An initial implementation left three marker requests outstanding; the established `drain_all` fixed profile correctness, but the corrected ABBA remained below control. |
| CQ drain after GM counter poll | `22.491 ms / 346.33 GB/s` ABBA average | reject | One isolated sample reached `21.159 ms / 367.98 GB/s`, but the four-run ABBA regressed; the isolated result was noise rather than a qualified gain. |
| Parallel three-round dissemination marker | `22.090 ms / 352.47 GB/s` single representative sample | reject | The three writes and three source-slot polls completed correctly, but the result was below control and did not justify another ABBA run. |
| Direct generation inline-write barrier | `33.682 ms / 231.17 GB/s` single representative sample | reject | Replacing each FAA arrival with an idempotent inline 64-bit write passed the eight-rank case but regressed badly. The transport path's FAA operation is faster here despite its apparent extra fetch result, so the source was restored to FAA. |

The dissemination variants were limited to `world_size == 8`, flat single-node
scale-up and did not alter non-8-rank or scale-out behavior. They are not in
the retained source. The retained barrier still uses FAA to all peers,
cache-bypassing `load_published` counter reads, one pending-peer poll loop,
and the established CQ drain before polling. This preserves the protocol and
the best measured end-to-end result while leaving the D4 control/barrier tail
as the next optimization target.

### 4.2 P7.1 early rank and expert plan

P7.1 starts after profiling the retained P7.0 tree, unless P7.0 already reaches
the final `2000 logical GB/s` target. The new profile determines which P7.1
substage is measured first; it does not make early count publication optional.

CUDA notify warps count rank and expert destinations while dispatch warps move
payload. P7.1 gives the Ascend producer the same semantic separation:

```text
route-ready generation:
    destination record counts
    destination-local expert counts

payload-ready generation:
    record bytes visible for consumer use
```

The sender derives expert counts from the same top-k traversal used by D1. A
grouping tile owns private rank and expert counts; a deterministic prefix and
reduction produce one count vector per destination rank. Count publication is
small and may precede D3 because it does not claim payload visibility.

The receiver may calculate source prefixes, local expert totals, aligned
expert prefixes, and exact work bounds after all route-ready generations are
observed. It must not read or validate record payload until the corresponding
payload-ready generation is observed.

P7.1 preserves exact public tensor shapes. The retained device-prefix path may
preallocate conservative capacity, but the final count readback and tensor
narrowing remain part of the public return contract.

### 4.3 P7.2 source-token chunk pipeline

The previous experimental pipeline divided each destination shard by slot
range. Its producer record path could revisit all tokens and filter records by
slot range for every chunk. That lifecycle added work and made the performance
result unstable.

P7.2 chunks the source-token space instead. Slot assignment is already stable
in source-token order. Let `[a, b)` be an interval of grouping-tile indices,
covering source tokens `[4a, min(4b, T))`. For destination rank `r`, existing
tile prefixes and the final destination count define one contiguous payload
interval:

```text
begin(r) = prefix[a][r]
end(r)   = b == tile_count ? destination_count[r] : prefix[b][r]
bytes(r) = (end(r) - begin(r)) * token_stride_bytes
```

The pipeline is:

```text
pack source-token chunk n
    -> publish contiguous destination intervals for chunk n
       -> HCOMM put/flush chunk n
          || pack source-token chunk n+1
```

Each token belongs to exactly one producer chunk. No chunk is allowed to scan
tokens outside its source-token interval. The fixed destination shards remain
disjoint, so payload storage does not require a second full staging buffer.
Only request state and completion generations use bounded ring slots.

The initial chunk size is derived from grouping tiles and screened with the
retained P7.0 producer. Chunk count, commands, source bytes, request-slot high
watermarks, D3/D4 overlap, and end-to-end time are all reported. A chunk size
that improves overlap but repeats enough control work to regress the operator
is rejected.

#### 4.3.1 Persistent VF root cause and revised execution model

The first persistent prototype launched several SIMT VF functions from the
same runtime chunk loop: begin, scalar packing, completion, release, request
wait, control, and barrier. EP2 white-box probes consistently observed a
command queue count of `2`, exactly one remote `put` followed by one `flush`
for the first chunk. The control generation remained zero and neither the
pipeline nor transport diagnostic marker for the control batch appeared.
Changing fences and merging the final control/barrier batch did not change
that result.

The evidence locates the failure before HCOMM service execution: later VF
batches were never appended. On this CANN path, `asc_vf_call` expands to an
asynchronous VF invocation and repeated calls from one running AICore kernel
cannot be treated as a general runtime task queue. P7.2 therefore adopts the
following rule:

```text
one persistent AICore stage -> exactly one persistent VF invocation
runtime chunk loop          -> doorbells and ordinary AICore functions only
```

The producer uses two levels of readiness:

```text
persistent producer VF                     producer AICore blocks
----------------------                     ----------------------
pack scalar/scale/top-k/metadata chunk n
all VF blocks complete scalar stores
publish scalar_ready(n) -----------------> wait scalar_ready(n)
                                             hidden load-once fanout
                          <---------------- publish hidden_done(block, n)
all VF blocks observe paired hidden_done
publish payload_ready(n)
```

Each AICore block writes hidden progress to an independent 64-byte cache line.
This avoids two cores writing different words in the same cached line and then
overwriting one another during cache-line writeback. The paired VF block turns
that per-block progress into a single atomic hidden-completion count; only the
last contributor publishes the chunk as ready for transport.

The release side uses an exact command-count doorbell:

```text
persistent release VF                      release AICore
---------------------                      --------------
wait payload_ready(n)
append put(s) + flush request
publish batch_target = request.command_end -> observe target > consumed
                                              service::execute()
                          <---------------- publish consumed_target = target
wait/validate request
publish slot completed

append final control + signal + barrier
publish final batch_target --------------> service::execute()
                          <---------------- publish final consumed_target
publish release completion
```

Both targets are monotonically increasing command counts for one transport
generation. The release VF cannot append the next batch until AICore has
acknowledged the current target. This bounds queue growth, preserves request
ownership, and allows repeated `service::execute()` calls without relying on
repeated asynchronous VF launches. The host schedule remains event-free: one
release kernel on the communication stream, one producer kernel on the
producer stream, then the normal epilogue stages.

### 4.4 P7.3 arrival-driven consumer tail

P7.3 starts after P7.2 unless the retained tree already reaches the final
target. It uses P7.1 source prefixes and per-source payload-ready generations
to process complete source intervals without waiting for unrelated source
ranks. The P7.2 profile determines whether D8, validation, or the final
all-rank barrier is the first P7.3 substage to address.

For normal Dispatch, the public compact order is source rank followed by
source slot. Once source counts are known, the final destination interval for
source rank `s` is fixed:

```text
output_begin(s) = source_prefix[s]
output_end(s)   = source_prefix[s] + source_count[s]
```

An arrival worker may validate and copy only a payload-ready source chunk into
that fixed interval. It may not publish final completion until:

- every source chunk is payload-ready and validated;
- all hidden, scale, top-k, weight, and source-metadata copies are visible;
- all rank and expert counts match their route-ready plan; and
- the transport and Dispatch generations have reached the required value.

The final barrier remains responsible for safe staging-buffer reuse and
cross-rank completion. P7.3 moves useful consumer work before that tail; it
does not delete the lifecycle boundary.

### 4.5 P7.4 acceptance and stop rule

After every retained workstream, run a fresh profile and recalculate the
remaining latency budget. P7 stops successfully when the representative mean
reaches `<= 3.893045 ms`. It does not continue adding protocol complexity
after the target is met.

If the complete P7.0-P7.3 path remains below `2000 logical GB/s`, the final
profile must identify the largest remaining owner. Symmetric-buffer-backed
public output is considered only if D8 remains the largest unavoidable span
and its measured removal is sufficient to cross the target. It is a separate
ABI design, not an implicit P7 fallback.

## 5. Latency Budget

The target budget guides prioritization; it is not permission to infer time
from independently aggregated stage maxima.

| Critical-path region | Target mean |
| --- | ---: |
| Route grouping, counts, and prefix | `<= 0.35 ms` |
| Token-resident packing overlapped with transport | `<= 2.60 ms` |
| Exposed consumer-copy tail | `<= 0.65 ms` |
| Completion, publication, and runtime margin | `<= 0.29 ms` |
| Total | `<= 3.89 ms` |

The packing and transport budget is deliberately combined. The representative
transport-only result is about `0.903 ms` under its narrower measurement, so
P7 must feed large contiguous payloads efficiently and hide most packing time
rather than require HCCS itself to become several times faster.

## 6. Protocol and Correctness Invariants

P7 must preserve:

1. one record for every unique `(source token, destination rank)` pair;
2. source-token order within every destination shard;
3. the existing encoded destination-slot format;
4. one master top-k lane per destination record;
5. all top-k indices and weights in every record, including inactive `-1`
   lanes;
6. exact FP8 hidden and scale-factor bytes;
7. the source rank, source token, master lane, and destination metadata
   contract used by Combine;
8. deterministic first-error selection before successful publication;
9. no payload-ready observation before all corresponding record bytes are
   visible;
10. no staging or request-slot reuse before transport completion;
11. exact public output shapes and handle compatibility; and
12. unchanged behavior for cached, expanded, hybrid, scale-out, async, and
    ineligible fallback paths.

Duplicate experts that map to the same destination rank still create one
record. Different top-k lanes that map to different local experts in that
rank remain represented inside that one record. An inactive `-1` lane creates
no destination but remains `-1` in every record produced for the token.

## 7. Measurement Requirements

### 7.1 P7.0 work counters

The report must contain or derive:

- input tokens;
- valid top-k routes;
- unique destination records;
- hidden vector bytes per token;
- expected source hidden reads;
- destination hidden writes;
- DataCopy source-load operations;
- DataCopy destination-store operations; and
- D3 stage span.

For the P7.0 candidate, expected source hidden bytes must be proportional to
`input_tokens * vector_hidden_bytes`, not `unique_records *
vector_hidden_bytes`.

### 7.2 Pipeline evidence

P7.2 and P7.3 additionally report:

- source-token chunk count and token bounds;
- per-destination slot and byte intervals;
- command count and payload bytes;
- SQ/CQ final depth and high-watermark;
- producer/transport and transport/consumer overlap;
- route-ready and payload-ready generation progression;
- per-rank device envelope; and
- end-to-end device-event samples.

Stage spans are independently aggregated diagnostics. Only the maximum-rank
device-event samples determine final bandwidth.

### 7.3 Retention rule

There is no fixed percentage threshold for intermediate candidates. Retain a
candidate only when:

- the intended mechanism is visible in counters or stage timing;
- mean and p95 move consistently beyond same-run pair noise;
- all correctness gates pass; and
- unrelated operations do not materially regress.

A faster isolated D3 stage with a flat or worse end-to-end result is not
sufficient. Its result is recorded, and the code is removed unless it is a
necessary, measured dependency of the next approved workstream.

## 8. Validation Sequence

Every workstream follows this order:

1. host tests for selector parsing, eligibility, copy plans, source-token
   ownership, prefix intervals, overflow, and fallback behavior;
2. source-contract tests for one source load per token tile and no
   destination-oriented source reload in the eligible path;
3. local focused Python and C++ contract suites;
4. clean Bisheng/AOT and production-extension build on NPU8P;
5. two-rank correctness on an allowed device pair;
6. hidden widths `7168` and `7184`;
7. normal routing, duplicate same-rank experts, inactive `-1` lanes, all-local
   and all-remote destinations, capacity boundaries, and two consecutive
   generations;
8. five-operation representative correctness to protect shared state;
9. EP8 same-binary ABBA screening and profiling; and
10. final 30-warmup, 30-sample EP8 acceptance.

NPU work must use TaskQueue. Normal development is limited to the locally
authorized two-device pairs. An eight-device task is submitted only after
explicit temporary authorization for P7 EP8 validation.

## 9. Work Breakdown

### P7.0A: Contracts and instrumentation

- Add a pure producer fan-out plan and checked eligibility helper.
- Add expected source-read and destination-write work-count tests.
- Add source contracts that reject a per-destination input-load loop in the
  eligible AICore function.
- Capture the retained P6 D3 profile with the final benchmark settings if the
  existing artifact cannot be reproduced from the current build.

### P7.0B: Single-buffer hidden fan-out

- Add the strict same-binary selector.
- Change only the aligned hidden body to token ownership.
- Reuse existing owners, tile prefixes, and destination slots.
- Keep scale, top-k, weights, metadata, and scalar tail unchanged.
- Verify source reads fall by the unique-destination factor and profile D3.

### P7.0C: MTE ordering and row-tile screening

- Screen qualified row-tile sizes only after P7.0B passes.
- Compare conservative per-store completion, batched store completion, and a
  two-buffer schedule one mechanism at a time.
- Retain the smallest ordering model that provides the best stable D3 and
  end-to-end result without UB or event failure.

### P7.1: Early route plan

- Add per-tile expert counts and deterministic reduction.
- Add separate route-ready and payload-ready state.
- Move source/expert prefix work before payload completion.
- Remove D5 expert rescanning only after count-plan parity is proven.

### P7.2: Source-token pipeline

- Replace destination-slot chunk traversal with source-token bounds.
- Derive contiguous per-rank intervals from grouping tile prefixes.
- Reuse bounded request slots and retain exact payload storage.
- Measure overlap, repeated work, commands, and queue state.

### P7.3: Arrival-driven copy

- Add per-source or per-source-chunk payload-ready observation.
- Copy only into fixed source-prefix output intervals.
- Preserve validation, final completion, and reuse barriers.
- Retain only when D8 or the barrier tail moves off the critical path.

### P7.4: Final acceptance

- Run the unchanged representative case with 30 warmups and 30 samples.
- Publish mean, p50, p95, logical bytes, and logical bandwidth.
- Publish the final stage profile and P6-to-P7 comparison.
- State explicitly whether `2000 logical GB/s` was reached.

## 10. Alternatives Not Selected

### Larger 512-byte tiles without load-once fan-out

Larger tiles reduce event frequency but still reload the same input row once
per unique destination. It is useful as a controlled subexperiment, not the
architecture for a `7.3x` end-to-end target.

### Symmetric receive buffer as public output

This could remove D8, but it changes tensor ownership, allocation, compact
layout, lifetime, and cached-handle assumptions. CUDA DeepEP also retains a
copy epilogue. P7 does not take this ABI risk before exhausting load-once and
overlap parity.

### Additional HCOMM channels

The transport-only probe reaches high throughput with one facade channel, and
the production queue high-watermark remains far below capacity. More channels
do not address D3 source amplification or serialized D8 and are deferred until
new telemetry proves saturation.

### One monolithic unmeasured rewrite

Combining token fan-out, count protocol, chunk lifecycle, and arrival-driven
copy in one change would make correctness failures and performance movement
unattributable. P7 keeps one retained baseline and one measured mechanism per
workstream while still aiming at the complete architectural target.

## 11. Source Map

CUDA references:

- `deep_ep/include/deep_ep/impls/dispatch.cuh`: token-oriented hidden TMA load,
  scale/top-k assembly, destination deduplication, direct NVLink store, Gin put,
  barrier, and programmatic launch completion.
- `deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh`: communication
  buffer load, final output TMA store, scale and metadata copy.

Ascend references:

- `csrc/backends/ascend/elastic/dispatch.asc`: D1-D4 producer, current vector
  payload function, D5-D8 epilogue, release protocol, and AOT launch wiring.
- `csrc/backends/ascend/elastic/topk_grouping.hpp`: subgroup destination
  grouping and owner election.
- `csrc/backends/ascend/elastic/layout.hpp`: grouping tiles, workspace, record,
  and pipeline layouts.
- `csrc/backends/ascend/elastic/dispatch_parallel.hpp`: consumer copy and
  expert-prefix plans.
- `csrc/backends/ascend/elastic/dispatch_pipeline_config.hpp`: current
  experimental destination-slot chunk selector.
- `csrc/backends/ascend/elastic/release_protocol.hpp`: payload, control,
  generation, signal, and acquire ordering.
- `csrc/backends/ascend/transport/aicore_transport_service.hpp`: command
  execution, SQ/CQ service, and barrier progression.
- `tests/ascend/benchmark/runtime.py`: maximum-rank event timing and logical
  traffic aggregation.
- `tests/ascend/hccs_benchmark/README.md`: P2P, all-to-all, and representative
  transport-only evidence.

## 12. Completion Criteria

P7 is complete when all of the following hold:

1. the eligible D3 path loads each token's aligned hidden body once and fans it
   out to all unique destination records;
2. the source-read work count no longer scales with unique destination count;
3. retained protocol or overlap changes have explicit profile evidence;
4. two-rank and five-operation correctness gates pass;
5. the final EP8 report uses the fixed workload and timing protocol;
6. mean Dispatch latency is no greater than `3.893045 ms`;
7. logical bandwidth is at least `2000 GB/s`;
8. p95 is stable and no unrelated operation materially regresses; and
9. rejected candidates are removed while their measurements remain documented.
