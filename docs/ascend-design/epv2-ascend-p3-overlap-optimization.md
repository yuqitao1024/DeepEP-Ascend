# EPv2 Ascend P3 Communication-Compute Overlap Optimization

## 1. Goal And Scope

P3 turns the current synchronous staged-transport execution into a measured,
request-driven pipeline. The work is deliberately ordered:

1. P3.0 measures producer, command publication, AICore service, completion
   wait, consumer, and epilogue work without changing the transport protocol.
2. P3.1 gives `DeviceRequest`, `flush_async()`, and `wait()` a real single-
   channel lifecycle.
3. P3.2 uses two request and workspace slots to overlap adjacent chunks.
4. P3.3 is considered only when the post-P3.2 SQ/CQ telemetry proves that one
   channel or one service drain is saturated.
5. P3.4 removes at most one measured resource or synchronization cost. P3.5
   remains outside this implementation.

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

The final candidate was commit
`17dd7ed9f23a9f5fdadfd3137a356d0ca9319e3f`, archived from a clean tree as
`deepep-p3-final-17dd7ed-88d09052.tar.gz` with SHA-256
`88d090525eb580d1c8e39bd08f7894af0252e4258344851ead779c2e0b1c93c0`.
The remote staging directory was
`/home/pyptouser/yuqitao/deepep-staging/p3-final-88d090525eb580d1`.
This final archive includes the corrected command accounting, all drain-wait
paths, observed SQ/CQ depths and high-water marks, strict host validation, and
the stream-ordered final profiling boundary. Results from the earlier
`f73da24` archive are superseded because its queue and wait telemetry was
incomplete.

All NPU work ran serially through TaskQueue with CANN 9.2,
`hcomm-deepep-current`, and the qualified Python 3.10 virtual environment:

| TaskQueue task | Scope | Terminal result |
| --- | --- | --- |
| `task_20260824_221053_12428769443` | Final archive extraction and production/SIMT/AICore builds | exit 0 |
| `task_20260824_224053_156495113079` | Focused two-rank `profile-mixed` and teardown retry on devices 6,7 | exit 0 |
| `task_20260824_224207_157664916330` | Full two-rank 12-case suite and production correctness on devices 6,7 | exit 0 |
| `task_20260824_163919_5752582788` | Immutable baseline reconstruction and build | exit 0 |
| `task_20260824_224637_16141996053` | Final EP8 disabled-profile ABBA control | exit 0 |
| `task_20260824_225210_166672421792` | Final EP8 enabled-profile representative case | exit 0 |

The two-rank transport suite passed `put`, `put-value64`, `faa64`, `signal`,
`signal-set`, `flush`, `payload-signal-order`, `barrier-repeat`, `queue-wrap`,
`profile-mixed`, `phase-boundary`, and `teardown`, all with
`diagnostics=kNone`. The production benchmark passed all five operations. Its
result is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-full-r1/production-correctness.json`
(SHA-256 `a1fcbf822889aab86098cf734fe47df56a55d822cc77e2de9e750b1f439ffdb1`).

The EP8 command used `torch.distributed.run --standalone --nproc-per-node=8`
with `bench_ep.py --cases ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0
--num-sms 72 --warmups 30 --iterations 30`; the enabled run additionally used
`--profile-stages`. It retained the byte-identical workload manifest
`/home/pyptouser/yuqitao/deepep-results/ascend950-representative-60e3d08/workload.json`
(SHA-256 `98d9dc5ff7b8f31afbc9589b037fd658d99b85b739b8572de1751b9e979eb623`,
fingerprint `d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00`).

The enabled schema-v3 result is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-profile/benchmark.json`
(SHA-256 `b6f456c9527428b5425807621a86f368b5c6714a4047af2f1d948567ce3451c7`).
The disabled ABBA result directory is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba`; its four
JSON SHA-256 values are recorded below.

### 10.1.1 Exact archive, environment, and build commands

The local candidate archive command below was re-run against the immutable
commit and reproduced the recorded candidate archive SHA-256. The remote copy
used for build and benchmark was
`/home/pyptouser/yuqitao/deepep-archives/deepep-p3-final-17dd7ed-88d09052.tar.gz`;
the staging command verifies it before extraction.

```bash
git archive --format=tar 17dd7ed9f23a9f5fdadfd3137a356d0ca9319e3f | gzip -n > /tmp/deepep-p3-final-17dd7ed.tar.gz
sha256sum /tmp/deepep-p3-final-17dd7ed.tar.gz
# 88d090525eb580d1c8e39bd08f7894af0252e4258344851ead779c2e0b1c93c0
```

Every remote task loaded this exact context before its build or benchmark:

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export HCOMM_ROOT=/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann
export PATH="$HCOMM_ROOT/bin:/home/pyptouser/yuqitao/tools/cmake-3.28.4/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$HCOMM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$HCOMM_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="$HCOMM_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CPATH="$HCOMM_ROOT/include${CPATH:+:$CPATH}"
export CMAKE_INCLUDE_PATH="$HCOMM_ROOT/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_LIBRARY_PATH="$HCOMM_ROOT/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
export PYTHONPATH="$HCOMM_ROOT/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
source /home/pyptouser/yuqitao/venvs/deepep-ascend-py310/bin/activate
```

Task `task_20260824_221053_12428769443` staged and built the candidate as
follows. Its durable provenance is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-build/provenance.txt`; the
runner path is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-build/simt_urma_runtime/deep_ep_ascend_urma_runner.so`.

```bash
archive=/home/pyptouser/yuqitao/deepep-archives/deepep-p3-final-17dd7ed-88d09052.tar.gz
archive_sha256=88d090525eb580d1c8e39bd08f7894af0252e4258344851ead779c2e0b1c93c0
source_dir=/home/pyptouser/yuqitao/deepep-staging/p3-final-88d090525eb580d1
result_dir=/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-build
runner_dir="$result_dir/simt_urma_runtime"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$archive_sha256"
test ! -e "$source_dir"
mkdir -p "$source_dir" "$result_dir"
tar -xzf "$archive" -C "$source_dir"
cd "$source_dir"
DEEP_EP_PLATFORM=ascend python setup.py build_ext --inplace
cmake -S tests/ascend/simt_urma -B "$runner_dir" -DDEEP_EP_ASCEND_STAGED_URMA=ON -DDEEP_EP_BUILD_URMA_RUNTIME=ON
cmake --build "$runner_dir" --target deep_ep_ascend_urma_runner --parallel 2
test -s "$runner_dir/deep_ep_ascend_urma_runner.so"
```

The original supplied baseline staging directory was absent before the first
AB attempt; this was staging loss, not a P3 benchmark failure. Task
`task_20260824_163919_5752582788` reconstructed and built the immutable
baseline, whose durable build provenance is
`/home/pyptouser/yuqitao/deepep-results/p3-5fc0940a-baseline-build/provenance.txt`.

```bash
archive=/home/pyptouser/yuqitao/deepep-archives/deepep-main-2bd268f-5fc0940a.tar.gz
archive_sha256=5fc0940a373391424243188400e876ffbd0bf56840c1fe4769384b4707b6e983
source_dir=/home/pyptouser/yuqitao/deepep-staging/baseline-5fc0940a37339142
result_dir=/home/pyptouser/yuqitao/deepep-results/p3-5fc0940a-baseline-build
test "$(sha256sum "$archive" | awk '{print $1}')" = "$archive_sha256"
test ! -e "$source_dir"
mkdir -p "$source_dir" "$result_dir"
tar -xzf "$archive" -C "$source_dir"
cd "$source_dir"
DEEP_EP_PLATFORM=ascend python setup.py build_ext --inplace
```

Task `task_20260824_224207_157664916330` used the final candidate staging and
runner above, unset profiling, and ran the full two-rank qualification. Its
durable result directory is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-full-r1` with
`simt-aicore.log`, `production-correctness.log`,
`production-correctness.json`, and `provenance.txt`.

```bash
unset DEEP_EP_ASCEND_PROFILE_STAGES
cd /home/pyptouser/yuqitao/deepep-staging/p3-final-88d090525eb580d1
python -m torch.distributed.run --standalone --nproc-per-node=2 tests/ascend/simt_urma/run_two_rank_probe.py --runner /home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-build/simt_urma_runtime/deep_ep_ascend_urma_runner.so --cases put,put-value64,faa64,signal,signal-set,flush,payload-signal-order,barrier-repeat,queue-wrap,profile-mixed,phase-boundary,teardown | tee /home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-full-r1/simt-aicore.log
python -m torch.distributed.run --standalone --nproc-per-node=2 tests/ascend/benchmark/bench_ep.py --cases ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0 --num-tokens 16 --hidden 128 --num-topk 2 --num-experts 4 --num-sms 72 --warmups 1 --iterations 1 --output /home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-full-r1/production-correctness.json | tee /home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-full-r1/production-correctness.log
```

Task `task_20260824_224637_16141996053` ran the disabled ABBA control after
the common environment and `unset DEEP_EP_ASCEND_PROFILE_STAGES`.

```bash
baseline_source=/home/pyptouser/yuqitao/deepep-staging/baseline-5fc0940a37339142
candidate_source=/home/pyptouser/yuqitao/deepep-staging/p3-final-88d090525eb580d1
manifest=/home/pyptouser/yuqitao/deepep-results/ascend950-representative-60e3d08/workload.json
result_dir=/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba
case_id=ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0
cd "$baseline_source" && python -m torch.distributed.run --standalone --nproc-per-node=8 tests/ascend/benchmark/bench_ep.py --cases "$case_id" --workload-manifest "$manifest" --num-sms 72 --warmups 30 --iterations 30 --output "$result_dir/baseline-a.json"
cd "$candidate_source" && python -m torch.distributed.run --standalone --nproc-per-node=8 tests/ascend/benchmark/bench_ep.py --cases "$case_id" --workload-manifest "$manifest" --num-sms 72 --warmups 30 --iterations 30 --output "$result_dir/candidate-a.json"
cd "$candidate_source" && python -m torch.distributed.run --standalone --nproc-per-node=8 tests/ascend/benchmark/bench_ep.py --cases "$case_id" --workload-manifest "$manifest" --num-sms 72 --warmups 30 --iterations 30 --output "$result_dir/candidate-b.json"
cd "$baseline_source" && python -m torch.distributed.run --standalone --nproc-per-node=8 tests/ascend/benchmark/bench_ep.py --cases "$case_id" --workload-manifest "$manifest" --num-sms 72 --warmups 30 --iterations 30 --output "$result_dir/baseline-b.json"
```

Task `task_20260824_225210_166672421792` ran the enabled candidate profile from
`/home/pyptouser/yuqitao/deepep-staging/p3-final-88d090525eb580d1`. Its durable
provenance is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-profile/provenance.txt`.

```bash
cd /home/pyptouser/yuqitao/deepep-staging/p3-final-88d090525eb580d1
python -m torch.distributed.run --standalone --nproc-per-node=8 tests/ascend/benchmark/bench_ep.py --cases ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0 --workload-manifest /home/pyptouser/yuqitao/deepep-results/ascend950-representative-60e3d08/workload.json --num-sms 72 --warmups 30 --iterations 30 --profile-stages --output /home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-profile/benchmark.json
```

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
| dispatch | 36.878 | 37.510 | +1.715% | 0.529% | 1.555% |
| expanded_dispatch | 39.049 | 38.497 | -1.414% | 0.299% | 4.179% |
| cached_dispatch | 84.303 | 84.635 | +0.393% | 1.310% | 0.008% |
| combine | 140.121 | 140.191 | +0.051% | 0.526% | 0.466% |
| reduced_combine | 167.462 | 169.555 | +1.250% | 0.668% | 0.515% |

Dispatch and reduced combine move by 1.715% and 1.250%; those changes remain
within the observed candidate pair variation or the combined control/candidate
run-to-run spread. Expanded dispatch varies by 4.179% between candidate runs,
which is larger than its -1.414% pair-mean delta. Cached dispatch and combine
are effectively flat. P3.0 profiling is therefore retained without applying
an arbitrary overhead threshold.

The full per-run values are durable evidence, not a temporary-report-only
artifact:

| JSON artifact | Durable result path | SHA-256 | Operation | Mean ms | p50 ms | p95 ms | Logical GB/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline A | `/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba/baseline-a.json` | `a560798f7f6ed7146b16295ef4a29e30741976b6ecc37c1469a6d43aa8a842e1` | dispatch | 36.780 | 36.854 | 39.333 | 211.691 |
| baseline A | same | same | expanded_dispatch | 39.107 | 39.095 | 41.882 | 236.566 |
| baseline A | same | same | cached_dispatch | 84.852 | 85.178 | 86.383 | 91.761 |
| baseline A | same | same | combine | 140.488 | 140.537 | 141.992 | 77.595 |
| baseline A | same | same | reduced_combine | 168.019 | 168.180 | 169.461 | 64.881 |
| candidate A | `/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba/candidate-a.json` | `125c1ee890bd36eafa369d72e9686aebe3698b0b7cb8941a84848d378d74a3e7` | dispatch | 37.799 | 37.906 | 40.523 | 205.984 |
| candidate A | same | same | expanded_dispatch | 39.285 | 39.285 | 41.508 | 235.498 |
| candidate A | same | same | cached_dispatch | 84.638 | 84.755 | 86.417 | 91.993 |
| candidate A | same | same | combine | 140.517 | 140.633 | 142.002 | 77.579 |
| candidate A | same | same | reduced_combine | 169.991 | 169.858 | 172.444 | 64.128 |
| candidate B | `/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba/candidate-b.json` | `92b1f947b0b853d80b4acecf7b773d53c2c01fb3306002270c3eb8e1fca06b4c` | dispatch | 37.221 | 37.159 | 38.873 | 209.186 |
| candidate B | same | same | expanded_dispatch | 37.709 | 37.323 | 39.812 | 245.338 |
| candidate B | same | same | cached_dispatch | 84.631 | 84.594 | 86.076 | 92.000 |
| candidate B | same | same | combine | 139.866 | 139.952 | 141.389 | 77.940 |
| candidate B | same | same | reduced_combine | 169.120 | 168.976 | 170.933 | 64.459 |
| baseline B | `/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba/baseline-b.json` | `1f07b19f85b5fd4aa822d1052f26ddec4b6855e7c8088c8360e7799d7a0e573c` | dispatch | 36.975 | 36.839 | 38.777 | 210.577 |
| baseline B | same | same | expanded_dispatch | 38.991 | 38.785 | 40.540 | 237.274 |
| baseline B | same | same | cached_dispatch | 83.755 | 83.872 | 85.316 | 92.963 |
| baseline B | same | same | combine | 139.753 | 139.960 | 142.142 | 78.003 |
| baseline B | same | same | reduced_combine | 166.904 | 166.830 | 169.279 | 65.314 |

The durable ABBA provenance file is
`/home/pyptouser/yuqitao/deepep-results/p3-final-88d09052-ep8-abba/provenance.txt`.
For each operation, `control_mean = (baseline_A + baseline_B) / 2`,
`candidate_mean = (candidate_A + candidate_B) / 2`, and
`delta_percent = (candidate_mean / control_mean - 1) * 100`. The corresponding
pair variation is exactly `(max(A, B) / min(A, B) - 1) * 100`; this is the
formula used for both control and candidate variation columns above.

### 10.3 Enabled EP8 latency and pipeline evidence

The following device-event figures are max-rank operation values. Logical
bandwidth uses the unchanged benchmark formulas.

| Operation | Mean ms | p50 ms | p95 ms | Logical GB/s | Producer cycles/ms | Network cycles/ms | Consumer cycles/ms | Ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 36.176 | 36.171 | 38.413 | 215.231 | 4,469,515 / 4.470 | 6,747,839 / 6.748 | 8,786,682 / 8.787 | 2.277x |
| expanded_dispatch | 38.347 | 38.304 | 40.340 | 241.257 | 4,483,951 / 4.484 | 6,642,344 / 6.642 | 10,299,972 / 10.300 | 2.080x |
| cached_dispatch | 85.689 | 85.699 | 87.071 | 90.865 | 50,759,224 / 50.759 | 10,934,981 / 10.935 | 8,813,497 / 8.813 | 1.389x |
| combine | 141.098 | 140.983 | 143.446 | 77.260 | 47,146,601 / 47.147 | 60,301,742 / 60.302 | 18,423,982 / 18.424 | 2.087x |
| reduced_combine | 168.954 | 168.657 | 171.146 | 64.522 | 74,430,398 / 74.430 | 62,895,605 / 62.896 | 18,419,423 / 18.419 | 2.092x |

The source of cycle conversion is the checked-in Ascend DevKit reference
`/root/aiagent/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md`:
Ascend 950PR/950DT `GetSystemCycle()` is 1 GHz. Thus one cycle is 0.001 us,
`time_us = cycles / 1000`, and `time_ms = cycles / 1000000`. The phase tables
below show cycles with the converted milliseconds; the detailed task report
also retains microseconds for every value.

The benchmark takes the maximum of each phase independently across ranks.
Consequently, a reported phase vector is a component-wise worst-rank
diagnostic: its producer, network, and consumer components can come from
different ranks and do not describe one observed rank's end-to-end trace. The
resulting overlap ceilings are directional optimistic bounds under that
aggregation, before fill/drain, launch, and resource-contention costs; they are
not measured per-rank speedups.

| Operation | Producer | Publication | Service submit | CQ wait | Consumer wait | Consumer compute | Epilogue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 4,469,515 / 4.470 ms | 163,049 / 0.163 ms | 5,734,015 / 5.734 ms | 850,775 / 0.851 ms | 604,662 / 0.605 ms | 8,178,839 / 8.179 ms | 3,181 / 0.003 ms |
| expanded_dispatch | 4,483,951 / 4.484 ms | 162,256 / 0.162 ms | 5,687,829 / 5.688 ms | 792,259 / 0.792 ms | 604,651 / 0.605 ms | 9,692,163 / 9.692 ms | 3,158 / 0.003 ms |
| cached_dispatch | 50,759,224 / 50.759 ms | 162,739 / 0.163 ms | 9,967,340 / 9.967 ms | 804,902 / 0.805 ms | 656,486 / 0.656 ms | 8,154,135 / 8.154 ms | 2,876 / 0.003 ms |
| combine | 47,146,601 / 47.147 ms | 51,360,567 / 51.361 ms | 7,383,877 / 7.384 ms | 1,557,298 / 1.557 ms | 10,782,474 / 10.782 ms | 7,639,255 / 7.639 ms | 2,253 / 0.002 ms |
| reduced_combine | 74,430,398 / 74.430 ms | 51,625,863 / 51.626 ms | 9,713,533 / 9.714 ms | 1,556,209 / 1.556 ms | 10,782,991 / 10.783 ms | 7,633,962 / 7.634 ms | 2,470 / 0.002 ms |

Raw named-stage spans are retained below as `cycles / us / ms`:

| Operation | Raw stage spans |
| --- | --- |
| dispatch | `producer_control` 6,964 / 6.964 / 0.007; `producer_group` 28,638 / 28.638 / 0.029; `producer_prefix` 575,271 / 575.271 / 0.575; `producer_record` 3,860,050 / 3860.050 / 3.860; `producer_release` 6,693,293 / 6693.293 / 6.693; `epilogue_acquire` 51,989 / 51.989 / 0.052; `epilogue_validate` 500,026 / 500.026 / 0.500; `epilogue_validate_reduce` 53,080 / 53.080 / 0.053; `epilogue_expert_count` 339,379 / 339.379 / 0.339; `epilogue_expert_prefix` 2,787,867 / 2787.867 / 2.788; `epilogue_metadata` 205,314 / 205.314 / 0.205; `epilogue_copy` 4,857,048 / 4857.048 / 4.857; `epilogue_complete` 3,181 / 3.181 / 0.003 |
| expanded_dispatch | `producer_control` 7,128 / 7.128 / 0.007; `producer_group` 28,185 / 28.185 / 0.028; `producer_prefix` 574,537 / 574.537 / 0.575; `producer_record` 3,875,302 / 3875.302 / 3.875; `producer_release` 6,606,836 / 6606.836 / 6.607; `epilogue_acquire` 51,767 / 51.767 / 0.052; `epilogue_validate` 500,857 / 500.857 / 0.501; `epilogue_validate_reduce` 53,015 / 53.015 / 0.053; `epilogue_expert_count` 340,163 / 340.163 / 0.340; `epilogue_expert_prefix` 2,787,268 / 2787.268 / 2.787; `epilogue_metadata` 527,412 / 527.412 / 0.527; `epilogue_copy` 6,047,142 / 6047.142 / 6.047; `epilogue_complete` 3,158 / 3.158 / 0.003 |
| cached_dispatch | `producer_control` 46,869,182 / 46869.182 / 46.869; `producer_group` 28,588 / 28.588 / 0.029; `producer_prefix` 687 / 0.687 / 0.001; `producer_record` 3,884,856 / 3884.856 / 3.885; `producer_release` 10,909,994 / 10909.994 / 10.910; `epilogue_acquire` 52,219 / 52.219 / 0.052; `epilogue_validate` 552,806 / 552.806 / 0.553; `epilogue_validate_reduce` 52,582 / 52.582 / 0.053; `epilogue_expert_count` 340,222 / 340.222 / 0.340; `epilogue_expert_prefix` 2,807,387 / 2807.387 / 2.807; `epilogue_metadata` 168,398 / 168.398 / 0.168; `epilogue_copy` 4,849,467 / 4849.467 / 4.849; `epilogue_complete` 2,876 / 2.876 / 0.003 |
| combine | `producer_control` 8,350 / 8.350 / 0.008; `producer_plan` 1,029,890 / 1029.890 / 1.030; `producer_plan_prefix` 776,143 / 776.143 / 0.776; `producer_record` 45,335,537 / 45335.537 / 45.336; `producer_release` 59,179,762 / 59179.762 / 59.180; `epilogue_acquire` 56,154 / 56.154 / 0.056; `epilogue_validate` 554,399 / 554.399 / 0.554; `epilogue_validate_reduce` 10,171,921 / 10171.921 / 10.172; `epilogue_reduce` 7,616,493 / 7616.493 / 7.616; `epilogue_weights` 24,562 / 24.562 / 0.025; `epilogue_complete` 2,253 / 2.253 / 0.002 |
| reduced_combine | `producer_control` 8,287 / 8.287 / 0.008; `producer_plan` 1,037,270 / 1037.270 / 1.037; `producer_plan_prefix` 775,225 / 775.225 / 0.775; `producer_record` 72,613,467 / 72613.467 / 72.613; `producer_release` 61,235,614 / 61235.614 / 61.236; `epilogue_acquire` 55,997 / 55.997 / 0.056; `epilogue_validate` 551,992 / 551.992 / 0.552; `epilogue_validate_reduce` 10,175,042 / 10175.042 / 10.175; `epilogue_reduce` 7,611,430 / 7611.430 / 7.611; `epilogue_weights` 24,996 / 24.996 / 0.025; `epilogue_complete` 2,470 / 2.470 / 0.002 |

Command and queue evidence:

| Operation | Commands | Puts | Payload bytes per rank | Final SQ/CQ | SQ/CQ HWM | Wait cycles per rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 30 | 7 | 285,118,320-287,399,024 | 0/0 | 4/4 | 762,883-850,775 |
| expanded_dispatch | 30 | 7 | 285,118,320-287,399,024 | 0/0 | 4/4 | 756,751-792,259 |
| cached_dispatch | 30 | 7 | 285,118,320-287,399,024 | 0/0 | 4/4 | 757,966-804,902 |
| combine | 30 | 7 | 543,110,512-548,352,112 | 0/0 | 4/4 | 1,479,098-1,557,298 |
| reduced_combine | 30 | 7 | 543,110,512-548,352,112 | 0/0 | 4/4 | 1,469,585-1,556,209 |

Each operation and rank emitted 30 commands with seven payload puts. The
observed SQ/CQ depths were 0/0 at completion and their corrected high-water
marks were 4/4. Dispatch payload bytes ranged from 285,118,320 to 287,399,024
per rank; combine payload bytes ranged from 543,110,512 to 548,352,112 per
rank. The measured wait-cycle ranges were 762,883-850,775 for dispatch,
756,751-792,259 for expanded dispatch, 757,966-804,902 for cached dispatch,
1,479,098-1,557,298 for combine, and 1,469,585-1,556,209 for reduced combine.

### 10.4 Decision

- P3.1 may proceed after review as the bounded single-channel request
  lifecycle described above; it was not implemented in P3.0.
- P3.2 may proceed after P3.1 review, first for dispatch and expanded dispatch:
  their 2.277x and 2.080x ceilings leave material producer/network/consumer
  overlap opportunity. Cached dispatch is producer-dominated (1.389x ceiling)
  and is not the initial chunking target.
- P3.3 is rejected/deferred. With final SQ/CQ depths 0/0 and high-water marks
  only 4/4 against the EP8 command capacity of 36, this qualification shows no
  queue saturation evidence for another channel or an additional service
  drain.

### 10.5 Disabled-template residual cleanup

Commit `5970e2261ffbb8358bec50a1d20436f48e961203` moves the disabled
`profile_block` lifetime and all profile calls inside caller-level
`if constexpr (ProfileEnabled)` branches. The immutable archive is
`/tmp/deepep-p3-profile-lifetime-5970e22.tar.gz`, SHA-256
`25f4fe0757fc6698ed84901ef1b0bfb58842acfc24e3e7c953979ee931b3c94f`.
The remote source is
`/home/pyptouser/yuqitao/deepep-staging/p3-profile-lifetime-25f4fe07`.
Generated code confirms that this is a real disabled-path reduction: the
disabled dispatch kernel shrank from `0x37c90` bytes at `ccc624e` to
`0x37aa0` bytes at `5970e22`, a 496-byte reduction.

TaskQueue task `task_20260825_001526_24835973056` built the archive;
`task_20260825_002211_25969545668` passed all 12 two-rank SIMT/AICore cases
with `diagnostics=kNone` and the production five-operation mini-case.
Task `task_20260825_002436_26393331534` then exposed a benchmark-method issue:
its ABBA process switched from baseline back to candidate, and candidate B
cached dispatch started with eight samples between roughly 181 and 303 ms
before returning to 70.6-78.8 ms. That mixed-binary result is retained as
diagnostic evidence but is not used for acceptance.

The final control used separate candidate-only and baseline-only processes:

| Task | Binary | Result directory | Terminal result |
| --- | --- | --- | --- |
| `task_20260825_003333_31844097674` | `5970e22` candidate, two consecutive runs | `/home/pyptouser/yuqitao/deepep-results/p3-profile-lifetime-25f4fe07-ep8-isolated` | exit 0 |
| `task_20260825_004855_37587428376` | `5fc0940a` baseline, two consecutive runs | `/home/pyptouser/yuqitao/deepep-results/p3-baseline-5fc0940a-ep8-isolated` | exit 0 |

| Run | Operation | Mean ms | p50 ms | p95 ms | Logical GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| candidate A | dispatch | 38.385 | 38.247 | 40.151 | 202.842 |
| candidate A | expanded dispatch | 39.375 | 39.378 | 40.664 | 234.958 |
| candidate A | cached dispatch | 85.093 | 85.085 | 86.945 | 91.501 |
| candidate A | combine | 139.757 | 139.815 | 141.879 | 78.001 |
| candidate A | reduced combine | 167.624 | 167.880 | 169.334 | 65.034 |
| candidate B | dispatch | 31.311 | 31.113 | 32.327 | 248.672 |
| candidate B | expanded dispatch | 36.419 | 36.750 | 38.506 | 254.031 |
| candidate B | cached dispatch | 78.204 | 77.531 | 81.432 | 99.562 |
| candidate B | combine | 136.447 | 136.854 | 139.560 | 79.893 |
| candidate B | reduced combine | 166.589 | 166.480 | 168.602 | 65.438 |
| baseline A | dispatch | 36.828 | 36.443 | 39.712 | 211.420 |
| baseline A | expanded dispatch | 37.776 | 37.507 | 39.937 | 244.905 |
| baseline A | cached dispatch | 84.087 | 84.008 | 86.331 | 92.595 |
| baseline A | combine | 139.734 | 139.757 | 142.073 | 78.014 |
| baseline A | reduced combine | 166.802 | 166.763 | 168.495 | 65.354 |
| baseline B | dispatch | 37.576 | 37.462 | 39.747 | 207.211 |
| baseline B | expanded dispatch | 39.781 | 39.610 | 41.888 | 232.561 |
| baseline B | cached dispatch | 82.852 | 83.133 | 84.737 | 93.976 |
| baseline B | combine | 141.303 | 141.444 | 143.029 | 77.148 |
| baseline B | reduced combine | 168.525 | 168.381 | 170.861 | 64.686 |

The isolated candidate pair is itself warm-state sensitive, especially for
dispatch, so the lower candidate pair mean is not claimed as a stable
speedup. The acceptance statement is narrower: no operation shows a stable
regression against both isolated baseline runs, and the disabled kernel no
longer contains the measured residual profile code. P3.0 is therefore closed.

| Artifact | SHA-256 |
| --- | --- |
| candidate A JSON | `b49ba5bb821628c278bd4e47ae5e671525a7bdbd60e75222470a092c1d067af7` |
| candidate B JSON | `f65258a72a4964170cdcf4101ee1e47990e62c6aba16b9fc75b4f159d027a5be` |
| baseline A JSON | `8d31e317e05f041ed10bb463228302ebe2035ebc8ea342d7b39ce0f583224562` |
| baseline B JSON | `50f52618b3e15542e9714f32a6e57e753d11b4b97b64c234268aa42544ec6969` |

## 11. P3.1 Hardware Qualification

P3.1 is accepted for the direct, single-host Ascend 950 scale-up path. It
adds the bounded request lifecycle described in section 6 and advertises
`kAsyncCompletion`. This is a correctness and ABI qualification; it does not
claim an isolated latency improvement.

The immutable input was captured from the P3.1 working tree based on commit
`5970e2261ffbb8358bec50a1d20436f48e961203` as
`/home/pyptouser/yuqitao/deepep-archives/deepep-p31-final-6edc5b69.tar.gz`.
Its SHA-256 is
`6edc5b69621750fb3227237f87599c67d2294da04250868577ea7e5828f6aecf`.
TaskQueue task `task_20260825_015319_90796624703` used CANN 9.2, the pinned
`hcomm-deepep-current` package, and the qualified Python 3.10 environment on
devices 0,1. It completed with exit 0 after performing all of the following:

- built the production extension with `kAsyncCompletion` advertised;
- built the SIMT/AICore runtime runner and the standalone facade probe;
- passed all 13 two-rank lifecycle cases with `diagnostics=kNone`, including
  `async-lifecycle`;
- passed the five-operation production mini-case.

The durable result directory is
`/home/pyptouser/yuqitao/deepep-results/p31-final-6edc5b69`. The exact
artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| production extension | `f63bfcbbe52f48be89cdeb055c4efccd2fb6a9fa2c9f9479ac73a96f5c209ede` |
| SIMT/AICore runner | `269286e1138ac6beb21eda93ffefda6a47e00014a14dbef088983bb61eb5b7d1` |
| production correctness JSON | `433d7a4e06c753cb97e8777a171dfd6936277048236cfda35bee303d5199fcd9` |
| SIMT/AICore validation log | `6ff02fe70d90789e7eceaacac6684cb0d43304ced7e8bfc18ac9c86201ecd8e1` |
| facade build log | `8fc183e703a7f716d36c4e6658f17d45fd3b64f9eab4753e9a452135eadacd6a` |

The device suite passed `put`, `put-value64`, `faa64`, `signal`,
`signal-set`, `flush`, `async-lifecycle`, `payload-signal-order`,
`barrier-repeat`, `queue-wrap`, `profile-mixed`, `phase-boundary`, and
`teardown`. The lifecycle case verifies that an asynchronously published
flush completes the exact request target and preserves the request generation
and terminal status. Host contract probes cover empty and pending misuse,
range and ABI failures, queue reset, stale service generation, diagnostic
propagation, short completion, timeout, and idempotent terminal waits.

## 12. P3.2 Hardware Qualification

P3.2 is retained as an opt-in direct-dispatch pipeline for the single-host
Ascend 950 scale-up path. The setting
`DEEP_EP_ASCEND_DISPATCH_PIPELINE_CHUNK_SLOTS=2048` divides an 8192-slot shard
into four chunks and alternates two request/workspace slots. It applies only
to synchronous, uncached, CPU-split dispatch and expanded dispatch. Cached
dispatch, combine, hybrid routing, physical scale-out, and async/event modes
continue to use their existing paths.

The immutable candidate archive is
`/home/pyptouser/yuqitao/deepep-archives/deepep-p32-1fae4521.tar.gz`, SHA-256
`1fae4521c5c8bebf19711fa4bf4e83a08c1cf5efce26a93804ccb1f7018622ef`.
It is based on P3.1 commit `d029ce73413bf09e238335d8c2bb482245061f8e`.
The production extension and SIMT/AICore runner built from that archive have
SHA-256 values `c9dfa5c7ec440cddc3bbe924340d04844250c12fc907890534b7fab8b6334b59`
and `4318e227d056610e12e1915ff55fa0e4a28096fdea19d3082d20e0895e0fdfca`.

### 12.1 Build and correctness gates

| TaskQueue task | Scope | Result |
| --- | --- | --- |
| `task_20260825_024612_124368928903` | Production extension, SIMT/AICore runner, facade, 13 lifecycle cases, control and pipeline mini-cases | exit 0 |
| `task_20260825_030215_137855619344` | Two-rank full 144-case matrix with two chunks | 144 cases and 720 operations passed, exit 0 |

The 144-case report is
`/home/pyptouser/yuqitao/deepep-results/p32-1fae4521-full144-r2/benchmark.json`,
SHA-256 `f753a34d0046125f622324a9d6c904bb6ac2670b8c8065a773b3128c08335d13`.
The first matrix submission, `task_20260825_025823_132996312277`, failed
before importing DeepEP because its submission wrapper expanded the CANN
library variables too early and could not load `libhccl.so`. The corrected
task kept the variables quoted until the TaskQueue shell and required no code
change.

### 12.2 EP8 chunk sweep

All runs used the representative FP8 case, the same workload manifest, 72
data blocks, 30 warmups, and 30 measured iterations. The first task swept
control, 4096, and 2048 slots; the next two tasks reversed or repeated the
control/2048 order to expose process and device warm-state sensitivity.

| TaskQueue task | Run order | Result directory | Result |
| --- | --- | --- | --- |
| `task_20260825_030415_140906913651` | control, 4096, 2048 | `/home/pyptouser/yuqitao/deepep-results/p32-1fae4521-ep8-sweep` | exit 0 |
| `task_20260825_030828_14772647950` | 2048, control | `/home/pyptouser/yuqitao/deepep-results/p32-1fae4521-ep8-reverse` | exit 0 |
| `task_20260825_031141_15008704710` | control, 2048 | `/home/pyptouser/yuqitao/deepep-results/p32-1fae4521-ep8-confirm` | exit 0 |

The 4096-slot result regressed dispatch mean from 37.242 ms to 39.972 ms and
is rejected. The 2048-slot setting has one cold run with little benefit, but
the run-level median across all three orders improves both targeted
operations and their tails:

| Operation | Control median mean | 2048 median mean | Delta | Control median p95 | 2048 median p95 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dispatch | 38.017 ms | 30.871 ms | -18.80% | 39.599 ms | 34.686 ms | -12.41% |
| expanded dispatch | 39.126 ms | 33.390 ms | -14.66% | 40.885 ms | 37.516 ms | -8.24% |

The three 2048-slot dispatch means are 28.144, 37.976, and 30.871 ms. The
expanded-dispatch means are 30.834, 39.760, and 33.390 ms. The cold result is
retained rather than excluded: the optimization has warm-state sensitivity,
so the table reports the median of complete runs instead of only its best
sample. Untargeted operations also moved with device state, but by smaller
median amounts: cached dispatch -8.72%, combine -3.48%, and reduced combine
-1.19%. This is why the evidence is reported as a three-order qualification,
not as a claim that every observed delta comes from chunk overlap.

The control/4096/2048 JSON SHA-256 values are respectively
`e8526653273a8c23ca469056e53fd6d44dd1e01cd5643dd98122cc825102ecdc`,
`955a68e39c5d9ad6514ba18cb3a8c9432a03c4f386b11381639e2cc6fdd0e935`,
and `df2d4f530b2ef8e28053b1b5e161d1abff78ccd1778739959ddbf3b08dd483c7`.
The reverse-run 2048/control hashes are
`921e519d651c785263fbefecb87c3525b14fd7fcd85a38c787f7071e7cda66ad`
and `fe1626cf94abfa0f44d343827ef72a79ae265b3ff69856d61dc14017d6bff3ce`.
The final confirmation control/2048 hashes are
`3f124d24b7b1c6aa8dc8dd6f7827b98efacb68942e63a4c46ce06f6cba5d3871`
and `ea66787952503e5bbbfd63b9cd1bcb24edf9791275132df65c92c280f5028838`.

### 12.3 Measured cross-chunk overlap

Task `task_20260825_031544_153457425045` profiled the retained 2048-slot
setting on the full EP8 workload. Its report is
`/home/pyptouser/yuqitao/deepep-results/p32-1fae4521-ep8-profile/benchmark.json`,
SHA-256 `abeeff416aa015f7dc483a9f2ebf1675c192baa281151848fd3028095925eb2a`.

Within one chunk, producer record completes before producer release starts.
Therefore the intersection of the aggregate record and release intervals can
only come from a later chunk being recorded while an earlier chunk is being
released. Every rank has a positive intersection: dispatch ranges from
3,804,069 to 3,912,264 cycles, and expanded dispatch ranges from 3,807,660 to
3,859,121 cycles. This is direct cross-chunk overlap evidence, not the
optimistic ceiling computed from independent phase maxima.

## 13. P3.3 Queue And Channel Decision

P3.3 is closed with no production change. Chunking increases the profiled
dispatch command count from 30 to 31, but every rank still ends at SQ/CQ depth
0/0 and the maximum observed SQ/CQ high-water marks remain 4/4. The EP8
command capacity is 36, so only about 11% of the queue is occupied at the
high-water point. Dispatch and expanded dispatch both show this result.

Adding another channel or a second service drain would therefore target no
observed saturation, while introducing another ordering, error-propagation,
and resource-contention surface. P3.3 remains deferred until repeated
telemetry approaches queue capacity or service submission becomes the
measured critical bottleneck.
