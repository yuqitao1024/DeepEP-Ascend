# Ascend EPv2 benchmark

This suite covers the supported BF16 and FP8 performance intersection plus the
accepted BF16 functional modes from `tests/elastic/test_ep.py` on Ascend 950.
Both profiles use the same
144-case enumeration, deterministic routing manifest, operation preparation,
correctness references, warmup/sample counts, rank aggregation, and logical
byte formulas.

The exhaustive case table and the statistical definitions are in
[`docs/superpowers/specs/2026-08-18-ascend-epv2-benchmark-parity-design.md`](../../../docs/superpowers/specs/2026-08-18-ascend-epv2-benchmark-parity-design.md).

## Suite classification

| Classification | Cases | Ascend behavior | Reason |
| --- | ---: | --- | --- |
| Current BF16 performance | 12 | Correctness preflight and timing | |
| Current FP8 performance | 12 | Correctness preflight and timing | |
| Current BF16 functional | 60 | Correctness preflight and timing | |
| Deferred FP8 functional | 60 | Listed, not executed | `fp8_full_row_deferred_3f` |
| Total inventory | 144 | 84 supported, 60 deferred | |

Each supported case checks normal dispatch, expanded dispatch, cached dispatch,
cached expanded dispatch with zero padding, combine, and reduced/expanded
combine. Correctness is a preflight gate for each supported case. The five
operations other than cached expanded padding are timed, producing 420
operation records per complete supported run.

Phase 3E.1 cached coverage and Phase 3E.2 non-cached coverage are recorded by
`tests/ascend/production/run_async_overlap.py`. Its intended matrix covers
normal, expanded, and cached BF16 dispatch plus BF16 combine in synchronous
and asynchronous modes, previous-event ordering, and communication-stream
allocation. Phase 3E.2 promotes the 60 BF16 functional rows. The 60 FP8
functional rows remain deferred because Phase 3E.2 does not add FP8 async
non-cached dispatch.

List cases without importing torch or torch_npu:

```bash
python3 tests/ascend/benchmark/bench_ep.py --list-cases --suite all
python3 tests/ascend/benchmark/bench_ep.py \
  --list-cases --suite performance --format json
python3 tests/ascend/benchmark/bench_ep.py \
  --list-cases --suite functional --format json
```

The inventory has 84 supported rows: 24 synchronous performance rows and 60
BF16 functional rows. The 60 FP8 functional rows remain listed but deferred.

## Ascend environment

The qualified target is Ascend 950, CANN 9.2.0, and AOT target `dav-3510`.
The HCOMM headers and `libhcomm.so` must come from the same package. On the
NPU8P host, use the validated symlink rather than selecting a weekly package:

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
test -n "$ASCEND_HOME_PATH"
DEEP_EP_PLATFORM=ascend python setup.py build_ext --inplace
```

Keep the environment exports and build/run command in the same TaskQueue
shell. The benchmark initializes HCCL, maps `LOCAL_RANK` to the local NPU,
uses `allow_hybrid_mode=False`, `num_sms=1`, and `num_qps=0`, and destroys the
buffer before the process group. Here `num_qps=0` means the CUDA QP tuning
argument is unused; HCOMM still owns the Ascend communication resources.

## TaskQueue acceptance

Run `task-submit --list` first and keep at most one submitted task active. The
current local policy permits only two-device pairs `0,1` and `6,7`; it forbids
`--device auto`, devices 2 through 5, and tasks with more than two devices.

A short two-rank build, correctness, and timing run is:

```bash
task-submit --list
task-submit --device 0,1 --max-time 300 --run '
cd /home/pyptouser/yuqitao/DeepEP-Ascend &&
source /usr/local/Ascend/cann-9.2.0/set_env.sh &&
export HCOMM_ROOT=/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann &&
export PATH="$HCOMM_ROOT/bin:/home/pyptouser/yuqitao/tools/cmake-3.28.4/bin${PATH:+:$PATH}" &&
export LD_LIBRARY_PATH="$HCOMM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" &&
export LIBRARY_PATH="$HCOMM_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}" &&
export CPLUS_INCLUDE_PATH="$HCOMM_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" &&
export CPATH="$HCOMM_ROOT/include${CPATH:+:$CPATH}" &&
export CMAKE_INCLUDE_PATH="$HCOMM_ROOT/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}" &&
export CMAKE_LIBRARY_PATH="$HCOMM_ROOT/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}" &&
export PYTHONPATH="$HCOMM_ROOT/python/site-packages${PYTHONPATH:+:$PYTHONPATH}" &&
source /home/pyptouser/yuqitao/venvs/deepep-ascend-py310/bin/activate &&
DEEP_EP_PLATFORM=ascend python setup.py build_ext --inplace &&
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/ascend/benchmark/bench_ep.py \
  --num-tokens 16 --hidden 128 --num-topk 2 --num-experts 4 \
  --warmups 1 --iterations 1 \
  --output /tmp/ascend-ep2-performance-smoke.json
'
```

Phase 3E.1 acceptance uses the same serialized policy and exact archive, then
runs the bounded event and full matrices separately:

```bash
python3 tests/ascend/production/run_async_overlap.py \
  --suite event --output /tmp/phase3e-event.json
python3 tests/ascend/production/run_async_overlap.py \
  --suite full --output /tmp/phase3e-full.json \
  --trace-dir /tmp/phase3e-traces
python3 tests/ascend/production/run_async_overlap.py \
  --suite uncached --output /tmp/phase3e2-uncached.json \
  --trace-dir /tmp/phase3e2-uncached-traces
```

Every runner case has a 45-second child-process bound. The complete TaskQueue
job remains bounded by `--max-time 300`. The runner atomically rewrites the
JSON report after every completed case, streams captured diagnostics for a
failed child, and stops at the first failure. The report summary distinguishes
selected, executed, failed, and not-run cases so an outer TaskQueue timeout
cannot be mistaken for a completed matrix.

The formal `overlap-vs-serialized` row fixes communication at 256 tokens,
hidden size 4096, and `num_sms=1`. Its compute workload is BF16 NCDHW Conv3D
with input `(1,64,24,96,96)`, weight `(64,64,3,3,3)`, stride `(1,1,1)`, zero
padding, dilation `(1,1,1)`, groups 1, and 256 iterations. It uses three
warmups and seven measured repetitions without dynamic calibration. Acceptance
requires all 21 matrix rows, at least 5% median improvement on both ranks,
`Conv3DV2=KERNEL_AICORE`, `dispatch_kernel=KERNEL_AIVEC`, distinct physical
streams, a positive overlap interval, and auxiliary AIV duration below 1% of
the primary Conv3D span. Auxiliary families remain inventoried and the report
keeps `compute_path_aic_only=false`; the complete path is not claimed as pure
AIC.

Retain the returned task ID and use `task-submit --wait ID`,
`task-submit --status ID`, and `task-submit --log ID`. Resolve a failed or
timed-out task before submitting another. A production-size run uses the same
command with the defaults or these explicit arguments:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/ascend/benchmark/bench_ep.py \
  --num-tokens 4096 --hidden 7168 --num-topk 6 --num-experts 256 \
  --warmups 30 --iterations 30 \
  --output /tmp/ascend-ep2-production.json
```

Four- and eight-rank commands require a different allocation policy or host
authorization. They must not be submitted through the current two-device
profile. Once the matching resource allocation is approved, the benchmark
commands are:

```bash
torchrun --standalone --nproc-per-node=4 \
  tests/ascend/benchmark/bench_ep.py \
  --warmups 1 --iterations 1 --output /tmp/ascend-ep4-smoke.json

torchrun --standalone --nproc-per-node=8 \
  tests/ascend/benchmark/bench_ep.py \
  --warmups 1 --iterations 1 --output /tmp/ascend-ep8-smoke.json
```

These short runs qualify launch and current-supported case coverage only.
They are not full topology performance qualification.

## CUDA parity and manifests

The CUDA profile retains automatic SM/QP selection and records the selected
values. Match `--num-processes`, model dimensions, reduction mode, warmups,
iterations, and case selection to the Ascend run. Generate and retain the
manifest so the exact routing and weights can be used on both machines:

```bash
python3 tests/elastic/test_ep.py \
  --benchmark-profile parity --num-processes 2 \
  --num-tokens 4096 --hidden 7168 --num-topk 6 --num-experts 256 \
  --warmups 30 --iterations 30 \
  --dump-manifest /tmp/ep2-workload.json \
  --benchmark-json /tmp/cuda-ep2.json

torchrun --standalone --nproc-per-node=2 \
  tests/ascend/benchmark/bench_ep.py \
  --workload-manifest /tmp/ep2-workload.json \
  --warmups 30 --iterations 30 \
  --output /tmp/ascend-ep2.json
```

The manifest file must be transferred byte-for-byte when CUDA and Ascend run
on different hosts. The report comparison rejects different fingerprints even
if the visible dimensions happen to match.

## Compare reports

```bash
python3 tests/ascend/benchmark/compare.py \
  /tmp/cuda-ep2.json /tmp/ascend-ep2.json

python3 tests/ascend/benchmark/compare.py \
  /tmp/cuda-ep2.json /tmp/ascend-ep2.json --format json
```

The table reports CUDA and Ascend mean/p50/p95 device latency, logical GB/s,
Ascend-over-CUDA latency ratio, and Ascend-over-CUDA bandwidth ratio.
Comparison is rejected when schema version, formula version, world size,
workload fingerprint, timing counts/aggregation, passed case IDs, operation
IDs, or logical bytes differ. The CUDA and NPU event timer names are expected
to differ and are not a rejection condition.

## JSON interpretation

The default report contains exactly 84 supported cases and 420 operation
records; the 144-row inventory is available separately through
`--list-cases`. Top-level identity and provenance fields include
`schema_version`, `formula_version`, `git_commit`, `platform`, `world_size`,
`workload`, and `workload_fingerprint`. `git_commit` is the repository `HEAD`,
or `unknown` when Git metadata is unavailable. `case_summary` reports `total`,
`pending`, `passed`, and `failed` for cases actually present. Each passed
operation records raw device/wall samples, mean/p50/p95 summaries, logical
bytes, formula version, per-rank summaries, and aggregated logical GB/s. Rank
latency uses the maximum for each sample; logical bytes are summed over ranks.

Do not compare internal kernel-stage profiler times across platforms. The
canonical comparable latency is the end-to-end CUDA/NPU event interval around
the public operation after the same barrier protocol. No Ascend performance
claim is valid until a fresh TaskQueue run has produced a complete report.
