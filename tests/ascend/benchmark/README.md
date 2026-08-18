# Ascend EPv2 benchmark

This suite compares the synchronous BF16 intersection of
`tests/elastic/test_ep.py` on CUDA and Ascend 950. Both profiles use the same
144-case enumeration, deterministic routing manifest, operation preparation,
correctness references, warmup/sample counts, rank aggregation, and logical
byte formulas.

The exhaustive case table and the statistical definitions are in
[`docs/superpowers/specs/2026-08-18-ascend-epv2-benchmark-parity-design.md`](../../../docs/superpowers/specs/2026-08-18-ascend-epv2-benchmark-parity-design.md).

## Suite classification

| Classification | Cases | Ascend behavior | Reason |
| --- | ---: | --- | --- |
| Current BF16 performance | 12 | Correctness preflight and timing | |
| Deferred FP8 performance | 12 | Listed, not executed | `fp8_runtime_deferred` |
| Functional previous event | 48 | Listed, not executed | `event_chaining_deferred` |
| Functional async without previous event | 48 | Listed, not executed | `async_overlap_deferred` |
| Functional comm allocation only | 24 | Listed, not executed | `comm_stream_allocation_deferred` |
| Total inventory | 144 | 12 current performance, 132 deferred | |

Each supported case checks normal dispatch, expanded dispatch, cached dispatch,
cached expanded dispatch with zero padding, combine, and reduced/expanded
combine. The correctness work is a preflight gate for each performance case,
not a separate functional case. The five operations other than cached expanded
padding are timed, producing 60 performance records per complete run. FP8
remains deferred while its separate implementation is in progress.

List cases without importing torch or torch_npu:

```bash
python3 tests/ascend/benchmark/bench_ep.py --list-cases --suite all
python3 tests/ascend/benchmark/bench_ep.py \
  --list-cases --suite performance --format json
python3 tests/ascend/benchmark/bench_ep.py \
  --list-cases --suite functional --format json
```

The inventory has 24 performance rows: 12 current BF16 and 12 deferred FP8.
The 120 functional rows never appear in a performance report.

## Ascend environment

The qualified target is Ascend 950, CANN 9.2.0, and AOT target `dav-3510`.
The HCOMM headers and `libhcomm.so` must come from the same package. On the
NPU8P host, use the validated symlink rather than selecting a weekly package:

```bash
source /home/pyptouser/yuqitao/venvs/deepep-ascend-py310/bin/activate
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export HCOMM_ROOT=/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann
export PATH="/home/pyptouser/yuqitao/tools/cmake-3.28.4/bin:$HCOMM_ROOT/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$HCOMM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$HCOMM_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="$HCOMM_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CPATH="$HCOMM_ROOT/include${CPATH:+:$CPATH}"
export CMAKE_INCLUDE_PATH="$HCOMM_ROOT/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_LIBRARY_PATH="$HCOMM_ROOT/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
export PYTHONPATH="$HCOMM_ROOT/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
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
task-submit --device 0,1 --max-time 1800 --run '
cd /home/pyptouser/yuqitao/DeepEP-Ascend &&
source /home/pyptouser/yuqitao/venvs/deepep-ascend-py310/bin/activate &&
source /usr/local/Ascend/cann-9.2.0/set_env.sh &&
export HCOMM_ROOT=/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann &&
export PATH="/home/pyptouser/yuqitao/tools/cmake-3.28.4/bin:$HCOMM_ROOT/bin${PATH:+:$PATH}" &&
export LD_LIBRARY_PATH="$HCOMM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" &&
export LIBRARY_PATH="$HCOMM_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}" &&
export CPLUS_INCLUDE_PATH="$HCOMM_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" &&
export CPATH="$HCOMM_ROOT/include${CPATH:+:$CPATH}" &&
export CMAKE_INCLUDE_PATH="$HCOMM_ROOT/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}" &&
export CMAKE_LIBRARY_PATH="$HCOMM_ROOT/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}" &&
export PYTHONPATH="$HCOMM_ROOT/python/site-packages${PYTHONPATH:+:$PYTHONPATH}" &&
DEEP_EP_PLATFORM=ascend python setup.py build_ext --inplace &&
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/ascend/benchmark/bench_ep.py \
  --num-tokens 16 --hidden 128 --num-topk 2 --num-experts 4 \
  --warmups 1 --iterations 1 \
  --output /tmp/ascend-ep2-performance-smoke.json
'
```

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

These short runs qualify launch and current-performance case coverage only.
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
workload fingerprint, passed case IDs, or operation IDs differ.

## JSON interpretation

The default report contains exactly 12 current BF16 performance cases and 60
operation records; the 144-row inventory is available separately through
`--list-cases`. Top-level identity fields include `schema_version`,
`formula_version`, `platform`, `world_size`, `workload`, and
`workload_fingerprint`. `case_summary` reports `total`, `pending`, `passed`,
and `failed` for cases actually present. Each passed operation records raw
device/wall samples, mean/p50/p95 summaries, logical bytes, formula version,
per-rank summaries, and aggregated logical GB/s. Rank latency uses the maximum
for each sample; logical bytes are summed over ranks.

Do not compare internal kernel-stage profiler times across platforms. The
canonical comparable latency is the end-to-end CUDA/NPU event interval around
the public operation after the same barrier protocol. No Ascend performance
claim is valid until a fresh TaskQueue run has produced a complete report.
