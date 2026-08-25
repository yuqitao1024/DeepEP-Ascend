# Ascend HCCS transport benchmark

This benchmark separates the communication path used by DeepEP into three
layers:

1. one-way, two-rank HCCS/HCOMM `put` throughput;
2. concurrent eight-rank ring and all-to-all throughput; and
3. the DeepEP staged transport data plane with representative EP8 payloads,
   excluding token grouping, offset calculation, production SQE assembly, and
   consumer copies.

The device benchmark uses the production `DeviceTransportFacade`, command
queue, AICore service, UBC/CTP team, symmetric window, `put`, `flush`, and CQ
completion path. It is therefore a measurement of the DeepEP transport data
plane, not a host-side `HcclSend`/`HcclRecv` benchmark or a raw wire test.
After all iterations, every byte in the final received payload is verified
outside the timed interval. Device diagnostics and CQ completion are checked
on every iteration.

## What is timed

Each measured iteration is aligned with an HCCL barrier. The timed device
region starts after the barrier and contains:

```text
SIMT producer: enqueue one put per nonzero remote peer, then one final flush
        |
        v
AICore service: consume commands -> HCOMM put/flush -> wait for CQ completion
```

The report records `producer`, `service`, and `total` cycles from the Ascend
950 1 GHz `GetSystemCycle()` counter. Host-to-device copies of the peer-size
table, HCCL barriers, stream synchronization, buffer reset, and payload
verification are outside this interval.

`Producer` is the AIV-side `asc_vf_call` issue window. The SIMT function may
continue asynchronously after that call returns, so its tail can overlap the
service interval. The producer and service columns are useful for detecting a
change in this benchmark's pipeline, but they are not independent standalone
kernel timings. `Total` is the acceptance latency.

This is deliberately a batched-put transport probe. Production direct
dispatch flushes and publishes control after each peer payload, whereas this
benchmark issues all nonzero peer puts followed by one flush. The probe is an
upper data-plane reference and does not reproduce the production command
schedule.

For iteration `i`, latency is the slowest participating rank:

```text
t_i = max_r(t_i,r)
```

The reported mean is the arithmetic mean of 50 measured `t_i` values. P50 and
P95 use the same rank-max samples. Logical bandwidth is:

```text
B_logical = sum(bytes sent by all active ranks) / mean(t_i) / 1e9
```

Consequently, an eight-rank aggregate bandwidth can be greater than the
bandwidth of one physical link. It must not be read as per-device or per-link
bandwidth.

## Build

Run from the repository root after selecting the CANN and HCOMM installations:

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export HCOMM_ROOT=/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann
export PATH="$HCOMM_ROOT/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$HCOMM_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$HCOMM_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="$HCOMM_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CPATH="$HCOMM_ROOT/include${CPATH:+:$CPATH}"
export CMAKE_INCLUDE_PATH="$HCOMM_ROOT/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_LIBRARY_PATH="$HCOMM_ROOT/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
export PYTHONPATH="$HCOMM_ROOT/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
source /home/pyptouser/yuqitao/venvs/deepep-ascend-py310/bin/activate

cmake -S tests/ascend/hccs_benchmark \
  -B build/hccs-benchmark \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/hccs-benchmark -j2
```

The output is `build/hccs-benchmark/hccs_benchmark.so`.

## Run

Choose unused master ports if these examples conflict with another distributed
job. The commands below use 20 warmups and 50 measured iterations.

Two-rank P2P, in both directions:

```bash
mkdir -p results/hccs-benchmark

python -m torch.distributed.run --nproc_per_node=2 --master_port=29742 \
  tests/ascend/hccs_benchmark/benchmark.py \
  --mode p2p --p2p-sender 0 \
  --runner build/hccs-benchmark/hccs_benchmark.so \
  --output results/hccs-benchmark/p2p-0-to-1.json \
  --sizes-mib 1,4,16,64 --warmups 20 --iterations 50

python -m torch.distributed.run --nproc_per_node=2 --master_port=29743 \
  tests/ascend/hccs_benchmark/benchmark.py \
  --mode p2p --p2p-sender 1 \
  --runner build/hccs-benchmark/hccs_benchmark.so \
  --output results/hccs-benchmark/p2p-1-to-0.json \
  --sizes-mib 1,4,16,64 --warmups 20 --iterations 50
```

Eight-rank ring, all-to-all, and representative transport-only:

```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29744 \
  tests/ascend/hccs_benchmark/benchmark.py \
  --mode ring \
  --runner build/hccs-benchmark/hccs_benchmark.so \
  --output results/hccs-benchmark/ring-ep8.json \
  --sizes-mib 4,16,32 --warmups 20 --iterations 50

python -m torch.distributed.run --nproc_per_node=8 --master_port=29745 \
  tests/ascend/hccs_benchmark/benchmark.py \
  --mode all-to-all \
  --runner build/hccs-benchmark/hccs_benchmark.so \
  --output results/hccs-benchmark/all-to-all-ep8.json \
  --sizes-mib 4,16,32 --warmups 20 --iterations 50

python -m torch.distributed.run --nproc_per_node=8 --master_port=29746 \
  tests/ascend/hccs_benchmark/benchmark.py \
  --mode transport-only \
  --runner build/hccs-benchmark/hccs_benchmark.so \
  --output results/hccs-benchmark/transport-only-ep8.json \
  --warmups 20 --iterations 50
```

## Ascend 950 results

These measurements were collected on eight `Ascend950PR_9599` devices with
CANN 9.2.0 and the pinned `hcomm-deepep-current` package. The benchmark was an
uncommitted addition on base commit
`de37ed747febb9ecacbc26baf4598e7852dc7c8f`; exact source hashes are recorded
below. Ring and all-to-all ran in
TaskQueue task `task_20260825_162727_1889313041`. The production-layout
transport-only rerun was `task_20260825_170857_118081824220`. Both completed
with exit code zero. Results are retained on NPU8P under:

```text
/home/pyptouser/yuqitao/deepep-results/hccs-benchmark-de37ed7
```

All latency and phase columns below are in microseconds. `GB/s` is decimal
logical bandwidth.

### One-way P2P

| Direction | Payload | Mean | P50 | P95 | GB/s | Producer | Service |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 -> 1 | 1 MiB | 39.153 | 39.130 | 39.351 | 26.781 | 0.104 | 39.049 |
| 0 -> 1 | 4 MiB | 98.441 | 98.385 | 98.666 | 42.607 | 0.104 | 98.337 |
| 0 -> 1 | 16 MiB | 335.393 | 335.297 | 336.199 | 50.023 | 0.105 | 335.288 |
| 0 -> 1 | 64 MiB | 1283.293 | 1282.987 | 1284.636 | 52.294 | 0.104 | 1283.188 |
| 1 -> 0 | 1 MiB | 39.725 | 39.720 | 39.867 | 26.396 | 0.110 | 39.626 |
| 1 -> 0 | 4 MiB | 98.964 | 98.920 | 99.124 | 42.382 | 0.110 | 98.864 |
| 1 -> 0 | 16 MiB | 335.948 | 335.823 | 336.774 | 49.940 | 0.110 | 335.848 |
| 1 -> 0 | 64 MiB | 1283.911 | 1283.563 | 1285.206 | 52.269 | 0.110 | 1283.811 |

The two directions converge at large payloads. Their 64 MiB average is
`52.282 GB/s`, and the directional difference is below `0.1%`.

### Eight-rank ring

Each rank sends one payload to `(rank + 1) mod 8`. The byte numerator is
`8 * payload`.

| Payload per sender | Aggregate bytes | Mean | P50 | P95 | Aggregate GB/s | Producer | Service |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 MiB | 32 MiB | 102.123 | 101.955 | 103.143 | 328.567 | 0.021 | 102.103 |
| 16 MiB | 128 MiB | 339.149 | 338.869 | 340.200 | 395.749 | 0.021 | 339.128 |
| 32 MiB | 256 MiB | 655.439 | 655.338 | 656.623 | 409.550 | 0.021 | 655.419 |

At 32 MiB per sender, the aggregate result is `409.550 GB/s`, or
`51.194 GB/s` per active sender. That is `97.92%` of the two-direction 64 MiB
P2P average.

### Eight-rank all-to-all

Each rank sends one equally sized payload to each of its seven remote peers.
The byte numerator is `8 * 7 * payload`.

| Payload per peer | Aggregate bytes | Mean | P50 | P95 | Aggregate GB/s | Producer | Service |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 MiB | 224 MiB | 165.117 | 165.051 | 165.586 | 1422.509 | 0.021 | 165.097 |
| 16 MiB | 896 MiB | 415.615 | 415.517 | 416.540 | 2260.564 | 0.021 | 415.594 |
| 32 MiB | 1792 MiB | 749.468 | 749.241 | 751.158 | 2507.178 | 0.021 | 749.447 |

An independent-link extrapolation from the large-message P2P average would be
`8 * 7 * 52.282 = 2927.775 GB/s`. The measured 32 MiB all-to-all reaches
`85.63%` of that reference. This percentage is a topology-contention
indicator, not a claimed physical hardware efficiency.

### Representative transport-only

This mode reuses the deterministic representative workload used by the full
EP benchmark:

| Setting | Value |
| --- | ---: |
| World size | 8 |
| Tokens per rank | 8192 maximum |
| Hidden width | 7168 |
| Top-k | 8 |
| Experts | 256 |
| Dispatch representation | FP8 |
| Production dispatch record stride | 7552 bytes |
| Workload fingerprint | `d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00` |

The benchmark derives the 7552-byte stride by calling the production
`build_core_tiling()` implementation. It includes hidden data, scale factors,
top-k indices and weights, source metadata, and production 32-byte alignment.
A nonuniform remote payload is then calculated for every source and
destination rank from the routing manifest. Local-rank traffic is omitted.

| Aggregate bytes | Mean | P50 | P95 | Aggregate GB/s | Producer | Service |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2290.272 MB | 903.444 | 902.592 | 906.702 | 2535.046 | 0.021 | 903.424 |

The nonuniform representative payload is `1.11%` faster in logical bandwidth
than the 32 MiB/peer all-to-all point. The two byte volumes are not identical,
but the result shows no material penalty from representative rank imbalance at
this message size.

## Interpretation for DeepEP dispatch

The representative full dispatch measurement used during this investigation
was `37.305 ms` and `208.715 logical GB/s`. Transport-only takes `0.903 ms`,
or `2.42%` of that latency. This does **not** mean that HCOMM accounts for
exactly 2.42% of production dispatch, for three reasons:

1. transport-only moves `2.290 GB` of remote, deduplicated payload, whereas the
   full dispatch logical-byte formula counts `7.786 GB`, including traffic
   categories that this probe deliberately omits;
2. transport-only issues at most seven large contiguous puts per rank, while
   production dispatch performs grouping, offset and metadata work and may
   fragment transport differently; and
3. the probe excludes scale movement, token packing, local-rank work, consumer
   copies, production SQE assembly, and synchronization between those stages.

The useful conclusion is narrower: large contiguous HCOMM `put` bandwidth and
eight-rank concurrency are already far above the end-to-end dispatch result.
The next high-priority investigation should align production dispatch with the
same remote-byte matrix and measure, separately, token grouping and offsets,
packing and scale movement, command construction, HCOMM service time, and
consumer copy. Link-level HCCS tuning should not be the first optimization
target unless that aligned profile shows production service time well above
this transport-only baseline.

### Bottleneck conclusion

The measurements separate several layers that are easy to conflate:

| Layer | Evidence | Conclusion |
| --- | --- | --- |
| Large-message HCCS link | P2P reaches `52.282 GB/s`; ring reaches `51.194 GB/s` per active sender, or `97.92%` of P2P | The one-peer path remains stable under eight-rank concurrency. |
| Eight-rank topology | The 32 MiB/peer all-to-all reaches `85.63%` of an independent-link extrapolation | Topology and concurrent traffic cost about `14.37%` against that reference, but still deliver `2507.178 GB/s` aggregate. |
| Representative transport data plane | The nonuniform workload reaches `2535.046 GB/s`, within `1.11%` of the uniform all-to-all point | Representative rank imbalance is not a material large-message penalty. |
| Full dispatch software path | Full dispatch takes `37.305 ms`, compared with `0.903 ms` for the narrower transport-only probe | Packing, command scheduling, consumer work, synchronization, and other omitted production stages dominate the gap. |

The last row is not a claim that HCOMM consumes only `2.42%` of dispatch time.
The byte formulas and command schedules differ, as described above. It does
show that raw link tuning is unlikely to produce the largest end-to-end gain.

The retained P3 representative profile gives a more specific view of the
production path. Its ordinary dispatch result was `36.176 ms`, with the
following independently aggregated max-rank phase times:

| Production phase | Time | Largest measured subphases |
| --- | ---: | --- |
| Producer | `4.470 ms` | record packing `3.860 ms`; rank prefix `0.575 ms` |
| Network | `6.748 ms` | service submit `5.734 ms`; CQ wait `0.851 ms`; publication `0.163 ms` |
| Consumer | `8.787 ms` | output copy `4.857 ms`; expert prefix `2.788 ms`; expert count `0.339 ms` |

The phase maxima may come from different ranks, so they must not be summed and
treated as an end-to-end trace. They are still useful for ranking work within
each phase:

- record packing accounts for most of the measured producer time;
- service submission, which includes command validation, address resolution,
  WQE/SQE construction, and request posting, is much larger than CQ polling;
- output copy and expert prefix account for most of consumer compute; and
- final SQ/CQ depths are `0/0`, with high-water marks of only `4/4` against a
  command capacity of 36, so queue saturation does not justify adding another
  transport channel.

The same P3 profile shows a different balance for combine. Combine takes
`141.098 ms` with `47.147 ms` producer, `60.302 ms` network, and `18.424 ms`
consumer phase maxima. Reduced combine takes `168.954 ms`, with `74.430 ms`
producer, `62.896 ms` network, and `18.419 ms` consumer phase maxima. Their
producer record spans are `45.336 ms` and `72.613 ms`, respectively. The
dispatch transport-only probe does not isolate combine traffic, but the P3
figures make combine record movement and production transport scheduling the
larger remaining targets.

### Optimization order implied by the measurements

1. Reproduce the transport-only remote-byte matrix inside production dispatch,
   then separate command parsing, WQE/SQE construction, SQ posting, HCOMM
   service, and CQ wait. This identifies how much of the `5.734 ms` service
   submit span comes from control work rather than payload movement.
2. Compare the production schedule of 30 commands and seven payload puts per
   rank with the probe's seven contiguous puts followed by one final flush.
   Test command batching and flush placement independently so ordering changes
   remain attributable and reviewable.
3. Reduce record staging and output copies. The main candidates are fusing
   packing with chunk publication, allowing transport to consume completed
   chunks immediately, and writing received records closer to their final
   expert-major layout.
4. Parallelize or reuse rank and expert prefix results. The measured
   `producer_prefix` and `epilogue_expert_prefix` spans are large enough to
   optimize independently of link throughput.
5. Apply producer/network/consumer overlap after shortening the serial work in
   each phase. Do not add channels unless new SQ/CQ telemetry shows sustained
   occupancy near capacity.

The detailed production phase definitions and aggregation caveats are in
[`epv2-ascend-p3-overlap-optimization.md`](../../../docs/ascend-design/epv2-ascend-p3-overlap-optimization.md).
The narrower comparison with external expert-count, prefix, scale-copy, and
SQE-construction measurements is in
[`epv2-ascend-stage-timing-analysis.md`](../../../docs/ascend-design/epv2-ascend-stage-timing-analysis.md).

## Result integrity

| Artifact | SHA-256 |
| --- | --- |
| `p2p-0-to-1.json` | `15d5dadca5557f5d1c87abb1f5e1289bcbc63dac02aafaf182f37c476423d981` |
| `p2p-1-to-0.json` | `65b34a525d425852b15924939a8bb6e34af38a1c51f9b40887f1b07cfb2a457a` |
| `ring-ep8.json` | `e553e543ae817d27fb962544e1b2cf755e8d0b1b5de041ad1488eda670a6827a` |
| `all-to-all-ep8.json` | `365d443051cd12457a52582367846c99dba678cfa93751b01472863eca1e5ab7` |
| `transport-only-ep8.json` | `fed68582520329a4d68e5f30c266e2d874dc2ed7dfbf5ff4fa8b320a16e66759` |

| Benchmark source | SHA-256 |
| --- | --- |
| `benchmark.py` | `286416149e58694800eb7937f5b7d66ef462411422c1356889c4b88c4fbff816` |
| `hccs_benchmark.asc` | `f413b6d09c057008885eb8ae0735e5435956074d7fec084cbdcd9a8110738b6a` |
| `hccs_benchmark.hpp` | `1130d512c2484b6d830f412563b30162d02e5c8e25baf8a50a1f5a292afecf42` |
| `hccs_benchmark_main.cpp` | `d52480ab4fdaf0e5b28e46f730886bf26ba22ac08851b0638475480bd5f9fb6c` |
| `CMakeLists.txt` | `5476d8f31b944711e30996ca9e8c15cb59fc18519ff43f1efedbacab0d23ec37` |
