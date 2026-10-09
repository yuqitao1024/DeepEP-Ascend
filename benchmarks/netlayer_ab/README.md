# Netlayer 0/1 A/B benchmark

This is a standalone communication benchmark.  It does not use DeepEP kernels,
PyTorch, NumPy, or Python third-party packages.  Runtime only needs Python 3.8+
from the standard distribution.  Building needs a C++17 compiler, CMake/Ninja
(normally available with the CANN environment), and CANN 9.3.

## What it measures

Every rank allocates one payload, registers it with HCCL, creates one AIV
channel per peer through the selected network layer, and issues one-sided
URMA WRITE requests to all peers.  The default case is EP8 all-to-all:

- payload per peer: 256 MiB by default in `run_ab.sh`
- request size: 16 MiB by default in `run_ab.sh`
- one independent channel per peer
- one AIV producer per rank
- host wall-clock timing around a synchronized kernel
- CQE drain and payload-marker validation

The benchmark intentionally uses independent per-peer queues.  It does not use
the shared-Jetty configuration that requires serial producers.

## Build

Use the CANN package that will run the binary.  For example:

```bash
source /usr/local/Ascend/cann-9.3.0/set_env.sh
cd benchmarks/netlayer_ab
./build_ab.sh
```

`build_ab.sh` builds two scenario directories:

- `build-layer0/`: layer-0/UB_CTP executable and mixed library
- `build-layer1/`: layer-1/UB_MEM executable and mixed library

The build script supports these overrides:

```bash
CANN_HOME=/path/to/cann CXX=/path/to/g++ BUILD_DIR=/tmp/netlayer-ab-build ./build.sh
```

## Run on NPU8P

Netlayer 0 on all eight devices:

```bash
PAYLOAD_BYTES=268435456 CHUNK_BYTES=16777216 \
ITERATIONS=20 WARMUP=5 WORLD_SIZE=8 ./run_ab.sh
```

On a netlayer=1 capable host:

the same command runs both binaries and compares them. On NPU8P, layer 1
currently compiles but channel acquisition fails with HCCL status 9; use a
layer-0 single-scenario run for NPU8P validation.

Useful tuning parameters:

```bash
./run.sh \
  --layer-index 0 \
  --payload-bytes 134217728 \
  --chunk-bytes 8388608 \
  --iterations 20
```

Increase payload and request size first if the measured bandwidth is far below
the physical link expectation.  A request that is too small or too few cannot
fill queue/credit pipelines.

## Output

Each rank prints:

```text
rank=0 layer=0 payload=67108864 chunk=4194304 ... mean_ms=... send_gibps=...
```

## Validated NPU8P layer-0 smoke result

The mixed host/device build passes on NPU8P with CANN 9.3.  The bootstrap race
is fixed by having rank 0 publish a device-context-generated root blob and a
ready marker; the other ranks wait for that marker.

NPU8P layer 0 is full-mesh.  The important rank-graph API detail is that
HcclRankGraphGetLinks may invalidate library-managed results from a prior
query, so the selected layer value must be copied before iterating peers.
After that correction, all eight ranks enumerate seven peers.  HcclBarrier must
receive the benchmark stream on this CANN build.

A validated smoke run used payload 16 MiB, 1 MiB requests, one warmup, and two
measured iterations:

```text
slowest_mean_ms=3.022310
mean_rank_ms=2.731774
send_sum_gibps=321.583691
slowest_rank_gibps=36.189206
peer_counts=7,7,7,7,7,7,7,7
```

This is a smoke result, not the final A/B number.  Use larger payloads and more
iterations for performance conclusions.

- `send_sum_gibps`: sum of all per-rank send-side logical bandwidth.
- `slowest_rank_gibps`: bandwidth attributed to the slowest rank envelope.

For an all-to-all comparison, keep both and compare the same layer under the
same payload/chunk/iteration settings.  Do not compare absolute device cycle
counters across NPUs; the host wall-clock result is the primary metric.

## Portability note

The benchmark contains the CANN 9.3 ABI layout used by the installed
HCOMM runtime.  If another environment uses a different CANN/HCOMM ABI, the
build or kernel may fail; copy the relevant installed headers and update
`netlayer_ab_abi.hpp` before interpreting a failure as a layer performance
difference.
