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
- dual timing: host wall-clock for the end-to-end diagnostic envelope, and
  msprof kernel duration for the reporting metric
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
rank=0 layer=0 payload=67108864 chunk=4194304 \\
  wall_mean_ms=... wall_send_gibps=...
```

There are two intentional metric classes:

- **Wall-clock (diagnostic)**: launch to stream synchronization on the host. It
  includes launch overhead, kernel execution, URMA WRITE, and CQE drain, but
  excludes the surrounding HCCL barriers, result copy, and validation.
- **Kernel msprof (reporting)**: use the Task Duration of
  netlayer_ab_kernel records in each rank's op_summary CSV. For each rank,
  use the mean kernel duration of the measured iterations, then compute
  `active_peer_count * payload_bytes / mean_kernel_duration`. This matches the
  official DeepEP-Ascend convention: average per-invocation duration first,
  compute bandwidth per rank, then report the mean/min/max/p50/p95 distribution
  across ranks.

Wall bandwidth uses GiB/s (1024^3 bytes per second). Kernel bandwidth uses
decimal GB/s (1e9 bytes per second), matching the official
`bytes / duration` convention.

## Netlayer-0 matrix

`run_layer0_matrix.sh` runs the eight-rank full-mesh matrix with 20 wall-clock
iterations and five msprof iterations per case. It writes one wall result and
one msprof summary per payload/chunk pair under `results-layer0-matrix`.

On NPU8P (CANN 9.3, devices 0-7), the completed 2026-10-14 run produced the
following five-case matrix. The matrix uses the default `per-peer` drain
schedule, so it characterizes the original serial-peer diagnostic path rather
than the hardware's multi-peer overlap ceiling:

| Payload/peer | WQE | Wall slowest rank | Wall mean rank | Kernel slowest rank | Kernel mean rank | Kernel mean GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 8 MiB | 256 KiB | 1.738914 ms | 1.641695 ms | 1.161183 ms | 1.158365 ms | 50.692905 |
| 16 MiB | 1 MiB | 3.063893 ms | 2.899906 ms | 2.283726 ms | 2.278619 ms | 51.540832 |
| 64 MiB | 4 MiB | 10.582948 ms | 10.458882 ms | 9.022279 ms | 9.003761 ms | 52.174513 |
| 128 MiB | 8 MiB | 18.622643 ms | 18.459079 ms | 18.005918 ms | 17.957575 ms | 52.320042 |
| 256 MiB | 16 MiB | 36.509876 ms | 36.416655 ms | 35.973041 ms | 35.893749 ms | 52.350814 |

The kernel bandwidth is close to the observed layer-0 unidirectional ceiling
and does not exceed it. Wall bandwidth is lower because it includes launch and
synchronization overhead; retain it as the diagnostic envelope, not as the
reporting metric.

## Peer-overlap A/B

`run_drain_ab.sh` compares two submission/completion schedules with the same
payload, WQE size, and rank count:

- `per-peer`: post all chunks to one peer, drain that peer, then move to the
  next peer. This is the original diagnostic schedule.
- `post-all`: post all chunks to all peers first, then drain all completion
  queues. This tests whether large writes to independent peer links overlap in
  hardware.

Both modes still use one AIV producer. If the post-all kernel time is close to
the single-peer transfer time rather than seven times it, the payload is
overlapping across peer links.
The script collects both wall-clock and msprof kernel duration. Use the
`drain_mode=... kernel_*` fields to decide whether peer traffic overlaps in
hardware; use wall-clock only as the end-to-end diagnostic envelope.

The 2026-10-14 NPU8P run (CANN 9.3, devices 0-7, 64 MiB per peer, 4 MiB WQEs,
20 wall iterations, five profiled iterations, one profile warmup) gave:

| Drain mode | Slowest-rank kernel mean | Mean-rank kernel mean | Kernel GB/s |
|---|---:|---:|---:|
| per-peer | 9.036023 ms | 9.016229 ms | 52.102356 |
| post-all | 1.440832 ms | 1.436804 ms | 326.949985 |

The 6.3x kernel-time reduction with identical logical traffic shows that the
layer-0 hardware can execute large writes to multiple peer links concurrently.
The wall-clock result shows the same direction but remains diagnostic because
it includes host launch/synchronization and completion-drain overhead. Raw
summaries are stored under `results-drain-ab/`.

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
measured iterations. It predates dual timing and retained only wall-clock
fields; use it for functional validation only. Use larger payloads and more
iterations for performance conclusions.

The reporting fields for a new run are:

- `parse_msprof.py` fields: per-rank bandwidth distribution and maximum
  per-rank mean kernel duration.
- `wall_*`: diagnostic end-to-end fields retained for regression analysis.

`run.py` no longer prints ACL-event kernel fields. On NPU8P/CANN 9.3,
`aclrtEventElapsedTime` returned zero for this mixed host/device launch, so
msprof is the required kernel-duration source.

## Portability note

The benchmark contains the CANN 9.3 ABI layout used by the installed
HCOMM runtime.  If another environment uses a different CANN/HCOMM ABI, the
build or kernel may fail; copy the relevant installed headers and update
`netlayer_ab_abi.hpp` before interpreting a failure as a layer performance
difference.
