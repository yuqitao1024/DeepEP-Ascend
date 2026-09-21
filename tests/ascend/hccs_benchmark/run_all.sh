#!/usr/bin/env bash

set -euo pipefail

# Rebuild and run the HCCS transport-only reference matrix. The caller is
# responsible for sourcing the CANN/HCOMM environment and activating a
# Torch-NPU-capable Python environment before invoking this script.

ROOT_DIR="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
BUILD_DIR="${HCCS_BENCHMARK_BUILD_DIR:-${ROOT_DIR}/build/hccs-benchmark}"
OUTPUT_DIR="${HCCS_BENCHMARK_OUTPUT_DIR:-${ROOT_DIR}/results/hccs-benchmark}"
RUNNER="${BUILD_DIR}/hccs_benchmark.so"
MASTER_PORT_BASE="${HCCS_BENCHMARK_MASTER_PORT_BASE:-29742}"
WARMUPS="${HCCS_BENCHMARK_WARMUPS:-20}"
ITERATIONS="${HCCS_BENCHMARK_ITERATIONS:-50}"

mkdir -p "${OUTPUT_DIR}"

cmake -S "${ROOT_DIR}/tests/ascend/hccs_benchmark" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${HCCS_BENCHMARK_BUILD_JOBS:-8}"

python -m torch.distributed.run \
  --nproc_per_node=2 --master_port="${MASTER_PORT_BASE}" \
  "${ROOT_DIR}/tests/ascend/hccs_benchmark/benchmark.py" \
  --mode p2p --p2p-sender 0 --runner "${RUNNER}" \
  --output "${OUTPUT_DIR}/p2p-0-to-1.json" \
  --sizes-mib 1,4,16,64 --warmups "${WARMUPS}" --iterations "${ITERATIONS}"

python -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$((MASTER_PORT_BASE + 1))" \
  "${ROOT_DIR}/tests/ascend/hccs_benchmark/benchmark.py" \
  --mode p2p --p2p-sender 1 --runner "${RUNNER}" \
  --output "${OUTPUT_DIR}/p2p-1-to-0.json" \
  --sizes-mib 1,4,16,64 --warmups "${WARMUPS}" --iterations "${ITERATIONS}"

python -m torch.distributed.run \
  --nproc_per_node=8 --master_port="$((MASTER_PORT_BASE + 2))" \
  "${ROOT_DIR}/tests/ascend/hccs_benchmark/benchmark.py" \
  --mode all-to-all --runner "${RUNNER}" \
  --output "${OUTPUT_DIR}/all-to-all-ep8.json" \
  --sizes-mib 4,16,32 --warmups "${WARMUPS}" --iterations "${ITERATIONS}"

python -m torch.distributed.run \
  --nproc_per_node=8 --master_port="$((MASTER_PORT_BASE + 3))" \
  "${ROOT_DIR}/tests/ascend/hccs_benchmark/benchmark.py" \
  --mode transport-only --runner "${RUNNER}" \
  --output "${OUTPUT_DIR}/transport-only-ep8.json" \
  --warmups "${WARMUPS}" --iterations "${ITERATIONS}"

echo "HCCS benchmark artifacts:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%p\n' | sort
