#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the H800 representative benchmark from installation through report checks.
This runner requires a single-node NVLink clique and does not require Gin.

Usage:
  bash tests/benchmark/run_h800_representative.sh

Environment overrides:
  CUDA_HOME           CUDA toolkit root (default: /usr/local/cuda)
  CUDA_VISIBLE_DEVICES
                      Eight visible H800 device IDs (default: 0,1,2,3,4,5,6,7)
  MASTER_PORT          Base port for the Gin preflight (default: 8361)
  DEEPEP_RESULT_ROOT   Result parent directory (default: ../deepep-results)
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if (( $# != 0 )); then
  echo "ERROR: unknown argument: $1" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEEPEP_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd -P)"
test -f "${DEEPEP_ROOT}/tests/benchmark/run_ep.py" || {
  echo "ERROR: cannot locate the DeepEP-Ascend repository root" >&2
  exit 1
}
cd "${DEEPEP_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=9.0
export DEEP_EP_PLATFORM=cuda
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-8361}"
export EP_DISABLE_GIN=1
export EP_SUPPRESS_NCCL_CHECK=0

command -v python3 >/dev/null || {
  echo "ERROR: python3 is not available" >&2
  exit 1
}
command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi is not available" >&2
  exit 1
}
test -x "${CUDA_HOME}/bin/nvcc" || {
  echo "ERROR: ${CUDA_HOME}/bin/nvcc is missing; set CUDA_HOME correctly" >&2
  exit 1
}
if ! test -x "${CUDA_HOME}/bin/cuobjdump"; then
  CUDA_RELEASE="$("${CUDA_HOME}/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n 1)"
  echo "ERROR: ${CUDA_HOME}/bin/cuobjdump is missing or not executable" >&2
  if [[ -n "${CUDA_RELEASE}" ]]; then
    CUDA_CUOBJDUMP_PACKAGE="cuda-cuobjdump-${CUDA_RELEASE/./-}"
    echo "Install it with: apt install ${CUDA_CUOBJDUMP_PACKAGE}" >&2
    echo "Or on RPM systems: dnf install ${CUDA_CUOBJDUMP_PACKAGE}" >&2
  else
    echo "Install the cuobjdump component matching the CUDA toolkit." >&2
  fi
  exit 1
fi

python3 -m pip install --upgrade setuptools wheel ninja packaging

DEEPEP_CUDA_MAJOR="$(python3 - <<'PY'
import torch
from packaging.version import Version

if Version(torch.__version__.split('+')[0]) < Version('2.10.0'):
    raise SystemExit(f'ERROR: PyTorch must be >= 2.10, found {torch.__version__}')
if not torch.cuda.is_available():
    raise SystemExit('ERROR: PyTorch cannot use CUDA')
if torch.version.cuda is None:
    raise SystemExit('ERROR: the installed PyTorch is not a CUDA build')
print(torch.version.cuda.split('.')[0])
PY
)"

case "${DEEPEP_CUDA_MAJOR}" in
  12|13) ;;
  *)
    echo "ERROR: only CUDA 12 or 13 is supported; PyTorch CUDA major=${DEEPEP_CUDA_MAJOR}" >&2
    exit 1
    ;;
esac

python3 - <<'PY'
import os
import re
import subprocess

import torch
from packaging.version import Version

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
if count != 8:
    raise SystemExit(f'ERROR: exactly 8 visible GPUs are required, found {count}')
bad = [f'{i}:{name}' for i, name in enumerate(names) if 'H800' not in name.upper()]
if bad:
    raise SystemExit('ERROR: non-H800 devices found: ' + ', '.join(bad))

nvcc = os.path.join(os.environ['CUDA_HOME'], 'bin', 'nvcc')
nvcc_output = subprocess.check_output([nvcc, '--version'], text=True)
match = re.search(r'release\s+([0-9]+\.[0-9]+)', nvcc_output)
if match is None:
    raise SystemExit('ERROR: cannot parse the nvcc version')
nvcc_version = Version(match.group(1))
if nvcc_version < Version('12.3'):
    raise SystemExit(f'ERROR: CUDA toolkit must be >= 12.3, found {nvcc_version}')
if str(nvcc_version.major) != torch.version.cuda.split('.')[0]:
    raise SystemExit(
        f'ERROR: nvcc major={nvcc_version.major} does not match '
        f'PyTorch CUDA={torch.version.cuda}'
    )
print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA toolkit:', nvcc_version)
print('cuobjdump:', os.path.join(os.environ['CUDA_HOME'], 'bin', 'cuobjdump'))
print('GPUs:', names)
PY

python3 tests/benchmark/check_cuda_nvlink.py --expected-gpus 8

"${CUDA_HOME}/bin/nvcc" --version
"${CUDA_HOME}/bin/cuobjdump" --version

python3 -m pip install \
  "nvidia-nccl-cu${DEEPEP_CUDA_MAJOR}==2.30.4" --no-deps
python3 -m pip install \
  "nvidia-nvshmem-cu${DEEPEP_CUDA_MAJOR}==3.3.20" --no-deps

git submodule update --init third-party/fmt

DEEP_EP_PLATFORM=cuda TORCH_CUDA_ARCH_LIST=9.0 \
  python3 setup.py build_ext --inplace --force

python3 - <<'PY'
import torch
import deep_ep
from deep_ep import _C

platform = _C.get_platform()
if platform != 'cuda':
    raise SystemExit(f'ERROR: extension platform is {platform!r}, not cuda')
print('DeepEP:', deep_ep.__version__)
print('DeepEP platform:', platform)
print('Device 0:', torch.cuda.get_device_name(0))
PY

DEEPEP_RESULT_ROOT="${DEEPEP_RESULT_ROOT:-$(dirname "${DEEPEP_ROOT}")/deepep-results}"
DEEPEP_RESULT_DIR="${DEEPEP_RESULT_ROOT}/h800-representative-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${DEEPEP_RESULT_DIR}"

python3 tests/benchmark/check_cuda_gin.py \
  --log-dir "${DEEPEP_RESULT_DIR}/gin-preflight" \
  --master-port "${MASTER_PORT}" \
  --single-node-nvlink

echo "Git commit: $(git rev-parse HEAD)"
echo "Result dir: ${DEEPEP_RESULT_DIR}"

python3 tests/benchmark/run_ep.py \
  --backend cuda \
  --profile representative \
  --output-dir "${DEEPEP_RESULT_DIR}"

test -s "${DEEPEP_RESULT_DIR}/workload.json"
test -s "${DEEPEP_RESULT_DIR}/benchmark.json"
test -s "${DEEPEP_RESULT_DIR}/benchmark.md"
test -s "${DEEPEP_RESULT_DIR}/run.log"

python3 - "${DEEPEP_RESULT_DIR}/benchmark.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding='utf-8'))
if report['case_summary'] != {'total': 1, 'pending': 0, 'passed': 1, 'failed': 0}:
    raise SystemExit(f"ERROR: case failed: {report['case_summary']}")
if report['failures']:
    raise SystemExit(f"ERROR: benchmark failures: {report['failures']}")

print('\nH800 representative result')
print(f"device: {report['device']['name']}")
print(f"fingerprint: {report['workload_fingerprint']}")
print(f"{'operation':<22} {'mean_us':>12} {'p50_us':>12} {'p95_us':>12} {'GB/s':>12}")
for op in report['cases'][0]['operations']:
    timing = op['device_seconds']
    print(
        f"{op['operation_id']:<22} "
        f"{timing['mean'] * 1e6:>12.3f} "
        f"{timing['p50'] * 1e6:>12.3f} "
        f"{timing['p95'] * 1e6:>12.3f} "
        f"{op['logical_gbps']:>12.3f}"
    )
PY

sha256sum \
  "${DEEPEP_RESULT_DIR}/workload.json" \
  "${DEEPEP_RESULT_DIR}/benchmark.json"

echo
echo "SUCCESS: ${DEEPEP_RESULT_DIR}"
echo "Readable report: ${DEEPEP_RESULT_DIR}/benchmark.md"
