import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "tests/benchmark/run_h800_representative.sh"


def write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_h800_runner_is_valid_bash():
    subprocess.run(["bash", "-n", RUNNER], check=True)


def test_h800_runner_help_documents_environment_overrides():
    completed = subprocess.run(
        ["bash", RUNNER, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run the H800 representative benchmark" in completed.stdout
    assert "CUDA_HOME" in completed.stdout
    assert "MASTER_PORT" in completed.stdout
    assert "DEEPEP_RESULT_ROOT" in completed.stdout
    assert "single-node NVLink" in completed.stdout


def test_h800_runner_reports_the_cuda_package_when_cuobjdump_is_missing(
    tmp_path,
):
    cuda_bin = tmp_path / "cuda/bin"
    cuda_bin.mkdir(parents=True)
    nvcc = cuda_bin / "nvcc"
    nvcc.write_text(
        "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.0'\n",
        encoding="utf-8",
    )
    nvcc.chmod(0o755)

    command_bin = tmp_path / "commands"
    command_bin.mkdir()
    nvidia_smi = command_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(cuda_bin.parent)
    environment["PATH"] = f"{command_bin}{os.pathsep}{environment['PATH']}"

    completed = subprocess.run(
        ["bash", RUNNER],
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 1
    assert f"{cuda_bin}/cuobjdump is missing" in completed.stderr
    assert "apt install cuda-cuobjdump-13-0" in completed.stderr


def test_h800_runner_trusts_manual_topology_and_uses_non_gin_path(tmp_path):
    command_bin = tmp_path / "commands"
    cuda_bin = tmp_path / "cuda/bin"
    command_bin.mkdir()
    cuda_bin.mkdir(parents=True)
    state_file = tmp_path / "runner-state"
    result_root = tmp_path / "results"

    write_executable(
        command_bin / "python3",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == "-" ]]; then
  body="$(cat)"
  if [[ ${body} == *"print(torch.version.cuda.split('.')[0])"* ]]; then
    echo 13
  fi
  exit 0
fi
if [[ ${1:-} == "tests/benchmark/check_cuda_nvlink.py" ]]; then
  echo "unexpected automatic topology check" >&2
  exit 77
fi
if [[ ${1:-} == "tests/benchmark/check_cuda_gin.py" ]]; then
  [[ " $* " == *" --single-node-nvlink "* ]]
  echo "preflight EP_DISABLE_GIN=${EP_DISABLE_GIN}" >> "${H800_RUNNER_TEST_STATE}"
  exit 0
fi
if [[ ${1:-} == "tests/benchmark/run_ep.py" ]]; then
  while (( $# )); do
    if [[ $1 == "--output-dir" ]]; then
      output_dir=$2
      break
    fi
    shift
  done
  mkdir -p "${output_dir}"
  echo '{}' > "${output_dir}/workload.json"
  echo '{}' > "${output_dir}/benchmark.json"
  echo report > "${output_dir}/benchmark.md"
  echo log > "${output_dir}/run.log"
  echo "benchmark EP_DISABLE_GIN=${EP_DISABLE_GIN}" >> "${H800_RUNNER_TEST_STATE}"
  exit 0
fi
exit 0
""",
    )
    write_executable(
        command_bin / "nvidia-smi",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    write_executable(
        command_bin / "git",
        """#!/usr/bin/env bash
if [[ ${1:-} == "rev-parse" ]]; then
  echo 0123456789abcdef
fi
exit 0
""",
    )
    write_executable(
        cuda_bin / "nvcc",
        "#!/usr/bin/env bash\necho 'Cuda compilation tools, release 13.0'\n",
    )
    write_executable(
        cuda_bin / "cuobjdump",
        "#!/usr/bin/env bash\necho 'cuobjdump 13.0'\n",
    )

    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(cuda_bin.parent)
    environment["DEEPEP_RESULT_ROOT"] = str(result_root)
    environment["H800_RUNNER_TEST_STATE"] = str(state_file)
    environment["PATH"] = f"{command_bin}{os.pathsep}{environment['PATH']}"

    subprocess.run(["bash", RUNNER], check=True, env=environment)

    assert state_file.read_text(encoding="utf-8").splitlines() == [
        "preflight EP_DISABLE_GIN=1",
        "benchmark EP_DISABLE_GIN=1",
    ]
