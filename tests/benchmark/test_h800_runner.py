import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "tests/benchmark/run_h800_representative.sh"


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
