"""Pinned copy completion and buffer lifetime contract without a device."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def test_small_host_transfer_completion_and_failure_lifetime():
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / 'small_host_transfer_probe'
        subprocess.run([
            'c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread',
            '-I', str(ROOT), str(ROOT / 'tests/ascend/small_host_transfer_probe.cpp'),
            '-o', str(binary)], check=True, capture_output=True, text=True)
        subprocess.run([str(binary)], check=True, capture_output=True, text=True)
