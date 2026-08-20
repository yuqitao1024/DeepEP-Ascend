import ast
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ascend.benchmark.runtime import AscendRuntime
from tests.elastic.test_ep import (
    build_parser,
    parity_buffer_settings,
    selected_parity_case_ids,
    test_loop as _test_loop,
)
from tests.utils.ep_benchmark_manifest import WorkloadSpec, build_manifest


ROOT = Path(__file__).resolve().parents[2]
CUDA_BENCH = ROOT / "tests/elastic/test_ep.py"
ASCEND_RUNTIME = ROOT / "tests/ascend/benchmark/runtime.py"


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_cuda_parser_keeps_defaults_and_adds_opt_in_parity():
    parser = build_parser()

    defaults = parser.parse_args([])
    parity = parser.parse_args(["--benchmark-profile", "parity"])

    assert defaults.num_processes == 8
    assert defaults.benchmark_profile == "upstream"
    assert defaults.warmups == 30
    assert defaults.iterations == 30
    assert parity.benchmark_profile == "parity"


def test_cuda_and_ascend_import_same_shared_core():
    assert "tests.utils.ep_benchmark_core" in _imported_modules(CUDA_BENCH)
    assert "tests.utils.ep_benchmark_core" in _imported_modules(ASCEND_RUNTIME)


def test_cuda_help_is_host_only_when_run_as_a_script():
    result = subprocess.run(
        [sys.executable, str(CUDA_BENCH), "--help"],
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 0, result.stderr
    assert "--benchmark-profile {upstream,parity}" in result.stdout


@pytest.mark.parametrize(
    "case_id",
    (
        "ep-fp8-align128-bias0-hcopy1-prev1-async0-alloc1",
        "ep-fp8-align128-bias0-hcopy1-prev0-async1-alloc0",
        "ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc1",
    ),
)
def test_cuda_parity_rejects_unsupported_case_before_torch_import(case_id):
    result = subprocess.run(
        [
            sys.executable,
            str(CUDA_BENCH),
            "--benchmark-profile",
            "parity",
            "--cases",
            case_id,
        ],
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 2
    assert "not a common supported case" in result.stderr
    assert "No module named 'torch'" not in result.stderr


def test_parity_case_selection_defaults_to_common_supported_intersection():
    selected = selected_parity_case_ids("")

    assert len(selected) == 84
    assert sum("-bf16-" in case_id for case_id in selected) == 72
    assert sum("-fp8-" in case_id for case_id in selected) == 12
    assert sum(
        "-prev0-async0-alloc0" in case_id for case_id in selected
    ) == 24
    assert any(
        case_id ==
        "ep-bf16-align128-bias0-hcopy1-prev1-async1-alloc1"
        for case_id in selected
    )
    assert all(
        "-prev0-async0-alloc0" in case_id
        for case_id in selected
        if "-fp8-" in case_id
    )

    with pytest.raises(ValueError, match="not a common supported case"):
        selected_parity_case_ids(
            "ep-fp8-align128-bias0-hcopy1-prev1-async0-alloc1"
        )


def test_shared_runtime_accepts_cuda_selected_sm_and_qp_counts():
    class FakeDist:
        @staticmethod
        def get_rank(_group):
            return 0

        @staticmethod
        def get_world_size(_group):
            return 2

    runtime = AscendRuntime(
        torch_module=object(),
        dist_module=FakeDist(),
        deep_ep_module=object(),
        group=object(),
        device=object(),
        args=object(),
        manifest=object(),
        num_sms=17,
        num_qps=9,
    )

    assert runtime.num_sms == 17
    assert runtime.num_qps == 9


def test_cuda_parity_buffer_dimensions_come_from_manifest():
    manifest = build_manifest(WorkloadSpec(
        world_size=2,
        num_tokens=17,
        hidden=64,
        num_topk=3,
        num_experts=8,
    ))
    args = SimpleNamespace(
        deterministic=False,
        allow_multiple_reduction=1,
        sl_idx=0,
        num_allocated_qps=0,
        num_qps=0,
        num_gpu_timeout_secs=100,
        num_cpu_timeout_secs=100,
    )

    settings = parity_buffer_settings(args, manifest)

    assert settings["num_max_tokens_per_rank"] == 17
    assert settings["hidden"] == 64
    assert settings["num_topk"] == 3
    assert settings["allow_hybrid_mode"] is False
    assert settings["use_fp8_dispatch"] is False


def test_cuda_spawn_entry_remains_picklable_after_lazy_import_refactor():
    assert pickle.loads(pickle.dumps(_test_loop)) is _test_loop
