import ast
from copy import deepcopy
from dataclasses import asdict
import json
import math
import re
import signal
import subprocess
from pathlib import Path
import sys

import pytest

from tests.benchmark.profiles import BenchmarkProfile, PROFILES, profile_manifest
from tests.benchmark.run_ep import (
    ASCEND_READINESS_COMMAND,
    RunConfig,
    build_backend_command,
    build_parser,
    execute_run,
    validate_ascend_readiness,
)
import tests.benchmark.run_ep as run_ep
from tests.benchmark.report_markdown import (
    comparison_rows,
    identify_profile,
    operation_records,
    render_backend_markdown,
    render_comparison_markdown,
    validate_complete_report,
    write_text_atomic,
)
from tests.utils.ep_benchmark_manifest import (
    enumerate_ep_mode_cases,
    load_manifest,
    write_manifest,
)


OPERATIONS = (
    "dispatch",
    "expanded_dispatch",
    "cached_dispatch",
    "combine",
    "reduced_combine",
)


def complete_report(platform, profile_name, device_name):
    profile = PROFILES[profile_name]
    manifest = profile_manifest(profile)
    summary = {
        "minimum": 1e-6,
        "mean": 1e-6,
        "p50": 1e-6,
        "p95": 1e-6,
        "maximum": 1e-6,
    }
    operations = [
        {
            "operation_id": operation_id,
            "formula_version": 1,
            "device_seconds": dict(summary),
            "wall_seconds": dict(summary),
            "device_samples": [1e-6] * profile.iterations,
            "wall_samples": [1e-6] * profile.iterations,
            "logical_bytes": {"scaleup": 2000},
            "logical_gbps": 2.0,
            "per_rank": [],
        }
        for operation_id in OPERATIONS
    ]
    cases = [
        {
            "case_id": case.case_id,
            "mode": asdict(case),
            "status": "passed",
            "reason": "",
            "operations": deepcopy(operations),
        }
        for case in enumerate_ep_mode_cases()
    ]
    return {
        "schema_version": 2,
        "formula_version": 1,
        "generated_at": "2026-08-20T00:00:00+00:00",
        "git_commit": "a" * 40,
        "platform": platform,
        "device": {"name": device_name},
        "world_size": 8,
        "workload": asdict(manifest.spec),
        "workload_fingerprint": manifest.fingerprint,
        "execution_protocol": {
            "allow_multiple_reduction": profile.allow_multiple_reduction,
        },
        "timing_protocol": {
            "timer": "cuda_event" if platform == "cuda" else "npu_event",
            "warmups": profile.warmups,
            "iterations": profile.iterations,
            "rank_aggregation": "maximum_latency",
            "logical_byte_aggregation": "sum",
        },
        "case_summary": {
            "total": 144,
            "pending": 0,
            "passed": 144,
            "failed": 0,
        },
        "cases": cases,
        "failures": [],
    }


def test_profiles_are_fixed_eight_rank_workloads():
    canonical = PROFILES["canonical"]
    smoke = PROFILES["smoke"]
    assert (
        canonical.world_size, canonical.num_tokens, canonical.hidden,
        canonical.num_topk, canonical.num_experts, canonical.seed,
        canonical.warmups, canonical.iterations,
    ) == (8, 4096, 7168, 6, 256, 0, 30, 30)
    assert (
        smoke.world_size, smoke.num_tokens, smoke.hidden,
        smoke.num_topk, smoke.num_experts, smoke.seed,
        smoke.warmups, smoke.iterations,
    ) == (8, 16, 128, 2, 8, 0, 1, 1)
    assert profile_manifest(canonical).ranks[7].num_tokens == 4089
    assert profile_manifest(smoke).ranks[7].num_tokens == 9


def test_profiles_keep_fixed_routing_and_reduction_settings():
    canonical = PROFILES["canonical"]
    smoke = PROFILES["smoke"]
    assert (
        canonical.unbalanced_ratio,
        canonical.precise_unbalanced_ratio,
        canonical.masked_ratio,
        canonical.allow_multiple_reduction,
    ) == (1.0, False, 0.0, 1)
    assert (
        smoke.unbalanced_ratio,
        smoke.precise_unbalanced_ratio,
        smoke.masked_ratio,
        smoke.allow_multiple_reduction,
    ) == (1.0, False, 0.0, 1)


def test_complete_report_requires_144_cases_and_720_operations():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    validate_complete_report(
        report, platform="cuda", profile=PROFILES["smoke"], require_h800=True
    )
    assert len(operation_records(report)) == 720


def test_complete_report_rejects_fabricated_profile():
    fabricated = BenchmarkProfile(
        name="fabricated",
        world_size=4,
        num_tokens=16,
        hidden=128,
        num_topk=2,
        num_experts=8,
        seed=0,
        warmups=1,
        iterations=1,
    )
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    manifest = profile_manifest(fabricated)
    report["world_size"] = fabricated.world_size
    report["workload"] = asdict(manifest.spec)
    report["workload_fingerprint"] = manifest.fingerprint

    with pytest.raises(ValueError, match="profile"):
        validate_complete_report(
            report, platform="cuda", profile=fabricated, require_h800=True
        )


def test_complete_report_rejects_unknown_platform():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["platform"] = "other"
    report["timing_protocol"]["timer"] = "npu_event"

    with pytest.raises(ValueError, match="platform"):
        validate_complete_report(
            report, platform="other", profile=PROFILES["smoke"], require_h800=True
        )


def test_identify_profile_matches_exact_workload_and_timing_tuple():
    report = complete_report("ascend", "canonical", device_name="Ascend 910B")

    assert identify_profile(report) is PROFILES["canonical"]


def test_identify_profile_rejects_unknown_workload_and_timing_tuple():
    report = complete_report("ascend", "smoke", device_name="Ascend 910B")
    report["timing_protocol"]["iterations"] = 2

    with pytest.raises(
        ValueError, match="report does not match canonical or smoke profile"
    ):
        identify_profile(report)


def test_identify_profile_rejects_noncanonical_timer():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["timing_protocol"]["timer"] = "host_clock"

    with pytest.raises(
        ValueError, match="report does not match canonical or smoke profile"
    ):
        identify_profile(report)


def test_identify_profile_rejects_schema_v1_with_field_specific_error():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["schema_version"] = 1

    with pytest.raises(ValueError, match="schema_version"):
        identify_profile(report)


def test_identify_profile_rejects_disabled_reduction_with_field_specific_error():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["execution_protocol"]["allow_multiple_reduction"] = 0

    with pytest.raises(
        ValueError,
        match="execution_protocol.allow_multiple_reduction",
    ):
        identify_profile(report)


@pytest.mark.parametrize(
    ("execution_protocol", "field"),
    (
        (
            {"allow_multiple_reduction": 0},
            "execution_protocol.allow_multiple_reduction",
        ),
        (
            {"allow_multiple_reduction": True},
            "execution_protocol.allow_multiple_reduction",
        ),
        (
            {"allow_multiple_reduction": 1, "unexpected": True},
            "execution_protocol",
        ),
        (None, "execution_protocol"),
    ),
)
def test_complete_report_requires_exact_enabled_execution_protocol(
    execution_protocol, field
):
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    report["execution_protocol"] = execution_protocol

    with pytest.raises(ValueError, match=re.escape(field)):
        validate_complete_report(
            report,
            platform="cuda",
            profile=PROFILES["smoke"],
            require_h800=True,
        )


def _set_wrong_platform(report):
    report["platform"] = "ascend"


def _set_wrong_world_size(report):
    report["world_size"] = 4


def _set_wrong_schema_version(report):
    report["schema_version"] = 1


def _set_non_h800_device(report):
    report["device"]["name"] = "NVIDIA A100"


def _set_wrong_workload(report):
    report["workload"]["hidden"] = 129


def _set_wrong_fingerprint(report):
    report["workload_fingerprint"] = "b" * 64


def _set_wrong_warmups(report):
    report["timing_protocol"]["warmups"] = 2


def _set_wrong_iterations(report):
    report["timing_protocol"]["iterations"] = 2


def _set_wrong_aggregation(report):
    report["timing_protocol"]["rank_aggregation"] = "mean_latency"


def _set_wrong_case_order(report):
    report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]


def _set_duplicate_case(report):
    report["cases"][1]["case_id"] = report["cases"][0]["case_id"]


def _set_failed_case(report):
    report["cases"][0]["status"] = "failed"


def _set_failures(report):
    report["failures"] = [{"case_id": report["cases"][0]["case_id"]}]


def _set_wrong_operation_order(report):
    operations = report["cases"][0]["operations"]
    operations[0], operations[1] = operations[1], operations[0]


def _set_wrong_formula_version(report):
    report["cases"][0]["operations"][0]["formula_version"] = 2


def _set_zero_latency(report):
    report["cases"][0]["operations"][0]["device_seconds"]["mean"] = 0.0


def _set_nan_latency(report):
    report["cases"][0]["operations"][0]["device_seconds"]["p95"] = math.nan


def _set_zero_gbps(report):
    report["cases"][0]["operations"][0]["logical_gbps"] = 0.0


def _set_nan_gbps(report):
    report["cases"][0]["operations"][0]["logical_gbps"] = math.nan


def _set_wrong_sample_count(report):
    report["cases"][0]["operations"][0]["device_samples"] = []


@pytest.mark.parametrize(
    ("mutate", "field"),
    (
        (_set_wrong_platform, "platform"),
        (_set_wrong_world_size, "world_size"),
        (_set_wrong_schema_version, "schema_version"),
        (_set_non_h800_device, "device.name"),
        (_set_wrong_workload, "workload.hidden"),
        (_set_wrong_fingerprint, "workload_fingerprint"),
        (_set_wrong_warmups, "timing_protocol.warmups"),
        (_set_wrong_iterations, "timing_protocol.iterations"),
        (_set_wrong_aggregation, "timing_protocol.rank_aggregation"),
        (_set_wrong_case_order, "cases[0].case_id"),
        (_set_duplicate_case, "cases[1].case_id"),
        (_set_failed_case, "cases[0].status"),
        (_set_failures, "failures"),
        (_set_wrong_operation_order, "cases[0].operations[0].operation_id"),
        (_set_wrong_formula_version, "cases[0].operations[0].formula_version"),
        (_set_zero_latency, "cases[0].operations[0].device_seconds.mean"),
        (_set_nan_latency, "cases[0].operations[0].device_seconds.p95"),
        (_set_zero_gbps, "cases[0].operations[0].logical_gbps"),
        (_set_nan_gbps, "cases[0].operations[0].logical_gbps"),
        (_set_wrong_sample_count, "cases[0].operations[0].device_samples"),
    ),
)
def test_complete_report_rejects_each_invalid_field(mutate, field):
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    mutate(report)

    with pytest.raises(ValueError, match=re.escape(field)):
        validate_complete_report(
            report,
            platform="cuda",
            profile=PROFILES["smoke"],
            require_h800=True,
        )


def test_backend_markdown_renders_smoke_warning_and_canonical_detail_rows():
    report = complete_report("cuda", "smoke", device_name="NVIDIA H800")

    markdown = render_backend_markdown(report, PROFILES["smoke"])

    assert "# EP Benchmark: CUDA / smoke" in markdown
    assert "NON-CANONICAL AUTOMATION VALIDATION" in markdown
    assert "| Case | Operation | Mean us | P50 us | P95 us | Logical GB/s |" in markdown
    assert markdown.count("\n| `ep-") == 720
    assert (
        "| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | dispatch | "
        "1.000 | 1.000 | 1.000 | 2.000 |"
    ) in markdown
    assert markdown.index("| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | dispatch") < markdown.index(
        "| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | cached_dispatch"
    )
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_backend_markdown_renders_canonical_provenance_workload_and_escaped_cells():
    report = complete_report("ascend", "canonical", device_name="Ascend|950\\rack\r\nnext")
    report["git_commit"] = "commit|abc\\def\r\nnext"

    markdown = render_backend_markdown(report, PROFILES["canonical"])

    assert "# EP Benchmark: Ascend / canonical" in markdown
    assert "NON-CANONICAL AUTOMATION VALIDATION" not in markdown
    assert "| Git commit | commit\\|abc\\\\def\\r\\nnext |" in markdown
    assert "| Device | Ascend\\|950\\\\rack\\r\\nnext |" in markdown
    assert "| World size | 8 |" in markdown
    assert "| Workload fingerprint | " + report["workload_fingerprint"] + " |" in markdown
    assert "| Tokens | 4096 |" in markdown
    assert "| Hidden | 7168 |" in markdown
    assert "| Top-k | 6 |" in markdown
    assert "| Experts | 256 |" in markdown


def _make_comparison_reports():
    cuda = complete_report("cuda", "smoke", device_name="NVIDIA H800")
    ascend = complete_report("ascend", "smoke", device_name="Ascend 950")
    cuda_operation = cuda["cases"][0]["operations"][0]
    ascend_operation = ascend["cases"][0]["operations"][0]
    cuda_operation["device_seconds"].update(mean=1e-6, p50=2e-6, p95=3e-6)
    ascend_operation["device_seconds"].update(mean=2e-6, p50=4e-6, p95=6e-6)
    cuda_operation["logical_gbps"] = 4.0
    ascend_operation["logical_gbps"] = 2.0
    return cuda, ascend


def test_comparison_rows_and_markdown_render_ascend_over_cuda_ratios():
    cuda, ascend = _make_comparison_reports()

    rows = comparison_rows(cuda, ascend)
    markdown = render_comparison_markdown(cuda, ascend, PROFILES["smoke"])

    assert len(rows) == 720
    assert rows[0]["latency_ratio"] == pytest.approx(2.0)
    assert rows[0]["bandwidth_ratio"] == pytest.approx(0.5)
    assert "# EP Benchmark Comparison: H800 vs Ascend" in markdown
    assert "NON-CANONICAL AUTOMATION VALIDATION" in markdown
    assert markdown.count("\n| `ep-") == 720
    assert (
        "| `ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0` | dispatch | "
        "1.000 | 2.000 | 3.000 | 4.000 | 2.000 | 4.000 | 6.000 | 2.000 | "
        "2.000 | 0.500 |"
    ) in markdown


def test_canonical_comparison_renders_workload_before_detail_without_warning():
    cuda = complete_report("cuda", "canonical", device_name="NVIDIA H800")
    ascend = complete_report("ascend", "canonical", device_name="Ascend 950")

    markdown = render_comparison_markdown(
        cuda, ascend, PROFILES["canonical"]
    )

    assert "NON-CANONICAL AUTOMATION VALIDATION" not in markdown
    assert markdown.index("## Provenance") < markdown.index("## Workload")
    assert markdown.index("## Workload") < markdown.index("## Detail")
    assert "| Tokens | 4096 |" in markdown
    assert "| Hidden | 7168 |" in markdown
    assert "| Top-k | 6 |" in markdown
    assert "| Experts | 256 |" in markdown
    assert "| Seed | 0 |" in markdown
    assert "| Warmups | 30 |" in markdown
    assert "| Iterations | 30 |" in markdown


def test_automation_comparison_rejects_matching_disabled_reduction_reports():
    cuda, ascend = _make_comparison_reports()
    cuda["execution_protocol"]["allow_multiple_reduction"] = 0
    ascend["execution_protocol"]["allow_multiple_reduction"] = 0

    with pytest.raises(
        ValueError,
        match="execution_protocol.allow_multiple_reduction",
    ):
        render_comparison_markdown(cuda, ascend, PROFILES["smoke"])


def _set_comparison_profile_mismatch(cuda, ascend):
    replacement = complete_report("ascend", "canonical", device_name="Ascend 950")
    ascend.clear()
    ascend.update(replacement)


def _set_comparison_schema_mismatch(cuda, ascend):
    ascend["schema_version"] = 1


def _set_comparison_formula_mismatch(cuda, ascend):
    ascend["formula_version"] = 2


def _set_comparison_fingerprint_mismatch(cuda, ascend):
    ascend["workload_fingerprint"] = "b" * 64


def _set_comparison_world_size_mismatch(cuda, ascend):
    ascend["world_size"] = 4


def _set_comparison_aggregation_mismatch(cuda, ascend):
    ascend["timing_protocol"]["rank_aggregation"] = "mean_latency"


def _set_comparison_execution_protocol_mismatch(cuda, ascend):
    ascend["execution_protocol"]["allow_multiple_reduction"] = 0


def _set_comparison_case_sequence_mismatch(cuda, ascend):
    ascend["cases"][0], ascend["cases"][1] = ascend["cases"][1], ascend["cases"][0]


def _set_comparison_operation_sequence_mismatch(cuda, ascend):
    operations = ascend["cases"][0]["operations"]
    operations[0], operations[1] = operations[1], operations[0]


def _set_comparison_logical_bytes_mismatch(cuda, ascend):
    ascend["cases"][0]["operations"][0]["logical_bytes"] = {"scaleup": 2001}


@pytest.mark.parametrize(
    ("mutate", "field"),
    (
        (_set_comparison_profile_mismatch, "profile"),
        (_set_comparison_schema_mismatch, "schema_version"),
        (_set_comparison_formula_mismatch, "formula_version"),
        (_set_comparison_fingerprint_mismatch, "workload_fingerprint"),
        (_set_comparison_world_size_mismatch, "world_size"),
        (_set_comparison_aggregation_mismatch, "timing_protocol.rank_aggregation"),
        (
            _set_comparison_execution_protocol_mismatch,
            "execution_protocol.allow_multiple_reduction",
        ),
        (_set_comparison_case_sequence_mismatch, "cases[0].case_id"),
        (_set_comparison_operation_sequence_mismatch, "cases[0].operations[0].operation_id"),
        (_set_comparison_logical_bytes_mismatch, "logical_bytes"),
    ),
)
def test_comparison_rejects_each_identity_mismatch_without_writing_output(
    tmp_path, mutate, field
):
    cuda, ascend = _make_comparison_reports()
    output = tmp_path / "comparison.md"
    mutate(cuda, ascend)

    with pytest.raises(ValueError, match=re.escape(field)):
        render_comparison_markdown(cuda, ascend, PROFILES["smoke"])

    assert not output.exists()


def _run_comparison_cli(cuda_path, ascend_path, output_path):
    repository_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "tests/benchmark/compare_ep.py",
            "--cuda",
            str(cuda_path),
            "--ascend",
            str(ascend_path),
            "--output",
            str(output_path),
        ],
        cwd=repository_root,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )


def _write_comparison_inputs(tmp_path, cuda, ascend):
    cuda_path = tmp_path / "cuda.json"
    ascend_path = tmp_path / "ascend.json"
    cuda_path.write_text(json.dumps(cuda), encoding="utf-8")
    ascend_path.write_text(json.dumps(ascend), encoding="utf-8")
    return cuda_path, ascend_path


def test_comparison_cli_writes_complete_noncanonical_markdown_with_empty_environment(
    tmp_path,
):
    cuda, ascend = _make_comparison_reports()
    cuda_path, ascend_path = _write_comparison_inputs(tmp_path, cuda, ascend)
    output = tmp_path / "comparison.md"

    result = _run_comparison_cli(cuda_path, ascend_path, output)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    markdown = output.read_text(encoding="utf-8")
    assert "NON-CANONICAL AUTOMATION VALIDATION" in markdown
    assert markdown.count("\n| `ep-") == 720


@pytest.mark.parametrize("input_name", ("cuda", "ascend"))
@pytest.mark.parametrize("alias_kind", ("normalized", "hardlink"))
def test_comparison_cli_rejects_output_aliasing_either_input_without_modifying_sources(
    tmp_path, input_name, alias_kind
):
    cuda, ascend = _make_comparison_reports()
    cuda_path, ascend_path = _write_comparison_inputs(tmp_path, cuda, ascend)
    source_paths = {"cuda": cuda_path, "ascend": ascend_path}
    original_bytes = {
        name: path.read_bytes() for name, path in source_paths.items()
    }
    aliased_input = source_paths[input_name]
    if alias_kind == "normalized":
        alias_directory = tmp_path / "alias-directory"
        alias_directory.mkdir()
        output = alias_directory / ".." / aliased_input.name
    else:
        output = tmp_path / f"{input_name}-alias.md"
        output.hardlink_to(aliased_input)

    result = _run_comparison_cli(cuda_path, ascend_path, output)

    assert result.returncode != 0
    assert f"--output must not alias --{input_name}" in result.stderr
    assert "Traceback" not in result.stderr
    assert cuda_path.read_bytes() == original_bytes["cuda"]
    assert ascend_path.read_bytes() == original_bytes["ascend"]
    assert output.read_bytes() == original_bytes[input_name]


@pytest.mark.parametrize(
    ("failure", "field"),
    (
        ("wrong_platform", "platform"),
        ("wrong_fingerprint", "workload_fingerprint"),
        ("wrong_case_count", "cases"),
    ),
)
def test_comparison_cli_rejects_invalid_reports_without_replacing_output(
    tmp_path, failure, field
):
    cuda, ascend = _make_comparison_reports()
    if failure == "wrong_platform":
        cuda["platform"] = "ascend"
    elif failure == "wrong_fingerprint":
        ascend["workload_fingerprint"] = "b" * 64
    else:
        ascend["cases"].pop()
    cuda_path, ascend_path = _write_comparison_inputs(tmp_path, cuda, ascend)
    output = tmp_path / "comparison.md"
    output.write_text("existing\n", encoding="utf-8")

    result = _run_comparison_cli(cuda_path, ascend_path, output)

    assert result.returncode != 0
    assert field in result.stderr
    assert "Traceback" not in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_comparison_cli_reports_missing_input_without_replacing_output(tmp_path):
    _, ascend = _make_comparison_reports()
    missing_cuda = tmp_path / "missing-cuda.json"
    ascend_path = tmp_path / "ascend.json"
    ascend_path.write_text(json.dumps(ascend), encoding="utf-8")
    output = tmp_path / "comparison.md"
    output.write_text("existing\n", encoding="utf-8")

    result = _run_comparison_cli(missing_cuda, ascend_path, output)

    assert result.returncode != 0
    assert str(missing_cuda) in result.stderr
    assert "No such file or directory" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_comparison_cli_reports_invalid_json_without_replacing_output(tmp_path):
    _, ascend = _make_comparison_reports()
    cuda_path = tmp_path / "cuda.json"
    ascend_path = tmp_path / "ascend.json"
    cuda_path.write_text("{invalid\n", encoding="utf-8")
    ascend_path.write_text(json.dumps(ascend), encoding="utf-8")
    output = tmp_path / "comparison.md"
    output.write_text("existing\n", encoding="utf-8")

    result = _run_comparison_cli(cuda_path, ascend_path, output)

    assert result.returncode != 0
    assert "Expecting property name enclosed in double quotes" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_comparison_cli_rejects_malformed_cuda_device_without_replacing_output(
    tmp_path,
):
    cuda, ascend = _make_comparison_reports()
    cuda["device"] = []
    cuda_path, ascend_path = _write_comparison_inputs(tmp_path, cuda, ascend)
    output = tmp_path / "comparison.md"
    output.write_text("existing\n", encoding="utf-8")

    result = _run_comparison_cli(cuda_path, ascend_path, output)

    assert result.returncode != 0
    assert "device" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_comparison_cli_rejects_malformed_ascend_device_without_creating_output(
    tmp_path,
):
    cuda, ascend = _make_comparison_reports()
    ascend["device"] = []
    cuda_path, ascend_path = _write_comparison_inputs(tmp_path, cuda, ascend)
    output = tmp_path / "comparison.md"

    result = _run_comparison_cli(cuda_path, ascend_path, output)

    assert result.returncode != 0
    assert "device" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_comparison_modules_are_runtime_free():
    repository_root = Path(__file__).resolve().parents[2]
    forbidden = (
        "torch",
        "torch_npu",
        "deep_ep",
        "tests.ascend.benchmark.runtime",
    )
    imported = {}
    for relative_path in (
        "tests/benchmark/compare_ep.py",
        "tests/benchmark/report_markdown.py",
        "tests/benchmark/profiles.py",
    ):
        source = (repository_root / relative_path).read_text(encoding="utf-8")
        modules = []
        for node in ast.walk(ast.parse(source, filename=relative_path)):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        imported[relative_path] = modules

    violations = {
        path: [
            module
            for module in modules
            if any(module == name or module.startswith(f"{name}.") for name in forbidden)
        ]
        for path, modules in imported.items()
    }
    assert violations == {
        "tests/benchmark/compare_ep.py": [],
        "tests/benchmark/report_markdown.py": [],
        "tests/benchmark/profiles.py": [],
    }


def test_write_text_atomic_replaces_destination_and_cleans_failed_temporary_file(
    tmp_path, monkeypatch
):
    output = tmp_path / "report.md"
    output.write_text("old\n", encoding="utf-8")
    write_text_atomic(output, "new\n")
    assert output.read_text(encoding="utf-8") == "new\n"

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("tests.benchmark.report_markdown.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_text_atomic(output, "uncommitted\n")

    assert output.read_text(encoding="utf-8") == "new\n"
    assert list(Path(tmp_path).glob(".report.md.*")) == []


def _adjacent_pairs(command):
    return tuple(zip(command, command[1:]))


def _argument_value(command, option):
    return command[command.index(option) + 1]


def test_run_parser_defaults_to_smoke_without_per_size_overrides(tmp_path):
    parser = build_parser()

    args = parser.parse_args([
        "--backend", "cuda", "--output-dir", str(tmp_path),
    ])

    assert args.profile == "smoke"
    help_text = parser.format_help()
    assert "--num-tokens" not in help_text
    assert "--hidden" not in help_text
    assert "--num-topk" not in help_text
    assert "--num-experts" not in help_text


def test_cuda_command_expands_canonical_profile_and_all_cases(tmp_path):
    staging = tmp_path / "benchmark.staging.json"
    manifest = tmp_path / "workload.json"
    command = build_backend_command(
        RunConfig("cuda", "canonical", tmp_path, None), staging, manifest
    )

    assert isinstance(command, tuple)
    assert command[:3] == (
        sys.executable, "tests/elastic/test_ep.py", "--benchmark-profile",
    )
    pairs = _adjacent_pairs(command)
    assert ("--benchmark-profile", "parity") in pairs
    assert ("--num-processes", "8") in pairs
    assert ("--num-tokens", "4096") in pairs
    assert ("--hidden", "7168") in pairs
    assert ("--num-topk", "6") in pairs
    assert ("--num-experts", "256") in pairs
    assert ("--warmups", "30") in pairs
    assert ("--iterations", "30") in pairs
    assert ("--workload-manifest", str(manifest)) in pairs
    assert ("--benchmark-json", str(staging)) in pairs
    assert "benchmark.json" not in command
    assert len(_argument_value(command, "--cases").split(",")) == 144
    assert "--num-sms" not in command
    assert "--num-qps" not in command


def test_ascend_command_uses_eight_rank_torchrun_and_staging_only(tmp_path):
    staging = tmp_path / "benchmark.staging.json"
    manifest = tmp_path / "workload.json"
    command = build_backend_command(
        RunConfig("ascend", "smoke", tmp_path, None), staging, manifest
    )

    assert isinstance(command, tuple)
    assert command[:4] == (
        "torchrun", "--standalone", "--nproc-per-node=8",
        "tests/ascend/benchmark/bench_ep.py",
    )
    pairs = _adjacent_pairs(command)
    assert ("--num-tokens", "16") in pairs
    assert ("--hidden", "128") in pairs
    assert ("--num-topk", "2") in pairs
    assert ("--num-experts", "8") in pairs
    assert ("--warmups", "1") in pairs
    assert ("--iterations", "1") in pairs
    assert ("--workload-manifest", str(manifest)) in pairs
    assert ("--output", str(staging)) in pairs
    assert "benchmark.json" not in command
    assert len(_argument_value(command, "--cases").split(",")) == 144


def test_ascend_readiness_requires_the_complete_supported_inventory():
    validate_ascend_readiness({
        "summary": {"total": 144, "supported": 144, "deferred": 0},
        "cases": [],
    })

    for summary in (
        {"total": 143, "supported": 143, "deferred": 0},
        {"total": 144, "supported": 84, "deferred": 60},
        {"total": 144, "supported": 144, "deferred": 1},
        {"total": 144, "supported": 144},
    ):
        with pytest.raises(ValueError, match="Ascend readiness"):
            validate_ascend_readiness({"summary": summary})


def test_live_ascend_inventory_satisfies_automation_readiness():
    result = subprocess.run(
        ASCEND_READINESS_COMMAND,
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"] == {
        "total": 144,
        "supported": 144,
        "deferred": 0,
    }
    validate_ascend_readiness(payload)


class FakeRunCommand:
    def __init__(
        self,
        report=None,
        exit_code=0,
        write_staging=True,
        readiness_summary=None,
    ):
        self.report = report
        self.exit_code = exit_code
        self.write_staging = write_staging
        self.readiness_summary = readiness_summary or {
            "total": 144, "supported": 144, "deferred": 0,
        }
        self.commands = []

    def __call__(self, command, log_handle):
        self.commands.append(command)
        if "--list-cases" in command:
            payload = {
                "summary": self.readiness_summary,
                "cases": [],
            }
            output = json.dumps(payload) + "\n"
            log_handle.write(output)
            return 0, output

        staging_option = "--benchmark-json" if "--benchmark-json" in command else "--output"
        if self.write_staging:
            staging = Path(_argument_value(command, staging_option))
            if self.report is None:
                staging.write_text("child diagnostic staging\n", encoding="utf-8")
            else:
                staging.write_text(json.dumps(self.report), encoding="utf-8")
        log_handle.write("fake child output\n")
        return self.exit_code, "fake child output\n"


@pytest.mark.parametrize(
    ("backend", "device_name"),
    (("cuda", "NVIDIA H800"), ("ascend", "Ascend 950")),
)
def test_execute_run_publishes_exactly_four_final_artifacts(
    tmp_path, backend, device_name
):
    output_dir = tmp_path / backend
    runner = FakeRunCommand(complete_report(backend, "smoke", device_name))

    result = execute_run(
        RunConfig(backend, "smoke", output_dir, None),
        command_runner=runner,
    )

    assert result == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "workload.json", "benchmark.json", "benchmark.md", "run.log",
    }
    assert load_manifest(output_dir / "workload.json") == profile_manifest(
        PROFILES["smoke"]
    )
    assert json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))[
        "platform"
    ] == backend
    assert (output_dir / "benchmark.md").read_text(encoding="utf-8").startswith(
        f"# EP Benchmark: {'CUDA' if backend == 'cuda' else 'Ascend'} / smoke\n"
    )
    log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert '"command": [' in log
    assert '"backend": "' + backend + '"' in log
    assert '"profile": "smoke"' in log
    assert "fake child output" in log
    if backend == "ascend":
        assert runner.commands[0] == ASCEND_READINESS_COMMAND


def test_execute_run_returns_child_exit_and_retains_failure_artifacts(tmp_path):
    output_dir = tmp_path / "failed-child"
    runner = FakeRunCommand(report=None, exit_code=17)

    result = execute_run(
        RunConfig("cuda", "smoke", output_dir, None),
        command_runner=runner,
    )

    assert result == 17
    assert (output_dir / "run.log").exists()
    assert (output_dir / "workload.json").exists()
    assert (output_dir / "benchmark.staging.json").read_text(
        encoding="utf-8"
    ) == "child diagnostic staging\n"
    assert not (output_dir / "benchmark.json").exists()
    assert not (output_dir / "benchmark.md").exists()
    log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "fake child output" in log
    assert '"event": "run_end"' in log
    assert '"exit_code": 17' in log


def test_ascend_readiness_failure_retains_workload_and_skips_device_command(tmp_path):
    output_dir = tmp_path / "not-ready"
    runner = FakeRunCommand(
        readiness_summary={"total": 144, "supported": 84, "deferred": 60}
    )

    result = execute_run(
        RunConfig("ascend", "smoke", output_dir, None),
        command_runner=runner,
    )

    assert result == 1
    assert len(runner.commands) == 1
    assert "--list-cases" in runner.commands[0]
    assert (output_dir / "workload.json").exists()
    assert not (output_dir / "benchmark.staging.json").exists()
    assert not (output_dir / "benchmark.json").exists()
    assert not (output_dir / "benchmark.md").exists()
    log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "Ascend readiness" in log
    assert '"event": "run_end"' in log
    assert '"exit_code": 1' in log


@pytest.mark.parametrize("failure", ("missing", "invalid"))
def test_execute_run_rejects_missing_or_invalid_staging_report(tmp_path, failure):
    output_dir = tmp_path / failure
    report = None
    write_staging = False
    if failure == "invalid":
        report = complete_report("cuda", "smoke", "NVIDIA A100")
        write_staging = True
    runner = FakeRunCommand(report=report, write_staging=write_staging)

    result = execute_run(
        RunConfig("cuda", "smoke", output_dir, None),
        command_runner=runner,
    )

    assert result == 1
    assert (output_dir / "run.log").exists()
    assert (output_dir / "workload.json").exists()
    assert not (output_dir / "benchmark.json").exists()
    assert not (output_dir / "benchmark.md").exists()
    log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert ("staging report" if failure == "missing" else "device.name") in log
    if failure == "invalid":
        assert (output_dir / "benchmark.staging.json").exists()


@pytest.mark.parametrize(
    "artifact_name",
    ("benchmark.json", "benchmark.md", "benchmark.staging.json"),
)
def test_execute_run_rejects_output_with_completed_or_staging_artifact(
    tmp_path, artifact_name
):
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    artifact = output_dir / artifact_name
    artifact.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already contains"):
        execute_run(
            RunConfig("cuda", "smoke", output_dir, None),
            command_runner=FakeRunCommand(),
        )

    assert artifact.read_text(encoding="utf-8") == "keep me\n"


def test_execute_run_rejects_repository_root_output():
    repository_root = Path(run_ep.__file__).resolve().parents[2]

    with pytest.raises(ValueError, match="repository root"):
        execute_run(
            RunConfig("cuda", "smoke", repository_root, None),
            command_runner=FakeRunCommand(),
        )


def test_execute_run_keeps_staging_diagnostics_when_markdown_rendering_fails(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "markdown-failure"
    runner = FakeRunCommand(complete_report("cuda", "smoke", "NVIDIA H800"))

    def fail_markdown(report, profile):
        raise RuntimeError("markdown exploded")

    monkeypatch.setattr(run_ep, "render_backend_markdown", fail_markdown)
    result = execute_run(
        RunConfig("cuda", "smoke", output_dir, None),
        command_runner=runner,
    )

    assert result == 1
    assert (output_dir / "run.log").exists()
    assert (output_dir / "workload.json").exists()
    assert (output_dir / "benchmark.staging.json").exists()
    assert not (output_dir / "benchmark.json").exists()
    assert not (output_dir / "benchmark.md").exists()
    assert list(output_dir.glob(".benchmark.md.*")) == []
    assert "markdown exploded" in (output_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("fail_on_promotion", "expected_promotions"),
    (
        pytest.param(1, ("benchmark.md",), id="first-promotion"),
        pytest.param(
            2,
            ("benchmark.md", "benchmark.json"),
            id="after-one-final-artifact",
        ),
    ),
)
def test_execute_run_rolls_back_injected_publication_failures(
    tmp_path,
    monkeypatch,
    fail_on_promotion,
    expected_promotions,
):
    output_dir = tmp_path / f"promotion-{fail_on_promotion}"
    report = complete_report("cuda", "smoke", "NVIDIA H800")
    runner = FakeRunCommand(report)
    real_replace = run_ep.os.replace
    promotions = []

    def fail_selected_promotion(source, destination):
        destination = Path(destination)
        if destination.name in ("benchmark.md", "benchmark.json"):
            promotions.append(destination.name)
            if len(promotions) == fail_on_promotion:
                raise OSError(
                    f"injected {destination.name} promotion failure"
                )
        return real_replace(source, destination)

    monkeypatch.setattr(run_ep.os, "replace", fail_selected_promotion)

    result = execute_run(
        RunConfig("cuda", "smoke", output_dir, None),
        command_runner=runner,
    )

    assert result == 1
    assert tuple(promotions) == expected_promotions
    assert not (output_dir / "benchmark.json").exists()
    assert not (output_dir / "benchmark.md").exists()
    assert json.loads(
        (output_dir / "benchmark.staging.json").read_text(encoding="utf-8")
    ) == report
    assert load_manifest(output_dir / "workload.json") == profile_manifest(
        PROFILES["smoke"]
    )
    assert list(output_dir.glob(".benchmark.md.*")) == []
    log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "injected benchmark." in log
    assert '"event": "run_end"' in log
    assert '"exit_code": 1' in log


def test_input_manifest_is_validated_and_copied_byte_for_byte(tmp_path):
    source = tmp_path / "shared.json"
    write_manifest(source, profile_manifest(PROFILES["smoke"]))
    source_bytes = source.read_bytes() + b" \n"
    source.write_bytes(source_bytes)
    output_dir = tmp_path / "copied"
    runner = FakeRunCommand(exit_code=9, write_staging=False)

    result = execute_run(
        RunConfig("cuda", "smoke", output_dir, source),
        command_runner=runner,
    )

    assert result == 9
    copied = output_dir / "workload.json"
    assert copied.read_bytes() == source_bytes
    assert ("--workload-manifest", str(copied.resolve())) in _adjacent_pairs(
        runner.commands[-1]
    )


@pytest.mark.parametrize("mutation", ("fingerprint", "profile"))
def test_input_manifest_rejects_bad_fingerprint_or_wrong_profile(tmp_path, mutation):
    source = tmp_path / "shared.json"
    write_manifest(source, profile_manifest(PROFILES["canonical"]))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if mutation == "fingerprint":
        payload["fingerprint"] = "0" * 64
    source.write_text(json.dumps(payload), encoding="utf-8")
    runner = FakeRunCommand()
    output_dir = tmp_path / mutation

    result = execute_run(
        RunConfig("cuda", "smoke", output_dir, source),
        command_runner=runner,
    )

    assert result == 1
    assert runner.commands == []
    assert not (output_dir / "benchmark.json").exists()
    assert not (output_dir / "benchmark.md").exists()
    assert ("fingerprint" if mutation == "fingerprint" else "profile") in (
        output_dir / "run.log"
    ).read_text(encoding="utf-8")


def test_generated_manifest_is_deterministic_and_child_consumes_output_copy(tmp_path):
    generated = []
    for name in ("one", "two"):
        output_dir = tmp_path / name
        runner = FakeRunCommand(exit_code=8, write_staging=False)
        result = execute_run(
            RunConfig("cuda", "smoke", output_dir, None),
            command_runner=runner,
        )
        assert result == 8
        manifest_path = (output_dir / "workload.json").resolve()
        generated.append(manifest_path.read_bytes())
        assert ("--workload-manifest", str(manifest_path)) in _adjacent_pairs(
            runner.commands[-1]
        )

    assert generated[0] == generated[1]


@pytest.mark.parametrize("failure_site", ("output", "git", "log"))
def test_run_cli_reports_setup_os_errors_without_traceback(
    tmp_path, monkeypatch, capsys, failure_site
):
    def fail_setup(*args, **kwargs):
        raise PermissionError("setup denied")

    if failure_site == "output":
        monkeypatch.setattr(run_ep, "_prepare_output_directory", fail_setup)
    elif failure_site == "git":
        monkeypatch.setattr(run_ep, "_git_commit", fail_setup)
    else:
        original_open = Path.open

        def fail_log_open(path, *args, **kwargs):
            if path.name == "run.log":
                fail_setup()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_log_open)

    exit_code = run_ep.main([
        "--backend", "cuda",
        "--output-dir", str(tmp_path / failure_site),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "run_ep.py: error: PermissionError: setup denied\n"
    assert "Traceback" not in captured.err


def test_run_cli_prints_spawn_diagnostic_and_retains_it_in_log(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "spawn-failure"
    monkeypatch.setattr(run_ep, "_git_commit", lambda: "a" * 40)

    def fail_spawn(*args, **kwargs):
        raise FileNotFoundError("backend executable missing")

    monkeypatch.setattr(run_ep.subprocess, "Popen", fail_spawn)

    exit_code = run_ep.main([
        "--backend", "cuda", "--output-dir", str(output_dir),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "run_ep.py: error: FileNotFoundError: backend executable missing\n"
    )
    assert "Traceback" not in captured.err
    log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "FileNotFoundError: backend executable missing" in log


class _TeeFailureStdout:
    def __init__(self):
        self.closed = False

    def __iter__(self):
        yield "child output before tee failure\n"

    def close(self):
        self.closed = True


class _LaunchedProcessDouble:
    def __init__(self, wait_outcomes=()):
        self.pid = 4321
        self.stdout = _TeeFailureStdout()
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls = []
        self.wait_outcomes = list(wait_outcomes)

    def poll(self):
        return None

    def terminate(self):
        self.terminate_called = True
        raise OSError("terminate failed")

    def kill(self):
        self.kill_called = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_outcomes:
            outcome = self.wait_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return -9


class _FailingLogTee:
    def write(self, content):
        raise OSError("run log write failed")

    def flush(self):
        raise AssertionError("flush must follow a successful write")


def test_run_logged_command_starts_child_in_new_session(monkeypatch, capsys):
    process = _LaunchedProcessDouble()
    popen_call = {}

    def capture_popen(command, **kwargs):
        popen_call["command"] = command
        popen_call["kwargs"] = kwargs
        return process

    monkeypatch.setattr(run_ep.subprocess, "Popen", capture_popen)

    exit_code, output = run_ep.run_logged_command(
        ("harmless-child",), _RecordingLogTee()
    )

    assert exit_code == -9
    assert output == "child output before tee failure\n"
    assert capsys.readouterr().out == output
    assert popen_call["command"] == ("harmless-child",)
    assert popen_call["kwargs"]["start_new_session"] is True


class _RecordingLogTee:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        pass


def test_run_logged_command_signals_process_group_and_uses_bounded_waits(
    monkeypatch
):
    process = _LaunchedProcessDouble(wait_outcomes=(
        run_ep.subprocess.TimeoutExpired("harmless-child", 5.0),
        -9,
    ))
    group_signals = []
    monkeypatch.setattr(run_ep.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        run_ep.os,
        "killpg",
        lambda process_group, sent_signal: group_signals.append(
            (process_group, sent_signal)
        ),
    )

    with pytest.raises(OSError, match="run log write failed"):
        run_ep.run_logged_command(("harmless-child",), _FailingLogTee())

    assert process.stdout.closed
    assert group_signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait_calls == [5.0, 5.0]
    assert not process.terminate_called
    assert not process.kill_called


def test_run_logged_command_never_waits_unbounded_when_group_cleanup_fails(
    monkeypatch
):
    process = _LaunchedProcessDouble(wait_outcomes=(
        run_ep.subprocess.TimeoutExpired("harmless-child", 5.0),
        run_ep.subprocess.TimeoutExpired("harmless-child", 5.0),
    ))
    group_signals = []
    monkeypatch.setattr(run_ep.subprocess, "Popen", lambda *args, **kwargs: process)

    def fail_group_signal(process_group, sent_signal):
        group_signals.append((process_group, sent_signal))
        raise OSError("group signal failed")

    monkeypatch.setattr(run_ep.os, "killpg", fail_group_signal)

    with pytest.raises(OSError, match="run log write failed"):
        run_ep.run_logged_command(("harmless-child",), _FailingLogTee())

    assert group_signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait_calls == [5.0, 5.0]
