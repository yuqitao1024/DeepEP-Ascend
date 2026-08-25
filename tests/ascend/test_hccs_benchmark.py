import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "tests" / "ascend" / "hccs_benchmark"


def benchmark_module():
    return importlib.import_module("tests.ascend.hccs_benchmark.benchmark")


def test_parse_size_mib_list_converts_to_bytes():
    benchmark = benchmark_module()

    assert benchmark.parse_size_mib_list("1,4,16,64") == (
        1 << 20,
        4 << 20,
        16 << 20,
        64 << 20,
    )


@pytest.mark.parametrize("value", ["", "0", "1,0", "-1", "1.5", "1,1"])
def test_parse_size_mib_list_rejects_invalid_values(value):
    benchmark = benchmark_module()

    with pytest.raises(ValueError):
        benchmark.parse_size_mib_list(value)


def test_summarize_measurement_uses_slowest_rank_and_aggregate_bytes():
    benchmark = benchmark_module()

    row = benchmark.summarize_measurement(
        mode="all-to-all",
        payload_bytes=16 << 20,
        active_senders=8,
        peers_per_sender=7,
        rank_samples_seconds=(
            (0.010, 0.012, 0.011),
            (0.009, 0.013, 0.010),
        ),
    )

    assert row["samples_seconds"] == pytest.approx([0.010, 0.013, 0.011])
    assert row["mean_seconds"] == pytest.approx(0.034 / 3)
    assert row["p50_seconds"] == pytest.approx(0.011)
    assert row["p95_seconds"] == pytest.approx(0.0128)
    assert row["bytes_per_iteration"] == 8 * 7 * (16 << 20)
    assert row["logical_gbps"] == pytest.approx(
        row["bytes_per_iteration"] / row["mean_seconds"] / 1e9
    )


def test_summarize_measurement_accepts_nonuniform_bytes_and_phase_samples():
    benchmark = benchmark_module()

    row = benchmark.summarize_measurement(
        mode="transport-only",
        payload_bytes=0,
        active_senders=8,
        peers_per_sender=7,
        aggregate_bytes=123_456,
        rank_samples_seconds=((0.010, 0.020), (0.015, 0.018)),
        phase_rank_samples_seconds={
            "producer": ((0.001, 0.002), (0.003, 0.001)),
            "service": ((0.007, 0.015), (0.010, 0.014)),
        },
    )

    assert row["bytes_per_iteration"] == 123_456
    assert row["phases"]["producer"]["samples_seconds"] == pytest.approx(
        [0.003, 0.002]
    )
    assert row["phases"]["service"]["mean_seconds"] == pytest.approx(0.0125)


def test_peer_payloads_define_each_communication_pattern():
    benchmark = benchmark_module()

    assert benchmark.peer_payloads("p2p", rank=0, world_size=2,
                                   payload_bytes=1024) == (0, 1024)
    assert benchmark.peer_payloads("p2p", rank=1, world_size=2,
                                   payload_bytes=1024) == (0, 0)
    assert benchmark.peer_payloads(
        "p2p", rank=1, world_size=2, payload_bytes=1024, p2p_sender=1
    ) == (1024, 0)
    assert benchmark.peer_payloads("ring", rank=3, world_size=4,
                                   payload_bytes=1024) == (
        1024, 0, 0, 0
    )
    assert benchmark.peer_payloads("all-to-all", rank=2, world_size=4,
                                   payload_bytes=1024) == (
        1024, 1024, 0, 1024
    )
    assert benchmark.peer_payloads(
        "transport-only", rank=0, world_size=8, payload_bytes=0,
        transport_record_bytes=7552,
    ) == (
        0,
        41_656_832,
        40_531_584,
        40_327_680,
        40_886_528,
        41_332_096,
        41_075_328,
        40_342_784,
    )


def test_transport_only_requires_production_record_stride():
    benchmark = benchmark_module()

    with pytest.raises(ValueError, match="record stride"):
        benchmark.peer_payloads(
            "transport-only", rank=0, world_size=8, payload_bytes=0)


def test_expected_payloads_transposes_sender_rows():
    benchmark = benchmark_module()
    matrix = (
        (0, 10, 20),
        (30, 0, 40),
        (50, 60, 0),
    )

    assert benchmark.expected_payloads(matrix, destination_rank=1) == (
        10, 0, 60
    )


def test_cli_parser_has_reproducible_measurement_defaults(tmp_path):
    benchmark = benchmark_module()

    args = benchmark.build_parser().parse_args([
        "--mode", "p2p",
        "--runner", "/tmp/hccs_benchmark.so",
        "--output", str(tmp_path / "p2p.json"),
    ])

    assert args.sizes_mib == "1,4,16,64"
    assert args.warmups == 20
    assert args.iterations == 50
    assert args.p2p_sender == 0
    assert args.collective_timeout_seconds == 300


def test_build_report_records_provenance_and_measurements():
    benchmark = benchmark_module()
    measurement = benchmark.summarize_measurement(
        mode="p2p",
        payload_bytes=4 << 20,
        active_senders=1,
        peers_per_sender=1,
        rank_samples_seconds=((0.001, 0.002),),
    )

    report = benchmark.build_report(
        git_commit="abc123",
        device_name="Ascend950PR_9599",
        world_size=2,
        warmups=20,
        iterations=2,
        hcomm_root="/opt/hcomm",
        measurements=[measurement],
    )

    assert report["schema_version"] == 1
    assert report["git_commit"] == "abc123"
    assert report["device"] == "Ascend950PR_9599"
    assert report["world_size"] == 2
    assert report["timing_protocol"] == {
        "timer": "ascend_system_cycle_1ghz",
        "warmups": 20,
        "iterations": 2,
        "rank_aggregation": "maximum_latency",
        "byte_aggregation": "sum_active_senders",
    }
    assert report["hcomm_root"] == "/opt/hcomm"
    assert report["measurements"] == [measurement]
    json.dumps(report, allow_nan=False)


def test_representative_peer_payloads_match_the_shared_workload():
    benchmark = benchmark_module()

    matrix = benchmark.representative_peer_payloads(7552)

    assert len(matrix) == 8
    assert all(len(row) == 8 for row in matrix)
    assert all(matrix[rank][rank] == 0 for rank in range(8))
    assert matrix[0][1] == 41_656_832
    assert matrix[7][6] == 40_267_264
    assert sum(matrix[0]) == 286_152_832
    assert sum(sum(row) for row in matrix) == 2_290_272_384


def test_representative_report_metadata_is_self_describing():
    benchmark = benchmark_module()

    metadata = benchmark.representative_report_metadata(7552)

    assert metadata["workload_fingerprint"] == (
        "d6338cb40be7a4b6d35c4a8c9ee106ea0385751cdd5a20f1ba366baa28324f00"
    )
    assert metadata["dispatch_record_bytes"] == 7552
    assert metadata["aggregate_remote_bytes"] == 2_290_272_384
    assert metadata["peer_payload_bytes"][0][1] == 41_656_832
    assert metadata["workload_spec"] == {
        "world_size": 8,
        "num_tokens": 8192,
        "hidden": 7168,
        "num_topk": 8,
        "num_experts": 256,
        "seed": 0,
        "unbalanced_ratio": 1.0,
        "precise_unbalanced_ratio": False,
        "masked_ratio": 0.0,
    }


def test_device_probe_uses_the_production_transport_facade():
    source = (BENCHMARK_ROOT / "hccs_benchmark.asc").read_text()

    for required in (
        "DeviceTransportFacade",
        "TransportTeam::kWorld",
        "transport.put(",
        "transport.flush(",
        "service::execute(context)",
        "AscendC::GetSystemCycle()",
        "producer_cycles",
        "service_cycles",
        "total_cycles",
    ):
        assert required in source


def test_benchmark_directory_contains_build_and_usage_entrypoints():
    expected = {
        "CMakeLists.txt",
        "README.md",
        "benchmark.py",
        "hccs_benchmark.asc",
        "hccs_benchmark.hpp",
        "hccs_benchmark_main.cpp",
    }

    assert expected <= {path.name for path in BENCHMARK_ROOT.iterdir()}
