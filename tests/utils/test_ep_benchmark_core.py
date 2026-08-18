import ast
from pathlib import Path

from tests.utils.ep_benchmark_core import (
    CORRECTNESS_OPERATIONS,
    PERFORMANCE_OPERATIONS,
    TensorBytes,
    calculate_combine_traffic,
    calculate_dispatch_traffic,
    count_unique_destinations,
    expanded_dispatch_copy_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
CUDA_BENCHMARK = ROOT / "tests/elastic/test_ep.py"


def test_operation_sets_match_upstream_sequence():
    assert CORRECTNESS_OPERATIONS == (
        "dispatch",
        "expanded_dispatch",
        "cached_dispatch",
        "cached_expanded_padding",
        "combine",
        "reduced_combine",
    )
    assert PERFORMANCE_OPERATIONS == (
        "dispatch",
        "expanded_dispatch",
        "cached_dispatch",
        "combine",
        "reduced_combine",
    )


def test_dispatch_traffic_counts_payload_index_and_weight_rows():
    traffic = calculate_dispatch_traffic(
        tensors=(
            TensorBytes(rows=4, row_bytes=16),
            TensorBytes(rows=4, row_bytes=16),
            TensorBytes(rows=4, row_bytes=8),
        ),
        num_recv_tokens=4,
        num_scaleup_recv_tokens=3,
        num_scaleout_send_tokens=0,
    )

    assert traffic.bytes_per_token == 40
    assert traffic.scaleup_bytes == 120
    assert traffic.scaleout_bytes == 0
    assert traffic.copy_bytes == 320


def test_expanded_dispatch_copy_bytes_adds_metadata_and_expanded_rows():
    assert expanded_dispatch_copy_bytes(
        num_recv_tokens=4,
        num_expanded_tokens=7,
        payload_bytes_per_token=40,
        metadata_bytes_per_token=24,
    ) == 536


def test_combine_traffic_counts_reduction_components():
    traffic = calculate_combine_traffic(
        num_scaleout_tokens=2,
        num_scaleup_tokens=3,
        num_reduction_read_tokens=4,
        bytes_per_token=20,
        bias_bytes=16,
        reduction_write_bytes=32,
    )

    assert traffic.scaleout_bytes == 40
    assert traffic.scaleup_bytes == 60
    assert traffic.reduction_bytes == 128


def test_unique_destinations_are_deduplicated_per_token():
    routes = ((0, 1, -1), (2, 2, 3), (-1, -1, -1))

    assert count_unique_destinations(routes) == 4
    assert count_unique_destinations(routes, divisor=2) == 2
    assert count_unique_destinations(routes, ignored_range=(0, 2)) == 2


def test_cuda_benchmark_imports_shared_case_and_traffic_logic():
    tree = ast.parse(CUDA_BENCHMARK.read_text(), filename=str(CUDA_BENCHMARK))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {
            "tests.utils.ep_benchmark_core",
            "tests.utils.ep_benchmark_manifest",
        }
        for alias in node.names
    }

    assert {
        "enumerate_ep_mode_cases",
        "calculate_dispatch_traffic",
        "calculate_combine_traffic",
    } <= imports
