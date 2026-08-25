from tests.benchmark import check_cuda_nvlink as nvlink


def topology_matrix(connections):
    labels = tuple(f"GPU{index}" for index in range(len(connections)))
    lines = ["        " + "  ".join(labels) + "  CPU Affinity"]
    for label, row in zip(labels, connections):
        lines.append(f"{label}    " + "  ".join(row) + "  0-31")
    return "\n".join(lines)


def test_validate_nvlink_clique_accepts_all_nv8_connections():
    matrix = topology_matrix([
        ["X" if source == target else "NV8" for target in range(8)]
        for source in range(8)
    ])

    result = nvlink.validate_nvlink_clique(matrix, expected_gpus=8)

    assert result.gpu_labels == tuple(f"GPU{index}" for index in range(8))
    assert result.error is None


def test_validate_nvlink_clique_rejects_a_non_nvlink_pair():
    connections = [
        ["X" if source == target else "NV8" for target in range(8)]
        for source in range(8)
    ]
    connections[2][5] = "SYS"

    result = nvlink.validate_nvlink_clique(
        topology_matrix(connections),
        expected_gpus=8,
    )

    assert result.error == "GPU2 -> GPU5 uses SYS, expected NV8"


def test_validate_nvlink_clique_rejects_a_narrower_nvlink_pair():
    connections = [
        ["X" if source == target else "NV8" for target in range(8)]
        for source in range(8)
    ]
    connections[1][7] = "NV4"

    result = nvlink.validate_nvlink_clique(
        topology_matrix(connections),
        expected_gpus=8,
    )

    assert result.error == "GPU1 -> GPU7 uses NV4, expected NV8"


def test_validate_nvlink_clique_rejects_the_wrong_gpu_count():
    matrix = topology_matrix([
        ["X" if source == target else "NV8" for target in range(2)]
        for source in range(2)
    ])

    result = nvlink.validate_nvlink_clique(matrix, expected_gpus=8)

    assert result.error == "expected 8 GPUs in topology matrix, found 2"
