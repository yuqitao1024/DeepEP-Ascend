import json

import pytest

from tests.utils.ep_benchmark_manifest import (
    WorkloadSpec,
    build_manifest,
    load_manifest,
    write_manifest,
)


def test_manifest_is_deterministic_and_rank_asymmetric():
    spec = WorkloadSpec(
        world_size=8,
        num_tokens=64,
        hidden=128,
        num_topk=6,
        num_experts=256,
        seed=7,
        unbalanced_ratio=1.0,
        precise_unbalanced_ratio=False,
        masked_ratio=0.0,
    )

    first = build_manifest(spec)
    second = build_manifest(spec)

    assert first.fingerprint == second.fingerprint
    assert first.ranks[0].num_tokens == 64
    assert first.ranks[7].num_tokens == 57
    assert first.ranks[0].topk_idx == second.ranks[0].topk_idx
    assert first.ranks[0].topk_weights == second.ranks[0].topk_weights


def test_manifest_changes_when_routing_seed_changes():
    base = WorkloadSpec(
        world_size=2,
        num_tokens=16,
        hidden=32,
        num_topk=2,
        num_experts=4,
        seed=1,
    )
    changed = WorkloadSpec(
        world_size=2,
        num_tokens=16,
        hidden=32,
        num_topk=2,
        num_experts=4,
        seed=2,
    )

    assert build_manifest(base).fingerprint != build_manifest(changed).fingerprint


def test_manifest_masking_preserves_weights_and_invalid_routes():
    spec = WorkloadSpec(
        world_size=2,
        num_tokens=8,
        hidden=32,
        num_topk=2,
        num_experts=4,
        seed=3,
        masked_ratio=1.0,
    )

    manifest = build_manifest(spec)

    assert all(
        expert == -1
        for rank in manifest.ranks
        for row in rank.topk_idx
        for expert in row
    )
    assert all(
        weight == 0.0
        for rank in manifest.ranks
        for row in rank.topk_weights
        for weight in row
    )


def test_manifest_json_round_trip_revalidates_fingerprint(tmp_path):
    manifest = build_manifest(WorkloadSpec(
        world_size=2,
        num_tokens=8,
        hidden=32,
        num_topk=2,
        num_experts=4,
        seed=11,
    ))
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)
    loaded = load_manifest(path)

    assert loaded == manifest

    payload = json.loads(path.read_text())
    payload["ranks"][0]["topk_idx"][0][0] = -1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        load_manifest(path)
