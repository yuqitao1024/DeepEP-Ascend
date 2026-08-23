from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from tests.utils.ep_benchmark_manifest import (
    BenchmarkManifest,
    EPModeCase,
    WorkloadSpec,
    build_manifest,
    enumerate_ep_mode_cases,
)


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    world_size: int
    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    seed: int
    warmups: int
    iterations: int
    unbalanced_ratio: float = 1.0
    precise_unbalanced_ratio: bool = False
    masked_ratio: float = 0.0
    allow_multiple_reduction: int = 1
    ascend_num_sms: int = 72
    case_ids: tuple[str, ...] = ()


PROFILES: Mapping[str, BenchmarkProfile] = MappingProxyType({
    "canonical": BenchmarkProfile(
        name="canonical",
        world_size=8,
        num_tokens=8192,
        hidden=7168,
        num_topk=8,
        num_experts=256,
        seed=0,
        warmups=30,
        iterations=30,
    ),
    "representative": BenchmarkProfile(
        name="representative",
        world_size=8,
        num_tokens=8192,
        hidden=7168,
        num_topk=8,
        num_experts=256,
        seed=0,
        warmups=30,
        iterations=30,
        case_ids=(
            "ep-fp8-align128-bias0-hcopy1-prev0-async0-alloc0",
        ),
    ),
    "smoke": BenchmarkProfile(
        name="smoke",
        world_size=8,
        num_tokens=16,
        hidden=128,
        num_topk=2,
        num_experts=8,
        seed=0,
        warmups=1,
        iterations=1,
    ),
})


def profile_manifest(profile: BenchmarkProfile) -> BenchmarkManifest:
    return _build_manifest(WorkloadSpec(
        world_size=profile.world_size,
        num_tokens=profile.num_tokens,
        hidden=profile.hidden,
        num_topk=profile.num_topk,
        num_experts=profile.num_experts,
        seed=profile.seed,
        unbalanced_ratio=profile.unbalanced_ratio,
        precise_unbalanced_ratio=profile.precise_unbalanced_ratio,
        masked_ratio=profile.masked_ratio,
    ))


@lru_cache(maxsize=None)
def _build_manifest(spec: WorkloadSpec) -> BenchmarkManifest:
    return build_manifest(spec)


def profile_cases(profile: BenchmarkProfile) -> tuple[EPModeCase, ...]:
    cases = enumerate_ep_mode_cases()
    if not profile.case_ids:
        return cases
    cases_by_id = {case.case_id: case for case in cases}
    try:
        selected = tuple(cases_by_id[case_id] for case_id in profile.case_ids)
    except KeyError as error:
        raise ValueError(f"unknown profile case: {error.args[0]}") from error
    if len(set(profile.case_ids)) != len(profile.case_ids):
        raise ValueError("profile cases must be unique")
    return selected
