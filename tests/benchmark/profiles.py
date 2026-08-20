from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from tests.utils.ep_benchmark_manifest import BenchmarkManifest, WorkloadSpec, build_manifest


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


PROFILES: Mapping[str, BenchmarkProfile] = MappingProxyType({
    "canonical": BenchmarkProfile(
        name="canonical",
        world_size=8,
        num_tokens=4096,
        hidden=7168,
        num_topk=6,
        num_experts=256,
        seed=0,
        warmups=30,
        iterations=30,
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
    return build_manifest(WorkloadSpec(
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
