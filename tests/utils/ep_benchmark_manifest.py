import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EPModeCase:
    do_handle_copy: bool
    expert_alignment: int
    use_fp8_dispatch: bool
    num_bias: int
    with_previous_event: bool
    async_with_compute_stream: bool
    allocate_on_comm_stream: bool

    @property
    def dtype_name(self) -> str:
        return "fp8" if self.use_fp8_dispatch else "bf16"

    @property
    def case_id(self) -> str:
        return (
            f"ep-{self.dtype_name}-align{self.expert_alignment}"
            f"-bias{self.num_bias}-hcopy{int(self.do_handle_copy)}"
            f"-prev{int(self.with_previous_event)}"
            f"-async{int(self.async_with_compute_stream)}"
            f"-alloc{int(self.allocate_on_comm_stream)}"
        )


def enumerate_ep_mode_cases() -> tuple[EPModeCase, ...]:
    cases = []
    for do_handle_copy in (True, False):
        for expert_alignment in (128, 1):
            for use_fp8_dispatch in (True, False):
                for num_bias in (0, 1, 2):
                    for with_previous_event in (False, True):
                        for async_with_compute_stream in (False, True):
                            allocations = (
                                (True,)
                                if with_previous_event
                                else (False, True)
                            )
                            for allocate_on_comm_stream in allocations:
                                cases.append(
                                    EPModeCase(
                                        do_handle_copy=do_handle_copy,
                                        expert_alignment=expert_alignment,
                                        use_fp8_dispatch=use_fp8_dispatch,
                                        num_bias=num_bias,
                                        with_previous_event=with_previous_event,
                                        async_with_compute_stream=(
                                            async_with_compute_stream
                                        ),
                                        allocate_on_comm_stream=(
                                            allocate_on_comm_stream
                                        ),
                                    )
                                )
    return tuple(cases)


def case_suite(case: EPModeCase) -> str:
    if (
        case.with_previous_event
        or case.async_with_compute_stream
        or case.allocate_on_comm_stream
    ):
        return "functional"
    return "performance"


@dataclass(frozen=True)
class WorkloadSpec:
    world_size: int
    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    seed: int = 0
    unbalanced_ratio: float = 1.0
    precise_unbalanced_ratio: bool = False
    masked_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.world_size < 2:
            raise ValueError("world_size must be at least two")
        if self.num_tokens <= 0 or self.hidden <= 0:
            raise ValueError("num_tokens and hidden must be positive")
        if self.num_experts <= 0 or self.num_experts % self.world_size != 0:
            raise ValueError("num_experts must be positive and rank-partitioned")
        if not 0 < self.num_topk <= self.num_experts:
            raise ValueError("num_topk must be in [1, num_experts]")
        if self.unbalanced_ratio < 1.0:
            raise ValueError("unbalanced_ratio must be at least one")
        if not 0.0 <= self.masked_ratio <= 1.0:
            raise ValueError("masked_ratio must be in [0, 1]")


@dataclass(frozen=True)
class RankWorkload:
    rank: int
    num_tokens: int
    topk_idx: tuple[tuple[int, ...], ...]
    topk_weights: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class BenchmarkManifest:
    schema_version: int
    generator_version: int
    spec: WorkloadSpec
    ranks: tuple[RankWorkload, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "spec": asdict(self.spec),
            "ranks": [asdict(rank) for rank in self.ranks],
            "fingerprint": self.fingerprint,
        }


def _rank_workload(spec: WorkloadSpec, rank: int) -> RankWorkload:
    rng = random.Random(spec.seed + rank)
    num_tokens = max(1, spec.num_tokens - rank)
    experts_per_rank = spec.num_experts // spec.world_size
    topk_indices = []
    topk_weights = []

    for _ in range(num_tokens):
        scores = []
        for expert in range(spec.num_experts):
            score = rng.random()
            if expert < experts_per_rank:
                score *= spec.unbalanced_ratio
            scores.append(score)

        selected = sorted(
            range(spec.num_experts),
            key=lambda expert: (scores[expert], expert),
            reverse=True,
        )[:spec.num_topk]
        rng.shuffle(selected)
        weights = [scores[expert] for expert in selected]

        for lane in range(spec.num_topk):
            if rng.random() < spec.masked_ratio:
                selected[lane] = -1
                weights[lane] = 0.0

        topk_indices.append(tuple(selected))
        topk_weights.append(tuple(weights))

    return RankWorkload(
        rank=rank,
        num_tokens=num_tokens,
        topk_idx=tuple(topk_indices),
        topk_weights=tuple(topk_weights),
    )


def _manifest_payload(
    spec: WorkloadSpec,
    ranks: tuple[RankWorkload, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator_version": 1,
        "spec": asdict(spec),
        "ranks": [asdict(rank) for rank in ranks],
    }


def manifest_fingerprint(
    spec: WorkloadSpec,
    ranks: tuple[RankWorkload, ...],
) -> str:
    encoded = json.dumps(
        _manifest_payload(spec, ranks),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(spec: WorkloadSpec) -> BenchmarkManifest:
    ranks = tuple(_rank_workload(spec, rank) for rank in range(spec.world_size))
    return BenchmarkManifest(
        schema_version=1,
        generator_version=1,
        spec=spec,
        ranks=ranks,
        fingerprint=manifest_fingerprint(spec, ranks),
    )


def write_manifest(path: str | Path, manifest: BenchmarkManifest) -> None:
    Path(path).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> BenchmarkManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    if payload.get("generator_version") != 1:
        raise ValueError("unsupported manifest generator_version")

    spec = WorkloadSpec(**payload["spec"])
    ranks = tuple(
        RankWorkload(
            rank=rank["rank"],
            num_tokens=rank["num_tokens"],
            topk_idx=tuple(tuple(row) for row in rank["topk_idx"]),
            topk_weights=tuple(
                tuple(row) for row in rank["topk_weights"]
            ),
        )
        for rank in payload["ranks"]
    )
    fingerprint = manifest_fingerprint(spec, ranks)
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("manifest fingerprint does not match its contents")
    return BenchmarkManifest(
        schema_version=payload["schema_version"],
        generator_version=payload["generator_version"],
        spec=spec,
        ranks=ranks,
        fingerprint=fingerprint,
    )
