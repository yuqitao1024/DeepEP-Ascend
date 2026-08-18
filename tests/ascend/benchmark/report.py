import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from tests.utils.ep_benchmark_manifest import EPModeCase


SCHEMA_VERSION = 1
FORMULA_VERSION = 1


@dataclass
class BenchmarkReport:
    platform: str
    workload_fingerprint: str
    world_size: int
    cases: list[dict[str, Any]]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION
    formula_version: int = FORMULA_VERSION
    git_commit: str = ""
    device: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    timing_protocol: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def empty_for_cases(
        cls,
        platform: str,
        cases: Iterable[EPModeCase],
        classify: Callable[[EPModeCase], Any],
        workload_fingerprint: str,
        world_size: int,
    ) -> "BenchmarkReport":
        records = []
        for case in cases:
            capability = classify(case)
            records.append({
                "case_id": case.case_id,
                "mode": asdict(case),
                "status": "pending" if capability.supported else "unsupported",
                "reason": capability.reason,
                "operations": [],
            })
        return cls(
            platform=platform,
            workload_fingerprint=workload_fingerprint,
            world_size=world_size,
            cases=records,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formula_version": self.formula_version,
            "generated_at": self.generated_at,
            "git_commit": self.git_commit,
            "platform": self.platform,
            "device": self.device,
            "world_size": self.world_size,
            "workload": self.workload,
            "workload_fingerprint": self.workload_fingerprint,
            "timing_protocol": self.timing_protocol,
            "case_summary": {
                "supported": sum(
                    case["status"] != "unsupported" for case in self.cases
                ),
                "unsupported": sum(
                    case["status"] == "unsupported" for case in self.cases
                ),
            },
            "cases": self.cases,
            "failures": self.failures,
        }


def write_report_atomic(path: Path | str, report: BenchmarkReport) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(report.to_dict(), temporary, sort_keys=True, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_comparable(
    left: BenchmarkReport,
    right: BenchmarkReport,
) -> None:
    identity_fields = (
        "schema_version",
        "formula_version",
        "workload_fingerprint",
        "world_size",
    )
    mismatches = [
        field_name
        for field_name in identity_fields
        if getattr(left, field_name) != getattr(right, field_name)
    ]
    if mismatches:
        raise ValueError(
            "benchmark reports are not comparable: " + ", ".join(mismatches)
        )
