from dataclasses import dataclass

from tests.utils.ep_benchmark_manifest import EPModeCase


@dataclass(frozen=True)
class Capability:
    suite: str
    supported: bool
    reason: str = ""


def phase_3e1_acceptance_operations():
    return (
        "cached-bf16-dispatch-sync",
        "cached-bf16-dispatch-async",
        "bf16-combine-sync",
        "bf16-combine-async",
        "previous-event-with-comm-allocation",
        "comm-stream-allocation",
    )


def classify_ascend_case(case: EPModeCase) -> Capability:
    return Capability("performance", True)
