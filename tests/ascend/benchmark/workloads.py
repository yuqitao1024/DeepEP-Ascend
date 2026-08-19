from dataclasses import dataclass

from tests.utils.ep_benchmark_manifest import EPModeCase, case_suite


@dataclass(frozen=True)
class Capability:
    suite: str
    supported: bool
    reason: str = ""


def classify_ascend_case(case: EPModeCase) -> Capability:
    suite = case_suite(case)
    if suite == "functional":
        if case.with_previous_event:
            return Capability(suite, False, "event_chaining_deferred")
        if case.async_with_compute_stream:
            return Capability(suite, False, "async_overlap_deferred")
        return Capability(suite, False, "comm_stream_allocation_deferred")
    return Capability(suite, True)
