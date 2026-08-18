from dataclasses import dataclass

from tests.utils.ep_benchmark_manifest import EPModeCase


@dataclass(frozen=True)
class Capability:
    supported: bool
    reason: str = ""


def classify_ascend_case(case: EPModeCase) -> Capability:
    if case.use_fp8_dispatch:
        return Capability(False, "fp8_runtime_deferred")
    if case.with_previous_event:
        return Capability(False, "event_chaining_deferred")
    if case.async_with_compute_stream:
        return Capability(False, "async_overlap_deferred")
    if case.allocate_on_comm_stream:
        return Capability(False, "comm_stream_allocation_deferred")
    return Capability(True)
