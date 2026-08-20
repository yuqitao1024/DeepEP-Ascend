from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from tests.utils.ep_benchmark_manifest import EPModeCase


CORRECTNESS_OPERATIONS = (
    "dispatch",
    "expanded_dispatch",
    "cached_dispatch",
    "cached_expanded_padding",
    "combine",
    "reduced_combine",
)

PERFORMANCE_OPERATIONS = (
    "dispatch",
    "expanded_dispatch",
    "cached_dispatch",
    "combine",
    "reduced_combine",
)


@dataclass(frozen=True)
class DispatchArguments:
    normal: dict[str, Any]
    expanded: dict[str, Any]
    _x: Any
    _topk_weights: Any
    _num_sms: int
    _num_qps: int
    _use_column_major_scale_layout: bool
    _async_with_compute_stream: bool
    _allocate_on_comm_stream: bool

    def cached(self, handle: Any) -> dict[str, Any]:
        return {
            "x": self._x,
            "handle": handle,
            "num_sms": self._num_sms,
            "num_qps": self._num_qps,
            "async_with_compute_stream": self._async_with_compute_stream,
            "allocate_on_comm_stream": self._allocate_on_comm_stream,
        }

    def cached_expanded(self, handle: Any) -> dict[str, Any]:
        return {
            **self.cached(handle),
            "topk_weights": self._topk_weights,
            "do_expand": True,
            "use_tma_aligned_col_major_sf": (
                self._use_column_major_scale_layout
            ),
            "do_zero_padding": True,
        }


def build_dispatch_arguments(
    case: EPModeCase,
    x: Any,
    topk_idx: Any,
    topk_weights: Any,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_sms: int,
    num_qps: int,
) -> DispatchArguments:
    normal = {
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_sms": num_sms,
        "num_qps": num_qps,
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "num_experts": num_experts,
        "expert_alignment": case.expert_alignment,
        "async_with_compute_stream": case.async_with_compute_stream,
        "allocate_on_comm_stream": case.allocate_on_comm_stream,
        "do_handle_copy": case.do_handle_copy,
        "do_cpu_sync": True,
    }
    expanded = {
        **normal,
        "do_expand": True,
        "use_tma_aligned_col_major_sf": case.use_fp8_dispatch,
    }
    return DispatchArguments(
        normal=normal,
        expanded=expanded,
        _x=x,
        _topk_weights=topk_weights,
        _num_sms=num_sms,
        _num_qps=num_qps,
        _use_column_major_scale_layout=case.use_fp8_dispatch,
        _async_with_compute_stream=case.async_with_compute_stream,
        _allocate_on_comm_stream=case.allocate_on_comm_stream,
    )


@dataclass(frozen=True)
class TensorBytes:
    rows: int
    row_bytes: int

    def __post_init__(self) -> None:
        if self.rows < 0 or self.row_bytes < 0:
            raise ValueError("tensor byte dimensions must be nonnegative")


@dataclass(frozen=True)
class DispatchTraffic:
    bytes_per_token: int
    scaleout_bytes: int
    scaleup_bytes: int
    copy_bytes: int


@dataclass(frozen=True)
class CombineTraffic:
    scaleout_bytes: int
    scaleup_bytes: int
    reduction_bytes: int


def calculate_dispatch_traffic(
    tensors: Iterable[TensorBytes],
    num_recv_tokens: int,
    num_scaleup_recv_tokens: int,
    num_scaleout_send_tokens: int,
) -> DispatchTraffic:
    tensors = tuple(tensors)
    if any(value < 0 for value in (
        num_recv_tokens,
        num_scaleup_recv_tokens,
        num_scaleout_send_tokens,
    )):
        raise ValueError("dispatch token counts must be nonnegative")
    if any(tensor.rows != tensors[0].rows for tensor in tensors[1:]):
        raise ValueError("dispatch tensors must have matching row counts")

    bytes_per_token = sum(tensor.row_bytes for tensor in tensors)
    return DispatchTraffic(
        bytes_per_token=bytes_per_token,
        scaleout_bytes=bytes_per_token * num_scaleout_send_tokens,
        scaleup_bytes=bytes_per_token * num_scaleup_recv_tokens,
        copy_bytes=2 * num_recv_tokens * bytes_per_token,
    )


def expanded_dispatch_copy_bytes(
    num_recv_tokens: int,
    num_expanded_tokens: int,
    payload_bytes_per_token: int,
    metadata_bytes_per_token: int,
) -> int:
    values = (
        num_recv_tokens,
        num_expanded_tokens,
        payload_bytes_per_token,
        metadata_bytes_per_token,
    )
    if any(value < 0 for value in values):
        raise ValueError("expanded dispatch byte inputs must be nonnegative")
    return (
        num_recv_tokens
        * (metadata_bytes_per_token + payload_bytes_per_token)
        + num_expanded_tokens * payload_bytes_per_token
    )


def calculate_combine_traffic(
    num_scaleout_tokens: int,
    num_scaleup_tokens: int,
    num_reduction_read_tokens: int,
    bytes_per_token: int,
    bias_bytes: int,
    reduction_write_bytes: int,
) -> CombineTraffic:
    values = (
        num_scaleout_tokens,
        num_scaleup_tokens,
        num_reduction_read_tokens,
        bytes_per_token,
        bias_bytes,
        reduction_write_bytes,
    )
    if any(value < 0 for value in values):
        raise ValueError("combine traffic inputs must be nonnegative")
    return CombineTraffic(
        scaleout_bytes=num_scaleout_tokens * bytes_per_token,
        scaleup_bytes=num_scaleup_tokens * bytes_per_token,
        reduction_bytes=(
            bias_bytes
            + num_reduction_read_tokens * bytes_per_token
            + reduction_write_bytes
        ),
    )


def count_unique_destinations(
    routes: Sequence[Sequence[int]],
    divisor: int = 1,
    ignored_range: Optional[tuple[int, int]] = None,
) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if ignored_range is not None and ignored_range[0] > ignored_range[1]:
        raise ValueError("ignored destination range must be ordered")

    count = 0
    for row in routes:
        destinations = set()
        for route in row:
            if route < 0:
                continue
            destination = route // divisor
            if (
                ignored_range is not None
                and ignored_range[0] <= destination < ignored_range[1]
            ):
                continue
            destinations.add(destination)
        count += len(destinations)
    return count
