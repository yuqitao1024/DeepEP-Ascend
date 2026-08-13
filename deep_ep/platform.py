from typing import Optional

import torch

# noinspection PyUnresolvedReferences
import deep_ep._C as _C


COMPILED_PLATFORM = _C.get_platform()
if COMPILED_PLATFORM not in ("cuda", "ascend"):
    raise RuntimeError(f"Unsupported compiled DeepEP platform: {COMPILED_PLATFORM}")


def is_cuda() -> bool:
    return COMPILED_PLATFORM == "cuda"


def require_cuda(operation: str) -> None:
    if not is_cuda():
        raise NotImplementedError(
            f"DeepEP Ascend backend: {operation} is not implemented in phase 1")


def get_comm_handle(group):
    if not is_cuda():
        return None
    from .utils.comm import get_nccl_comm_handle
    return get_nccl_comm_handle(group)


def comm_handle_value(handle: Optional[object]) -> int:
    return 0 if handle is None else handle.get()


def capture_event():
    return _C.EventHandle()


def unwrap_event(event):
    return None if event is None else getattr(event, "event", event)


def validate_device_type(tensor, operation: str) -> None:
    require_cuda(operation)
    if tensor.device.type != "cuda":
        raise ValueError(
            f"DeepEP CUDA backend: {operation} requires a CUDA tensor")


def synchronize() -> None:
    if is_cuda():
        torch.cuda.synchronize()


def wrap_stream(stream):
    require_cuda("get_comm_stream")
    return torch.cuda.Stream(stream_id=stream.stream_id,
                             device_index=stream.device_index,
                             device_type=stream.device_type)
