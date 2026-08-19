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
            f"DeepEP Ascend backend: {operation} is unavailable until the Ascend "
            "device transport is implemented")


def get_comm_handle(group, force_new_comm: bool = False):
    if is_cuda():
        from .utils.comm import get_nccl_comm_handle
        return get_nccl_comm_handle(group, force_new_comm=force_new_comm)
    from .utils.hccl import get_hccl_comm_handle
    return get_hccl_comm_handle(group)


def comm_handle_value(handle: Optional[object]) -> int:
    if handle is None:
        return 0
    if isinstance(handle, int):
        return handle
    return handle.get()


def capture_event():
    return _C.EventHandle()


def ascend_current_stream_is_capturing() -> bool:
    if is_cuda():
        return False
    query = getattr(torch.npu, "is_current_stream_capturing", None)
    return bool(query()) if callable(query) else False


def unwrap_event(event):
    return None if event is None else getattr(event, "event", event)


def validate_device_type(tensor, operation: str) -> None:
    expected = "cuda" if is_cuda() else "npu"
    if tensor.device.type != expected:
        platform_name = "CUDA" if is_cuda() else "Ascend"
        device_requirement = "a CUDA" if is_cuda() else "an NPU"
        raise ValueError(
            f"DeepEP {platform_name} backend: {operation} requires "
            f"{device_requirement} tensor")


def synchronize() -> None:
    if is_cuda():
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()


def wrap_stream(stream):
    if is_cuda():
        return torch.cuda.Stream(stream_id=stream.stream_id,
                                 device_index=stream.device_index,
                                 device_type=stream.device_type)
    return torch.npu.Stream(stream_id=stream.stream_id,
                            device_index=stream.device_index,
                            device_type=stream.device_type)
