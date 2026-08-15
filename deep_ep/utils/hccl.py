import torch


def get_hccl_comm_handle(group) -> int:
    local_rank = int(group.rank())
    candidates = [group]
    get_backend = getattr(group, "_get_backend", None)
    if get_backend is not None:
        candidates.append(get_backend(torch.device("npu", local_rank)))

    for backend in candidates:
        get_communicator = getattr(backend, "get_hccl_comm", None)
        if get_communicator is None:
            continue
        communicator = get_communicator(local_rank)
        if isinstance(communicator, (tuple, list)):
            if len(communicator) != 1:
                raise RuntimeError(
                    f"expected one HCCL communicator, got {communicator!r}")
            communicator = communicator[0]
        handle = int(communicator or 0)
        if handle == 0:
            raise RuntimeError(
                "ProcessGroupHCCL returned a null HCCL communicator")
        return handle

    raise RuntimeError("ProcessGroupHCCL.get_hccl_comm is unavailable")
