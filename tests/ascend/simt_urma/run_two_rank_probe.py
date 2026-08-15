#!/usr/bin/env python3

import argparse
import ctypes
import datetime
import json
import os
import pathlib
import sys


TIMEOUT_MS = 30_000
ORDERING_ITERATIONS = 1_000


def runtime_contract():
    phased = {"phases": ["producer", "service", "consumer"]}
    cases = {
        "put": dict(phased),
        "put-value64": dict(phased),
        "faa64": dict(phased),
        "signal": dict(phased),
        "flush": dict(phased),
        "payload-signal-order": {
            **phased, "iterations": ORDERING_ITERATIONS,
        },
        "barrier-repeat": {**phased, "iterations": 2},
        "queue-wrap": {**phased, "requires_sq_wrap": True},
        "phase-boundary": dict(phased),
        "teardown": {"phases": ["host"]},
    }
    return {
        "cases": cases,
        "communicator": "backend.get_hccl_comm(local_rank)",
        "runner_loading": "in-process-shared-library",
        "timeout_ms": TIMEOUT_MS,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--runner")
    parser.add_argument("--cases")
    return parser.parse_args()


def resolve_runner(path):
    candidate = pathlib.Path(path).resolve()
    candidates = [candidate]
    if candidate.suffix != ".so":
        candidates.append(candidate.with_suffix(".so"))
    for item in candidates:
        if item.is_file():
            return item
    raise FileNotFoundError(f"runtime runner not found: {path}")


def load_library(path):
    return ctypes.CDLL(str(resolve_runner(path)))


def load_runner(path):
    library = load_library(path)
    run_case = library.deep_ep_ascend_urma_run_case
    run_case.argtypes = [
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    run_case.restype = ctypes.c_int
    return run_case


def run_local_smoke(path):
    import torch
    import torch_npu

    del torch_npu
    torch.npu.set_device(0)
    library = load_library(path)
    smoke = library.deep_ep_ascend_urma_run_local_phase_boundary
    smoke.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    smoke.restype = ctypes.c_int
    error = ctypes.create_string_buffer(2048)
    status = smoke(error, len(error))
    if status != 0:
        raise RuntimeError(error.value.decode(errors="replace"))
    print("case=phase-boundary device=0 local-smoke PASS", flush=True)
    return 0


def hccl_communicator(process_group, local_rank, torch):
    candidates = [process_group]
    get_backend = getattr(process_group, "_get_backend", None)
    if get_backend is not None:
        candidates.append(get_backend(torch.device("npu", local_rank)))
    for backend in candidates:
        get_comm = getattr(backend, "get_hccl_comm", None)
        if get_comm is None:
            continue
        communicator = get_comm(local_rank)
        if isinstance(communicator, (tuple, list)):
            if len(communicator) != 1:
                raise RuntimeError(
                    f"expected one HCCL communicator, got {communicator!r}")
            communicator = communicator[0]
        return int(communicator)
    raise RuntimeError("ProcessGroupHCCL.get_hccl_comm is unavailable")


def selected_cases(value):
    known = runtime_contract()["cases"]
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in known]
    if not names or unknown:
        raise ValueError(f"invalid cases: {unknown or names}")
    teardown_count = names.count("teardown")
    if teardown_count == 0:
        raise ValueError("teardown case is required")
    if teardown_count != 1 or names[-1] != "teardown":
        raise ValueError("teardown must be the final selected case")
    return names


def run_case_sequence(
        dist, run_case, communicator, rank, world_size, cases, contract):
    teardown_completed = False
    primary_error = None

    def invoke_case(case_name):
        iterations = contract[case_name].get("iterations", 1)
        error = ctypes.create_string_buffer(2048)
        status = run_case(
            communicator, rank, world_size, case_name.encode(),
            iterations, error, len(error))
        local_result = {
            "rank": rank,
            "status": status,
            "error": error.value.decode(errors="replace"),
        }
        results = [None] * world_size
        dist.all_gather_object(results, local_result)
        failures = [result for result in results if result["status"] != 0]
        if failures:
            raise RuntimeError(f"case {case_name} failed: {failures}")
        if rank == 0:
            print(f"case={case_name} ranks=2 diagnostics=kNone PASS",
                  flush=True)

    def execute_case(case_name):
        dist.barrier()
        invoke_case(case_name)

    def cleanup_teardown():
        cleanup_error = None
        try:
            dist.barrier()
        except BaseException as error:
            cleanup_error = error
        try:
            invoke_case("teardown")
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error

    try:
        for case_name in cases:
            execute_case(case_name)
            teardown_completed = case_name == "teardown"
        return 0
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if not teardown_completed:
            try:
                cleanup_teardown()
            except BaseException as teardown_error:
                if primary_error is None:
                    raise
                print(
                    f"teardown after failed runtime case also failed: "
                    f"{teardown_error}", file=sys.stderr, flush=True)


def run_runtime(args, cases):
    import torch
    import torch.distributed as dist
    import torch_npu

    del torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"runtime probe requires two ranks, got {world_size}")

    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl", timeout=datetime.timedelta(milliseconds=TIMEOUT_MS * 4))
    try:
        process_group = dist.distributed_c10d._get_default_group()
        communicator = hccl_communicator(process_group, local_rank, torch)
        run_case = load_runner(args.runner)
        return run_case_sequence(
            dist, run_case, communicator, rank, world_size, cases,
            runtime_contract()["cases"])
    finally:
        dist.destroy_process_group()


def main():
    args = parse_args()
    if args.describe:
        print(json.dumps(runtime_contract(), sort_keys=True))
        return 0
    if args.local_smoke:
        if not args.runner:
            raise SystemExit("--runner is required for --local-smoke")
        return run_local_smoke(args.runner)
    if not args.runner or not args.cases:
        raise SystemExit("--runner and --cases are required for runtime validation")
    return run_runtime(args, selected_cases(args.cases))


if __name__ == "__main__":
    raise SystemExit(main())
