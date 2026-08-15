import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import deep_ep
import deep_ep._C as extension

from api_surface import assert_no_testing_diagnostic_surface


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"deferred argument was inspected: {name}")


def _expect_gate(operation, call):
    try:
        call()
    except NotImplementedError as error:
        message = str(error)
        if not message.startswith(f"DeepEP Ascend backend: {operation} "):
            raise AssertionError(message) from error
    else:
        raise AssertionError(f"{operation} was unexpectedly available")


def _make_buffer(group):
    return deep_ep.ElasticBuffer(
        group,
        num_bytes=2 * 1024 * 1024,
        allow_hybrid_mode=False,
        explicitly_destroy=True,
    )


def run(inject_diagnostic):
    assert_no_testing_diagnostic_surface(extension)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    rank = dist.get_rank(group)

    try:
        for _ in range(3):
            temporary = _make_buffer(group)
            temporary.destroy()
            temporary.destroy()
            dist.barrier(group)

        buffer = _make_buffer(group)
        try:
            _check(buffer.get_logical_domain_size() == (1, 2),
                   "unexpected logical domain")
            _check(buffer.get_physical_domain_size() == (1, 2),
                   "unexpected physical domain")
            _check((buffer.scaleout_rank_idx, buffer.scaleup_rank_idx) ==
                   (0, rank), "unexpected rank mapping")

            for generation in range(1, 101):
                buffer.barrier(
                    with_cpu_sync=generation % 17 == 0,
                    sequential=True,
                )

            try:
                buffer.barrier(sequential=False)
            except RuntimeError as error:
                _check("requires sequential=True" in str(error),
                       "unexpected non-sequential barrier error")
            else:
                raise AssertionError("non-sequential barrier was accepted")

            poison = _Poison()
            _expect_gate("dispatch", lambda: buffer.dispatch(poison))
            _expect_gate("combine", lambda: buffer.combine(poison, poison))

            if inject_diagnostic:
                os.environ["DEEP_EP_ASCEND_TEST_DIAGNOSTIC"] = \
                    "completion_timeout"
                try:
                    buffer.barrier()
                except RuntimeError as error:
                    message = str(error)
                    expected_fields = (
                        f"barrier failed on rank {rank}",
                        "backend error 0",
                        "completion_timeout",
                        "command_index=0",
                        "opcode=6",
                        f"peer={rank}",
                        "channel=0",
                        "generation=101",
                    )
                    for expected in expected_fields:
                        _check(expected in message,
                               f"diagnostic omitted {expected!r}: {message}")
                else:
                    raise AssertionError("injected diagnostic was not reported")
                finally:
                    os.environ.pop("DEEP_EP_ASCEND_TEST_DIAGNOSTIC", None)
        finally:
            buffer.destroy()

        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(
            gathered,
            {"rank": rank, "barriers": 100, "diagnostic": inject_diagnostic},
            group=group,
        )
        if rank == 0:
            print(f"Phase 2E two-rank barrier validation passed: {gathered}",
                  flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-diagnostic", action="store_true")
    args = parser.parse_args()
    run(args.inject_diagnostic)


if __name__ == "__main__":
    main()
