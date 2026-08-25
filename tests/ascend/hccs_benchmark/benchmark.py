#!/usr/bin/env python3
"""Run and report the Ascend HCCS transport benchmarks."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timedelta, timezone
from statistics import fmean
from typing import Iterable, Sequence

from tests.utils.ep_benchmark_manifest import WorkloadSpec, build_manifest


MIB = 1 << 20


def representative_manifest():
    return build_manifest(WorkloadSpec(
        world_size=8,
        num_tokens=8192,
        hidden=7168,
        num_topk=8,
        num_experts=256,
    ))


def representative_peer_payloads(
    dispatch_record_bytes: int,
) -> tuple[tuple[int, ...], ...]:
    """Return remote FP8 dispatch bytes for the shared EP8 workload."""
    if dispatch_record_bytes <= 0:
        raise ValueError("dispatch record stride must be positive")
    manifest = representative_manifest()
    spec = manifest.spec
    experts_per_rank = spec.num_experts // spec.world_size
    matrix = []
    for source in manifest.ranks:
        destination_counts = [0] * spec.world_size
        for routes in source.topk_idx:
            destinations = {
                expert // experts_per_rank for expert in routes if expert >= 0
            }
            for destination in destinations:
                if destination != source.rank:
                    destination_counts[destination] += 1
        matrix.append(tuple(
            count * dispatch_record_bytes for count in destination_counts
        ))
    return tuple(matrix)


def representative_report_metadata(
    dispatch_record_bytes: int,
) -> dict[str, object]:
    manifest = representative_manifest()
    matrix = representative_peer_payloads(dispatch_record_bytes)
    return {
        "workload_fingerprint": manifest.fingerprint,
        "workload_spec": asdict(manifest.spec),
        "dispatch_record_bytes": dispatch_record_bytes,
        "aggregate_remote_bytes": sum(sum(row) for row in matrix),
        "peer_payload_bytes": matrix,
    }


def peer_payloads(
    mode: str, *, rank: int, world_size: int, payload_bytes: int,
    p2p_sender: int = 0, transport_record_bytes: int = 0,
) -> tuple[int, ...]:
    """Return the bytes sent by one rank to every world peer."""
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in the world")
    if mode == "p2p":
        if world_size != 2:
            raise ValueError("p2p mode requires exactly two ranks")
        if p2p_sender not in (0, 1):
            raise ValueError("p2p sender must be rank zero or one")
        receiver = 1 - p2p_sender
        return tuple(
            payload_bytes if rank == p2p_sender and peer == receiver else 0
            for peer in range(world_size)
        )
    if mode == "ring":
        if world_size < 2:
            raise ValueError("ring mode requires at least two ranks")
        destination = (rank + 1) % world_size
        return tuple(
            payload_bytes if peer == destination else 0
            for peer in range(world_size)
        )
    if mode == "all-to-all":
        if world_size < 2:
            raise ValueError("all-to-all mode requires at least two ranks")
        return tuple(
            0 if peer == rank else payload_bytes for peer in range(world_size)
        )
    if mode == "transport-only":
        if world_size != 8:
            raise ValueError("transport-only mode requires exactly eight ranks")
        if transport_record_bytes <= 0:
            raise ValueError(
                "transport-only mode requires a production record stride")
        return representative_peer_payloads(transport_record_bytes)[rank]
    raise ValueError(f"unsupported benchmark mode: {mode}")


def expected_payloads(
    sender_rows: Sequence[Sequence[int]], *, destination_rank: int
) -> tuple[int, ...]:
    """Return bytes expected from every sender at one destination."""
    if not sender_rows or not 0 <= destination_rank < len(sender_rows):
        raise ValueError("invalid sender matrix or destination rank")
    if any(len(row) != len(sender_rows) for row in sender_rows):
        raise ValueError("sender matrix must be square")
    return tuple(row[destination_rank] for row in sender_rows)


def parse_size_mib_list(value: str) -> tuple[int, ...]:
    """Parse a unique comma-separated list of positive integer MiB sizes."""
    if not value:
        raise ValueError("payload size list must not be empty")
    try:
        sizes_mib = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("payload sizes must be integer MiB values") from error
    if any(size <= 0 for size in sizes_mib):
        raise ValueError("payload sizes must be positive")
    if len(set(sizes_mib)) != len(sizes_mib):
        raise ValueError("payload sizes must be unique")
    return tuple(size * MIB for size in sizes_mib)


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_measurement(
    *,
    mode: str,
    payload_bytes: int,
    active_senders: int,
    peers_per_sender: int,
    rank_samples_seconds: Sequence[Sequence[float]],
    aggregate_bytes: int | None = None,
    phase_rank_samples_seconds: dict[
        str, Sequence[Sequence[float]]
    ] | None = None,
) -> dict[str, object]:
    """Aggregate every iteration with the slowest participating rank."""
    if not rank_samples_seconds:
        raise ValueError("rank samples must not be empty")
    sample_count = len(rank_samples_seconds[0])
    if sample_count == 0 or any(
        len(samples) != sample_count for samples in rank_samples_seconds
    ):
        raise ValueError("rank sample counts must be equal and nonzero")
    samples = [
        max(rank_samples[index] for rank_samples in rank_samples_seconds)
        for index in range(sample_count)
    ]
    ordered = sorted(samples)
    mean_seconds = fmean(samples)
    bytes_per_iteration = (
        payload_bytes * active_senders * peers_per_sender
        if aggregate_bytes is None else aggregate_bytes
    )
    result = {
        "mode": mode,
        "payload_bytes": payload_bytes,
        "active_senders": active_senders,
        "peers_per_sender": peers_per_sender,
        "bytes_per_iteration": bytes_per_iteration,
        "samples_seconds": samples,
        "mean_seconds": mean_seconds,
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "logical_gbps": bytes_per_iteration / mean_seconds / 1e9,
    }
    if phase_rank_samples_seconds:
        phases = {}
        for name, rank_samples in phase_rank_samples_seconds.items():
            if len(rank_samples) != len(rank_samples_seconds) or any(
                len(samples) != sample_count for samples in rank_samples
            ):
                raise ValueError("phase samples must match total samples")
            phase_samples = [
                max(samples[index] for samples in rank_samples)
                for index in range(sample_count)
            ]
            phase_ordered = sorted(phase_samples)
            phases[name] = {
                "samples_seconds": phase_samples,
                "mean_seconds": fmean(phase_samples),
                "p50_seconds": _percentile(phase_ordered, 0.50),
                "p95_seconds": _percentile(phase_ordered, 0.95),
            }
        result["phases"] = phases
    return result


def build_report(
    *,
    git_commit: str,
    device_name: str,
    world_size: int,
    warmups: int,
    iterations: int,
    hcomm_root: str,
    measurements: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Build the stable machine-readable benchmark report."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "device": device_name,
        "world_size": world_size,
        "hcomm_root": hcomm_root,
        "timing_protocol": {
            "timer": "ascend_system_cycle_1ghz",
            "warmups": warmups,
            "iterations": iterations,
            "rank_aggregation": "maximum_latency",
            "byte_aggregation": "sum_active_senders",
        },
        "measurements": list(measurements),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure HCCS and DeepEP staged transport throughput")
    parser.add_argument(
        "--mode", required=True,
        choices=("p2p", "ring", "all-to-all", "transport-only"))
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes-mib", default="1,4,16,64")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--p2p-sender", type=int, choices=(0, 1), default=0)
    parser.add_argument("--collective-timeout-seconds", type=int, default=300)
    return parser


class NativeCycles(ctypes.Structure):
    _fields_ = [
        ("producer_cycles", ctypes.c_uint64),
        ("service_cycles", ctypes.c_uint64),
        ("total_cycles", ctypes.c_uint64),
    ]


class NativeBenchmark:
    def __init__(self, path: str):
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        self.create = self.library.deep_ep_hccs_benchmark_create
        self.create.argtypes = [
            ctypes.c_int64, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.create.restype = ctypes.c_void_p
        self.reset = self.library.deep_ep_hccs_benchmark_reset
        self.reset.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.reset.restype = ctypes.c_int
        self.run = self.library.deep_ep_hccs_benchmark_run
        self.run.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64, ctypes.POINTER(NativeCycles),
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.run.restype = ctypes.c_int
        self.verify = self.library.deep_ep_hccs_benchmark_verify
        self.verify.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.verify.restype = ctypes.c_int
        self.destroy = self.library.deep_ep_hccs_benchmark_destroy
        self.destroy.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.destroy.restype = ctypes.c_int
        self.representative_record_bytes = (
            self.library.deep_ep_hccs_benchmark_representative_record_bytes)
        self.representative_record_bytes.argtypes = []
        self.representative_record_bytes.restype = ctypes.c_uint64


def _native_error(function, *arguments):
    error = ctypes.create_string_buffer(2048)
    status = function(*arguments, error, len(error))
    return status, error.value.decode(errors="replace")


def _collective_status(dist, rank: int, status: int, error: str, label: str):
    local = {"rank": rank, "status": status, "error": error}
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    failures = [row for row in gathered if row["status"] != 0]
    if failures:
        raise RuntimeError(f"{label} failed: {failures}")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _matrix_for_point(
    mode: str, world_size: int, payload_bytes: int, p2p_sender: int,
    transport_record_bytes: int,
) -> tuple[tuple[int, ...], ...]:
    if mode == "transport-only":
        return representative_peer_payloads(transport_record_bytes)
    return tuple(
        peer_payloads(
            mode, rank=rank, world_size=world_size,
            payload_bytes=payload_bytes, p2p_sender=p2p_sender,
            transport_record_bytes=transport_record_bytes)
        for rank in range(world_size)
    )


def _run_distributed(args) -> int:
    import torch
    import torch.distributed as dist
    import torch_npu
    from tests.ascend.simt_urma.run_two_rank_probe import hccl_communicator

    del torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    required_world = 2 if args.mode == "p2p" else 8
    if world_size != required_world:
        raise RuntimeError(
            f"{args.mode} requires world size {required_world}, got {world_size}")
    if (args.warmups < 0 or args.iterations <= 0 or
            args.collective_timeout_seconds <= 0):
        raise ValueError(
            "warmups must be nonnegative and iterations and timeout positive")

    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        timeout=timedelta(seconds=args.collective_timeout_seconds),
    )
    handle = None
    native = NativeBenchmark(args.runner)
    try:
        group = dist.distributed_c10d._get_default_group()
        communicator = hccl_communicator(group, local_rank, torch)
        sizes = (
            (0,) if args.mode == "transport-only"
            else parse_size_mib_list(args.sizes_mib)
        )
        transport_record_bytes = (
            int(native.representative_record_bytes())
            if args.mode == "transport-only" else 0
        )
        if args.mode == "transport-only" and transport_record_bytes <= 0:
            raise RuntimeError("production representative record layout failed")
        matrices = tuple(
            _matrix_for_point(
                args.mode, world_size, payload_bytes, args.p2p_sender,
                transport_record_bytes)
            for payload_bytes in sizes
        )
        peer_stride = max(
            value for matrix in matrices for row in matrix for value in row)
        peer_stride = max(MIB, ((peer_stride + MIB - 1) // MIB) * MIB)

        create_error = ctypes.create_string_buffer(2048)
        handle = native.create(
            communicator, rank, world_size, peer_stride,
            create_error, len(create_error))
        _collective_status(
            dist, rank, 0 if handle else 1,
            create_error.value.decode(errors="replace"), "create")

        measurements = []
        generation = 0
        for payload_bytes, matrix in zip(sizes, matrices):
            local_peer_bytes = (ctypes.c_uint64 * world_size)(*matrix[rank])
            expected = (ctypes.c_uint64 * world_size)(
                *expected_payloads(matrix, destination_rank=rank))
            status, error = _native_error(native.reset, handle)
            _collective_status(dist, rank, status, error, "reset")
            dist.barrier()

            samples = {"producer": [], "service": [], "total": []}
            for iteration in range(args.warmups + args.iterations):
                generation += 1
                cycles = NativeCycles()
                status, error = _native_error(
                    native.run, handle, local_peer_bytes, generation,
                    ctypes.byref(cycles))
                _collective_status(
                    dist, rank, status, error,
                    f"generation {generation}")
                if iteration >= args.warmups:
                    samples["producer"].append(
                        cycles.producer_cycles / 1e9)
                    samples["service"].append(
                        cycles.service_cycles / 1e9)
                    samples["total"].append(cycles.total_cycles / 1e9)

            dist.barrier()
            status, error = _native_error(native.verify, handle, expected)
            _collective_status(dist, rank, status, error, "verify")
            gathered = [None] * world_size
            dist.all_gather_object(gathered, samples)
            if rank == 0:
                aggregate_bytes = sum(
                    sum(sender_row) for sender_row in matrix)
                active_senders = sum(any(row) for row in matrix)
                peers_per_sender = max(
                    sum(value > 0 for value in row) for row in matrix)
                measurements.append(summarize_measurement(
                    mode=args.mode,
                    payload_bytes=payload_bytes,
                    active_senders=active_senders,
                    peers_per_sender=peers_per_sender,
                    aggregate_bytes=aggregate_bytes,
                    rank_samples_seconds=tuple(
                        row["total"] for row in gathered),
                    phase_rank_samples_seconds={
                        "producer": tuple(
                            row["producer"] for row in gathered),
                        "service": tuple(
                            row["service"] for row in gathered),
                    },
                ))

        if rank == 0:
            report = build_report(
                git_commit=_git_commit(),
                device_name=torch.npu.get_device_name(local_rank),
                world_size=world_size,
                warmups=args.warmups,
                iterations=args.iterations,
                hcomm_root=os.environ.get("HCOMM_ROOT", "unknown"),
                measurements=measurements,
            )
            report["mode"] = args.mode
            if args.mode == "p2p":
                report["p2p_sender"] = args.p2p_sender
            if args.mode == "transport-only":
                report.update(representative_report_metadata(
                    transport_record_bytes))
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            print(f"wrote {output}", flush=True)
        dist.barrier()
        return 0
    finally:
        if handle is not None:
            status, error = _native_error(native.destroy, handle)
            try:
                _collective_status(dist, rank, status, error, "destroy")
            finally:
                handle = None
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    return _run_distributed(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
