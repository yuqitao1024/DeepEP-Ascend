import argparse
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GIN_UNAVAILABLE_MESSAGE = "NCCL GIN is unavailable"


class ProbeStatus(Enum):
    AVAILABLE = "AVAILABLE"
    GIN_UNAVAILABLE = "GIN_UNAVAILABLE"
    FAILED = "FAILED"


class Conclusion(Enum):
    CUOBJDUMP_UNAVAILABLE = "CUOBJDUMP_UNAVAILABLE"
    SINGLE_NODE_NVLINK_READY = "SINGLE_NODE_NVLINK_READY"
    DIRECT_GIN_AVAILABLE = "DIRECT_GIN_AVAILABLE"
    HYBRID_GIN_ONLY = "HYBRID_GIN_ONLY"
    GIN_UNAVAILABLE_BASE_RUNTIME_OK = "GIN_UNAVAILABLE_BASE_RUNTIME_OK"
    PROBE_FAILED = "PROBE_FAILED"


@dataclass(frozen=True)
class CuobjdumpCheck:
    path: Path | None
    error: str | None


def _build_probe_command(hybrid_mode):
    return (
        sys.executable,
        str(REPOSITORY_ROOT / "tests/elastic/test_ep.py"),
        "--num-processes", "2",
        "--num-tokens", "16",
        "--hidden", "256",
        "--num-topk", "2",
        "--num-experts", "8",
        "--allow-hybrid-mode", "1" if hybrid_mode else "0",
        "--test-first-only",
        "--skip-perf-test",
    )


def _find_cuda_home():
    configured_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured_home:
        return Path(configured_home)

    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).parent.parent

    default_home = Path("/usr/local/cuda")
    return default_home if default_home.exists() else None


def _check_cuobjdump():
    cuda_home = _find_cuda_home()
    if cuda_home is None:
        return CuobjdumpCheck(
            path=None,
            error=(
                "CUDA_HOME could not be resolved. Set CUDA_HOME to the CUDA "
                "toolkit directory that contains bin/cuobjdump."
            ),
        )

    cuobjdump = cuda_home / "bin" / "cuobjdump"
    if not cuobjdump.is_file():
        return CuobjdumpCheck(
            path=cuobjdump,
            error=f"Required CUDA binary is missing: {cuobjdump}",
        )
    if not os.access(cuobjdump, os.X_OK):
        return CuobjdumpCheck(
            path=cuobjdump,
            error=f"Required CUDA binary is not executable: {cuobjdump}",
        )
    return CuobjdumpCheck(path=cuobjdump, error=None)


def _run_probe(mode, command, environment, log_path, timeout_seconds):
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        for group_signal in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, group_signal)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        output += f"\nERROR: {mode} Gin probe timed out\n"
        exit_code = 124

    log_path.write_text(output, encoding="utf-8")
    return exit_code, output


def _probe_status(exit_code, output):
    if exit_code == 0:
        return ProbeStatus.AVAILABLE
    if GIN_UNAVAILABLE_MESSAGE in output:
        return ProbeStatus.GIN_UNAVAILABLE
    return ProbeStatus.FAILED


def run_preflight(
    log_dir,
    command_runner=_run_probe,
    cuobjdump_checker=_check_cuobjdump,
    timeout_seconds=120,
    master_port=8361,
    single_node_nvlink=False,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cuobjdump = cuobjdump_checker()
    if cuobjdump.error is not None:
        print(f"Gin preflight prerequisite failed: {cuobjdump.error}", file=sys.stderr)
        return Conclusion.CUOBJDUMP_UNAVAILABLE
    print(f"Gin preflight cuobjdump: {cuobjdump.path}")

    def probe(mode, hybrid_mode, disable_gin):
        environment = os.environ.copy()
        environment["EP_DISABLE_GIN"] = "1" if disable_gin else "0"
        environment["EP_BUFFER_DEBUG"] = "1"
        environment["WORLD_SIZE"] = "1"
        environment["RANK"] = "0"
        environment["MASTER_ADDR"] = "127.0.0.1"
        environment["MASTER_PORT"] = str(
            master_port + ("direct", "hybrid", "fallback").index(mode)
        )
        environment.setdefault("NCCL_DEBUG", "INFO")
        environment.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,ENV")
        command = _build_probe_command(hybrid_mode)
        log_path = log_dir / f"{mode}.log"
        exit_code, output = command_runner(
            mode, command, environment, log_path, timeout_seconds
        )
        print(f"Gin preflight {mode}: exit={exit_code}, log={log_path}")
        return _probe_status(exit_code, output)

    if single_node_nvlink:
        fallback = probe("fallback", hybrid_mode=False, disable_gin=True)
        if fallback is ProbeStatus.AVAILABLE:
            return Conclusion.SINGLE_NODE_NVLINK_READY
        return Conclusion.PROBE_FAILED

    direct = probe("direct", hybrid_mode=False, disable_gin=False)
    if direct is ProbeStatus.AVAILABLE:
        return Conclusion.DIRECT_GIN_AVAILABLE
    if direct is ProbeStatus.FAILED:
        return Conclusion.PROBE_FAILED

    hybrid = probe("hybrid", hybrid_mode=True, disable_gin=False)
    if hybrid is ProbeStatus.AVAILABLE:
        return Conclusion.HYBRID_GIN_ONLY
    if hybrid is ProbeStatus.FAILED:
        return Conclusion.PROBE_FAILED

    fallback = probe("fallback", hybrid_mode=False, disable_gin=True)
    if fallback is ProbeStatus.AVAILABLE:
        return Conclusion.GIN_UNAVAILABLE_BASE_RUNTIME_OK
    return Conclusion.PROBE_FAILED


def build_parser():
    parser = argparse.ArgumentParser(
        description="Classify CUDA direct, hybrid, and non-Gin readiness",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--master-port", type=int, default=8361)
    parser.add_argument(
        "--single-node-nvlink",
        action="store_true",
        help=(
            "run only the non-Gin 2-rank path after the caller has verified "
            "that all benchmark GPUs form one NVLink clique"
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    conclusion = run_preflight(
        args.log_dir,
        single_node_nvlink=args.single_node_nvlink,
        timeout_seconds=args.timeout_seconds,
        master_port=args.master_port,
    )
    print(f"GIN_PREFLIGHT_CONCLUSION={conclusion.value}")

    if conclusion is Conclusion.CUOBJDUMP_UNAVAILABLE:
        print(
            "Install the cuobjdump component for the active CUDA toolkit "
            "(for CUDA 13.0: `apt install cuda-cuobjdump-13-0` or "
            "`dnf install cuda-cuobjdump-13-0`), then verify CUDA_HOME.",
            file=sys.stderr,
        )
        return 5
    if conclusion is Conclusion.SINGLE_NODE_NVLINK_READY:
        print(
            "The non-Gin 2-rank path is ready for the verified single-node "
            "NVLink topology. Keep EP_DISABLE_GIN=1 for the representative run."
        )
        return 0
    if conclusion is Conclusion.DIRECT_GIN_AVAILABLE:
        print("Direct Gin is available; representative parity can continue.")
        return 0
    if conclusion is Conclusion.HYBRID_GIN_ONLY:
        print(
            "Hybrid Gin works, but representative parity requires direct Gin. "
            "Check the multi-plane network configuration or use a separately "
            "declared hybrid benchmark profile.",
            file=sys.stderr,
        )
        return 2
    if conclusion is Conclusion.GIN_UNAVAILABLE_BASE_RUNTIME_OK:
        print(
            "The non-Gin fallback works, but neither direct nor hybrid Gin is "
            "available. Check the NCCL network plugin, RDMA, and GPUDirect RDMA.",
            file=sys.stderr,
        )
        return 3

    print(
        "The probe failed for a reason other than Gin availability. Inspect the "
        f"logs in {args.log_dir}.",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
