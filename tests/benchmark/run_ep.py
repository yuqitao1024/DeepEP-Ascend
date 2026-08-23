import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import TextIO
import uuid


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.benchmark.profiles import PROFILES, profile_cases, profile_manifest
from tests.benchmark.report_markdown import (
    render_backend_markdown,
    validate_complete_report,
    write_text_atomic,
)
from tests.utils.ep_benchmark_manifest import (
    load_manifest,
    write_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGING_REPORT_NAME = "benchmark.staging.json"
FINAL_REPORT_NAME = "benchmark.json"
FINAL_MARKDOWN_NAME = "benchmark.md"
WORKLOAD_NAME = "workload.json"
RUN_LOG_NAME = "run.log"
CHILD_CLEANUP_TIMEOUT_SECONDS = 5.0
ASCEND_READINESS_COMMAND = (
    sys.executable,
    "tests/ascend/benchmark/bench_ep.py",
    "--list-cases", "--suite", "all", "--format", "json",
)


@dataclass(frozen=True)
class RunConfig:
    backend: str
    profile: str
    output_dir: Path
    workload_manifest: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an eight-rank EP benchmark backend",
    )
    parser.add_argument("--backend", choices=("cuda", "ascend"), required=True)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="smoke"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path)
    return parser


def _profile_arguments(profile) -> tuple[str, ...]:
    arguments = (
        "--num-tokens", str(profile.num_tokens),
        "--hidden", str(profile.hidden),
        "--num-topk", str(profile.num_topk),
        "--num-experts", str(profile.num_experts),
        "--seed", str(profile.seed),
        "--unbalanced-ratio", str(profile.unbalanced_ratio),
        "--masked-ratio", str(profile.masked_ratio),
        "--allow-multiple-reduction", str(profile.allow_multiple_reduction),
        "--warmups", str(profile.warmups),
        "--iterations", str(profile.iterations),
    )
    if profile.precise_unbalanced_ratio:
        arguments += ("--precise-unbalanced-ratio",)
    return arguments


def build_backend_command(
    config: RunConfig,
    staging_report: Path,
    manifest_path: Path,
) -> tuple[str, ...]:
    if config.backend not in ("cuda", "ascend"):
        raise ValueError(f"unknown backend: {config.backend}")
    try:
        profile = PROFILES[config.profile]
    except KeyError as error:
        raise ValueError(f"unknown profile: {config.profile}") from error

    case_ids = ",".join(case.case_id for case in profile_cases(profile))
    profile_arguments = _profile_arguments(profile)
    common = (
        *profile_arguments,
        "--cases", case_ids,
        "--workload-manifest", str(Path(manifest_path)),
    )
    if config.backend == "cuda":
        return (
            sys.executable,
            "tests/elastic/test_ep.py",
            "--benchmark-profile", "parity",
            "--num-processes", str(profile.world_size),
            *common,
            "--benchmark-json", str(Path(staging_report)),
        )
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={profile.world_size}",
        "tests/ascend/benchmark/bench_ep.py",
        "--num-sms", str(profile.ascend_num_sms),
        *common,
        "--output", str(Path(staging_report)),
    )


def validate_ascend_readiness(payload: dict) -> None:
    expected = {"total": 144, "supported": 144, "deferred": 0}
    if not isinstance(payload, dict) or payload.get("summary") != expected:
        raise ValueError(
            "Ascend readiness requires total=144, supported=144, deferred=0"
        )


def run_logged_command(
    command: tuple[str, ...], log_handle: TextIO
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output = []
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output.append(line)
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
        process.stdout.close()
        return process.wait(), "".join(output)
    except BaseException:
        _stop_and_reap_child(process)
        raise


def _stop_and_reap_child(process) -> None:
    try:
        process.stdout.close()
    except BaseException:
        pass

    for group_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, group_signal)
        except BaseException:
            pass
        try:
            process.wait(timeout=CHILD_CLEANUP_TIMEOUT_SECONDS)
        except BaseException:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def _log_event(log_handle: TextIO, **payload) -> None:
    log_handle.write(json.dumps(payload, sort_keys=True) + "\n")
    log_handle.flush()


def _print_diagnostic(error: BaseException) -> None:
    print(
        f"run_ep.py: error: {type(error).__name__}: {error}",
        file=sys.stderr,
    )


def _normalize_command_result(result) -> tuple[int, str]:
    if isinstance(result, int):
        return result, ""
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], int)
        and isinstance(result[1], str)
    ):
        return result
    raise TypeError("command runner must return an exit code or (exit code, output)")


def _invoke_command(command_runner, command, log_handle, config):
    _log_event(
        log_handle,
        event="command",
        backend=config.backend,
        profile=config.profile,
        command=list(command),
        timestamp_utc=_utc_now(),
    )
    exit_code, output = _normalize_command_result(
        command_runner(command, log_handle)
    )
    _log_event(
        log_handle,
        event="command_exit",
        backend=config.backend,
        profile=config.profile,
        exit_code=exit_code,
        timestamp_utc=_utc_now(),
    )
    return exit_code, output


def _validate_config(config: RunConfig) -> None:
    if config.backend not in ("cuda", "ascend"):
        raise ValueError(f"unknown backend: {config.backend}")
    if config.profile not in PROFILES:
        raise ValueError(f"unknown profile: {config.profile}")


def _prepare_output_directory(config: RunConfig) -> Path:
    output_dir = Path(config.output_dir).resolve()
    if output_dir == REPOSITORY_ROOT:
        raise ValueError("output directory must not be the repository root")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output directory is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = tuple(
        name
        for name in (
            FINAL_REPORT_NAME,
            FINAL_MARKDOWN_NAME,
            STAGING_REPORT_NAME,
        )
        if (output_dir / name).exists()
    )
    if occupied:
        raise ValueError(
            "output directory already contains benchmark artifacts: "
            + ", ".join(occupied)
        )
    return output_dir


def _resolve_workload(config: RunConfig, output_dir: Path) -> Path:
    profile = PROFILES[config.profile]
    expected = profile_manifest(profile)
    output = (output_dir / WORKLOAD_NAME).resolve()
    if config.workload_manifest is None:
        write_manifest(output, expected)
        return output

    source = Path(config.workload_manifest).resolve()
    supplied = load_manifest(source)
    if supplied != expected:
        raise ValueError(
            f"workload manifest does not match the {config.profile} profile"
        )
    if source != output:
        shutil.copyfile(source, output)
    return output


def execute_run(config: RunConfig, command_runner=run_logged_command) -> int:
    _validate_config(config)
    output_dir = _prepare_output_directory(config)
    log_path = output_dir / RUN_LOG_NAME
    staging_report = output_dir / STAGING_REPORT_NAME
    final_report = output_dir / FINAL_REPORT_NAME
    final_markdown = output_dir / FINAL_MARKDOWN_NAME
    pending_markdown: Path | None = None
    commit = _git_commit()
    run_exit_code = 1

    with log_path.open("a", encoding="utf-8") as log_handle:
        _log_event(
            log_handle,
            event="run_start",
            backend=config.backend,
            profile=config.profile,
            git_commit=commit,
            timestamp_utc=_utc_now(),
        )
        try:
            manifest_path = _resolve_workload(config, output_dir)
            if config.backend == "ascend":
                exit_code, output = _invoke_command(
                    command_runner,
                    ASCEND_READINESS_COMMAND,
                    log_handle,
                    config,
                )
                if exit_code != 0:
                    run_exit_code = exit_code
                    return exit_code
                validate_ascend_readiness(json.loads(output))

            command = build_backend_command(
                config, staging_report, manifest_path
            )
            exit_code, _ = _invoke_command(
                command_runner, command, log_handle, config
            )
            if exit_code != 0:
                run_exit_code = exit_code
                return exit_code
            if not staging_report.is_file():
                raise ValueError("backend did not produce a staging report")

            report = json.loads(staging_report.read_text(encoding="utf-8"))
            profile = PROFILES[config.profile]
            validate_complete_report(
                report,
                platform=config.backend,
                profile=profile,
                require_h800=config.backend == "cuda",
            )
            markdown = render_backend_markdown(report, profile)
            pending_markdown = output_dir / (
                f".{FINAL_MARKDOWN_NAME}.{uuid.uuid4().hex}"
            )
            write_text_atomic(pending_markdown, markdown)

            os.replace(pending_markdown, final_markdown)
            pending_markdown = None
            try:
                os.replace(staging_report, final_report)
            except BaseException:
                final_markdown.unlink(missing_ok=True)
                raise

            run_exit_code = 0
            return 0
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            _log_event(
                log_handle,
                event="diagnostic",
                backend=config.backend,
                profile=config.profile,
                diagnostic=f"{type(error).__name__}: {error}",
                exit_code=1,
                timestamp_utc=_utc_now(),
            )
            _print_diagnostic(error)
            return 1
        finally:
            if pending_markdown is not None:
                pending_markdown.unlink(missing_ok=True)
            _log_event(
                log_handle,
                event="run_end",
                backend=config.backend,
                profile=config.profile,
                exit_code=run_exit_code,
                timestamp_utc=_utc_now(),
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        backend=args.backend,
        profile=args.profile,
        output_dir=args.output_dir,
        workload_manifest=args.workload_manifest,
    )
    try:
        return execute_run(config)
    except ValueError as error:
        _print_diagnostic(error)
        return 2
    except (OSError, subprocess.SubprocessError) as error:
        _print_diagnostic(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
