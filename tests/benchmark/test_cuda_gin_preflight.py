import signal
import subprocess

import pytest

from tests.benchmark import check_cuda_gin as preflight


GIN_UNAVAILABLE = "NCCL GIN is unavailable"


@pytest.mark.parametrize(
    ("results", "expected_conclusion", "expected_modes"),
    (
        ([(0, "direct ready")], "DIRECT_GIN_AVAILABLE", ["direct"]),
        (
            [(1, GIN_UNAVAILABLE), (0, "hybrid ready")],
            "HYBRID_GIN_ONLY",
            ["direct", "hybrid"],
        ),
        (
            [(1, GIN_UNAVAILABLE), (1, GIN_UNAVAILABLE), (0, "fallback ready")],
            "GIN_UNAVAILABLE_BASE_RUNTIME_OK",
            ["direct", "hybrid", "fallback"],
        ),
        ([(1, "CUDA out of memory")], "PROBE_FAILED", ["direct"]),
    ),
)
def test_preflight_classifies_probe_results(
    tmp_path, results, expected_conclusion, expected_modes
):
    calls = []

    def fake_runner(mode, command, environment, log_path, timeout_seconds):
        calls.append((mode, command, environment, log_path, timeout_seconds))
        return results[len(calls) - 1]

    conclusion = preflight.run_preflight(
        tmp_path,
        command_runner=fake_runner,
        timeout_seconds=17,
        master_port=18361,
    )

    assert conclusion.value == expected_conclusion
    assert [call[0] for call in calls] == expected_modes
    assert all(call[4] == 17 for call in calls)
    assert all("--num-processes" in call[1] for call in calls)
    assert all(call[3].parent == tmp_path for call in calls)

    for call_index, (mode, command, environment, _, _) in enumerate(calls):
        expected_hybrid = "1" if mode == "hybrid" else "0"
        hybrid_index = command.index("--allow-hybrid-mode")
        assert command[hybrid_index + 1] == expected_hybrid
        assert environment["EP_DISABLE_GIN"] == (
            "1" if mode == "fallback" else "0"
        )
        assert environment["WORLD_SIZE"] == "1"
        assert environment["RANK"] == "0"
        assert environment["MASTER_ADDR"] == "127.0.0.1"
        assert environment["MASTER_PORT"] == str(18361 + call_index)


def test_preflight_stops_when_hybrid_fails_for_an_unrelated_reason(tmp_path):
    results = iter(((1, GIN_UNAVAILABLE), (1, "invalid device ordinal")))
    modes = []

    def fake_runner(mode, command, environment, log_path, timeout_seconds):
        modes.append(mode)
        return next(results)

    conclusion = preflight.run_preflight(tmp_path, command_runner=fake_runner)

    assert conclusion.value == "PROBE_FAILED"
    assert modes == ["direct", "hybrid"]


def test_probe_timeout_stops_the_entire_process_group(tmp_path, monkeypatch):
    process_kwargs = {}
    signals = []

    class TimedOutProcess:
        pid = 4321

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired(
                "gin-probe", timeout, output="partial output"
            )

        def wait(self, timeout):
            if len(signals) == 1:
                raise subprocess.TimeoutExpired("gin-probe", timeout)
            return -signal.SIGKILL

    def fake_popen(command, **kwargs):
        process_kwargs.update(kwargs)
        return TimedOutProcess()

    monkeypatch.setattr(preflight.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        preflight.os, "killpg", lambda process_group, sent_signal: signals.append(
            (process_group, sent_signal)
        )
    )
    log_path = tmp_path / "direct.log"

    exit_code, output = preflight._run_probe(
        "direct", ("probe",), {}, log_path, timeout_seconds=3
    )

    assert exit_code == 124
    assert "partial output" in output
    assert "timed out" in output
    assert process_kwargs["start_new_session"] is True
    assert signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]
    assert log_path.read_text(encoding="utf-8") == output
