"""CPU-only regression tests for the ET-05 workspace subprocess timeout fix.

The workspace dispatch layer (`_run_cli_command`) used to launch delegated
benchmark runners through ``subprocess.run`` without any timeout, so a hung
runner blocked the workspace forever. It now goes through
``run_logged_subprocess`` with an optional parent-side backstop timeout and
kills the child's process group on expiry.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# The package __init__ re-exports only public names (``from .dispatch import *``),
# so the private helpers under test must be imported from the dispatch module.
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry.dispatch import (
    _run_cli_command,
    _workspace_subprocess_timeout,
)


def test_run_cli_command_timeout_kills_child_and_raises(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(120)"]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        _run_cli_command(
            command,
            output_dir=tmp_path,
            benchmark_id="fake-bench",
            delegate_runner="fake_runner",
            request={"command": command},
            log_callback=None,
            timeout=1.5,
        )
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"timeout enforcement took too long: {elapsed:.1f}s"
    assert (tmp_path / "workspace_runner_stdout.log").is_file()
    assert (tmp_path / "workspace_runner_stderr.log").is_file()


def test_run_cli_command_success_still_parses_scorecard(tmp_path: Path) -> None:
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_payload = {
        "run": {"status": "completed", "returncode": 0},
        "metrics": {"summary": {"sample_count": 1}},
    }
    command = [
        sys.executable,
        "-c",
        f"import json; json.dump({scorecard_payload!r}, open({str(scorecard_path)!r}, 'w')); print('runner done')",
    ]
    result = _run_cli_command(
        command,
        output_dir=tmp_path,
        benchmark_id="fake-bench",
        delegate_runner="fake_runner",
        request={"command": command},
        log_callback=None,
        timeout=60.0,
    )
    assert result["status"] == "completed"
    assert "runner done" in (tmp_path / "workspace_runner_stdout.log").read_text(encoding="utf-8")


def test_run_cli_command_missing_scorecard_error_includes_log_tail(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('boom: dataset missing\\n'); sys.exit(3)",
    ]
    with pytest.raises(RuntimeError, match="dataset missing"):
        _run_cli_command(
            command,
            output_dir=tmp_path,
            benchmark_id="fake-bench",
            delegate_runner="fake_runner",
            request={"command": command},
            log_callback=None,
            timeout=60.0,
        )


def test_workspace_subprocess_timeout_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", raising=False)
    assert _workspace_subprocess_timeout({}) is None

    assert _workspace_subprocess_timeout({"timeout": 10}) == pytest.approx(70.0)
    assert _workspace_subprocess_timeout({"timeout": "25"}) == pytest.approx(85.0)

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "100")
    assert _workspace_subprocess_timeout({}) == pytest.approx(160.0)
    # Explicit workspace config wins over the environment default.
    assert _workspace_subprocess_timeout({"timeout": 10}) == pytest.approx(70.0)

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "not-a-number")
    assert _workspace_subprocess_timeout({}) is None
