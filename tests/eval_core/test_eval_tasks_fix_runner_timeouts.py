"""ET-03/ET-06 regression tests: runner subprocess timeout backstop and registry default.

Pure CPU: uses a fake sleeping script as the upstream benchmark process; no GPU,
network, or heavyweight model imports.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.framework.official_runner import (
    default_benchmark_timeout,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import (
    VIDEO_RUNNER_REGISTRY,
    VideoRunnerSpec,
)
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import _timeout_seconds


def test_default_benchmark_timeout_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", raising=False)
    assert default_benchmark_timeout() is None

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "7200")
    assert default_benchmark_timeout() == 7200.0

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "1.5")
    assert default_benchmark_timeout() == 1.5

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "not-a-number")
    assert default_benchmark_timeout() is None

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "")
    assert default_benchmark_timeout() is None


def test_orchestration_timeout_uses_shared_environment_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "42.5")
    assert _timeout_seconds(None) == 42.5
    assert _timeout_seconds(7) == 7.0

    monkeypatch.delenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT")
    assert _timeout_seconds(None) is None


def test_phygenbench_upstream_overall_times_out_on_hung_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ET-06: a hung upstream process must raise TimeoutExpired, not block forever."""
    from worldfoundry.evaluation.tasks.execution.runners.phygenbench import phygenbench_runtime

    repo_root = tmp_path / "phygenbench_repo"
    overall = repo_root / "PhyGenEval" / "overall.py"
    overall.parent.mkdir(parents=True)
    overall.write_text(
        textwrap.dedent(
            """
            import time
            time.sleep(600)
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "1")
    with pytest.raises(subprocess.TimeoutExpired):
        phygenbench_runtime._run_upstream_overall(repo_root=repo_root, model_name="dummy-model")


def test_phygenbench_upstream_overall_unbounded_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the env var the historical behavior (no timeout) is preserved."""
    from worldfoundry.evaluation.tasks.execution.runners.phygenbench import phygenbench_runtime

    repo_root = tmp_path / "phygenbench_repo"
    overall = repo_root / "PhyGenEval" / "overall.py"
    overall.parent.mkdir(parents=True)
    result_dir = repo_root / "result"
    overall.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--root")
            parser.add_argument("--model-name")
            args = parser.parse_args()
            out = Path(args.root) / "result" / f"{args.model_name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"ok": True}))
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("WORLDFOUNDRY_BENCHMARK_TIMEOUT", raising=False)
    result_path = phygenbench_runtime._run_upstream_overall(
        repo_root=repo_root, model_name="dummy-model"
    )
    assert result_path == result_dir / "dummy-model.json"
    assert result_path.is_file()


def test_video_runner_spec_results_flag_default() -> None:
    """ET-03: results_flag has a shared default; existing entries keep the same value."""
    spec = VideoRunnerSpec("some/script.py")
    assert spec.results_flag == "--official-results-path"

    # Every registered entry still resolves to the shared flag.
    for benchmark_id, registered in VIDEO_RUNNER_REGISTRY.items():
        assert registered.results_flag == "--official-results-path", benchmark_id
