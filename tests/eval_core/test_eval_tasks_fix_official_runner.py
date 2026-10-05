"""CPU-only regression tests for evaluation-tasks review fixes ET-04/10/12/14/15.

Covers the shared official runner CLI framework (`official_runner.py`) failure
contract, stale-result rejection, alias matching, declared-scale normalization,
and the worldscore standalone runner failure contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from worldfoundry.evaluation.tasks.execution.framework import official_runner as ors


def _config(tmp_path: Path, **overrides) -> ors.BenchRunnerConfig:
    defaults = dict(
        benchmark_id="fixture-bench",
        display_name="Fixture Bench",
        root_env="",
        results_path_env="FIXTURE_BENCH_RESULTS",
        default_repo_subdir="",
        metric_order=("score_a",),
        metric_specs={"score_a": {"name": "Score A", "group": "official", "higher_is_better": True}},
        metric_aliases={},
        average_metric_id="fixture_average",
        official_entry="fixture_entry.py",
        official_output_globs=("results/*.json",),
    )
    defaults.update(overrides)
    return ors.BenchRunnerConfig(**defaults)


def _run_main(config: ors.BenchRunnerConfig, hooks: ors.RunnerHooks, output_dir: Path, *extra: str) -> int:
    argv = [
        "--output-dir",
        str(output_dir),
        "--run-official",
        "--generated-video-dir",
        str(output_dir / "videos"),
        *extra,
    ]
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    return ors.run_main(config, hooks, argv)


def _read_scorecard(output_dir: Path) -> dict:
    scorecard_path = output_dir / "scorecard.json"
    assert scorecard_path.is_file(), "failure contract: scorecard.json must exist"
    return json.loads(scorecard_path.read_text(encoding="utf-8"))


def test_timeout_writes_failed_scorecard_and_returns_nonzero(tmp_path: Path) -> None:
    """ET-04: an official-command timeout must produce a failed scorecard."""
    output_dir = tmp_path / "out"
    hooks = ors.RunnerHooks(
        build_official_command=lambda **kwargs: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    exit_code = _run_main(_config(tmp_path), hooks, output_dir, "--timeout", "1")

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 1
    assert scorecard["run"]["status"] == "failed"
    assert any("timed out" in reason for reason in scorecard["validation"]["blocked_reasons"])
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["normalization_ok"] is False


def test_unexpected_exception_writes_failed_scorecard(tmp_path: Path) -> None:
    """ET-04: exceptions outside the old whitelist still leave a failed scorecard."""
    output_dir = tmp_path / "out"

    def _boom(**kwargs):
        raise KeyError("boom")

    exit_code = _run_main(_config(tmp_path), ors.RunnerHooks(build_official_command=_boom), output_dir)

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 1
    assert scorecard["run"]["status"] == "failed"
    assert "KeyError" in scorecard["run"]["error"]


def test_missing_results_writes_schema_complete_failed_scorecard(tmp_path: Path) -> None:
    """ET-04: validation failures after output resolution still write artifacts."""
    output_dir = tmp_path / "out"

    exit_code = ors.run_main(
        _config(tmp_path),
        ors.RunnerHooks(build_official_command=lambda **kwargs: None),
        ["--output-dir", str(output_dir)],
    )

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 2
    assert scorecard["run"]["status"] == "failed"
    assert scorecard["artifacts"]["upstream_results"].endswith("blocked_results.json")
    assert ors.runner_result_payload(_config(tmp_path), scorecard, output_dir=output_dir)["upstream_results"]


def test_command_secret_flags_are_redacted() -> None:
    assert ors.redact_command_secrets(
        ["python", "runner.py", "--gemini-api-key", "secret", "--auth_token=value", "--safe", "ok"]
    ) == [
        "python",
        "runner.py",
        "--gemini-api-key",
        "<redacted>",
        "--auth_token=<redacted>",
        "--safe",
        "ok",
    ]


def test_nonzero_exit_does_not_score_fresh_results(tmp_path: Path) -> None:
    """ET-15: a failing official command must not be scored, even with results present."""
    output_dir = tmp_path / "out"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True)
    script = (
        "import json, pathlib, sys;"
        f"pathlib.Path({str(results_dir / 'fresh.json')!r}).write_text(json.dumps({{'score_a': 0.9}}));"
        "sys.exit(2)"
    )
    hooks = ors.RunnerHooks(build_official_command=lambda **kwargs: [sys.executable, "-c", script])

    exit_code = _run_main(_config(tmp_path), hooks, output_dir)

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 1
    assert scorecard["run"]["status"] == "failed"
    assert any("exit code 2" in reason for reason in scorecard["validation"]["blocked_reasons"])
    assert scorecard["metrics"]["leaderboard"] == {}
    assert scorecard["dataset"]["upstream_results"].endswith("blocked_results.json")


def test_stale_results_from_previous_run_are_rejected(tmp_path: Path) -> None:
    """ET-15: results whose mtime predates the run must not masquerade as fresh."""
    output_dir = tmp_path / "out"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True)
    stale = results_dir / "old.json"
    stale.write_text(json.dumps({"score_a": 0.9}), encoding="utf-8")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    hooks = ors.RunnerHooks(build_official_command=lambda **kwargs: [sys.executable, "-c", "pass"])

    exit_code = _run_main(_config(tmp_path), hooks, output_dir)

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 1
    assert scorecard["run"]["status"] == "failed"
    assert any("stale" in reason for reason in scorecard["validation"]["blocked_reasons"])
    assert scorecard["metrics"]["leaderboard"] == {}


def test_fresh_successful_run_is_still_scored(tmp_path: Path) -> None:
    """Successful official runs keep producing scored scorecards after the fix."""
    output_dir = tmp_path / "out"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True)
    script = (
        "import json, pathlib;"
        f"pathlib.Path({str(results_dir / 'fresh.json')!r}).write_text(json.dumps({{'score_a': 0.9}}))"
    )
    hooks = ors.RunnerHooks(build_official_command=lambda **kwargs: [sys.executable, "-c", script])

    exit_code = _run_main(_config(tmp_path), hooks, output_dir)

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 0
    assert scorecard["run"]["status"] == "official_bounded"
    assert scorecard["metrics"]["leaderboard"]["score_a"] == pytest.approx(0.9)


def test_worldscore_failure_writes_failed_scorecard(tmp_path: Path) -> None:
    """ET-14: the standalone worldscore runner keeps the failure contract."""
    from worldfoundry.evaluation.tasks.execution.runners.worldscore import (
        run_worldscore_official_runner as worldscore,
    )

    bad_results = tmp_path / "bad.json"
    bad_results.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    output_dir = tmp_path / "worldscore-out"

    exit_code = worldscore.main(
        [
            "--output-dir",
            str(output_dir),
            "--official-results-path",
            str(bad_results),
            "--json",
        ]
    )

    scorecard = _read_scorecard(output_dir)
    assert exit_code == 1
    assert scorecard["run"]["status"] == "failed"
    assert scorecard["official_benchmark_verified"] is False


def test_metric_id_from_key_rejects_ambiguous_substring_matches() -> None:
    """ET-12: substring alias matches must be unique to be trusted."""
    config = _config(
        Path("."),
        metric_aliases={
            "subject_consistency": "metric_one",
            "background_consistency": "metric_two",
        },
    )

    assert ors.metric_id_from_key("subject_consistency", config) == "metric_one"
    # "consistency" is a substring of two aliases mapping to different metrics.
    assert ors.metric_id_from_key("consistency", config) is None
    # Unique fuzzy hits still resolve (upstream column drift tolerance).
    assert ors.metric_id_from_key("subject_consistency_v2", config) == "metric_one"


def test_declared_scale_normalizer_overrides_percent_heuristic(tmp_path: Path) -> None:
    """ET-10: catalog-declared scales replace the blind (1,100] /100 heuristic."""
    config = _config(tmp_path, benchmark_id="video-bench")
    declared = ors.declared_metric_normalizers("video-bench")
    likert_metric = next((m for m, spec in declared.items() if spec == "scale_max:5"), None)
    assert likert_metric is not None, "video-bench declares scale_max:5 metrics in the catalog"

    # 4.2 on a 1-5 judge scale: declared scale gives 0.84, the old heuristic gave 0.042.
    assert ors.normalized_metric_score(config, likert_metric, 4.2) == pytest.approx(4.2 / 5.0)

    # Undeclared metrics keep the legacy heuristic (backwards compatible).
    fixture = _config(tmp_path)
    assert ors.normalized_metric_score(fixture, "score_a", 75.0) == pytest.approx(0.75)
    assert ors.normalized_metric_score(fixture, "score_a", 0.4) == pytest.approx(0.4)
