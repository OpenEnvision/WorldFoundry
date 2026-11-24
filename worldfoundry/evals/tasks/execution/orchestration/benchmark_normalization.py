"""Normalize official benchmark results into WorldFoundry scorecards and metric rows."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from worldfoundry.core.execution.process import run_logged_subprocess
from worldfoundry.core.io.formats.serialization import write_json
from worldfoundry.core.observability.time import utc_now_iso
from worldfoundry.evals.tasks.catalog.dispatch import (
    embodied_result_normalizer_tracks,
    pass_generated_artifact_dir_benchmarks,
    specialized_result_normalizer_scripts,
)

from ...catalog.schema import BenchmarkZooEntry
from .benchmark_commands import (
    REPO_ROOT,
    SCORECARD_SCHEMA_VERSION,
    JsonValue,
    _apply_benchmark_data_root_env,
    _artifact_path,
    _ensure_repo_on_pythonpath,
    _env_mapping,
    _runtime_spec_env,
    _timeout_seconds,
)


def _scorecard_normalization_available(scorecard: Mapping[str, JsonValue]) -> bool:
    """Return True when ``scorecard`` exposes at least one normalized metric."""
    if scorecard.get("normalization_ok") is True:
        return True
    evaluation = scorecard.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("available") is True:
        return True
    metrics = scorecard.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    leaderboard = metrics.get("leaderboard")
    if isinstance(leaderboard, Mapping) and bool(leaderboard):
        return True
    per_metric = metrics.get("per_metric")
    if isinstance(per_metric, Mapping):
        return any(isinstance(item, Mapping) and item.get("available") is True for item in per_metric.values())
    return False


def _run_specialized_result_normalizer(
    *,
    benchmark_id: str,
    output_dir: Path,
    results_path: Path | None,
    runtime_spec: Mapping[str, JsonValue],
    generated_artifact_dir: JsonValue = None,
    run_official: bool = False,
    kwargs: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue] | None:
    """Spawn a per-benchmark specialized normalizer subprocess."""
    spec = specialized_result_normalizer_scripts().get(benchmark_id)
    if spec is None:
        return None
    script, result_flag = spec
    script_path = REPO_ROOT / script
    if not script_path.is_file():
        return None
    kwargs = kwargs or {}
    stdout_path = output_dir / "specialized_normalizer_stdout.log"
    stderr_path = output_dir / "specialized_normalizer_stderr.log"
    command = [
        sys.executable,
        str(script_path),
        "--benchmark-id",
        benchmark_id,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if run_official:
        command.append("--run-official")
    if results_path is not None:
        command.extend([result_flag, str(results_path)])
    from worldfoundry.evals.tasks.catalog.dispatch import official_runner_spec

    runner_spec = official_runner_spec(benchmark_id)
    result_model_id = kwargs.get("result_model_id")
    if runner_spec is not None and runner_spec.model_arg and result_model_id:
        command.extend([runner_spec.model_arg, str(result_model_id)])
    score_dir = kwargs.get("score_dir") if benchmark_id == "camerabench" else None
    if score_dir not in (None, ""):
        command.extend(["--score-dir", str(score_dir), "--task", "all", "--no-gpt"])
    if benchmark_id in pass_generated_artifact_dir_benchmarks() and generated_artifact_dir is not None:
        command.extend(["--generated-artifact-dir", str(generated_artifact_dir)])
    if benchmark_id == "pawbench":
        dataset_root = kwargs.get("dataset_root") or kwargs.get("benchmark_data_root")
        if dataset_root:
            command.extend(["--dataset-root", str(dataset_root)])
        model_name = kwargs.get("model_name") or kwargs.get("model_id")
        if model_name:
            command.extend(["--model-name", str(model_name)])
    if benchmark_id == "phyfps-bench-gen" and kwargs.get("prompt_manifest"):
        command.extend(["--prompt-manifest", str(kwargs["prompt_manifest"])])
    env = os.environ.copy()
    env.update(
        {
            "WORLDFOUNDRY_BENCHMARK_ID": benchmark_id,
            "WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR": str(output_dir),
        }
    )
    _ensure_repo_on_pythonpath(env)
    env.update(
        _runtime_spec_env(
            runtime_spec,
            benchmark_id=benchmark_id,
            generated_artifact_dir=generated_artifact_dir,
        )
    )
    _apply_benchmark_data_root_env(env, benchmark_id, kwargs)
    env.update(_env_mapping(kwargs.get("env_overrides", kwargs.get("env"))))
    timeout_seconds = _timeout_seconds(kwargs.get("timeout_seconds", kwargs.get("timeout")))
    start = time.monotonic()
    completed = run_logged_subprocess(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=REPO_ROOT,
        env=env,
        timeout=timeout_seconds,
    )
    duration_seconds = time.monotonic() - start
    scorecard_path = output_dir / "scorecard.json"
    if not scorecard_path.is_file():
        return {
            "ok": False,
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": duration_seconds,
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "scorecard": None,
            "error": "specialized normalizer did not write scorecard.json",
        }
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if not isinstance(scorecard, dict):
        return {
            "ok": False,
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": duration_seconds,
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "scorecard": None,
            "error": "specialized normalizer wrote a non-object scorecard.json",
        }
    normalization_ok = scorecard.get("normalization_ok") is True or _scorecard_normalization_available(scorecard)
    scorecard["normalization_ok"] = normalization_ok
    validation = scorecard.get("validation")
    validation_normalizer_only = validation.get("normalizer_only") if isinstance(validation, Mapping) else None
    if isinstance(validation_normalizer_only, bool):
        scorecard["normalizer_only"] = validation_normalizer_only
    else:
        scorecard.setdefault("normalizer_only", scorecard.get("integration_evidence") is not True)
    scorecard["runtime"] = dict(runtime_spec)
    run = scorecard.get("run")
    if isinstance(run, Mapping):
        scorecard["run"] = {
            **dict(run),
            "runner": run.get("runner") or "benchmark_zoo_specialized_result_normalizer",
            "command": command,
            "returncode": completed.returncode,
            "duration_seconds": duration_seconds,
        }
    artifacts = dict(scorecard.get("artifacts") or {})
    artifacts["stdout"] = str(stdout_path.resolve())
    artifacts["stderr"] = str(stderr_path.resolve())
    scorecard["artifacts"] = artifacts
    write_json(scorecard_path, scorecard)
    return {
        "ok": normalization_ok,
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "scorecard": scorecard,
        "error": None if normalization_ok else "specialized normalizer produced no available metrics",
    }


def _run_embodied_result_normalizer(
    *,
    benchmark_id: str,
    output_dir: Path,
    results_path: Path,
    runtime_spec: Mapping[str, JsonValue],
    kwargs: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue] | None:
    """Normalize embodied/VLA results in-process via ``embodied.normalizer``."""
    track = embodied_result_normalizer_tracks().get(benchmark_id)
    if track is None:
        return None
    kwargs = kwargs or {}
    normalizer_overrides = kwargs.get("normalizers")
    normalizers = dict(normalizer_overrides) if isinstance(normalizer_overrides, Mapping) else {}
    start = time.monotonic()
    try:
        from worldfoundry.evals.tasks.embodied.normalizer import normalize_results

        scorecard = normalize_results(
            input_paths=[results_path],
            output_dir=output_dir,
            benchmark_id=benchmark_id,
            track=track,
            normalizers={str(key): str(value) for key, value in normalizers.items()},
        )
        returncode = 0
        error = None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        scorecard = {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "official_benchmark_verified": False,
            "integration_evidence": False,
            "leaderboard_valid": False,
            "normalizer_only": True,
            "normalization_ok": False,
            "run": {
                "status": "failed",
                "started_at": utc_now_iso(),
                "runner": "worldfoundry.evals.tasks.embodied.normalizer",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "benchmark": {
                "benchmark_id": benchmark_id,
                "track": track,
                "contract_only": False,
            },
            "evaluation": {
                "available": False,
                "kind": "vla_va_wam_official_result_normalizer",
                "num_results": 0,
            },
            "metrics": {
                "leaderboard": {},
                "per_metric": {},
                "summary": {"normalized_result_rows": 0},
            },
            "artifacts": {"scorecard": _artifact_path(output_dir, "scorecard.json")},
        }
        write_json(output_dir / "scorecard.json", scorecard)
        returncode = 1
        error = str(exc)
    duration_seconds = time.monotonic() - start
    scorecard = dict(scorecard)
    normalization_ok = _scorecard_normalization_available(scorecard)
    scorecard["normalization_ok"] = normalization_ok
    scorecard["normalizer_only"] = True
    scorecard["runtime"] = dict(runtime_spec)
    run = scorecard.get("run")
    if isinstance(run, Mapping):
        scorecard["run"] = {
            **dict(run),
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        }
    write_json(output_dir / "scorecard.json", scorecard)
    return {
        "ok": normalization_ok,
        "command": [
            "wf",
            "normalize",
            "embodied",
            "--benchmark-id",
            benchmark_id,
            "--track",
            track,
            "--input",
            str(results_path),
            "--output-dir",
            str(output_dir),
        ],
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "stdout_path": None,
        "stderr_path": None,
        "scorecard": scorecard,
        "error": error
        if error
        else (None if normalization_ok else "embodied normalizer produced no available metrics"),
    }


def _metric_result_value(result: Mapping[str, JsonValue]) -> JsonValue:
    """Prefer ``normalized_value``, else ``raw_value``."""
    value = result.get("normalized_value")
    return result.get("raw_value") if value is None else value


def _metric_scorecard_entry(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Convert one metric result row into scorecard ``per_metric`` shape."""
    value = _metric_result_value(result)
    valid = result.get("valid") is True and value is not None
    if valid:
        return {
            "available": True,
            "raw_score": result.get("raw_value"),
            "normalized_score": result.get("normalized_value"),
            "coverage": result.get("coverage", 1.0),
        }
    return {
        "available": False,
        "reason": result.get("skip_reason") or "metric_not_available",
        "diagnostics": result.get("diagnostics", {}),
    }


def _metric_raw_row(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Expose raw metric row with ``available`` flag."""
    row = dict(result)
    row["available"] = row.get("valid") is True and _metric_result_value(row) is not None
    if not row["available"]:
        row["reason"] = row.get("skip_reason") or "metric_not_available"
    return row


def _dataset_commands(entry: BenchmarkZooEntry) -> tuple[tuple[str, ...], ...]:
    """Build ``hf download`` commands from manifest dataset refs."""
    commands: list[tuple[str, ...]] = []
    for ref in entry.dataset_refs or (entry.dataset,):
        if not ref.hf_dataset_id:
            continue
        command = ["hf", "download", ref.hf_dataset_id, "--repo-type", "dataset"]
        if ref.revision:
            command.extend(("--revision", ref.revision))
        commands.append(tuple(command))
    return tuple(commands)


def _expected_paths(entry: BenchmarkZooEntry) -> tuple[str, ...]:
    """Collect expected dataset and artifact paths from the manifest."""
    paths: list[str] = []
    for ref in entry.dataset_refs or (entry.dataset,):
        if ref.path and ref.path not in paths:
            paths.append(ref.path)
    for artifact in entry.expected_artifacts:
        if isinstance(artifact, str):
            paths.append(artifact)
        elif isinstance(artifact, Mapping):
            path = artifact.get("path") or artifact.get("uri")
            if path:
                paths.append(str(path))
    return tuple(dict.fromkeys(paths))
