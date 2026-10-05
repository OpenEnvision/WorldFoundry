"""EWMBench official scorer runtime (mock CSV or optional evaluate.py dispatch)."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.execution.process import read_text_tail, run_logged_subprocess
from worldfoundry.core.io import materialize_file
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_metrics import METRIC_SPECS
from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_paths import resolve_ewmbench_root
from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_prompts import (
    load_prompt_records,
    unique_prompt_records,
)

EWM_DIMENSIONS = ("scene_consistency", "trajectory_consistency", "semantics", "diversity")
OFFICIAL_OUTPUT_GLOBS = ("**/ewmbm_final_table.csv", "**/*final*.csv")
SEMANTIC_RUBRIC_REL = Path("semantic_caption_rubric.txt")
CONFIG_TEMPLATE_REL = Path("config.template.yaml")


def resolve_semantic_rubric_path(*, repo_root: Path | None = None) -> Path | None:
    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

    env_rubric = os.environ.get("WORLDFOUNDRY_EWMBENCH_SEMANTIC_RUBRIC")
    if env_rubric:
        rubric_path = Path(env_rubric).expanduser().resolve()
        return rubric_path if rubric_path.is_file() else None
    bundled = bundled_benchmark_asset("ewmbench", SEMANTIC_RUBRIC_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_ewmbench_root()
    if root is None:
        return None
    candidate = root / SEMANTIC_RUBRIC_REL
    return candidate if candidate.is_file() else None


def resolve_config_template_path(*, repo_root: Path | None = None) -> Path | None:
    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

    env_config = os.environ.get("WORLDFOUNDRY_EWMBENCH_CONFIG_PATH")
    if env_config:
        config_path = Path(env_config).expanduser().resolve()
        if config_path.is_file():
            return config_path
    bundled = bundled_benchmark_asset("ewmbench", CONFIG_TEMPLATE_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_ewmbench_root()
    if root is None:
        return None
    for relative in (Path("config.yaml"), CONFIG_TEMPLATE_REL):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def resolve_evaluate_script(*, repo_root: Path | None = None) -> Path | None:
    root = repo_root or resolve_ewmbench_root()
    if root is None:
        return None
    candidate = root / "evaluate.py"
    return candidate if candidate.is_file() else None


def discover_official_results(search_roots: list[Path]) -> Path | None:
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file():
            return root
        for pattern in OFFICIAL_OUTPUT_GLOBS:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[-1]
    return None



@dataclass(frozen=True)
class EwmbenchScorerConfig:
    backend: str
    repo_root: Path | None
    config_path: Path | None
    strict: bool = False


def scorer_config_from_env(*, repo_root: Path | None = None) -> EwmbenchScorerConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_EWMBENCH_SCORER_BACKEND")
        or os.environ.get("WORLDFOUNDRY_EWMBENCH_RUNTIME_BACKEND")
        or "mock"
    ).strip().lower()
    repo_root = resolve_ewmbench_root(repo_root)
    strict = os.environ.get("WORLDFOUNDRY_EWMBENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return EwmbenchScorerConfig(
        backend=backend,
        repo_root=repo_root,
        config_path=resolve_config_template_path(repo_root=repo_root),
        strict=strict,
    )


def _deterministic_mock_score(*, seed: str, metric_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{metric_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 25
    higher_is_better = METRIC_SPECS[metric_id]["higher_is_better"]
    if higher_is_better:
        return round(0.55 + bucket / 100.0, 4)
    return round(0.05 + bucket / 500.0, 4)


def _matching_videos(*, generated_artifact_dir: Path, prompt_ids: set[str]) -> list[str]:
    if not generated_artifact_dir.is_dir():
        return []
    matched: list[str] = []
    for path in sorted(generated_artifact_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.stem in prompt_ids:
            matched.append(path.stem)
    return matched


def _write_mock_csv(*, output_path: Path, seed: str, prompt_records: list[Mapping[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in prompt_records:
        episode_seed = f"{seed}:{record['prompt_id']}"
        rows.append(
            {
                "episode_id": record["prompt_id"],
                "scene_consistency": _deterministic_mock_score(seed=episode_seed, metric_id="scene_consistency"),
                "trajectory_consistency": _deterministic_mock_score(
                    seed=episode_seed, metric_id="motion_correctness"
                ),
                "semantics": _deterministic_mock_score(seed=episode_seed, metric_id="semantic_alignment"),
                "diversity": _deterministic_mock_score(seed=episode_seed, metric_id="diversity"),
            }
        )
    if not rows:
        rows.append(
            {
                "episode_id": "mock_episode",
                **{
                    source_col: _deterministic_mock_score(seed=seed, metric_id=metric_id)
                    for source_col, metric_id in (
                        ("scene_consistency", "scene_consistency"),
                        ("trajectory_consistency", "motion_correctness"),
                        ("semantics", "semantic_alignment"),
                        ("diversity", "diversity"),
                    )
                },
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode_id", "scene_consistency", "trajectory_consistency", "semantics", "diversity"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _run_upstream_evaluate(
    *,
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    script = resolve_evaluate_script(repo_root=repo_root)
    if script is None:
        raise FileNotFoundError(f"EWMBench evaluate.py not found under {repo_root}")
    command = [
        os.environ.get("WORLDFOUNDRY_UNIFIED_PYTHON", sys.executable),
        str(script.resolve()),
        "--dimension",
        *EWM_DIMENSIONS,
        "--config_path",
        str(config_path.resolve()),
    ]
    if overwrite:
        command.append("--overwrite")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root.resolve()), env.get("PYTHONPATH")) if part
    )
    timeout_value = os.environ.get("WORLDFOUNDRY_EWMBENCH_TIMEOUT_SECONDS", "").strip()
    timeout_seconds = float(timeout_value) if timeout_value else None
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("WORLDFOUNDRY_EWMBENCH_TIMEOUT_SECONDS must be positive")
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    completed = run_logged_subprocess(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=str(repo_root.resolve()),
        env=env,
        timeout=timeout_seconds,
        start_new_session=False,
    )
    if completed.returncode != 0:
        detail = read_text_tail(stderr_path) or read_text_tail(stdout_path)
        raise RuntimeError(
            "EWMBench upstream evaluate.py failed "
            f"(exit={completed.returncode}): {detail}"
        )
    discovered = discover_official_results([output_dir, repo_root, repo_root / "output"])
    if discovered is None:
        raise FileNotFoundError(
            "EWMBench upstream evaluate.py did not produce ewmbm_final_table.csv under "
            f"{repo_root}"
        )
    return discovered


def run_ewmbench_scorer(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: EwmbenchScorerConfig,
    task_manifest_path: Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ewmbench_results.csv"
    prompt_records = unique_prompt_records(load_prompt_records(task_manifest_path=task_manifest_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_ids = {record["prompt_id"] for record in prompt_records}
    matched_videos = _matching_videos(generated_artifact_dir=generated_artifact_dir, prompt_ids=prompt_ids)

    if config.backend == "mock":
        seed = f"ewmbench-mock:{len(matched_videos)}:{limit or 'all'}"
        _write_mock_csv(output_path=results_path, seed=seed, prompt_records=prompt_records)
        return {
            "backend": "mock",
            "results_path": str(results_path.resolve()),
            "video_count": len(matched_videos),
            "prompt_count": len(prompt_records),
        }

    repo_root = config.repo_root
    if repo_root is None:
        raise FileNotFoundError(
            "EWMBench in-tree runtime is missing. Expected runtime/ewmbench next to this runner, "
            "or set WORLDFOUNDRY_EWMBENCH_ROOT to an equivalent local copy."
        )
    config_path = config.config_path
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            "EWMBench config.yaml is required for official scoring. Set WORLDFOUNDRY_EWMBENCH_CONFIG_PATH."
        )
    upstream_results = _run_upstream_evaluate(
        repo_root=repo_root,
        config_path=config_path,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    materialize_file(upstream_results, results_path)
    return {
        "backend": "official",
        "results_path": str(results_path.resolve()),
        "upstream_results_path": str(upstream_results.resolve()),
        "repo_root": str(repo_root.resolve()),
        "config_path": str(config_path.resolve()),
        "video_count": len(matched_videos),
        "prompt_count": len(prompt_records),
    }
