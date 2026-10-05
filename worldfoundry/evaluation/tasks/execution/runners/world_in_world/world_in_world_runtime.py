"""World-in-World evaluator runtime for WorldFoundry-generated artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io import materialize_file
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_official_runtime import (
    run_official_world_in_world_runtime,
)
from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_prompts import (
    DEFAULT_TASK,
    METRICS_JSON_NAME,
    load_prompt_records,
    resolve_world_in_world_root,
    unique_prompt_records,
)


def discover_metrics_json(search_roots: list[Path]) -> Path | None:
    metric_names = {METRICS_JSON_NAME, "world_in_world_metrics.json"}
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.name in metric_names:
            return root
        matches = sorted(
            path
            for name in metric_names
            for path in root.glob(f"**/{name}")
        )
        if matches:
            return matches[-1]
    return None



@dataclass(frozen=True)
class WorldInWorldRuntimeConfig:
    backend: str
    repo_root: Path | None
    task: str
    exp_id: str
    strict: bool = False


def runtime_config_from_env(
    *,
    task: str | None = None,
    exp_id: str | None = None,
    repo_root: Path | None = None,
) -> WorldInWorldRuntimeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_RUNTIME_BACKEND")
        or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_SCORER_BACKEND")
        or "official"
    ).strip().lower()
    resolved_task = (task or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_TASK") or DEFAULT_TASK).strip()
    resolved_exp_id = exp_id or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_EXP_ID") or "worldfoundry_artifact"
    strict = os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return WorldInWorldRuntimeConfig(
        backend=backend,
        repo_root=resolve_world_in_world_root(repo_root),
        task=resolved_task,
        exp_id=resolved_exp_id,
        strict=strict,
    )


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


def _env_metrics_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def _artifact_metrics_path(*, generated_artifact_dir: Path | None, output_dir: Path) -> Path | None:
    candidates = [_env_metrics_path()]
    if generated_artifact_dir is not None:
        candidates.append(discover_metrics_json([generated_artifact_dir]))
        candidates.extend(
            path
            for path in (
                generated_artifact_dir / "world_in_world_metrics.json",
                generated_artifact_dir / METRICS_JSON_NAME,
            )
            if path.is_file()
        )
    candidates.append(discover_metrics_json([output_dir]))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _copy_artifact_results(*, source_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialize_file(source_path, output_path)
    return output_path


def run_world_in_world_evaluator(
    *,
    generated_artifact_dir: Path | None,
    output_dir: Path,
    config: WorldInWorldRuntimeConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "world_in_world_metrics.json"
    prompt_records: list[dict[str, Any]] = []
    try:
        prompt_records = unique_prompt_records(
            load_prompt_records(task=config.task, repo_root=config.repo_root)
        )
        if limit is not None:
            prompt_records = prompt_records[: int(limit)]
    except FileNotFoundError:
        if config.strict:
            raise
    prompt_ids = {str(record["prompt_id"]) for record in prompt_records}
    matched_videos = (
        _matching_videos(generated_artifact_dir=generated_artifact_dir, prompt_ids=prompt_ids)
        if generated_artifact_dir is not None
        else []
    )

    if config.backend in {"official", "world-in-world", "world_in_world"}:
        summary = run_official_world_in_world_runtime(
            generated_artifact_dir=generated_artifact_dir,
            output_dir=output_dir,
            task=config.task,
            exp_id=config.exp_id,
        )
        summary.update(
            {
                "video_count": len(matched_videos),
                "prompt_count": len(prompt_records),
            }
        )
        return summary

    if config.backend in {"artifact", "worldfoundry"}:
        source_path = _artifact_metrics_path(
            generated_artifact_dir=generated_artifact_dir,
            output_dir=output_dir,
        )
        if source_path is None:
            raise FileNotFoundError(
                "World-in-World artifact evaluation requires an existing metrics file. "
                "Set WORLDFOUNDRY_WORLD_IN_WORLD_RESULTS_PATH or place metrics.json / "
                "world_in_world_metrics.json under --generated-artifact-dir."
            )
        _copy_artifact_results(source_path=source_path, output_path=results_path)
        return {
            "backend": "artifact",
            "results_path": str(results_path.resolve()),
            "source_results_path": str(source_path.resolve()),
            "task": config.task,
            "exp_id": config.exp_id,
            "video_count": len(matched_videos),
            "prompt_count": len(prompt_records),
        }

    raise ValueError(
        "Unsupported World-in-World runtime backend "
        f"{config.backend!r}. Use 'artifact' to import metrics produced by the "
        "WorldFoundry evaluation pipeline."
    )
