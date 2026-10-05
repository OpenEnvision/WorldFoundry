"""PhyGround judge runtime for WorldFoundry-generated artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io import materialize_file
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_prompts import (
    load_prompt_records,
    resolve_phyground_root,
    resolve_prompts_json_path,
    unique_generation_records,
)



@dataclass(frozen=True)
class PhyGroundJudgeConfig:
    backend: str
    repo_root: Path | None
    prompts_json_path: Path | None
    prompt_config: str = "default.yaml"
    api_base: str | None = None
    strict: bool = False


def judge_config_from_env() -> PhyGroundJudgeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_PHYGROUND_JUDGE_BACKEND")
        or os.environ.get("WORLDFOUNDRY_PHYGROUND_SCORER_BACKEND")
        or "artifact"
    ).strip().lower()
    repo_root = resolve_phyground_root()
    prompts_json_path = None
    try:
        prompts_json_path = resolve_prompts_json_path(repo_root=repo_root)
    except FileNotFoundError:
        prompts_json_path = None
    strict = os.environ.get("WORLDFOUNDRY_PHYGROUND_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return PhyGroundJudgeConfig(
        backend=backend,
        repo_root=repo_root,
        prompts_json_path=prompts_json_path,
        prompt_config=os.environ.get("WORLDFOUNDRY_PHYGROUND_PROMPT_CONFIG", "default.yaml"),
        api_base=os.environ.get("WORLDFOUNDRY_PHYGROUND_JUDGE_API_BASE"),
        strict=strict,
    )


def _video_stems(generated_artifact_dir: Path) -> list[str]:
    if not generated_artifact_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in generated_artifact_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _env_scores_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_PHYGROUND_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def _artifact_scores_path(*, generated_artifact_dir: Path, output_dir: Path) -> Path | None:
    for candidate in (
        _env_scores_path(),
        generated_artifact_dir / "scores.json",
        generated_artifact_dir / "phyground_scores.json",
        output_dir / "scores.json",
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def run_phyground_judge(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: PhyGroundJudgeConfig,
    prompts_json_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_json = output_dir / "scores.json"
    prompts_path = prompts_json_path or config.prompts_json_path
    if prompts_path is None:
        prompts_path = resolve_prompts_json_path(repo_root=config.repo_root)
    prompt_records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_path))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]

    if config.backend not in {"artifact", "worldfoundry"}:
        raise ValueError(
            "PhyGround runner no longer launches benchmark-local shell judges. "
            "Run the judge through WorldFoundry/base-model infrastructure and pass "
            "the resulting scores.json via WORLDFOUNDRY_PHYGROUND_RESULTS_PATH or "
            "--generated-artifact-dir."
        )
    source_scores = _artifact_scores_path(generated_artifact_dir=generated_artifact_dir, output_dir=output_dir)
    if source_scores is None:
        raise FileNotFoundError(
            "PhyGround artifact evaluation requires scores.json. Set "
            "WORLDFOUNDRY_PHYGROUND_RESULTS_PATH or place scores.json under --generated-artifact-dir."
        )
    if source_scores.resolve() != scores_json.resolve():
        materialize_file(source_scores, scores_json)
        return {
            "backend": "artifact",
            "scores_json": str(scores_json.resolve()),
            "source_scores_json": str(source_scores.resolve()),
            "video_count": len(_video_stems(generated_artifact_dir)),
            "prompt_count": len(prompt_records),
            "prompts_json_path": str(prompts_path.resolve()),
        }
    return {
        "backend": "artifact",
        "scores_json": str(scores_json.resolve()),
        "source_scores_json": str(source_scores.resolve()),
        "video_count": len(_video_stems(generated_artifact_dir)),
        "prompt_count": len(prompt_records),
        "prompts_json_path": str(prompts_path.resolve()),
    }
