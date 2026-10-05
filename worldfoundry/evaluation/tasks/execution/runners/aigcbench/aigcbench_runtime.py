"""AIGCBench artifact/result importer for WorldFoundry-generated evaluations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io import materialize_file
from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_metrics import METRIC_ORDER
from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_prompts import (
    load_prompt_records,
    resolve_prompt_manifest_path,
    unique_prompt_records,
)


@dataclass(frozen=True)
class AIGCBenchScorerConfig:
    backend: str
    prompt_manifest_path: Path | None
    strict: bool = False


def scorer_config_from_env() -> AIGCBenchScorerConfig:
    backend = os.environ.get("WORLDFOUNDRY_AIGCBENCH_SCORER_BACKEND", "artifact").strip().lower()
    prompt_manifest_path = resolve_prompt_manifest_path()
    strict = os.environ.get("WORLDFOUNDRY_AIGCBENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return AIGCBenchScorerConfig(
        backend=backend,
        prompt_manifest_path=prompt_manifest_path,
        strict=strict,
    )


def _matching_videos(*, generated_artifact_dir: Path, prompt_ids: set[str]) -> list[str]:
    if not generated_artifact_dir.is_dir():
        return []
    matched: list[str] = []
    for path in sorted(generated_artifact_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        if path.stem in prompt_ids:
            matched.append(path.stem)
    return matched


def discover_aigcbench_results(search_roots: list[Path]) -> Path | None:
    globs = (
        "results_summary.csv",
        "aigcbench_results*.csv",
        "aigcbench_results*.json",
        "*aigcbench*.csv",
        "*aigcbench*.json",
    )
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in {".csv", ".json"}:
            return root
        for pattern in globs:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[-1]
    return None


def _env_results_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_AIGCBENCH_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def _artifact_results_path(*, generated_artifact_dir: Path | None, output_dir: Path) -> Path | None:
    search_roots = [output_dir]
    if generated_artifact_dir is not None:
        search_roots.insert(0, generated_artifact_dir)
    for candidate in (_env_results_path(), discover_aigcbench_results(search_roots)):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _copy_artifact_results(*, source_path: Path, output_dir: Path) -> Path:
    suffix = source_path.suffix if source_path.suffix else ".csv"
    output_path = output_dir / f"aigcbench_results{suffix}"
    materialize_file(source_path, output_path)
    return output_path


def run_aigcbench_scorer(
    *,
    generated_artifact_dir: Path,
    output_dir: Path,
    config: AIGCBenchScorerConfig,
    prompt_manifest_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = prompt_manifest_path or config.prompt_manifest_path
    prompt_records = unique_prompt_records(load_prompt_records(prompt_manifest_path=manifest))
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_ids = {record["prompt_id"] for record in prompt_records}
    matched_videos = _matching_videos(generated_artifact_dir=generated_artifact_dir, prompt_ids=prompt_ids)

    if config.backend not in {"artifact", "worldfoundry", "official"}:
        raise ValueError(
            "Unsupported AIGCBench scorer backend "
            f"{config.backend!r}. Use 'artifact' to import results produced by the "
            "WorldFoundry evaluation pipeline."
        )
    source_path = _artifact_results_path(
        generated_artifact_dir=generated_artifact_dir,
        output_dir=output_dir,
    )
    if source_path is None:
        raise FileNotFoundError(
            "AIGCBench artifact evaluation requires an existing CSV/JSON results file. "
            "Set WORLDFOUNDRY_AIGCBENCH_RESULTS_PATH or place aigcbench_results*.csv/json "
            "under --generated-artifact-dir."
        )
    results_path = _copy_artifact_results(source_path=source_path, output_dir=output_dir)
    return {
        "backend": "artifact",
        "results_path": str(results_path.resolve()),
        "results_csv": str(results_path.resolve()) if results_path.suffix.lower() == ".csv" else None,
        "source_results_path": str(source_path.resolve()),
        "video_count": len(matched_videos),
        "prompt_count": len(prompt_records),
        "prompt_manifest": None if manifest is None else str(manifest),
        "metric_ids": list(METRIC_ORDER),
    }
