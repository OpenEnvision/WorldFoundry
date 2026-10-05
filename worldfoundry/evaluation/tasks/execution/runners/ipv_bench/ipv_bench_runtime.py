"""IPV-Bench artifact/result importer for WorldFoundry-generated evaluations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io import materialize_file
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_metrics import (
    METRIC_ORDER,
)
from worldfoundry.evaluation.tasks.execution.runners.ipv_bench.ipv_bench_prompts import (
    load_prompt_records,
    unique_prompt_records,
)


def discover_official_results(search_roots: list[Path]) -> Path | None:
    globs = (
        "*_pred_ipv_judgement.json",
        "*_pred_ipv_mcqa.json",
        "*_pred_ipv_openqa.json",
        "ipv_results*.csv",
        "ipv_results*.json",
        "ipv_results*.jsonl",
    )
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file():
            return root
        for pattern in globs:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[-1]
    return None



@dataclass(frozen=True)
class IpvBenchRuntimeConfig:
    backend: str
    strict: bool = False


def runtime_config_from_env() -> IpvBenchRuntimeConfig:
    backend = (
        os.environ.get("WORLDFOUNDRY_IPV_BENCH_RUNTIME_BACKEND")
        or os.environ.get("WORLDFOUNDRY_IPV_BENCH_SCORER_BACKEND")
        or "artifact"
    ).strip().lower()
    strict = os.environ.get("WORLDFOUNDRY_IPV_BENCH_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
    return IpvBenchRuntimeConfig(
        backend=backend,
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


def _env_results_path() -> Path | None:
    value = os.environ.get("WORLDFOUNDRY_IPV_BENCH_RESULTS_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def _artifact_results_path(*, generated_artifact_dir: Path | None, output_dir: Path) -> Path | None:
    search_roots = [output_dir]
    if generated_artifact_dir is not None:
        search_roots.insert(0, generated_artifact_dir)
    for candidate in (_env_results_path(), discover_official_results(search_roots)):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _copy_artifact_results(*, source_path: Path, output_dir: Path) -> Path:
    suffix = source_path.suffix if source_path.suffix else ".json"
    output_path = output_dir / f"ipv_bench_results{suffix}"
    materialize_file(source_path, output_path)
    return output_path


def run_ipv_bench_evaluator(
    *,
    generated_artifact_dir: Path | None,
    output_dir: Path,
    config: IpvBenchRuntimeConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = unique_prompt_records(load_prompt_records())
    if limit is not None:
        prompt_records = prompt_records[: int(limit)]
    prompt_ids = {str(record["prompt_id"]) for record in prompt_records}
    matched_videos = (
        _matching_videos(generated_artifact_dir=generated_artifact_dir, prompt_ids=prompt_ids)
        if generated_artifact_dir is not None
        else []
    )

    if config.backend not in {"artifact", "worldfoundry", "official"}:
        raise ValueError(
            "Unsupported IPV-Bench runtime backend "
            f"{config.backend!r}. Use 'artifact' to import results produced by the "
            "WorldFoundry evaluation pipeline."
        )
    source_path = _artifact_results_path(
        generated_artifact_dir=generated_artifact_dir,
        output_dir=output_dir,
    )
    if source_path is None:
        raise FileNotFoundError(
            "IPV-Bench artifact evaluation requires an existing results file. "
            "Set WORLDFOUNDRY_IPV_BENCH_RESULTS_PATH or place ipv_results*.csv/json/jsonl "
            "or *_pred_ipv_*.json under --generated-artifact-dir."
        )
    results_path = _copy_artifact_results(source_path=source_path, output_dir=output_dir)
    return {
        "backend": "artifact",
        "results_path": str(results_path.resolve()),
        "source_results_path": str(source_path.resolve()),
        "video_count": len(matched_videos),
        "prompt_count": len(prompt_records),
        "metric_ids": list(METRIC_ORDER),
    }
