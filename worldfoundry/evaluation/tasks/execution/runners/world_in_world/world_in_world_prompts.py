"""World-in-World closed-loop task prompt materialization."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    first_existing_dir,
    resolve_env_path,
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)

BENCHMARK_ID = "world-in-world"
METRICS_JSON_NAME = "metrics.json"

TASK_EPISODE_PATHS: dict[str, tuple[Path, ...]] = {
    "AR": (
        Path("data/WIW_datasets/eval_datasets/AR/episodes_AR.json.gz"),
        Path("data/WIW_datasets/eval_datasets/AR"),
    ),
    "AEQA": (
        Path("data/WIW_datasets/eval_datasets/AEQA/episodes_AEQA.json.gz"),
        Path("subtrees/open-eqa/data/open-eqa-184.json"),
        Path("subtrees/open-eqa/data/open-eqa-41.json"),
    ),
    "IGNav": (
        Path("data/WIW_datasets/eval_datasets/IGNav/episodes_IGNav.json.gz"),
        Path("data/WIW_datasets/eval_datasets/IGNav/igdataset_goal_imgs.zip"),
    ),
}

CANONICAL_TASKS = ("AR", "IGNav", "AEQA", "Manip")
CANONICAL_AEQA_PROMPT_COUNT = 184

DEFAULT_TASK = "AEQA"



def resolve_world_in_world_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_WORLD_IN_WORLD_ASSETS_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_episode_source(
    task: str,
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.exists() else None
    env_key = f"WORLDFOUNDRY_WORLD_IN_WORLD_{task.upper()}_EPISODES"
    env_path = resolve_env_path(env_key)
    if env_path is not None:
        return env_path if env_path.exists() else None
    env_manifest = resolve_env_path("WORLDFOUNDRY_WORLD_IN_WORLD_PROMPT_MANIFEST")
    if env_manifest is not None:
        return env_manifest if env_manifest.exists() else None
    for relative in TASK_EPISODE_PATHS.get(task.upper(), ()):
        bundled = bundled_benchmark_asset(BENCHMARK_ID, relative)
        if bundled.exists():
            return bundled
    root = repo_root or resolve_world_in_world_root()
    if root is None:
        return None
    for relative in TASK_EPISODE_PATHS.get(task.upper(), ()):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _prompt_id_for_record(*, task: str, record: Mapping[str, Any], index: int) -> str:
    if task == "AEQA":
        return str(record.get("question_id") or record.get("prompt_id") or index)
    if task in {"AR", "IGNav"}:
        scene_id = str(record.get("scene_id") or "scene").split("/")[-1].split(".")[0]
        episode_id = record.get("episode_id", index)
        return f"{scene_id}_E{int(episode_id):03d}"
    return str(record.get("prompt_id") or record.get("episode_id") or index)


def _prompt_text_for_record(*, task: str, record: Mapping[str, Any]) -> str:
    if task == "AEQA":
        question = str(record.get("question") or record.get("prompt") or "").strip()
        if question:
            return question
    if task == "AR":
        target = record.get("target_categrory") or record.get("target_category")
        if target:
            return f"Active recognition: identify {target} in the scene."
    if task == "IGNav":
        goal = record.get("goal_image") or record.get("instruction") or record.get("prompt")
        if goal:
            return str(goal)
    return str(record.get("prompt") or record.get("instruction") or record.get("question") or "").strip()


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return [dict(item) for item in episodes if isinstance(item, dict)]
        return [dict(payload)]
    raise ValueError(f"Unsupported episode payload at {path}")


def load_prompt_records(
    *,
    task: str | None = None,
    episodes_path: Path | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    resolved_task = (task or DEFAULT_TASK).strip()
    source = resolve_episode_source(resolved_task, explicit=episodes_path, repo_root=repo_root)
    if source is None:
        raise FileNotFoundError(
            f"World-in-World episodes for task {resolved_task} are missing. "
            "Set WORLDFOUNDRY_WORLD_IN_WORLD_ASSETS_ROOT or task-specific episode env vars."
        )
    raw_records = _load_json_records(source)
    records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        prompt_id = _prompt_id_for_record(task=resolved_task, record=record, index=index)
        prompt = _prompt_text_for_record(task=resolved_task, record=record)
        if not prompt_id:
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt or f"world-in-world:{resolved_task}:{prompt_id}",
                "task": resolved_task,
                "official_video_name": f"{prompt_id}.mp4",
                **record,
            }
        )
    if not records:
        raise ValueError(f"World-in-World prompt records are empty after validation: {source}")
    return records


def unique_prompt_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return records


def materialize_world_in_world_generation_requests(
    *,
    limit: int | None = None,
    task: str | None = None,
    repo_root: Path | None = None,
) -> tuple[GenerationRequest, ...]:
    root = repo_root or resolve_world_in_world_root()
    resolved_task = (task or os.environ.get("WORLDFOUNDRY_WORLD_IN_WORLD_TASK") or DEFAULT_TASK).strip()
    records = unique_prompt_records(load_prompt_records(task=resolved_task, repo_root=root))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        requests.append(
            GenerationRequest(
                sample_id=str(record["prompt_id"]),
                task_name="world-in-world",
                inputs={
                    "prompt": str(record["prompt"]),
                    "prompt_id": record["prompt_id"],
                    "task": record.get("task", resolved_task),
                    "official_video_name": record.get("official_video_name"),
                    "answer": record.get("answer"),
                    "category": record.get("category"),
                },
                output_schema={
                    "generated_video": {"kind": "video"},
                    "interaction_trace": {"kind": "structured_trace"},
                },
            )
        )
    return tuple(requests)
