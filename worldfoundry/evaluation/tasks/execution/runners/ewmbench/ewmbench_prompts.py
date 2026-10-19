"""EWMBench prompt materialization from bundled task manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.core.io.filesystem.file_utils import materialize_file
from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.runner_common import (
    iter_generation_video_artifacts,
    reset_generation_artifact_staging,
)
from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_paths import load_task_manifest
from worldfoundry.evaluation.utils import write_jsonl


def load_prompt_records(*, task_manifest_path: Path | None = None) -> list[dict[str, Any]]:
    manifest = load_task_manifest(task_manifest_path=task_manifest_path)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("EWMBench task manifest must contain an episodes list")
    records: list[dict[str, Any]] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        episode_id = str(episode.get("episode_id") or "").strip()
        introduction = str(episode.get("introduction") or "").strip()
        if not episode_id or not introduction:
            continue
        records.append(
            {
                "prompt_id": episode_id,
                "episode_id": episode_id,
                "task_id": episode.get("task_id"),
                "episode_name": episode.get("episode_name"),
                "prompt": introduction,
                "introduction": introduction,
                "trials": episode.get("trials") or manifest.get("trials_per_episode") or [1],
            }
        )
    if not records:
        raise ValueError("EWMBench prompt records are empty after validation")
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


def official_video_filename_for_record(record: dict[str, Any]) -> str:
    return f"{record['prompt_id']}.mp4"


def materialize_ewmbench_generation_requests(
    *,
    limit: int | None = None,
    task_manifest_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_records(task_manifest_path=task_manifest_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="ewmbench",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "episode_id": record["episode_id"],
                    "task_id": record.get("task_id"),
                    "action_context": record.get("introduction"),
                    "generation_text": record["prompt"],
                    "trials": record.get("trials"),
                    "official_video_name": official_video_filename_for_record(record),
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_ewmbench_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    task_manifest_path: Path | None = None,
) -> tuple[int, int]:
    reset_generation_artifact_staging(generation_output_dir, generated_artifact_dir)
    records = unique_prompt_records(load_prompt_records(task_manifest_path=task_manifest_path))
    record_by_id = {record["prompt_id"]: record for record in records}
    materialized = 0
    placeholders = 0
    manifest_rows: list[dict[str, Any]] = []

    for result, _, source_path in iter_generation_video_artifacts(generation_output_dir, output_artifact):
        if source_path is None or not source_path.is_file():
            continue
        record = record_by_id.get(result.sample_id, {"prompt_id": result.sample_id})
        target_path = generated_artifact_dir / official_video_filename_for_record(record)
        if target_path.is_file():
            continue
        materialize_file(source_path, target_path, writable=False)
        materialized += 1
        manifest_rows.append({"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target_path)})

    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders
