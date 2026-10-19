"""VideoVerse prompt materialization for model generation and artifact layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.filesystem.file_utils import materialize_file
from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.execution.framework.runner_common import (
    iter_generation_video_artifacts,
    reset_generation_artifact_staging,
    resolve_env_path,
)
from worldfoundry.evaluation.utils import write_jsonl

from .run_videoverse_official_runner import (
    _default_prompt_manifest,
    _load_json,
)


def resolve_prompt_manifest_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_path = resolve_env_path("WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST")
    if env_path is not None:
        return env_path
    default = _default_prompt_manifest()
    if default is None:
        raise ValueError(
            "VideoVerse prompt manifest is missing. Restore the bundled asset under "
            "worldfoundry/data/benchmarks/assets/videoverse/ or set "
            "WORLDFOUNDRY_VIDEOVERSE_PROMPT_MANIFEST."
        )
    return default


def load_prompt_manifest(explicit: Path | None = None) -> dict[str, dict[str, Any]]:
    path = resolve_prompt_manifest_path(explicit)
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"VideoVerse prompt manifest must be an object keyed by prompt id: {path}")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, Mapping)}


def _t2v_prompt(record: Mapping[str, Any]) -> str:
    following = record.get("t2v_following_prompt")
    if isinstance(following, Mapping):
        prompt = following.get("t2v_prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    for key in ("prompt", "t2v_prompt", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def materialize_videoverse_generation_requests(
    *,
    limit: int | None = None,
    prompt_manifest_path: Path | None = None,
    split: str = "default",
) -> tuple[GenerationRequest, ...]:
    """Build GenerationRequest rows for VideoVerse T2V model-benchmark runs."""
    manifest = load_prompt_manifest(prompt_manifest_path)
    prompt_ids = sorted(manifest)
    if limit is not None:
        prompt_ids = prompt_ids[: int(limit)]
    requests: list[GenerationRequest] = []
    for prompt_id in prompt_ids:
        record = manifest[prompt_id]
        prompt = _t2v_prompt(record)
        if not prompt:
            continue
        requests.append(
            GenerationRequest(
                sample_id=prompt_id,
                task_name="videoverse",
                split=split,
                inputs={
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "generation_text": prompt,
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_videoverse_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
) -> tuple[int, int]:
    """Copy model outputs into ``{prompt_id}.mp4`` files expected by VideoVerse."""
    rows: list[dict[str, Any]] = []
    reset_generation_artifact_staging(generation_output_dir, generated_artifact_dir)
    materialized = 0
    for result, artifact, source_path in iter_generation_video_artifacts(generation_output_dir, output_artifact):
        destination = generated_artifact_dir / f"{result.sample_id}.mp4"
        row = {
            "sample_id": result.sample_id,
            "artifact_name": output_artifact,
            "source_uri": artifact.uri,
            "destination": str(destination),
            "status": "missing",
            "placeholder": False,
        }
        if source_path is not None and source_path.is_file():
            if source_path.resolve() != destination.resolve():
                materialize_file(source_path, destination, writable=False)
            row["status"] = "copied"
            materialized += 1
        rows.append(row)
    write_jsonl(artifact_manifest_path, rows, atomic=False)
    return materialized, 0


def export_prompt_manifest_subset(
    *,
    output_path: Path,
    limit: int | None = None,
    prompt_manifest_path: Path | None = None,
) -> Path:
    manifest = load_prompt_manifest(prompt_manifest_path)
    prompt_ids = sorted(manifest)
    if limit is not None:
        prompt_ids = prompt_ids[: int(limit)]
    payload = {prompt_id: manifest[prompt_id] for prompt_id in prompt_ids}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
