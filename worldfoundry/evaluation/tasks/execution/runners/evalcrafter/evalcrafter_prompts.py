"""EvalCrafter prompt materialization from bundled prompt700.txt and metadata.json."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)
from worldfoundry.evaluation.utils import write_jsonl

BENCHMARK_ID = "evalcrafter"
PROMPT700_REL = Path("prompt700.txt")
METADATA_REL = Path("metadata.json")
CANONICAL_PROMPT_COUNT = 700
EXPECTED_VIDEO_COUNT = 700

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def resolve_evalcrafter_root(explicit: Path | None = None) -> Path | None:
    for candidate in (
        explicit,
        _env_path("WORLDFOUNDRY_EVALCRAFTER_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate.expanduser().resolve()
    return None


def resolve_prompt700_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EvalCrafter prompt700.txt not found: {path}")
        return path
    env_manifest = _env_path("WORLDFOUNDRY_EVALCRAFTER_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"EvalCrafter prompt700.txt not found: {env_manifest}")
        return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT700_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_evalcrafter_root()
    if root is None:
        raise FileNotFoundError(
            "EvalCrafter prompt700.txt is missing. Set WORLDFOUNDRY_EVALCRAFTER_PROMPT_MANIFEST "
            "or WORLDFOUNDRY_EVALCRAFTER_ROOT."
        )
    candidate = root / PROMPT700_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"EvalCrafter prompt700.txt not found: {candidate}")
    return candidate


def resolve_metadata_path(*, repo_root: Path | None = None) -> Path | None:
    env_metadata = _env_path("WORLDFOUNDRY_EVALCRAFTER_METADATA")
    if env_metadata is not None:
        return env_metadata if env_metadata.is_file() else None
    bundled = bundled_benchmark_asset(BENCHMARK_ID, METADATA_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_evalcrafter_root()
    if root is None:
        return None
    candidate = root / METADATA_REL
    return candidate if candidate.is_file() else None


def resolve_official_metric_prompt_path(prompt_id: str, *, repo_root: Path | None = None) -> Path:
    """Resolve the per-id prompt files read by EvalCrafter's official metric scripts."""

    filename = f"{prompt_id}.txt"
    candidates = (
        repo_root / "prompts" / filename if repo_root is not None else None,
        bundled_benchmark_assets_root(BENCHMARK_ID) / "prompts" / filename,
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"EvalCrafter official metric prompt is missing for {prompt_id}; expected prompts/{filename}"
    )


def load_prompt_lines(*, prompt700_path: Path | None = None) -> list[str]:
    path = resolve_prompt700_path(explicit=prompt700_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"EvalCrafter prompt700.txt is empty: {path}")
    return lines


def load_metadata(*, metadata_path: Path | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    path = metadata_path or resolve_metadata_path(repo_root=repo_root)
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_prompt_records(
    *,
    prompt700_path: Path | None = None,
    metadata_path: Path | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    lines = load_prompt_lines(prompt700_path=prompt700_path)
    metadata = load_metadata(metadata_path=metadata_path, repo_root=repo_root)
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(lines):
        prompt_id = f"{index:04d}"
        entry = metadata.get(prompt_id) if isinstance(metadata.get(prompt_id), dict) else {}
        attributes = entry.get("attributes") if isinstance(entry.get("attributes"), dict) else {}
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "attributes": attributes,
                "official_video_name": f"{prompt_id}.mp4",
            }
        )
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


def materialize_evalcrafter_generation_requests(
    *,
    limit: int | None = None,
    prompt700_path: Path | None = None,
    metadata_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(
        load_prompt_records(prompt700_path=prompt700_path, metadata_path=metadata_path)
    )
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="evalcrafter",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "generation_text": record["prompt"],
                    "attributes": record.get("attributes") or {},
                    "official_video_name": official_video_filename_for_record(record),
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_evalcrafter_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    prompt700_path: Path | None = None,
) -> tuple[int, int]:
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    records = unique_prompt_records(load_prompt_records(prompt700_path=prompt700_path))
    record_by_id = {record["prompt_id"]: record for record in records}
    materialized = 0
    placeholders = 0
    manifest_rows: list[dict[str, Any]] = []

    for sample_dir in sorted(path for path in generation_output_dir.iterdir() if path.is_dir()):
        result_path = sample_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        sample_id = str(payload.get("sample_id") or sample_dir.name)
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        source = outputs.get(output_artifact) or outputs.get("generated_video")
        if not source:
            continue
        source_path = Path(str(source))
        if not source_path.is_file():
            official_name = official_video_filename_for_record(
                record_by_id.get(sample_id, {"prompt_id": sample_id})
            )
            sibling = source_path.parent / official_name
            if sibling.is_file():
                source_path = sibling
        if not source_path.is_file():
            continue
        target_path = generated_artifact_dir / official_video_filename_for_record(
            record_by_id.get(sample_id, {"prompt_id": sample_id})
        )
        shutil.copy2(source_path, target_path)
        materialized += 1
        manifest_rows.append({"sample_id": sample_id, "artifact": output_artifact, "path": str(target_path)})

    results_path = generation_output_dir / "results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = GenerationResult.from_dict(json.loads(line))
            artifact = result.artifacts.get(output_artifact) or result.artifacts.get("generated_video")
            if artifact is None:
                continue
            from worldfoundry.evaluation.utils import local_path_for_uri

            source_path = local_path_for_uri(str(artifact))
            if source_path is None or not source_path.is_file():
                continue
            record = record_by_id.get(result.sample_id, {"prompt_id": result.sample_id})
            target_path = generated_artifact_dir / official_video_filename_for_record(record)
            if target_path.is_file():
                continue
            shutil.copy2(source_path, target_path)
            materialized += 1
            manifest_rows.append(
                {"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target_path)}
            )

    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders


def canonical_prompt_count(*, prompt700_path: Path | None = None) -> int:
    return len(load_prompt_lines(prompt700_path=prompt700_path))
