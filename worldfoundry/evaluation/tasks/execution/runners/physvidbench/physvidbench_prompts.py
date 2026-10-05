"""PhysVidBench prompt materialization and artifact layout helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from worldfoundry.core.io.filesystem.file_utils import materialize_file
from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import (
    first_existing_dir,
    resolve_env_path,
    bundled_benchmark_asset,
    bundled_benchmark_assets_root,
)
from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_SUFFIXES
from worldfoundry.evaluation.utils import write_jsonl

BENCHMARK_ID = "physvidbench"
PROMPT_MANIFEST_REL = Path("prompts_questions.csv")




def resolve_physvidbench_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_PHYSVIDBENCH_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_prompt_manifest_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PhysVidBench prompt manifest not found: {path}")
        return path
    env_manifest = resolve_env_path("WORLDFOUNDRY_PHYSVIDBENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"PhysVidBench prompt manifest not found: {env_manifest}")
        return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_physvidbench_root()
    if root is None:
        raise FileNotFoundError(
            "PhysVidBench prompt manifest is missing. Set WORLDFOUNDRY_PHYSVIDBENCH_PROMPT_MANIFEST "
            "or WORLDFOUNDRY_PHYSVIDBENCH_ROOT."
        )
    candidate = root / PROMPT_MANIFEST_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"PhysVidBench prompt manifest not found: {candidate}")
    return candidate


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def load_prompt_rows(*, prompt_manifest_path: Path | None = None) -> list[dict[str, str]]:
    path = resolve_prompt_manifest_path(explicit=prompt_manifest_path)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _truthy(row.get("Upsampled")):
                continue
            prompt_id = str(row.get("PromptID") or "").strip()
            prompt = str(row.get("Prompt") or "").strip()
            if not prompt_id or not prompt:
                continue
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "question": str(row.get("Question") or "").strip(),
                    "types": str(row.get("Types") or "").strip(),
                    "difficulty": str(row.get("Difficulty") or "").strip(),
                }
            )
    if not rows:
        raise ValueError(f"PhysVidBench prompt manifest is empty after Upsampled filtering: {path}")
    return rows


def unique_prompt_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for row in rows:
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return sorted(records, key=lambda item: int(item["prompt_id"]) if item["prompt_id"].isdigit() else item["prompt_id"])


def video_filename_for_prompt_id(prompt_id: str) -> str:
    return f"{int(prompt_id):04d}.mp4" if prompt_id.isdigit() else f"{prompt_id}.mp4"


def materialize_physvidbench_generation_requests(
    *,
    limit: int | None = None,
    prompt_manifest_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_rows(prompt_manifest_path=prompt_manifest_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        requests.append(
            GenerationRequest(
                sample_id=record["prompt_id"],
                task_name="physvidbench",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": record["prompt_id"],
                    "generation_text": record["prompt"],
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_physvidbench_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    allow_placeholders: bool = False,
) -> tuple[int, int]:
    import json

    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
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
            continue
        target_path = generated_artifact_dir / video_filename_for_prompt_id(sample_id)
        materialize_file(source_path, target_path, writable=False)
        materialized += 1
        manifest_rows.append({"sample_id": sample_id, "artifact": output_artifact, "path": str(target_path)})
    if materialized == 0:
        for path in sorted(generation_output_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            sample_id = path.stem
            target_path = generated_artifact_dir / video_filename_for_prompt_id(sample_id)
            materialize_file(path, target_path, writable=False)
            materialized += 1
            manifest_rows.append({"sample_id": sample_id, "artifact": output_artifact, "path": str(target_path)})
    if materialized == 0 and allow_placeholders:
        for record in unique_prompt_records(load_prompt_rows())[:1]:
            target_path = generated_artifact_dir / video_filename_for_prompt_id(record["prompt_id"])
            target_path.write_bytes(b"placeholder")
            placeholders += 1
            manifest_rows.append(
                {
                    "sample_id": record["prompt_id"],
                    "artifact": output_artifact,
                    "path": str(target_path),
                    "placeholder": True,
                }
            )
    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders


def attach_generation_results(
    *,
    requests: tuple[GenerationRequest, ...],
    generation_output_dir: Path,
) -> tuple[GenerationResult, ...]:
    results: list[GenerationResult] = []
    for request in requests:
        sample_dir = generation_output_dir / request.sample_id
        result_path = sample_dir / "result.json"
        if result_path.is_file():
            import json

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(GenerationResult.from_dict(payload))
            continue
        results.append(
            GenerationResult(
                sample_id=request.sample_id,
                task_name=request.task_name,
                split=request.split,
                status="missing",
                outputs={},
            )
        )
    return tuple(results)
