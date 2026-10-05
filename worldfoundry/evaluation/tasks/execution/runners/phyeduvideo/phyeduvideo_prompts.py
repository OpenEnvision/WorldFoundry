"""PhyEduVideo prompt materialization and generated-video layout helpers."""

from __future__ import annotations

import json
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

BENCHMARK_ID = "phyeduvideo"
PROMPTS_DIR_REL = Path("Prompts")
PROMPTS_FILE_REL = PROMPTS_DIR_REL / "Prompts.json"
CAP_FILE_REL = PROMPTS_DIR_REL / "cap.json"
SA_FILE_REL = PROMPTS_DIR_REL / "SA.json"
PC1_FILE_REL = PROMPTS_DIR_REL / "PC-1.json"
PC2_FILE_REL = PROMPTS_DIR_REL / "PC-2.json"
PC3_FILE_REL = PROMPTS_DIR_REL / "PC-3.json"
SCRIPTS_DIR_REL = Path("scripts")

CANONICAL_PROMPT_COUNT = 205



def resolve_phyeduvideo_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_PHYEDUVIDEO_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def _resolve_repo_file(
    *,
    explicit: Path | None,
    env_name: str,
    relative: Path,
    repo_root: Path | None,
    label: str,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PhyEduVideo {label} not found: {path}")
        return path
    env_path = resolve_env_path(env_name)
    if env_path is not None:
        if not env_path.is_file():
            raise FileNotFoundError(f"PhyEduVideo {label} not found: {env_path}")
        return env_path
    bundled = bundled_benchmark_asset(BENCHMARK_ID, relative)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_phyeduvideo_root()
    if root is None:
        raise FileNotFoundError(
            f"PhyEduVideo {label} is missing. Set WORLDFOUNDRY_PHYEDUVIDEO_ROOT or {env_name}."
        )
    candidate = root / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"PhyEduVideo {label} not found: {candidate}")
    return candidate


def resolve_prompts_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    return _resolve_repo_file(
        explicit=explicit,
        env_name="WORLDFOUNDRY_PHYEDUVIDEO_PROMPTS_FILE",
        relative=PROMPTS_FILE_REL,
        repo_root=repo_root,
        label="prompt suite (Prompts.json)",
    )


def resolve_cap_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    return _resolve_repo_file(
        explicit=explicit,
        env_name="WORLDFOUNDRY_PHYEDUVIDEO_CAP_FILE",
        relative=CAP_FILE_REL,
        repo_root=repo_root,
        label="caption prompt file (cap.json)",
    )


def _flatten_prompt_payload(payload: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for concept in payload:
        concept_id = int(concept["Id"])
        concept_name = str(concept.get("concept") or "").strip()
        category = str(concept.get("category") or "").strip()
        teaching_points = concept.get("teaching_points")
        if not isinstance(teaching_points, dict):
            continue
        for teaching_point_id, teaching_point_data in teaching_points.items():
            if not isinstance(teaching_point_data, dict):
                continue
            prompt = str(teaching_point_data.get("prompt") or "").strip()
            teaching_point = str(teaching_point_data.get("teaching_point") or "").strip()
            if not prompt:
                continue
            prompt_id = f"Id{concept_id}_{teaching_point_id}"
            records.append(
                {
                    "prompt_id": prompt_id,
                    "concept_id": str(concept_id),
                    "teaching_point_id": str(teaching_point_id),
                    "concept": concept_name,
                    "category": category,
                    "prompt": prompt,
                    "teaching_point": teaching_point,
                }
            )
    return records


def load_prompt_records(*, prompts_path: Path | None = None) -> list[dict[str, str]]:
    path = resolve_prompts_path(explicit=prompts_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"PhyEduVideo prompt suite must be a JSON list: {path}")
    records = _flatten_prompt_payload(payload)
    if not records:
        raise ValueError(f"PhyEduVideo prompt suite is empty after flattening: {path}")
    return records


def load_cap_records(*, cap_path: Path | None = None) -> list[dict[str, str]]:
    path = resolve_cap_path(explicit=cap_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"PhyEduVideo cap.json must be a JSON list: {path}")
    records = _flatten_prompt_payload(payload)
    if not records:
        raise ValueError(f"PhyEduVideo cap.json is empty after flattening: {path}")
    return records


def unique_prompt_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for row in rows:
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return sorted(records, key=lambda item: (int(item["concept_id"]), item["teaching_point_id"]))


def video_filename_for_prompt_id(prompt_id: str) -> str:
    return f"{prompt_id}.mp4"


def materialize_phyeduvideo_generation_requests(
    *,
    limit: int | None = None,
    prompts_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_records(prompts_path=prompts_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        requests.append(
            GenerationRequest(
                sample_id=record["prompt_id"],
                task_name="phyeduvideo",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": record["prompt_id"],
                    "generation_text": record["prompt"],
                    "concept_id": record["concept_id"],
                    "teaching_point_id": record["teaching_point_id"],
                    "concept": record["concept"],
                    "category": record["category"],
                    "teaching_point": record["teaching_point"],
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_phyeduvideo_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    allow_placeholders: bool = False,
) -> tuple[int, int]:
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
        for record in unique_prompt_records(load_prompt_records())[:1]:
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
