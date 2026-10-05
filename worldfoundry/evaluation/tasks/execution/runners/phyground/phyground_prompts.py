"""PhyGround prompt materialization and generated-video layout helpers."""

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
from worldfoundry.evaluation.utils import write_jsonl

BENCHMARK_ID = "phyground"
IN_TREE_PHYGROUND_ROOT = Path(__file__).resolve().parent / "runtime" / "phyground"
PROMPTS_JSON_REL = Path("data/prompts/phyground.json")
PROMPTS_JSON_ALT_REL = Path("prompts/phyground.json")
FIRST_IMAGES_REL = Path("data/first_images")
FIRST_IMAGES_ALT_REL = Path("first_images")
CANONICAL_PROMPT_COUNT = 250




def resolve_phyground_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_PHYGROUND_ROOT"),
        IN_TREE_PHYGROUND_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_data_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_PHYGROUND_DATA_ROOT"),
        resolve_env_path("WORLDFOUNDRY_BENCHMARK_DATA_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_prompts_json_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PhyGround prompts JSON not found: {path}")
        return path
    env_manifest = resolve_env_path("WORLDFOUNDRY_PHYGROUND_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"PhyGround prompts JSON not found: {env_manifest}")
        return env_manifest
    for rel in (PROMPTS_JSON_REL, PROMPTS_JSON_ALT_REL):
        bundled = bundled_benchmark_asset(BENCHMARK_ID, rel)
        if bundled.is_file():
            return bundled
    root = data_root or resolve_data_root()
    if root is not None:
        for rel in (PROMPTS_JSON_REL, PROMPTS_JSON_ALT_REL):
            candidate = root / rel
            if candidate.is_file():
                return candidate
        candidate = root / "prompts" / "phyground.json"
        if candidate.is_file():
            return candidate
    repo = repo_root or resolve_phyground_root()
    if repo is not None:
        for rel in (PROMPTS_JSON_REL, PROMPTS_JSON_ALT_REL):
            candidate = repo / rel
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "PhyGround prompts JSON is missing. Set WORLDFOUNDRY_PHYGROUND_DATA_ROOT, "
        "WORLDFOUNDRY_PHYGROUND_PROMPT_MANIFEST, or WORLDFOUNDRY_PHYGROUND_ROOT."
    )


def resolve_first_images_dir(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
    data_root: Path | None = None,
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    env_dir = resolve_env_path("WORLDFOUNDRY_PHYGROUND_FIRST_IMAGES_DIR")
    if env_dir is not None and env_dir.is_dir():
        return env_dir
    root = data_root or resolve_data_root()
    if root is not None:
        for rel in (FIRST_IMAGES_REL, FIRST_IMAGES_ALT_REL):
            candidate = root / rel
            if candidate.is_dir():
                return candidate
    repo = repo_root or resolve_phyground_root()
    if repo is not None:
        for rel in (FIRST_IMAGES_REL, FIRST_IMAGES_ALT_REL):
            candidate = repo / rel
            if candidate.is_dir():
                return candidate
    return None


def _load_prompt_payload(*, prompts_json_path: Path | None = None) -> list[dict[str, Any]]:
    path = resolve_prompts_json_path(explicit=prompts_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_prompts = payload.get("prompts")
        if isinstance(raw_prompts, dict):
            for key, row in raw_prompts.items():
                if isinstance(row, dict):
                    records.append({"prompt_id": str(row.get("video") or key), **row})
        elif isinstance(raw_prompts, list):
            for index, row in enumerate(raw_prompts):
                if isinstance(row, dict):
                    records.append({"prompt_id": str(row.get("video") or index), **row})
    if not records:
        raise ValueError(f"PhyGround prompts JSON is empty or unsupported: {path}")
    return records


def load_prompt_records(*, prompts_json_path: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _load_prompt_payload(prompts_json_path=prompts_json_path):
        prompt_id = str(row.get("video") or row.get("prompt_id") or "").strip()
        prompt = str(row.get("prompt") or row.get("description") or "").strip()
        if not prompt_id or not prompt:
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "physical_laws": row.get("physical_laws"),
                "first_frame_image": str(row.get("first_frame_image") or ""),
                "domain": str(row.get("_domain") or row.get("domain") or ""),
            }
        )
    if not records:
        raise ValueError("PhyGround prompt records are empty after validation.")
    return records


def unique_generation_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        records.append(row)
    return sorted(records, key=lambda item: item["prompt_id"])


def video_filename_for_record(record: dict[str, Any]) -> str:
    prompt_id = record["prompt_id"]
    return prompt_id if prompt_id.endswith(".mp4") else f"{prompt_id}.mp4"


def resolve_conditioning_image(record: dict[str, Any], *, first_images_dir: Path | None) -> str | None:
    image_name = str(record.get("first_frame_image") or "").strip()
    if not image_name or first_images_dir is None:
        return None
    candidate = first_images_dir / image_name
    return str(candidate.resolve()) if candidate.is_file() else None


def materialize_phyground_generation_requests(
    *,
    limit: int | None = None,
    prompts_json_path: Path | None = None,
    first_images_dir: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_json_path))
    images_dir = first_images_dir or resolve_first_images_dir()
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        conditioning_image = resolve_conditioning_image(record, first_images_dir=images_dir)
        inputs: dict[str, Any] = {
            "prompt": record["prompt"],
            "prompt_id": sample_id,
            "generation_text": record["prompt"],
            "physical_laws": record.get("physical_laws"),
            "domain": record.get("domain"),
        }
        if conditioning_image is not None:
            inputs["conditioning_image"] = conditioning_image
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="phyground",
                split=split,
                inputs=inputs,
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_phyground_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
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
        target_path = generated_artifact_dir / video_filename_for_record({"prompt_id": sample_id})
        materialize_file(source_path, target_path, writable=False)
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
            target_path = generated_artifact_dir / video_filename_for_record({"prompt_id": result.sample_id})
            if target_path.is_file():
                continue
            materialize_file(source_path, target_path, writable=False)
            materialized += 1
            manifest_rows.append(
                {"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target_path)}
            )

    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders
