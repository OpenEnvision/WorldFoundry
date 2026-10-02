"""VideoPhy2 prompt materialization from bundled in-tree assets."""

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

BENCHMARK_ID = "videophy2"
PROMPTS_JSON_REL = Path("prompts.json")
CANONICAL_PROMPT_COUNT = 200




def resolve_videophy2_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_VIDEOPHY2_ROOT"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_prompts_json_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VideoPhy2 prompts JSON not found: {path}")
        return path
    env_manifest = resolve_env_path("WORLDFOUNDRY_VIDEOPHY2_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"VideoPhy2 prompts JSON not found: {env_manifest}")
        return env_manifest
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPTS_JSON_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_videophy2_root()
    if root is None:
        raise FileNotFoundError(
            "VideoPhy2 prompts JSON is missing. Set WORLDFOUNDRY_VIDEOPHY2_PROMPT_MANIFEST "
            "or provide bundled assets under worldfoundry/data/benchmarks/assets/videophy2/."
        )
    candidate = root / PROMPTS_JSON_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"VideoPhy2 prompts JSON not found: {candidate}")
    return candidate


def _physics_rules(row: dict[str, Any]) -> list[str]:
    rules = row.get("physics_rules") or row.get("rules") or []
    if isinstance(rules, str):
        return [rules] if rules.strip() else []
    if isinstance(rules, list):
        return [str(item) for item in rules if str(item).strip()]
    return []


def load_prompt_records(*, prompts_json_path: Path | None = None) -> list[dict[str, Any]]:
    path = resolve_prompts_json_path(explicit=prompts_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"VideoPhy2 prompts JSON must be a list: {path}")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        prompt_id = str(row.get("prompt_id") or index + 1)
        caption = str(row.get("caption") or row.get("prompt") or row.get("upsampled_caption") or "").strip()
        if not caption:
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt_index": index,
                "prompt": caption,
                "caption": caption,
                "physics_rules": _physics_rules(row),
                "is_hard": row.get("is_hard"),
                "action": row.get("action"),
                "category": row.get("category"),
            }
        )
    if not records:
        raise ValueError(f"VideoPhy2 prompt records are empty after validation: {path}")
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
    return sorted(records, key=lambda item: int(item["prompt_id"]))


def official_video_filename_for_record(record: dict[str, Any]) -> str:
    return f"{record['prompt_id']}.mp4"


def materialize_videophy2_generation_requests(
    *,
    limit: int | None = None,
    prompts_json_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_json_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="videophy2",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "generation_text": record["prompt"],
                    "caption": record.get("caption"),
                    "physics_rules": record.get("physics_rules"),
                    "official_video_name": official_video_filename_for_record(record),
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_videophy2_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    prompts_json_path: Path | None = None,
) -> tuple[int, int]:
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    records = unique_generation_records(load_prompt_records(prompts_json_path=prompts_json_path))
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
            continue
        record = record_by_id.get(sample_id, {"prompt_id": sample_id})
        target_path = generated_artifact_dir / official_video_filename_for_record(record)
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
            record = record_by_id.get(result.sample_id, {"prompt_id": result.sample_id})
            target_path = generated_artifact_dir / official_video_filename_for_record(record)
            if target_path.is_file():
                continue
            materialize_file(source_path, target_path, writable=False)
            materialized += 1
            manifest_rows.append(
                {"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target_path)}
            )

    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders
