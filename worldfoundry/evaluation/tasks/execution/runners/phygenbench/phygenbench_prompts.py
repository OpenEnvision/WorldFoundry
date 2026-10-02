"""PhyGenBench prompt materialization and generated-video layout helpers."""

from __future__ import annotations

import json
import os
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

BENCHMARK_ID = "phygenbench"
IN_TREE_PHYGENBENCH_ROOT = Path(__file__).resolve().parent / "runtime" / "phygenbench"
PROMPTS_JSON_REL = Path("PhyGenBench") / "prompts.json"
EXPLICIT_PROMPTS_JSON_REL = Path("PhyGenBench") / "explicit_prompts.json"
CANONICAL_PROMPT_COUNT = 160




def resolve_phygenbench_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_PHYGENBENCH_ROOT"),
        IN_TREE_PHYGENBENCH_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_prompts_json_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
    prefer_explicit_captions: bool | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PhyGenBench prompts JSON not found: {path}")
        return path
    env_manifest = resolve_env_path("WORLDFOUNDRY_PHYGENBENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"PhyGenBench prompts JSON not found: {env_manifest}")
        return env_manifest
    use_explicit = prefer_explicit_captions
    if use_explicit is None:
        use_explicit = os.environ.get("WORLDFOUNDRY_PHYGENBENCH_USE_EXPLICIT_PROMPTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if use_explicit:
        bundled_explicit = bundled_benchmark_asset(BENCHMARK_ID, EXPLICIT_PROMPTS_JSON_REL)
        if bundled_explicit.is_file():
            return bundled_explicit
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPTS_JSON_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_phygenbench_root()
    if root is None:
        raise FileNotFoundError(
            "PhyGenBench prompts JSON is missing. Set WORLDFOUNDRY_PHYGENBENCH_PROMPT_MANIFEST "
            "or WORLDFOUNDRY_PHYGENBENCH_ROOT."
        )
    if use_explicit:
        candidate = root / EXPLICIT_PROMPTS_JSON_REL
        if candidate.is_file():
            return candidate
    candidate = root / PROMPTS_JSON_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"PhyGenBench prompts JSON not found: {candidate}")
    return candidate


def _generation_text(row: dict[str, Any]) -> str:
    explicit = str(row.get("explicit_caption") or "").strip()
    if explicit:
        return explicit
    return str(row.get("caption") or row.get("prompt") or "").strip()


def load_prompt_records(*, prompts_json_path: Path | None = None) -> list[dict[str, Any]]:
    path = resolve_prompts_json_path(explicit=prompts_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"PhyGenBench prompts JSON must be a list: {path}")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        prompt_id = str(index + 1)
        prompt = _generation_text(row)
        if not prompt:
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt_index": index,
                "prompt": prompt,
                "caption": str(row.get("caption") or prompt),
                "physical_laws": row.get("physical_laws"),
                "sub_category": row.get("sub_category"),
                "main_category": row.get("main_category"),
            }
        )
    if not records:
        raise ValueError(f"PhyGenBench prompt records are empty after validation: {path}")
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
    prompt_id = str(record["prompt_id"])
    return f"output_video_{prompt_id}.mp4"


def video_filename_for_record(record: dict[str, Any]) -> str:
    return official_video_filename_for_record(record)


def materialize_phygenbench_generation_requests(
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
                task_name="phygenbench",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "generation_text": record["prompt"],
                    "caption": record.get("caption"),
                    "physical_laws": record.get("physical_laws"),
                    "main_category": record.get("main_category"),
                    "sub_category": record.get("sub_category"),
                    "official_video_name": official_video_filename_for_record(record),
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def _resolve_source_video(
    *,
    sample_id: str,
    source_path: Path,
    record_by_id: dict[str, dict[str, Any]],
) -> Path:
    if source_path.is_file():
        return source_path
    stem = source_path.stem
    if stem.isdigit():
        return source_path
    record = record_by_id.get(sample_id)
    if record is not None:
        official_name = official_video_filename_for_record(record)
        sibling = source_path.parent / official_name
        if sibling.is_file():
            return sibling
    return source_path


def copy_phygenbench_generated_videos(
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
        source_path = _resolve_source_video(
            sample_id=sample_id,
            source_path=Path(str(source)),
            record_by_id=record_by_id,
        )
        if not source_path.is_file():
            continue
        target_path = generated_artifact_dir / video_filename_for_record(
            record_by_id.get(sample_id, {"prompt_id": sample_id})
        )
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
            target_path = generated_artifact_dir / video_filename_for_record(record)
            if target_path.is_file():
                continue
            materialize_file(source_path, target_path, writable=False)
            materialized += 1
            manifest_rows.append(
                {"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target_path)}
            )

    write_jsonl(artifact_manifest_path, manifest_rows)
    return materialized, placeholders
