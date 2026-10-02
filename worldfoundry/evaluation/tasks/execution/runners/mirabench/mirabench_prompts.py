"""MiraBench prompt materialization and generated-video layout helpers."""

from __future__ import annotations

import csv
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

BENCHMARK_ID = "mirabench"
IN_TREE_MIRABENCH_ROOT = Path(__file__).resolve().parent / "runtime" / "mirabench"
META_GENERATED_REL = Path("data/evaluation_example/meta_generated.csv")
META_GT_REL = Path("data/evaluation_example/meta_gt.csv")
CALCULATE_SCORE_REL = Path("calculate_score.py")
CANONICAL_PROMPT_COUNT = 150

META_COLUMNS = (
    "video_idx",
    "video_path",
    "short_caption",
    "dense_caption",
    "main_object_caption",
    "background_caption",
    "style_caption",
    "camera_caption",
)



def resolve_mirabench_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_MIRABENCH_ROOT"),
        IN_TREE_MIRABENCH_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_meta_csv_path(
    *,
    explicit: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MiraBench meta CSV not found: {path}")
        return path
    env_manifest = resolve_env_path("WORLDFOUNDRY_MIRABENCH_META_CSV")
    if env_manifest is not None:
        if not env_manifest.is_file():
            raise FileNotFoundError(f"MiraBench meta CSV not found: {env_manifest}")
        return env_manifest
    env_prompt = resolve_env_path("WORLDFOUNDRY_MIRABENCH_PROMPT_MANIFEST")
    if env_prompt is not None:
        if not env_prompt.is_file():
            raise FileNotFoundError(f"MiraBench meta CSV not found: {env_prompt}")
        return env_prompt
    bundled = bundled_benchmark_asset(BENCHMARK_ID, META_GENERATED_REL)
    if bundled.is_file():
        return bundled
    root = repo_root or resolve_mirabench_root()
    if root is None:
        raise FileNotFoundError(
            "MiraBench meta CSV is missing. Set WORLDFOUNDRY_MIRABENCH_META_CSV, "
            "WORLDFOUNDRY_MIRABENCH_PROMPT_MANIFEST, or WORLDFOUNDRY_MIRABENCH_ROOT."
        )
    candidate = root / META_GENERATED_REL
    if not candidate.is_file():
        raise FileNotFoundError(f"MiraBench meta CSV not found: {candidate}")
    return candidate


def _generation_text(row: dict[str, Any]) -> str:
    dense = str(row.get("dense_caption") or "").strip()
    if dense:
        return dense
    return str(row.get("short_caption") or row.get("prompt") or "").strip()


def load_prompt_records(*, meta_csv_path: Path | None = None) -> list[dict[str, Any]]:
    path = resolve_meta_csv_path(explicit=meta_csv_path)
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row:
                continue
            prompt_id = str(row.get("video_idx") or row.get("prompt_id") or "").strip()
            prompt = _generation_text(row)
            if not prompt_id or not prompt:
                continue
            records.append(
                {
                    "prompt_id": prompt_id,
                    "video_idx": prompt_id,
                    "prompt": prompt,
                    "short_caption": row.get("short_caption"),
                    "dense_caption": row.get("dense_caption"),
                    "main_object_caption": row.get("main_object_caption"),
                    "background_caption": row.get("background_caption"),
                    "style_caption": row.get("style_caption"),
                    "camera_caption": row.get("camera_caption"),
                    "video_path": row.get("video_path"),
                }
            )
    if not records:
        raise ValueError(f"MiraBench prompt records are empty after validation: {path}")
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
    return sorted(records, key=lambda item: int(item["prompt_id"]) if str(item["prompt_id"]).isdigit() else item["prompt_id"])


def official_video_filename_for_record(record: dict[str, Any]) -> str:
    return f"{record['prompt_id']}.mp4"


def video_filename_for_record(record: dict[str, Any]) -> str:
    return official_video_filename_for_record(record)


def materialize_mirabench_generation_requests(
    *,
    limit: int | None = None,
    meta_csv_path: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(load_prompt_records(meta_csv_path=meta_csv_path))
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        sample_id = record["prompt_id"]
        requests.append(
            GenerationRequest(
                sample_id=sample_id,
                task_name="mirabench",
                split=split,
                inputs={
                    "prompt": record["prompt"],
                    "prompt_id": sample_id,
                    "generation_text": record["prompt"],
                    "short_caption": record.get("short_caption"),
                    "dense_caption": record.get("dense_caption"),
                    "main_object_caption": record.get("main_object_caption"),
                    "background_caption": record.get("background_caption"),
                    "style_caption": record.get("style_caption"),
                    "camera_caption": record.get("camera_caption"),
                    "official_video_name": official_video_filename_for_record(record),
                },
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def write_meta_csv(
    *,
    records: list[dict[str, Any]],
    generated_artifact_dir: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(META_COLUMNS))
        writer.writeheader()
        for record in records:
            video_path = generated_artifact_dir / official_video_filename_for_record(record)
            writer.writerow(
                {
                    "video_idx": record["prompt_id"],
                    "video_path": str(video_path.resolve()),
                    "short_caption": record.get("short_caption") or "",
                    "dense_caption": record.get("dense_caption") or record.get("prompt") or "",
                    "main_object_caption": record.get("main_object_caption") or "",
                    "background_caption": record.get("background_caption") or "",
                    "style_caption": record.get("style_caption") or "",
                    "camera_caption": record.get("camera_caption") or "",
                }
            )
    return output_path


def _resolve_source_video(
    *,
    sample_id: str,
    source_path: Path,
    record_by_id: dict[str, dict[str, Any]],
) -> Path:
    if source_path.is_file():
        return source_path
    record = record_by_id.get(sample_id)
    if record is not None:
        official_name = official_video_filename_for_record(record)
        sibling = source_path.parent / official_name
        if sibling.is_file():
            return sibling
    return source_path


def copy_mirabench_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str = "generated_video",
    meta_csv_path: Path | None = None,
) -> tuple[int, int]:
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    records = unique_prompt_records(load_prompt_records(meta_csv_path=meta_csv_path))
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
