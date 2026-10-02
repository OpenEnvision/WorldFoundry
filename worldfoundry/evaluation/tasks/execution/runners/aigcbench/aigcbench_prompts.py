"""AIGCBench prompt materialization and generated-video layout helpers."""

from __future__ import annotations

import json
import re
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
from worldfoundry.evaluation.utils import worldfoundry_hfd_dataset_root, write_jsonl

BENCHMARK_ID = "aigcbench"
IN_TREE_AIGCBENCH_ROOT = Path(__file__).resolve().parent / "runtime" / "aigcbench"
WEBVID_LIST_REL = Path("webvid_eval_1000.txt")
LAION_LIST_REL = Path("Laion-aesthetics_select_samples.txt")
T2I_DIR_NAMES = ("t2i_625", "t2i_aspect_ratio_625", "AIGCBench/t2i_625")
PROMPT_MANIFEST_REL = Path("prompt_suite.json")

WEBVID_PROMPT_COUNT = 1000
LAION_PROMPT_COUNT = 925
DEFAULT_OURS_PROMPT_COUNT = 2002
CANONICAL_PROMPT_COUNT = WEBVID_PROMPT_COUNT + LAION_PROMPT_COUNT + DEFAULT_OURS_PROMPT_COUNT
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})



def resolve_aigcbench_root(explicit: Path | None = None) -> Path | None:
    return first_existing_dir(
        explicit,
        resolve_env_path("WORLDFOUNDRY_AIGCBENCH_ROOT"),
        IN_TREE_AIGCBENCH_ROOT,
        bundled_benchmark_assets_root(BENCHMARK_ID),
    )


def resolve_dataset_root(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    for candidate in (
        resolve_env_path("WORLDFOUNDRY_AIGCBENCH_DATASET_ROOT"),
        resolve_env_path("WORLDFOUNDRY_AIGCBENCH_PROMPT_SUITE_DIR"),
        bundled_benchmark_assets_root(BENCHMARK_ID),
    ):
        if candidate is not None and candidate.is_dir():
            return candidate
    hfd_root = worldfoundry_hfd_dataset_root()
    for local_name in ("stevenfan__AIGCBench_v1.0", "stevenfan/AIGCBench_v1.0"):
        candidate = hfd_root / local_name
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_dataset_file(
    *,
    explicit: Path | None,
    env_name: str,
    relative: Path,
    dataset_root: Path | None,
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_file() else None
    env_path = resolve_env_path(env_name)
    if env_path is not None:
        return env_path if env_path.is_file() else None
    bundled = bundled_benchmark_asset(BENCHMARK_ID, relative)
    if bundled.is_file():
        return bundled
    root = dataset_root or resolve_dataset_root()
    if root is None:
        return None
    candidate = root / relative
    return candidate if candidate.is_file() else None


def resolve_webvid_list_path(
    *,
    explicit: Path | None = None,
    dataset_root: Path | None = None,
) -> Path | None:
    return _resolve_dataset_file(
        explicit=explicit,
        env_name="WORLDFOUNDRY_AIGCBENCH_WEBVID_LIST",
        relative=WEBVID_LIST_REL,
        dataset_root=dataset_root,
    )


def resolve_laion_list_path(
    *,
    explicit: Path | None = None,
    dataset_root: Path | None = None,
) -> Path | None:
    return _resolve_dataset_file(
        explicit=explicit,
        env_name="WORLDFOUNDRY_AIGCBENCH_LAION_LIST",
        relative=LAION_LIST_REL,
        dataset_root=dataset_root,
    )


def resolve_prompt_manifest_path(
    *,
    explicit: Path | None = None,
    dataset_root: Path | None = None,
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_file() else None
    env_manifest = resolve_env_path("WORLDFOUNDRY_AIGCBENCH_PROMPT_MANIFEST")
    if env_manifest is not None:
        return env_manifest if env_manifest.is_file() else None
    bundled = bundled_benchmark_asset(BENCHMARK_ID, PROMPT_MANIFEST_REL)
    if bundled.is_file():
        return bundled
    root = dataset_root or resolve_dataset_root()
    if root is None:
        return None
    candidate = root / PROMPT_MANIFEST_REL
    return candidate if candidate.is_file() else None


def resolve_t2i_image_dir(*, dataset_root: Path | None = None) -> Path | None:
    env_dir = resolve_env_path("WORLDFOUNDRY_AIGCBENCH_T2I_DIR")
    if env_dir is not None and env_dir.is_dir():
        return env_dir
    root = dataset_root or resolve_dataset_root()
    if root is None:
        return None
    for name in T2I_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _parse_laion_line(line: str) -> dict[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if ".jpg " in stripped:
        image_name, prompt = stripped.split(".jpg ", 1)
        image_name = f"{image_name}.jpg"
    else:
        parts = stripped.split(" ", 1)
        if len(parts) != 2:
            return None
        image_name, prompt = parts
    prompt = prompt.strip().strip('"')
    if not image_name or not prompt:
        return None
    prompt_id = Path(image_name).stem
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_type": "laion",
        "reference_image": image_name,
    }


def _parse_webvid_line(line: str) -> dict[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    video_name = Path(stripped).name
    prompt_id = Path(video_name).stem
    return {
        "prompt_id": prompt_id,
        "prompt": "",
        "prompt_type": "webvid",
        "reference_video": video_name,
    }


def _parse_t2i_filename(path: Path) -> dict[str, str] | None:
    stem = path.stem
    match = re.match(r"^(\d+)_(.+)$", stem)
    if match is None:
        return None
    prompt_id = match.group(1)
    prompt = match.group(2).replace("_", " ").strip()
    if not prompt:
        return None
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_type": "ours",
        "reference_image": path.name,
    }


def _load_manifest_records(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("prompts") or payload.get("records") or payload.get("rows") or []
        else:
            raise ValueError(f"AIGCBench prompt manifest must be JSON list/object: {path}")
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported AIGCBench prompt manifest format: {path}")
    records: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt_id = str(row.get("prompt_id") or row.get("sample_id") or row.get("id") or "").strip()
        prompt = str(row.get("prompt") or row.get("text") or row.get("caption") or "").strip()
        prompt_type = str(row.get("prompt_type") or row.get("subset") or "standard").strip()
        if not prompt_id:
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_type": prompt_type,
                **{
                    key: str(value)
                    for key, value in row.items()
                    if key in {"reference_image", "reference_video", "reference_image_path", "reference_video_path"}
                    and value not in (None, "")
                },
            }
        )
    if not records:
        raise ValueError(f"AIGCBench prompt manifest is empty: {path}")
    return records


def load_prompt_records(
    *,
    prompt_manifest_path: Path | None = None,
    dataset_root: Path | None = None,
) -> list[dict[str, str]]:
    manifest = resolve_prompt_manifest_path(explicit=prompt_manifest_path, dataset_root=dataset_root)
    if manifest is not None:
        return _load_manifest_records(manifest)

    root = dataset_root or resolve_dataset_root()
    records: list[dict[str, str]] = []

    webvid_path = resolve_webvid_list_path(dataset_root=root)
    if webvid_path is not None:
        for line in webvid_path.read_text(encoding="utf-8").splitlines():
            row = _parse_webvid_line(line)
            if row is not None:
                records.append(row)

    laion_path = resolve_laion_list_path(dataset_root=root)
    if laion_path is not None:
        for line in laion_path.read_text(encoding="utf-8").splitlines():
            row = _parse_laion_line(line)
            if row is not None:
                records.append(row)

    t2i_dir = resolve_t2i_image_dir(dataset_root=root)
    if t2i_dir is not None:
        for path in sorted(t2i_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            row = _parse_t2i_filename(path)
            if row is not None:
                records.append(row)

    if not records:
        raise FileNotFoundError(
            "AIGCBench prompt suite is missing. Set WORLDFOUNDRY_AIGCBENCH_DATASET_ROOT, "
            "WORLDFOUNDRY_AIGCBENCH_PROMPT_MANIFEST, or place HF dataset files "
            "(webvid_eval_1000.txt, Laion-aesthetics_select_samples.txt, t2i_625/)."
        )
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
    return sorted(records, key=lambda item: (item.get("prompt_type", ""), item["prompt_id"]))


def video_filename_for_prompt_id(prompt_id: str) -> str:
    return f"{prompt_id}.mp4"


def materialize_aigcbench_generation_requests(
    *,
    limit: int | None = None,
    prompt_manifest_path: Path | None = None,
    dataset_root: Path | None = None,
    split: str = "standard",
) -> tuple[GenerationRequest, ...]:
    records = unique_prompt_records(
        load_prompt_records(prompt_manifest_path=prompt_manifest_path, dataset_root=dataset_root)
    )
    if limit is not None:
        records = records[: int(limit)]
    requests: list[GenerationRequest] = []
    for record in records:
        inputs: dict[str, Any] = {
            "prompt": record.get("prompt") or record["prompt_id"],
            "prompt_id": record["prompt_id"],
            "generation_text": record.get("prompt") or record["prompt_id"],
            "prompt_type": record.get("prompt_type") or "standard",
        }
        for key in ("reference_image", "reference_video", "reference_image_path", "reference_video_path"):
            if record.get(key):
                inputs[key] = record[key]
        requests.append(
            GenerationRequest(
                sample_id=record["prompt_id"],
                task_name="aigcbench",
                split=split,
                inputs=inputs,
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
    return tuple(requests)


def copy_aigcbench_generated_videos(
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
