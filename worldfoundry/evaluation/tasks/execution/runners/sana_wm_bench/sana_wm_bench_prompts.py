"""Materialize SANA-WM Bench generation requests from its public manifests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, is_generation_result_successful, local_path_for_uri

BENCHMARK_ID = "sana-wm-bench"
SPLITS = {
    "simple_60s": "benchmark_v2_smooth_60s",
    "hard_60s": "benchmark_v2_hard_60s",
}
CANONICAL_SCENE_COUNT = 80
CANONICAL_FRAME_COUNT = 961
CANONICAL_FPS = 16


def normalize_split(split: str | None) -> str:
    value = (split or "simple_60s").strip().lower().replace("-", "_")
    aliases = {"simple": "simple_60s", "hard": "hard_60s"}
    value = aliases.get(value, value)
    if value not in SPLITS:
        raise ValueError(f"unsupported SANA-WM Bench split {split!r}; expected one of {tuple(SPLITS)}")
    return value


def split_manifest_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / SPLITS[split] / "sanawm_export_v2" / "run_manifest.jsonl"


def load_manifest_rows(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    path = split_manifest_path(dataset_root, split)
    if not path.is_file():
        raise FileNotFoundError(f"SANA-WM Bench run manifest is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"SANA-WM Bench manifest must contain JSON objects: {path}")
    ids = [str(row.get("id") or "") for row in rows]
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"SANA-WM Bench manifest has missing or duplicate scene ids: {path}")
    return [dict(row) for row in rows]


def materialize_sana_wm_bench_generation_requests(
    *,
    limit: int | None = None,
    dataset_root: Path | None = None,
    split: str | None = None,
) -> tuple[GenerationRequest, ...]:
    """Return official image/prompt/camera requests for one public split."""
    if dataset_root is None:
        raise ValueError("SANA-WM Bench requests require --dataset-root / WORLDFOUNDRY_SANA_WM_BENCH_DATASET_ROOT")
    root = Path(dataset_root).expanduser().resolve()
    selected = normalize_split(split)
    requests: list[GenerationRequest] = []
    for row in load_manifest_rows(root, selected):
        scene_id = str(row["id"])
        image_rel = str(row.get("image_path") or f"images/{scene_id}.png")
        camera_rel = str(row.get("camera_path") or "")
        if not camera_rel:
            raise ValueError(f"SANA-WM Bench manifest row has no camera_path: {scene_id}")
        image_path = root / image_rel
        camera_path = root / camera_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"SANA-WM Bench first frame is missing: {image_path}")
        if not camera_path.is_file():
            raise FileNotFoundError(f"SANA-WM Bench camera trajectory is missing: {camera_path}")
        requests.append(
            GenerationRequest(
                sample_id=f"{selected}-{scene_id}",
                task_name=BENCHMARK_ID,
                split=selected,
                inputs={
                    "prompt": str(row.get("prompt") or scene_id),
                    "first_frame": str(image_path.resolve()),
                    "camera_path": str(camera_path.resolve()),
                    "scene_id": scene_id,
                    "official_video_name": f"{scene_id}_generated.mp4",
                    "fps": CANONICAL_FPS,
                    "num_frames": CANONICAL_FRAME_COUNT,
                },
                controls={"camera_trajectory": "c2w_npz", "camera_intrinsics": "embedded_npz"},
                output_schema={"generated_video": {"kind": "video"}},
            )
        )
        if limit is not None and len(requests) >= int(limit):
            break
    return tuple(requests)


def copy_sana_wm_bench_generated_videos(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str,
) -> tuple[int, int]:
    """Copy successful generated videos to the exact official filename layout."""
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    placeholders = 0
    requests_path = generation_output_dir / "requests.jsonl"
    results_path = generation_output_dir / "results.jsonl"
    if not requests_path.is_file() or not results_path.is_file():
        raise FileNotFoundError("SANA-WM Bench generation must contain requests.jsonl and results.jsonl")
    names = {
        str(row.get("sample_id")): str((row.get("inputs") or {}).get("official_video_name") or "")
        for row in (json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = GenerationResult.from_dict(json.loads(line))
        if not is_generation_result_successful(result):
            continue
        artifact = result.artifacts.get(output_artifact) or result.artifacts.get("generated_video")
        if artifact is None:
            continue
        source = local_path_for_uri(artifact.uri, base_dir=generation_output_dir)
        if source is None or not source.is_file():
            continue
        official_name = names.get(result.sample_id) or f"{result.sample_id.rsplit('-', 1)[-1]}_generated.mp4"
        if Path(official_name).name != official_name or not official_name.endswith("_generated.mp4"):
            raise ValueError(f"unsafe SANA-WM Bench official video name: {official_name!r}")
        target = generated_artifact_dir / official_name
        shutil.copy2(source, target)
        rows.append({"sample_id": result.sample_id, "artifact": output_artifact, "path": str(target.resolve())})
        copied += 1
    artifact_manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return copied, placeholders
