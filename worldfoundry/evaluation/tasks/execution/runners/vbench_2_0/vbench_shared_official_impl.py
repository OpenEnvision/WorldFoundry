#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import cache_root_path
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.utils import REPO_ROOT
from worldfoundry.runtime.env import resolve_hf_cache_dir  # type: ignore[reportMissingImports]

RUNNERS_ROOT = Path(__file__).resolve().parents[1]
BASE_VBENCH_RUNTIME_ROOT = RUNNERS_ROOT / "vbench" / "runtime"
IN_TREE_VBENCH_PLUS_PLUS_ROOT = RUNNERS_ROOT / "vbench_plus_plus" / "runtime"
IN_TREE_VBENCH2_ROOT = RUNNERS_ROOT / "vbench_2_0" / "runtime"
DEFAULT_VBENCH_ROOT = IN_TREE_VBENCH_PLUS_PLUS_ROOT
VBENCH_FULL_INFO_ASSET = bundled_benchmark_asset("vbench", "VBench_full_info.json")
VBENCH_PROMPTS_ASSET_ROOT = bundled_benchmark_asset("vbench", "prompts")
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_EXTENSIONS = frozenset({".mp4", ".gif", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
VBENCH2_HF_DATASETS = (
    "Vchitect/VBench-2.0_sampled_videos",
    "Vchitect/VBench-2.0_human_annotation",
    "Vchitect/VBench-2.0_human_anomaly",
)
VBENCH2_DIMENSION_FILES = {
    "camera_motion": "Camera_Motion.json",
    "complex_landscape": "Complex_Landscape.json",
    "complex_plot": "Complex_Plot.json",
    "composition": "Composition.json",
    "diversity": "Diversity.json",
    "dynamic_attribute": "Dynamic_Attribute.json",
    "dynamic_spatial_relationship": "Dynamic_Spatial_Relationship.json",
    "human_anatomy": "Human_Anatomy.json",
    "human_clothes": "Human_Clothes.json",
    "human_identity": "Human_Identity.json",
    "human_interaction": "Human_Interaction.json",
    "instance_preservation": "Instance_Preservation.json",
    "material": "Material.json",
    "mechanics": "Mechanics.json",
    "motion_order_understanding": "Motion_Order_Understanding.json",
    "motion_rationality": "Motion_Rationality.json",
    "multi_view_consistency": "Multi-View_Consistency.json",
    "thermotics": "Thermotics.json",
}
VBENCH2_ANNOTATION_EXTRA_FILES = ("VBench2_arena_feedback.csv",)
VBENCH2_ANOMALY_ARCHIVE_FILES = ("opensource.zip", "opensource.z01", "opensource.z02")
VBENCH2_CATEGORY_GROUPS = {
    "vbench2_creativity": ("composition", "diversity"),
    "vbench2_commonsense": ("instance_preservation", "motion_rationality"),
    "vbench2_controllability": (
        "camera_motion",
        "complex_landscape",
        "complex_plot",
        "dynamic_attribute",
        "dynamic_spatial_relationship",
        "human_interaction",
        "motion_order_understanding",
    ),
    "vbench2_human_fidelity": ("human_anatomy", "human_clothes", "human_identity"),
    "vbench2_physics": ("material", "mechanics", "multi_view_consistency", "thermotics"),
}
PLUS_VARIANT_AVERAGE = {
    "i2v": "vbench_plus_plus_i2v_average",
    "long": "vbench_plus_plus_long_average",
    "trustworthiness": "vbench_plus_plus_trustworthiness_average",
}
CANONICAL_FULL_INFO_ASSETS = {
    "i2v": bundled_benchmark_asset("vbench-plus-plus", "i2v", "vbench2_i2v_full_info.json"),
    "long": bundled_benchmark_asset("vbench-plus-plus", "long", "VBench_full_info.json"),
    "trustworthiness": bundled_benchmark_asset(
        "vbench-plus-plus", "trustworthiness", "vbench2_trustworthy.json"
    ),
    "vbench2": bundled_benchmark_asset("vbench-2.0", "VBench2_full_info.json"),
}
AGGREGATE_METRIC_IDS = frozenset(
    {
        *VBENCH2_CATEGORY_GROUPS,
        "vbench2_total",
        *PLUS_VARIANT_AVERAGE.values(),
        "vbench_plus_plus_average",
    }
)


def hf_cache_dir_candidates(explicit_cache_dir: Path | None) -> list[Path]:
    """Return local Hugging Face cache roots checked for VBench-2.0 datasets.

    Args:
        explicit_cache_dir: User-provided cache root from CLI or environment.
    """
    raw_paths = [
        explicit_cache_dir,
        env_path("WORLDFOUNDRY_HF_CACHE_DIR"),
        env_path("HF_HUB_CACHE"),
        (env_path("HF_HOME") / "hub") if env_path("HF_HOME") else None,
        REPO_ROOT / "cache" / "benchmark_zoo" / "hf_datasets",
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        path = raw_path.expanduser()
        key = str(path.resolve())
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def hf_cache_dataset_dir(cache_dir: Path, dataset_id: str) -> Path:
    """Build the Hugging Face cache directory name for a dataset repo.

    Args:
        cache_dir: Hugging Face cache root.
        dataset_id: Dataset repo id such as Vchitect/VBench-2.0_human_annotation.
    """
    return cache_dir / f"datasets--{dataset_id.replace('/', '--')}"


def vbench2_dataset_names(dataset_id: str) -> tuple[str, ...]:
    """Return common local directory names for a VBench-2.0 dataset.

    Args:
        dataset_id: Hugging Face dataset repo id.
    """
    owner, name = dataset_id.split("/", 1)
    return (name, f"{owner}--{name}", f"datasets--{owner}--{name}")


def snapshot_dirs(dataset_dir: Path) -> list[Path]:
    """List materialized Hugging Face snapshot directories.

    Args:
        dataset_dir: Hugging Face cache dataset directory.
    """
    snapshots = dataset_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return [path for path in sorted(snapshots.iterdir()) if path.is_dir()]


def vbench2_dataset_candidates(
    dataset_id: str,
    *,
    dataset_root: Path | None,
    hf_cache_dir: Path | None,
) -> list[Path]:
    """Return possible local roots for one VBench-2.0 dataset.

    Args:
        dataset_id: Hugging Face dataset repo id.
        dataset_root: User-provided root containing downloaded VBench-2.0 data.
        hf_cache_dir: User-provided or default Hugging Face cache root.
    """
    candidates: list[Path] = []
    if dataset_root is not None:
        root = dataset_root.expanduser()
        candidates.append(root)
        for name in vbench2_dataset_names(dataset_id):
            candidates.append(root / name)
        for name in vbench2_dataset_names(dataset_id):
            candidates.extend(snapshot_dirs(root / name))

    for cache_dir in hf_cache_dir_candidates(hf_cache_dir):
        dataset_dir = hf_cache_dataset_dir(cache_dir, dataset_id)
        candidates.append(dataset_dir)
        candidates.extend(snapshot_dirs(dataset_dir))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def existing_videos(path: Path | None, *, limit: int | None = None) -> list[Path]:
    """List supported video files under a local path.

    Args:
        path: File or directory to inspect.
        limit: Optional maximum number of files to collect.
    """
    return [Path(item) for item in list_video_files(path, limit=limit)]


def load_json_list(path: Path) -> list[Any]:
    """Load an annotation JSON file that must contain a list.

    Args:
        path: JSON file path.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"VBench-2.0 annotation JSON must contain a list: {path}")
    return payload


def vbench2_annotation_rows(annotation_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize local VBench-2.0 human annotation prompts.

    Args:
        annotation_root: Root containing VBench-2.0 human annotation JSON files.
    """
    rows: list[dict[str, Any]] = []
    reference_video_names: list[str] = []
    for dimension, filename in sorted(VBENCH2_DIMENSION_FILES.items()):
        path = annotation_root / filename
        if not path.is_file():
            continue
        for index, item in enumerate(load_json_list(path)):
            if not isinstance(item, dict):
                raise ValueError(f"VBench-2.0 annotation row must be an object: {path}#{index}")
            videos = item.get("videos")
            videos = videos if isinstance(videos, dict) else {}
            for video_path in videos.values():
                if isinstance(video_path, str) and Path(video_path).suffix.lower() in VIDEO_EXTENSIONS:
                    name = Path(video_path).name
                    if name not in reference_video_names:
                        reference_video_names.append(name)
            rows.append(
                {
                    "prompt_id": f"{dimension}:{index}",
                    "dimension": dimension,
                    "prompt": item.get("prompt_en") or item.get("prompt"),
                    "reference_videos": videos,
                    "source_file": filename,
                    "row_index": index,
                }
            )
    return rows, reference_video_names


def select_vbench2_root(candidates: list[Path], expected_files: tuple[str, ...], *, allow_videos: bool = False) -> Path | None:
    """Pick the first candidate root containing expected VBench-2.0 material.

    Args:
        candidates: Candidate local roots.
        expected_files: File names expected at the selected root.
        allow_videos: Treat any supported video file as materialized data.
    """
    for path in candidates:
        if not path.exists():
            continue
        if any((path / filename).is_file() for filename in expected_files):
            return path
        if allow_videos and existing_videos(path, limit=1):
            return path
    return None


def discover_vbench2_datasets(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Discover local VBench-2.0 real-data roots and write prompt contract rows.

    Args:
        args: Parsed runner CLI arguments.
        output_dir: Runner output directory.
    """
    dataset_root = args.vbench2_dataset_root
    hf_cache_dir = args.hf_cache_dir
    specs = {
        "sampled_videos": {
            "dataset_id": "Vchitect/VBench-2.0_sampled_videos",
            "expected_files": (),
            "allow_videos": True,
        },
        "human_annotation": {
            "dataset_id": "Vchitect/VBench-2.0_human_annotation",
            "expected_files": tuple(VBENCH2_DIMENSION_FILES.values()) + VBENCH2_ANNOTATION_EXTRA_FILES,
            "allow_videos": False,
        },
        "human_anomaly": {
            "dataset_id": "Vchitect/VBench-2.0_human_anomaly",
            "expected_files": ("README.md", *VBENCH2_ANOMALY_ARCHIVE_FILES),
            "allow_videos": True,
        },
    }
    datasets: dict[str, Any] = {}
    prompt_rows: list[dict[str, Any]] = []
    reference_video_names: list[str] = []

    for key, spec in specs.items():
        dataset_id = str(spec["dataset_id"])
        expected_files = tuple(str(item) for item in spec["expected_files"])
        root = select_vbench2_root(
            vbench2_dataset_candidates(dataset_id, dataset_root=dataset_root, hf_cache_dir=hf_cache_dir),
            expected_files,
            allow_videos=bool(spec["allow_videos"]),
        )
        present_files = [filename for filename in expected_files if root is not None and (root / filename).is_file()]
        videos = existing_videos(root, limit=args.discovery_video_limit)
        status = "missing"
        if root is not None and (present_files or videos):
            status = "ready" if len(present_files) == len(expected_files) or videos else "partial"
        datasets[key] = {
            "hf_dataset_id": dataset_id,
            "local_root": None if root is None else str(root.resolve()),
            "status": status,
            "ready": status == "ready",
            "expected_files": list(expected_files),
            "present_files": present_files,
            "video_file_count": len(videos),
            "video_file_count_is_truncated": args.discovery_video_limit is not None
            and len(videos) >= args.discovery_video_limit,
        }
        if key == "human_annotation" and root is not None:
            prompt_rows, reference_video_names = vbench2_annotation_rows(root)
            datasets[key]["prompt_count"] = len(prompt_rows)
            datasets[key]["reference_video_count"] = len(reference_video_names)

    prompt_manifest_path = output_dir / "vbench2_prompt_manifest.jsonl"
    write_jsonl(prompt_manifest_path, prompt_rows)
    return {
        "schema_version": "worldfoundry-vbench2-dataset-discovery",
        "benchmark_id": "vbench-2.0",
        "generated_at": utc_now_iso(),
        "hf_dataset_ids": list(VBENCH2_HF_DATASETS),
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "hf_cache_dirs": [str(path) for path in hf_cache_dir_candidates(hf_cache_dir)],
        "datasets": datasets,
        "prompt_manifest": str(prompt_manifest_path.resolve()),
        "prompt_count": len(prompt_rows),
        "reference_video_names": reference_video_names,
        "reference_video_count": len(reference_video_names),
        "ready": any(item.get("ready") for item in datasets.values()) or bool(prompt_rows),
    }


def build_vbench2_video_coverage(
    videos_path: Path | None,
    dataset_manifest: dict[str, Any] | None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build generated-video coverage diagnostics for VBench-2.0.

    Args:
        videos_path: Generated artifact directory or file.
        dataset_manifest: VBench-2.0 dataset discovery payload.
    """
    video_files = existing_videos(videos_path, limit=limit)
    generated_names = [path.name for path in video_files]
    reference_names = []
    if dataset_manifest is not None:
        reference_names = [str(item) for item in dataset_manifest.get("reference_video_names", [])]
    matched_names = sorted(set(generated_names) & set(reference_names))
    return {
        "schema_version": "worldfoundry-vbench2-video-coverage",
        "videos_path": None if videos_path is None else str(videos_path),
        "generated_file_count": len(video_files),
        "generated_file_count_is_truncated": limit is not None and len(video_files) >= limit,
        "generated_files": [str(path) for path in video_files],
        "reference_video_count": len(reference_names),
        "matched_reference_video_count": len(matched_names),
        "matched_reference_video_names": matched_names,
    }


def build_vbench2_benchmark_contract(
    *,
    benchmark_id: str,
    dataset_manifest: dict[str, Any] | None,
    video_coverage: dict[str, Any],
    normalizer_only: bool,
) -> dict[str, Any]:
    """Build the VBench-2.0 benchmark contract separate from official validation.

    Args:
        benchmark_id: Scorecard benchmark id.
        dataset_manifest: Local dataset discovery payload.
        video_coverage: Generated-video coverage diagnostics.
        normalizer_only: True when only an existing official result JSON is normalized.
    """
    return {
        "schema_version": "worldfoundry-vbench2-runner-contract",
        "benchmark_id": benchmark_id,
        "input_contract": {
            "generated_video_dir": "Directory or file passed by --videos-path / WORLDFOUNDRY_GENERATED_ARTIFACT_DIR.",
            "official_results_path": "Existing official *_eval_results.json accepted by --official-results-path.",
            "real_data": list(VBENCH2_HF_DATASETS),
        },
        "output_contract": {
            "scorecard": "scorecard.json",
            "raw_metric_table": "raw_metric_table.jsonl",
            "dimension_scores": "dimension_scores.json",
            "dataset_manifest": "vbench2_dataset_manifest.json",
            "video_coverage": "vbench2_video_coverage.json",
        },
        "official_validation_boundary": {
            "normalizer_only": normalizer_only,
            "official_runtime_required_for_integration_evidence": True,
            "requires_upstream_vbench2_repo": True,
            "requires_model_weights_or_judges": True,
        },
        "dataset_discovery_ready": bool(dataset_manifest and dataset_manifest.get("ready")),
        "generated_file_count": video_coverage["generated_file_count"],
    }


def split_values(values: list[str] | None, env_name: str) -> list[str]:
    raw_values = values or [os.environ.get(env_name, "")]
    results: list[str] = []
    for raw_value in raw_values:
        for item in str(raw_value).replace(",", " ").split():
            item = item.strip()
            if item and item not in results:
                results.append(item)
    return results


def canonical_metric_id(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


def canonical_declared_dimensions(variant: str) -> set[str]:
    """Read the declared dimension set from the bundled canonical full-info asset."""
    path = CANONICAL_FULL_INFO_ASSETS[variant]
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, list):
        return set()
    dimensions: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_dimensions = item.get("dimension", item.get("dimensions", []))
        if isinstance(raw_dimensions, str):
            raw_dimensions = [raw_dimensions]
        dimensions.update(
            canonical_metric_id(str(dimension))
            for dimension in raw_dimensions
            if str(dimension).strip()
        )
    return dimensions


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
        return score if math.isfinite(score) else None
    if isinstance(value, str):
        try:
            score = float(value.strip())
            return score if math.isfinite(score) else None
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        for item in value:
            score = scalar(item)
            if score is not None:
                return score
    if isinstance(value, dict):
        for key in ("score", "value", "mean", "average", "avg", "result", "all_results"):
            if key in value:
                score = scalar(value[key])
                if score is not None:
                    return score
    return None


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def normalize_score(raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    if 1.0 < raw_score <= 100.0:
        return raw_score / 100.0
    return max(0.0, min(1.0, raw_score))


def load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"VBench-series result JSON must be an object: {path}")
    return payload


def latest_results(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*_eval_results.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"official VBench-series run did not create *_eval_results.json under: {output_dir}")
    return candidates[-1]


def direct_video_files(path: Path | None) -> list[Path]:
    if path is None or not path.is_dir():
        return []
    return sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS)


def link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        try:
            target.symlink_to(source.resolve())
            return "symlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy"


def ensure_torchvision_write_video_shim(output_dir: Path) -> Path:
    """Create a subprocess shim for upstream code expecting torchvision.io.write_video."""
    shim_dir = output_dir / "runtime_shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """
from __future__ import annotations

try:
    import numpy as _np
    import torchvision.io as _torchvision_io

    if not hasattr(_torchvision_io, "write_video"):

        def _write_video(filename, video_array, fps, *args, **kwargs):
            import cv2 as _cv2

            if hasattr(video_array, "detach"):
                video_array = video_array.detach().cpu().numpy()
            frames = _np.asarray(video_array)
            if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
                raise ValueError("write_video shim expects video frames shaped [T, H, W, C]")
            if frames.dtype != _np.uint8:
                frames = _np.clip(frames, 0, 255).astype(_np.uint8)
            height, width = int(frames.shape[1]), int(frames.shape[2])
            writer = _cv2.VideoWriter(str(filename), _cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV VideoWriter could not open {filename}")
            try:
                for frame in frames:
                    if frame.shape[-1] == 1:
                        frame = _np.repeat(frame, 3, axis=-1)
                    elif frame.shape[-1] == 4:
                        frame = frame[:, :, :3]
                    writer.write(_cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

        _torchvision_io.write_video = _write_video
except Exception:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    return shim_dir


def ensure_long_custom_input_split(videos_path: Path | None) -> dict[str, Any]:
    """Prepare the split_clip layout expected by upstream VBench++ long.

    Upstream VBench++ long uses torchvision.write_video to create clips. That path
    is fragile across torchvision/PyAV versions; for bounded custom-input validations a
    whole-video clip is equivalent and keeps the official scorer on its read-only
    evaluation path.
    """

    if videos_path is None or not videos_path.is_dir():
        raise ValueError("VBench++ long_custom_input requires --videos-path to be a directory")
    videos = direct_video_files(videos_path)
    if not videos:
        raise ValueError(f"VBench++ long_custom_input found no root video files under: {videos_path}")

    split_root = videos_path / "split_clip"
    split_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, video in enumerate(videos):
        clip_dir = split_root / f"{video.stem}-{index}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clip_dir / f"{video.stem}-{index}_full{video.suffix.lower()}"
        method = link_or_copy(video, clip_path)
        entries.append(
            {
                "source": str(video),
                "clip_dir": str(clip_dir),
                "clip": str(clip_path),
                "method": method,
            }
        )
    ready_folder_count = sum(1 for child in split_root.iterdir() if child.is_dir() and child.name.rsplit("-", 1)[-1].isdigit())
    return {
        "schema_version": "worldfoundry-vbench-plus-plus-long-presplit-v1",
        "videos_path": str(videos_path),
        "split_clip_root": str(split_root),
        "source_video_count": len(videos),
        "ready_folder_count": ready_folder_count,
        "official_preprocess_skip_ready": ready_folder_count == len(videos),
        "entries": entries,
    }


def list_video_files(path: Path | None, *, limit: int | None = None, exclude_dirs: set[str] | None = None) -> list[str]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [str(path)] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    files: list[str] = []
    skipped = exclude_dirs or set()
    for item in path.rglob("*"):
        if skipped and skipped.intersection(item.relative_to(path).parts[:-1]):
            continue
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(str(item))
            if limit is not None and len(files) >= limit:
                break
    return sorted(files)


def raw_dimension_rows(raw_results: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for raw_name in sorted(raw_results):
        metric_id = canonical_metric_id(raw_name)
        raw_score = scalar(raw_results[raw_name])
        normalized_score = normalize_score(raw_score)
        is_aggregate = metric_id in AGGREGATE_METRIC_IDS
        row = {
            "metric_id": metric_id,
            "raw_metric_name": raw_name,
            "available": raw_score is not None and not is_aggregate,
            "raw_score": None if is_aggregate else raw_score,
            "normalized_score": None if is_aggregate else normalized_score,
            "raw_value": raw_results[raw_name],
            "source": "official_vbench_series_results",
        }
        if is_aggregate:
            row["reason"] = "aggregate_recomputed_only_from_complete_declared_dimensions"
        elif raw_score is None:
            row["reason"] = "score_not_found_in_vbench_series_results"
        else:
            scores[metric_id] = raw_score
        rows.append(row)
    return rows, scores


def aggregate_rows(
    variant: str,
    dimension_scores: dict[str, float],
    *,
    expected_dimensions: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if variant == "vbench2":
        group_scores: dict[str, float] = {}
        for group_id, members in VBENCH2_CATEGORY_GROUPS.items():
            group_complete = set(members).issubset(dimension_scores)
            raw_score = mean([dimension_scores[member] for member in members]) if group_complete else None
            if raw_score is not None:
                group_scores[group_id] = raw_score
            rows.append(
                {
                    "metric_id": group_id,
                    "available": raw_score is not None,
                    "raw_score": raw_score,
                    "normalized_score": normalize_score(raw_score),
                    "raw_score_range": [0.0, 1.0],
                    "source": "computed_from_official_vbench2_dimensions",
                    "dimensions": list(members),
                    **({} if raw_score is not None else {"reason": "required_vbench2_dimensions_missing"}),
                }
            )
        full_suite_complete = (
            bool(expected_dimensions)
            and expected_dimensions.issubset(dimension_scores)
            and set(VBENCH2_CATEGORY_GROUPS).issubset(group_scores)
        )
        total = mean(list(group_scores.values())) if full_suite_complete else None
        rows.append(
            {
                "metric_id": "vbench2_total",
                "available": total is not None,
                "raw_score": total,
                "normalized_score": normalize_score(total),
                "raw_score_range": [0.0, 1.0],
                "source": "computed_from_vbench2_group_scores",
                "dimensions": list(VBENCH2_CATEGORY_GROUPS),
                **({} if total is not None else {"reason": "complete_vbench2_suite_required"}),
            }
        )
        return rows

    variant_complete = bool(expected_dimensions) and expected_dimensions.issubset(dimension_scores)
    raw_score = (
        mean([dimension_scores[metric_id] for metric_id in sorted(expected_dimensions)])
        if variant_complete
        else None
    )
    variant_average = PLUS_VARIANT_AVERAGE[variant]
    rows.append(
        {
            "metric_id": variant_average,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalize_score(raw_score),
            "raw_score_range": [0.0, 1.0],
            "source": f"computed_from_vbench_plus_plus_{variant}_dimensions",
            "dimensions": sorted(expected_dimensions),
            **({} if raw_score is not None else {"reason": "complete_variant_dimension_suite_required"}),
        }
    )
    rows.append(
        {
            "metric_id": "vbench_plus_plus_average",
            "available": False,
            "raw_score": None,
            "normalized_score": None,
            "raw_score_range": [0.0, 1.0],
            "source": "requires_cross_variant_aggregation",
            "dimensions": list(PLUS_VARIANT_AVERAGE.values()),
            "reason": "all_vbench_plus_plus_variants_required",
        }
    )
    return rows


def normalize_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    variant: str,
    output_dir: Path,
    upstream_results_path: Path,
    videos_path: Path | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None,
    stderr_path: Path | None,
    dataset_manifest: dict[str, Any] | None = None,
    video_coverage: dict[str, Any] | None = None,
    long_presplit: dict[str, Any] | None = None,
    suite_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    dimension_scores_path = output_dir / "dimension_scores.json"
    dataset_manifest_path = output_dir / "vbench2_dataset_manifest.json"
    video_coverage_path = output_dir / "vbench2_video_coverage.json"
    benchmark_contract_path = output_dir / "vbench2_benchmark_contract.json"
    long_presplit_path = output_dir / "vbenchpp_long_presplit_manifest.json"
    video_files = list_video_files(videos_path, limit=1000, exclude_dirs={"split_clip"} if long_presplit is not None else None)
    video_coverage = video_coverage or build_vbench2_video_coverage(videos_path, dataset_manifest, limit=1000)

    suite_coverage = suite_coverage or {
        "full_suite_complete": False,
        "expected_dimensions": [],
        "reasons": ["canonical suite coverage was not evaluated"],
    }
    coverage_dimensions = {
        canonical_metric_id(str(item))
        for item in suite_coverage.get("expected_dimensions", [])
    }
    expected_dimensions = canonical_declared_dimensions(variant) or coverage_dimensions

    dimension_rows, dimension_scores = raw_dimension_rows(raw_results)
    metric_rows = [
        {key: value for key, value in row.items() if key != "raw_value"}
        for row in dimension_rows
    ]
    metric_rows.extend(
        aggregate_rows(
            variant,
            dimension_scores,
            expected_dimensions=expected_dimensions,
        )
    )

    per_metric = {row["metric_id"]: row for row in metric_rows}
    leaderboard = {
        row["metric_id"]: row["raw_score"]
        for row in metric_rows
        if row["available"] and row["raw_score"] is not None
    }
    available_count = sum(1 for row in metric_rows if row["available"])

    write_jsonl(raw_metric_table_path, metric_rows)
    write_json(
        dimension_scores_path,
        {
            "variant": variant,
            "upstream_results": str(upstream_results_path.resolve()),
            "dimensions": dimension_rows,
        },
    )
    normalizer_only = command is None
    normalization_ok = returncode == 0 and available_count > 0
    integration_evidence = command is not None and normalization_ok
    full_metric_coverage = bool(expected_dimensions) and expected_dimensions.issubset(dimension_scores)
    official_verified = (
        integration_evidence
        and full_metric_coverage
        and bool(suite_coverage.get("full_suite_complete", False))
    )
    normalized = normalizer_only and normalization_ok
    if variant == "vbench2":
        if dataset_manifest is None:
            dataset_manifest = discover_vbench2_datasets(
                argparse.Namespace(vbench2_dataset_root=None, hf_cache_dir=None),
                output_dir,
            )
        write_json(dataset_manifest_path, dataset_manifest)
        write_json(video_coverage_path, video_coverage)
        write_json(
            benchmark_contract_path,
            build_vbench2_benchmark_contract(
                benchmark_id=benchmark_id,
                dataset_manifest=dataset_manifest,
                video_coverage=video_coverage,
                normalizer_only=normalizer_only,
            ),
        )
    if long_presplit is not None:
        write_json(long_presplit_path, long_presplit)

    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": "official_verified"
            if official_verified
            else "official_bounded"
            if integration_evidence
            else "normalized"
            if normalized
            else "failed",
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_vbench_series_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "VBench-2.0" if variant == "vbench2" else "VBench++",
            "variant": variant,
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_model_weights": True,
        },
        "dataset": {
            "generated_artifact_dir": None if videos_path is None else str(videos_path),
            "generated_file_count": len(video_files),
            **(
                {
                    "hf_dataset_ids": list(VBENCH2_HF_DATASETS),
                    "real_data_ready": bool(dataset_manifest and dataset_manifest.get("ready")),
                    "dataset_manifest": str(dataset_manifest_path.resolve()),
                    "video_coverage": str(video_coverage_path.resolve()),
                    "prompt_count": 0 if dataset_manifest is None else int(dataset_manifest.get("prompt_count", 0)),
                    "matched_reference_video_count": video_coverage["matched_reference_video_count"],
                }
                if variant == "vbench2"
                else {}
            ),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "official VBench-series result normalization or runtime validation; full leaderboard evidence requires complete official prompt suites and dependency stack",
            ],
        },
        "generation": {
            "successful": len(video_files),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "dimensions": [row["metric_id"] for row in dimension_rows if row["available"]],
                "aggregates": [
                    row["metric_id"]
                    for row in metric_rows
                    if row["available"] and row.get("source", "").startswith("computed_")
                ],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(video_files),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": returncode == 0 and available_count > 0,
            "kind": "official_vbench_series",
            "variant": variant,
            "upstream_results": str(upstream_results_path.resolve()),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_validation_required_for_integration_evidence": True,
            "full_metric_coverage": full_metric_coverage,
            "canonical_suite_coverage": suite_coverage,
            **(
                {
                    "dataset_discovery_ready": bool(dataset_manifest and dataset_manifest.get("ready")),
                    "video_coverage": video_coverage,
                }
                if variant == "vbench2"
                else {}
            ),
            **({"long_custom_input_presplit": long_presplit} if long_presplit is not None else {}),
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "dimension_scores": str(dimension_scores_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
            **(
                {
                    "vbench2_dataset_manifest": str(dataset_manifest_path.resolve()),
                    "vbench2_video_coverage": str(video_coverage_path.resolve()),
                    "vbench2_benchmark_contract": str(benchmark_contract_path.resolve()),
                    "vbench2_prompt_manifest": None
                    if dataset_manifest is None
                    else str(dataset_manifest.get("prompt_manifest")),
                }
                if variant == "vbench2"
                else {}
            ),
            **(
                {"vbenchpp_long_presplit_manifest": str(long_presplit_path.resolve())}
                if long_presplit is not None
                else {}
            ),
        },
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def default_results_path(variant: str) -> Path | None:
    env_names = {
        "i2v": "WORLDFOUNDRY_VBENCH_PLUS_PLUS_RESULTS_PATH",
        "long": "WORLDFOUNDRY_VBENCH_PLUS_PLUS_RESULTS_PATH",
        "trustworthiness": "WORLDFOUNDRY_VBENCH_PLUS_PLUS_RESULTS_PATH",
        "vbench2": "WORLDFOUNDRY_VBENCH2_RESULTS_PATH",
    }
    return env_path(env_names[variant], env_path("WORLDFOUNDRY_VBENCH_SERIES_RESULTS_PATH"))


def default_vbench_cache_dir() -> Path:
    return env_path("WORLDFOUNDRY_VBENCH_CACHE_DIR", cache_root_path() / "models" / "vbench")


def default_vbench2_cache_dir() -> Path:
    return env_path("WORLDFOUNDRY_VBENCH2_CACHE_DIR", cache_root_path() / "models" / "vbench2")


def default_runtime_root(variant: str) -> Path:
    if variant == "vbench2":
        return env_path("WORLDFOUNDRY_VBENCH2_ROOT", env_path("WORLDFOUNDRY_VBENCH_ROOT", IN_TREE_VBENCH2_ROOT))
    return env_path(
        "WORLDFOUNDRY_VBENCH_PLUS_PLUS_ROOT",
        env_path("WORLDFOUNDRY_VBENCH_ROOT", IN_TREE_VBENCH_PLUS_PLUS_ROOT),
    )


def default_full_json(args: argparse.Namespace) -> Path:
    bundled = CANONICAL_FULL_INFO_ASSETS[args.variant]
    if bundled.is_file():
        return bundled
    fallback = {
        "i2v": args.vbench_root / "vbench2_beta_i2v" / "vbench2_i2v_full_info.json",
        "long": args.vbench_root / "vbench2_beta_long" / "VBench_full_info.json",
        "trustworthiness": args.vbench_root / "vbench2_beta_trustworthiness" / "vbench2_trustworthy.json",
        "vbench2": args.vbench_root / "vbench2" / "VBench2_full_info.json",
    }
    return fallback[args.variant]


def canonical_suite_coverage(args: argparse.Namespace) -> dict[str, Any]:
    """Prove canonical prompt, sample, and dimension coverage conservatively."""
    full_json_path = args.full_json_dir or default_full_json(args)
    requested_dimensions = {
        canonical_metric_id(item)
        for item in split_values(args.dimension, "WORLDFOUNDRY_VBENCH_SERIES_DIMENSIONS")
    }
    canonical_mode = (
        not args.custom_input
        if args.variant == "trustworthiness"
        else args.mode == ("long_vbench_standard" if args.variant == "long" else "vbench_standard")
    )
    report: dict[str, Any] = {
        "schema_version": "worldfoundry-vbench-canonical-suite-coverage-v1",
        "variant": args.variant,
        "full_json_dir": str(full_json_path),
        "full_json_exists": full_json_path.is_file(),
        "canonical_mode": canonical_mode,
        "requested_dimensions": sorted(requested_dimensions),
        "expected_dimensions": [],
        "complete_dimension_coverage": False,
        "generated_file_count": 0,
        "expected_prompt_count": 0,
        "expected_video_count": 0,
        "covered_video_count": 0,
        "complete_sample_coverage": False,
        "full_suite_complete": False,
        "reasons": [],
    }
    if not full_json_path.is_file():
        report["reasons"].append(f"canonical full-info JSON was not found: {full_json_path}")
        return report
    try:
        payload = json.loads(full_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["reasons"].append(f"canonical full-info JSON could not be loaded: {exc}")
        return report
    if not isinstance(payload, list):
        report["reasons"].append("canonical full-info JSON must contain a list")
        return report

    expected_dimensions: set[str] = set()
    prompt_sample_counts: dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_dimensions = item.get("dimension", item.get("dimensions", []))
        if isinstance(raw_dimensions, str):
            raw_dimensions = [raw_dimensions]
        item_dimensions = {
            canonical_metric_id(str(dimension))
            for dimension in raw_dimensions
            if str(dimension).strip()
        }
        expected_dimensions.update(item_dimensions)
        prompt = item.get("prompt_en") or item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt = prompt[:180] if args.variant == "vbench2" else prompt
        if args.variant == "vbench2":
            samples = 20 if "diversity" in item_dimensions else 3
        elif args.variant == "trustworthiness":
            samples = 10
        else:
            samples = 5
        prompt_sample_counts[prompt] = max(prompt_sample_counts.get(prompt, 0), samples)

    expected_stems = {
        f"{prompt}-{index}"
        for prompt, samples in prompt_sample_counts.items()
        for index in range(samples)
    }
    video_files = direct_video_files(args.videos_path)
    actual_stems = {path.stem for path in video_files}
    covered_stems = expected_stems & actual_stems
    report.update(
        {
            "expected_dimensions": sorted(expected_dimensions),
            "complete_dimension_coverage": requested_dimensions == expected_dimensions,
            "generated_file_count": len(video_files),
            "expected_prompt_count": len(prompt_sample_counts),
            "expected_video_count": len(expected_stems),
            "covered_video_count": len(covered_stems),
            "complete_sample_coverage": bool(expected_stems) and expected_stems.issubset(actual_stems),
            "sample_missing_videos": sorted(expected_stems - actual_stems)[:10],
        }
    )
    if not canonical_mode:
        report["reasons"].append("the run did not use the canonical benchmark mode")
    if requested_dimensions != expected_dimensions:
        report["reasons"].append("requested dimensions do not exactly match the canonical dimension suite")
    if not report["complete_sample_coverage"]:
        report["reasons"].append("generated videos do not cover every canonical prompt/sample pair")
    report["full_suite_complete"] = (
        canonical_mode
        and report["complete_dimension_coverage"]
        and report["complete_sample_coverage"]
    )
    return report


def resolved_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def vbench_subprocess_env(args: argparse.Namespace, *, shim_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = []
    if shim_dir is not None:
        pythonpath_entries.append(str(shim_dir))
    pythonpath_entries.append(str(REPO_ROOT))
    pythonpath_entries.append(str(args.vbench_root))
    if args.variant in PLUS_VARIANT_AVERAGE:
        pythonpath_entries.append(str(BASE_VBENCH_RUNTIME_ROOT))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("VBENCH_CACHE_DIR", str(default_vbench_cache_dir()))
    env.setdefault("VBENCH2_CACHE_DIR", str(default_vbench2_cache_dir()))
    if VBENCH_FULL_INFO_ASSET.is_file():
        env.setdefault("WORLDFOUNDRY_VBENCH_FULL_INFO", str(VBENCH_FULL_INFO_ASSET))
    if VBENCH_PROMPTS_ASSET_ROOT.is_dir():
        env.setdefault("WORLDFOUNDRY_VBENCH_PROMPTS_ROOT", str(VBENCH_PROMPTS_ASSET_ROOT))
    return env


def build_official_command(args: argparse.Namespace, upstream_output_dir: Path) -> list[str]:
    scripts = {
        "i2v": "worldfoundry.evaluation.tasks.execution.runners.vbench_plus_plus.runtime.entrypoints.i2v",
        "long": "worldfoundry.evaluation.tasks.execution.runners.vbench_plus_plus.runtime.entrypoints.long",
        "trustworthiness": "worldfoundry.evaluation.tasks.execution.runners.vbench_plus_plus.runtime.entrypoints.trustworthiness",
        "vbench2": "worldfoundry.evaluation.tasks.execution.runners.vbench_2_0.runtime.entrypoints.vbench2",
    }
    dimensions = split_values(args.dimension, "WORLDFOUNDRY_VBENCH_SERIES_DIMENSIONS")
    if not dimensions:
        raise ValueError("--dimension or WORLDFOUNDRY_VBENCH_SERIES_DIMENSIONS is required unless --official-results-path is used")
    if args.videos_path is None:
        raise ValueError("--videos-path or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required unless --official-results-path is used")

    command = [
        args.python,
        "-m",
        scripts[args.variant],
        "--output_path",
        str(upstream_output_dir.resolve()),
        "--full_json_dir",
        str((args.full_json_dir or default_full_json(args)).resolve()),
        "--videos_path",
        str(args.videos_path.resolve()),
        "--dimension",
        *dimensions,
    ]
    if args.variant in {"i2v", "long", "vbench2"}:
        command.extend(["--mode", args.mode])
    if args.variant == "i2v":
        if args.ratio:
            command.extend(["--ratio", args.ratio])
        if args.custom_image_folder:
            command.extend(["--custom_image_folder", str(args.custom_image_folder.resolve())])
    if args.variant in {"long", "vbench2"}:
        if args.prompt:
            command.extend(["--prompt", args.prompt])
        if args.prompt_file:
            command.extend(["--prompt_file", str(args.prompt_file.resolve())])
        if args.category:
            command.extend(["--category", args.category])
    if args.variant == "trustworthiness" and args.custom_input:
        command.append("--custom_input")
    return command


def run_series(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.vbench_root = (args.vbench_root or default_runtime_root(args.variant)).expanduser().resolve()
    args.from_upstream_results = resolved_path(args.from_upstream_results)
    args.videos_path = resolved_path(args.videos_path)
    args.vbench2_dataset_root = resolved_path(args.vbench2_dataset_root)
    args.hf_cache_dir = resolved_path(args.hf_cache_dir)
    args.full_json_dir = resolved_path(args.full_json_dir)
    args.prompt_file = resolved_path(args.prompt_file)
    args.custom_image_folder = resolved_path(args.custom_image_folder)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    dataset_manifest = discover_vbench2_datasets(args, output_dir) if args.variant == "vbench2" else None
    video_coverage = (
        build_vbench2_video_coverage(args.videos_path, dataset_manifest, limit=args.video_scan_limit)
        if args.variant == "vbench2"
        else None
    )
    suite_coverage = canonical_suite_coverage(args)

    results_path = args.from_upstream_results or default_results_path(args.variant)
    if results_path is not None:
        raw_results = load_results(results_path)
        return normalize_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            variant=args.variant,
            output_dir=output_dir,
            upstream_results_path=results_path,
            videos_path=args.videos_path,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            dataset_manifest=dataset_manifest,
            video_coverage=video_coverage,
            suite_coverage=suite_coverage,
        )

    upstream_output_dir = output_dir / "upstream"
    long_presplit = (
        ensure_long_custom_input_split(args.videos_path)
        if args.variant == "long" and args.mode == "long_custom_input"
        else None
    )
    shim_dir = ensure_torchvision_write_video_shim(output_dir)
    command = build_official_command(args, upstream_output_dir)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=args.vbench_root,
        env=vbench_subprocess_env(args, shim_dir=shim_dir),
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    try:
        upstream_results_path = latest_results(upstream_output_dir)
    except FileNotFoundError as exc:
        stderr_tail = "\n".join(completed.stderr.splitlines()[-20:])
        stdout_tail = "\n".join(completed.stdout.splitlines()[-20:])
        detail = stderr_tail or stdout_tail or "no upstream stdout/stderr"
        raise FileNotFoundError(f"{exc}; returncode={completed.returncode}; upstream_tail={detail}") from exc
    raw_results = load_results(upstream_results_path)
    return normalize_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        variant=args.variant,
        output_dir=output_dir,
        upstream_results_path=upstream_results_path,
        videos_path=args.videos_path,
        command=command,
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        dataset_manifest=dataset_manifest,
        video_coverage=video_coverage,
        long_presplit=long_presplit,
        suite_coverage=suite_coverage,
    )


def build_parser(
    *,
    variant_choices: tuple[str, ...] | None = None,
    description: str = "Run or normalize VBench++ / VBench-2.0 official outputs.",
) -> argparse.ArgumentParser:
    variants = variant_choices or tuple(sorted((*PLUS_VARIANT_AVERAGE, "vbench2")))
    env_variant = os.environ.get("WORLDFOUNDRY_VBENCH_SERIES_VARIANT")
    default_variant = env_variant if env_variant in variants else variants[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "vbench-plus-plus"))
    parser.add_argument("--variant", choices=variants, default=default_variant)
    parser.add_argument("--vbench-root", type=Path, default=None)
    parser.add_argument("--official-results-path", dest="from_upstream_results", type=Path)
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--videos-path", type=Path, default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"))
    parser.add_argument("--vbench2-dataset-root", type=Path, default=env_path("WORLDFOUNDRY_VBENCH2_DATASET_ROOT"))
    parser.add_argument("--hf-cache-dir", type=Path, default=resolve_hf_cache_dir())
    parser.add_argument("--full-json-dir", type=Path)
    parser.add_argument("--dimension", action="append")
    parser.add_argument("--mode", default=os.environ.get("WORLDFOUNDRY_VBENCH_SERIES_MODE", "vbench_standard"))
    parser.add_argument("--prompt", default=os.environ.get("WORLDFOUNDRY_VBENCH_SERIES_PROMPT", ""))
    parser.add_argument("--prompt-file", type=Path, default=env_path("WORLDFOUNDRY_VBENCH_SERIES_PROMPT_FILE"))
    parser.add_argument("--category", default=os.environ.get("WORLDFOUNDRY_VBENCH_SERIES_CATEGORY"))
    parser.add_argument("--ratio", default=os.environ.get("WORLDFOUNDRY_VBENCH_I2V_RATIO"))
    parser.add_argument("--custom-image-folder", type=Path, default=env_path("WORLDFOUNDRY_VBENCH_I2V_IMAGE_FOLDER"))
    parser.add_argument("--custom-input", action="store_true")
    parser.add_argument("--python", default=os.environ.get("WORLDFOUNDRY_UPSTREAM_PYTHON", sys.executable))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_VBENCH_SERIES_TIMEOUT", "7200")))
    parser.add_argument(
        "--discovery-video-limit",
        type=int,
        default=int(os.environ.get("WORLDFOUNDRY_VBENCH_DISCOVERY_VIDEO_LIMIT", "100")),
        help="Maximum local dataset videos to sample while discovering VBench-2.0 roots.",
    )
    parser.add_argument(
        "--video-scan-limit",
        type=int,
        default=int(os.environ.get("WORLDFOUNDRY_VBENCH_VIDEO_SCAN_LIMIT", "1000")),
        help="Maximum generated videos to enumerate for coverage diagnostics.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    variant_choices: tuple[str, ...] | None = None,
    description: str = "Run or normalize VBench++ / VBench-2.0 official outputs.",
) -> int:
    args = build_parser(variant_choices=variant_choices, description=description).parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    if args.variant == "vbench2" and args.benchmark_id == "vbench-plus-plus":
        args.benchmark_id = "vbench-2.0"

    try:
        scorecard = run_series(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "variant": args.variant,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "dimension_scores": scorecard["artifacts"]["dimension_scores"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}/{args.variant}: VBench-series evaluation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
