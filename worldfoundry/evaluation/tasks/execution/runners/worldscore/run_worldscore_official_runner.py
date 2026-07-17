#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.utils import REPO_ROOT

from worldfoundry.runtime.env import first_env_value, resolve_hf_cache_dir  # type: ignore[reportMissingImports]  # noqa: E402
from worldfoundry.evaluation.utils import HFD_DATASET_CACHE_ROOT, worldfoundry_hfd_dataset_root
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, load_json, utc_now_iso, write_json, write_jsonl

DEFAULT_WORLDSCORE_ROOT = (
    REPO_ROOT
    / "worldfoundry"
    / "evaluation"
    / "tasks"
    / "execution"
    / "runners"
    / "worldscore"
    / "runtime"
    / "worldscore"
)
DEFAULT_WORLDSCORE_CONFIG_ROOT = bundled_benchmark_asset("worldscore", "config")
WORLDSCORE_HF_CACHE_DIR = "datasets--Howieeeee--WorldScore"
WORLDSCORE_HFD_DIR_NAME = "Howieeeee__WorldScore"
WORLDSCORE_DATASET_DIR_NAME = "WorldScore-Dataset"
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"

CONTROLLABILITY_ASPECTS = ("camera_control", "object_control")
QUALITY_ASPECTS = (
    "content_alignment",
    "3d_consistency",
    "photometric_consistency",
    "style_consistency",
    "subjective_quality",
)
DYNAMICS_ASPECTS = ("motion_accuracy", "motion_magnitude", "motion_smoothness")
METRIC_ORDER = ("controllability", "quality", "dynamics", "worldscore_average")
METRIC_ALIASES = {
    "controllability": "controllability",
    "quality": "quality",
    "dynamics": "dynamics",
    "worldscore_average": "worldscore_average",
    "WorldScore Average": "worldscore_average",
}
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".avi", ".webm"}


def env_value(*names: str) -> str | None:
    """Resolve the first configured environment value.

    Args:
        *names: Environment variable names ordered by priority.
    """

    return first_env_value(names, os.environ)


def latest_hf_dataset_snapshot(cache_dir: Path) -> Path | None:
    dataset_dir = cache_dir / WORLDSCORE_HF_CACHE_DIR
    snapshots_dir = dataset_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return snapshots[-1] if snapshots else None


def local_worldscore_data_paths(cache_dir: Path) -> list[Path]:
    """Return local DATA_PATH candidates that contain WorldScore-Dataset.

    Args:
        cache_dir: Hugging Face cache root configured for this run.
    """

    candidates = [
        worldfoundry_hfd_dataset_root() / WORLDSCORE_HFD_DIR_NAME,
        cache_dir / WORLDSCORE_HFD_DIR_NAME,
        REPO_ROOT / "cache" / "worldfoundry" / "assets" / "worldscore_official" / "data",
        HFD_DATASET_CACHE_ROOT / WORLDSCORE_HFD_DIR_NAME,
    ]
    selected: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        if (candidate / WORLDSCORE_DATASET_DIR_NAME).is_dir():
            selected.append(candidate)
    return selected


def resolve_worldscore_data_path(args: argparse.Namespace) -> Path | None:
    if args.data_path is not None:
        return args.data_path
    snapshot = latest_hf_dataset_snapshot(args.hf_cache_dir)
    if snapshot is not None:
        return snapshot
    local_paths = local_worldscore_data_paths(args.hf_cache_dir)
    if local_paths:
        return local_paths[0]
    return None


def read_simple_yaml_scalar(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not stripped.startswith(prefix):
            continue
        value = stripped.removeprefix(prefix).strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        return value.strip("'\"") or None
    return None


def render_worldscore_path_template(value: str, args: argparse.Namespace) -> str:
    replacements = {
        "${oc.env:MODEL_PATH}": "" if args.model_path is None else str(args.model_path),
        "${oc.env:DATA_PATH}": "" if args.data_path is None else str(args.data_path),
        "${oc.env:WORLDSCORE_PATH}": str(args.worldscore_root),
    }
    rendered = value
    for source, target in replacements.items():
        rendered = rendered.replace(source, target)
    return os.path.expandvars(rendered)


def derive_worldscore_output_dir(args: argparse.Namespace) -> Path | None:
    if args.model_name is None:
        return None
    model_config_path = args.worldscore_config_root / "model_configs" / f"{args.model_name}.yaml"
    base_config_path = args.worldscore_config_root / "base_config.yaml"
    runs_root = read_simple_yaml_scalar(model_config_path, "runs_root")
    if not runs_root:
        return None
    output_dir = read_simple_yaml_scalar(model_config_path, "output_dir") or read_simple_yaml_scalar(base_config_path, "output_dir")
    if not output_dir:
        return None
    return Path(render_worldscore_path_template(runs_root, args)) / render_worldscore_path_template(output_dir, args)


def read_worldscore_config_scalar(args: argparse.Namespace, key: str) -> str | None:
    """Read a scalar from the model config, falling back to base config.

    Args:
        args: Parsed WorldScore runner arguments.
        key: YAML key to read from official config files.
    """

    if args.model_name is None:
        return None
    model_config_path = args.worldscore_config_root / "model_configs" / f"{args.model_name}.yaml"
    base_config_path = args.worldscore_config_root / "base_config.yaml"
    return read_simple_yaml_scalar(model_config_path, key) or read_simple_yaml_scalar(base_config_path, key)


def read_worldscore_config_int(args: argparse.Namespace, key: str, default: int) -> int:
    """Read an integer from official config files.

    Args:
        args: Parsed WorldScore runner arguments.
        key: YAML key to read.
        default: Value used when the key is absent.
    """

    value = read_worldscore_config_scalar(args, key)
    return default if value is None else int(value)


def read_worldscore_resolution(args: argparse.Namespace) -> tuple[int, int]:
    """Read the official generated input-image resolution.

    Args:
        args: Parsed WorldScore runner arguments.
    """

    value = read_worldscore_config_scalar(args, "resolution")
    if value is None:
        return (832, 480)
    parsed = json.loads(value)
    return (int(parsed[0]), int(parsed[1]))


def worldscore_dataset_root(data_path: Path) -> Path:
    """Return the official WorldScore-Dataset directory.

    Args:
        data_path: DATA_PATH value passed to the official WorldScore runtime.
    """

    return data_path / WORLDSCORE_DATASET_DIR_NAME


def load_worldscore_records(data_path: Path, split: str) -> list[dict[str, Any]]:
    """Load official WorldScore metadata records for one split.

    Args:
        data_path: DATA_PATH value passed to the official WorldScore runtime.
        split: Split name such as "dynamic" or "static".
    """

    manifest_path = worldscore_dataset_root(data_path) / split / f"{split}.json"
    records = load_json(manifest_path)
    if not isinstance(records, list):
        raise ValueError(f"WorldScore {split} manifest must be a list: {manifest_path}")
    return [record for record in records if isinstance(record, dict)]


def center_crop_worldscore_input(image_path: Path, resolution: tuple[int, int], output_path: Path) -> None:
    """Write the official runner-style cropped input_image.png.

    Args:
        image_path: Official dataset image path.
        resolution: Target (width, height) read from model config.
        output_path: Destination input_image.png path.
    """

    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        target_width, target_height = resolution
        target_ratio = target_width / target_height
        current_ratio = width / height
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / target_ratio)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        cropped = image.crop((left, top, left + new_width, top + new_height))
        cropped.resize(resolution).save(output_path)


def copy_stage_frames(source_path: Path, frames_dir: Path, target_frames: int) -> int:
    """Extract or copy generated frames into official frame filenames.

    Args:
        source_path: Generated video file or directory of generated frames.
        frames_dir: Destination official frames directory.
        target_frames: Required bounded frame count for this staged sample.
    """

    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        source_frames = sorted(
            path for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
        )
        if len(source_frames) < target_frames:
            raise ValueError(f"source frame directory has {len(source_frames)} frames but {target_frames} are required")
        for index, frame_path in enumerate(source_frames[:target_frames]):
            with Image.open(frame_path) as image:
                image.convert("RGB").save(frames_dir / f"{index:03d}.png")
        return target_frames

    import imageio.v3 as iio

    written = 0
    for frame in iio.imiter(source_path):
        if written >= target_frames:
            break
        Image.fromarray(frame).convert("RGB").save(frames_dir / f"{written:03d}.png")
        written += 1
    if written < target_frames:
        raise ValueError(f"source video has {written} frames but {target_frames} are required")
    return written


def resolve_stage_dynamic_source(source_path: Path) -> Path:
    if not source_path.is_dir():
        return source_path
    frame_sources = sorted(
        path for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    )
    if frame_sources:
        return source_path
    video_sources = sorted(
        path for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not video_sources:
        raise ValueError(f"stage source directory contains no frames or videos: {source_path}")
    return video_sources[0]


def dynamic_instance_dir(output_root: Path, image_data: dict[str, Any]) -> Path:
    """Return the official dynamic instance directory for a dataset record.

    Args:
        output_root: WorldScore output_dir root.
        image_data: One official dynamic metadata record.
    """

    image_name = Path(str(image_data["image"])).stem
    return output_root / "dynamic" / str(image_data["visual_style"]) / str(image_data["motion_type"]) / image_name


def stage_dynamic_worldscore_output(args: argparse.Namespace) -> dict[str, Any]:
    """Stage one bounded dynamic generated output in official layout.

    Args:
        args: Parsed runner arguments containing stage source and official paths.
    """

    if args.data_path is None:
        raise ValueError("--data-path, DATA_PATH, or local WorldScore-Dataset is required for dynamic staging")
    output_root = args.worldscore_output_dir or derive_worldscore_output_dir(args)
    if output_root is None:
        raise ValueError("--worldscore-output-dir or resolvable --model-name/--model-path is required for dynamic staging")
    records = load_worldscore_records(args.data_path, "dynamic")
    image_data = dict(records[args.stage_sample_index])
    target_frames = args.stage_target_frames or read_worldscore_config_int(args, "frames", 50)
    official_frames = read_worldscore_config_int(args, "frames", target_frames)
    instance_dir = dynamic_instance_dir(output_root, image_data)
    if instance_dir.exists() and not args.stage_overwrite:
        raise FileExistsError(f"staged WorldScore instance already exists: {instance_dir}; pass --stage-overwrite")

    (output_root / "static").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(instance_dir, ignore_errors=True)
    frames_dir = instance_dir / "frames"
    source_path = resolve_stage_dynamic_source(args.stage_dynamic_source)
    frame_count = copy_stage_frames(source_path, frames_dir, target_frames)
    source_image = relative_to_data_root(str(image_data["image"]), args.data_path)
    center_crop_worldscore_input(source_image, read_worldscore_resolution(args), instance_dir / "input_image.png")

    image_data["num_scenes"] = len(image_data.get("camera_path", [])) or 1
    image_data["total_frames"] = frame_count
    write_json(instance_dir / "image_data.json", image_data)

    video_output_dir = instance_dir / "videos"
    if source_path.is_file():
        video_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, video_output_dir / "output.mp4")

    return {
        "split": "dynamic",
        "requested_source_path": str(args.stage_dynamic_source),
        "source_path": str(source_path),
        "output_dir": str(output_root),
        "instance_dir": str(instance_dir),
        "sample_index": args.stage_sample_index,
        "frame_count": frame_count,
        "official_config_frames": official_frames,
        "bounded": frame_count != official_frames,
        "leaderboard_valid": False,
    }


def count_frame_files(frames_dir: Path) -> int:
    """Count image frames accepted by the official WorldScore evaluator.

    Args:
        frames_dir: Candidate frames directory under one WorldScore instance.
    """

    if not frames_dir.is_dir():
        return 0
    return sum(1 for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES)


def relative_to_data_root(path_value: str, data_path: Path | None) -> Path:
    """Resolve a WorldScore dataset-relative path for validation.

    Args:
        path_value: Path stored in image_data.json.
        data_path: Runner data path passed as DATA_PATH to official WorldScore.
    """

    path = Path(path_value)
    if path.is_absolute() or data_path is None:
        return path
    return data_path / "WorldScore-Dataset" / path


def validate_worldscore_output_tree(generated_root: Path | None, data_path: Path | None) -> dict[str, Any]:
    """Validate generated files against the official WorldScore directory contract.

    Args:
        generated_root: WorldScore output_dir containing static and dynamic trees.
        data_path: Runner data path used to resolve dynamic masks.
    """

    if generated_root is None:
        return {"valid": False, "blockers": ["generated_root was not resolved"], "samples": []}
    if not generated_root.exists():
        return {"valid": False, "blockers": [f"generated_root does not exist: {generated_root}"], "samples": []}

    samples: list[dict[str, Any]] = []
    blockers: list[str] = []
    for image_data_path in sorted(generated_root.rglob("image_data.json")):
        instance_dir = image_data_path.parent
        image_data = load_json(image_data_path)
        split = "dynamic" if "dynamic" in image_data_path.parts else "static"
        total_frames = image_data.get("total_frames")
        frame_count = count_frame_files(instance_dir / "frames")
        issues: list[str] = []

        if not (instance_dir / "input_image.png").is_file():
            issues.append("missing input_image.png")
        if not (instance_dir / "frames").is_dir():
            issues.append("missing frames directory")
        if isinstance(total_frames, int) and frame_count < total_frames:
            issues.append(f"frames directory has {frame_count} frames but image_data.total_frames is {total_frames}")

        camera_count = None
        if split == "static":
            camera_data_path = instance_dir / "camera_data.json"
            if not camera_data_path.is_file():
                issues.append("missing camera_data.json")
            else:
                camera_data = load_json(camera_data_path)
                cameras = camera_data.get("cameras_interp")
                camera_count = len(cameras) if isinstance(cameras, list) else None
                if isinstance(total_frames, int) and camera_count is not None and camera_count != total_frames:
                    issues.append(f"camera_data.cameras_interp has {camera_count} cameras but image_data.total_frames is {total_frames}")
                if camera_count is not None and frame_count != camera_count:
                    issues.append(f"frames directory has {frame_count} frames but camera_data.cameras_interp has {camera_count} cameras")

        missing_masks: list[str] = []
        if split == "dynamic":
            masks = image_data.get("masks")
            if not isinstance(masks, list):
                issues.append("dynamic image_data.json is missing required masks list")
            else:
                missing_masks = [mask for mask in masks if not relative_to_data_root(mask, data_path).is_file()]
                if missing_masks:
                    issues.append(f"dynamic masks are missing from dataset root: {', '.join(missing_masks)}")

        sample = {
            "instance_dir": str(instance_dir),
            "split": split,
            "total_frames": total_frames,
            "frame_count": frame_count,
            "camera_count": camera_count,
            "video_exists": (instance_dir / "videos" / "output.mp4").is_file(),
            "missing_masks": missing_masks,
            "issues": issues,
        }
        samples.append(sample)
        blockers.extend(f"{sample['instance_dir']}: {issue}" for issue in issues)

    if not samples:
        blockers.append(f"no image_data.json files found under: {generated_root}")
    return {"valid": not blockers, "blockers": blockers, "samples": samples}


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, list) or isinstance(value, tuple):
        values = [scalar(item) for item in value]
        values = [item for item in values if item is not None]
        if values:
            return sum(values) / len(values)
    if isinstance(value, dict):
        for key in ("score", "raw_score", "normalized_score", "value", "mean", "average", "avg", "overall"):
            if key in value:
                number = scalar(value[key])
                if number is not None:
                    return number
    return None


def iter_numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        number = scalar(value)
        return [] if number is None else [number]
    if isinstance(value, list) or isinstance(value, tuple):
        values: list[float] = []
        for item in value:
            values.extend(iter_numeric_values(item))
        return values
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(iter_numeric_values(item))
        return values
    return []


def normalize_worldscore_value(raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    if raw_score > 1.5:
        return raw_score / 100.0
    return raw_score


def score_from_aspects(scores: dict[str, Any], aspects: tuple[str, ...]) -> tuple[float | None, dict[str, float]]:
    values: dict[str, float] = {}
    for aspect in aspects:
        number = scalar(scores.get(aspect))
        if number is not None:
            values[aspect] = number
    if not values:
        return None, {}
    return sum(values.values()) / len(values), values


def metric_source_payload(raw_results: dict[str, Any]) -> dict[str, Any]:
    for key in ("scores", "metrics", "leaderboard", "leaderboard_metrics"):
        value = raw_results.get(key)
        if isinstance(value, dict):
            return value
    return raw_results


def extract_scores(raw_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = metric_source_payload(raw_results)
    extracted: dict[str, dict[str, Any]] = {}

    for raw_key, raw_value in payload.items():
        metric_id = METRIC_ALIASES.get(str(raw_key))
        if not metric_id or metric_id in extracted:
            continue
        raw_score = scalar(raw_value)
        extracted[metric_id] = {
            "raw_score": raw_score,
            "sub_scores": {},
            "source": raw_key,
        }

    grouped = {
        "controllability": CONTROLLABILITY_ASPECTS,
        "quality": QUALITY_ASPECTS,
        "dynamics": DYNAMICS_ASPECTS,
    }
    for metric_id, aspects in grouped.items():
        if metric_id not in extracted:
            raw_score, sub_scores = score_from_aspects(payload, aspects)
            if raw_score is not None:
                extracted[metric_id] = {
                    "raw_score": raw_score,
                    "sub_scores": sub_scores,
                    "source": "computed_from_worldscore_aspects",
                }

    if "worldscore_average" not in extracted:
        for key in ("WorldScore-Dynamic", "WorldScore-Static"):
            raw_score = scalar(payload.get(key))
            if raw_score is not None:
                extracted["worldscore_average"] = {
                    "raw_score": raw_score,
                    "sub_scores": {},
                    "source": key,
                }
                break

    if "worldscore_average" not in extracted:
        component_scores = [
            item["raw_score"]
            for metric_id, item in extracted.items()
            if metric_id in {"controllability", "quality", "dynamics"} and item["raw_score"] is not None
        ]
        if component_scores:
            extracted["worldscore_average"] = {
                "raw_score": sum(component_scores) / len(component_scores),
                "sub_scores": {},
                "source": "computed_from_worldscore_metric_families",
            }
    return extracted


def iter_per_sample_rows(raw_results: dict[str, Any], generated_root: Path | None) -> list[dict[str, Any]]:
    explicit_rows = raw_results.get("per_sample_metrics")
    if isinstance(explicit_rows, list):
        return [row for row in explicit_rows if isinstance(row, dict)]
    if generated_root is None or not generated_root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(generated_root.rglob("evaluation.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "evaluation_path": str(path),
                "relative_path": str(path.relative_to(generated_root)),
                "metrics": payload,
            }
        )
    return rows


def sample_scores_from_evaluation_payload(payload: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for aspect in (*CONTROLLABILITY_ASPECTS, *QUALITY_ASPECTS, *DYNAMICS_ASPECTS):
        if aspect in payload:
            number = scalar(payload[aspect])
            if number is not None:
                scores[aspect] = number
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if key in scores:
            continue
        nested_values = iter_numeric_values(value)
        if nested_values:
            scores[key] = sum(nested_values) / len(nested_values)
    return scores


def aggregate_worldscore_evaluation_tree(generated_root: Path | None) -> dict[str, Any] | None:
    if generated_root is None or not generated_root.exists():
        return None
    per_sample_metrics: list[dict[str, Any]] = []
    aspect_values: dict[str, list[float]] = {}
    for path in sorted(generated_root.rglob("evaluation.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        sample_scores = sample_scores_from_evaluation_payload(payload)
        if not sample_scores:
            continue
        for key, value in sample_scores.items():
            aspect_values.setdefault(key, []).append(value)
        per_sample_metrics.append(
            {
                "evaluation_path": str(path),
                "relative_path": str(path.relative_to(generated_root)),
                "metrics": payload,
                "scores": sample_scores,
            }
        )
    if not per_sample_metrics:
        return None

    summary = {key: sum(values) / len(values) for key, values in sorted(aspect_values.items()) if values}
    raw_results: dict[str, Any] = {
        **summary,
        "per_sample_metrics": per_sample_metrics,
        "normalization_source": "generated_evaluation_tree",
    }
    extracted = extract_scores(raw_results)
    if "worldscore_average" in extracted:
        raw_results["worldscore_average"] = extracted["worldscore_average"]["raw_score"]
    return raw_results


def latest_worldscore_results(search_root: Path | None) -> Path | None:
    if search_root is None or not search_root.exists():
        return None
    candidates = sorted(search_root.rglob("worldscore.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def resolve_worldscore_results_path(
    args: argparse.Namespace,
    *,
    fallback_output_dir: Path,
    strict_explicit: bool,
) -> tuple[Path, Path | None]:
    if args.worldscore_results_path is not None:
        if args.worldscore_results_path.is_file():
            return args.worldscore_results_path, args.worldscore_results_path.parent
        raise FileNotFoundError(f"WorldScore result JSON not found: {args.worldscore_results_path}")

    if args.worldscore_output_dir is not None:
        results_path = args.worldscore_output_dir / "worldscore.json"
        if results_path.is_file():
            return results_path, args.worldscore_output_dir
        if strict_explicit and args.generated_root is None:
            raise FileNotFoundError(f"WorldScore worldscore.json not found under: {args.worldscore_output_dir}")
        return fallback_output_dir / "official" / "missing_worldscore.json", args.worldscore_output_dir

    derived_output_dir = derive_worldscore_output_dir(args)
    if derived_output_dir is not None:
        results_path = derived_output_dir / "worldscore.json"
        if results_path.is_file():
            return results_path, derived_output_dir

    results_path = latest_worldscore_results(args.model_path)
    if results_path is not None:
        return results_path, results_path.parent
    results_path = latest_worldscore_results(args.generated_root)
    if results_path is not None:
        return results_path, results_path.parent
    return fallback_output_dir / "official" / "missing_worldscore.json", derived_output_dir


def load_or_aggregate_worldscore_results(
    results_path: Path,
    *,
    generated_root: Path | None,
    fallback_output_dir: Path,
) -> tuple[dict[str, Any], Path, str]:
    if results_path.is_file():
        raw_results = load_json(results_path)
        if not isinstance(raw_results, dict):
            raise ValueError(f"WorldScore result JSON must be an object: {results_path}")
        return raw_results, results_path, "worldscore_json"

    aggregated = aggregate_worldscore_evaluation_tree(generated_root)
    if aggregated is not None:
        aggregate_path = fallback_output_dir / "official" / "worldscore_from_evaluation_tree.json"
        write_json(aggregate_path, aggregated)
        return aggregated, aggregate_path, "generated_evaluation_tree"

    write_json(results_path, {"error": "missing WorldScore result artifact"})
    raw_results = load_json(results_path)
    if not isinstance(raw_results, dict):
        raise ValueError(f"WorldScore result JSON must be an object: {results_path}")
    return raw_results, results_path, "missing_worldscore_json"


def normalize_worldscore_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    output_dir: Path,
    official_results_path: Path,
    data_path: Path | None,
    generated_root: Path | None,
    evaluation_output_dir: Path | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    contract_validation_import: bool = False,
    normalization_source: str = "worldscore_json",
    contract_validation: dict[str, Any] | None = None,
    run_scope: str = "official",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder_stdout = output_dir / "official_stdout.log"
    placeholder_stderr = output_dir / "official_stderr.log"
    resolved_stdout = stdout_path if stdout_path is not None else placeholder_stdout
    resolved_stderr = stderr_path if stderr_path is not None else placeholder_stderr
    if stdout_path is None:
        resolved_stdout.write_text("", encoding="utf-8")
    if stderr_path is None:
        resolved_stderr.write_text("", encoding="utf-8")
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_metrics_path = output_dir / "per_sample_metrics.jsonl"

    extracted_scores = extract_scores(raw_results)
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}

    for metric_id in METRIC_ORDER:
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        normalized_score = normalize_worldscore_value(raw_score)
        row = {
            "metric_id": metric_id,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "source": item.get("source"),
            "sub_scores": item.get("sub_scores") or {},
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_worldscore_results"
        else:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    per_sample_rows = iter_per_sample_rows(raw_results, generated_root)
    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_metrics_path, per_sample_rows)

    available_count = sum(1 for row in metric_rows if row["available"])
    contract_blockers = [] if contract_validation is None else contract_validation.get("blockers", [])
    bounded_run = run_scope.endswith("_bounded")
    run_status = "official_verified" if returncode == 0 and available_count else "failed"
    if contract_blockers and available_count == 0:
        run_status = "blocked"
    elif returncode == 0 and available_count == 0 and contract_validation and contract_validation.get("valid"):
        run_status = "contract_validated"
    normalization_ok = returncode == 0 and available_count > 0
    normalizer_only = command is None
    official_verified = command is not None and normalization_ok
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_worldscore_official_runner",
            "scope": run_scope,
            "bounded": bounded_run,
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "WorldScore",
            "contract_only": False,
            "requires_in_tree_official_runtime": True,
        },
        "dataset": {
            "data_path": None if data_path is None else str(data_path),
            "generated_artifact_dir": None if generated_root is None else str(generated_root),
            "per_sample_metric_count": len(per_sample_rows),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "WorldScore runtime validation; full leaderboard evidence requires complete evaluation trees and submission protocol",
            ],
        },
        "generation": {
            "successful": len(per_sample_rows),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "worldscore_metric_families": [row["metric_id"] for row in metric_rows if row["available"]],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(per_sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "official_worldscore",
            "scope": run_scope,
            "official_results": str(official_results_path),
            "evaluation_output_dir": None if evaluation_output_dir is None else str(evaluation_output_dir),
            "normalization_source": normalization_source,
            "num_results": len(per_sample_rows),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
            "blockers": contract_blockers,
        },
        "contract_validation": contract_validation,
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_metrics": str(per_sample_metrics_path.resolve()),
            "official_results": str(official_results_path.resolve()),
            "evaluation_output_dir": None if evaluation_output_dir is None else str(evaluation_output_dir.resolve()),
            "official_stdout": str(resolved_stdout.resolve()),
            "official_stderr": str(resolved_stderr.resolve()),
        },
        "validation": {
            "official_contract": True,
            "scope": run_scope,
            "bounded": bounded_run,
            "leaderboard_valid": False,
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_results_imported": normalizer_only and normalization_ok,
        },
        "official_benchmark_verified": official_verified,
        "integration_evidence": official_verified,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
        "worldfoundry_contract_validation_evidence": contract_validation_import
        and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def build_official_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(args.worldscore_root / "worldscore" / "run_evaluate.py"),
        "--model_name",
        args.model_name,
    ]
    if args.num_jobs is not None:
        command.extend(["--num_jobs", str(args.num_jobs)])
    if args.only_calculate_mean:
        command.extend(["--only_calculate_mean", "True"])
    return command


def worldscore_pythonpath_roots(worldscore_root: Path) -> list[Path]:
    return [
        REPO_ROOT,
        worldscore_root,
    ]


def run_command_with_timeout(command: list[str], cwd: Path, env: dict[str, str], timeout: int) -> dict[str, Any]:
    """Run an official command while preserving timeout stdout and stderr.

    Args:
        command: Command line to execute.
        cwd: Working directory for the official process.
        env: Environment passed to the official process.
        timeout: Maximum runtime in seconds.
    """

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    timed_out = process.poll() is None
    if timed_out:
        process.kill()
    stdout, stderr = process.communicate()
    return {
        "returncode": 124 if timed_out else process.returncode,
        "stdout": stdout,
        "stderr": f"Command timed out after {timeout} seconds\n{stderr}" if timed_out else stderr,
    }


def run_official_worldscore(args: argparse.Namespace) -> dict[str, Any]:
    args.worldscore_root = args.worldscore_root.expanduser().resolve()
    args.worldscore_config_root = args.worldscore_config_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser().resolve()
    if args.data_path is not None:
        args.data_path = args.data_path.expanduser().resolve()
    if args.generated_root is not None:
        args.generated_root = args.generated_root.expanduser().resolve()
    if args.worldscore_results_path is not None:
        args.worldscore_results_path = args.worldscore_results_path.expanduser().resolve()
    if args.worldscore_output_dir is not None:
        args.worldscore_output_dir = args.worldscore_output_dir.expanduser().resolve()
    if args.official_results_path is not None:
        args.official_results_path = args.official_results_path.expanduser().resolve()
    if args.stage_dynamic_source is not None:
        args.stage_dynamic_source = args.stage_dynamic_source.expanduser().resolve()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "official_stdout.log"
    stderr_path = output_dir / "official_stderr.log"
    args.data_path = resolve_worldscore_data_path(args)
    stage_result = None
    run_scope = "official"
    normalizer_mode = bool(args.official_results_path) or (
        args.worldscore_output_dir is not None and args.model_name is None
    )
    should_stage_dynamic = args.stage_dynamic_source is not None and (args.stage_only or not normalizer_mode)
    if should_stage_dynamic:
        stage_result = stage_dynamic_worldscore_output(args)
        args.generated_root = Path(stage_result["output_dir"])
        args.worldscore_output_dir = Path(stage_result["output_dir"])
        run_scope = "official_bounded" if stage_result["bounded"] else "official"

    if args.official_results_path:
        raw_results = load_json(args.official_results_path)
        if not isinstance(raw_results, dict):
            raise ValueError(f"WorldScore result JSON must be an object: {args.official_results_path}")
        evaluation_output_dir = args.worldscore_output_dir or args.official_results_path.parent
        generated_root = args.generated_root or args.worldscore_output_dir
        return normalize_worldscore_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            official_results_path=args.official_results_path,
            data_path=args.data_path,
            generated_root=generated_root,
            evaluation_output_dir=evaluation_output_dir,
            command=None,
            duration_seconds=None,
            returncode=0,
            contract_validation_import=True,
            normalization_source="worldscore_json",
            contract_validation=validate_worldscore_output_tree(generated_root, args.data_path),
            run_scope=run_scope,
        )

    if args.worldscore_output_dir and args.model_name is None:
        results_path, evaluation_output_dir = resolve_worldscore_results_path(
            args,
            fallback_output_dir=output_dir,
            strict_explicit=True,
        )
        generated_root = args.generated_root or evaluation_output_dir
        raw_results, resolved_results_path, normalization_source = load_or_aggregate_worldscore_results(
            results_path,
            generated_root=generated_root,
            fallback_output_dir=output_dir,
        )
        return normalize_worldscore_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            official_results_path=resolved_results_path,
            data_path=args.data_path,
            generated_root=generated_root,
            evaluation_output_dir=evaluation_output_dir,
            command=None,
            duration_seconds=None,
            returncode=0,
            contract_validation_import=True,
            normalization_source=normalization_source,
            contract_validation=validate_worldscore_output_tree(generated_root, args.data_path),
            run_scope=run_scope,
        )

    if args.stage_only:
        results_path, evaluation_output_dir = resolve_worldscore_results_path(
            args,
            fallback_output_dir=output_dir,
            strict_explicit=False,
        )
        generated_root = args.generated_root or evaluation_output_dir
        raw_results, resolved_results_path, normalization_source = load_or_aggregate_worldscore_results(
            results_path,
            generated_root=generated_root,
            fallback_output_dir=output_dir,
        )
        return normalize_worldscore_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            official_results_path=resolved_results_path,
            data_path=args.data_path,
            generated_root=generated_root,
            evaluation_output_dir=evaluation_output_dir,
            command=None,
            duration_seconds=None,
            returncode=0,
            contract_validation_import=True,
            normalization_source=normalization_source,
            contract_validation=validate_worldscore_output_tree(generated_root, args.data_path),
            run_scope=run_scope,
        )

    if args.model_name is None:
        raise ValueError("--model-name or WORLDFOUNDRY_WORLDSCORE_MODEL_NAME is required unless --official-results-path is used")
    if args.model_path is None:
        raise ValueError("--model-path, MODEL_PATH, or WORLDFOUNDRY_WORLDSCORE_MODEL_PATH is required unless --official-results-path is used")
    if args.data_path is None:
        raise ValueError("--data-path, DATA_PATH, or WORLDFOUNDRY_WORLDSCORE_DATA_PATH is required unless --official-results-path is used")
    if not (args.worldscore_root / "worldscore" / "run_evaluate.py").is_file():
        raise FileNotFoundError(f"WorldScore run_evaluate.py not found under: {args.worldscore_root}")

    command = build_official_command(args)
    env = os.environ.copy()
    env["WORLDSCORE_PATH"] = str(args.worldscore_root)
    env["WORLDSCORE_CONFIG_ROOT"] = str(args.worldscore_config_root)
    env["MODEL_PATH"] = str(args.model_path)
    env["DATA_PATH"] = str(args.data_path)
    pythonpath_entries = [str(path) for path in worldscore_pythonpath_roots(args.worldscore_root) if path.exists()]
    if env.get("PYTHONPATH"):
        pythonpath_entries.extend(entry for entry in env["PYTHONPATH"].split(os.pathsep) if entry)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    start = time.monotonic()
    completed = run_command_with_timeout(command, args.worldscore_root, env, args.timeout)
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(completed["stdout"], encoding="utf-8")
    stderr_path.write_text(completed["stderr"], encoding="utf-8")

    results_path, evaluation_output_dir = resolve_worldscore_results_path(args, fallback_output_dir=output_dir, strict_explicit=False)
    generated_root = args.generated_root or evaluation_output_dir or args.model_path
    raw_results, resolved_results_path, normalization_source = load_or_aggregate_worldscore_results(
        results_path,
        generated_root=generated_root,
        fallback_output_dir=output_dir,
    )

    return normalize_worldscore_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        official_results_path=resolved_results_path,
        data_path=args.data_path,
        generated_root=generated_root,
        evaluation_output_dir=evaluation_output_dir,
        command=command,
        duration_seconds=duration_seconds,
        returncode=completed["returncode"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        normalization_source=normalization_source,
        contract_validation=validate_worldscore_output_tree(generated_root, args.data_path),
        run_scope=run_scope,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official WorldScore and normalize its output to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "worldscore"))
    parser.add_argument("--worldscore-root", type=Path, default=env_path("WORLDFOUNDRY_WORLDSCORE_ROOT", default=DEFAULT_WORLDSCORE_ROOT))
    parser.add_argument(
        "--worldscore-config-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDSCORE_CONFIG_ROOT", default=DEFAULT_WORLDSCORE_CONFIG_ROOT),
    )
    parser.add_argument("--model-name", default=env_value("WORLDFOUNDRY_WORLDSCORE_MODEL_NAME", "WORLDFOUNDRY_MODEL_NAME"))
    parser.add_argument("--model-path", type=Path, default=env_path("WORLDFOUNDRY_WORLDSCORE_MODEL_PATH", "MODEL_PATH"))
    parser.add_argument(
        "--data-path",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDSCORE_DATA_PATH", "WORLDFOUNDRY_BENCHMARK_DATA_ROOT", "DATA_PATH"),
        help="WorldScore dataset root; defaults to local HF cache snapshot when available.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=resolve_hf_cache_dir())
    parser.add_argument("--generated-root", type=Path, default=env_path("WORLDFOUNDRY_WORLDSCORE_GENERATED_ROOT"))
    parser.add_argument(
        "--worldscore-results-path",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDSCORE_RESULTS_PATH"),
        help="Explicit WorldScore result artifact path.",
    )
    parser.add_argument(
        "--worldscore-output-dir",
        "--evaluation-output-dir",
        dest="worldscore_output_dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDSCORE_OUTPUT_DIR"),
    )
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_WORLDSCORE_TIMEOUT", "7200")))
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--only-calculate-mean", action="store_true")
    parser.add_argument("--official-results-path", type=Path)
    parser.add_argument(
        "--stage-dynamic-source",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDSCORE_STAGE_DYNAMIC_SOURCE", "WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
        help="Generated video or frames directory to stage into the official dynamic WorldScore layout.",
    )
    parser.add_argument("--stage-sample-index", type=int, default=int(os.environ.get("WORLDFOUNDRY_WORLDSCORE_STAGE_SAMPLE_INDEX", "0")))
    parser.add_argument(
        "--stage-target-frames",
        type=int,
        default=None,
        help="Bounded frame count to stage; defaults to the official model config frame count.",
    )
    parser.add_argument("--stage-overwrite", action="store_true")
    parser.add_argument("--stage-only", action="store_true", help="Stage and validate/import without running official metrics.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_official_worldscore(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["official_benchmark_verified"] and scorecard["integration_evidence"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_metrics": scorecard["artifacts"]["per_sample_metrics"],
        "official_results": scorecard["artifacts"]["official_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official WorldScore validation {status}")
        print(f"scorecard: {result['scorecard']}")
    artifact_run_ok = bool(scorecard.get("normalization_ok")) or args.stage_only
    return 0 if result["ok"] or artifact_run_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
