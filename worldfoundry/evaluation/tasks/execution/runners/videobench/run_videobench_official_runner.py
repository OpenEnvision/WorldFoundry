#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.utils import REPO_ROOT

from worldfoundry.runtime.env import (  # type: ignore[reportMissingImports]  # noqa: E402
    resolve_data_dir,
    resolve_hf_cache_dir,
)
from worldfoundry.evaluation.utils import HFD_DATASET_CACHE_ROOT, worldfoundry_hfd_dataset_root
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, load_json, utc_now_iso, write_json, write_jsonl

DEFAULT_VIDEOBENCH_ROOT = Path(__file__).resolve().parent / "runtime"
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VIDEO_ARTIFACT_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".gif"})
HUMAN_ANNOTATION_REPO_ID = "Video-Bench/Video-Bench_human_annotation"
OFFICIAL_VIDEOS_REPO_ID = "Video-Bench/Video-Bench_videos"
VIDEOBENCH_FULL_JSON_ASSET = bundled_benchmark_asset("video-bench", "VideoBench_full.json")

DIMENSION_SPECS: dict[str, dict[str, Any]] = {
    "imaging_quality": {
        "metric_id": "imaging_quality",
        "name": "Imaging Quality",
        "group": "static_quality",
        "max_score": 5.0,
    },
    "aesthetic_quality": {
        "metric_id": "aesthetic_quality",
        "name": "Aesthetic Quality",
        "group": "static_quality",
        "max_score": 5.0,
    },
    "temporal_consistency": {
        "metric_id": "temporal_consistency",
        "name": "Temporal Consistency",
        "group": "dynamic_quality",
        "max_score": 5.0,
    },
    "motion_effects": {
        "metric_id": "motion_effects",
        "name": "Motion Effects",
        "group": "dynamic_quality",
        "max_score": 5.0,
    },
    "video-text consistency": {
        "metric_id": "video_text_consistency",
        "name": "Video-Text Consistency",
        "group": "video_text_alignment",
        "max_score": 5.0,
    },
    "object_class": {
        "metric_id": "object_class_consistency",
        "name": "Object-Class Consistency",
        "group": "video_text_alignment",
        "max_score": 3.0,
    },
    "color": {
        "metric_id": "color_consistency",
        "name": "Color Consistency",
        "group": "video_text_alignment",
        "max_score": 3.0,
    },
    "action": {
        "metric_id": "action_consistency",
        "name": "Action Consistency",
        "group": "video_text_alignment",
        "max_score": 3.0,
    },
    "scene": {
        "metric_id": "scene_consistency",
        "name": "Scene Consistency",
        "group": "video_text_alignment",
        "max_score": 3.0,
    },
}
METRIC_ORDER = tuple(spec["metric_id"] for spec in DIMENSION_SPECS.values()) + ("videobench_average",)


def first_env_path(*names: str) -> Path | None:
    """
    Resolve the first non-empty path from a list of environment variables.

    Args:
        *names: Environment variable names ordered from highest to lowest priority.
    """
    return env_path(*names)


def env_list(name: str) -> list[str] | None:
    value = os.environ.get(name)
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def hf_cache_roots() -> list[Path]:
    """
    Return standard Hugging Face cache roots that may contain downloaded dataset snapshots.

    Args:
        None.
    """
    roots: list[Path] = [resolve_hf_cache_dir()]

    resolved: list[Path] = []
    for root in roots:
        path = root.expanduser()
        if path not in resolved:
            resolved.append(path)
    return resolved


def hf_snapshot_candidates(repo_id: str) -> list[Path]:
    """
    Locate local Hugging Face dataset snapshot directories for a repo id.

    Args:
        repo_id: Hugging Face dataset id such as ``Video-Bench/Video-Bench_human_annotation``.
    """
    repo_dir_name = "datasets--" + repo_id.replace("/", "--")
    candidates: list[Path] = []
    for cache_root in hf_cache_roots():
        snapshots_root = cache_root / repo_dir_name / "snapshots"
        if snapshots_root.is_dir():
            candidates.extend(path for path in sorted(snapshots_root.iterdir()) if path.is_dir())
    return candidates


def common_dataset_candidates(repo_id: str) -> list[Path]:
    """
    Build local dataset directory candidates for explicit downloads and HF snapshots.

    Args:
        repo_id: Hugging Face dataset id.
    """
    owner, name = repo_id.split("/", 1)
    data_dir = resolve_data_dir()
    benchmark_data_root = worldfoundry_hfd_dataset_root()
    hfd_name = repo_id.replace("/", "__")
    return [
        benchmark_data_root / hfd_name,
        benchmark_data_root / owner / name,
        data_dir / owner / name,
        data_dir / hfd_name,
        data_dir / "hfd_datasets" / hfd_name,
        HFD_DATASET_CACHE_ROOT / hfd_name,
        REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "external" / owner / name,
        REPO_ROOT / "data" / "benchmark_zoo" / owner / name,
        REPO_ROOT / "tmp" / "benchmark_zoo" / "datasets" / owner / name,
        REPO_ROOT / "tmp" / "benchmark_zoo" / "datasets" / name,
        *hf_snapshot_candidates(repo_id),
    ]


def looks_like_annotation_root(path: Path) -> bool:
    """
    Check whether a directory contains Video-Bench human annotation JSON files.

    Args:
        path: Candidate annotation dataset root.
    """
    return path.is_dir() and any((path / f"{dimension}.json").is_file() for dimension in DIMENSION_SPECS)


def looks_like_video_root(path: Path) -> bool:
    """
    Check whether a directory resembles a Video-Bench video tree.

    Args:
        path: Candidate generated or official video dataset root.
    """
    if not path.is_dir():
        return False
    dimension_names = set(DIMENSION_SPECS) | {"overall_consistency", "video-text consistency"}
    if any((path / name).exists() for name in dimension_names):
        return True
    if any(child.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS for child in path.iterdir() if child.is_file()):
        return True
    return any(child.suffix.lower() == ".zip" for child in path.iterdir() if child.is_file())


def resolve_annotation_root(args: argparse.Namespace, *, required: bool = False) -> Path | None:
    """
    Resolve the Video-Bench human annotation root from CLI, env, or local HF cache.

    Args:
        args: Parsed CLI namespace.
        required: Whether a missing annotation root should fail the run.
    """
    explicit = args.annotation_root or first_env_path(
        "WORLDFOUNDRY_VIDEOBENCH_ANNOTATION_ROOT",
        "WORLDFOUNDRY_VIDEOBENCH_HUMAN_ANNOTATION_ROOT",
    )
    if explicit is not None:
        root = explicit.expanduser()
        if looks_like_annotation_root(root):
            return root
        raise FileNotFoundError(f"Video-Bench annotation root must contain dimension JSON files: {root}")

    for candidate in common_dataset_candidates(HUMAN_ANNOTATION_REPO_ID):
        if looks_like_annotation_root(candidate.expanduser()):
            return candidate.expanduser()
    if required:
        raise FileNotFoundError(
            "Video-Bench prompt suite not found; pass --full-json-dir or --annotation-root "
            "pointing at Video-Bench/Video-Bench_human_annotation"
        )
    return None


def resolve_official_videos_root(args: argparse.Namespace) -> Path | None:
    """
    Resolve a downloaded official Video-Bench videos dataset root when available.

    Args:
        args: Parsed CLI namespace.
    """
    explicit = args.official_videos_root or first_env_path(
        "WORLDFOUNDRY_VIDEOBENCH_OFFICIAL_VIDEOS_ROOT",
        "WORLDFOUNDRY_VIDEOBENCH_VIDEO_DATA_ROOT",
    )
    if explicit is not None:
        root = explicit.expanduser()
        if looks_like_video_root(root):
            return root
        raise FileNotFoundError(f"Video-Bench official videos root is not a video tree: {root}")

    for candidate in common_dataset_candidates(OFFICIAL_VIDEOS_REPO_ID):
        if looks_like_video_root(candidate.expanduser()):
            return candidate.expanduser()
    return None


def resolve_videos_path(args: argparse.Namespace) -> Path | None:
    """
    Resolve the video directory consumed by official Video-Bench ``--videos_path``.

    Args:
        args: Parsed CLI namespace.
    """
    explicit = (
        args.videos_path
        or args.generated_video_dir
        or args.model_output_dir
        or first_env_path(
            "WORLDFOUNDRY_VIDEOBENCH_VIDEOS_PATH",
            "WORLDFOUNDRY_VIDEOBENCH_GENERATED_VIDEO_DIR",
            "WORLDFOUNDRY_VIDEOBENCH_MODEL_OUTPUT_DIR",
            "WORLDFOUNDRY_GENERATED_ARTIFACT_DIR",
            "WORLDFOUNDRY_MODEL_OUTPUT_DIR",
        )
        or args.official_videos_root
        or resolve_official_videos_root(args)
    )
    if explicit is None:
        return None
    root = explicit.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Video-Bench videos path not found: {root}")
    return root


def video_tree_summary(path: Path | None) -> dict[str, Any]:
    """
    Summarize a generated or official video tree without reading video payloads.

    Args:
        path: Optional video tree root.
    """
    if path is None:
        return {"available": False}
    dimension_dirs = [child.name for child in sorted(path.iterdir()) if child.is_dir()]
    zip_files = [child.name for child in sorted(path.iterdir()) if child.is_file() and child.suffix.lower() == ".zip"]
    video_count = 0
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in VIDEO_ARTIFACT_EXTENSIONS:
            video_count += 1
    return {
        "available": True,
        "path": str(path.resolve()),
        "dimension_dirs": dimension_dirs,
        "zip_files": zip_files,
        "video_file_count": video_count,
    }


def annotation_root_summary(path: Path | None) -> dict[str, Any]:
    """
    Summarize a Video-Bench human annotation root.

    Args:
        path: Optional annotation dataset root.
    """
    if path is None:
        return {"available": False}
    files = [f"{dimension}.json" for dimension in DIMENSION_SPECS if (path / f"{dimension}.json").is_file()]
    return {
        "available": bool(files),
        "path": str(path.resolve()),
        "dimension_files": files,
        "dimension_count": len(files),
    }


def build_full_json_from_annotations(annotation_root: Path, output_path: Path, dimensions: list[str]) -> Path:
    """
    Build official ``VideoBench_full.json`` prompt metadata from downloaded HF annotations.

    Args:
        annotation_root: Root of ``Video-Bench/Video-Bench_human_annotation``.
        output_path: JSON file written for the official evaluator.
        dimensions: Canonical Video-Bench dimensions requested for this run.
    """
    prompt_dimensions: dict[str, set[str]] = {}
    requested = set(dimensions)
    for dimension in DIMENSION_SPECS:
        if dimension not in requested:
            continue
        annotation_path = annotation_root / f"{dimension}.json"
        if not annotation_path.is_file():
            continue
        rows = load_json(annotation_path)
        if not isinstance(rows, list):
            raise ValueError(f"Video-Bench annotation JSON must be a list: {annotation_path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt") or row.get("prompt_en")
            if isinstance(prompt, str) and prompt.strip():
                prompt_dimensions.setdefault(prompt.strip(), set()).add(dimension)

    if not prompt_dimensions:
        raise FileNotFoundError(f"no Video-Bench prompts found in annotation root: {annotation_root}")

    dimension_order = list(DIMENSION_SPECS)
    payload = [
        {
            "prompt": prompt,
            "dimension": [dimension for dimension in dimension_order if dimension in dimensions_for_prompt],
        }
        for prompt, dimensions_for_prompt in prompt_dimensions.items()
    ]
    write_json(output_path, payload)
    return output_path


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("score", "raw_score", "value", "mean", "average"):
            if key in value:
                number = scalar(value[key])
                if number is not None:
                    return number
    return None


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def canonical_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_dimension(value: str | None) -> str | None:
    if not value:
        return None
    key = canonical_key(value)
    for dimension, spec in DIMENSION_SPECS.items():
        if key in {canonical_key(dimension), canonical_key(spec["metric_id"]), canonical_key(spec["name"])}:
            return dimension
    return None


def infer_dimension_from_path(path: Path) -> str | None:
    candidates = [path.stem, path.parent.name]
    if path.name.endswith("_score_results.json"):
        candidates.append(path.name.removesuffix("_score_results.json").removeprefix("results_"))
    for candidate in candidates:
        dimension = canonical_dimension(candidate)
        if dimension is not None:
            return dimension
    return None


def normalize_videobench_score(raw_score: float | None, max_score: float) -> float | None:
    if raw_score is None:
        return None
    if raw_score < 1.0:
        return max(0.0, min(1.0, raw_score))
    if 1.0 <= raw_score <= max_score and max_score > 1.0:
        return max(0.0, min(1.0, (raw_score - 1.0) / (max_score - 1.0)))
    if raw_score == 1.0 and max_score <= 1.0:
        return raw_score
    return max(0.0, min(1.0, raw_score / max_score))


def load_upstream_results(path: Path) -> tuple[dict[str, Any], Path]:
    if path.is_file():
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Video-Bench result JSON must be an object: {path}")
        inferred_dimension = infer_dimension_from_path(path)
        if inferred_dimension is not None and {"average_scores", "scores"} & set(payload):
            payload = {inferred_dimension: payload}
        return payload, path

    if path.is_dir():
        payload: dict[str, Any] = {}
        for result_path in sorted(path.rglob("*_score_results.json")):
            dimension = infer_dimension_from_path(result_path)
            if dimension is None:
                continue
            result = load_json(result_path)
            if isinstance(result, dict):
                payload[dimension] = result
        if not payload:
            raise FileNotFoundError(f"no Video-Bench *_score_results.json files found under: {path}")
        return payload, path

    raise FileNotFoundError(f"Video-Bench results path not found: {path}")


def dimension_blocks(raw_results: dict[str, Any], upstream_results_path: Path) -> dict[str, dict[str, Any]]:
    blocks: dict[str, dict[str, Any]] = {}
    if {"average_scores", "scores"} & set(raw_results):
        dimension = canonical_dimension(raw_results.get("dimension")) or infer_dimension_from_path(upstream_results_path)
        if dimension is None:
            raise ValueError("single Video-Bench result JSON requires a dimension field or dimension-named path")
        return {dimension: raw_results}

    for key, value in raw_results.items():
        dimension = canonical_dimension(str(key))
        if dimension is not None and isinstance(value, dict):
            blocks[dimension] = value
    return blocks


def sample_rows_for_dimension(dimension: str, block: dict[str, Any], upstream_results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = block.get("scores")
    if not isinstance(scores, dict):
        return rows

    spec = DIMENSION_SPECS[dimension]
    for sample_id, sample in scores.items():
        if not isinstance(sample, dict):
            continue
        prompt = sample.get("prompt_en") or sample.get("prompt") or sample.get("text")
        for model_name, value in sample.items():
            if str(model_name) in {"prompt_en", "prompt", "text"}:
                continue
            raw_score = scalar(value)
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "dimension": dimension,
                    "metric_id": spec["metric_id"],
                    "model_name": str(model_name),
                    "prompt": prompt,
                    "available": raw_score is not None,
                    "raw_score": raw_score,
                    "normalized_score": normalize_videobench_score(raw_score, spec["max_score"]),
                    "official_scale": [1.0, spec["max_score"]],
                    "source": str(upstream_results_path.resolve()),
                }
            )
    return rows


def dimension_score(dimension: str, block: dict[str, Any], sample_rows: list[dict[str, Any]]) -> tuple[float | None, str, int]:
    average_scores = block.get("average_scores")
    if isinstance(average_scores, dict):
        values = [scalar(value) for value in average_scores.values()]
        score = mean(values)
        count = len([value for value in values if value is not None])
        if score is not None:
            return score, "mean_official_average_scores", count

    values = [scalar(row.get("raw_score")) for row in sample_rows if row.get("dimension") == dimension]
    score = mean(values)
    count = len([value for value in values if value is not None])
    if score is not None:
        return score, "mean_sample_scores", count

    direct = scalar(block.get("score"))
    if direct is not None:
        return direct, "score", 1
    return None, "score_not_found", 0


def normalize_videobench_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    output_dir: Path,
    upstream_results_path: Path,
    annotation_root: Path | None,
    official_videos_root: Path | None,
    videos_path: Path | None,
    full_json_path: Path | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path,
    stderr_path: Path,
    blocked_reasons: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_scores_path = output_dir / "per_sample_scores.jsonl"

    blocks = dimension_blocks(raw_results, upstream_results_path)
    all_sample_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}

    for dimension, spec in DIMENSION_SPECS.items():
        block = blocks.get(dimension, {})
        sample_rows = sample_rows_for_dimension(dimension, block, upstream_results_path) if block else []
        all_sample_rows.extend(sample_rows)
        raw_score, source, sample_count = dimension_score(dimension, block, sample_rows) if block else (None, "dimension_missing", 0)
        normalized_score = normalize_videobench_score(raw_score, spec["max_score"])
        row = {
            "metric_id": spec["metric_id"],
            "name": spec["name"],
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "higher_is_better": True,
            "official_scale": [1.0, spec["max_score"]],
            "source": source,
            "sample_count": sample_count,
            "dimension": dimension,
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_videobench_results"
        else:
            leaderboard[spec["metric_id"]] = raw_score
        metric_rows.append(row)
        per_metric[spec["metric_id"]] = row

    normalized_components = [
        row["normalized_score"]
        for row in metric_rows
        if row["metric_id"] != "videobench_average" and row["normalized_score"] is not None
    ]
    average_score = mean(normalized_components)
    average_row = {
        "metric_id": "videobench_average",
        "name": "Video-Bench Average",
        "available": average_score is not None,
        "raw_score": average_score,
        "normalized_score": average_score,
        "higher_is_better": True,
        "official_scale": [0.0, 1.0],
        "source": "mean_normalized_dimension_scores" if average_score is not None else "score_not_found",
        "sample_count": len([row for row in all_sample_rows if row["available"]]),
    }
    if average_score is None:
        average_row["reason"] = "no_available_videobench_dimension_scores"
    else:
        leaderboard["videobench_average"] = average_score
    metric_rows.append(average_row)
    per_metric["videobench_average"] = average_row

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_scores_path, all_sample_rows)

    available_count = sum(1 for row in metric_rows if row["available"])
    normalizer_only = command is None and not blocked_reasons
    normalized = normalizer_only and returncode == 0 and available_count > 0
    official_verified = command is not None and returncode == 0 and available_count > 0
    normalization_ok = (official_verified or normalized) and not blocked_reasons
    run_status = "blocked" if blocked_reasons else "official_verified" if official_verified else "normalized" if normalized else "failed"
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_videobench_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "Video-Bench",
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_external_judge_api": True,
        },
        "dataset": {
            "sample_count": len(all_sample_rows),
            "upstream_results": str(upstream_results_path.resolve()),
            "human_annotation": annotation_root_summary(annotation_root),
            "official_videos": video_tree_summary(official_videos_root),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "official Video-Bench score normalization validation; leaderboard parity requires the official videos/annotations, prompt suite, and external MLLM judge configuration",
            ],
        },
        "generation": {
            "successful": len({(row["dimension"], row["sample_id"], row["model_name"]) for row in all_sample_rows if row["available"]}),
            "failed": len([row for row in all_sample_rows if not row["available"]]),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "static_quality": ["imaging_quality", "aesthetic_quality"],
                "dynamic_quality": ["temporal_consistency", "motion_effects"],
                "video_text_alignment": [
                    "video_text_consistency",
                    "object_class_consistency",
                    "color_consistency",
                    "action_consistency",
                    "scene_consistency",
                ],
                "aggregate": ["videobench_average"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(all_sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": (official_verified or normalized) and not blocked_reasons,
            "kind": "official_videobench",
            "upstream_results": str(upstream_results_path.resolve()),
            "num_results": len(all_sample_rows),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
            "external_judge_api_required": True,
            "videos_path": None if videos_path is None else str(videos_path.resolve()),
            "full_json_dir": None if full_json_path is None else str(full_json_path.resolve()),
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_validation_required_for_integration_evidence": True,
            "blocked_reasons": blocked_reasons or [],
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_scores": str(per_sample_scores_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": str(stdout_path.resolve()),
            "upstream_stderr": str(stderr_path.resolve()),
        },
        "official_benchmark_verified": official_verified,
        "integration_evidence": command is not None and official_verified,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def resolved_dimensions(args: argparse.Namespace) -> list[str]:
    raw_dimensions = args.dimension or env_list("WORLDFOUNDRY_VIDEOBENCH_DIMENSIONS") or list(DIMENSION_SPECS)
    dimensions: list[str] = []
    for raw in raw_dimensions:
        dimension = canonical_dimension(raw)
        if dimension is None:
            raise ValueError(f"unknown Video-Bench dimension: {raw}")
        if dimension not in dimensions:
            dimensions.append(dimension)
    return dimensions


def required_secret(name: str, value: str | None) -> str:
    if value:
        return value
    raise ValueError(f"official Video-Bench runtime requires {name}; use --official-results-path for offline normalization")


def missing_judge_api_requirements(args: argparse.Namespace) -> list[str]:
    """
    Return missing API credential requirements for the official Video-Bench judge.

    Args:
        args: Parsed CLI namespace containing resolved judge credential values.
    """
    missing: list[str] = []
    if not args.gpt4o_api_key:
        missing.append("WORLDFOUNDRY_VIDEOBENCH_GPT4O_API_KEY or OPENAI_API_KEY")
    if not args.gpt4o_mini_api_key:
        missing.append("WORLDFOUNDRY_VIDEOBENCH_GPT4O_MINI_API_KEY or OPENAI_API_KEY")
    return missing


def write_runtime_config(args: argparse.Namespace, output_dir: Path, dimensions: list[str]) -> Path:
    template_path = args.videobench_root / "config.json"
    config = load_json(template_path) if template_path.is_file() else {}
    if not isinstance(config, dict):
        config = {}

    config["GPT4o_API_KEY"] = required_secret("WORLDFOUNDRY_VIDEOBENCH_GPT4O_API_KEY or OPENAI_API_KEY", args.gpt4o_api_key)
    config["GPT4o_BASE_URL"] = args.gpt4o_base_url or ""
    config["GPT4o_mini_API_KEY"] = required_secret(
        "WORLDFOUNDRY_VIDEOBENCH_GPT4O_MINI_API_KEY or OPENAI_API_KEY",
        args.gpt4o_mini_api_key,
    )
    config["GPT4o_mini_BASE_URL"] = args.gpt4o_mini_base_url or args.gpt4o_base_url or ""

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for dimension in dimensions:
        safe_name = canonical_key(dimension)
        config[f"log_path_{dimension}"] = str((logs_dir / f"{safe_name}.log").resolve())
        if dimension == "video-text consistency":
            config["log_path_overall_consistency"] = str((logs_dir / "video_text_consistency.log").resolve())

    config_path = output_dir / "videobench_config.private.json"
    write_json(config_path, config)
    config_path.chmod(0o600)
    return config_path


def resolve_full_json_path(args: argparse.Namespace, output_dir: Path, dimensions: list[str], annotation_root: Path | None) -> Path:
    """
    Resolve or build the prompt metadata JSON required by official Video-Bench.

    Args:
        args: Parsed CLI namespace.
        output_dir: Runner output directory for generated prompt metadata.
        dimensions: Canonical Video-Bench dimensions requested for this run.
        annotation_root: Optional HF human annotation root used to build the prompt suite.
    """
    explicit = args.full_json_dir or first_env_path("WORLDFOUNDRY_VIDEOBENCH_FULL_JSON_DIR")
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video-Bench full prompt JSON not found: {path}")
        return path

    if VIDEOBENCH_FULL_JSON_ASSET.is_file():
        return VIDEOBENCH_FULL_JSON_ASSET

    upstream_default = args.videobench_root / "videobench" / "VideoBench_full.json"
    if upstream_default.is_file():
        return upstream_default

    if annotation_root is None:
        annotation_root = resolve_annotation_root(args, required=True)
    return build_full_json_from_annotations(annotation_root, output_dir / "VideoBench_full.from_annotations.json", dimensions)


def collect_official_output(output_dir: Path) -> tuple[dict[str, Any], Path]:
    return load_upstream_results(output_dir)


def build_official_command(
    args: argparse.Namespace,
    config_path: Path,
    upstream_output_dir: Path,
    dimensions: list[str],
    videos_path: Path,
    full_json_path: Path,
) -> list[str]:
    command = [
        args.python,
        str(args.videobench_root / "evaluate.py"),
        "--output_path",
        str(upstream_output_dir),
        "--config_path",
        str(config_path),
        "--log_path",
        str((args.output_dir / "logs").resolve()),
        "--full_json_dir",
        str(full_json_path),
        "--videos_path",
        str(videos_path),
        "--dimension",
        *dimensions,
        "--mode",
        args.mode,
    ]
    if args.models:
        command.extend(["--models", *args.models])
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.prompt_file:
        command.extend(["--prompt_file", str(args.prompt_file)])
    return command


def run_videobench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    annotation_root = resolve_annotation_root(args, required=False)
    official_videos_root = resolve_official_videos_root(args)

    if args.from_upstream_results:
        raw_results, upstream_results_path = load_upstream_results(args.from_upstream_results)
        return normalize_videobench_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=upstream_results_path,
            annotation_root=annotation_root,
            official_videos_root=official_videos_root,
            videos_path=None,
            full_json_path=None,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    videos_path = resolve_videos_path(args)
    if videos_path is None:
        raise ValueError("--videos-path, --generated-video-dir, --model-output-dir, or --official-results-path is required")
    if not args.videobench_root.is_dir():
        raise FileNotFoundError(f"Video-Bench root not found: {args.videobench_root}")

    dimensions = resolved_dimensions(args)
    upstream_output_dir = output_dir / "upstream"
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    full_json_path = resolve_full_json_path(args, output_dir, dimensions, annotation_root)
    missing_api_requirements = missing_judge_api_requirements(args)
    if missing_api_requirements:
        blocked_results_path = upstream_output_dir / "blocked_missing_judge_api_score_results.json"
        write_json(
            blocked_results_path,
            {
                "dimension": dimensions[0],
                "average_scores": {},
                "scores": {},
                "blocked": True,
                "blocked_reasons": missing_api_requirements,
            },
        )
        return normalize_videobench_results(
            {},
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=blocked_results_path,
            annotation_root=annotation_root,
            official_videos_root=official_videos_root,
            videos_path=videos_path,
            full_json_path=full_json_path,
            command=None,
            duration_seconds=None,
            returncode=1,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            blocked_reasons=missing_api_requirements,
        )

    config_path = write_runtime_config(args, output_dir, dimensions)
    command = build_official_command(args, config_path, upstream_output_dir, dimensions, videos_path, full_json_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.videobench_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=args.videobench_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    raw_results, upstream_results_path = collect_official_output(upstream_output_dir)
    return normalize_videobench_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        upstream_results_path=upstream_results_path,
        annotation_root=annotation_root,
        official_videos_root=official_videos_root,
        videos_path=videos_path,
        full_json_path=full_json_path,
        command=command,
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official Video-Bench or normalize official score outputs to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "video-bench"))
    parser.add_argument("--videobench-root", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_ROOT", DEFAULT_VIDEOBENCH_ROOT))
    parser.add_argument(
        "--official-results-path",
        dest="from_upstream_results",
        type=Path,
        default=env_path("WORLDFOUNDRY_VIDEOBENCH_RESULTS_PATH"),
    )
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--videos-path", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_VIDEOS_PATH"))
    parser.add_argument("--generated-video-dir", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_GENERATED_VIDEO_DIR"))
    parser.add_argument("--model-output-dir", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_MODEL_OUTPUT_DIR"))
    parser.add_argument("--annotation-root", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_ANNOTATION_ROOT"))
    parser.add_argument("--official-videos-root", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_OFFICIAL_VIDEOS_ROOT"))
    parser.add_argument("--full-json-dir", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_FULL_JSON_DIR"))
    parser.add_argument("--dimension", action="append", default=None)
    parser.add_argument("--mode", choices=("standard", "custom_static", "custom_nonstatic"), default=os.environ.get("WORLDFOUNDRY_VIDEOBENCH_MODE", "standard"))
    parser.add_argument("--models", nargs="+", default=env_list("WORLDFOUNDRY_VIDEOBENCH_MODELS") or [])
    parser.add_argument("--prompt", default=os.environ.get("WORLDFOUNDRY_VIDEOBENCH_PROMPT"))
    parser.add_argument("--prompt-file", type=Path, default=env_path("WORLDFOUNDRY_VIDEOBENCH_PROMPT_FILE"))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_VIDEOBENCH_TIMEOUT", "3600")))
    parser.add_argument("--gpt4o-api-key", default=os.environ.get("WORLDFOUNDRY_VIDEOBENCH_GPT4O_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--gpt4o-base-url", default=os.environ.get("WORLDFOUNDRY_VIDEOBENCH_GPT4O_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--gpt4o-mini-api-key",
        default=os.environ.get("WORLDFOUNDRY_VIDEOBENCH_GPT4O_MINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--gpt4o-mini-base-url",
        default=os.environ.get("WORLDFOUNDRY_VIDEOBENCH_GPT4O_MINI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_videobench(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "full_official_ok": scorecard["official_benchmark_verified"] and scorecard["integration_evidence"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_scores": scorecard["artifacts"]["per_sample_scores"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official Video-Bench validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["full_official_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
