#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import cache_root_path
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.utils import REPO_ROOT

IN_TREE_VBENCH_ROOT = Path(__file__).resolve().parent / "runtime"
DEFAULT_VBENCH_ROOT = IN_TREE_VBENCH_ROOT
VBENCH_FULL_INFO_ASSET = bundled_benchmark_asset("vbench", "VBench_full_info.json")
VBENCH_PROMPTS_ASSET_ROOT = bundled_benchmark_asset("vbench", "prompts")
DEFAULT_BERT_BASE_UNCASED = REPO_ROOT.parent / "ckpt" / "WorldScore" / "bert-base-uncased"
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)
VBENCH_QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
VBENCH_TEMPORAL_QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
)
VBENCH_FRAME_QUALITY_DIMENSIONS = ("aesthetic_quality", "imaging_quality")
VBENCH_CUSTOM_INPUT_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
VBENCH_SEMANTIC_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)
VBENCH_DIMENSION_PRESETS = {
    "all": VBENCH_DIMENSIONS,
    "full_16": VBENCH_DIMENSIONS,
    "official": VBENCH_DIMENSIONS,
    "validation": ("aesthetic_quality",),
    "custom_supported": VBENCH_CUSTOM_INPUT_DIMENSIONS,
    "quality": VBENCH_QUALITY_DIMENSIONS,
    "temporal_quality": VBENCH_TEMPORAL_QUALITY_DIMENSIONS,
    "temporal": VBENCH_TEMPORAL_QUALITY_DIMENSIONS,
    "frame_quality": VBENCH_FRAME_QUALITY_DIMENSIONS,
    "frame": VBENCH_FRAME_QUALITY_DIMENSIONS,
    "semantic": VBENCH_SEMANTIC_DIMENSIONS,
    "text_alignment": VBENCH_SEMANTIC_DIMENSIONS,
}
VBENCH_DIMENSION_WEIGHTS = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "temporal_flickering": 1.0,
    "motion_smoothness": 1.0,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
    "object_class": 1.0,
    "multiple_objects": 1.0,
    "human_action": 1.0,
    "color": 1.0,
    "spatial_relationship": 1.0,
    "scene": 1.0,
    "appearance_style": 1.0,
    "temporal_style": 1.0,
    "overall_consistency": 1.0,
}
VBENCH_NORMALIZATION = {
    "subject_consistency": {"min": 0.1462, "max": 1.0},
    "background_consistency": {"min": 0.2615, "max": 1.0},
    "temporal_flickering": {"min": 0.6293, "max": 1.0},
    "motion_smoothness": {"min": 0.706, "max": 0.9975},
    "dynamic_degree": {"min": 0.0, "max": 1.0},
    "aesthetic_quality": {"min": 0.0, "max": 1.0},
    "imaging_quality": {"min": 0.0, "max": 1.0},
    "object_class": {"min": 0.0, "max": 1.0},
    "multiple_objects": {"min": 0.0, "max": 1.0},
    "human_action": {"min": 0.0, "max": 1.0},
    "color": {"min": 0.0, "max": 1.0},
    "spatial_relationship": {"min": 0.0, "max": 1.0},
    "scene": {"min": 0.0, "max": 0.8222},
    "appearance_style": {"min": 0.0009, "max": 0.2855},
    "temporal_style": {"min": 0.0, "max": 0.364},
    "overall_consistency": {"min": 0.0, "max": 0.364},
}


@dataclass(frozen=True)
class VBenchRunRequest:
    """Script-local request for running or normalizing official VBench results."""

    output_dir: str | Path
    videos_path: str | Path | None = None
    dimensions: tuple[str, ...] = ()
    presets: tuple[str, ...] = ()
    benchmark_id: str = "vbench"
    vbench_root: str | Path = DEFAULT_VBENCH_ROOT
    mode: str = "vbench_standard"
    prompt: str | None = None
    prompt_file: str | Path | None = None
    category: str | None = None
    imaging_quality_preprocessing_mode: str = "longer"
    full_json_dir: str | Path | None = None
    python: str = sys.executable
    timeout: int = 1800
    load_ckpt_from_local: bool = False
    read_frame: bool = False
    from_upstream_results: str | Path | None = None

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            benchmark_id=self.benchmark_id,
            vbench_root=Path(self.vbench_root),
            videos_path=None if self.videos_path is None else Path(self.videos_path),
            output_dir=Path(self.output_dir),
            dimension=list(self.dimensions),
            preset=list(self.presets),
            mode=self.mode,
            prompt=self.prompt,
            prompt_file=None if self.prompt_file is None else Path(self.prompt_file),
            category=self.category,
            imaging_quality_preprocessing_mode=self.imaging_quality_preprocessing_mode,
            full_json_dir=None if self.full_json_dir is None else Path(self.full_json_dir),
            python=self.python,
            timeout=self.timeout,
            load_ckpt_from_local=self.load_ckpt_from_local,
            read_frame=self.read_frame,
            from_upstream_results=None
            if self.from_upstream_results is None
            else Path(self.from_upstream_results),
            json=False,
        )


def split_dimensions(values: list[str] | None, presets: list[str] | None = None) -> list[str]:
    raw_values = [*(presets or ()), *(values or ())]
    if not raw_values:
        raw_values = [os.environ.get("WORLDFOUNDRY_VBENCH_PRESET") or os.environ.get("WORLDFOUNDRY_VBENCH_DIMENSIONS", "")]
    dimensions: list[str] = []
    for raw_value in raw_values:
        for item in str(raw_value).replace(",", " ").split():
            item = canonical_dimension_id(item)
            expanded = VBENCH_DIMENSION_PRESETS.get(item, (item,))
            for dimension in expanded:
                if dimension and dimension not in dimensions:
                    dimensions.append(dimension)
    return dimensions


def default_vbench_root() -> Path:
    return env_path("WORLDFOUNDRY_VBENCH_ROOT", IN_TREE_VBENCH_ROOT)


def default_vbench_cache_dir() -> Path:
    return env_path("WORLDFOUNDRY_VBENCH_CACHE_DIR", cache_root_path() / "models" / "vbench")


def list_video_files(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [str(path)] if path.suffix.lower() in {".mp4", ".gif"} else []
    return [
        str(item)
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in {".mp4", ".gif"}
    ]


def materialize_custom_prompt_file(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.mode != "custom_input" or not args.prompt or args.prompt_file is not None:
        return None
    if args.videos_path is None or args.videos_path.is_file():
        return None

    videos_root = args.videos_path.resolve()
    video_files = list_video_files(videos_root)
    if not video_files:
        return None

    prompts: dict[str, str] = {}
    for item in video_files:
        path = Path(item).resolve()
        try:
            key = path.relative_to(videos_root).as_posix()
        except ValueError:
            key = path.name
        prompts[key] = args.prompt

    prompt_file = output_dir / "custom_prompts.json"
    write_json(prompt_file, prompts)
    args.prompt_file = prompt_file
    args.prompt = None
    return prompt_file


def extract_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, list) or isinstance(value, tuple):
        for item in value:
            scalar = extract_scalar(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(value, dict):
        for key in ("score", "value", "mean", "average", "avg", "all_results", "result"):
            if key in value:
                scalar = extract_scalar(value[key])
                if scalar is not None:
                    return scalar
    return None


def canonical_dimension_id(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def raw_result_for_dimension(raw_results: dict[str, Any], dimension: str) -> Any:
    metric_id = canonical_dimension_id(dimension)
    candidates = (metric_id, metric_id.replace("_", " "))
    for key in candidates:
        if key in raw_results:
            return raw_results[key]
    return None


def load_upstream_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"VBench result JSON must be an object: {path}")
    return payload


def resolved_full_json_path(full_json_dir: Path | None, vbench_root: Path) -> Path:
    path = full_json_dir or VBENCH_FULL_INFO_ASSET
    if not path.exists():
        path = vbench_root / "vbench" / "VBench_full_info.json"
    return path / "VBench_full_info.json" if path.is_dir() else path


def official_prompt_suite_requested(args: argparse.Namespace) -> bool:
    presets = {canonical_dimension_id(value) for value in (getattr(args, "preset", None) or [])}
    return args.mode == "vbench_standard" or bool(presets & {"full_16", "official"})


def _payload_dimensions(payload: dict[str, Any]) -> set[str]:
    raw_dimensions = payload.get("dimension", payload.get("dimensions"))
    if isinstance(raw_dimensions, str):
        return {canonical_dimension_id(raw_dimensions)}
    if isinstance(raw_dimensions, list) or isinstance(raw_dimensions, tuple):
        return {
            canonical_dimension_id(str(item))
            for item in raw_dimensions
            if str(item).strip()
        }
    return set()


def _matches_requested_dimensions(payload: dict[str, Any], dimensions: list[str]) -> bool:
    payload_dimensions = _payload_dimensions(payload)
    requested_dimensions = {canonical_dimension_id(item) for item in dimensions}
    return not payload_dimensions or not requested_dimensions or bool(payload_dimensions & requested_dimensions)


def _prompt_text(payload: dict[str, Any]) -> str | None:
    for key in ("prompt_en", "prompt", "prompt_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _collect_prompt_suite_entries(
    payload: Any,
    prompts: list[str],
    video_names: set[str],
    *,
    dimensions: list[str],
) -> None:
    if isinstance(payload, dict):
        prompt = _prompt_text(payload)
        if prompt and _matches_requested_dimensions(payload, dimensions):
            prompts.append(prompt)
        for key, value in payload.items():
            key_id = str(key).lower()
            if key_id in {"video", "videos", "video_path", "videos_path", "video_list", "filename", "file"}:
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if isinstance(item, str) and Path(item).suffix.lower() in {".mp4", ".gif"}:
                        video_names.add(Path(item).name)
            _collect_prompt_suite_entries(value, prompts, video_names, dimensions=dimensions)
    elif isinstance(payload, list):
        for item in payload:
            _collect_prompt_suite_entries(item, prompts, video_names, dimensions=dimensions)


def _expected_standard_prompt_video_names(prompts: list[str], video_files: list[str]) -> set[str]:
    suffixes = sorted({Path(path).suffix for path in video_files if Path(path).suffix.lower() in {".mp4", ".gif"}})
    if not suffixes:
        suffixes = [".mp4"]
    return {f"{prompt}-{index}{suffix}" for prompt in prompts for suffix in suffixes for index in range(5)}


def validate_prompt_suite_materialization(args: argparse.Namespace) -> dict[str, Any]:
    requested = official_prompt_suite_requested(args)
    full_json_path = resolved_full_json_path(args.full_json_dir, args.vbench_root)
    video_files = (
        [
            str(item)
            for item in sorted(args.videos_path.iterdir())
            if item.is_file() and item.suffix.lower() in {".mp4", ".gif"}
        ]
        if args.videos_path is not None and args.videos_path.is_dir()
        else list_video_files(args.videos_path)
    )
    actual_video_names = {Path(path).name for path in video_files}
    requested_dimensions = {canonical_dimension_id(item) for item in args.dimension}
    report: dict[str, Any] = {
        "ok": True,
        "leaderboard_valid": False,
        "requested": requested,
        "canonical_mode": args.mode == "vbench_standard",
        "mode": args.mode,
        "presets": list(getattr(args, "preset", None) or []),
        "requested_dimensions": sorted(requested_dimensions),
        "expected_dimensions": list(VBENCH_DIMENSIONS),
        "complete_dimension_coverage": requested_dimensions == set(VBENCH_DIMENSIONS),
        "full_json_dir": str(full_json_path),
        "full_json_exists": full_json_path.is_file(),
        "videos_path": None if args.videos_path is None else str(args.videos_path),
        "generated_file_count": len(video_files),
        "expected_prompt_count": 0,
        "expected_video_count": 0,
        "covered_video_count": 0,
        "complete_sample_coverage": False,
        "full_suite_complete": False,
        "issues": [],
        "reasons": [],
    }
    if not requested:
        return report

    prompts: list[str] = []
    expected_video_names: set[str] = set()
    if not full_json_path.is_file():
        report["issues"].append(
            {
                "code": "missing_full_json_dir",
                "message": f"official VBench prompt suite metadata was not found at {full_json_path}",
            }
        )
    else:
        try:
            payload = json.loads(full_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report["issues"].append(
                {
                    "code": "invalid_full_json_dir",
                    "message": f"official VBench prompt suite metadata is not valid JSON: {exc}",
                }
            )
        else:
            _collect_prompt_suite_entries(payload, prompts, expected_video_names, dimensions=list(args.dimension))

    if not expected_video_names and prompts:
        expected_video_names = _expected_standard_prompt_video_names(prompts, video_files)
    covered_video_names = actual_video_names & expected_video_names
    report["expected_prompt_count"] = len(prompts)
    report["expected_video_count"] = len(expected_video_names)
    report["covered_video_count"] = len(covered_video_names)
    report["complete_sample_coverage"] = bool(expected_video_names) and covered_video_names == expected_video_names
    report["covered_videos"] = sorted(covered_video_names)
    report["sample_expected_videos"] = sorted(expected_video_names)[:10]

    if not video_files:
        report["issues"].append(
            {
                "code": "no_generated_videos",
                "message": "videos_path did not contain any .mp4 or .gif files for official VBench evaluation",
            }
        )
    elif full_json_path.is_file() and not prompts and not expected_video_names:
        report["issues"].append(
            {
                "code": "no_prompt_suite_entries_for_dimensions",
                "message": "official VBench prompt suite metadata did not contain entries for the requested dimensions",
            }
        )
    elif expected_video_names and not covered_video_names:
        report["issues"].append(
            {
                "code": "no_materialized_prompt_or_video",
                "message": "videos_path did not contain any generated video whose filename matches the requested official VBench prompt suite",
            }
        )

    report["ok"] = not report["issues"]
    report["reasons"] = [
        issue["message"]
        for issue in report["issues"]
        if isinstance(issue, dict) and issue.get("message")
    ]
    report["full_suite_complete"] = (
        report["ok"]
        and report["requested"]
        and report["canonical_mode"]
        and report["complete_dimension_coverage"]
        and report["complete_sample_coverage"]
    )
    return report


def latest_upstream_results(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*_eval_results.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def official_normalized_weighted_score(metric_id: str, raw_score: float) -> float:
    bounds = VBENCH_NORMALIZATION[metric_id]
    normalized = (raw_score - bounds["min"]) / (bounds["max"] - bounds["min"])
    return normalized * VBENCH_DIMENSION_WEIGHTS[metric_id]


def aggregate_dimension_group(raw_scores: dict[str, float], dimensions: tuple[str, ...]) -> float | None:
    if any(metric_id not in raw_scores for metric_id in dimensions):
        return None
    weighted_scores = [official_normalized_weighted_score(metric_id, raw_scores[metric_id]) for metric_id in dimensions]
    weight_total = sum(VBENCH_DIMENSION_WEIGHTS[metric_id] for metric_id in dimensions)
    return sum(weighted_scores) / weight_total


def compute_vbench_aggregates(raw_scores: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    temporal_quality = aggregate_dimension_group(raw_scores, VBENCH_TEMPORAL_QUALITY_DIMENSIONS)
    frame_quality = aggregate_dimension_group(raw_scores, VBENCH_FRAME_QUALITY_DIMENSIONS)
    quality_score = aggregate_dimension_group(raw_scores, VBENCH_QUALITY_DIMENSIONS)
    text_alignment = aggregate_dimension_group(raw_scores, VBENCH_SEMANTIC_DIMENSIONS)
    aggregate_specs = [
        ("temporal_quality", temporal_quality, VBENCH_TEMPORAL_QUALITY_DIMENSIONS),
        ("frame_quality", frame_quality, VBENCH_FRAME_QUALITY_DIMENSIONS),
        ("text_alignment", text_alignment, VBENCH_SEMANTIC_DIMENSIONS),
    ]
    if quality_score is not None and text_alignment is not None:
        aggregate_specs.insert(
            0,
            (
                "overall_quality",
                (quality_score * 4.0 + text_alignment) / 5.0,
                (*VBENCH_QUALITY_DIMENSIONS, *VBENCH_SEMANTIC_DIMENSIONS),
            ),
        )

    for metric_id, score, dimensions in aggregate_specs:
        if score is None:
            continue
        rows.append(
            {
                "metric_id": metric_id,
                "available": True,
                "raw_score": score,
                "normalized_score": score,
                "aggregate": True,
                "raw_value": {
                    "computed_from": list(dimensions),
                    "normalization": "official_vbench_minmax_and_weights",
                },
            }
        )
    return rows


def normalize_vbench_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    dimensions: list[str],
    output_dir: Path,
    upstream_results_path: Path,
    videos_path: Path | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    prompt_suite_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_files = list_video_files(videos_path)
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    requested_dimensions = dimensions or sorted(raw_results)
    raw_scores: dict[str, float] = {}

    for dimension in requested_dimensions:
        metric_id = canonical_dimension_id(dimension)
        raw_value = raw_result_for_dimension(raw_results, metric_id)
        score = extract_scalar(raw_value)
        row = {
            "metric_id": metric_id,
            "available": score is not None,
            "raw_score": score,
            "normalized_score": score,
            "raw_value": raw_value,
        }
        if score is None:
            row["reason"] = "score_not_found_in_vbench_results"
        else:
            leaderboard[metric_id] = score
            if metric_id in VBENCH_DIMENSIONS:
                raw_scores[metric_id] = score
        metric_rows.append(row)
        per_metric[metric_id] = {key: value for key, value in row.items() if key != "raw_value"}

    aggregate_rows = compute_vbench_aggregates(raw_scores)
    for row in aggregate_rows:
        metric_id = str(row["metric_id"])
        leaderboard[metric_id] = float(row["raw_score"])
        metric_rows.append(row)
        per_metric[metric_id] = {key: value for key, value in row.items() if key != "raw_value"}

    available_count = sum(1 for row in metric_rows if row["available"])
    dimension_metrics = [
        str(row["metric_id"])
        for row in metric_rows
        if row["available"] and row.get("aggregate") is not True
    ]
    aggregate_metrics = [
        str(row["metric_id"])
        for row in aggregate_rows
        if row["available"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    prompt_suite_validation = prompt_suite_validation or {
        "ok": True,
        "leaderboard_valid": False,
        "requested": False,
        "issues": [],
    }
    validation_reasons = [
        issue["message"]
        for issue in prompt_suite_validation.get("issues", [])
        if isinstance(issue, dict) and issue.get("message")
    ]
    integration_evidence = (
        command is not None
        and returncode == 0
        and bool(dimension_metrics)
        and bool(prompt_suite_validation.get("ok", True))
    )
    full_metric_coverage = set(VBENCH_DIMENSIONS).issubset(dimension_metrics)
    official_verified = (
        integration_evidence
        and full_metric_coverage
        and bool(prompt_suite_validation.get("full_suite_complete", False))
    )
    normalized = command is None and returncode == 0 and available_count > 0

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
            "runner": "benchmark_zoo_vbench_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "VBench",
            "contract_only": False,
            "requires_upstream_runtime": True,
        },
        "dataset": {
            "generated_artifact_dir": None if videos_path is None else str(videos_path),
            "generated_file_count": len(video_files),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                *validation_reasons,
                "official VBench runtime validation; use the full upstream prompt suite and submission protocol for leaderboard evidence",
            ],
        },
        "generation": {
            "successful": len(video_files),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "vbench_dimensions": dimension_metrics,
                "vbench_aggregates": aggregate_metrics,
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
            "kind": "official_vbench",
            "upstream_results": str(upstream_results_path),
            "num_results": available_count,
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": command is None,
            "official_runtime_executed": command is not None,
            "full_metric_coverage": full_metric_coverage,
            "prompt_suite_materialization": prompt_suite_validation,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "normalization_ok": returncode == 0 and available_count > 0,
        "official_benchmark_verified": official_verified,
        "integration_evidence": integration_evidence,
        "leaderboard_valid": False,
        "official_results_imported": normalized,
    }

    write_json(scorecard_path, scorecard)
    write_jsonl(raw_metric_table_path, metric_rows)
    return scorecard


def build_official_command(args: argparse.Namespace, upstream_output_dir: Path) -> list[str]:
    full_json_path = resolved_full_json_path(args.full_json_dir, args.vbench_root)
    command = [
        args.python,
        "-m",
        "worldfoundry.evaluation.tasks.execution.runners.vbench.runtime.entrypoints.base",
        "--output_path",
        str(upstream_output_dir),
        "--full_json_dir",
        str(full_json_path),
        "--videos_path",
        str(args.videos_path),
        "--dimension",
        *args.dimension,
        "--mode",
        args.mode,
        "--imaging_quality_preprocessing_mode",
        args.imaging_quality_preprocessing_mode,
    ]
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.prompt_file:
        command.extend(["--prompt_file", str(args.prompt_file)])
    if args.category:
        command.extend(["--category", args.category])
    if args.load_ckpt_from_local:
        command.extend(["--load_ckpt_from_local", "True"])
    if args.read_frame:
        command.extend(["--read_frame", "True"])
    return command


def validate_vbench_args(args: argparse.Namespace) -> None:
    if args.prompt and args.prompt_file:
        raise ValueError("--prompt and --prompt-file cannot be used together")
    if (args.prompt or args.prompt_file) and args.mode != "custom_input":
        raise ValueError("--prompt/--prompt-file require --mode custom_input")
    if args.category and args.mode != "vbench_category":
        raise ValueError("--category requires --mode vbench_category")
    if args.mode == "vbench_category" and not args.category:
        raise ValueError("--mode vbench_category requires --category")
    if args.mode == "custom_input":
        unsupported = [dimension for dimension in args.dimension if dimension not in VBENCH_CUSTOM_INPUT_DIMENSIONS]
        if unsupported:
            supported = ", ".join(VBENCH_CUSTOM_INPUT_DIMENSIONS)
            raise ValueError(
                "custom_input only supports these VBench dimensions: "
                f"{supported}. Unsupported: {', '.join(unsupported)}"
            )


def run_official_vbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    args.output_dir = output_dir
    if args.vbench_root is None:
        args.vbench_root = default_vbench_root()
    args.vbench_root = args.vbench_root.expanduser().resolve()
    if args.videos_path is not None:
        args.videos_path = args.videos_path.expanduser().resolve()
    if args.full_json_dir is not None:
        args.full_json_dir = args.full_json_dir.expanduser().resolve()
    if args.from_upstream_results is not None:
        args.from_upstream_results = args.from_upstream_results.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    upstream_output_dir = output_dir / "upstream"
    upstream_output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_upstream_results:
        raw_results = load_upstream_results(args.from_upstream_results)
        prompt_suite_validation = validate_prompt_suite_materialization(args)
        return normalize_vbench_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            dimensions=args.dimension,
            output_dir=output_dir,
            upstream_results_path=args.from_upstream_results,
            videos_path=args.videos_path,
            command=None,
            duration_seconds=None,
            returncode=0,
            prompt_suite_validation=prompt_suite_validation,
        )

    if args.videos_path is None:
        raise ValueError("--videos-path or WORLDFOUNDRY_GENERATED_ARTIFACT_DIR is required unless --official-results-path is used")
    full_json_path = resolved_full_json_path(args.full_json_dir, args.vbench_root)
    if not full_json_path.is_file():
        raise FileNotFoundError(f"VBench full-info asset not found: {full_json_path}")
    if not args.dimension:
        raise ValueError("--dimension or WORLDFOUNDRY_VBENCH_DIMENSIONS is required")
    validate_vbench_args(args)
    materialize_custom_prompt_file(args, output_dir)
    prompt_suite_validation = validate_prompt_suite_materialization(args)

    command = build_official_command(args, upstream_output_dir)
    if not prompt_suite_validation.get("ok", True):
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(
            "skipped official VBench execution because prompt suite materialization preflight failed\n",
            encoding="utf-8",
        )
        failure_results = {
            dimension: {
                "error": "prompt suite materialization preflight failed",
                "issues": prompt_suite_validation.get("issues", []),
            }
            for dimension in args.dimension
        }
        upstream_results_path = upstream_output_dir / "preflight_failed_eval_results.json"
        write_json(upstream_results_path, failure_results)
        return normalize_vbench_results(
            failure_results,
            benchmark_id=args.benchmark_id,
            dimensions=args.dimension,
            output_dir=output_dir,
            upstream_results_path=upstream_results_path,
            videos_path=args.videos_path,
            command=command,
            duration_seconds=0.0,
            returncode=2,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            prompt_suite_validation=prompt_suite_validation,
        )

    env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT), str(args.vbench_root)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("VBENCH_CACHE_DIR", str(default_vbench_cache_dir()))
    if VBENCH_PROMPTS_ASSET_ROOT.is_dir():
        env.setdefault("WORLDFOUNDRY_VBENCH_PROMPTS_ROOT", str(VBENCH_PROMPTS_ASSET_ROOT))
    if VBENCH_FULL_INFO_ASSET.is_file():
        env.setdefault("WORLDFOUNDRY_VBENCH_FULL_INFO", str(VBENCH_FULL_INFO_ASSET))
    if "VBENCH_BERT_BASE_UNCASED" not in env and DEFAULT_BERT_BASE_UNCASED.is_dir():
        env["VBENCH_BERT_BASE_UNCASED"] = str(DEFAULT_BERT_BASE_UNCASED)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=args.vbench_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    duration_seconds = time.monotonic() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    upstream_results_path = latest_upstream_results(upstream_output_dir)
    if upstream_results_path is None:
        failure_results = {dimension: {"error": "missing upstream VBench *_eval_results.json"} for dimension in args.dimension}
        upstream_results_path = upstream_output_dir / "missing_eval_results.json"
        write_json(upstream_results_path, failure_results)
        raw_results = failure_results
    else:
        raw_results = load_upstream_results(upstream_results_path)

    return normalize_vbench_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        dimensions=args.dimension,
        output_dir=output_dir,
        upstream_results_path=upstream_results_path,
        videos_path=args.videos_path,
        command=command,
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        prompt_suite_validation=prompt_suite_validation,
    )


def run_vbench(request: VBenchRunRequest | None = None, **kwargs: Any) -> dict[str, Any]:
    if request is not None and kwargs:
        raise TypeError("pass either VBenchRunRequest or keyword arguments, not both")
    if request is None:
        request = VBenchRunRequest(**kwargs)
    args = request.to_namespace()
    args.dimension = split_dimensions(args.dimension, args.preset)
    return run_official_vbench(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official VBench and normalize its output to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "vbench"))
    parser.add_argument("--vbench-root", type=Path, default=None)
    parser.add_argument("--videos-path", type=Path, default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--dimension", action="append")
    parser.add_argument("--preset", action="append")
    parser.add_argument("--mode", default=os.environ.get("WORLDFOUNDRY_VBENCH_MODE", "vbench_standard"))
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--imaging-quality-preprocessing-mode", default=os.environ.get("WORLDFOUNDRY_VBENCH_IMAGING_QUALITY_PREPROCESSING_MODE", "longer"))
    parser.add_argument("--full-json-dir", type=Path)
    parser.add_argument("--python", default=os.environ.get("WORLDFOUNDRY_VBENCH_PYTHON", sys.executable))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_VBENCH_TIMEOUT", "1800")))
    parser.add_argument("--load-ckpt-from-local", action="store_true")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument("--official-results-path", dest="from_upstream_results", type=Path)
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    return parser


def build_dimensions_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List VBench dimensions and WorldFoundry presets.")
    parser.add_argument("--json", action="store_true")
    return parser


def vbench_dimensions_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "vbench",
        "dimensions": list(VBENCH_DIMENSIONS),
        "presets": {name: list(dimensions) for name, dimensions in sorted(VBENCH_DIMENSION_PRESETS.items())},
        "primary_presets": {
            "all": list(VBENCH_DIMENSIONS),
            "quality": list(VBENCH_QUALITY_DIMENSIONS),
            "semantic": list(VBENCH_SEMANTIC_DIMENSIONS),
            "custom_supported": list(VBENCH_CUSTOM_INPUT_DIMENSIONS),
        },
    }


def dimensions_main(argv: list[str] | None = None) -> int:
    args = build_dimensions_parser().parse_args(argv)
    payload = vbench_dimensions_payload()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("dimensions:")
        for dimension in payload["dimensions"]:
            print(f"  {dimension}")
        print("presets:")
        for name, dimensions in payload["presets"].items():
            print(f"  {name}: {', '.join(dimensions)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dimension = split_dimensions(args.dimension, args.preset)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_official_vbench(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "normalization_ok": scorecard["normalization_ok"],
        "integration_evidence": scorecard["integration_evidence"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: VBench evaluation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


def dispatch_main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"run", "evaluate", "normalize"}:
        return main(args[1:])
    if args and args[0] in {"dimensions", "list-dimensions"}:
        return dimensions_main(args[1:])
    return main(args)


if __name__ == "__main__":
    raise SystemExit(dispatch_main())
