#!/usr/bin/env python3
from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any


from worldfoundry.evaluation.utils import REPO_ROOT

from worldfoundry.evaluation.tasks.execution.framework.benchmark_data import (
    build_generated_video_manifest,
    build_local_dataset_manifest,
    discover_metadata_records,
    expected_stems_from_records,
)
from worldfoundry.evaluation.tasks.execution.framework.io import (
    env_path,
    load_json,
    mean_numeric,
    scalar_number,
    utc_now_iso,
    write_json,
    write_jsonl,
)


DEFAULT_CHRONOMAGIC_ROOT = (
    REPO_ROOT
    / "worldfoundry"
    / "evaluation"
    / "tasks"
    / "execution"
    / "runners"
    / "chronomagic_bench"
    / "runtime"
    / "chronomagic_bench"
)
HF_DATASET_ID = "BestWishYsh/ChronoMagic-Bench"
HF_DATASET_CONFIG = "default"
HF_DATASET_SPLIT = "test"
HF_DATASET_EXPECTED_ROWS = 1799
METRIC_ORDER = ("chronomagic_score", "temporal_transformation")
VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"})
OPEN_EVAL_PARTS = ("1", "2", "3")
FULL_COMPONENTS = ("chscore", "gpt4o_mtscore", "mtscore")
COMPONENT_RESULT_KEYS = {
    "chscore": "Average_CHScore",
    "gpt4o_mtscore": "Average_GPT4o-MTScore",
    "mtscore": "Average_MTScore",
}
LEADERBOARD_RESULT_KEYS = ("Average_CHScore", "Average_GPT4o-MTScore", "Average_MTScore", "UMT-FVD", "UMTScore")
COMPONENT_SCRIPT_PATHS = {
    "chscore": (Path("CHScore") / "step0-get_CHScore.py",),
    "gpt4o_mtscore": (
        Path("GPT4o_MTScore") / "step0-extract_video_frames.py",
        Path("GPT4o_MTScore") / "step1-get_temp_results.py",
        Path("GPT4o_MTScore") / "step2-get_GPT4o-MTScore.py",
    ),
    "mtscore": (Path("MTScore") / "step0-get_MTScore.py",),
}
COMPONENT_ALIASES = {
    "ch": "chscore",
    "chscore": "chscore",
    "gpt4o": "gpt4o_mtscore",
    "gpt4o-mtscore": "gpt4o_mtscore",
    "gpt4o_mtscore": "gpt4o_mtscore",
    "mt": "mtscore",
    "mtscore": "mtscore",
}


def parse_components(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip().lower() in {"", "all", "full", "official"}:
        return FULL_COMPONENTS
    components: list[str] = []
    for token in value.replace("+", ",").split(","):
        normalized = COMPONENT_ALIASES.get(token.strip().lower())
        if normalized is None:
            valid = ", ".join(("all", *sorted(set(COMPONENT_ALIASES))))
            raise ValueError(f"unsupported ChronoMagic component {token!r}; expected one of: {valid}")
        if normalized not in components:
            components.append(normalized)
    return tuple(component for component in FULL_COMPONENTS if component in components)


def component_mode(components: tuple[str, ...]) -> bool:
    return components != FULL_COMPONENTS


scalar = partial(scalar_number, dict_keys=("score", "raw_score", "value", "mean", "average"))
mean = mean_numeric


def required_component_keys(components: tuple[str, ...]) -> dict[str, str]:
    return {component: COMPONENT_RESULT_KEYS[component] for component in components}


def normalize_component(key: str, value: float | None) -> float | None:
    if value is None:
        return None
    if key in {"Average_MTScore", "UMTScore"}:
        return max(0.0, min(1.0, value))
    if key == "Average_GPT4o-MTScore":
        if 1.0 <= value <= 5.0:
            return (value - 1.0) / 4.0
        return max(0.0, min(1.0, value))
    if key == "Average_CHScore":
        if 0.0 <= value <= 5.0:
            return value / 5.0
        if 5.0 < value <= 100.0:
            return value / 100.0
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 100.0:
        return value / 100.0
    return max(0.0, min(1.0, value))


def choose_model_rows(raw_results: dict[str, Any], model_name: str | None) -> dict[str, dict[str, Any]]:
    if model_name:
        value = raw_results.get(model_name)
        if not isinstance(value, dict):
            raise ValueError(f"model {model_name!r} not found in ChronoMagic results")
        return {model_name: value}
    model_rows = {key: value for key, value in raw_results.items() if isinstance(value, dict)}
    if not model_rows:
        raise ValueError("ChronoMagic result JSON does not contain model result objects")
    return model_rows


def extract_scores(raw_results: dict[str, Any], model_name: str | None) -> dict[str, dict[str, Any]]:
    model_rows = choose_model_rows(raw_results, model_name)
    chronomagic_values: list[float] = []
    chronomagic_normalized: list[float] = []
    temporal_values: list[float] = []
    temporal_normalized: list[float] = []
    for row in model_rows.values():
        components = {
            "Average_CHScore": scalar(row.get("Average_CHScore")),
            "Average_MTScore": scalar(row.get("Average_MTScore")),
            "Average_GPT4o-MTScore": scalar(row.get("Average_GPT4o-MTScore")),
        }
        for key, value in components.items():
            if value is not None:
                chronomagic_values.append(value)
            normalized = normalize_component(key, value)
            if normalized is not None:
                chronomagic_normalized.append(normalized)
        for key in ("Average_MTScore", "Average_GPT4o-MTScore"):
            value = components.get(key)
            if value is not None:
                temporal_values.append(value)
            normalized = normalize_component(key, value)
            if normalized is not None:
                temporal_normalized.append(normalized)

    return {
        "chronomagic_score": {
            "raw_score": mean(chronomagic_values),
            "normalized_score": mean(chronomagic_normalized),
            "source": "mean_Average_CHScore_Average_MTScore_Average_GPT4o_MTScore",
            "sample_count": len(model_rows),
        },
        "temporal_transformation": {
            "raw_score": mean(temporal_values),
            "normalized_score": mean(temporal_normalized),
            "source": "mean_Average_MTScore_Average_GPT4o_MTScore",
            "sample_count": len(model_rows),
        },
    }


def component_result_availability(
    raw_results: dict[str, Any],
    model_name: str | None,
    components: tuple[str, ...],
) -> dict[str, Any]:
    required = required_component_keys(components)
    model_rows = choose_model_rows(raw_results, model_name)
    missing_by_model: dict[str, list[str]] = {}
    available_by_model: dict[str, list[str]] = {}
    for name, row in model_rows.items():
        missing: list[str] = []
        available: list[str] = []
        for component, result_key in required.items():
            if scalar(row.get(result_key)) is None:
                missing.append(component)
            else:
                available.append(component)
        missing_by_model[name] = missing
        available_by_model[name] = available

    missing_required = sorted({component for missing in missing_by_model.values() for component in missing})
    available_components = [
        component
        for component in components
        if component not in missing_required
    ]
    return {
        "required_components": list(components),
        "required_result_keys": required,
        "available_components": available_components,
        "missing_required_components": missing_required,
        "per_model_available_components": available_by_model,
        "per_model_missing_components": missing_by_model,
        "required_components_available": not missing_required,
    }


def leaderboard_field_availability(raw_results: dict[str, Any], model_name: str | None) -> dict[str, Any]:
    model_rows = choose_model_rows(raw_results, model_name)
    missing_by_model: dict[str, list[str]] = {}
    available_by_model: dict[str, list[str]] = {}
    for name, row in model_rows.items():
        missing: list[str] = []
        available: list[str] = []
        umt_fvd = scalar(row.get("UMT-FVD"))
        for key in LEADERBOARD_RESULT_KEYS:
            value = scalar(row.get(key))
            placeholder_umt = key in {"UMT-FVD", "UMTScore"} and umt_fvd == -1
            if value is None or placeholder_umt:
                missing.append(key)
            else:
                available.append(key)
        missing_by_model[name] = missing
        available_by_model[name] = available

    missing_required = sorted({key for missing in missing_by_model.values() for key in missing})
    return {
        "required_result_keys": list(LEADERBOARD_RESULT_KEYS),
        "available_result_keys": [key for key in LEADERBOARD_RESULT_KEYS if key not in missing_required],
        "missing_result_keys": missing_required,
        "per_model_available_result_keys": available_by_model,
        "per_model_missing_result_keys": missing_by_model,
        "required_fields_available": not missing_required,
    }


def per_model_rows(raw_results: dict[str, Any], model_name: str | None, upstream_results_path: Path) -> list[dict[str, Any]]:
    rows = []
    for name, row in choose_model_rows(raw_results, model_name).items():
        rows.append(
            {
                "model_name": name,
                "average_mtscore": scalar(row.get("Average_MTScore")),
                "average_chscore": scalar(row.get("Average_CHScore")),
                "average_gpt4o_mtscore": scalar(row.get("Average_GPT4o-MTScore")),
                "umt_fvd": scalar(row.get("UMT-FVD")),
                "umt_score": scalar(row.get("UMTScore")),
                "source": str(upstream_results_path.resolve()),
                "raw": row,
            }
        )
    return rows


def per_sample_rows_from_all_results(all_results_dir: Path | None) -> list[dict[str, Any]]:
    if all_results_dir is None or not all_results_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(all_results_dir.glob("*.json")):
        raw = load_json(path)
        if not isinstance(raw, dict):
            continue
        if path.name.endswith("_CHScore.json"):
            for item in raw.get("all_scores", []):
                if isinstance(item, dict):
                    for video_id, metrics in item.items():
                        rows.append({"metric_family": "CHScore", "video_id": video_id, "metrics": metrics, "source": str(path.resolve())})
        elif path.name.endswith("_MTScore.json"):
            for item in raw.get("video_scores", []):
                if isinstance(item, dict):
                    rows.append({"metric_family": "MTScore", "video_id": item.get("video_name"), "metrics": item, "source": str(path.resolve())})
        elif path.name.endswith("_GPT4o-MTScore.json"):
            formatted = raw.get("Formatted Data")
            if isinstance(formatted, dict):
                for video_id, metrics in formatted.items():
                    rows.append({"metric_family": "GPT4o-MTScore", "video_id": video_id, "metrics": metrics, "source": str(path.resolve())})
    return rows


def result_path_from_arg(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "ChronoMagic-Bench-Input.json"
        if candidate.is_file():
            return candidate
    return path


def normalize_chronomagic_results(
    raw_results: dict[str, Any],
    *,
    benchmark_id: str,
    output_dir: Path,
    upstream_results_path: Path,
    all_results_dir: Path | None,
    dataset_root: Path | None,
    generated_video_dir: Path | None,
    model_name: str | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None,
    stderr_path: Path | None,
    components: tuple[str, ...] = FULL_COMPONENTS,
    commands: list[list[str]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_metrics_path = output_dir / "per_sample_metrics.jsonl"
    generated_video_manifest_path = output_dir / "generated_video_manifest.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"

    extracted_scores = extract_scores(raw_results, model_name)
    component_availability = component_result_availability(raw_results, model_name, components)
    leaderboard_availability = leaderboard_field_availability(raw_results, model_name)
    sample_rows = per_model_rows(raw_results, model_name, upstream_results_path)
    sample_rows.extend(per_sample_rows_from_all_results(all_results_dir))
    dataset_records = discover_metadata_records(dataset_root)
    expected_stems = expected_stems_from_records(dataset_records, ("videoid", "video_id", "id"))
    generated_video_manifest = build_generated_video_manifest(
        generated_video_dir,
        expected_count=HF_DATASET_EXPECTED_ROWS,
        expected_stems=expected_stems,
    )
    dataset_manifest = build_local_dataset_manifest(
        dataset_root,
        dataset_id=HF_DATASET_ID,
        config=HF_DATASET_CONFIG,
        split=HF_DATASET_SPLIT,
        expected_rows=HF_DATASET_EXPECTED_ROWS,
    )
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in METRIC_ORDER:
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        row = {
            "metric_id": metric_id,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": item.get("normalized_score"),
            "raw_score_range": "official_mixed_scale",
            "source": item.get("source"),
            "sample_count": item.get("sample_count"),
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_chronomagic_results"
        else:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_metrics_path, sample_rows)
    write_json(generated_video_manifest_path, generated_video_manifest)
    write_json(dataset_manifest_path, dataset_manifest)
    available_count = sum(1 for row in metric_rows if row["available"])
    required_components_available = bool(component_availability["required_components_available"])
    normalization_ok = returncode == 0 and required_components_available
    normalizer_only = command is None
    official_verified = command is not None and normalization_ok
    is_component_run = component_mode(components)
    full_prompt_coverage = bool(generated_video_manifest["coverage_complete"])
    leaderboard_valid = (
        official_verified
        and not is_component_run
        and full_prompt_coverage
        and bool(leaderboard_availability["required_fields_available"])
    )
    missing_full_components = [component for component in FULL_COMPONENTS if component not in components]
    eligibility_reasons = [
        "official ChronoMagic result normalization or runtime validation; leaderboard validity requires full prompt set, CHScore, MTScore, GPT4o-MTScore, UMT-FVD, UMTScore, and official submission protocol",
    ]
    if is_component_run:
        eligibility_reasons.append(
            "bounded official component validation only; missing full official components: "
            + ", ".join(missing_full_components)
        )
    if not required_components_available:
        eligibility_reasons.append(
            "missing required ChronoMagic official component outputs: "
            + ", ".join(component_availability["missing_required_components"])
        )
    if not full_prompt_coverage:
        eligibility_reasons.append("generated videos do not cover the complete ChronoMagic prompt set")
    if not leaderboard_availability["required_fields_available"]:
        eligibility_reasons.append(
            "missing or placeholder ChronoMagic leaderboard fields: "
            + ", ".join(leaderboard_availability["missing_result_keys"])
        )
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": (
                "official_verified"
                if official_verified
                else "official_results_imported"
                if normalizer_only and normalization_ok
                else "failed"
            ),
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_chronomagic_official_runner",
            "command": command,
            "commands": commands,
            "components": list(components),
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "ChronoMagic-Bench",
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_model_weights": True,
            "requires_api": "gpt4o_mtscore" in components,
        },
        "dataset": {
            "hf_dataset_id": HF_DATASET_ID,
            "hf_config": HF_DATASET_CONFIG,
            "hf_split": HF_DATASET_SPLIT,
            "expected_rows": HF_DATASET_EXPECTED_ROWS,
            "sample_count": len(sample_rows),
            "local_dataset": dataset_manifest,
            "generated_videos": generated_video_manifest,
        },
        "eligibility": {
            "leaderboard_valid": leaderboard_valid,
            "reasons": eligibility_reasons,
            "full_prompt_coverage": full_prompt_coverage,
            "leaderboard_field_availability": leaderboard_availability,
        },
        "generation": {
            "successful": len(sample_rows),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "metamorphic": ["chronomagic_score", "temporal_transformation"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
            "component_availability": component_availability,
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "official_chronomagic",
            "upstream_results": str(upstream_results_path.resolve()),
            "all_results_dir": None if all_results_dir is None else str(all_results_dir.resolve()),
            "dataset_root": None if dataset_root is None else str(dataset_root.resolve()),
            "generated_video_dir": None if generated_video_dir is None else str(generated_video_dir.resolve()),
            "components": list(components),
            "component_run": is_component_run,
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
            "required_components_available": required_components_available,
            "missing_required_components": component_availability["missing_required_components"],
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_results_imported": normalizer_only and normalization_ok,
            "required_result_keys": component_availability["required_result_keys"],
            "missing_required_components": component_availability["missing_required_components"],
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_metrics": str(per_sample_metrics_path.resolve()),
            "generated_video_manifest": str(generated_video_manifest_path.resolve()),
            "dataset_manifest": str(dataset_manifest_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "official_benchmark_verified": official_verified,
        "official_component_verified": is_component_run and official_verified,
        "integration_evidence": official_verified,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def build_official_command(args: argparse.Namespace, result_root: Path) -> list[str]:
    if args.input_folder is None:
        raise ValueError("--input-folder or --official-results-path is required")
    if not args.openai_api:
        raise ValueError("--openai-api or OPENAI_API_KEY is required for official ChronoMagic runtime execution")
    command = [
        args.python,
        str(args.chronomagic_root / "evaluate.py"),
        "--eval_type",
        args.eval_type,
        "--model_names",
        args.model_name or "worldfoundry",
        "--input_folder",
        str(args.input_folder),
        "--output_folder",
        str(result_root),
        "--video_frames_folder",
        args.video_frames_folder,
        "--model_pth_CHScore",
        str(args.model_pth_chscore),
        "--model_pth_MTScore",
        str(args.model_pth_mtscore),
        "--num_workers",
        str(args.num_workers),
        "--openai_api",
        args.openai_api,
    ]
    if args.api_base_url:
        command.extend(["--api_base_url", args.api_base_url])
    return command


def generated_video_dir_for_run(args: argparse.Namespace) -> Path | None:
    if args.generated_video_dir is not None:
        return args.generated_video_dir
    if args.input_folder is None:
        return None
    if args.model_name:
        return args.input_folder / args.model_name
    return args.input_folder


def model_names(args: argparse.Namespace) -> list[str]:
    return [args.model_name or "worldfoundry"]


def list_video_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)


def list_direct_video_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS)


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


def stage_generated_video_input(args: argparse.Namespace, output_dir: Path) -> None:
    """Stage arbitrary generated videos into ChronoMagic's input_folder/model_name layout."""
    if args.input_folder is not None or args.generated_video_dir is None:
        return
    source = args.generated_video_dir.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"ChronoMagic generated video directory not found: {source}")
    args.model_name = args.model_name or "worldfoundry"
    staged_root = output_dir / "staged_input"
    model_dir = staged_root / args.model_name
    rows = []

    if args.eval_type == "open":
        part_root = source / args.model_name if (source / args.model_name).is_dir() else source
        videos_by_part = {part: list_direct_video_files(part_root / part) for part in OPEN_EVAL_PARTS}
        missing_parts = [part for part, videos in videos_by_part.items() if not videos]
        if missing_parts:
            raise ValueError(
                "ChronoMagic open eval generated-video staging requires direct video files under "
                f"{part_root}/{{1,2,3}}; missing or empty parts: {', '.join(missing_parts)}"
            )
        for part, videos in videos_by_part.items():
            part_dir = model_dir / part
            part_dir.mkdir(parents=True, exist_ok=True)
            for index, video in enumerate(videos):
                target = part_dir / video.name
                if target.exists() and target.resolve() != video.resolve():
                    target = part_dir / f"{video.stem}-{index}{video.suffix.lower()}"
                rows.append({"part": part, "source": str(video), "target": str(target), "method": link_or_copy(video, target)})
    else:
        videos = list_video_files(source)
        if not videos:
            raise FileNotFoundError(f"ChronoMagic generated video directory contains no videos: {source}")
        model_dir.mkdir(parents=True, exist_ok=True)
        for index, video in enumerate(videos):
            target = model_dir / video.name
            if target.exists() and target.resolve() != video.resolve():
                target = model_dir / f"{video.stem}-{index}{video.suffix.lower()}"
            rows.append({"source": str(video), "target": str(target), "method": link_or_copy(video, target)})
    write_json(
        output_dir / "staged_input_manifest.json",
        {
            "schema_version": "worldfoundry-chronomagic-staged-input-v1",
            "source": str(source),
            "input_folder": str(staged_root),
            "model_name": args.model_name,
            "video_count": len(rows),
            "videos": rows,
        },
    )
    args.input_folder = staged_root


def validate_chronomagic_root(chronomagic_root: Path, components: tuple[str, ...]) -> None:
    required = [Path("evaluate.py"), Path("get_uploaded_json.py")]
    for component in components:
        required.extend(COMPONENT_SCRIPT_PATHS[component])
    missing = [path.as_posix() for path in required if not (chronomagic_root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"ChronoMagic official repo root is missing required scripts under {chronomagic_root}: "
            + ", ".join(missing)
        )


def validate_input_layout(args: argparse.Namespace) -> None:
    if args.input_folder is None:
        raise ValueError("--input-folder or --generated-video-dir is required for ChronoMagic component execution")
    input_folder = args.input_folder
    if not input_folder.is_dir():
        raise FileNotFoundError(f"ChronoMagic input folder not found: {input_folder}")

    missing: list[str] = []
    for model_name in model_names(args):
        model_dir = input_folder / model_name
        if args.eval_type == "open":
            for part in OPEN_EVAL_PARTS:
                part_dir = model_dir / part
                if not part_dir.is_dir():
                    missing.append(f"{model_name}/{part}/")
                elif not list_direct_video_files(part_dir):
                    missing.append(f"{model_name}/{part}/*.mp4")
        else:
            if not model_dir.is_dir():
                missing.append(f"{model_name}/")
            elif not list_direct_video_files(model_dir):
                missing.append(f"{model_name}/*.mp4")
    if missing:
        layout = "input_folder/model_name/{1,2,3}/*.mp4" if args.eval_type == "open" else "input_folder/model_name/*.mp4"
        raise FileNotFoundError(
            f"ChronoMagic {args.eval_type} eval input layout must match {layout}; missing: "
            + ", ".join(missing)
        )


def build_component_commands(args: argparse.Namespace, result_root: Path, components: tuple[str, ...]) -> list[list[str]]:
    if args.input_folder is None:
        raise ValueError("--input-folder is required for ChronoMagic component execution")

    commands: list[list[str]] = []
    output_all = result_root / "all"
    names = model_names(args)
    if "chscore" in components:
        commands.append(
            [
                args.python,
                str(args.chronomagic_root / "CHScore" / "step0-get_CHScore.py"),
                "--model_names",
                *names,
                "--input_folder",
                str(args.input_folder),
                "--output_folder",
                str(output_all),
                "--model_pth",
                str(args.model_pth_chscore),
                "--eval_type",
                args.eval_type,
            ]
        )
    if "gpt4o_mtscore" in components:
        if not args.openai_api:
            raise ValueError("--openai-api or OPENAI_API_KEY is required for GPT4o-MTScore component execution")
        frames_dir = result_root / "GPT4o-MTScores_temp" / args.video_frames_folder
        scores_dir = result_root / "GPT4o-MTScores_temp" / "scores_temp"
        gpt4o_score_command = [
            args.python,
            str(args.chronomagic_root / "GPT4o_MTScore" / "step1-get_temp_results.py"),
            "--num_workers",
            str(args.num_workers),
            "--openai_api",
            args.openai_api,
            "--input_dir",
            str(frames_dir),
            "--output_dir",
            str(scores_dir),
            "--model_names",
            *names,
            "--eval_type",
            args.eval_type,
        ]
        if args.api_base_url:
            gpt4o_score_command.extend(["--base_url", args.api_base_url])
        commands.extend(
            [
                [
                    args.python,
                    str(args.chronomagic_root / "GPT4o_MTScore" / "step0-extract_video_frames.py"),
                    "--input_dir",
                    str(args.input_folder),
                    "--output_dir",
                    str(frames_dir),
                    "--model_names",
                    *names,
                    "--eval_type",
                    args.eval_type,
                ],
                gpt4o_score_command,
                [
                    args.python,
                    str(args.chronomagic_root / "GPT4o_MTScore" / "step2-get_GPT4o-MTScore.py"),
                    "--input_dir",
                    str(scores_dir),
                    "--output_dir",
                    str(output_all),
                    "--model_names",
                    *names,
                    "--eval_type",
                    args.eval_type,
                ],
            ]
        )
    if "mtscore" in components:
        commands.append(
            [
                args.python,
                str(args.chronomagic_root / "MTScore" / "step0-get_MTScore.py"),
                "--model_names",
                *names,
                "--input_folder",
                str(args.input_folder),
                "--output_folder",
                str(output_all),
                "--model_pth",
                str(args.model_pth_mtscore),
                "--eval_type",
                args.eval_type,
            ]
        )
    commands.append(
        [
            args.python,
            str(args.chronomagic_root / "get_uploaded_json.py"),
            "--input_path",
            str(output_all),
            "--output_path",
            str(result_root),
        ]
    )
    return commands


def run_component_commands(
    args: argparse.Namespace,
    commands: list[list[str]],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    start = time.monotonic()
    for command in commands:
        header = "$ " + shlex.join(command) + "\n"
        completed = subprocess.run(
            command,
            cwd=args.chronomagic_root,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        stdout_parts.append(header + completed.stdout)
        stderr_parts.append(header + completed.stderr)
        if completed.returncode != 0:
            stdout_path.write_text("\n".join(stdout_parts), encoding="utf-8")
            stderr_path.write_text("\n".join(stderr_parts), encoding="utf-8")
            return completed.returncode, time.monotonic() - start

    stdout_path.write_text("\n".join(stdout_parts), encoding="utf-8")
    stderr_path.write_text("\n".join(stderr_parts), encoding="utf-8")
    return 0, time.monotonic() - start


def run_chronomagic(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    components = parse_components(args.components)
    stage_generated_video_input(args, output_dir)

    if args.from_upstream_results:
        upstream_results_path = result_path_from_arg(args.from_upstream_results)
        raw_results = load_json(upstream_results_path)
        if not isinstance(raw_results, dict):
            raise ValueError(f"ChronoMagic result JSON must be an object: {upstream_results_path}")
        all_results_dir = args.all_results_dir
        if all_results_dir is None and upstream_results_path.parent.joinpath("all").is_dir():
            all_results_dir = upstream_results_path.parent / "all"
        return normalize_chronomagic_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=upstream_results_path,
            all_results_dir=all_results_dir,
            dataset_root=args.dataset_root,
            generated_video_dir=generated_video_dir_for_run(args),
            model_name=args.model_name,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            components=components,
        )

    validate_chronomagic_root(args.chronomagic_root, components)
    validate_input_layout(args)
    upstream_output_dir = output_dir / "upstream"
    commands = build_component_commands(args, upstream_output_dir, components)
    command = ["bash", "-lc", " && ".join(shlex.join(item) for item in commands)]
    returncode, duration_seconds = run_component_commands(args, commands, stdout_path, stderr_path)
    upstream_results_path = upstream_output_dir / "ChronoMagic-Bench-Input.json"
    if upstream_results_path.is_file():
        raw_results = load_json(upstream_results_path)
    else:
        raw_results = {
            model_names(args)[0]: {
                "error": "missing upstream ChronoMagic-Bench-Input.json",
                "returncode": returncode,
            }
        }
        write_json(upstream_results_path, raw_results)
        raw_results = load_json(upstream_results_path)
    if not isinstance(raw_results, dict):
        raise ValueError(f"ChronoMagic result JSON must be an object: {upstream_results_path}")
    return normalize_chronomagic_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        upstream_results_path=upstream_results_path,
        all_results_dir=upstream_output_dir / "all",
        dataset_root=args.dataset_root,
        generated_video_dir=generated_video_dir_for_run(args),
        model_name=args.model_name,
        command=command,
        duration_seconds=duration_seconds,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        components=components,
        commands=commands,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official ChronoMagic-Bench or normalize official results to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "chronomagic-bench"))
    parser.add_argument("--chronomagic-root", type=Path, default=env_path("WORLDFOUNDRY_CHRONOMAGIC_ROOT", DEFAULT_CHRONOMAGIC_ROOT))
    parser.add_argument(
        "--official-results-path",
        dest="from_upstream_results",
        type=Path,
        default=env_path("WORLDFOUNDRY_CHRONOMAGIC_RESULTS_PATH"),
    )
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--all-results-dir", type=Path, default=env_path("WORLDFOUNDRY_CHRONOMAGIC_ALL_RESULTS_DIR"))
    parser.add_argument("--model-name", default=os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_MODEL_NAME"))
    parser.add_argument("--input-folder", type=Path, default=env_path("WORLDFOUNDRY_CHRONOMAGIC_INPUT_FOLDER"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_CHRONOMAGIC_DATASET_ROOT"),
        help="Local BestWishYsh/ChronoMagic-Bench dataset root for discovery evidence.",
    )
    parser.add_argument(
        "--generated-video-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
        help="Generated ChronoMagic video directory; defaults to WORLDFOUNDRY_GENERATED_ARTIFACT_DIR.",
    )
    parser.add_argument("--eval-type", choices=("open", "close"), default=os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_EVAL_TYPE", "close"))
    parser.add_argument(
        "--components",
        default=os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_COMPONENTS", "all"),
        help="Comma-separated official components to run: all, chscore, mtscore, gpt4o_mtscore.",
    )
    parser.add_argument("--model-pth-chscore", type=Path, default=env_path("WORLDFOUNDRY_CHRONOMAGIC_CHSCORE_CKPT", Path("cotracker2.pth")))
    parser.add_argument("--model-pth-mtscore", type=Path, default=env_path("WORLDFOUNDRY_CHRONOMAGIC_MTSCORE_CKPT", Path("InternVideo2-stage2_1b-224p-f4.pt")))
    parser.add_argument("--video-frames-folder", default=os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_VIDEO_FRAMES_FOLDER", "video_frames_folder_temp"))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_NUM_WORKERS", "8")))
    parser.add_argument("--openai-api", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_OPENAI_API"))
    parser.add_argument("--api-base-url", default=os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_API_BASE_URL"))
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_CHRONOMAGIC_TIMEOUT", "7200")))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_chronomagic(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_metrics": scorecard["artifacts"]["per_sample_metrics"],
        "generated_video_manifest": scorecard["artifacts"]["generated_video_manifest"],
        "dataset_manifest": scorecard["artifacts"]["dataset_manifest"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "official_component_verified": scorecard["official_component_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
        "components": scorecard["run"]["components"],
        "component_run": scorecard["evaluation"]["component_run"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official ChronoMagic-Bench validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
