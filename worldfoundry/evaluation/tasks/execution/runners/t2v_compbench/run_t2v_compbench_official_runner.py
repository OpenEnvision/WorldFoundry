#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.benchmark_data import (
    VIDEO_EXTENSIONS,
    build_generated_video_manifest,
)
from worldfoundry.evaluation.tasks.execution.framework.io import (
    env_path,
    load_json,
    normalize_unit_score,
    scalar_number,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.execution.runners.t2v_compbench.t2v_compbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    GENERATION_MANIFEST_NAME,
    PROMPTS_PER_CATEGORY,
    materialize_t2v_compbench_official_layout,
)
from worldfoundry.evaluation.utils import REPO_ROOT

DEFAULT_T2V_COMPBENCH_ROOT = (
    REPO_ROOT
    / "worldfoundry"
    / "evaluation"
    / "tasks"
    / "execution"
    / "runners"
    / "t2v_compbench"
    / "runtime"
    / "t2v_compbench"
)
DEFAULT_T2V_COMPBENCH_ASSETS = (
    REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "assets" / "t2v-compbench"
)
SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
HF_DATASET_ID = "Kaiyue/T2V-CompBench-Videos"
HF_DATASET_CONFIG = "default"
HF_DATASET_SPLIT = "train"
HF_DATASET_EXPECTED_ROWS = 25200
EXPECTED_PROMPT_COUNT = CANONICAL_PROMPT_COUNT
COMPONENT_METRICS = (
    "consistent_attribute_binding",
    "dynamic_attribute_binding",
    "spatial_relationships",
    "motion_binding",
    "action_binding",
    "object_interactions",
    "generative_numeracy",
)
METRIC_ORDER = (*COMPONENT_METRICS, "t2v_compbench_average")

CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "consistent_attribute_binding": {
        "display_name": "Consistent Attribute Binding",
        "metric_type": "mllm",
        "script": "mllm_metrics/compbench_eval_consistent_attr.py",
        "cwd": ".",
        "prompt_file": "meta_data/consistent_attribute_binding.json",
        "video_subdir": "consistent_attr",
        "csv_subdir": "csv_consistent_attr",
        "csv_suffix": "_consistent_attr_score.csv",
    },
    "dynamic_attribute_binding": {
        "display_name": "Dynamic Attribute Binding",
        "metric_type": "mllm",
        "script": "mllm_metrics/compbench_eval_dynamic_attr.py",
        "cwd": ".",
        "prompt_file": "meta_data/dynamic_attribute_binding.json",
        "video_subdir": "dynamic_attr",
        "csv_subdir": "csv_dynamic_attr",
        "csv_suffix": "_dynamic_attr_score.csv",
    },
    "spatial_relationships": {
        "display_name": "Spatial Relationships",
        "metric_type": "detection",
        "script": "grounded_sam_metrics/compbench_eval_spatial_relationships.py",
        "cwd": ".",
        "prompt_file": "meta_data/spatial_relationships.json",
        "video_subdir": "spatial_relationships",
        "csv_subdir": "csv_spatial",
        "csv_suffix": "_spatial_score.csv",
    },
    "motion_binding": {
        "display_name": "Motion Binding",
        "metric_type": "tracking",
        "script": "dot/compbench_eval_motion_binding.py",
        "cwd": "dot",
        "prompt_file": "meta_data/motion_binding.json",
        "video_subdir": "motion_binding",
        "csv_subdir": "csv_motion_binding",
        "csv_suffix": "_motion_score.csv",
    },
    "action_binding": {
        "display_name": "Action Binding",
        "metric_type": "mllm",
        "script": "mllm_metrics/compbench_eval_action_binding.py",
        "cwd": ".",
        "prompt_file": "meta_data/action_binding.json",
        "video_subdir": "action_binding",
        "csv_subdir": "csv_action_binding",
        "csv_suffix": "_action_binding_score.csv",
    },
    "object_interactions": {
        "display_name": "Object Interactions",
        "metric_type": "mllm",
        "script": "mllm_metrics/compbench_eval_interaction.py",
        "cwd": ".",
        "prompt_file": "meta_data/object_interactions.json",
        "video_subdir": "interaction",
        "csv_subdir": "csv_object_interactions",
        "csv_suffix": "_object_interactions_score.csv",
    },
    "generative_numeracy": {
        "display_name": "Generative Numeracy",
        "metric_type": "detection",
        "script": "grounded_sam_metrics/compbench_eval_numeracy.py",
        "cwd": ".",
        "prompt_file": "meta_data/generative_numeracy.json",
        "video_subdir": "generative_numeracy",
        "csv_subdir": "csv_numeracy",
        "csv_suffix": "_numeracy_video.csv",
    },
}

METRIC_ALIASES = {
    "consistent_attr": "consistent_attribute_binding",
    "consistent_attribute": "consistent_attribute_binding",
    "consistent_attribute_binding": "consistent_attribute_binding",
    "dynamic_attr": "dynamic_attribute_binding",
    "dynamic_attribute": "dynamic_attribute_binding",
    "dynamic_attribute_binding": "dynamic_attribute_binding",
    "spatial": "spatial_relationships",
    "spatial_relationship": "spatial_relationships",
    "spatial_relationships": "spatial_relationships",
    "motion": "motion_binding",
    "motion_binding": "motion_binding",
    "action": "action_binding",
    "action_binding": "action_binding",
    "interaction": "object_interactions",
    "interactions": "object_interactions",
    "object_interaction": "object_interactions",
    "object_interactions": "object_interactions",
    "numeracy": "generative_numeracy",
    "generative_numeracy": "generative_numeracy",
    "t2v_compbench": "t2v_compbench_average",
    "t2v_compbench_average": "t2v_compbench_average",
    "average": "t2v_compbench_average",
    "overall": "t2v_compbench_average",
}

HF_LABEL_TO_METRIC = {
    "consistent_attr_1": "consistent_attribute_binding",
    "dynamic_attr_2": "dynamic_attribute_binding",
    "spatial_3": "spatial_relationships",
    "motion_4": "motion_binding",
    "action_5": "action_binding",
    "interaction_6": "object_interactions",
    "numeracy_7": "generative_numeracy",
}

CATEGORY_VIDEO_SUBDIR_ALIASES: dict[str, tuple[str, ...]] = {
    "consistent_attribute_binding": ("consistent_attr", "consistent_attr_1"),
    "dynamic_attribute_binding": ("dynamic_attr", "dynamic_attr_2"),
    "spatial_relationships": ("spatial_relationships", "spatial_3"),
    "motion_binding": ("motion_binding", "motion_4"),
    "action_binding": ("action_binding", "action_5"),
    "object_interactions": ("interaction", "object_interactions", "interaction_6"),
    "generative_numeracy": ("generative_numeracy", "numeracy", "numeracy_7"),
}

CATEGORY_RUNTIME_GROUPS: dict[str, tuple[str, ...]] = {
    "consistent_attribute_binding": ("mllm",),
    "dynamic_attribute_binding": ("mllm",),
    "action_binding": ("mllm",),
    "object_interactions": ("mllm",),
    "spatial_relationships": ("detection",),
    "generative_numeracy": ("detection",),
    "motion_binding": ("detection", "tracking"),
}

PREFLIGHT_IMPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "mllm": (
        "cv2",
        "torch",
        "torchvision",
        "PIL",
        "worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.constants",
        "worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.model.builder",
    ),
    "detection": (
        "cv2",
        "torch",
        "torchvision",
        "PIL",
        "matplotlib",
        "tqdm",
        "worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v1.dpt",
        "worldfoundry.base_models.perception_core.detection.grounding_dino.models",
        "worldfoundry.base_models.perception_core.segment.sam_v1",
    ),
    "tracking": (
        "torch",
        "scipy",
        "matplotlib",
        "tqdm",
        "worldfoundry.base_models.perception_core.tracking.dot.models",
        "worldfoundry.base_models.perception_core.tracking.dot.utils.io",
    ),
}


scalar = partial(
    scalar_number,
    dict_keys=("score", "raw_score", "value", "mean", "average", "avg", "overall"),
)


def grounding_dino_checkpoint_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_GROUNDING_DINO_CKPT")
    if explicit:
        return Path(explicit).expanduser()
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        for candidate in (
            root / "hfd" / "ShilongLiu--GroundingDINO" / "groundingdino_swint_ogc.pth",
            root / "GroundingDINO" / "groundingdino_swint_ogc.pth",
            root / "evalcrafter" / "GroundingDINO" / "groundingdino_swint_ogc.pth",
        ):
            if candidate.is_file():
                return candidate
    evalcrafter_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if evalcrafter_dir:
        candidate = Path(evalcrafter_dir).expanduser() / "GroundingDINO" / "groundingdino_swint_ogc.pth"
        if candidate.is_file():
            return candidate
    return Path("checkpoints") / "ckpt" / "groundingdino_swint_ogc.pth"


def sam_v1_checkpoint_path(model_type: str = "vit_b") -> Path:
    if model_type == "vit_b":
        explicit = os.environ.get("WORLDFOUNDRY_SAM_VIT_B_CKPT")
        filename = "sam_vit_b_01ec64.pth"
    elif model_type == "vit_h":
        explicit = os.environ.get("WORLDFOUNDRY_SAM_VIT_H_CKPT")
        filename = "sam_vit_h_4b8939.pth"
    else:
        raise ValueError(f"Unsupported SAM v1 model_type: {model_type}")
    if explicit:
        return Path(explicit).expanduser()
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        for subdir in ("", "evalcrafter", "sam"):
            candidate = (root / subdir / filename) if subdir else root / filename
            if candidate.is_file():
                return candidate
    evalcrafter_dir = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    if evalcrafter_dir:
        candidate = Path(evalcrafter_dir).expanduser() / "SAM" / filename
        if candidate.is_file():
            return candidate
    return Path("checkpoints") / "ckpt" / filename


def canonical_metric_id(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return METRIC_ALIASES.get(normalized)


normalize_score = normalize_unit_score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_metric_from_path(path: Path) -> str | None:
    name = path.name.lower()
    for metric_id, spec in CATEGORY_SPECS.items():
        if name.endswith(str(spec["csv_suffix"]).lower()):
            return metric_id
    stem_parts = path.stem.lower().replace("-", "_").split("_")
    for width in range(len(stem_parts), 0, -1):
        candidate = "_".join(stem_parts[-width:])
        metric_id = canonical_metric_id(candidate)
        if metric_id in COMPONENT_METRICS:
            return metric_id
    return None


def detect_generated_category(path: Path) -> str | None:
    """Infer the T2V-CompBench category from a generated video path.

    Args:
        path: Generated video path under a model/category layout.
    """
    parts = {part.lower() for part in path.parts}
    for metric_id, spec in CATEGORY_SPECS.items():
        if str(spec["video_subdir"]).lower() in parts:
            return metric_id
    for label, metric_id in HF_LABEL_TO_METRIC.items():
        if label.lower() in parts:
            return metric_id
    return None


def score_row_index(rows: list[list[str]]) -> tuple[int | None, float | None]:
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        if not row:
            continue
        label = str(row[0]).strip().lower()
        if label.startswith("score"):
            for value in row[1:]:
                number = scalar(value)
                if number is not None:
                    return index, number
    return None, None


def row_to_mapping(header: list[str], row: list[str]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for index, key in enumerate(header):
        if key:
            mapping[key] = row[index] if index < len(row) else ""
    return mapping


def parse_per_sample_rows(metric_id: str, path: Path, rows: list[list[str]], final_index: int | None) -> list[dict[str, Any]]:
    if not rows:
        return []
    header = [item.strip() for item in rows[0]]
    score_column = next((key for key in header if key.lower() == "score"), None)
    if score_column is None:
        score_column = next((key for key in header if key.lower().endswith("score")), None)
    body_end = final_index if final_index is not None else len(rows)
    sample_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows[1:body_end], start=1):
        if not any(cell.strip() for cell in row):
            continue
        mapping = row_to_mapping(header, row)
        sample_score = scalar(mapping.get(score_column)) if score_column else scalar(row[-1] if row else None)
        sample_rows.append(
            {
                "metric_id": metric_id,
                "sample_index": row_index - 1,
                "sample_id": mapping.get("name") or mapping.get("video_name") or mapping.get("id") or row[0],
                "prompt": mapping.get("prompt"),
                "raw_score": sample_score,
                "normalized_score": normalize_score(sample_score),
                "source_csv": str(path.resolve()),
                "raw": mapping,
            }
        )
    return sample_rows


def parse_official_csv(path: Path) -> dict[str, Any] | None:
    metric_id = detect_metric_from_path(path)
    if metric_id is None:
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    final_index, raw_score = score_row_index(rows)
    sample_rows = parse_per_sample_rows(metric_id, path, rows, final_index)
    return {
        "metric_id": metric_id,
        "raw_score": raw_score,
        "sample_count": len(sample_rows),
        "source": "official_csv_final_score",
        "source_csv": path,
        "per_sample_rows": sample_rows,
        "csv_sha256": sha256_file(path),
        "csv_bytes": path.stat().st_size,
    }


def collect_csv_results(csv_dir: Path, model_name: str | None) -> list[dict[str, Any]]:
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"T2V-CompBench CSV directory not found: {csv_dir}")
    parsed: list[dict[str, Any]] = []
    for path in sorted(csv_dir.rglob("*.csv")):
        item = parse_official_csv(path)
        if item is None:
            continue
        if model_name and not path.name.startswith(f"{model_name}_"):
            item["model_name_mismatch"] = True
        parsed.append(item)

    by_metric: dict[str, list[dict[str, Any]]] = {}
    for item in parsed:
        by_metric.setdefault(item["metric_id"], []).append(item)

    selected: list[dict[str, Any]] = []
    for metric_id in COMPONENT_METRICS:
        candidates = by_metric.get(metric_id, [])
        if model_name:
            preferred = [item for item in candidates if not item.get("model_name_mismatch")]
            if preferred:
                candidates = preferred
        candidates = [item for item in candidates if item.get("raw_score") is not None]
        if candidates:
            selected.append(candidates[0])
    return selected


def collect_json_results(raw_results: Any) -> list[dict[str, Any]]:
    payload = raw_results
    if isinstance(raw_results, dict):
        for key in ("scores", "metrics", "leaderboard"):
            if isinstance(raw_results.get(key), dict):
                payload = raw_results[key]
                break
    if not isinstance(payload, dict):
        return []

    results: list[dict[str, Any]] = []
    for raw_key, raw_value in payload.items():
        metric_id = canonical_metric_id(str(raw_key))
        if metric_id not in METRIC_ORDER:
            continue
        raw_score = scalar(raw_value)
        results.append(
            {
                "metric_id": metric_id,
                "raw_score": raw_score,
                "sample_count": None,
                "source": str(raw_key),
                "source_csv": None,
                "per_sample_rows": [],
                "csv_sha256": None,
                "csv_bytes": None,
            }
        )
    return results


def build_metric_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    by_metric = {item["metric_id"]: item for item in results if item.get("metric_id") in METRIC_ORDER}
    component_scores = [
        item["raw_score"] for metric_id, item in by_metric.items() if metric_id in COMPONENT_METRICS and item["raw_score"] is not None
    ]
    if len(component_scores) == len(COMPONENT_METRICS) and "t2v_compbench_average" not in by_metric:
        by_metric["t2v_compbench_average"] = {
            "metric_id": "t2v_compbench_average",
            "raw_score": sum(component_scores) / len(component_scores),
            "sample_count": min((item.get("sample_count") or 0 for item in by_metric.values()), default=0),
            "source": "computed_from_all_official_t2v_compbench_categories",
            "source_csv": None,
        }

    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in METRIC_ORDER:
        item = by_metric.get(metric_id, {})
        raw_score = item.get("raw_score")
        row = {
            "metric_id": metric_id,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalize_score(raw_score),
            "raw_score_range": [0.0, 1.0],
            "source": item.get("source"),
            "sample_count": item.get("sample_count"),
            "source_csv": None if item.get("source_csv") is None else str(Path(item["source_csv"]).resolve()),
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_t2v_compbench_outputs"
        else:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row
    return metric_rows, per_metric, leaderboard


def build_csv_manifest(results: list[dict[str, Any]], source_dir: Path | None) -> dict[str, Any]:
    files = []
    for item in results:
        source_csv = item.get("source_csv")
        if source_csv is None:
            continue
        path = Path(source_csv)
        files.append(
            {
                "metric_id": item["metric_id"],
                "path": str(path.resolve()),
                "sha256": item.get("csv_sha256"),
                "bytes": item.get("csv_bytes"),
                "sample_count": item.get("sample_count"),
            }
        )
    found_metrics = {item["metric_id"] for item in results}
    return {
        "source_dir": None if source_dir is None else str(source_dir.resolve()),
        "files": files,
        "found_metrics": sorted(found_metrics),
        "missing_metrics": [metric for metric in COMPONENT_METRICS if metric not in found_metrics],
    }


def build_t2v_dataset_manifest(root: Path | None) -> dict[str, Any]:
    """Describe the HF T2V-CompBench video mirror without recursively scanning all videos."""
    exists = bool(root is not None and root.exists())
    direct_children = sorted(root.iterdir()) if root is not None and root.is_dir() else []
    direct_files = [path for path in direct_children if path.is_file()]
    direct_dirs = [path for path in direct_children if path.is_dir()]
    zip_files = [path for path in direct_files if path.suffix.lower() == ".zip"]
    media_dirs = [path for path in direct_dirs if not path.name.startswith(".")]
    return {
        "hf_dataset_id": HF_DATASET_ID,
        "config": HF_DATASET_CONFIG,
        "split": HF_DATASET_SPLIT,
        "expected_rows": HF_DATASET_EXPECTED_ROWS,
        "root": None if root is None else str(root),
        "exists": exists,
        "file_count": len(direct_files),
        "direct_file_count": len(direct_files),
        "direct_dir_count": len(direct_dirs),
        "archive_count": len(zip_files),
        "model_video_dir_count": len(media_dirs),
        "sample_archives": [path.name for path in zip_files[:20]],
        "sample_model_video_dirs": [path.name for path in media_dirs[:20]],
    }


def normalize_t2v_compbench_results(
    results: list[dict[str, Any]],
    *,
    benchmark_id: str,
    output_dir: Path,
    source_dir: Path | None,
    dataset_root: Path | None,
    video_root: Path | None,
    upstream_results_path: Path | None,
    command: list[list[str]] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None,
    stderr_path: Path | None,
    requested_categories: list[str] | None = None,
    generation_bridge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_scores_path = output_dir / "per_sample_scores.jsonl"
    csv_manifest_path = output_dir / "leaderboard_csv_manifest.json"
    generated_video_manifest_path = output_dir / "generated_video_manifest.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"

    metric_rows, per_metric, leaderboard = build_metric_rows(results)
    per_sample_rows = [row for item in results for row in item.get("per_sample_rows", [])]
    csv_manifest = build_csv_manifest(results, source_dir)
    generated_video_manifest = build_generated_video_manifest(
        video_root,
        expected_count=EXPECTED_PROMPT_COUNT,
        category_from_path=detect_generated_category,
    )
    dataset_manifest = build_t2v_dataset_manifest(dataset_root)

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_scores_path, per_sample_rows)
    write_json(csv_manifest_path, csv_manifest)
    write_json(generated_video_manifest_path, generated_video_manifest)
    write_json(dataset_manifest_path, dataset_manifest)

    available_count = sum(1 for row in metric_rows if row["available"])
    complete_category_count = sum(1 for metric in COMPONENT_METRICS if metric in leaderboard)
    normalization_ok = returncode == 0 and available_count > 0
    normalizer_only = command is None
    result_by_metric = {str(item.get("metric_id")): item for item in results}
    requested = list(
        dict.fromkeys(
            requested_categories
            or [metric for metric in COMPONENT_METRICS if metric in result_by_metric]
        )
    )
    runtime_results_complete = bool(requested) and all(
        result_by_metric.get(metric, {}).get("raw_score") is not None
        and int(result_by_metric.get(metric, {}).get("sample_count") or 0) > 0
        for metric in requested
    )
    official_component_verified = (
        command is not None
        and returncode == 0
        and runtime_results_complete
    )
    full_metric_outputs = set(requested) == set(COMPONENT_METRICS) and all(
        result_by_metric.get(metric, {}).get("raw_score") is not None
        and int(result_by_metric.get(metric, {}).get("sample_count") or 0) >= PROMPTS_PER_CATEGORY
        for metric in COMPONENT_METRICS
    )
    full_video_coverage = (
        generated_video_manifest["file_count"] == EXPECTED_PROMPT_COUNT
        and all(
            generated_video_manifest["by_category"].get(metric, 0) == PROMPTS_PER_CATEGORY
            for metric in COMPONENT_METRICS
        )
    )
    official_benchmark_verified = (
        official_component_verified and full_metric_outputs and full_video_coverage
    )
    if official_benchmark_verified:
        eligibility_reasons: list[str] = []
    elif normalizer_only:
        eligibility_reasons = [
            "existing result normalization does not prove that WorldFoundry executed the official evaluator"
        ]
    else:
        eligibility_reasons = [
            "bounded official component evidence is not a full T2V-CompBench leaderboard run; "
            "all seven categories require 200 scored videos each"
        ]
    run_status = (
        "official_benchmark_verified"
        if official_benchmark_verified
        else "official_component_verified"
        if official_component_verified
        else "official_results_imported"
        if normalization_ok
        else "failed"
    )
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_t2v_compbench_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "T2V-CompBench",
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_model_weights": True,
        },
        "dataset": {
            "sample_count": len(per_sample_rows),
            "hf_dataset_id": HF_DATASET_ID,
            "hf_config": HF_DATASET_CONFIG,
            "hf_split": HF_DATASET_SPLIT,
            "expected_prompt_count": EXPECTED_PROMPT_COUNT,
            "expected_dataset_rows": HF_DATASET_EXPECTED_ROWS,
            "category_count": complete_category_count,
            "local_dataset": dataset_manifest,
            "generated_videos": generated_video_manifest,
        },
        "eligibility": {
            "leaderboard_valid": official_benchmark_verified,
            "reasons": eligibility_reasons,
        },
        "generation": {
            "successful": len(per_sample_rows),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "mllm": [
                    "consistent_attribute_binding",
                    "dynamic_attribute_binding",
                    "action_binding",
                    "object_interactions",
                ],
                "detection": ["spatial_relationships", "generative_numeracy"],
                "tracking": ["motion_binding"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(per_sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
                "complete_category_count": complete_category_count,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "official_t2v_compbench",
            "source_dir": None if source_dir is None else str(source_dir.resolve()),
            "dataset_root": None if dataset_root is None else str(dataset_root.resolve()),
            "generated_video_root": None if video_root is None else str(video_root.resolve()),
            "generation_bridge": generation_bridge,
            "scope": "full" if official_benchmark_verified else "bounded",
            "requested_categories": requested,
            "upstream_results": None if upstream_results_path is None else str(upstream_results_path.resolve()),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_component_verified": official_component_verified,
            "full_metric_outputs": full_metric_outputs,
            "full_video_coverage": full_video_coverage,
            "official_results_imported": normalizer_only and normalization_ok,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_scores": str(per_sample_scores_path.resolve()),
            "leaderboard_csv_manifest": str(csv_manifest_path.resolve()),
            "generated_video_manifest": str(generated_video_manifest_path.resolve()),
            "dataset_manifest": str(dataset_manifest_path.resolve()),
            "generation_bridge": (
                str((output_dir / "generation_bridge.json").resolve())
                if generation_bridge is not None
                else None
            ),
            "upstream_results": None if upstream_results_path is None else str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "official_benchmark_verified": official_benchmark_verified,
        "official_component_verified": official_component_verified,
        "integration_evidence": official_component_verified,
        "normalizer_only": normalizer_only,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def category_list_from_args(args: argparse.Namespace) -> list[str]:
    raw_values: list[str] = []
    if args.category:
        raw_values.extend(args.category)
    env_value = os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_CATEGORY")
    if env_value and not raw_values:
        raw_values.extend(env_value.split(","))
    if not raw_values:
        raw_values = ["consistent_attribute_binding"]

    categories: list[str] = []
    for raw_value in raw_values:
        for part in raw_value.split(","):
            part = part.strip()
            if not part:
                continue
            if part.lower() == "all":
                categories.extend(COMPONENT_METRICS)
                continue
            metric_id = canonical_metric_id(part)
            if metric_id not in COMPONENT_METRICS:
                raise ValueError(f"unknown T2V-CompBench category: {part}")
            categories.append(metric_id)
    return list(dict.fromkeys(categories))


def has_explicit_category_selection(args: argparse.Namespace) -> bool:
    return bool(args.category or os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_CATEGORY"))


def prepare_generated_video_bridge(args: argparse.Namespace) -> dict[str, Any] | None:
    """Strictly stage generic generated artifacts for the official evaluators."""

    existing = getattr(args, "generation_bridge", None)
    if isinstance(existing, dict):
        return existing
    if args.generated_video_dir is None:
        return None
    if args.video_root is not None:
        raise ValueError("--generated-video-dir and --video-root are mutually exclusive")
    selected_categories = category_list_from_args(args) if has_explicit_category_selection(args) else None
    report = materialize_t2v_compbench_official_layout(
        generated_video_dir=args.generated_video_dir,
        official_layout_dir=args.output_dir / "official_video_layout",
        generation_manifest_path=args.generation_manifest,
        selected_categories=selected_categories,
        assets_root=args.t2v_compbench_assets,
    )
    args.video_root = Path(report["official_video_root"])
    if not has_explicit_category_selection(args):
        args.category = list(report["selected_categories"])
    args.generation_bridge = report
    write_json(args.output_dir / "generation_bridge.json", report)
    return report


def category_video_aliases(category: str) -> tuple[str, ...]:
    aliases = CATEGORY_VIDEO_SUBDIR_ALIASES.get(category)
    if aliases:
        return aliases
    return (str(CATEGORY_SPECS[category]["video_subdir"]),)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def path_has_video_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
            return True
    return False


def category_video_candidates(args: argparse.Namespace, category: str) -> list[Path]:
    if args.video_root is None:
        return []
    video_root = args.video_root
    aliases = category_video_aliases(category)
    candidates: list[Path] = []

    if video_root.name in aliases:
        candidates.append(video_root)
    direct_candidates = [video_root / alias for alias in aliases]
    candidates.extend(direct_candidates)

    model_name = args.model_name
    if model_name:
        model_root = video_root / model_name
        candidates.extend(model_root / alias for alias in aliases)
        candidates.extend(model_root / model_name / alias for alias in aliases)

    has_direct_layout = any(candidate.exists() for candidate in direct_candidates)
    if video_root.is_dir() and not has_direct_layout:
        for child in sorted(path for path in video_root.iterdir() if path.is_dir())[:256]:
            candidates.extend(child / alias for alias in aliases)
            candidates.extend(child / child.name / alias for alias in aliases)

    return dedupe_paths(candidates)


def category_video_path(args: argparse.Namespace, category: str) -> Path:
    if args.video_root is None:
        raise ValueError("--video-root or WORLDFOUNDRY_T2V_COMPBENCH_VIDEO_ROOT is required for official execution")
    candidates = category_video_candidates(args, category)
    for candidate in candidates:
        if path_has_video_files(candidate):
            return candidate
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if existing:
        return existing[0]
    rendered = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"T2V-CompBench video directory not found for {category}; tried:\n  - {rendered}"
    )


def category_prompt_file(args: argparse.Namespace, category: str) -> Path:
    return args.t2v_compbench_assets / str(CATEGORY_SPECS[category]["prompt_file"])


def maybe_add(command: list[str], flag: str, value: str | Path | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_single_category_command(args: argparse.Namespace, category: str, output_root: Path) -> tuple[Path, list[str]]:
    root = args.t2v_compbench_root
    spec = CATEGORY_SPECS[category]
    script = root / spec["script"]
    if not script.is_file():
        raise FileNotFoundError(f"T2V-CompBench script not found: {script}")

    csv_dir = output_root / "csv" / spec["csv_subdir"]
    visual_dir = output_root / "visualizations" / category
    model_name = args.model_name or "worldfoundry"
    command = [
        args.python,
        str(script),
        "--video-path",
        str(category_video_path(args, category)),
        "--output-path",
        str(csv_dir),
        "--read-prompt-file",
        str(category_prompt_file(args, category)),
    ]

    metric_type = spec["metric_type"]
    if metric_type == "mllm":
        command.extend(["--t2v-model", model_name])
        maybe_add(command, "--model-path", args.llava_model_path)
        maybe_add(command, "--model-base", args.llava_model_base)
    elif category == "spatial_relationships":
        command.extend(
            [
                "--depth_folder",
                str(args.depth_folder or (output_root / "depth" / category)),
                "--t2v-model",
                model_name,
                "--output_dir",
                str(visual_dir),
                "--device",
                args.device,
            ]
        )
        maybe_add(command, "--config", args.grounded_config)
        maybe_add(command, "--grounded_checkpoint", args.grounded_checkpoint)
        maybe_add(command, "--sam_checkpoint", args.sam_checkpoint)
    elif category == "generative_numeracy":
        command.extend(["--t2v-model", model_name, "--output_dir", str(visual_dir)])
        maybe_add(command, "--config_file", args.grounded_config)
        maybe_add(command, "--checkpoint_path", args.grounded_checkpoint)
    elif category == "motion_binding":
        raise ValueError("motion_binding requires --run-motion-two-stage; use --csv-dir for existing official outputs")
    else:
        raise ValueError(f"unsupported T2V-CompBench category execution: {category}")

    cwd = root / spec["cwd"]
    return cwd, command


def build_motion_commands(args: argparse.Namespace, output_root: Path) -> list[tuple[Path, list[str]]]:
    if not args.run_motion_two_stage:
        raise ValueError("motion_binding official execution requires --run-motion-two-stage or precomputed --csv-dir")
    root = args.t2v_compbench_root
    model_name = args.model_name or "worldfoundry"
    seg_script = root / "grounded_sam_metrics" / "compbench_motion_binding_seg.py"
    eval_script = root / "dot" / "compbench_eval_motion_binding.py"
    for script in (seg_script, eval_script):
        if not script.is_file():
            raise FileNotFoundError(f"T2V-CompBench motion script not found: {script}")
    seg_output = output_root / "motion_binding_seg"
    eval_output = output_root / "visualizations" / "motion_binding"
    csv_dir = output_root / "csv" / CATEGORY_SPECS["motion_binding"]["csv_subdir"]
    motion_video_path = category_video_path(args, "motion_binding")
    standard_video_path = args.motion_standard_video_path or (
        motion_video_path.parent / "video_standard" / motion_video_path.name
    )

    stage1 = [
        args.python,
        str(seg_script),
        "--video-path",
        str(motion_video_path),
        "--read-prompt-file",
        str(category_prompt_file(args, "motion_binding")),
        "--t2v-model",
        model_name,
        "--total_frame",
        str(args.total_frame),
        "--fps",
        str(args.fps),
        "--output_dir",
        str(seg_output),
    ]
    maybe_add(stage1, "--grounded_checkpoint", args.grounded_checkpoint)
    maybe_add(stage1, "--sam_checkpoint", args.sam_checkpoint)

    stage2 = [
        args.python,
        str(eval_script),
        "--video-path",
        str(standard_video_path),
        "--mask_folder",
        str(seg_output),
        "--read-prompt-file",
        str(category_prompt_file(args, "motion_binding")),
        "--t2v-model",
        model_name,
        "--output-path",
        str(csv_dir),
        "--output_dir",
        str(eval_output),
    ]
    return [(root, stage1), (root / "dot", stage2)]


def build_official_commands(args: argparse.Namespace, output_root: Path) -> list[tuple[Path, list[str]]]:
    commands: list[tuple[Path, list[str]]] = []
    for category in category_list_from_args(args):
        if category == "motion_binding":
            commands.extend(build_motion_commands(args, output_root))
        else:
            commands.append(build_single_category_command(args, category, output_root))
    return commands


def build_official_env(root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    pythonpath_entries = []
    extra_prepend = env.get("WORLDFOUNDRY_T2V_COMPBENCH_PREPEND_PYTHONPATH")
    if extra_prepend:
        pythonpath_entries.extend(item for item in extra_prepend.split(os.pathsep) if item)
    pythonpath_entries.extend(
        [
            str(REPO_ROOT),
            str(root),
            str(root / "dot"),
        ]
    )
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries).rstrip(os.pathsep)
    return env


def run_command_sequence(
    commands: list[tuple[Path, list[str]]],
    *,
    root: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[list[list[str]], float, int]:
    env = build_official_env(root)

    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    command_lines: list[list[str]] = []
    start = time.monotonic()
    final_returncode = 0
    for cwd, command in commands:
        command_lines.append(command)
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        with stdout_path.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.write(completed.stdout)
            if completed.stdout and not completed.stdout.endswith("\n"):
                handle.write("\n")
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.write(completed.stderr)
            if completed.stderr and not completed.stderr.endswith("\n"):
                handle.write("\n")
        if completed.returncode != 0 and final_returncode == 0:
            final_returncode = completed.returncode
            break
    return command_lines, time.monotonic() - start, final_returncode


def preflight_check(
    name: str,
    ok: bool,
    *,
    required: bool = True,
    path: Path | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "path": None if path is None else str(path),
        "detail": detail,
    }


def count_direct_video_files(path: Path | None) -> int:
    if path is None or not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS)


def selected_runtime_groups(categories: list[str]) -> list[str]:
    groups: list[str] = []
    for category in categories:
        groups.extend(CATEGORY_RUNTIME_GROUPS.get(category, ()))
    return list(dict.fromkeys(groups))


def python_import_preflight(args: argparse.Namespace, group: str) -> dict[str, Any]:
    modules = PREFLIGHT_IMPORT_GROUPS[group]
    code = """
import importlib
import json
import sys

modules = json.loads(sys.argv[1])
for module in modules:
    try:
        importlib.import_module(module)
        result = {"module": module, "ok": True, "error": None}
    except Exception as exc:
        result = {"module": module, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False), flush=True)
"""
    def parse_module_results(stdout: str | bytes | None) -> list[dict[str, Any]]:
        if stdout is None:
            return []
        text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
        results: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "module" in item:
                results.append(item)
        return results

    try:
        completed = subprocess.run(
            [args.python, "-c", code, json.dumps(list(modules))],
            cwd=args.t2v_compbench_root,
            env=build_official_env(args.t2v_compbench_root),
            capture_output=True,
            text=True,
            timeout=args.preflight_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        module_results = parse_module_results(exc.stdout)
        completed_modules = {item.get("module") for item in module_results}
        module_results.extend(
            {
                "module": module,
                "ok": False,
                "error": f"TimeoutExpired: import probe exceeded {args.preflight_timeout}s",
            }
            for module in modules
            if module not in completed_modules
        )
        return {
            "group": group,
            "ok": False,
            "returncode": None,
            "modules": module_results,
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    module_results = parse_module_results(completed.stdout)
    if not module_results:
        module_results = [
            {
                "module": module,
                "ok": False,
                "error": "import probe did not return JSON",
            }
            for module in modules
        ]
    return {
        "group": group,
        "ok": completed.returncode == 0 and all(item.get("ok") for item in module_results),
        "returncode": completed.returncode,
        "modules": module_results,
        "stderr": completed.stderr.strip()[-4000:],
    }


def build_t2v_compbench_preflight(args: argparse.Namespace) -> dict[str, Any]:
    generation_bridge = prepare_generated_video_bridge(args)
    categories = category_list_from_args(args)
    checks: list[dict[str, Any]] = []
    root = args.t2v_compbench_root
    assets_root = args.t2v_compbench_assets
    checks.append(preflight_check("official_repo_root", root.is_dir(), path=root))
    checks.append(preflight_check("asset_root", assets_root.is_dir(), path=assets_root))

    for category in categories:
        spec = CATEGORY_SPECS[category]
        script = root / spec["script"]
        prompt_file = category_prompt_file(args, category)
        checks.append(preflight_check(f"{category}.script", script.is_file(), path=script))
        checks.append(preflight_check(f"{category}.prompt_file", prompt_file.is_file(), path=prompt_file))

    if args.dataset_root is None:
        checks.append(
            preflight_check(
                "dataset_root",
                False,
                required=False,
                path=None,
                detail="optional Kaiyue/T2V-CompBench-Videos mirror; official prompt assets are already in tree",
            )
        )
    else:
        checks.append(
            preflight_check(
                "dataset_root",
                args.dataset_root.is_dir(),
                required=False,
                path=args.dataset_root,
            )
        )
        checks.append(
            preflight_check(
                "dataset_readme",
                (args.dataset_root / "README.md").is_file(),
                required=False,
                path=args.dataset_root / "README.md",
            )
        )

    video_checks: dict[str, Any] = {}
    if args.csv_dir is None and args.from_upstream_results is None:
        if args.video_root is None:
            checks.append(preflight_check("video_root", False, path=None, detail="--video-root or WORLDFOUNDRY_T2V_COMPBENCH_VIDEO_ROOT is required for official execution"))
        else:
            checks.append(preflight_check("video_root", args.video_root.is_dir(), path=args.video_root))
            for category in categories:
                candidates = category_video_candidates(args, category)
                selected = next((candidate for candidate in candidates if path_has_video_files(candidate)), None)
                count = count_direct_video_files(selected)
                expected_count = (
                    int(generation_bridge["by_category"].get(category, 0))
                    if generation_bridge is not None
                    else PROMPTS_PER_CATEGORY
                )
                video_checks[category] = {
                    "selected_path": None if selected is None else str(selected),
                    "video_file_count": count,
                    "expected_file_count": expected_count,
                    "candidate_paths": [str(candidate) for candidate in candidates],
                }
                checks.append(
                    preflight_check(
                        f"{category}.video_files",
                        expected_count > 0 and count == expected_count,
                        path=selected,
                        detail=f"found={count}, expected={expected_count}",
                    )
                )
    elif args.csv_dir is not None:
        checks.append(preflight_check("csv_dir", args.csv_dir.is_dir(), path=args.csv_dir))
    elif args.from_upstream_results is not None:
        checks.append(preflight_check("from_upstream_results", args.from_upstream_results.is_file(), path=args.from_upstream_results))

    groups = selected_runtime_groups(categories)
    import_reports = [python_import_preflight(args, group) for group in groups]
    for report in import_reports:
        missing = [item["module"] for item in report["modules"] if not item.get("ok")]
        checks.append(
            preflight_check(
                f"{report['group']}.python_imports",
                report["ok"],
                detail="missing=" + ",".join(missing) if missing else None,
            )
        )

    if "mllm" in groups:
        checks.append(
            preflight_check(
                "mllm.llava_model_path",
                bool(args.llava_model_path),
                path=None if args.llava_model_path is None else Path(args.llava_model_path),
                detail="set --llava-model-path or WORLDFOUNDRY_T2V_COMPBENCH_LLAVA_MODEL_PATH for reproducible official MLLM scoring",
            )
        )

    if "detection" in groups:
        if args.grounded_config is not None:
            checks.append(
                preflight_check(
                    "detection.groundingdino_config",
                    args.grounded_config.is_file(),
                    path=args.grounded_config,
                )
            )
        checks.append(
            preflight_check(
                "detection.groundingdino_checkpoint",
                bool(args.grounded_checkpoint and args.grounded_checkpoint.is_file()),
                path=args.grounded_checkpoint,
            )
        )
        checks.append(
            preflight_check(
                "detection.sam_checkpoint",
                bool(args.sam_checkpoint and args.sam_checkpoint.is_file()),
                path=args.sam_checkpoint,
            )
        )

    if "tracking" in groups:
        dot_checkpoint_dir = root / "dot" / "checkpoints"
        dot_checkpoints = sorted(dot_checkpoint_dir.glob("*.pth")) if dot_checkpoint_dir.is_dir() else []
        checks.append(
            preflight_check(
                "tracking.dot_checkpoints",
                len(dot_checkpoints) >= 1,
                path=dot_checkpoint_dir,
                detail=f"found={len(dot_checkpoints)}",
            )
        )

    required = [check for check in checks if check["required"]]
    ready = all(check["ok"] for check in required) and all(report["ok"] for report in import_reports)
    missing_names = {check["name"] for check in required if not check["ok"]}
    next_actions: list[str] = []
    import_failures = [
        item
        for report in import_reports
        for item in report["modules"]
        if not item.get("ok")
    ]
    if any("TimeoutExpired" in str(item.get("error")) for item in import_failures):
        next_actions.append(
            "The selected Python import probe timed out. Increase --preflight-timeout or investigate the "
            "reported slow module before starting the official evaluator."
        )
    if any(
        "TimeoutExpired" not in str(item.get("error"))
        for item in import_failures
    ):
        next_actions.append(
            "Install the WorldFoundry benchmark dependencies into the selected Python with "
            "scripts/setup/model_env_install.sh --model evaluation-benchmarks. "
            "T2V-CompBench now imports reusable foundation models from worldfoundry.base_models "
            "instead of runner-local LLaVA/GroundingDINO/SAM forks."
        )
    if "mllm.llava_model_path" in missing_names:
        next_actions.append(
            "Set WORLDFOUNDRY_T2V_COMPBENCH_LLAVA_MODEL_PATH or pass --llava-model-path to the LLaVA checkpoint used for MLLM metrics."
        )
    if "detection.groundingdino_checkpoint" in missing_names:
        next_actions.append(
            "Set WORLDFOUNDRY_T2V_COMPBENCH_GROUNDINGDINO_CKPT or WORLDFOUNDRY_GROUNDING_DINO_CKPT to groundingdino_swint_ogc.pth."
        )
    if "detection.sam_checkpoint" in missing_names:
        next_actions.append(
            "Set WORLDFOUNDRY_T2V_COMPBENCH_SAM_CKPT or WORLDFOUNDRY_SAM_VIT_H_CKPT to sam_vit_h_4b8939.pth."
        )
    if "tracking.dot_checkpoints" in missing_names:
        next_actions.append(
            f"Download the official DOT checkpoints into {root / 'dot' / 'checkpoints'} before running motion_binding."
        )
    return {
        "schema_version": "worldfoundry-t2v-compbench-preflight-v1",
        "benchmark_id": args.benchmark_id,
        "ready": ready,
        "categories": categories,
        "runtime_groups": groups,
        "official_repo_root": str(root),
        "dataset_root": None if args.dataset_root is None else str(args.dataset_root),
        "video_root": None if args.video_root is None else str(args.video_root),
        "generation_bridge": generation_bridge,
        "video_checks": video_checks,
        "import_reports": import_reports,
        "checks": checks,
        "missing_required": [check for check in required if not check["ok"]],
        "next_actions": next_actions,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = build_t2v_compbench_preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "preflight_report.json", report)
    return report


def run_t2v_compbench(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_bridge = prepare_generated_video_bridge(args)
    requested_categories = category_list_from_args(args)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    if args.from_upstream_results:
        raw_results = load_json(args.from_upstream_results)
        results = collect_json_results(raw_results)
        return normalize_t2v_compbench_results(
            results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            source_dir=None,
            dataset_root=args.dataset_root,
            video_root=args.video_root,
            upstream_results_path=args.from_upstream_results,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            requested_categories=requested_categories,
            generation_bridge=generation_bridge,
        )

    if args.csv_dir:
        results = collect_csv_results(args.csv_dir, args.model_name)
        return normalize_t2v_compbench_results(
            results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            source_dir=args.csv_dir,
            dataset_root=args.dataset_root,
            video_root=args.video_root,
            upstream_results_path=None,
            command=None,
            duration_seconds=None,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            requested_categories=requested_categories,
            generation_bridge=generation_bridge,
        )

    upstream_output_root = output_dir / "upstream"
    upstream_output_root.mkdir(parents=True, exist_ok=True)
    commands = build_official_commands(args, upstream_output_root)
    command_lines, duration_seconds, returncode = run_command_sequence(
        commands,
        root=args.t2v_compbench_root,
        timeout=args.timeout,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    try:
        results = collect_csv_results(upstream_output_root / "csv", args.model_name)
    except FileNotFoundError:
        if returncode == 0:
            raise
        results = []
    return normalize_t2v_compbench_results(
        results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        source_dir=upstream_output_root / "csv",
        dataset_root=args.dataset_root,
        video_root=args.video_root,
        upstream_results_path=None,
        command=command_lines,
        duration_seconds=duration_seconds,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        requested_categories=requested_categories,
        generation_bridge=generation_bridge,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official T2V-CompBench category scripts or normalize official CSV outputs to a WorldFoundry scorecard."
    )
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "t2v-compbench"))
    parser.add_argument(
        "--t2v-compbench-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_ROOT", DEFAULT_T2V_COMPBENCH_ROOT),
    )
    parser.add_argument(
        "--t2v-compbench-assets",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_ASSETS", DEFAULT_T2V_COMPBENCH_ASSETS),
    )
    parser.add_argument("--csv-dir", type=Path, default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_CSV_DIR"))
    parser.add_argument("--video-root", type=Path, default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_VIDEO_ROOT"))
    parser.add_argument(
        "--generated-video-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_GENERATED_VIDEO_DIR"),
        help=(
            "Model-independent generated videos. Requires an exact generation manifest and is staged "
            "into the official category/NNNN.mp4 layout."
        ),
    )
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_GENERATION_MANIFEST"),
        help=f"JSONL mapping manifest; defaults to <generated-video-dir>/{GENERATION_MANIFEST_NAME}.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_DATASET_ROOT"),
        help="Local Kaiyue/T2V-CompBench-Videos dataset root for discovery evidence.",
    )
    parser.add_argument("--model-name", default=os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_MODEL_NAME"))
    parser.add_argument("--category", action="append", help="Category id, comma-separated ids, or all.")
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_TIMEOUT", "7200")))
    parser.add_argument("--official-results-path", dest="from_upstream_results", type=Path)
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--llava-model-path", default=os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_LLAVA_MODEL_PATH"))
    parser.add_argument("--llava-model-base", default=os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_LLAVA_MODEL_BASE"))
    parser.add_argument(
        "--grounded-config",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_GROUNDINGDINO_CONFIG"),
        help="Optional GroundingDINO config override, including a local bert_base_uncased_path when offline.",
    )
    parser.add_argument(
        "--grounded-checkpoint",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_GROUNDINGDINO_CKPT")
        or env_path("WORLDFOUNDRY_GROUNDING_DINO_CKPT")
        or grounding_dino_checkpoint_path(),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_SAM_CKPT")
        or env_path("WORLDFOUNDRY_SAM_VIT_H_CKPT")
        or sam_v1_checkpoint_path("vit_h"),
    )
    parser.add_argument("--depth-folder", type=Path, default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_DEPTH_DIR"))
    parser.add_argument("--device", default=os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_DEVICE", "cuda"))
    parser.add_argument("--run-motion-two-stage", action="store_true")
    parser.add_argument("--motion-standard-video-path", type=Path, default=env_path("WORLDFOUNDRY_T2V_COMPBENCH_MOTION_STANDARD_VIDEO_PATH"))
    parser.add_argument("--total-frame", default=os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_TOTAL_FRAME", "16"))
    parser.add_argument("--fps", default=os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_FPS", "8"))
    parser.add_argument("--preflight", action="store_true", help="Check official T2V-CompBench runtime, assets, and video layout without running metrics.")
    parser.add_argument(
        "--preflight-timeout",
        type=int,
        default=int(os.environ.get("WORLDFOUNDRY_T2V_COMPBENCH_PREFLIGHT_TIMEOUT", "180")),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def resolve_path_arg(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def normalize_path_args(args: argparse.Namespace) -> None:
    for attr in (
        "t2v_compbench_root",
        "t2v_compbench_assets",
        "csv_dir",
        "video_root",
        "generated_video_dir",
        "generation_manifest",
        "dataset_root",
        "output_dir",
        "from_upstream_results",
        "grounded_config",
        "grounded_checkpoint",
        "sam_checkpoint",
        "depth_folder",
        "motion_standard_video_path",
    ):
        setattr(args, attr, resolve_path_arg(getattr(args, attr)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight and args.output_dir is None:
        args.output_dir = REPO_ROOT / "tmp" / "benchmark_zoo" / "t2v_compbench_preflight"
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    normalize_path_args(args)

    if args.preflight:
        try:
            report = run_preflight(args)
        except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            status = "ready" if report["ready"] else "blocked"
            print(f"{args.benchmark_id}: official T2V-CompBench preflight {status}")
            print(f"report: {args.output_dir / 'preflight_report.json'}")
        return 0 if report["ready"] else 1

    try:
        scorecard = run_t2v_compbench(args)
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["integration_evidence"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_scores": scorecard["artifacts"]["per_sample_scores"],
        "leaderboard_csv_manifest": scorecard["artifacts"]["leaderboard_csv_manifest"],
        "generated_video_manifest": scorecard["artifacts"]["generated_video_manifest"],
        "dataset_manifest": scorecard["artifacts"]["dataset_manifest"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "official_component_verified": scorecard["official_component_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official T2V-CompBench validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
