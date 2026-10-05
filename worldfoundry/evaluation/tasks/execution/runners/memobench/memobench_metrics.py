"""MemoBench metric normalization.

MemoBench emits several files across its three official stages:

* ``evaluation/run_eval.py`` writes per-clip automated metric CSVs.
* ``evaluation/compute_ors.py`` writes ``ors_scores.csv``.
* ``evaluation/vqa/llm-vqa.py`` writes per-clip VQA CSVs and ``overall.csv``.
* ``leaderboard/leaderboard.py`` writes a model-level ``leaderboard.csv``.

This module accepts any mix of those files and returns WorldFoundry-style metric
rows with normalized scores in [0, 1].
"""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric

METRIC_ORDER = (
    "visual_quality",
    "motion_smoothness",
    "object_identity_consistency",
    "geo3d_consistency",
    "camera_controllability",
    "image_reward_score",
    "object_revisit_score",
    "gt_all_psnr",
    "gt_all_ssim",
    "gt_all_lpips",
    "instruction_following",
    "object_background",
    "continuity_of_memory",
    "physics_adherence",
    "memobench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "visual_quality": {
        "name": "Visual Quality",
        "group": "automated",
        "higher_is_better": True,
        "description": "Perceptual quality from CLIP-IQA+ and LAION aesthetic scores.",
    },
    "motion_smoothness": {
        "name": "Motion Smoothness",
        "group": "automated",
        "higher_is_better": True,
        "description": "Temporal coherence via optical-flow warp error over visible and reappear phases.",
    },
    "object_identity_consistency": {
        "name": "Object Identity Consistency",
        "group": "automated",
        "higher_is_better": True,
        "description": "DINOv2 object-centric identity consistency after reappearance.",
    },
    "geo3d_consistency": {
        "name": "Geo3D Consistency",
        "group": "automated",
        "higher_is_better": True,
        "description": "Depth-based geometric consistency over visible and reappear phases.",
    },
    "camera_controllability": {
        "name": "Camera Controllability",
        "group": "automated",
        "higher_is_better": True,
        "description": "Agreement between generated and target camera motion.",
    },
    "image_reward_score": {
        "name": "Image Reward Score",
        "group": "automated",
        "higher_is_better": True,
        "description": "Human-preference alignment score normalized by MemoBench.",
    },
    "object_revisit_score": {
        "name": "Object Revisit Score",
        "group": "object_permanence",
        "higher_is_better": True,
        "description": "SAM-3 detection score for target-object reappearance.",
    },
    "gt_all_psnr": {
        "name": "GT All PSNR",
        "group": "pixel_fidelity",
        "higher_is_better": True,
        "description": "PSNR against ground-truth frames across all phases.",
    },
    "gt_all_ssim": {
        "name": "GT All SSIM",
        "group": "pixel_fidelity",
        "higher_is_better": True,
        "description": "SSIM against ground-truth frames across all phases.",
    },
    "gt_all_lpips": {
        "name": "GT All LPIPS",
        "group": "pixel_fidelity",
        "higher_is_better": False,
        "description": "LPIPS against ground-truth frames across all phases; lower is better.",
    },
    "instruction_following": {
        "name": "Instruction Following",
        "group": "vqa",
        "higher_is_better": True,
        "description": "VQA pass rate for requested camera movements and events.",
    },
    "object_background": {
        "name": "Object and Background",
        "group": "vqa",
        "higher_is_better": True,
        "description": "VQA pass rate for object identity and background consistency.",
    },
    "continuity_of_memory": {
        "name": "Continuity of Memory",
        "group": "vqa",
        "higher_is_better": True,
        "description": "VQA pass rate for memory of target-object state while out of frame.",
    },
    "physics_adherence": {
        "name": "Physics Adherence",
        "group": "vqa",
        "higher_is_better": True,
        "description": "VQA pass rate for plausible lighting, shadows, and motion.",
    },
    "memobench_average": {
        "name": "MemoBench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean normalized score over available MemoBench component metrics.",
        "primary": True,
    },
}

METRIC_ALIASES = {
    "visualquality": "visual_quality",
    "visual_quality": "visual_quality",
    "visqual": "visual_quality",
    "motion_smoothness": "motion_smoothness",
    "motionsmoothness": "motion_smoothness",
    "motsmooth": "motion_smoothness",
    "objidentityconsistency": "object_identity_consistency",
    "object_identity_consistency": "object_identity_consistency",
    "objconsist": "object_identity_consistency",
    "geo3dconsistency": "geo3d_consistency",
    "geo_3d_consistency": "geo3d_consistency",
    "3dconsist": "geo3d_consistency",
    "camera_controllability": "camera_controllability",
    "cameracontrollability": "camera_controllability",
    "camctrl": "camera_controllability",
    "image_reward_score": "image_reward_score",
    "imagerewardscore": "image_reward_score",
    "imagerewardscore_pct": "image_reward_score",
    "imgreward": "image_reward_score",
    "ors": "object_revisit_score",
    "object_revisit_score": "object_revisit_score",
    "objectrevisitscore": "object_revisit_score",
    "gt_all_psnr": "gt_all_psnr",
    "psnr": "gt_all_psnr",
    "gt_all_ssim": "gt_all_ssim",
    "ssim": "gt_all_ssim",
    "gt_all_lpips": "gt_all_lpips",
    "lpips": "gt_all_lpips",
    "instructionfollowing": "instruction_following",
    "instruction_following": "instruction_following",
    "vqa_if": "instruction_following",
    "objectbackground": "object_background",
    "object_background": "object_background",
    "object_and_background": "object_background",
    "vqa_ob": "object_background",
    "continuityofmemory": "continuity_of_memory",
    "continuity_of_memory": "continuity_of_memory",
    "vqa_cm": "continuity_of_memory",
    "physicsadherence": "physics_adherence",
    "physics_adherence": "physics_adherence",
    "vqa_pa": "physics_adherence",
    "memobench_average": "memobench_average",
    "memobench": "memobench_average",
    "average": "memobench_average",
    "overall": "memobench_average",
}

MODEL_KEYS = ("model", "Model", "model_name", "Model Name")
SCORE_JSON_KEYS = ("score", "scores")


def canonical_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonical_model(value: Any) -> str:
    return canonical_key(value).replace("_", "")


def metric_id_for_key(value: Any) -> str | None:
    key = canonical_key(value)
    return METRIC_ALIASES.get(key) or METRIC_ALIASES.get(key.replace("_", ""))


def _number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def normalize_score(metric_id: str, raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    value = float(raw_score)
    if metric_id == "gt_all_lpips":
        unit = value if 0.0 <= value <= 1.0 else value / 100.0
        return _clamp01(1.0 - unit)
    if metric_id in {"gt_all_ssim", "object_revisit_score"}:
        return _clamp01(value if 0.0 <= value <= 1.0 else value / 100.0)
    if metric_id == "gt_all_psnr":
        return _clamp01(value if 0.0 <= value <= 1.0 else value / 100.0)
    return _clamp01(value if 0.0 <= value <= 1.0 else value / 100.0)


def _parse_score_json(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    return parsed if isinstance(parsed, Mapping) else None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "records", "results", "metrics"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
        return [dict(payload)]
    return []


def _candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = [
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in {".csv", ".json", ".jsonl"}
    ]
    preferred: list[Path] = []
    for child in files:
        name = child.name.lower()
        if (
            name == "leaderboard.csv"
            or name.startswith("eval_")
            or name == "ors_scores.csv"
            or name == "overall.csv"
            or "memobench" in name
        ):
            preferred.append(child)
    return preferred or files


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(path)
    if suffix == ".json":
        return _read_json_rows(path)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
        return rows
    return []


def _row_model(row: Mapping[str, Any]) -> str | None:
    for key in MODEL_KEYS:
        if key in row and str(row[key]).strip():
            return str(row[key])
    return None


def _should_include_row(row: Mapping[str, Any], model_name: str | None) -> bool:
    if model_name is None:
        return True
    row_model = _row_model(row)
    if row_model is None:
        return True
    return canonical_model(row_model) == canonical_model(model_name)


def _iter_metric_values(row: Mapping[str, Any]) -> Iterable[tuple[str, float]]:
    # The official run_eval.py CSV keeps both ImageRewardScore and its _pct
    # diagnostic column after replacing ImageRewardScore with the composite.
    # Count the metric once per clip even when both aliases are present.
    seen: set[str] = set()
    for key in SCORE_JSON_KEYS:
        score_map = _parse_score_json(row.get(key))
        if score_map:
            for score_key, score_value in score_map.items():
                metric_id = metric_id_for_key(score_key)
                score = _number(score_value)
                if metric_id is not None and score is not None and metric_id not in seen:
                    seen.add(metric_id)
                    yield metric_id, score

    for raw_key, raw_value in row.items():
        if raw_key in SCORE_JSON_KEYS:
            continue
        metric_id = metric_id_for_key(raw_key)
        score = _number(raw_value)
        if metric_id is not None and score is not None and metric_id not in seen:
            seen.add(metric_id)
            yield metric_id, score


def compute_memobench_metrics(
    paths: Sequence[Path],
    *,
    model_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for input_path in paths:
        for path in _candidate_files(input_path):
            try:
                rows = _load_rows(path)
            except (OSError, csv.Error, json.JSONDecodeError):
                continue
            for row in rows:
                if not _should_include_row(row, model_name):
                    continue
                for metric_id, raw_score in _iter_metric_values(row):
                    normalized = normalize_score(metric_id, raw_score)
                    if normalized is None:
                        continue
                    buckets[metric_id].append((raw_score, normalized, str(path)))

    metrics: dict[str, dict[str, Any]] = {}
    for metric_id, values in buckets.items():
        raw_mean = mean_numeric(raw for raw, _normalized, _source in values)
        normalized_mean = mean_numeric(normalized for _raw, normalized, _source in values)
        sources = sorted({source for _raw, _normalized, source in values})
        metrics[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_mean,
            "normalized_score": normalized_mean,
            "source": ";".join(sources[:5]),
            "sample_count": len(values),
        }

    if "memobench_average" not in metrics:
        components = [
            row["normalized_score"]
            for metric_id, row in metrics.items()
            if metric_id != "memobench_average" and row.get("normalized_score") is not None
        ]
        average = mean_numeric(components)
        if average is not None:
            metrics["memobench_average"] = {
                "metric_id": "memobench_average",
                "raw_score": average,
                "normalized_score": average,
                "source": "mean_available_component_metrics",
                "sample_count": len(components),
            }
    return metrics


def metric_rows(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        item = metrics.get(metric_id, {})
        normalized = item.get("normalized_score")
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": normalized is not None,
                "raw_score": item.get("raw_score"),
                "normalized_score": normalized,
                "score": normalized,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "source": item.get("source"),
                "sample_count": item.get("sample_count"),
                "reason": None if normalized is not None else "score_not_available_in_memobench_results",
            }
        )
    return rows


__all__ = [
    "METRIC_ALIASES",
    "METRIC_ORDER",
    "METRIC_SPECS",
    "compute_memobench_metrics",
    "metric_rows",
    "normalize_score",
]

