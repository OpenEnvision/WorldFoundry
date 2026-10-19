"""MiraBench metric normalization from official summary or per-sample exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean
from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float

METRIC_ORDER = (
    "dynamic_degree",
    "tracking_strength",
    "dino_temporal_consistency",
    "clip_temporal_consistency",
    "temporal_motion_smoothness",
    "mean_absolute_error",
    "root_mean_square_error",
    "aesthetic_quality",
    "imaging_quality",
    "camera_alignment",
    "main_object_alignment",
    "background_alignment",
    "style_alignment",
    "overall_alignment",
    "fvd",
    "fid",
    "kid",
    "mirabench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "dynamic_degree": {
        "name": "Dynamic Degree",
        "group": "motion_strength",
        "higher_is_better": True,
        "description": "MiraBench RAFT optical-flow dynamic degree for motion strength.",
    },
    "tracking_strength": {
        "name": "Tracking Strength",
        "group": "motion_strength",
        "higher_is_better": True,
        "description": "MiraBench tracking-based long-range motion strength.",
    },
    "dino_temporal_consistency": {
        "name": "DINO Temporal Consistency",
        "group": "temporal_consistency",
        "higher_is_better": True,
        "description": "MiraBench DINO adjacent-frame structural temporal consistency.",
    },
    "clip_temporal_consistency": {
        "name": "CLIP Temporal Consistency",
        "group": "temporal_consistency",
        "higher_is_better": True,
        "description": "MiraBench CLIP adjacent-frame semantic temporal consistency.",
    },
    "temporal_motion_smoothness": {
        "name": "Temporal Motion Smoothness",
        "group": "temporal_consistency",
        "higher_is_better": True,
        "description": "MiraBench AMT-prior temporal motion smoothness.",
    },
    "mean_absolute_error": {
        "name": "Mean Absolute Error",
        "group": "three_d_consistency",
        "higher_is_better": False,
        "description": "MiraBench 3D reconstruction mean absolute error; lower is better.",
    },
    "root_mean_square_error": {
        "name": "Root Mean Square Error",
        "group": "three_d_consistency",
        "higher_is_better": False,
        "description": "MiraBench 3D reconstruction root mean square error; lower is better.",
    },
    "aesthetic_quality": {
        "name": "Aesthetic Quality",
        "group": "visual_quality",
        "higher_is_better": True,
        "description": "MiraBench LAION aesthetic quality score.",
    },
    "imaging_quality": {
        "name": "Imaging Quality",
        "group": "visual_quality",
        "higher_is_better": True,
        "description": "MiraBench MUSIQ imaging quality score.",
    },
    "camera_alignment": {
        "name": "Camera Alignment",
        "group": "text_video_alignment",
        "higher_is_better": True,
        "description": "MiraBench ViCLIP camera-caption alignment.",
    },
    "main_object_alignment": {
        "name": "Main Object Alignment",
        "group": "text_video_alignment",
        "higher_is_better": True,
        "description": "MiraBench ViCLIP main-object-caption alignment.",
    },
    "background_alignment": {
        "name": "Background Alignment",
        "group": "text_video_alignment",
        "higher_is_better": True,
        "description": "MiraBench ViCLIP background-caption alignment.",
    },
    "style_alignment": {
        "name": "Style Alignment",
        "group": "text_video_alignment",
        "higher_is_better": True,
        "description": "MiraBench ViCLIP style-caption alignment.",
    },
    "overall_alignment": {
        "name": "Overall Alignment",
        "group": "text_video_alignment",
        "higher_is_better": True,
        "description": "MiraBench ViCLIP overall text-video alignment.",
    },
    "fvd": {
        "name": "FVD",
        "group": "distribution_similarity",
        "higher_is_better": False,
        "description": "MiraBench Fréchet Video Distance distribution metric; lower is better.",
    },
    "fid": {
        "name": "FID",
        "group": "distribution_similarity",
        "higher_is_better": False,
        "description": "MiraBench Fréchet Inception Distance distribution metric; lower is better.",
    },
    "kid": {
        "name": "KID",
        "group": "distribution_similarity",
        "higher_is_better": False,
        "description": "MiraBench Kernel Inception Distance distribution metric; lower is better.",
    },
    "mirabench_average": {
        "name": "MiraBench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "MiraBench official aggregate when supplied by the official scorer or normalized output.",
        "primary": True,
    },
}

UPSTREAM_METRIC_ALIASES = {
    "dynamic_degree": "dynamic_degree",
    "tracking_strength": "tracking_strength",
    "temporal_dino_consistency": "dino_temporal_consistency",
    "dino_temporal_consistency": "dino_temporal_consistency",
    "temporal_clip_consistency": "clip_temporal_consistency",
    "clip_temporal_consistency": "clip_temporal_consistency",
    "temporal_motion_smoothness": "temporal_motion_smoothness",
    "3d_consistency_mean_err": "mean_absolute_error",
    "3d_consistency_rmse": "root_mean_square_error",
    "mean_absolute_error": "mean_absolute_error",
    "root_mean_square_error": "root_mean_square_error",
    "aesthetic_quality": "aesthetic_quality",
    "imaging_quality": "imaging_quality",
    "camera_alignment": "camera_alignment",
    "main_object_alignment": "main_object_alignment",
    "background_alignment": "background_alignment",
    "style_alignment": "style_alignment",
    "overall_consistency": "overall_alignment",
    "overall_alignment": "overall_alignment",
    "fvd": "fvd",
    "kvd": "fvd",
    "fid": "fid",
    "kid": "kid",
    "mirabench_average": "mirabench_average",
    "average": "mirabench_average",
}


def _canonical_metric_id(raw: str) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return UPSTREAM_METRIC_ALIASES.get(key) or UPSTREAM_METRIC_ALIASES.get(key.replace(" ", "_"))


def _is_summary_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and any(
        _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "")) for row in rows
    )


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        metric_id = _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or ""))
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if metric_id is None or score is None:
            continue
        buckets.setdefault(metric_id, []).append(score)
    metrics = {metric_id: _mean(values) for metric_id, values in buckets.items() if values}
    if metrics.get("mirabench_average") is None:
        component_values = [value for key, value in metrics.items() if key != "mirabench_average" and value is not None]
        if component_values:
            metrics["mirabench_average"] = sum(component_values) / len(component_values)
    return metrics


def _aggregate_sample_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in METRIC_ORDER if metric_id != "mirabench_average"}
    for row in rows:
        for raw_key, raw_value in row.items():
            metric_id = _canonical_metric_id(str(raw_key))
            score = _to_float(raw_value)
            if metric_id is None or score is None or metric_id == "mirabench_average":
                continue
            buckets.setdefault(metric_id, []).append(score)
    metrics = {metric_id: _mean(values) for metric_id, values in buckets.items() if values}
    component_values = [value for key, value in metrics.items() if value is not None]
    if component_values:
        metrics["mirabench_average"] = sum(component_values) / len(component_values)
    else:
        metrics["mirabench_average"] = None
    return metrics


def compute_mirabench_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_path: Path | None = None,
) -> dict[str, Any]:
    if _is_summary_rows(rows):
        direct_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    if len(rows) == 1 and isinstance(rows[0], Mapping):
        payload = rows[0]
        direct_metrics = {}
        for raw_key, raw_value in payload.items():
            metric_id = _canonical_metric_id(str(raw_key))
            score = _to_float(raw_value)
            if metric_id is None or score is None:
                continue
            direct_metrics[metric_id] = score
        if direct_metrics:
            if direct_metrics.get("mirabench_average") is None:
                component_values = [
                    value for key, value in direct_metrics.items() if key != "mirabench_average" and value is not None
                ]
                if component_values:
                    direct_metrics["mirabench_average"] = sum(component_values) / len(component_values)
            return {
                "metrics": direct_metrics,
                "components": {"format": "summary_json_object"},
            }

    direct_metrics = _aggregate_sample_rows(rows)
    return {
        "metrics": direct_metrics,
        "components": {
            "sample_count": len(rows),
            "format": "video_score_csv",
        },
    }


def load_results_rows(results_path: Path) -> list[dict[str, Any]]:
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("results", "rows", "records", "metrics"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
            return [dict(payload)]
        raise ValueError(f"Unsupported MiraBench JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
