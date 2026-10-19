"""World-in-World metric normalization from evaluator summaries or caller exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean

METRIC_ORDER = (
    "active_recognition_success_rate",
    "image_goal_navigation_success_rate",
    "image_goal_navigation_spl",
    "active_embodied_qa_score",
    "active_embodied_qa_spl",
    "robotic_manipulation_success_rate",
    "interaction_trace_consistency",
    "world_in_world_average",
)

VIDEO_QUALITY_METRIC_IDS = ("fvd", "ssim", "psnr", "lpips")

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "active_recognition_success_rate": {
        "name": "Active Recognition Success Rate",
        "group": "utility",
        "higher_is_better": True,
        "description": "Active Recognition task accuracy from official result exports.",
    },
    "image_goal_navigation_success_rate": {
        "name": "Image Goal Navigation Success Rate",
        "group": "utility",
        "higher_is_better": True,
        "description": "IGNav success rate (sr) from official result exports.",
    },
    "image_goal_navigation_spl": {
        "name": "Image Goal Navigation SPL",
        "group": "utility",
        "higher_is_better": True,
        "description": "IGNav SPL from official result exports.",
    },
    "active_embodied_qa_score": {
        "name": "Active Embodied QA Score",
        "group": "utility",
        "higher_is_better": True,
        "description": "AEQA mean score normalized to [0, 1].",
    },
    "active_embodied_qa_spl": {
        "name": "Active Embodied QA SPL",
        "group": "utility",
        "higher_is_better": True,
        "description": "AEQA mean efficiency (SPL-style) normalized to [0, 1].",
    },
    "robotic_manipulation_success_rate": {
        "name": "Robotic Manipulation Success Rate",
        "group": "utility",
        "higher_is_better": True,
        "description": "Manipulation task success rate when supplied by official result exports.",
    },
    "interaction_trace_consistency": {
        "name": "Interaction Trace Consistency",
        "group": "utility",
        "higher_is_better": True,
        "description": "Closed-loop interaction trace consistency score.",
    },
    "world_in_world_average": {
        "name": "World In World Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean over available World-in-World utility metrics.",
        "primary": True,
    },
    "fvd": {
        "name": "FVD",
        "group": "video_quality",
        "higher_is_better": False,
        "description": "Fréchet Video Distance from evaluation/FVD/cal_4metrics.py.",
    },
    "ssim": {
        "name": "SSIM",
        "group": "video_quality",
        "higher_is_better": True,
        "description": "Structural similarity from evaluation/FVD/cal_4metrics.py.",
    },
    "psnr": {
        "name": "PSNR",
        "group": "video_quality",
        "higher_is_better": True,
        "description": "Peak signal-to-noise ratio from evaluation/FVD/cal_4metrics.py.",
    },
    "lpips": {
        "name": "LPIPS",
        "group": "video_quality",
        "higher_is_better": False,
        "description": "Learned perceptual image patch similarity from evaluation/FVD/calculate_lpips.py.",
    },
}

OFFICIAL_METRIC_ALIASES = {
    "active_recognition_success_rate": "active_recognition_success_rate",
    "ar_success": "active_recognition_success_rate",
    "accuracy": "active_recognition_success_rate",
    "image_goal_navigation_success_rate": "image_goal_navigation_success_rate",
    "ignav_success": "image_goal_navigation_success_rate",
    "sr": "image_goal_navigation_success_rate",
    "success_rate": "image_goal_navigation_success_rate",
    "image_goal_navigation_spl": "image_goal_navigation_spl",
    "ignav_spl": "image_goal_navigation_spl",
    "spl": "image_goal_navigation_spl",
    "active_embodied_qa_score": "active_embodied_qa_score",
    "aeqa_score": "active_embodied_qa_score",
    "mean_score": "active_embodied_qa_score",
    "active_embodied_qa_spl": "active_embodied_qa_spl",
    "aeqa_spl": "active_embodied_qa_spl",
    "mean_efficiency": "active_embodied_qa_spl",
    "robotic_manipulation_success_rate": "robotic_manipulation_success_rate",
    "manipulation_success_rate": "robotic_manipulation_success_rate",
    "interaction_trace_consistency": "interaction_trace_consistency",
    "trace_consistency": "interaction_trace_consistency",
    "world_in_world_average": "world_in_world_average",
    "average": "world_in_world_average",
    "fvd": "fvd",
    "ssim": "ssim",
    "psnr": "psnr",
    "lpips": "lpips",
}

TASK_SUMMARY_FIELD_MAP: dict[str, dict[str, str]] = {
    "AR": {"accuracy": "active_recognition_success_rate"},
    "IGNav": {
        "sr": "image_goal_navigation_success_rate",
        "spl": "image_goal_navigation_spl",
    },
    "AEQA": {
        "mean_score": "active_embodied_qa_score",
        "mean_efficiency": "active_embodied_qa_spl",
    },
    "Manip": {
        "success_rate": "robotic_manipulation_success_rate",
        "sr": "robotic_manipulation_success_rate",
    },
}


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        for key in ("value", "score", "mean"):
            nested = _to_float(value.get(key))
            if nested is not None:
                return nested
        if "per_video" in value and isinstance(value["per_video"], list) and value["per_video"]:
            return _to_float(value["per_video"])
        return None
    if isinstance(value, (list, tuple)) and value:
        return _to_float(value[-1])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_metric_id(raw: str) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return OFFICIAL_METRIC_ALIASES.get(key) or OFFICIAL_METRIC_ALIASES.get(key.replace(" ", "_"))


def _normalize_unit(value: float, metric_id: str) -> float:
    if metric_id in {"active_embodied_qa_score", "active_embodied_qa_spl"} and value > 1.0:
        return value / 100.0
    if metric_id in {
        "image_goal_navigation_success_rate",
        "image_goal_navigation_spl",
        "active_recognition_success_rate",
    }:
        if value > 1.0:
            return value / 100.0
    if metric_id == "psnr" and value > 1.0:
        return value
    if metric_id in {"fvd", "lpips", "ssim"} and value > 1.0 and metric_id != "psnr":
        return value
    return value


def _compute_average(metrics: dict[str, float]) -> float | None:
    if metrics.get("world_in_world_average") is not None:
        return metrics["world_in_world_average"]
    component_values = [
        value
        for key, value in metrics.items()
        if key in METRIC_ORDER and key != "world_in_world_average" and value is not None
    ]
    return _mean(component_values)


def _summary_from_evaluator(payload: Mapping[str, Any], *, task: str | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        field_map = TASK_SUMMARY_FIELD_MAP.get(str(task or payload.get("task") or "").strip(), {})
        for raw_key, raw_value in summary.items():
            metric_id = field_map.get(str(raw_key)) or _canonical_metric_id(str(raw_key))
            score = _to_float(raw_value)
            if metric_id is None or score is None:
                continue
            metrics[metric_id] = _normalize_unit(score, metric_id)
    for raw_key, raw_value in payload.items():
        if raw_key in {"summary", "details", "task"}:
            continue
        metric_id = _canonical_metric_id(str(raw_key))
        score = _to_float(raw_value)
        if metric_id is None or score is None:
            continue
        metrics[metric_id] = _normalize_unit(score, metric_id)
    video_metrics = payload.get("video_metrics")
    if isinstance(video_metrics, Mapping):
        for raw_key, raw_value in video_metrics.items():
            metric_id = _canonical_metric_id(str(raw_key))
            score = _to_float(raw_value)
            if metric_id is None or score is None:
                continue
            metrics[metric_id] = score
    average = _compute_average(metrics)
    if average is not None:
        metrics["world_in_world_average"] = average
    return metrics


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        metric_id = _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or ""))
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if metric_id is None or score is None:
            continue
        buckets.setdefault(metric_id, []).append(_normalize_unit(score, metric_id))
    metrics = {metric_id: _mean(values) for metric_id, values in buckets.items() if values}
    average = _compute_average(metrics)
    if average is not None:
        metrics["world_in_world_average"] = average
    return metrics


def compute_world_in_world_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_path: Path | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    if len(rows) == 1 and isinstance(rows[0], Mapping):
        payload = rows[0]
        if isinstance(payload.get("summary"), Mapping) or any(_canonical_metric_id(str(key)) for key in payload.keys()):
            direct_metrics = _summary_from_evaluator(payload, task=task)
            video_quality = {
                metric_id: direct_metrics[metric_id]
                for metric_id in VIDEO_QUALITY_METRIC_IDS
                if metric_id in direct_metrics
            }
            return {
                "metrics": direct_metrics,
                "components": {
                    "format": "evaluator_summary_json",
                    "task": task or payload.get("task"),
                    "video_quality": video_quality,
                },
            }

    if rows and all(
        _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or ""))
        for row in rows[: min(3, len(rows))]
    ):
        direct_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    direct_metrics = _summary_metrics(rows)
    return {
        "metrics": direct_metrics,
        "components": {"row_count": len(rows), "format": "generic_rows"},
    }


def load_results_rows(results_path: Path) -> list[dict[str, Any]]:
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            return [dict(payload)]
        raise ValueError(f"Unsupported World-in-World JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
