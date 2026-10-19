"""AIGCBench metric normalization from official summary or per-sample exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean
from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float

METRIC_ORDER = (
    "mse_first",
    "ssim_first",
    "image_genvideo_clip",
    "genvideo_text_clip",
    "genvideo_refvideo_clip_keyframes",
    "flow_square_mean",
    "genvideo_refvideo_clip_corresponding_frames",
    "genvideo_clip_adjacent_frames",
    "frame_count",
    "dover",
    "genvideo_refvideo_ssim",
    "aigcbench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "mse_first": {
        "name": "MSE (First)",
        "group": "control_video_alignment",
        "higher_is_better": False,
        "description": "MSE between condition image and first generated frame.",
    },
    "ssim_first": {
        "name": "SSIM (First)",
        "group": "control_video_alignment",
        "higher_is_better": True,
        "description": "SSIM between condition image and first generated frame.",
    },
    "image_genvideo_clip": {
        "name": "Image-GenVideo CLIP",
        "group": "control_video_alignment",
        "higher_is_better": True,
        "description": "CLIP similarity between condition image and generated video.",
    },
    "genvideo_text_clip": {
        "name": "GenVideo-Text CLIP",
        "group": "control_video_alignment",
        "higher_is_better": True,
        "description": "CLIP similarity between generated video and text prompt.",
    },
    "genvideo_refvideo_clip_keyframes": {
        "name": "GenVideo-RefVideo CLIP (Keyframes)",
        "group": "control_video_alignment",
        "higher_is_better": True,
        "description": "CLIP similarity between generated and reference video keyframes.",
    },
    "flow_square_mean": {
        "name": "Flow-Square-Mean",
        "group": "motion_effects",
        "higher_is_better": True,
        "description": "Optical-flow square mean for generated video motion effects.",
    },
    "genvideo_refvideo_clip_corresponding_frames": {
        "name": "GenVideo-RefVideo CLIP (Corresponding frames)",
        "group": "motion_effects",
        "higher_is_better": True,
        "description": "CLIP similarity for generated and reference corresponding frames.",
    },
    "genvideo_clip_adjacent_frames": {
        "name": "GenVideo CLIP (Adjacent frames)",
        "group": "temporal_consistency",
        "higher_is_better": True,
        "description": "Adjacent-frame CLIP similarity for temporal consistency.",
    },
    "frame_count": {
        "name": "Frame Count",
        "group": "video_quality",
        "higher_is_better": True,
        "description": "Generated video frame count.",
    },
    "dover": {
        "name": "DOVER",
        "group": "video_quality",
        "higher_is_better": True,
        "description": "DOVER video quality score.",
    },
    "genvideo_refvideo_ssim": {
        "name": "GenVideo-RefVideo SSIM",
        "group": "video_quality",
        "higher_is_better": True,
        "description": "SSIM between generated and reference videos.",
    },
    "aigcbench_average": {
        "name": "AIGCBench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean over available AIGCBench leaf metrics.",
        "primary": True,
    },
}

METRIC_ALIASES = {
    "mse_first": "mse_first",
    "mse": "mse_first",
    "mse (first)": "mse_first",
    "ssim_first": "ssim_first",
    "ssim": "ssim_first",
    "ssim (first)": "ssim_first",
    "image_genvideo_clip": "image_genvideo_clip",
    "image-genvideo clip": "image_genvideo_clip",
    "image_genvideo_clip_per": "image_genvideo_clip",
    "clip_per": "image_genvideo_clip",
    "genvideo_text_clip": "genvideo_text_clip",
    "genvideo-text clip": "genvideo_text_clip",
    "genvideo_refvideo_clip_keyframes": "genvideo_refvideo_clip_keyframes",
    "genvideo-refvideo clip (keyframes)": "genvideo_refvideo_clip_keyframes",
    "genvideo_refvideo_clip_corresponding_frames": "genvideo_refvideo_clip_corresponding_frames",
    "genvideo-refvideo clip (corresponding frames)": "genvideo_refvideo_clip_corresponding_frames",
    "genvideo_clip_adjacent_frames": "genvideo_clip_adjacent_frames",
    "genvideo clip (adjacent frames)": "genvideo_clip_adjacent_frames",
    "flow_square_mean": "flow_square_mean",
    "flow-square-mean": "flow_square_mean",
    "frame_count": "frame_count",
    "frame count": "frame_count",
    "dover": "dover",
    "genvideo_refvideo_ssim": "genvideo_refvideo_ssim",
    "genvideo-refvideo ssim": "genvideo_refvideo_ssim",
    "aigcbench_average": "aigcbench_average",
    "average": "aigcbench_average",
    "clip": "image_genvideo_clip",
}


def _canonical_metric_id(raw: str) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return METRIC_ALIASES.get(key) or METRIC_ALIASES.get(key.replace(" ", "_"))


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
    if metrics.get("aigcbench_average") is None:
        component_values = [
            value for key, value in metrics.items() if key not in {"aigcbench_average"} and value is not None
        ]
        if component_values:
            metrics["aigcbench_average"] = sum(component_values) / len(component_values)
    return metrics


def _aggregate_sample_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in METRIC_ORDER if metric_id != "aigcbench_average"}
    for row in rows:
        for raw_key, raw_value in row.items():
            metric_id = _canonical_metric_id(str(raw_key))
            score = _to_float(raw_value)
            if metric_id is None or score is None or metric_id == "aigcbench_average":
                continue
            buckets.setdefault(metric_id, []).append(score)
    metrics = {metric_id: _mean(values) for metric_id, values in buckets.items() if values}
    component_values = [value for key, value in metrics.items() if value is not None]
    if component_values:
        metrics["aigcbench_average"] = sum(component_values) / len(component_values)
    else:
        metrics["aigcbench_average"] = None
    return metrics


def compute_aigcbench_metrics(*, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if _is_summary_rows(rows):
        direct_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    if len(rows) == 1 and isinstance(rows[0], Mapping):
        payload = rows[0]
        direct_metrics = {
            metric_id: score for metric_id in METRIC_ORDER if (score := _to_float(payload.get(metric_id))) is not None
        }
        if direct_metrics:
            component_values = [
                value for key, value in direct_metrics.items() if key != "aigcbench_average" and value is not None
            ]
            if direct_metrics.get("aigcbench_average") is None and component_values:
                direct_metrics["aigcbench_average"] = sum(component_values) / len(component_values)
            return {
                "metrics": direct_metrics,
                "components": {"format": "summary_json_object"},
            }

    direct_metrics = _aggregate_sample_rows(rows)
    return {
        "metrics": direct_metrics,
        "components": {
            "sample_count": len(rows),
            "format": "per_sample_rows",
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
        raise ValueError(f"Unsupported AIGCBench JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
