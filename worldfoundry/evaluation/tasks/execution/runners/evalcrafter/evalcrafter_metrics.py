"""EvalCrafter metric normalization from official final_result.txt exports."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

METRIC_ORDER = (
    "visual_quality",
    "text_video_alignment",
    "motion_quality",
    "temporal_consistency",
    "evalcrafter_total",
    "vqa_aesthetic",
    "vqa_technical",
    "inception_score",
    "clip_temp_score",
    "warping_error",
    "face_consistency_score",
    "action_score",
    "motion_ac_score",
    "flow_score",
    "clip_score",
    "blip_bleu",
    "sd_score",
    "detection_score",
    "color_score",
    "count_score",
    "ocr_error",
    "celebrity_id_error",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "visual_quality": {"name": "Visual Quality", "group": "aggregate", "higher_is_better": True},
    "text_video_alignment": {"name": "Text-Video Alignment", "group": "aggregate", "higher_is_better": True},
    "motion_quality": {"name": "Motion Quality", "group": "aggregate", "higher_is_better": True},
    "temporal_consistency": {"name": "Temporal Consistency", "group": "aggregate", "higher_is_better": True},
    "evalcrafter_total": {
        "name": "EvalCrafter Total",
        "group": "aggregate",
        "higher_is_better": True,
        "primary": True,
    },
    "vqa_aesthetic": {"name": "VQA Aesthetic", "group": "visual_quality", "higher_is_better": True},
    "vqa_technical": {"name": "VQA Technical", "group": "visual_quality", "higher_is_better": True},
    "inception_score": {"name": "Inception Score", "group": "visual_quality", "higher_is_better": True},
    "clip_temp_score": {"name": "CLIP Temporal Score", "group": "temporal_consistency", "higher_is_better": True},
    "warping_error": {"name": "Warping Error", "group": "temporal_consistency", "higher_is_better": False},
    "face_consistency_score": {
        "name": "Face Consistency Score",
        "group": "temporal_consistency",
        "higher_is_better": True,
    },
    "action_score": {"name": "Action Score", "group": "motion_quality", "higher_is_better": True},
    "motion_ac_score": {"name": "Motion AC Score", "group": "motion_quality", "higher_is_better": True},
    "flow_score": {"name": "Flow Score", "group": "motion_quality", "higher_is_better": True},
    "clip_score": {"name": "CLIP Score", "group": "text_video_alignment", "higher_is_better": True},
    "blip_bleu": {"name": "BLIP BLEU", "group": "text_video_alignment", "higher_is_better": True},
    "sd_score": {"name": "SD Score", "group": "text_video_alignment", "higher_is_better": True},
    "detection_score": {"name": "Detection Score", "group": "text_video_alignment", "higher_is_better": True},
    "color_score": {"name": "Color Score", "group": "text_video_alignment", "higher_is_better": True},
    "count_score": {"name": "Count Score", "group": "text_video_alignment", "higher_is_better": True},
    "ocr_error": {"name": "OCR Error", "group": "text_video_alignment", "higher_is_better": False},
    "celebrity_id_error": {"name": "Celebrity ID Error", "group": "text_video_alignment", "higher_is_better": False},
}

RAW_METRIC_MAP = {
    "VQA_A": "vqa_aesthetic",
    "VQA_T": "vqa_technical",
    "IS": "inception_score",
    "clip_temp_score": "clip_temp_score",
    "warping_error": "warping_error",
    "face_consistency_score": "face_consistency_score",
    "action_score": "action_score",
    "motion_ac_score": "motion_ac_score",
    "flow_score": "flow_score",
    "clip_score": "clip_score",
    "blip_bleu": "blip_bleu",
    "sd_score": "sd_score",
    "detection_score": "detection_score",
    "color_score": "color_score",
    "count_score": "count_score",
    "ocr_score": "ocr_error",
    "celebrity_id_score": "celebrity_id_error",
}

RESULT_PATTERNS = {
    "visual_quality": r"Visual Quality\s+([-+]?\d+(?:\.\d+)?)",
    "text_video_alignment": r"Text-Video Alignment\s+([-+]?\d+(?:\.\d+)?)",
    "motion_quality": r"Motion Quality\s+([-+]?\d+(?:\.\d+)?)",
    "temporal_consistency": r"Temporal Consistency\s+([-+]?\d+(?:\.\d+)?)",
    "evalcrafter_total": r"Total\s+([-+]?\d+(?:\.\d+)?)",
}

def scalar(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, str):
        try:
            return scalar(float(value.strip()))
        except ValueError:
            return None
    return None


def normalized_score(raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    if 1.0 < raw_score <= 100.0:
        return raw_score / 100.0
    return None


def parse_metrics_dict(text: str) -> dict[str, float | None]:
    match = re.search(r"Metrics:\s*(\{.*?\})", text, flags=re.DOTALL)
    if not match:
        return {}
    literal = re.sub(r"(?<=[:\[, ])nan(?=[,\]}])", "None", match.group(1))
    payload = ast.literal_eval(literal)
    if not isinstance(payload, dict):
        return {}
    return {str(key): scalar(value) for key, value in payload.items()}


def parse_result_summary(text: str) -> dict[str, float | None]:
    extracted: dict[str, float | None] = {}
    for metric_id, pattern in RESULT_PATTERNS.items():
        match = re.search(pattern, text)
        extracted[metric_id] = None if not match else scalar(match.group(1))
    return extracted


def load_upstream_results(results_path: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_runtime import latest_final_result

    final_result_path = latest_final_result(results_path)
    text = final_result_path.read_text(encoding="utf-8", errors="replace")
    extracted: dict[str, dict[str, Any]] = {}
    for raw_name, raw_score in parse_metrics_dict(text).items():
        metric_id = RAW_METRIC_MAP.get(raw_name)
        if metric_id:
            extracted[metric_id] = {
                "raw_score": raw_score,
                "source": f"metrics_dict.{raw_name}",
                "sample_count": None,
            }
    for metric_id, raw_score in parse_result_summary(text).items():
        extracted[metric_id] = {
            "raw_score": raw_score,
            "source": "results_summary",
            "sample_count": None,
        }
    return extracted, final_result_path
