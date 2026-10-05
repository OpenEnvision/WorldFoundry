"""Single video-quality contract table.

Lookups go through :func:`video_quality_spec`. YAML entrypoints bind onto
:func:`evaluate_video_quality_contract_file` in ``video_quality_contract``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.lazy_registry import (
    normalize_registry_key,
    supported_ids,
)

_BLOCKED_REASON = {
    "aigcbench": "asset_required",
    "fetv": "asset_required",
    "mirabench": "asset_required",
    "genai-bench": "judge_required",
    "ipv-bench": "judge_required",
}

_BLOCKED_DETAIL = {
    "aigcbench": (
        "official AIGCBench scorer outputs are required; this evaluator only "
        "normalizes existing official result files"
    ),
    "fetv": (
        "official FETV metric or annotation outputs are required; this evaluator "
        "only normalizes existing official result files"
    ),
    "mirabench": (
        "official MiraBench scorer outputs are required; this evaluator only "
        "normalizes existing official result files"
    ),
    "genai-bench": (
        "human preference labels and model or judge predictions are required for "
        "pairwise preference accuracy"
    ),
    "ipv-bench": (
        "human annotation, official judge outputs, or official understanding "
        "predictions are required for IPV-Bench metrics"
    ),
}

_REQUIRED_INPUTS = {
    "aigcbench": (
        "official_results_path or official_results",
        "metric_id/name plus score/value columns",
        "generated_video",
        "prompt_id",
        "prompt",
    ),
    "mirabench": (
        "official_results_path or official_results",
        "metric_id/name plus score/value columns",
        "generated_video",
        "prompt_id",
        "prompt",
        "category or prompt_type for per-category aggregation",
    ),
    "fetv": (
        "official_results_path or official_results",
        "metric_id/name plus score/value columns",
        "generated_video",
        "prompt_id",
        "prompt",
    ),
    "genai-bench": (
        "human_label or human_preference",
        "prediction or judge_prediction",
        "task",
        "generated_video for video_generation rows",
    ),
    "ipv-bench": (
        "visual_quality annotation",
        "prompt_following annotation",
        "judgement/mcqa/open_qa labels or official judge scores",
        "prediction or model_answer for understanding tasks",
    ),
}

_AGGREGATE_COMPONENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "aigcbench": {
        "control_video_alignment": (
            "mse_first",
            "ssim_first",
            "image_genvideo_clip",
            "genvideo_text_clip",
            "genvideo_refvideo_clip_keyframes",
        ),
        "motion_effects": (
            "flow_square_mean",
            "genvideo_refvideo_clip_corresponding_frames",
        ),
        "temporal_consistency": (
            "genvideo_clip_adjacent_frames",
            "genvideo_refvideo_clip_corresponding_frames",
        ),
        "video_quality": (
            "frame_count",
            "dover",
            "genvideo_refvideo_ssim",
        ),
        "aigcbench_average": (
            "control_video_alignment",
            "motion_effects",
            "temporal_consistency",
            "video_quality",
        ),
    },
    "mirabench": {
        "motion_strength": (
            "dynamic_degree",
            "tracking_strength",
        ),
        "temporal_consistency": (
            "dino_temporal_consistency",
            "clip_temporal_consistency",
            "temporal_motion_smoothness",
        ),
        "three_d_consistency": (
            "mean_absolute_error",
            "root_mean_square_error",
        ),
        "visual_quality": (
            "aesthetic_quality",
            "imaging_quality",
        ),
        "text_video_alignment": (
            "camera_alignment",
            "main_object_alignment",
            "background_alignment",
            "style_alignment",
            "overall_alignment",
        ),
        "distribution_similarity": (
            "fvd",
            "fid",
            "kid",
        ),
        "mirabench_average": (
            "motion_strength",
            "temporal_consistency",
            "three_d_consistency",
            "visual_quality",
            "text_video_alignment",
            "distribution_similarity",
        ),
    },
    "fetv": {
        "fetv_average": (
            "static_quality",
            "temporal_quality",
            "overall_alignment",
            "fine_grained_alignment",
            "clip_score",
            "blip_score",
            "fid",
            "fvd",
        ),
    },
    "ipv-bench": {
        "ipv_bench_average": (
            "visual_quality",
            "prompt_following",
            "impossible_video_score",
            "judgement_accuracy",
            "mcqa_accuracy",
            "open_qa_score",
        ),
    },
}

_METRIC_ALIASES: dict[str, dict[str, str]] = {
    "aigcbench": {
        "controlvideoalignment": "control_video_alignment",
        "control_video_alignment": "control_video_alignment",
        "imagevideoalignment": "control_video_alignment",
        "image_video_alignment": "control_video_alignment",
        "motioneffects": "motion_effects",
        "motion_effects": "motion_effects",
        "temporalconsistency": "temporal_consistency",
        "temporal_consistency": "temporal_consistency",
        "videoquality": "video_quality",
        "video_quality": "video_quality",
        "msefirst": "mse_first",
        "mse_first": "mse_first",
        "ssimfirst": "ssim_first",
        "ssim_first": "ssim_first",
        "imagegenvideoclip": "image_genvideo_clip",
        "image_genvideo_clip": "image_genvideo_clip",
        "genvideotextclip": "genvideo_text_clip",
        "genvideo_text_clip": "genvideo_text_clip",
        "genvideorefvideoclipkeyframes": "genvideo_refvideo_clip_keyframes",
        "genvideo_refvideo_clip_keyframes": "genvideo_refvideo_clip_keyframes",
        "flowsquaremean": "flow_square_mean",
        "flow_square_mean": "flow_square_mean",
        "genvideorefvideoclipcorrespondingframes": "genvideo_refvideo_clip_corresponding_frames",
        "genvideo_refvideo_clip_corresponding_frames": "genvideo_refvideo_clip_corresponding_frames",
        "genvideoclipadjacentframes": "genvideo_clip_adjacent_frames",
        "genvideo_clip_adjacent_frames": "genvideo_clip_adjacent_frames",
        "framecount": "frame_count",
        "frame_count": "frame_count",
        "dover": "dover",
        "genvideorefvideossim": "genvideo_refvideo_ssim",
        "genvideo_refvideo_ssim": "genvideo_refvideo_ssim",
        "aigcbenchaverage": "aigcbench_average",
        "aigcbench_average": "aigcbench_average",
    },
    "mirabench": {
        "dd": "dynamic_degree",
        "dynamicdegree": "dynamic_degree",
        "dynamic_degree": "dynamic_degree",
        "ts": "tracking_strength",
        "trackingstrength": "tracking_strength",
        "tracking_strength": "tracking_strength",
        "dtc": "dino_temporal_consistency",
        "dinotemporalconsistency": "dino_temporal_consistency",
        "dino_temporal_consistency": "dino_temporal_consistency",
        "ctc": "clip_temporal_consistency",
        "cliptemporalconsistency": "clip_temporal_consistency",
        "clip_temporal_consistency": "clip_temporal_consistency",
        "tms": "temporal_motion_smoothness",
        "temporalmotionsmoothness": "temporal_motion_smoothness",
        "temporal_motion_smoothness": "temporal_motion_smoothness",
        "mae": "mean_absolute_error",
        "meanabsoluteerror": "mean_absolute_error",
        "mean_absolute_error": "mean_absolute_error",
        "rmse": "root_mean_square_error",
        "rootmeansquareerror": "root_mean_square_error",
        "root_mean_square_error": "root_mean_square_error",
        "aq": "aesthetic_quality",
        "aestheticquality": "aesthetic_quality",
        "aesthetic_quality": "aesthetic_quality",
        "iq": "imaging_quality",
        "imaging_quality": "imaging_quality",
        "imaging_quality": "imaging_quality",
        "ca": "camera_alignment",
        "cameraalignment": "camera_alignment",
        "camera_alignment": "camera_alignment",
        "moa": "main_object_alignment",
        "mainobjectalignment": "main_object_alignment",
        "main_object_alignment": "main_object_alignment",
        "ba": "background_alignment",
        "backgroundalignment": "background_alignment",
        "background_alignment": "background_alignment",
        "sa": "style_alignment",
        "stylealignment": "style_alignment",
        "style_alignment": "style_alignment",
        "oa": "overall_alignment",
        "overallalignment": "overall_alignment",
        "overall_alignment": "overall_alignment",
        "fvd": "fvd",
        "fid": "fid",
        "kid": "kid",
        "distributionconsistency": "distribution_similarity",
        "distribution_consistency": "distribution_similarity",
        "distributionsimilarity": "distribution_similarity",
        "distribution_similarity": "distribution_similarity",
        "mirabenchaverage": "mirabench_average",
        "mirabench_average": "mirabench_average",
    },
    "fetv": {
        "staticquality": "static_quality",
        "static_quality": "static_quality",
        "temporalquality": "temporal_quality",
        "temporal_quality": "temporal_quality",
        "overallalignment": "overall_alignment",
        "overall_alignment": "overall_alignment",
        "finegrainedalignment": "fine_grained_alignment",
        "fine_grained_alignment": "fine_grained_alignment",
        "clipscore": "clip_score",
        "clip_score": "clip_score",
        "blipscore": "blip_score",
        "blip_score": "blip_score",
        "fid": "fid",
        "fvd": "fvd",
        "fetvaverage": "fetv_average",
        "fetv_average": "fetv_average",
    },
    "ipv-bench": {
        "visualquality": "visual_quality",
        "visual_quality": "visual_quality",
        "promptfollowing": "prompt_following",
        "prompt_following": "prompt_following",
        "ipvscore": "impossible_video_score",
        "impossible_video_score": "impossible_video_score",
        "impossiblevideoscore": "impossible_video_score",
        "judgementaccuracy": "judgement_accuracy",
        "judgement_accuracy": "judgement_accuracy",
        "judgmentaccuracy": "judgement_accuracy",
        "judgment_accuracy": "judgement_accuracy",
        "mcqaaccuracy": "mcqa_accuracy",
        "mcqa_accuracy": "mcqa_accuracy",
        "openqascore": "open_qa_score",
        "open_qa_score": "open_qa_score",
        "openqa_score": "open_qa_score",
        "ipvbenchaverage": "ipv_bench_average",
        "ipv_bench_average": "ipv_bench_average",
    },
}

GENAI_TASK_METRICS = {
    "image_generation": "image_generation_preference_accuracy",
    "image_editing": "image_editing_preference_accuracy",
    "video_generation": "video_preference_accuracy",
}


def supported_video_quality_benchmark_ids() -> tuple[str, ...]:
    from worldfoundry.evaluation.tasks.catalog.dispatch import scoring_benchmark_ids

    catalog_ids = scoring_benchmark_ids("video_quality")
    return catalog_ids if catalog_ids else supported_ids(_BLOCKED_REASON)


@lru_cache(maxsize=None)
def video_quality_spec(benchmark_id: str) -> dict[str, Any]:
    key = normalize_registry_key(benchmark_id)
    if key not in _BLOCKED_REASON:
        known = ", ".join(supported_video_quality_benchmark_ids())
        raise KeyError(f"unsupported video-quality benchmark {benchmark_id!r}; known: {known}")
    spec: dict[str, Any] = {
        "benchmark_id": key,
        "blocked_reason": _BLOCKED_REASON[key],
        "blocked_detail": _BLOCKED_DETAIL[key],
        "required_inputs": _REQUIRED_INPUTS[key],
        "aggregate_components": _AGGREGATE_COMPONENTS.get(key, {}),
        "metric_aliases": _METRIC_ALIASES.get(key, {}),
    }
    if key == "genai-bench":
        spec["genai_task_metrics"] = dict(GENAI_TASK_METRICS)
    return spec
