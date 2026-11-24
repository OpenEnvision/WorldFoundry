"""Registration bindings for mapping custom local evaluators to benchmarks.

This module maps target benchmarks (e.g. vbench, camerabench, genai-bench) and
embodied task groups to their corresponding local formulas.
"""

from __future__ import annotations

from collections.abc import Mapping

MetricEvaluatorBinding = tuple[str, str, str, str]

FORMULA_EVALUATOR_BINDINGS: tuple[MetricEvaluatorBinding, ...] = (
    (
        "*",
        "pairwise_preference_accuracy",
        "pairwise_preference_accuracy",
        "Pairwise preference agreement from already judged rows.",
    ),
    ("vbench", "overall_quality", "vbench_final_score", "VBench final weighted score from official dimension scores."),
    (
        "vbench",
        "text_alignment",
        "vbench_final_score",
        "VBench semantic/text-alignment score from official dimension scores.",
    ),
    (
        "vbench-plus-plus",
        "vbench_plus_plus_i2v_average",
        "vbench_final_score",
        "VBench++ I2V final weighted score from official dimension scores.",
    ),
    (
        "camerabench",
        "camera_motion_average_precision",
        "camera_binary_classification",
        "CameraBench binary AP from score records.",
    ),
    (
        "camerabench",
        "camera_motion_roc_auc",
        "camera_binary_classification",
        "CameraBench binary ROC-AUC from score records.",
    ),
    (
        "camerabench",
        "camera_vqa_accuracy",
        "camera_vqa",
        "CameraBench VQA question accuracy from yes/no score quads.",
    ),
    (
        "camerabench",
        "camera_retrieval_accuracy",
        "camera_retrieval",
        "CameraBench retrieval group accuracy from score quads.",
    ),
    (
        "chronomagic-bench",
        "chronomagic_score",
        "chronomagic_average_chscore",
        "ChronoMagic CHScore average from official per-video CHScore JSON records.",
    ),
    (
        "genai-bench",
        "pairwise_accuracy",
        "genai_bench_accuracy",
        "GenAI-Bench pairwise judge accuracy from existing correct/incorrect rows.",
    ),
    (
        "genai-bench",
        "image_generation_preference_accuracy",
        "genai_bench_accuracy",
        "GenAI-Bench image-generation preference accuracy from existing correct/incorrect rows.",
    ),
    (
        "genai-bench",
        "image_editing_preference_accuracy",
        "genai_bench_accuracy",
        "GenAI-Bench image-editing preference accuracy from existing correct/incorrect rows.",
    ),
    (
        "genai-bench",
        "video_preference_accuracy",
        "genai_bench_accuracy",
        "GenAI-Bench video-generation preference accuracy from existing correct/incorrect rows.",
    ),
    (
        "genai-bench",
        "genai_bench_average",
        "genai_bench_accuracy",
        "GenAI-Bench average pairwise judge accuracy from existing correct/incorrect rows.",
    ),
    (
        "videoscore",
        "visual_quality",
        "videoscore_spearman",
        "VideoScore visual-quality Spearman correlation from existing ref/ans score vectors.",
    ),
    (
        "videoscore",
        "temporal_consistency",
        "videoscore_spearman",
        "VideoScore temporal-consistency Spearman correlation from existing ref/ans score vectors.",
    ),
    (
        "videoscore",
        "dynamic_degree",
        "videoscore_spearman",
        "VideoScore dynamic-degree Spearman correlation from existing ref/ans score vectors.",
    ),
    (
        "videoscore",
        "text_to_video_alignment",
        "videoscore_spearman",
        "VideoScore text-to-video-alignment Spearman correlation from existing ref/ans score vectors.",
    ),
    (
        "videoscore",
        "factual_consistency",
        "videoscore_spearman",
        "VideoScore factual-consistency Spearman correlation from existing ref/ans score vectors.",
    ),
    (
        "videoscore",
        "videoscore_average",
        "videoscore_spearman",
        "VideoScore average Spearman correlation from existing ref/ans score vectors.",
    ),
    (
        "videoverse",
        "qa_accuracy",
        "videoverse_subquestion_accuracy",
        "VideoVerse sub-question accuracy from judge result JSON.",
    ),
    (
        "videoverse",
        "videoverse_average",
        "videoverse_subquestion_accuracy",
        "VideoVerse video-level pass rate from judge result JSON.",
    ),
    (
        "video-bench",
        "mcqa_accuracy",
        "mcqa_accuracy",
        "Video-Bench multiple-choice accuracy from predictions and answers.",
    ),
    ("ipv-bench", "mcqa_accuracy", "mcqa_accuracy", "IPV-Bench multiple-choice accuracy from predictions and answers."),
    (
        "worldmodelbench",
        "instruction_following",
        "worldmodelbench_result_scores",
        "WorldModelBench category score aggregation.",
    ),
    (
        "worldmodelbench",
        "common_sense",
        "worldmodelbench_result_scores",
        "WorldModelBench category score aggregation.",
    ),
    (
        "worldmodelbench",
        "physical_adherence",
        "worldmodelbench_result_scores",
        "WorldModelBench category score aggregation.",
    ),
    (
        "worldmodelbench",
        "world_model_average",
        "worldmodelbench_result_scores",
        "WorldModelBench total score aggregation.",
    ),
    (
        "*",
        "jedi_score",
        "jedi_mmd",
        "VideoJEDi MMD score from precomputed V-JEPA train/test feature arrays.",
    ),
)

SUCCESS_METRIC_IDS_BY_BENCHMARK: Mapping[str, tuple[str, ...]] = {
    "robotwin": (
        "success_rate",
        "task_success",
        "episode_success",
        "bimanual_task_success",
        "domain_randomization_success",
        "clean_success",
        "randomized_success",
    ),
    "calvin": ("success_rate", "sequence_success", "task_success", "completion"),
    "metaworld": ("success_rate", "task_success"),
    "bridgedata-v2": ("success_rate", "task_success", "goal_success", "rollout_success"),
    "libero": ("success_rate", "task_success", "episode_success"),
    "simpler-env": ("success_rate", "task_success", "episode_success"),
    "robocasa": ("success_rate", "task_success", "episode_success"),
    "maniskill": ("success_rate", "task_success"),
    "rlbench": ("success_rate", "task_success", "episode_success"),
}


def success_metric_bindings() -> tuple[tuple[str, str], ...]:
    """Compile and return list of all benchmark ID and metric ID success bindings.

    Returns:
        Tuple of (benchmark_id, metric_id) string pairs.
    """
    return tuple(
        (benchmark_id, metric_id)
        for benchmark_id, metric_ids in SUCCESS_METRIC_IDS_BY_BENCHMARK.items()
        for metric_id in metric_ids
    )


__all__ = [
    "FORMULA_EVALUATOR_BINDINGS",
    "SUCCESS_METRIC_IDS_BY_BENCHMARK",
    "MetricEvaluatorBinding",
    "success_metric_bindings",
]
