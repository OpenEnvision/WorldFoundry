"""Mathematical formulas and metrics aggregation algorithms for standard benchmarks.

This module implements scoring algorithms, accuracy computations, retrieval metrics,
classification metrics, and averages for various benchmarks such as CameraBench,
VBench, VideoScore, and multiple choice questions.
"""

from __future__ import annotations

import ast
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonValue = Any

_VBENCH_TASK_INFO = (
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
)
_VBENCH_QUALITY_LIST = (
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
)
_VBENCH_SEMANTIC_LIST = (
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
)
_VBENCH_DIM_WEIGHT = {
    "subject consistency": 1.0,
    "background consistency": 1.0,
    "temporal flickering": 1.0,
    "motion smoothness": 1.0,
    "dynamic degree": 0.5,
    "aesthetic quality": 1.0,
    "imaging quality": 1.0,
    "object class": 1.0,
    "multiple objects": 1.0,
    "human action": 1.0,
    "color": 1.0,
    "spatial relationship": 1.0,
    "scene": 1.0,
    "appearance style": 1.0,
    "temporal style": 1.0,
    "overall consistency": 1.0,
}
_VBENCH_NORMALIZE = {
    "subject consistency": (0.1462, 1.0),
    "background consistency": (0.2615, 1.0),
    "temporal flickering": (0.6293, 1.0),
    "motion smoothness": (0.706, 0.9975),
    "dynamic degree": (0.0, 1.0),
    "aesthetic quality": (0.0, 1.0),
    "imaging quality": (0.0, 1.0),
    "object class": (0.0, 1.0),
    "multiple objects": (0.0, 1.0),
    "human action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance style": (0.0009, 0.2855),
    "temporal style": (0.0, 0.364),
    "overall consistency": (0.0, 0.364),
}

_VBENCH_I2V_TASK_INFO = (
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
)
_VBENCH_I2V_LIST = (
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
)
_VBENCH_I2V_QUALITY_LIST = (
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
)
_VBENCH_I2V_DIM_WEIGHT = {
    "Video-Text Camera Motion": 0.1,
    "Video-Image Subject Consistency": 1.0,
    "Video-Image Background Consistency": 1.0,
    "Subject Consistency": 1.0,
    "Background Consistency": 1.0,
    "Motion Smoothness": 1.0,
    "Dynamic Degree": 0.5,
    "Aesthetic Quality": 1.0,
    "Imaging Quality": 1.0,
}
_VBENCH_I2V_NORMALIZE = {
    "Video-Text Camera Motion": (0.0, 1.0),
    "Video-Image Subject Consistency": (0.1462, 1.0),
    "Video-Image Background Consistency": (0.2615, 1.0),
    "Subject Consistency": (0.1462, 1.0),
    "Background Consistency": (0.2615, 1.0),
    "Motion Smoothness": (0.7060, 0.9975),
    "Dynamic Degree": (0.0, 1.0),
    "Aesthetic Quality": (0.0, 1.0),
    "Imaging Quality": (0.0, 1.0),
}
_VBENCH_I2V_ALIASES = {
    "camera_motion": "Video-Text Camera Motion",
    "i2v_subject": "Video-Image Subject Consistency",
    "i2v_background": "Video-Image Background Consistency",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
}


def boolean_accuracy(
    records: Sequence[Mapping[str, JsonValue]],
    *,
    value_key: str = "correct",
) -> dict[str, JsonValue]:
    """Compute accuracy from rows that already contain boolean correctness labels."""

    correct = 0
    total = 0
    per_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        if value_key not in record:
            continue
        value = _truthy_label(record.get(value_key))
        if value is None:
            continue
        correct += int(value)
        total += 1
        task = record.get("task", record.get("category", record.get("split")))
        if task:
            bucket = per_task[str(task)]
            bucket[0] += int(value)
            bucket[1] += 1
    return {
        "accuracy": _safe_div(correct, total),
        "num_correct": correct,
        "num_total": total,
        "per_task": _ratio_counts(per_task),
    }


def camera_binary_classification_metrics(records: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Compute CameraBench-style AP/AUC from yes/no score records.

    Upstream source checked: ``t2v_metrics/camerabench/binary_classification_evaluation.py``.
    WorldFoundry keeps only the score/label math and does not copy plotting or sklearn dependency code.
    """

    scores: list[float] = []
    labels: list[int] = []
    for record in records:
        if record.get("error") is not None:
            continue
        score = _finite_score(record.get("score"))
        label = _binary_label(record.get("ground_truth_label", record.get("label", record.get("answer"))))
        if label is None:
            continue
        scores.append(score)
        labels.append(label)

    num_samples = len(scores)
    num_positive = sum(1 for label in labels if label == 1)
    num_negative = num_samples - num_positive
    if num_samples == 0 or num_positive == 0 or num_negative == 0:
        average_precision = 0.0
        roc_auc = 0.0
    else:
        average_precision = _average_precision(scores, labels)
        roc_auc = _roc_auc(scores, labels)
    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "num_samples": num_samples,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "positive_ratio": (num_positive / num_samples) if num_samples else 0.0,
    }


def camera_vqa_metrics(records: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Compute CameraBench VQA binary and question accuracy from yes/no score quads."""

    binary_correct = 0
    question_correct = 0
    total_binary = 0
    total_questions = 0
    num_samples = 0
    for record in records:
        pair = _camera_score_pair(record)
        if pair is None:
            continue
        yes_scores, no_scores = pair
        binary_correct += int(yes_scores["pos_text_pos_image"] > no_scores["pos_text_pos_image"])
        binary_correct += int(no_scores["pos_text_neg_image"] > yes_scores["pos_text_neg_image"])
        binary_correct += int(no_scores["neg_text_pos_image"] > yes_scores["neg_text_pos_image"])
        binary_correct += int(yes_scores["neg_text_neg_image"] > no_scores["neg_text_neg_image"])
        total_binary += 4

        pos_question_correct = (
            yes_scores["pos_text_pos_image"] > no_scores["pos_text_pos_image"]
            and no_scores["pos_text_neg_image"] > yes_scores["pos_text_neg_image"]
        )
        neg_question_correct = (
            no_scores["neg_text_pos_image"] > yes_scores["neg_text_pos_image"]
            and yes_scores["neg_text_neg_image"] > no_scores["neg_text_neg_image"]
        )
        question_correct += int(pos_question_correct) + int(neg_question_correct)
        total_questions += 2
        num_samples += 1

    return {
        "binary_acc": (binary_correct / total_binary) if total_binary else 0.0,
        "question_acc": (question_correct / total_questions) if total_questions else 0.0,
        "num_samples": num_samples,
    }


def camera_retrieval_metrics(records: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Compute CameraBench text/image/group retrieval accuracy from score quads."""

    text_correct = 0
    image_correct = 0
    group_correct = 0
    total = 0
    for record in records:
        scores = _retrieval_score_dict(record)
        if scores is None:
            continue
        text_ok = (
            scores["pos_text_pos_image"] > scores["neg_text_pos_image"]
            and scores["neg_text_neg_image"] > scores["pos_text_neg_image"]
        )
        image_ok = (
            scores["pos_text_pos_image"] > scores["pos_text_neg_image"]
            and scores["neg_text_neg_image"] > scores["neg_text_pos_image"]
        )
        text_correct += int(text_ok)
        image_correct += int(image_ok)
        group_correct += int(text_ok and image_ok)
        total += 1
    return {
        "text": (text_correct / total) if total else 0.0,
        "image": (image_correct / total) if total else 0.0,
        "group": (group_correct / total) if total else 0.0,
        "num_samples": total,
    }


def videoverse_subquestion_metrics(data: Mapping[str, Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Compute VideoVerse sub-question, check-level, and video-level accuracy."""

    total_sub_questions = 0
    yes_count = 0
    no_count = 0
    wrong_count = 0
    total_checks = 0
    passed_checks = 0
    total_videos = 0
    passed_videos = 0
    type_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "sub_total": 0,
            "sub_yes": 0,
            "sub_no": 0,
            "checks_total": 0,
            "checks_pass": 0,
        }
    )

    for video_record in data.values():
        if not isinstance(video_record, Mapping):
            continue
        total_videos += 1
        video_pass = True
        checks = video_record.get("verification_checks", ())
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            checks = ()
        for check in checks:
            if not isinstance(check, Mapping) or "sub_question_results" not in check:
                continue
            check_type = str(check.get("check_type") or "Unknown")
            sub_results = check.get("sub_question_results") or ()
            total_checks += 1
            type_stats[check_type]["checks_total"] += 1
            check_pass = True
            if not isinstance(sub_results, Sequence) or isinstance(sub_results, (str, bytes)):
                sub_results = ()
            for item in sub_results:
                if not isinstance(item, Mapping):
                    continue
                result = str(item.get("res") or "").strip().lower()
                total_sub_questions += 1
                type_stats[check_type]["sub_total"] += 1
                if result == "yes":
                    yes_count += 1
                    type_stats[check_type]["sub_yes"] += 1
                elif result == "no":
                    no_count += 1
                    type_stats[check_type]["sub_no"] += 1
                    check_pass = False
                else:
                    wrong_count += 1
                    check_pass = False
            if check_pass:
                passed_checks += 1
                type_stats[check_type]["checks_pass"] += 1
            else:
                video_pass = False
        if video_pass:
            passed_videos += 1

    per_check_type = {
        check_type: {
            **stats,
            "sub_question_accuracy": _safe_div(stats["sub_yes"], stats["sub_yes"] + stats["sub_no"]),
            "check_accuracy": _safe_div(stats["checks_pass"], stats["checks_total"]),
        }
        for check_type, stats in type_stats.items()
    }
    return {
        "sub_question_accuracy": _safe_div(yes_count, yes_count + no_count),
        "check_accuracy": _safe_div(passed_checks, total_checks),
        "video_accuracy": _safe_div(passed_videos, total_videos),
        "total_sub_questions": total_sub_questions,
        "yes": yes_count,
        "no": no_count,
        "wrong": wrong_count,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "total_videos": total_videos,
        "passed_videos": passed_videos,
        "per_check_type": per_check_type,
    }


def multiple_choice_accuracy(
    predictions: Mapping[str, Mapping[str, JsonValue]] | Sequence[Mapping[str, JsonValue]],
    answers: Mapping[str, Mapping[str, JsonValue]] | Sequence[Mapping[str, JsonValue]] | None = None,
    *,
    taxonomy_by_video: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> dict[str, JsonValue]:
    """Compute IPV/Video-Bench-style multiple-choice accuracy from prediction rows."""

    rows = _join_prediction_answer_rows(predictions, answers)
    correct = 0
    total = 0
    taxonomy_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    spatial_temporal_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for row in rows:
        pred = _choice_letter(row.get("pred", row.get("prediction", row.get("model_answer"))))
        answer = _choice_letter(row.get("answer", row.get("label", row.get("gt"))))
        if not pred or not answer:
            continue
        is_correct = pred == answer
        correct += int(is_correct)
        total += 1
        video_name = row.get("video_name") or row.get("video_id")
        tax = taxonomy_by_video.get(str(video_name), {}) if taxonomy_by_video and video_name is not None else {}
        for label in tax.get("taxonomy_label_list", ()) if isinstance(tax.get("taxonomy_label_list"), Sequence) else ():
            bucket = taxonomy_counts[str(label)]
            bucket[0] += int(is_correct)
            bucket[1] += 1
        spatial_temporal_label = tax.get("spatial_temporal_label")
        if spatial_temporal_label:
            bucket = spatial_temporal_counts[str(spatial_temporal_label)]
            bucket[0] += int(is_correct)
            bucket[1] += 1

    return {
        "accuracy": _safe_div(correct, total),
        "num_correct": correct,
        "num_total": total,
        "per_taxonomy": _ratio_counts(taxonomy_counts),
        "per_spatial_temporal": _ratio_counts(spatial_temporal_counts),
    }


def chronomagic_average_scores(
    records: Mapping[str, Mapping[str, JsonValue]] | Sequence[Mapping[str, JsonValue]],
    *,
    score_key: str,
    suffix: str,
) -> dict[str, dict[str, float]]:
    """Merge ChronoMagic per-video CHScore/MTScore JSON records by filename prefix."""

    grouped: dict[str, list[float]] = defaultdict(list)
    pattern = re.compile(rf"^(.*)_.*_{re.escape(suffix)}\.json$")
    iterable = records.items() if isinstance(records, Mapping) else ((_record_name(item), item) for item in records)
    for name, payload in iterable:
        match = pattern.match(Path(str(name)).name)
        if not match or not isinstance(payload, Mapping):
            continue
        value = payload.get(score_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            grouped[match.group(1)].append(float(value))
    output_key = f"Average_{suffix}"
    return {prefix: {output_key: sum(values) / len(values)} for prefix, values in sorted(grouped.items()) if values}


def worldmodelbench_score(
    accs: Mapping[str, Sequence[JsonValue]],
    *,
    num_instances: int,
) -> dict[str, JsonValue]:
    """Aggregate WorldModelBench judge scores with the upstream category grouping."""

    category_mapping = {
        2: ("framewise", "temporal"),
        5: ("newton", "mass", "fluid", "penetration", "gravity"),
    }
    categories: dict[str, JsonValue] = {}
    total_score = 0.0
    for category, values in accs.items():
        scores = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not scores:
            categories[str(category)] = {"overall": 0.0, "sub_scores": {}, "num_scores": 0}
            continue
        if num_instances <= 0 or len(scores) % num_instances != 0:
            raise ValueError("num_instances must divide every WorldModelBench category score count.")
        num_sub = len(scores) // num_instances
        if num_sub == 1:
            overall = sum(scores) / len(scores)
            sub_scores: dict[str, float] = {}
        elif num_sub in category_mapping:
            sub_scores = {
                sub_name: sum(scores[index::num_sub]) / len(scores[index::num_sub])
                for index, sub_name in enumerate(category_mapping[num_sub])
            }
            overall = sum(sub_scores.values())
        else:
            raise ValueError(f"Unexpected number of WorldModelBench subcategories: {num_sub}")
        total_score += overall
        categories[str(category)] = {
            "overall": overall,
            "sub_scores": sub_scores,
            "num_scores": len(scores),
            "num_subcategories": num_sub,
        }
    return {"total_score": total_score, "categories": categories}


def vbench_final_score(scores: Mapping[str, JsonValue], *, i2v: bool = False) -> dict[str, JsonValue]:
    """Aggregate already-computed VBench dimension scores into official final scores."""

    if i2v:
        normalized = _weighted_normalized_scores(
            _normalize_score_keys(scores, aliases=_VBENCH_I2V_ALIASES),
            task_names=_VBENCH_I2V_TASK_INFO,
            normalize_ranges=_VBENCH_I2V_NORMALIZE,
            dim_weights=_VBENCH_I2V_DIM_WEIGHT,
        )
        quality_score = _weighted_group_mean(normalized, _VBENCH_I2V_QUALITY_LIST, _VBENCH_I2V_DIM_WEIGHT)
        i2v_score = _weighted_group_mean(normalized, _VBENCH_I2V_LIST, _VBENCH_I2V_DIM_WEIGHT)
        final_score = (quality_score + i2v_score) / 2.0
        return {
            "final_score": final_score,
            "quality_score": quality_score,
            "i2v_score": i2v_score,
            "normalized_scores": normalized,
        }

    normalized = _weighted_normalized_scores(
        _normalize_score_keys(scores),
        task_names=_VBENCH_TASK_INFO,
        normalize_ranges=_VBENCH_NORMALIZE,
        dim_weights=_VBENCH_DIM_WEIGHT,
    )
    quality_score = _weighted_group_mean(normalized, _VBENCH_QUALITY_LIST, _VBENCH_DIM_WEIGHT)
    semantic_score = _weighted_group_mean(normalized, _VBENCH_SEMANTIC_LIST, _VBENCH_DIM_WEIGHT)
    final_score = (quality_score * 4.0 + semantic_score) / 5.0
    return {
        "final_score": final_score,
        "quality_score": quality_score,
        "semantic_score": semantic_score,
        "normalized_scores": normalized,
    }


def pairwise_preference_accuracy(records: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Compute agreement with pairwise preference labels from already judged rows."""

    correct = 0
    total = 0
    ties = 0
    per_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        prediction = _preference_label(record.get("prediction", record.get("pred", record.get("winner"))))
        reference = _preference_label(record.get("reference", record.get("label", record.get("preferred"))))
        if not prediction or not reference:
            score_a = record.get("score_a", record.get("a_score"))
            score_b = record.get("score_b", record.get("b_score"))
            if prediction is None:
                prediction = _preference_from_scores(score_a, score_b)
            if reference is None:
                reference = _preference_label(record.get("human_preference", record.get("gt")))
        if not prediction or not reference:
            continue
        if prediction == "tie" or reference == "tie":
            ties += 1
        is_correct = prediction == reference
        correct += int(is_correct)
        total += 1
        category = record.get("category", record.get("source", record.get("benchmark")))
        if category:
            bucket = per_category[str(category)]
            bucket[0] += int(is_correct)
            bucket[1] += 1
    return {
        "accuracy": _safe_div(correct, total),
        "num_correct": correct,
        "num_total": total,
        "num_ties": ties,
        "per_category": _ratio_counts(per_category),
    }


def score_vector_spearman(
    records: Sequence[Mapping[str, JsonValue]],
    *,
    ref_key: str = "ref",
    ans_key: str = "ans",
    scale: float = 100.0,
) -> dict[str, JsonValue]:
    """Compute VideoScore-style per-aspect Spearman correlations from score vectors."""

    ref_vectors: list[list[float]] = []
    ans_vectors: list[list[float]] = []
    for record in records:
        ref_vector = _score_vector(record.get(ref_key, record.get("reference")))
        ans_vector = _score_vector(record.get(ans_key, record.get("prediction", record.get("scores"))))
        if not ref_vector or not ans_vector:
            continue
        width = min(len(ref_vector), len(ans_vector))
        ref_vectors.append(ref_vector[:width])
        ans_vectors.append(ans_vector[:width])

    num_aspects = min((len(vector) for vector in (*ref_vectors, *ans_vectors)), default=0)
    spearman_list: list[float | None] = []
    for index in range(num_aspects):
        ref_column = [vector[index] for vector in ref_vectors]
        ans_column = [vector[index] for vector in ans_vectors]
        rho = _spearman(ref_column, ans_column)
        spearman_list.append(round(rho * scale, 4) if rho is not None else None)
    valid = [value for value in spearman_list if value is not None]
    return {
        "spearman_list": spearman_list,
        "spearman_average": (sum(valid) / len(valid)) if valid else 0.0,
        "num_records": len(ref_vectors),
        "num_aspects": num_aspects,
    }


def success_rate(records: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Aggregate embodied/simulation success rows into overall and per-task rates."""

    successes = 0
    trials = 0
    per_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        success_count, trial_count = _success_trial_counts(record)
        if trial_count <= 0:
            continue
        successes += success_count
        trials += trial_count
        task = record.get("task", record.get("task_name", record.get("env", record.get("suite"))))
        if task:
            bucket = per_task[str(task)]
            bucket[0] += success_count
            bucket[1] += trial_count
    return {
        "success_rate": _safe_div(successes, trials),
        "num_success": successes,
        "num_trials": trials,
        "per_task": _success_ratio_counts(per_task),
    }


def parse_worldmodelbench_score(text: JsonValue) -> float:
    """Parse the final numeric score from a WorldModelBench judge response."""

    try:
        return float(str(text).split(":")[-1].strip(" ."))
    except ValueError:
        return 0.0


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Calculate the average precision (AP) of scores relative to binary labels.

    Args:
        scores: Sequence of scores.
        labels: Sequence of binary labels (0 or 1).

    Returns:
        Average precision float value.
    """
    sorted_pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    seen_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(sorted_pairs, start=1):
        if label == 1:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / positives


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Calculate the area under the receiver operating characteristic curve (ROC AUC).

    Args:
        scores: Sequence of scores.
        labels: Sequence of binary labels (0 or 1).

    Returns:
        ROC AUC float value.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = _average_ranks(scores)
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Calculate average fractional ranks for scores to correctly handle ties.

    Args:
        values: Sequence of numeric values.

    Returns:
        List of average ranks matching the input length.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute Spearman's rank correlation coefficient between two score vectors.

    Args:
        left: First sequence of numeric values.
        right: Second sequence of numeric values.

    Returns:
        Spearman correlation coefficient float, or None if invalid or too short.
    """
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=False)
    )
    left_denominator = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_denominator = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_denominator == 0.0 or right_denominator == 0.0:
        return None
    return numerator / (left_denominator * right_denominator)


def _camera_score_pair(record: Mapping[str, JsonValue]) -> tuple[dict[str, float], dict[str, float]] | None:
    """Helper to extract positive/negative scores for camera text/image quads.

    Args:
        record: Score record mapping.

    Returns:
        Tuple of (yes_scores_dict, no_scores_dict), or None if missing.
    """
    if record.get("error") is not None:
        return None
    yes_scores = record.get("yes_scores")
    no_scores = record.get("no_scores")
    if not isinstance(yes_scores, Mapping) or not isinstance(no_scores, Mapping):
        return None
    keys = ("pos_text_pos_image", "pos_text_neg_image", "neg_text_pos_image", "neg_text_neg_image")
    try:
        return (
            {key: float(yes_scores[key]) for key in keys},
            {key: float(no_scores[key]) for key in keys},
        )
    except (KeyError, TypeError, ValueError):
        return None


def _retrieval_score_dict(record: Mapping[str, JsonValue]) -> dict[str, float] | None:
    """Helper to extract score dictionary for retrieval task evaluations.

    Args:
        record: Score record mapping.

    Returns:
        Score dict, or None if invalid.
    """
    if all(
        key in record
        for key in ("pos_text_pos_image", "pos_text_neg_image", "neg_text_pos_image", "neg_text_neg_image")
    ):
        try:
            return {
                key: float(record[key])
                for key in ("pos_text_pos_image", "pos_text_neg_image", "neg_text_pos_image", "neg_text_neg_image")
            }
        except (TypeError, ValueError):
            return None
    pair = _camera_score_pair(record)
    if pair is None:
        return None
    yes_scores, _ = pair
    return yes_scores


def _binary_label(value: JsonValue) -> int | None:
    """Standardize input label to integer binary 0 or 1.

    Args:
        value: Input label value.

    Returns:
        1 or 0, or None if unrecognized.
    """
    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "positive", "pos"}:
        return 1
    if text in {"no", "false", "0", "negative", "neg"}:
        return 0
    return None


def _finite_score(value: JsonValue) -> float:
    """Parse and return a finite float value, with a very low fallback for infinite/invalid values.

    Args:
        value: Raw numeric score value.

    Returns:
        Finite float score.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return -1e10
    return score if math.isfinite(score) else -1e10


def _choice_letter(value: JsonValue) -> str:
    """Extract choice letter option (e.g. 'A') from predictions/references.

    Args:
        value: Option string or mapping.

    Returns:
        Lower-cased first letter character of option.
    """
    text = str(value or "").lower().replace(".", "").replace("(", "").replace(")", "").strip()
    return text[:1]


def _join_prediction_answer_rows(
    predictions: Mapping[str, Mapping[str, JsonValue]] | Sequence[Mapping[str, JsonValue]],
    answers: Mapping[str, Mapping[str, JsonValue]] | Sequence[Mapping[str, JsonValue]] | None,
) -> tuple[dict[str, JsonValue], ...]:
    """Join multiple-choice predictions and answers by matching sample/question ID.

    Args:
        predictions: Collection of predictions.
        answers: Collection of reference answers.

    Returns:
        Joined collection of rows.
    """
    pred_map = _row_map(predictions)
    if answers is None:
        return tuple(dict(row) for row in pred_map.values())
    answer_map = _row_map(answers)
    return tuple(
        {**dict(pred_row), **dict(answer_map.get(key, {}))} for key, pred_row in pred_map.items() if key in answer_map
    )


def _row_map(
    rows: Mapping[str, Mapping[str, JsonValue]] | Sequence[Mapping[str, JsonValue]],
) -> dict[str, Mapping[str, JsonValue]]:
    """Map collection sequence rows to dictionaries keyed by question/sample ID.

    Args:
        rows: Rows sequence or dictionary.

    Returns:
        Dictionary mapping IDs to row metadata.
    """
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items() if isinstance(value, Mapping)}
    output: dict[str, Mapping[str, JsonValue]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        key = row.get("question_id", row.get("id", row.get("sample_id", index)))
        output[str(key)] = row
    return output


def _ratio_counts(values: Mapping[str, Sequence[int]]) -> dict[str, dict[str, JsonValue]]:
    """Convert raw correct/total counts per-task into percentage accuracy ratios.

    Args:
        values: Dictionary mapping task name to [correct_count, total_count] list.

    Returns:
        Structured accuracy mapping per-task.
    """
    return {
        key: {"accuracy": _safe_div(counts[0], counts[1]), "num_correct": counts[0], "num_total": counts[1]}
        for key, counts in values.items()
    }


def _success_ratio_counts(values: Mapping[str, Sequence[int]]) -> dict[str, dict[str, JsonValue]]:
    """Convert raw successes/trials counts per-task into success rate ratios.

    Args:
        values: Dictionary mapping task name to [success_count, trial_count] list.

    Returns:
        Structured success rate mapping per-task.
    """
    return {
        key: {"success_rate": _safe_div(counts[0], counts[1]), "num_success": counts[0], "num_trials": counts[1]}
        for key, counts in values.items()
    }


def _record_name(record: Mapping[str, JsonValue]) -> str:
    """Retrieve filename, name, or path from a record safely.

    Args:
        record: Score/result record mapping.

    Returns:
        String identifier.
    """
    return str(record.get("filename") or record.get("name") or record.get("path") or "")


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    """Perform float division safely, returning 0.0 if denominator is zero.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.

    Returns:
        Division result float.
    """
    return float(numerator) / float(denominator) if denominator else 0.0


def _normalize_score_keys(
    scores: Mapping[str, JsonValue], *, aliases: Mapping[str, str] | None = None
) -> dict[str, JsonValue]:
    """Map raw or colloquial dimension names to their canonical VBench task names.

    Args:
        scores: Input scores mapping.
        aliases: Name mapping overrides.

    Returns:
        Normalized scores mapping.
    """
    output: dict[str, JsonValue] = {}
    aliases = aliases or {}
    for key, value in scores.items():
        text = str(key)
        canonical = aliases.get(text, text.replace("_", " ").strip())
        output[canonical] = (
            value[0] if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value else value
        )
    return output


def _weighted_normalized_scores(
    scores: Mapping[str, JsonValue],
    *,
    task_names: Sequence[str],
    normalize_ranges: Mapping[str, tuple[float, float]],
    dim_weights: Mapping[str, float],
) -> dict[str, float]:
    """Calculate the normalized and weighted value for each VBench task.

    Args:
        scores: Key-value dimension scores.
        task_names: Mandatory task names sequence.
        normalize_ranges: Min/max normalization ranges per-dimension.
        dim_weights: Importance weights per-dimension.

    Returns:
        Dictionary mapping dimension names to weighted float scores in [0.0, 1.0].
    """
    normalized: dict[str, float] = {}
    for name in task_names:
        raw_value = _finite_score(scores.get(name, 0.0))
        min_value, max_value = normalize_ranges[name]
        value = 0.0 if max_value == min_value else (raw_value - min_value) / (max_value - min_value)
        normalized[name] = value * dim_weights[name]
    return normalized


def _weighted_group_mean(
    weighted_scores: Mapping[str, float],
    names: Sequence[str],
    dim_weights: Mapping[str, float],
) -> float:
    """Calculate the weighted group mean score over selected dimensions.

    Args:
        weighted_scores: Previously weighted scores.
        names: Sub-dimension keys list.
        dim_weights: Baseline dimension weights.

    Returns:
        Calculated mean float.
    """
    return sum(weighted_scores[name] for name in names) / sum(dim_weights[name] for name in names)


def _preference_label(value: JsonValue) -> str | None:
    """Normalize preference Winner labels into a stable 'a', 'b', or 'tie' value.

    Args:
        value: Input raw winner label.

    Returns:
        'a', 'b', or 'tie' string, or None if unrecognized.
    """
    text = str(value or "").strip().lower()
    if text in {"a", "left", "first", "model_a", "model a", "0", "1"}:
        return "a"
    if text in {"b", "right", "second", "model_b", "model b", "2"}:
        return "b"
    if text in {"tie", "draw", "same", "equal", "both"}:
        return "tie"
    return None


def _preference_from_scores(score_a: JsonValue, score_b: JsonValue) -> str | None:
    """Derive preference winner from raw scores comparisons.

    Args:
        score_a: Model A score.
        score_b: Model B score.

    Returns:
        Winner indicator ('a', 'b', or 'tie').
    """
    try:
        a_value = float(score_a)
        b_value = float(score_b)
    except (TypeError, ValueError):
        return None
    if a_value == b_value:
        return "tie"
    return "a" if a_value > b_value else "b"


def _success_trial_counts(record: Mapping[str, JsonValue]) -> tuple[int, int]:
    """Parse out success and trial integers from a given episode or metrics mapping.

    Args:
        record: Episode or metrics record.

    Returns:
        Tuple of (success_count, trial_count).
    """
    for success_key, trial_key in (
        ("num_success", "num_trials"),
        ("successes", "trials"),
        ("suc_num", "test_num"),
        ("success_count", "episode_count"),
    ):
        if success_key in record or trial_key in record:
            success_count = int(float(record.get(success_key, 0) or 0))
            trial_count = int(float(record.get(trial_key, 0) or 0))
            return success_count, trial_count
    for key in ("success", "succeeded", "is_success", "done_success", "task_success"):
        if key in record:
            return int(bool(record[key])), 1
    if "success_rate" in record:
        trials = int(float(record.get("num_trials", record.get("trials", 1)) or 1))
        return round(float(record["success_rate"]) * trials), trials
    return 0, 0


def _score_vector(value: JsonValue) -> list[float]:
    """Safely parse a list of finite float values from a string or sequence representation.

    Args:
        value: Input sequence or string list representation.

    Returns:
        List of parsed float values.
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        output.append(number)
    return output


def _truthy_label(value: JsonValue) -> bool | None:
    """Evaluate whether an input label indicates truthiness or boolean correctness.

    Args:
        value: Input label value.

    Returns:
        True if truthy, False if falsy, or None if unrecognized.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1", "correct", "right"}:
            return True
        if normalized in {"false", "no", "0", "incorrect", "wrong"}:
            return False
    return None


def jedi_mmd_score(
    train_features: Any,
    test_features: Any,
) -> float:
    """Compute the VideoJEDi polynomial-kernel MMD score from feature arrays."""
    from worldfoundry.evals.tasks.metrics.distribution.jedi.wrapper import compute_jedi_from_features

    return compute_jedi_from_features(train_features, test_features)


__all__ = [
    "boolean_accuracy",
    "camera_binary_classification_metrics",
    "camera_retrieval_metrics",
    "camera_vqa_metrics",
    "chronomagic_average_scores",
    "jedi_mmd_score",
    "multiple_choice_accuracy",
    "pairwise_preference_accuracy",
    "parse_worldmodelbench_score",
    "score_vector_spearman",
    "success_rate",
    "vbench_final_score",
    "videoverse_subquestion_metrics",
    "worldmodelbench_score",
]
