"""Local deterministic evaluators lookup map.

This module maps standard evaluation strings (like 'artifact_count', 'camera_vqa')
to concrete helper functions implementing metrics calculation locally.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from worldfoundry.evals.api import MetricResult
from worldfoundry.evals.tasks.metrics.registry import missing_artifacts, normalize_artifact_records

from .formulas import (
    boolean_accuracy,
    camera_binary_classification_metrics,
    camera_retrieval_metrics,
    camera_vqa_metrics,
    chronomagic_average_scores,
    multiple_choice_accuracy,
    pairwise_preference_accuracy,
    score_vector_spearman,
    success_rate,
    vbench_final_score,
    videoverse_subquestion_metrics,
    worldmodelbench_score,
)
from .protocols import payload_from_request_parts, records_from_payload

MetricEvaluationCallable = Callable[[Any, Any], MetricResult]


def _metric_key(value: str) -> str:
    """Normalize a metric key string to lowercase hyphenated.

    Args:
        value: Raw key string.

    Returns:
        Normalized key.
    """
    return value.strip().casefold().replace("_", "-")


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Coerce any sequence or single value into a tuple of strings.

    Args:
        value: Input value or sequence.

    Returns:
        Tuple of string items.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _required_artifacts(entry: Any, request: Any) -> tuple[str, ...]:
    """Compile the combined required artifact names from evaluator entry and request metadata.

    Args:
        entry: Evaluator entry specification.
        request: Request specification.

    Returns:
        Deduplicated tuple of required artifact name strings.
    """
    metadata_required = _tuple_of_str(request.task_metadata.get("required_artifacts"))
    return tuple(dict.fromkeys((*entry.required_artifacts, *metadata_required)))


def _failure_result(
    request: Any,
    *,
    skip_reason: str,
    message: str,
    diagnostics: Mapping[str, Any] | None = None,
) -> MetricResult:
    """Build a standard invalid metric result for local evaluation failures.

    Args:
        request: Metric evaluation request.
        skip_reason: Reason for skipping or failure.
        message: Explanatory error details.
        diagnostics: Additional structured diagnostics.

    Returns:
        Invalid MetricResult.
    """
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        valid=False,
        coverage=0.0,
        skip_reason=skip_reason,
        diagnostics={"message": message, **dict(diagnostics or {})},
    )


def _artifact_count_metric(request: Any, entry: Any) -> MetricResult:
    """Compute artifact count metrics based on generated artifact records.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = normalize_artifact_records(request.generated_artifact_manifest)
    required = _required_artifacts(entry, request)
    missing = missing_artifacts(required, records, base_dir=request.artifact_base_dir)
    if missing:
        return _failure_result(
            request,
            skip_reason="missing_artifact",
            message="required generated artifacts are missing",
            diagnostics={"missing_artifacts": list(missing), "required_artifacts": list(required)},
        )
    value = float(len(records))
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=int(value),
        normalized_value=value,
        components={"artifact_count": int(value)},
        diagnostics={"evaluation_kind": entry.evaluation_kind},
    )


def _required_artifacts_present_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate whether all requested/required artifacts are present.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = normalize_artifact_records(request.generated_artifact_manifest)
    required = _required_artifacts(entry, request)
    if not required:
        return MetricResult(
            sample_id=request.sample_id,
            metric_id=request.metric_id,
            raw_value=1,
            normalized_value=1.0,
            components={"required_artifacts": []},
            diagnostics={"evaluation_kind": entry.evaluation_kind},
        )
    missing = missing_artifacts(required, records, base_dir=request.artifact_base_dir)
    if missing:
        return _failure_result(
            request,
            skip_reason="missing_artifact",
            message="required generated artifacts are missing",
            diagnostics={"missing_artifacts": list(missing), "required_artifacts": list(required)},
        )
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=1,
        normalized_value=1.0,
        components={"required_artifacts": list(required)},
        diagnostics={"evaluation_kind": entry.evaluation_kind},
    )


def _metric_payload(request: Any) -> Any:
    """Helper to extract metric payload from requests.

    Args:
        request: Request specification.

    Returns:
        The metric payload.
    """
    return payload_from_request_parts(
        generated_artifact_manifest=request.generated_artifact_manifest,
        task_metadata=request.task_metadata,
        reference=request.reference,
    )


def _camera_binary_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate CameraBench binary classification metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="CameraBench binary metrics require score records with score and ground_truth_label fields.",
        )
    metrics = camera_binary_classification_metrics(records)
    value_key = "roc_auc" if _metric_key(request.metric_id).endswith("roc-auc") else "average_precision"
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics[value_key],
        normalized_value=float(metrics[value_key]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "camerabench.in_tree_metrics"},
    )


def _camera_vqa_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate CameraBench VQA metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="CameraBench VQA metrics require records with yes_scores and no_scores fields.",
        )
    metrics = camera_vqa_metrics(records)
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics["question_acc"],
        normalized_value=float(metrics["question_acc"]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "camerabench.in_tree_metrics"},
    )


def _camera_retrieval_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate CameraBench retrieval metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="CameraBench retrieval metrics require score records.",
        )
    metrics = camera_retrieval_metrics(records)
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics["group"],
        normalized_value=float(metrics["group"]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "camerabench.in_tree_metrics"},
    )


def _chronomagic_chscore_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate ChronoMagic CHScore or MTScore metrics from records.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    payload = _metric_payload(request)
    if not isinstance(payload, (Mapping, Sequence)) or isinstance(payload, (str, bytes)):
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="ChronoMagic CHScore aggregation requires official per-video CHScore JSON records.",
        )
    metrics = chronomagic_average_scores(
        payload,
        score_key=str(request.task_metadata.get("score_key") or "total_average_score"),
        suffix=str(request.task_metadata.get("suffix") or "CHScore"),
    )
    if not metrics:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="ChronoMagic CHScore aggregation found no matching CHScore records.",
        )
    values = [
        float(model_scores["Average_CHScore"])
        for model_scores in metrics.values()
        if isinstance(model_scores, Mapping) and isinstance(model_scores.get("Average_CHScore"), (int, float))
    ]
    value = sum(values) / len(values) if values else 0.0
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=value,
        normalized_value=float(value),
        components={"average_chscore": value, "per_model": metrics, "num_models": len(values)},
        diagnostics={
            "evaluation_kind": entry.evaluation_kind,
            "source": "ChronoMagic-Bench.CHScore.step1-get_merged_CHScore",
        },
    )


def _genai_bench_accuracy_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate GenAI-Bench pairwise accuracy metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="GenAI-Bench accuracy requires rows with a boolean correct field.",
        )
    metric_key = _metric_key(request.metric_id)
    task_filter = {
        "image-generation-preference-accuracy": ("image_generation",),
        "image-editing-preference-accuracy": ("image_edition", "image_editing"),
        "video-preference-accuracy": ("video_generation",),
    }.get(metric_key)
    if task_filter is not None:
        records = tuple(
            record
            for record in records
            if str(record.get("task", record.get("category", ""))).strip().casefold().replace("-", "_") in task_filter
        )
    metrics = boolean_accuracy(records)
    value = float(metrics["accuracy"])
    if metric_key == "genai-bench-average" and metrics["per_task"]:
        value = sum(float(item["accuracy"]) for item in metrics["per_task"].values()) / len(metrics["per_task"])
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=value,
        normalized_value=value,
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "GenAI-Bench.show_results"},
    )


def _videoverse_subquestion_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate VideoVerse sub-question and video accuracy metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    payload = _metric_payload(request)
    if not isinstance(payload, Mapping):
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="VideoVerse metrics require a mapping of video ids to verification_checks.",
        )
    metrics = videoverse_subquestion_metrics(payload)
    value_key = "video_accuracy" if _metric_key(request.metric_id) == "videoverse-average" else "sub_question_accuracy"
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics[value_key],
        normalized_value=float(metrics[value_key]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "VideoVerse.eval_sub_question_acc"},
    )


def _videoscore_spearman_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate VideoScore aspect Spearman correlations.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="VideoScore Spearman metrics require rows with ref and ans score vectors.",
        )
    metrics = score_vector_spearman(records)
    aspect_index = {
        "visual-quality": 0,
        "temporal-consistency": 1,
        "dynamic-degree": 2,
        "text-to-video-alignment": 3,
        "factual-consistency": 4,
    }.get(_metric_key(request.metric_id))
    if aspect_index is None:
        value = float(metrics["spearman_average"])
    else:
        spearman_list = metrics["spearman_list"]
        value = float(spearman_list[aspect_index] or 0.0) if aspect_index < len(spearman_list) else 0.0
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=value,
        normalized_value=value / 100.0,
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "VideoScore.benchmark.get_spearman_corr"},
    )


def _mcqa_accuracy_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate MCQA prediction accuracy from expected and predicted columns.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    predictions = request.reference.get(
        "predictions", request.reference.get("pred", request.reference.get("official_results"))
    )
    answers = request.reference.get("answers", request.reference.get("ground_truth", request.reference.get("labels")))
    taxonomy = request.reference.get("taxonomy_by_video")
    if predictions is None:
        predictions = _metric_payload(request)
    if predictions is None:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="MCQA metrics require predictions and answers or rows with prediction/answer fields.",
        )
    metrics = multiple_choice_accuracy(
        predictions,
        answers,
        taxonomy_by_video=taxonomy if isinstance(taxonomy, Mapping) else None,
    )
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics["accuracy"],
        normalized_value=float(metrics["accuracy"]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind},
    )


def _worldmodelbench_result_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate WorldModelBench results category score and overall total score.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    payload = _metric_payload(request)
    if not isinstance(payload, Mapping):
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="WorldModelBench metrics require an accs mapping and num_instances.",
        )
    accs = payload.get("accs", payload)
    if not isinstance(accs, Mapping):
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="WorldModelBench accs must be a mapping from category to score list.",
        )
    num_instances = int(request.reference.get("num_instances") or request.task_metadata.get("num_instances") or 0)
    metrics = worldmodelbench_score(accs, num_instances=num_instances)
    category = request.metric_id
    if category == "world_model_average":
        value = metrics["total_score"]
    else:
        category_metrics = metrics["categories"].get(category)
        if not isinstance(category_metrics, Mapping):
            return _failure_result(
                request,
                skip_reason="missing_metric_input",
                message=f"WorldModelBench category {category!r} was not present in accs.",
            )
        value = category_metrics["overall"]
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=value,
        normalized_value=float(value),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "WorldModelBench.evaluation.process_results"},
    )


def _vbench_final_score_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate and aggregate VBench dimension scores.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    payload = _metric_payload(request)
    if not isinstance(payload, Mapping):
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="VBench final-score aggregation requires a mapping of dimension scores.",
        )
    i2v = bool(request.task_metadata.get("i2v")) or _metric_key(request.metric_id) == "vbench-plus-plus-i2v-average"
    metrics = vbench_final_score(payload, i2v=i2v)
    value_key = "final_score"
    if _metric_key(request.metric_id) in {"quality-score", "overall-quality"}:
        value_key = "final_score"
    elif _metric_key(request.metric_id) in {"semantic-score", "text-alignment"} and "semantic_score" in metrics:
        value_key = "semantic_score"
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics[value_key],
        normalized_value=float(metrics[value_key]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "VBench.scripts.cal_final_score"},
    )


def _pairwise_preference_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate pairwise preference prediction metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="Pairwise preference metrics require rows with prediction/ref preference labels or score_a/score_b.",
        )
    metrics = pairwise_preference_accuracy(records)
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics["accuracy"],
        normalized_value=float(metrics["accuracy"]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind},
    )


def _success_rate_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate embodied benchmark success rate metrics.

    Args:
        request: Request specification.
        entry: Evaluator entry spec.

    Returns:
        The computed MetricResult.
    """
    records = records_from_payload(_metric_payload(request))
    if not records:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="Success-rate metrics require episode rows or aggregate rows with success/trial counts.",
        )
    metrics = success_rate(_filter_success_records(records, request.metric_id))
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=metrics["success_rate"],
        normalized_value=float(metrics["success_rate"]),
        components=metrics,
        diagnostics={"evaluation_kind": entry.evaluation_kind},
    )


def _jedi_mmd_metric(request: Any, entry: Any) -> MetricResult:
    """Evaluate VideoJEDi score from precomputed train/test feature arrays."""
    from worldfoundry.evals.tasks.metrics.distribution.jedi.jedi_runtime import score_from_feature_payload

    payload = _metric_payload(request)
    if not isinstance(payload, Mapping) and isinstance(request.reference, Mapping):
        payload = request.reference
    if not isinstance(payload, Mapping):
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="JEDi metrics require train/test feature arrays or feature file paths.",
        )
    value = score_from_feature_payload(payload)
    if value is None:
        return _failure_result(
            request,
            skip_reason="missing_metric_input",
            message="JEDi metrics require both train and test feature payloads.",
        )
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        raw_value=value,
        normalized_value=float(value),
        components={"jedi_score": value},
        diagnostics={"evaluation_kind": entry.evaluation_kind, "source": "jedi.in_tree"},
    )


def _filter_success_records(
    records: Sequence[Mapping[str, Any]],
    metric_id: str,
) -> tuple[Mapping[str, Any], ...]:
    """Filter success records matching the target metric name or tokens.

    Args:
        records: Input sequence of records.
        metric_id: Metric identifier.

    Returns:
        Tuple of matching records.
    """
    key = _metric_key(metric_id)
    if key in {
        "success-rate",
        "task-success",
        "episode-success",
        "sequence-success",
        "completion",
        "goal-success",
        "rollout-success",
    }:
        return tuple(records)
    tokens = tuple(token for token in key.split("-") if token not in {"success", "rate", "task", "episode"})
    if not tokens:
        return tuple(records)
    filtered = []
    for record in records:
        haystack = (
            " ".join(
                str(record.get(name, ""))
                for name in ("metric_id", "condition", "split", "domain", "task", "task_name", "suite", "category")
            )
            .casefold()
            .replace("_", "-")
        )
        if all(token in haystack for token in tokens):
            filtered.append(record)
    return tuple(filtered) if filtered else tuple(records)


LOCAL_EVALUATORS: Mapping[str, MetricEvaluationCallable] = {
    "artifact_count": _artifact_count_metric,
    "camera_binary_classification": _camera_binary_metric,
    "camera_retrieval": _camera_retrieval_metric,
    "camera_vqa": _camera_vqa_metric,
    "chronomagic_average_chscore": _chronomagic_chscore_metric,
    "genai_bench_accuracy": _genai_bench_accuracy_metric,
    "jedi_mmd": _jedi_mmd_metric,
    "mcqa_accuracy": _mcqa_accuracy_metric,
    "pairwise_preference_accuracy": _pairwise_preference_metric,
    "required_artifacts_present": _required_artifacts_present_metric,
    "success_rate": _success_rate_metric,
    "vbench_final_score": _vbench_final_score_metric,
    "videoscore_spearman": _videoscore_spearman_metric,
    "videoverse_subquestion_accuracy": _videoverse_subquestion_metric,
    "worldmodelbench_result_scores": _worldmodelbench_result_metric,
}


__all__ = [
    "LOCAL_EVALUATORS",
    "MetricEvaluationCallable",
]
