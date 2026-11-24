"""Benchmark-specific metric normalization for catalog scorecards.

These modules map official benchmark outputs and catalog metric ids to
WorldFoundry ``MetricResult`` rows. Reusable ``compute_*`` metrics live under
``worldfoundry.evals.tasks.metrics`` instead.
"""

from __future__ import annotations

from .bindings import (
    FORMULA_EVALUATOR_BINDINGS,
    SUCCESS_METRIC_IDS_BY_BENCHMARK,
    MetricEvaluatorBinding,
    success_metric_bindings,
)
from .evaluators import (
    BLOCKED_EVALUATION_KINDS,
    ExternalMetricEvaluationRequest,
    ExternalMetricEvaluatorEntry,
    ExternalMetricEvaluatorRegistry,
    MetricEvaluationCallable,
    default_external_metric_evaluator_registry,
    evaluate_external_metric,
    get_external_metric_evaluator,
    list_external_metric_evaluators,
)
from .formulas import (
    boolean_accuracy,
    camera_binary_classification_metrics,
    camera_retrieval_metrics,
    camera_vqa_metrics,
    chronomagic_average_scores,
    jedi_mmd_score,
    multiple_choice_accuracy,
    pairwise_preference_accuracy,
    parse_worldmodelbench_score,
    score_vector_spearman,
    success_rate,
    vbench_final_score,
    videoverse_subquestion_metrics,
    worldmodelbench_score,
)
from .local_evaluators import LOCAL_EVALUATORS
from .protocols import (
    BenchmarkMetricInput,
    BenchmarkMetricProtocol,
    payload_from_request_parts,
    records_from_payload,
)

__all__ = [
    "BLOCKED_EVALUATION_KINDS",
    "FORMULA_EVALUATOR_BINDINGS",
    "LOCAL_EVALUATORS",
    "SUCCESS_METRIC_IDS_BY_BENCHMARK",
    "BenchmarkMetricInput",
    "BenchmarkMetricProtocol",
    "ExternalMetricEvaluationRequest",
    "ExternalMetricEvaluatorEntry",
    "ExternalMetricEvaluatorRegistry",
    "MetricEvaluationCallable",
    "MetricEvaluatorBinding",
    "boolean_accuracy",
    "camera_binary_classification_metrics",
    "camera_retrieval_metrics",
    "camera_vqa_metrics",
    "chronomagic_average_scores",
    "default_external_metric_evaluator_registry",
    "evaluate_external_metric",
    "get_external_metric_evaluator",
    "jedi_mmd_score",
    "list_external_metric_evaluators",
    "multiple_choice_accuracy",
    "pairwise_preference_accuracy",
    "parse_worldmodelbench_score",
    "payload_from_request_parts",
    "records_from_payload",
    "score_vector_spearman",
    "success_metric_bindings",
    "success_rate",
    "vbench_final_score",
    "videoverse_subquestion_metrics",
    "worldmodelbench_score",
]
