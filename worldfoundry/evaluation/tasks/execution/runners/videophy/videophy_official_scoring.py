"""VideoPhy official SA/PC/joint result normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import (
    OfficialMetricScore,
    _joint_score_from_records,
    _score_from_records,
)

BENCHMARK_ID = "videophy"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "judge_required",
    "required_inputs": [
        "VideoPhy public test captions/videos or generated videos for the official captions",
        "human SA/PC labels or VideoCon-Physics AutoEvaluator checkpoint outputs",
        "per-sample sa, pc, and optional joint score fields",
    ],
}


def official_scores_from_records(
    records: list[Mapping[str, Any]],
    official_results_path: Path | None,
    *,
    average_id: str | None = "videophy_average",
    rating_scale_max: float | None = None,
) -> dict[str, OfficialMetricScore]:
    if not records:
        return {}
    semantic = _score_from_records(
        records,
        "semantic_adherence",
        official_results_path,
        aliases=("sa", "SA"),
        scale_max=rating_scale_max,
    )
    physical = _score_from_records(
        records,
        "physical_commonsense",
        official_results_path,
        aliases=("pc", "PC"),
        scale_max=rating_scale_max,
    )
    joint = _joint_score_from_records(records, official_results_path)
    scores: dict[str, OfficialMetricScore] = {}
    if semantic is not None:
        scores["semantic_adherence"] = semantic
    if physical is not None:
        scores["physical_commonsense"] = physical
    if joint is not None:
        scores["joint_score"] = joint
    average = (
        _score_from_records(records, average_id, official_results_path)
        if average_id is not None
        else None
    )
    if average_id is not None and average is not None:
        scores[average_id] = average
    elif average_id is not None:
        component_values = [
            scores[metric_id].score
            for metric_id in ("semantic_adherence", "physical_commonsense")
            if metric_id in scores
        ]
        if len(component_values) == 2:
            value = sum(component_values) / len(component_values)
            scores[average_id] = OfficialMetricScore(
                score=value,
                raw_value=value,
                evidence={
                    "source_path": None if official_results_path is None else str(official_results_path),
                    "aggregation": "mean_of_sa_and_pc",
                    "components": ["semantic_adherence", "physical_commonsense"],
                },
            )
    return scores
