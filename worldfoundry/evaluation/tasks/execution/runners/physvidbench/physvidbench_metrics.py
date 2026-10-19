"""PhysVidBench metric formulas from official QA output rows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean

METRIC_ORDER = (
    "physical_commonsense_accuracy",
    "affordance_understanding",
    "tool_use_consistency",
    "material_property_consistency",
    "temporal_dynamics_consistency",
    "physvidbench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "physical_commonsense_accuracy": {
        "name": "Physical Commonsense Accuracy",
        "group": "physics",
        "higher_is_better": True,
        "description": "Overall physical commonsense QA accuracy.",
    },
    "affordance_understanding": {
        "name": "Affordance Understanding",
        "group": "physics",
        "higher_is_better": True,
        "description": "Affordance and interaction understanding score.",
    },
    "tool_use_consistency": {
        "name": "Tool Use Consistency",
        "group": "physics",
        "higher_is_better": True,
        "description": "Tool-mediated physical interaction consistency.",
    },
    "material_property_consistency": {
        "name": "Material Property Consistency",
        "group": "physics",
        "higher_is_better": True,
        "description": "Material property consistency.",
    },
    "temporal_dynamics_consistency": {
        "name": "Temporal Dynamics Consistency",
        "group": "temporal",
        "higher_is_better": True,
        "description": "Physical temporal dynamics consistency.",
    },
    "physvidbench_average": {
        "name": "PhysVidBench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean over available physical commonsense score families.",
        "primary": True,
    },
}

TYPE_TO_METRIC: dict[str, str] = {
    "Object Properties & Affordances": "affordance_understanding",
    "Action & Procedural Understanding": "tool_use_consistency",
    "Material Interaction & Transformation": "material_property_consistency",
    "Temporal Dynamics": "temporal_dynamics_consistency",
}


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unit_score(value: float) -> float:
    return value / 100.0 if 1.0 < value <= 100.0 else value


def _parse_match(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        metric_id = str(
            row.get("metric_id")
            or row.get("metric")
            or row.get("Metric")
            or row.get("name")
            or row.get("leaderboard_key")
            or ""
        ).strip()
        if metric_id not in METRIC_SPECS:
            continue
        score = _to_float(
            row.get("score")
            if row.get("score") is not None
            else row.get("value")
            if row.get("value") is not None
            else row.get("raw_score")
            if row.get("raw_score") is not None
            else row.get("normalized_score")
        )
        if score is not None:
            metrics[metric_id] = _unit_score(score)
    component_values = [
        value
        for metric_id, value in metrics.items()
        if metric_id not in {"physical_commonsense_accuracy", "physvidbench_average"}
    ]
    if "physvidbench_average" not in metrics:
        if component_values:
            metrics["physvidbench_average"] = sum(component_values) / len(component_values)
        elif "physical_commonsense_accuracy" in metrics:
            metrics["physvidbench_average"] = metrics["physical_commonsense_accuracy"]
    return metrics


def compute_physvidbench_metrics(*, qa_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(
        str(row.get("metric_id") or row.get("metric") or row.get("Metric") or row.get("leaderboard_key") or "").strip()
        in METRIC_SPECS
        for row in qa_rows
    ):
        direct_metrics = _summary_metrics(qa_rows)
        return {
            "metrics": direct_metrics,
            "components": {
                "question_count": 0,
                "format": "summary_metric_rows",
                "type_bucket_counts": {metric_id: 0 for metric_id in TYPE_TO_METRIC.values()},
            },
        }

    overall: list[float] = []
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in TYPE_TO_METRIC.values()}

    for row in qa_rows:
        match = _parse_match(row.get("Match"))
        if match is None:
            continue
        score = 1.0 if match else 0.0
        overall.append(score)
        types = str(row.get("Types") or "")
        for tag in (part.strip() for part in types.split(",")):
            metric_id = TYPE_TO_METRIC.get(tag)
            if metric_id is not None:
                buckets[metric_id].append(score)

    direct_metrics = {
        "physical_commonsense_accuracy": _mean(overall),
        "affordance_understanding": _mean(buckets["affordance_understanding"]),
        "tool_use_consistency": _mean(buckets["tool_use_consistency"]),
        "material_property_consistency": _mean(buckets["material_property_consistency"]),
        "temporal_dynamics_consistency": _mean(buckets["temporal_dynamics_consistency"]),
    }
    component_values = [
        value for key, value in direct_metrics.items() if key != "physical_commonsense_accuracy" and value is not None
    ]
    direct_metrics["physvidbench_average"] = _mean(component_values) if component_values else _mean(overall)

    return {
        "metrics": direct_metrics,
        "components": {
            "question_count": len(overall),
            "type_bucket_counts": {metric_id: len(values) for metric_id, values in buckets.items()},
        },
    }
