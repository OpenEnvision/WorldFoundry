"""EWMBench metric normalization from official CSV/JSON exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean
from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float

METRIC_ORDER = (
    "scene_consistency",
    "motion_correctness",
    "semantic_alignment",
    "diversity",
    "ewmbench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "scene_consistency": {
        "name": "Scene Consistency",
        "group": "quality",
        "higher_is_better": True,
    },
    "motion_correctness": {
        "name": "Motion Correctness",
        "group": "motion",
        "higher_is_better": True,
    },
    "semantic_alignment": {
        "name": "Semantic Alignment",
        "group": "semantics",
        "higher_is_better": True,
    },
    "diversity": {
        "name": "Diversity",
        "group": "diversity",
        "higher_is_better": True,
    },
    "ewmbench_average": {
        "name": "EWMBench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "primary": True,
    },
}

UPSTREAM_METRIC_ALIASES = {
    "scene_consistency": "scene_consistency",
    "trajectory_consistency": "motion_correctness",
    "motion_correctness": "motion_correctness",
    "semantics": "semantic_alignment",
    "semantic_alignment": "semantic_alignment",
    "diversity": "diversity",
    "ewmbench_average": "ewmbench_average",
    "average": "ewmbench_average",
}

CSV_COLUMN_MAP = {
    "scene_consistency": "scene_consistency",
    "trajectory_consistency": "motion_correctness",
    "semantics": "semantic_alignment",
    "diversity": "diversity",
}


def _canonical_metric_id(raw: str) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return UPSTREAM_METRIC_ALIASES.get(key) or UPSTREAM_METRIC_ALIASES.get(key.replace(" ", "_"))


def _aggregate_csv_columns(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {metric_id: [] for metric_id in CSV_COLUMN_MAP.values()}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for source_col, metric_id in CSV_COLUMN_MAP.items():
            score = _to_float(row.get(source_col))
            if score is not None:
                buckets[metric_id].append(score)
    metrics = {metric_id: _mean(values) for metric_id, values in buckets.items() if values}
    component_values = [value for key, value in metrics.items() if key != "ewmbench_average" and value is not None]
    if component_values:
        metrics["ewmbench_average"] = sum(component_values) / len(component_values)
    return metrics


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        metric_id = _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or ""))
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if metric_id is None or score is None:
            continue
        buckets.setdefault(metric_id, []).append(score)
    metrics = {metric_id: _mean(values) for metric_id, values in buckets.items() if values}
    if metrics.get("ewmbench_average") is None:
        component_values = [value for key, value in metrics.items() if key != "ewmbench_average" and value is not None]
        if component_values:
            metrics["ewmbench_average"] = sum(component_values) / len(component_values)
    return metrics


def _direct_object_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_key, raw_value in payload.items():
        metric_id = _canonical_metric_id(str(raw_key))
        score = _to_float(raw_value)
        if metric_id is None or score is None:
            continue
        metrics[metric_id] = score
    if metrics.get("ewmbench_average") is None:
        component_values = [value for key, value in metrics.items() if key != "ewmbench_average" and value is not None]
        if component_values:
            metrics["ewmbench_average"] = sum(component_values) / len(component_values)
    return metrics


def compute_ewmbench_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_path: Path | None = None,
) -> dict[str, Any]:
    if rows and any(
        _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or ""))
        for row in rows
        if isinstance(row, Mapping)
    ):
        direct_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    if len(rows) == 1 and isinstance(rows[0], Mapping):
        direct_metrics = _direct_object_metrics(rows[0])
        if direct_metrics:
            return {
                "metrics": direct_metrics,
                "components": {"format": "summary_json_object"},
            }

    direct_metrics = _aggregate_csv_columns(rows)
    if direct_metrics:
        return {
            "metrics": direct_metrics,
            "components": {"sample_count": len(rows), "format": "ewmbm_final_table_csv"},
        }

    return {"metrics": {}, "components": {"sample_count": len(rows), "format": "unknown"}}


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
        raise ValueError(f"Unsupported EWMBench JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
