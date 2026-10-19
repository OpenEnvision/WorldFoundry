"""PhyEduVideo metric formulas from official summary or per-sample result rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean
from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float

METRIC_ORDER = (
    "semantic_adherence",
    "physics_commonsense",
    "motion_smoothness",
    "temporal_flickering",
    "phyeduvideo_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "semantic_adherence": {
        "name": "Semantic Adherence",
        "group": "semantics",
        "higher_is_better": True,
        "description": "Semantic adherence to teaching-point prompts and rubric objects/actions.",
    },
    "physics_commonsense": {
        "name": "Physics Commonsense",
        "group": "physics",
        "higher_is_better": True,
        "description": "Physics commonsense QA score aggregated over PC-1/PC-2/PC-3 rubrics.",
    },
    "motion_smoothness": {
        "name": "Motion Smoothness",
        "group": "motion",
        "higher_is_better": True,
        "description": "Motion smoothness score for generated explanatory videos.",
    },
    "temporal_flickering": {
        "name": "Temporal Flickering",
        "group": "temporal",
        "higher_is_better": True,
        "description": "Temporal flickering score for generated explanatory videos.",
    },
    "phyeduvideo_average": {
        "name": "PhyEduVideo Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean over available PhyEduVideo score families.",
        "primary": True,
    },
}

METRIC_ALIASES = {
    "semantic_adherence": "semantic_adherence",
    "sa": "semantic_adherence",
    "semantic adherence": "semantic_adherence",
    "physics_commonsense": "physics_commonsense",
    "physics commonsense": "physics_commonsense",
    "pc": "physics_commonsense",
    "motion_smoothness": "motion_smoothness",
    "motion smoothness": "motion_smoothness",
    "temporal_flickering": "temporal_flickering",
    "temporal flickering": "temporal_flickering",
    "phyeduvideo_average": "phyeduvideo_average",
    "average": "phyeduvideo_average",
}


def _normalize_unit_score(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _canonical_metric_id(raw: str) -> str | None:
    key = str(raw or "").strip().lower().replace("-", "_")
    return METRIC_ALIASES.get(key)


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        metric_id = _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or ""))
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if metric_id is None or score is None:
            continue
        metrics[metric_id] = _normalize_unit_score(score)
    component_values = [
        value for key, value in metrics.items() if key not in {"phyeduvideo_average"} and value is not None
    ]
    if metrics.get("phyeduvideo_average") is None and component_values:
        metrics["phyeduvideo_average"] = sum(component_values) / len(component_values)
    return metrics


def _is_summary_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and any(
        _canonical_metric_id(str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "")) for row in rows
    )


def _sa_score(row: Mapping[str, Any]) -> float | None:
    for key in ("semantic_adherence", "SA_internVL35", "sa_score", "object_score", "score"):
        value = _to_float(row.get(key))
        if value is not None:
            return _normalize_unit_score(value / 3.0 if key == "SA_internVL35" and value > 1.0 else value)
    object_score = _to_float(row.get("object_score"))
    action_score = _to_float(row.get("action_score"))
    if object_score is not None and action_score is not None:
        return _normalize_unit_score((object_score + action_score) / 3.0)
    return None


def _pc_score(row: Mapping[str, Any], model_prefix: str | None = None) -> float | None:
    if model_prefix:
        for suffix in ("_score", "_descrete", "_discrete"):
            value = _to_float(row.get(f"{model_prefix}{suffix}"))
            if value is not None:
                return _normalize_unit_score(value / 3.0 if value > 1.0 else value)
    for key in ("physics_commonsense", "pc_score", "score", "clip_score"):
        value = _to_float(row.get(key))
        if value is not None:
            return _normalize_unit_score(value / 3.0 if value > 1.0 else value)
    return None


def _aggregate_sample_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    sa_values: list[float] = []
    pc_values: list[float] = []
    ms_values: list[float] = []
    tf_values: list[float] = []
    for row in rows:
        sa = _sa_score(row)
        if sa is not None:
            sa_values.append(sa)
        pc = _pc_score(row)
        if pc is not None:
            pc_values.append(pc)
        ms = _to_float(row.get("motion_smoothness"))
        if ms is not None:
            ms_values.append(_normalize_unit_score(ms))
        tf = _to_float(row.get("temporal_flickering"))
        if tf is not None:
            tf_values.append(_normalize_unit_score(tf))
    return {
        "semantic_adherence": _mean(sa_values),
        "physics_commonsense": _mean(pc_values),
        "motion_smoothness": _mean(ms_values),
        "temporal_flickering": _mean(tf_values),
    }


def compute_phyeduvideo_metrics(*, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if _is_summary_rows(rows):
        direct_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    if len(rows) == 1 and isinstance(rows[0], Mapping):
        payload = rows[0]
        direct_metrics = {
            metric_id: _normalize_unit_score(score)
            for metric_id in METRIC_ORDER
            if (score := _to_float(payload.get(metric_id))) is not None
        }
        if direct_metrics:
            component_values = [
                value for key, value in direct_metrics.items() if key != "phyeduvideo_average" and value is not None
            ]
            if direct_metrics.get("phyeduvideo_average") is None and component_values:
                direct_metrics["phyeduvideo_average"] = sum(component_values) / len(component_values)
            return {
                "metrics": direct_metrics,
                "components": {"format": "summary_json_object"},
            }

    direct_metrics = _aggregate_sample_rows(rows)
    component_values = [value for key, value in direct_metrics.items() if value is not None]
    if component_values:
        direct_metrics["phyeduvideo_average"] = sum(component_values) / len(component_values)
    else:
        direct_metrics["phyeduvideo_average"] = None

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
        raise ValueError(f"Unsupported PhyEduVideo JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
