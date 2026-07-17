"""VideoPhy2 metric normalization (Likert SA/PC/joint and rule classification)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.runners.videophy2.videophy2_official_scoring import (
    official_scores_from_records,
)

METRIC_ORDER = (
    "semantic_adherence",
    "physical_commonsense",
    "joint_score",
    "rule_followed_rate",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "semantic_adherence": {
        "name": "Semantic Adherence",
        "group": "general",
        "higher_is_better": True,
        "description": "Mean 1-5 semantic adherence rating normalized to unit interval.",
    },
    "physical_commonsense": {
        "name": "Physical Commonsense",
        "group": "general",
        "higher_is_better": True,
        "description": "Mean 1-5 physical commonsense rating normalized to unit interval.",
    },
    "joint_score": {
        "name": "Joint Score",
        "group": "general",
        "higher_is_better": True,
        "description": "Fraction of prompts with SA>=4 and PC>=4.",
        "primary": True,
    },
    "rule_followed_rate": {
        "name": "Rule Followed Rate",
        "group": "rule",
        "higher_is_better": True,
        "description": "Fraction of determined physical rules judged as followed rather than violated.",
    },
}

SUMMARY_METRIC_ALIASES = {
    "semantic_adherence": "semantic_adherence",
    "physical_commonsense": "physical_commonsense",
    "joint_score": "joint_score",
    "rule_followed_rate": "rule_followed_rate",
    "physical_rule_followed_rate": "rule_followed_rate",
}
DEPRECATED_METRIC_IDS = frozenset({"rule_classification_accuracy", "videophy2_average"})


def _is_summary_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "").strip()
        for row in rows[: min(3, len(rows))]
    )


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    metrics: dict[str, float] = {}
    deprecated_metrics: dict[str, float] = {}
    for row in rows:
        raw_metric_id = str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "").strip()
        score = row.get("score") if "score" in row else row.get("value")
        if not raw_metric_id or score in (None, ""):
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        value = value / 100.0 if value > 1.0 else value
        if raw_metric_id in DEPRECATED_METRIC_IDS:
            deprecated_metrics[raw_metric_id] = value
            continue
        metric_id = SUMMARY_METRIC_ALIASES.get(raw_metric_id)
        if metric_id is not None:
            metrics[metric_id] = value
    return metrics, deprecated_metrics


def _deprecated_row_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    deprecated: dict[str, float] = {}
    for metric_id in DEPRECATED_METRIC_IDS:
        values: list[float] = []
        for row in rows:
            value = row.get(metric_id)
            try:
                if value not in (None, ""):
                    values.append(float(value))
            except (TypeError, ValueError):
                continue
        if values:
            deprecated[metric_id] = sum(values) / len(values)
    return deprecated


def load_results_rows(results_path: Path) -> list[dict[str, Any]]:
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("results", "rows", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
        raise ValueError(f"Unsupported VideoPhy2 JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def compute_videophy2_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_path: Path | None = None,
) -> dict[str, Any]:
    if _is_summary_rows(rows):
        direct_metrics, deprecated_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {
                "row_count": len(rows),
                "format": "summary_csv",
                "deprecated_metrics": deprecated_metrics,
            },
        }

    official_scores = official_scores_from_records(list(rows), results_path)
    direct_metrics: dict[str, float | None] = {
        metric_id: (official_scores[metric_id].score if metric_id in official_scores else None)
        for metric_id in METRIC_ORDER
    }
    return {
        "metrics": direct_metrics,
        "components": {
            "row_count": len(rows),
            "format": "official_dimension_rows",
            "deprecated_metrics": _deprecated_row_metrics(rows),
        },
    }
