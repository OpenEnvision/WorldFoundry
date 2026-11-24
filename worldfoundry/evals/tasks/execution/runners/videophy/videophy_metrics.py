"""VideoPhy metric normalization (SA/PC/joint binary AutoEval or human labels)."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.runners.videophy.videophy_official_scoring import (
    official_scores_from_records,
)

METRIC_ORDER = (
    "semantic_adherence",
    "physical_commonsense",
    "joint_score",
    "videophy_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "semantic_adherence": {
        "name": "Semantic Adherence",
        "group": "general",
        "higher_is_better": True,
        "description": "Fraction of prompts with semantic adherence (SA=1 or P(Yes) threshold).",
    },
    "physical_commonsense": {
        "name": "Physical Commonsense",
        "group": "general",
        "higher_is_better": True,
        "description": "Fraction of prompts with physical commonsense (PC=1).",
    },
    "joint_score": {
        "name": "Joint Score",
        "group": "general",
        "higher_is_better": True,
        "description": "Fraction of prompts with SA=1 and PC=1.",
    },
    "videophy_average": {
        "name": "VideoPhy Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean of SA and PC rates.",
        "primary": True,
    },
}


def _is_summary_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "").strip()
        for row in rows[: min(3, len(rows))]
    )


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        metric_id = str(row.get("metric_id") or row.get("Metric") or row.get("metric") or "").strip()
        score = row.get("score") if "score" in row else row.get("value")
        if not metric_id or score in (None, ""):
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        metrics[metric_id] = value / 100.0 if value > 1.0 else value
    return metrics


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
        raise ValueError(f"Unsupported VideoPhy JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def compute_videophy_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_path: Path | None = None,
) -> dict[str, Any]:
    if _is_summary_rows(rows):
        direct_metrics = _summary_metrics(rows)
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "summary_csv"},
        }

    official_scores = official_scores_from_records(list(rows), results_path)
    direct_metrics: dict[str, float | None] = {
        metric_id: (official_scores[metric_id].score if metric_id in official_scores else None)
        for metric_id in METRIC_ORDER
    }
    return {
        "metrics": direct_metrics,
        "components": {"row_count": len(rows), "format": "official_dimension_rows"},
    }
