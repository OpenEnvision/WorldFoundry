"""PhyGround metric formulas from official scores.json or summary rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float
from worldfoundry.evaluation.tasks.execution.runners.phyground.phyground_official_scoring import (
    official_scores_from_records,
)

METRIC_ORDER = (
    "semantic_adherence",
    "physical_temporal_validity",
    "persistence",
    "solid_body_score",
    "fluid_dynamics_score",
    "optics_score",
    "phyground_overall",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "semantic_adherence": {
        "name": "Semantic Adherence",
        "group": "general",
        "higher_is_better": True,
        "description": "Semantic adherence (SA) normalized benchmark score.",
    },
    "physical_temporal_validity": {
        "name": "Physical Temporal Validity",
        "group": "general",
        "higher_is_better": True,
        "description": "Physical temporal validity (PTV) normalized benchmark score.",
    },
    "persistence": {
        "name": "Persistence",
        "group": "general",
        "higher_is_better": True,
        "description": "Object persistence normalized benchmark score.",
    },
    "solid_body_score": {
        "name": "Solid-Body Score",
        "group": "physics",
        "higher_is_better": True,
        "description": "Solid-body mechanics law aggregate.",
    },
    "fluid_dynamics_score": {
        "name": "Fluid Dynamics Score",
        "group": "physics",
        "higher_is_better": True,
        "description": "Fluid dynamics law aggregate.",
    },
    "optics_score": {
        "name": "Optics Score",
        "group": "physics",
        "higher_is_better": True,
        "description": "Optics law aggregate.",
    },
    "phyground_overall": {
        "name": "PhyGround Overall",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean over general and domain PhyGround scores.",
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
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if not metric_id or score is None:
            continue
        metrics[metric_id] = score / 100.0 if score > 1.0 else score
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
        raise ValueError(f"Unsupported PhyGround JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def compute_phyground_metrics(
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
        "components": {
            "video_count": len(rows),
            "format": "scores_json",
        },
    }
