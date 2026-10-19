"""PhyGenBench metric formulas from official summary rows or PhyGenEval result JSON."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float
from worldfoundry.evaluation.tasks.execution.runners.phygenbench.phygenbench_official_scoring import (
    official_scores_from_records,
)

METRIC_ORDER = (
    "physical_commonsense",
    "physical_law_adherence",
    "semantic_adherence",
    "phygenbench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "physical_commonsense": {
        "name": "Physical Commonsense",
        "group": "general",
        "higher_is_better": True,
        "description": "Physical commonsense normalized benchmark score.",
    },
    "physical_law_adherence": {
        "name": "Physical Law Adherence",
        "group": "general",
        "higher_is_better": True,
        "description": "Physical law adherence normalized benchmark score.",
    },
    "semantic_adherence": {
        "name": "Semantic Adherence",
        "group": "general",
        "higher_is_better": True,
        "description": "Semantic adherence normalized benchmark score.",
    },
    "phygenbench_average": {
        "name": "PhyGenBench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "description": "Mean over PhyGenBench component scores.",
        "primary": True,
    },
}

_STAGE_SCALE_MAX = 3.0
_AVERAGE_SUFFIX = re.compile(r"^(?P<prefix>.+)_(average|closed|open)$")


def _normalize_unit(value: float, *, scale_max: float | None = None) -> float:
    if scale_max is not None and value > 1.0:
        return value / scale_max
    if value > 1.0:
        return value / 100.0
    return value


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
        metrics[metric_id] = _normalize_unit(score)
    return metrics


def _average_field_name(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for row in rows:
        for key in row:
            match = _AVERAGE_SUFFIX.match(str(key))
            if match is not None:
                return str(key)
    return None


def _stage_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "single": _to_float(row.get("single")),
        "multi_gpt": _to_float(row.get("multi_gpt")),
        "video_gpt": _to_float(row.get("video_gpt")),
        "semantic_score": _to_float(
            row.get("semantic_adherence")
            or row.get("semantic_score")
            or row.get("total_Score")
            or row.get("object_score")
        ),
    }


def _derive_per_sample_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    average_field = _average_field_name(rows)
    commonsense_values: list[float] = []
    law_values: list[float] = []
    semantic_values: list[float] = []
    average_values: list[float] = []

    for row in rows:
        stages = _stage_values(row)
        single = stages["single"]
        multi = stages["multi_gpt"]
        video = stages["video_gpt"]
        semantic = stages["semantic_score"]

        if single is not None and multi is not None:
            law_values.append(_normalize_unit((single + multi) / 2.0, scale_max=_STAGE_SCALE_MAX))
        if single is not None and multi is not None and video is not None:
            commonsense_values.append(_normalize_unit((single + multi + video) / 3.0, scale_max=_STAGE_SCALE_MAX))
        if semantic is not None:
            semantic_values.append(_normalize_unit(semantic, scale_max=5.0 if semantic > 1.0 else None))
        elif video is not None:
            semantic_values.append(_normalize_unit(video, scale_max=_STAGE_SCALE_MAX))

        if average_field is not None:
            average = _to_float(row.get(average_field))
            if average is not None:
                average_values.append(_normalize_unit(average, scale_max=_STAGE_SCALE_MAX))

    metrics: dict[str, float] = {}
    if law_values:
        metrics["physical_law_adherence"] = sum(law_values) / len(law_values)
    if commonsense_values:
        metrics["physical_commonsense"] = sum(commonsense_values) / len(commonsense_values)
    if semantic_values:
        metrics["semantic_adherence"] = sum(semantic_values) / len(semantic_values)

    if average_values:
        metrics["phygenbench_average"] = sum(average_values) / len(average_values)
    else:
        component_ids = ("physical_commonsense", "physical_law_adherence", "semantic_adherence")
        component_values = [metrics[metric_id] for metric_id in component_ids if metric_id in metrics]
        if len(component_values) == len(component_ids):
            metrics["phygenbench_average"] = sum(component_values) / len(component_values)
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
        raise ValueError(f"Unsupported PhyGenBench JSON results shape: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def compute_phygenbench_metrics(
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
    if official_scores:
        direct_metrics: dict[str, float | None] = {
            metric_id: (official_scores[metric_id].score if metric_id in official_scores else None)
            for metric_id in METRIC_ORDER
        }
        return {
            "metrics": direct_metrics,
            "components": {"row_count": len(rows), "format": "official_dimension_rows"},
        }

    derived = _derive_per_sample_metrics(rows)
    direct_metrics = {metric_id: derived.get(metric_id) for metric_id in METRIC_ORDER}
    return {
        "metrics": direct_metrics,
        "components": {
            "sample_count": len(rows),
            "format": "phygeneval_per_sample_json",
        },
    }
