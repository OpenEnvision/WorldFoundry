"""PhyGround official scores.json normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.framework.official_result_scoring import (
    OfficialMetricScore,
    _first_nested_value,
    _first_record_field,
    _looks_likert,
    _score_from_records,
    _unit_score,
)

JsonValue = Any

PHYGROUND_LAW_GROUPS = {
    "solid_body_score": ("gravity", "inertia", "momentum", "impenetrability", "collision", "material"),
    "fluid_dynamics_score": ("buoyancy", "displacement", "flow_dynamics", "boundary_interaction", "fluid_continuity"),
    "optics_score": ("reflection", "shadow"),
}


def _first_value(row: Mapping[str, JsonValue], *keys: str) -> JsonValue:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _field_aliases(metric_id: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    base = [metric_id, metric_id.replace("_", "-"), metric_id.replace("_", " "), metric_id.upper()]
    return tuple(dict.fromkeys([*base, *aliases]))


def _phyground_scores(
    records: list[Mapping[str, JsonValue]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Normalize PhyGround scores.json outputs from PhyJudge/API judges.

    Args:
        records: PhyGround per-video results or aggregate rows.
        official_results_path: Source path for diagnostics.
    """

    scores: dict[str, OfficialMetricScore] = {}
    for metric_id, aliases in {
        "semantic_adherence": ("SA", "sa"),
        "physical_temporal_validity": ("PTV", "ptv"),
        "persistence": ("persistence",),
    }.items():
        score = _phyground_general_score(records, metric_id, aliases, official_results_path)
        if score is None:
            score = _score_from_records(records, metric_id, official_results_path, aliases=aliases, scale_max=5.0)
        if score is not None:
            scores[metric_id] = score
    for metric_id, laws in PHYGROUND_LAW_GROUPS.items():
        score = _phyground_law_group_score(records, metric_id, laws, official_results_path)
        if score is not None:
            scores[metric_id] = score
    overall = _score_from_records(records, "phyground_overall", official_results_path, aliases=("overall",))
    if overall is None:
        component_ids = (
            "semantic_adherence",
            "physical_temporal_validity",
            "persistence",
            "solid_body_score",
            "fluid_dynamics_score",
            "optics_score",
        )
        component_values = [scores[metric_id].score for metric_id in component_ids if metric_id in scores]
        if len(component_values) == len(component_ids):
            value = sum(component_values) / len(component_values)
            overall = OfficialMetricScore(
                score=value,
                raw_value=value,
                evidence={
                    "source_path": None if official_results_path is None else str(official_results_path),
                    "aggregation": "mean_of_general_and_domain_scores",
                    "components": list(component_ids),
                },
            )
    if overall is not None:
        scores["phyground_overall"] = overall
    return scores


def _phyground_general_score(
    records: list[Mapping[str, JsonValue]],
    metric_id: str,
    aliases: tuple[str, ...],
    official_results_path: Path | None,
) -> OfficialMetricScore | None:
    """Macro-average PhyGround general scores over videos.

    Args:
        records: PhyGround per-video or per-annotation rows.
        metric_id: Canonical metric id.
        aliases: Official field aliases to accept.
        official_results_path: Source path for diagnostics.
    """

    fields = _field_aliases(metric_id, aliases)
    grouped: dict[str, list[float]] = {}
    source_fields: list[str] = []
    for record in records:
        field = _first_record_field(record, fields)
        if field is None:
            continue
        normalized = _unit_score(_first_nested_value(record, (field,)), scale_max=5.0)
        if normalized is None:
            continue
        grouped.setdefault(_phyground_sample_key(record), []).append(normalized)
        source_fields.append(field)
    if not grouped:
        return None
    video_scores = [sum(values) / len(values) for values in grouped.values() if values]
    score = sum(video_scores) / len(video_scores)
    return OfficialMetricScore(
        score=score,
        raw_value=score,
        evidence={
            "source_path": None if official_results_path is None else str(official_results_path),
            "source_fields": sorted(set(source_fields)),
            "sample_count": len(video_scores),
            "annotation_count": sum(len(values) for values in grouped.values()),
            "aggregation": "macro_average_over_video_level_scores",
            "normalization": "scale_max:5",
        },
    )


def _phyground_law_group_score(
    records: list[Mapping[str, JsonValue]],
    metric_id: str,
    laws: tuple[str, ...],
    official_results_path: Path | None,
) -> OfficialMetricScore | None:
    """Aggregate PhyGround per-law scores into a domain score.

    Args:
        records: PhyGround scores.json result rows.
        metric_id: Domain metric id.
        laws: Physical-law keys belonging to the domain.
        official_results_path: Source path for diagnostics.
    """

    values = []
    law_counts: dict[str, int] = {}
    grouped: dict[str, list[float]] = {}
    for record in records:
        law_scores = _phyground_law_scores(record, laws)
        if law_scores:
            grouped.setdefault(_phyground_sample_key(record), []).extend(law_scores.values())
        for law, score in law_scores.items():
            values.append(score)
            law_counts[law] = law_counts.get(law, 0) + 1
    if not values:
        return _score_from_records(records, metric_id, official_results_path)
    video_scores = [sum(scores) / len(scores) for scores in grouped.values() if scores]
    score = sum(video_scores) / len(video_scores)
    return OfficialMetricScore(
        score=score,
        raw_value=score,
        evidence={
            "source_path": None if official_results_path is None else str(official_results_path),
            "aggregation": "macro_average_over_video_level_applicable_laws",
            "laws": list(laws),
            "law_counts": law_counts,
            "sample_count": len(video_scores),
            "annotation_law_score_count": len(values),
        },
    )


def _phyground_law_scores(record: Mapping[str, JsonValue], laws: tuple[str, ...]) -> dict[str, float]:
    """Extract normalized PhyGround law scores from one row.

    Args:
        record: Per-video PhyGround judge result.
        laws: Laws to extract.
    """

    result: dict[str, float] = {}
    physical = record.get("physical")
    law_container = physical.get("laws") if isinstance(physical, Mapping) else None
    scores = record.get("scores")
    if not isinstance(law_container, Mapping) and isinstance(scores, Mapping):
        physical_scores = scores.get("physical")
        if isinstance(physical_scores, Mapping):
            law_container = physical_scores
    if not isinstance(law_container, Mapping):
        law_container = record.get("laws")
    if not isinstance(law_container, Mapping):
        return result
    for law in laws:
        row = law_container.get(law)
        score = row.get("score") if isinstance(row, Mapping) else row
        normalized = _unit_score(score, scale_max=5.0 if _looks_likert(score) else None)
        if normalized is not None:
            result[law] = normalized
    return result


def _phyground_sample_key(record: Mapping[str, JsonValue]) -> str:
    """Return a PhyGround sample key, including model id when present."""

    sample_id = _first_value(record, "video", "id_stem", "sample_id", "video_id", "prompt_id", "id", "uid")
    model_id = record.get("model")
    if model_id not in (None, "") and sample_id not in (None, ""):
        return f"{model_id}:{sample_id}"
    if sample_id not in (None, ""):
        return str(sample_id)
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


official_scores_from_records = _phyground_scores
