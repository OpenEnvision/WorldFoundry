"""Shared helpers for normalizing official benchmark result exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.io import optional_float

JsonValue = Any


@dataclass(frozen=True)
class OfficialMetricScore:
    score: float
    raw_value: JsonValue
    evidence: Mapping[str, JsonValue]


def _dimension_scores(
    records: list[Mapping[str, JsonValue]],
    dimension_ids: tuple[str, ...],
    average_id: str,
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Aggregate common official dimension score files.

    Args:
        records: Official rows with either aggregate or per-sample dimension fields.
        dimension_ids: Component metric ids to average independently.
        average_id: Published aggregate metric id.
        official_results_path: Source path for diagnostics.
    """

    scores = {
        metric_id: score
        for metric_id in dimension_ids
        if (score := _score_from_records(records, metric_id, official_results_path)) is not None
    }
    aggregate = _score_from_records(records, average_id, official_results_path)
    if aggregate is not None:
        scores[average_id] = aggregate
    else:
        component_values = [scores[metric_id].score for metric_id in dimension_ids if metric_id in scores]
        if len(component_values) == len(dimension_ids):
            value = sum(component_values) / len(component_values)
            scores[average_id] = OfficialMetricScore(
                score=value,
                raw_value=value,
                evidence={
                    "source_path": None if official_results_path is None else str(official_results_path),
                    "aggregation": "mean_of_official_dimensions",
                    "components": list(dimension_ids),
                },
            )
    return scores




def _score_from_records(
    records: list[Mapping[str, JsonValue]],
    metric_id: str,
    official_results_path: Path | None,
    *,
    aliases: tuple[str, ...] = (),
    scale_max: float | None = None,
) -> OfficialMetricScore | None:
    """Find and average numeric official score fields.

    Args:
        records: Loaded official result records.
        metric_id: Canonical metric id.
        official_results_path: Source path for diagnostics.
        aliases: Official field aliases to accept.
        scale_max: Optional maximum scale value for Likert-style scores.
    """

    fields = _field_aliases(metric_id, aliases)
    values: list[float] = []
    source_fields: list[str] = []
    for record in records:
        field = _first_record_field(record, fields)
        if field is None:
            continue
        normalized = _unit_score(_first_nested_value(record, (field,)), scale_max=scale_max)
        if normalized is None:
            continue
        values.append(normalized)
        source_fields.append(field)
    if not values:
        return None
    score = sum(values) / len(values)
    return OfficialMetricScore(
        score=score,
        raw_value=score,
        evidence={
            "source_path": None if official_results_path is None else str(official_results_path),
            "source_fields": sorted(set(source_fields)),
            "sample_count": len(values),
            "normalization": "percent_or_fraction_to_unit" if scale_max is None else f"scale_max:{scale_max:g}",
        },
    )




def _joint_score_from_records(
    records: list[Mapping[str, JsonValue]],
    official_results_path: Path | None,
) -> OfficialMetricScore | None:
    """Compute VideoPhy joint performance from official SA/PC labels.

    Args:
        records: Official rows with joint or SA/PC fields.
        official_results_path: Source path for diagnostics.
    """

    explicit = _score_from_records(records, "joint_score", official_results_path, aliases=("joint", "SA=1, PC=1", "SA_PC"))
    if explicit is not None:
        return explicit
    values = []
    for record in records:
        sa = _first_nested_value(record, _field_aliases("semantic_adherence", ("sa", "SA")))
        pc = _first_nested_value(record, _field_aliases("physical_commonsense", ("pc", "PC")))
        sa_score = _unit_score(sa, scale_max=5.0 if _looks_likert(sa) else None)
        pc_score = _unit_score(pc, scale_max=5.0 if _looks_likert(pc) else None)
        if sa_score is not None and pc_score is not None:
            values.append(sa_score >= 0.8 and pc_score >= 0.8)
    if not values:
        return None
    passed = sum(1 for value in values if value)
    return OfficialMetricScore(
        score=passed / len(values),
        raw_value={"passed": passed, "total": len(values)},
        evidence={
            "source_path": None if official_results_path is None else str(official_results_path),
            "formula": "fraction(sa >= 4 and pc >= 4) for 5-point labels, or fraction(sa >= 0.8 and pc >= 0.8) after normalization",
            "sample_count": len(values),
        },
    )




def _rule_followed_rate(
    records: list[Mapping[str, JsonValue]],
    official_results_path: Path | None,
) -> OfficialMetricScore | None:
    """Compute the VideoPhy2 followed-rule rate from official rule labels.

    Args:
        records: Official rows with physical-rule result lists.
        official_results_path: Source path for diagnostics.
    """

    correct = 0
    total = 0
    for record in records:
        followed = _sequence_value(_first_nested_value(record, ("physics_rules_followed", "rules_followed", "followed_rules")))
        violated = _sequence_value(_first_nested_value(record, ("physics_rules_unfollowed", "rules_unfollowed", "violated_rules")))
        for _ in followed:
            correct += 1
            total += 1
        for _ in violated:
            total += 1
    if total == 0:
        return None
    return OfficialMetricScore(
        score=correct / total,
        raw_value={"followed": correct, "scored_rules": total},
        evidence={
            "source_path": None if official_results_path is None else str(official_results_path),
            "formula": "followed_rules / (followed_rules + violated_rules); cannot-be-determined rules excluded",
        },
    )




def _field_aliases(metric_id: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Build common official result field aliases for one metric.

    Args:
        metric_id: Canonical metric id.
        aliases: Benchmark-specific aliases.
    """

    base = [metric_id, metric_id.replace("_", "-"), metric_id.replace("_", " "), metric_id.upper()]
    return tuple(dict.fromkeys([*base, *aliases]))




def _first_record_field(record: Mapping[str, JsonValue], fields: tuple[str, ...]) -> str | None:
    """Return the first field present at the top level or inside score containers.

    Args:
        record: Official result row.
        fields: Candidate field aliases.
    """

    for field in fields:
        if _first_nested_value(record, (field,)) not in (None, ""):
            return field
    metric_id = record.get("metric_id") or record.get("name") or record.get("key")
    if metric_id is not None and _metric_key(str(metric_id)) in {_metric_key(field) for field in fields}:
        for field in ("score", "value", "raw_value", "normalized_value"):
            if record.get(field) not in (None, ""):
                return field
    return None




def _first_nested_value(record: Mapping[str, JsonValue], fields: tuple[str, ...]) -> JsonValue:
    """Return a field from a row or common nested score containers.

    Args:
        record: Official result row.
        fields: Candidate field aliases.
    """

    for field in fields:
        if field in record:
            return record[field]
    for container_name in ("scores", "metrics", "benchmark_metric_results", "metric_scores", "general"):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            for field in fields:
                if field in container:
                    return container[field]
            for nested_name in ("general", "physical", "laws"):
                nested = container.get(nested_name)
                if isinstance(nested, Mapping):
                    for field in fields:
                        if field in nested:
                            return nested[field]
    return None




def _unit_score(value: JsonValue, *, scale_max: float | None = None) -> float | None:
    """Normalize official numeric labels to the unit interval.

    Args:
        value: Official score, percentage, boolean, or Likert value.
        scale_max: Optional scale maximum for ordinal scores.
    """

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = optional_float(value)
    if numeric is None:
        return None
    if scale_max is not None:
        return min(1.0, max(0.0, numeric / scale_max))
    if numeric < 0:
        return None
    if numeric <= 1:
        return numeric
    if numeric <= 100:
        return numeric / 100.0
    return None




def _looks_likert(value: JsonValue) -> bool:
    """Return whether a value appears to be a 1-5 ordinal rating.

    Args:
        value: Candidate official score value.
    """

    numeric = optional_float(value)
    return numeric is not None and 1.0 <= numeric <= 5.0 and float(numeric).is_integer()




def _sequence_value(value: JsonValue) -> tuple[JsonValue, ...]:
    """Return a JSON sequence while excluding string-like values.

    Args:
        value: Candidate list-like object.
    """

    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    return ()




def _metric_key(value: str) -> str:
    """Normalize metric and category keys for alias matching.

    Args:
        value: Raw metric or field key.
    """

    return value.strip().casefold().replace("_", "-").replace(" ", "-")




def _loose_id(value: str) -> str:
    """Normalize ids for model/result matching while preserving explicit ids elsewhere."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())




def _float_list(value: JsonValue) -> list[float]:
    """Parse official list-valued CSV cells into floats.

    Args:
        value: CSV cell, JSON list, or scalar field.
    """

    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [float(item) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if isinstance(value, str):
        return [float(item) for item in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []
