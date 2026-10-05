"""LikePhys official ``results_<model>.json`` normalization for the contract evaluator.

The zoo contract evaluator flattens whatever the caller passes into a record list. For
LikePhys a record is normally one full ``results_<model>.json`` payload, so three shapes
are accepted here:

1. Payloads carrying ``scene_evaluations`` — recomputed with the official mis-rank formula.
2. Payloads carrying only ``misrank_metrics`` — the per-variation ratios ``evaluator.py``
   already wrote alongside the raw losses.
3. Flat summary rows carrying ``metric_id``/``score`` pairs.

Scenario attribution is kept when a payload records it (``scenario``/``data``/``dataset``);
otherwise variations are pooled without scenario-specific report filtering, and that is
recorded in the evidence so partial imports are never mistaken for the published number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import OfficialMetricScore
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_metrics import (
    DOMAIN_METRIC_IDS,
    METRIC_ORDER,
    TEMPORAL_DISORDER_VARIATION,
    VARIATION_WEIGHTED,
    misrank_by_variation,
)
from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_scenarios import (
    DOMAIN_SCENARIOS,
    SCENARIO_ORDER,
    domain_for_scenario,
    reported_variation_filter,
    scenario_id_for_variations,
)

JsonValue = Any

_SCENARIO_KEYS = ("scenario", "scenario_id", "data", "dataset", "data_name")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _to_float(value: JsonValue) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_scenario_id(record: Mapping[str, JsonValue]) -> str | None:
    for key in _SCENARIO_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value in SCENARIO_ORDER:
            return value
    return None


def _observed_variations(record: Mapping[str, JsonValue]) -> set[str]:
    scene_evaluations = record.get("scene_evaluations")
    if isinstance(scene_evaluations, Mapping):
        return {
            str(variation)
            for subgroup in scene_evaluations.values()
            if isinstance(subgroup, Mapping)
            for variation in subgroup
        }
    misrank_metrics = record.get("misrank_metrics")
    if isinstance(misrank_metrics, Mapping):
        return {str(name) for name in misrank_metrics}
    return set()


def _variations_from_record(record: Mapping[str, JsonValue]) -> tuple[str | None, dict[str, float], str]:
    """Return ``(scenario_id, {variation: misrank_rate}, source_kind)`` for one record.

    Result artifacts carry no scenario field, so when the caller did not supply one the
    scenario is recovered from its impossible-variation signature, which is unique across
    the 12 scenarios. Recovering it is what lets the reported variation filter apply.
    """
    scenario_id = _record_scenario_id(record) or scenario_id_for_variations(_observed_variations(record))
    scene_evaluations = record.get("scene_evaluations")
    if isinstance(scene_evaluations, Mapping):
        table = misrank_by_variation(
            scene_evaluations,
            scenario_id=scenario_id,
            apply_reported_filter=scenario_id is not None,
        )
        return scenario_id, {name: payload["misrank_rate"] for name, payload in table.items()}, "scene_evaluations"

    misrank_metrics = record.get("misrank_metrics")
    if isinstance(misrank_metrics, Mapping):
        excluded = reported_variation_filter(scenario_id) if scenario_id else frozenset()
        rates: dict[str, float] = {}
        for name, payload in misrank_metrics.items():
            if str(name) in excluded:
                continue
            value = payload.get("misrank_ratio") if isinstance(payload, Mapping) else payload
            rate = _to_float(value)
            if rate is not None:
                rates[str(name)] = rate
        return scenario_id, rates, "misrank_metrics"

    return scenario_id, {}, "none"


def _summary_metrics(records: Sequence[Mapping[str, JsonValue]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for record in records:
        metric_id = str(record.get("metric_id") or record.get("metric") or record.get("Metric") or "").strip()
        if metric_id not in METRIC_ORDER:
            continue
        score = _to_float(record.get("score") if "score" in record else record.get("value"))
        if score is None:
            continue
        metrics[metric_id] = score / 100.0 if score > 1.0 else score
    return metrics


def _likephys_scores(
    records: list[Mapping[str, JsonValue]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    """Normalize LikePhys official artifacts into WorldFoundry metric scores."""
    source_path = None if official_results_path is None else str(official_results_path)
    scores: dict[str, OfficialMetricScore] = {}

    per_scenario: dict[str, dict[str, float]] = {}
    unattributed: list[dict[str, float]] = []
    source_kinds: set[str] = set()
    for record in records:
        scenario_id, rates, source_kind = _variations_from_record(record)
        if not rates:
            continue
        source_kinds.add(source_kind)
        if scenario_id is None:
            unattributed.append(rates)
        else:
            per_scenario.setdefault(scenario_id, {}).update(rates)

    if not per_scenario and not unattributed:
        for metric_id, value in _summary_metrics(records).items():
            scores[metric_id] = OfficialMetricScore(
                score=value,
                raw_value=value,
                evidence={"source_path": source_path, "format": "summary_rows"},
            )
        return scores

    grouped_rates = [*per_scenario.values(), *unattributed]
    pooled = [rate for rates in grouped_rates for rate in rates.values()]
    scenario_means = [mean for rates in grouped_rates if (mean := _mean(list(rates.values()))) is not None]
    evidence_base: dict[str, JsonValue] = {
        "source_path": source_path,
        "source_kinds": sorted(source_kinds),
        "scenario_count": len(grouped_rates),
        "attributed_scenarios": sorted(per_scenario),
        "unattributed_record_count": len(unattributed),
        "reported_variation_filter_applied": bool(per_scenario) and not unattributed,
    }

    overall = _mean(pooled)
    if overall is not None:
        scores["likephys_misrank_rate"] = OfficialMetricScore(
            score=overall,
            raw_value=overall,
            evidence={**evidence_base, "aggregation": VARIATION_WEIGHTED, "variation_count": len(pooled)},
        )
    scenario_weighted = _mean(scenario_means)
    if scenario_weighted is not None:
        scores["likephys_dataset_weighted_misrank_rate"] = OfficialMetricScore(
            score=scenario_weighted,
            raw_value=scenario_weighted,
            evidence={**evidence_base, "aggregation": "dataset_weighted"},
        )

    for domain, scenario_ids in DOMAIN_SCENARIOS.items():
        domain_rates = [
            rate
            for scenario_id, rates in per_scenario.items()
            if domain_for_scenario(scenario_id) == domain
            for rate in rates.values()
        ]
        value = _mean(domain_rates)
        if value is None:
            continue
        scores[DOMAIN_METRIC_IDS[domain]] = OfficialMetricScore(
            score=value,
            raw_value=value,
            evidence={
                **evidence_base,
                "aggregation": VARIATION_WEIGHTED,
                "domain": domain,
                "scenarios": [scenario_id for scenario_id in scenario_ids if scenario_id in per_scenario],
            },
        )

    temporal = _mean(
        [rates[TEMPORAL_DISORDER_VARIATION] for rates in grouped_rates if TEMPORAL_DISORDER_VARIATION in rates]
    )
    if temporal is not None:
        scores["temporal_disorder_misrank_rate"] = OfficialMetricScore(
            score=temporal,
            raw_value=temporal,
            evidence={**evidence_base, "variation": TEMPORAL_DISORDER_VARIATION},
        )
    return scores


official_scores_from_records = _likephys_scores


__all__ = ["official_scores_from_records"]
