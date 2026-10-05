"""LikePhys mis-rank metric formulas over official ``results_<model>.json`` artifacts.

The official protocol has two aggregation layers:

``evaluator.py:compute_misrank_normalized``
    Per scenario, for every impossible variation, form all (valid, invalid) loss pairs
    inside a subgroup, count the pairs where the *valid* clip received the *higher*
    denoising loss, and average that ratio over subgroups.

``read_exp_final.py``
    Pools those per-variation ratios into scenario and model level numbers. It supports
    ``variation_weighted`` averaging (every variation counts equally — the default used
    for the published ranking) and ``dataset_weighted`` averaging (every scenario counts
    equally), and drops the variations listed in ``filter_config`` from the report.

Both layers are reproduced here. Mis-rank is an error rate: lower is better, ``0.0``
means the model ranked every valid clip below every impossible counterpart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.runners.likephys.likephys_scenarios import (
    DOMAIN_SCENARIOS,
    SCENARIO_ORDER,
    VALID_VARIATION,
    reported_variation_filter,
    scenario_id_for_variations,
)

JsonValue = Any

VARIATION_WEIGHTED = "variation_weighted"
DATASET_WEIGHTED = "dataset_weighted"

# Cross-cutting variation present in every scenario; reported as its own diagnostic.
TEMPORAL_DISORDER_VARIATION = "temporal_disorder"

METRIC_ORDER = (
    "likephys_misrank_rate",
    "likephys_dataset_weighted_misrank_rate",
    "rigid_body_misrank_rate",
    "fluid_misrank_rate",
    "deformable_misrank_rate",
    "optics_misrank_rate",
    "temporal_disorder_misrank_rate",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "likephys_misrank_rate": {
        "name": "LikePhys Mis-Rank Rate",
        "group": "aggregate",
        "higher_is_better": False,
        "description": (
            "Variation-weighted mean mis-rank rate over every scored (scenario, impossible "
            "variation) pair; the published LikePhys ranking metric."
        ),
        "primary": True,
    },
    "likephys_dataset_weighted_misrank_rate": {
        "name": "LikePhys Scenario-Weighted Mis-Rank Rate",
        "group": "aggregate",
        "higher_is_better": False,
        "description": "Mean over per-scenario mis-rank rates, weighting each of the 12 scenarios equally.",
    },
    "rigid_body_misrank_rate": {
        "name": "Rigid-Body Mis-Rank Rate",
        "group": "domain",
        "higher_is_better": False,
        "description": "Variation-weighted mis-rank over ball drop, ball collision, pendulum, block slide, and pyramid.",
    },
    "fluid_misrank_rate": {
        "name": "Fluid Mis-Rank Rate",
        "group": "domain",
        "higher_is_better": False,
        "description": "Variation-weighted mis-rank over droplet, faucet, and river fluid scenarios.",
    },
    "deformable_misrank_rate": {
        "name": "Deformable Mis-Rank Rate",
        "group": "domain",
        "higher_is_better": False,
        "description": "Variation-weighted mis-rank over cloth drape and waving flag scenarios.",
    },
    "optics_misrank_rate": {
        "name": "Optics Mis-Rank Rate",
        "group": "domain",
        "higher_is_better": False,
        "description": "Variation-weighted mis-rank over moving-light and moving-camera shadow scenarios.",
    },
    "temporal_disorder_misrank_rate": {
        "name": "Temporal Disorder Mis-Rank Rate",
        "group": "variation",
        "higher_is_better": False,
        "description": "Mis-rank restricted to the temporal_disorder variation, which every scenario provides.",
    },
}

DOMAIN_METRIC_IDS: Mapping[str, str] = {
    "rigid_body": "rigid_body_misrank_rate",
    "fluid": "fluid_misrank_rate",
    "deformable": "deformable_misrank_rate",
    "optics": "optics_misrank_rate",
}

RESULTS_FILENAME_PREFIX = "results_"
RESULTS_FILENAME_SUFFIX = ".json"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def video_loss(entry: Mapping[str, JsonValue]) -> float | None:
    """Return the ELBO-surrogate loss for one probed clip.

    ``read_exp_final.py`` averages ``loss_array`` (the per-timestep losses); ``loss`` is
    the same number precomputed by ``evaluator.py``. The array is preferred so partially
    written artifacts stay consistent with the official reader.
    """
    losses = entry.get("loss_array")
    if isinstance(losses, (list, tuple)):
        numeric = [float(value) for value in losses if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if numeric:
            return sum(numeric) / len(numeric)
    value = entry.get("loss")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _variation_losses(variation_entry: JsonValue) -> list[float]:
    if not isinstance(variation_entry, Mapping):
        return []
    losses: list[float] = []
    for entry in variation_entry.values():
        if not isinstance(entry, Mapping):
            continue
        loss = video_loss(entry)
        if loss is not None:
            losses.append(loss)
    return losses


def subgroup_misrank(valid_losses: Sequence[float], invalid_losses: Sequence[float]) -> tuple[float, int] | None:
    """Return ``(misrank_ratio, pair_count)`` for one subgroup, or ``None`` when unpairable.

    A pair is mis-ranked when the physically valid clip receives the higher loss, i.e. the
    model assigned the impossible clip the higher likelihood.
    """
    if not valid_losses or not invalid_losses:
        return None
    pair_count = len(valid_losses) * len(invalid_losses)
    misranked = sum(1 for valid in valid_losses for invalid in invalid_losses if valid > invalid)
    return misranked / pair_count, pair_count


def misrank_by_variation(
    scene_evaluations: Mapping[str, JsonValue],
    *,
    scenario_id: str | None = None,
    apply_reported_filter: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute per-variation mis-rank for one scenario's ``scene_evaluations`` block.

    Args:
        scene_evaluations: ``{subgroup_id: {variation: {video_name: entry}}}``.
        scenario_id: Scenario the block belongs to; enables the reported variation filter.
        apply_reported_filter: Drop the variations excluded from the published aggregates.
    """
    excluded = reported_variation_filter(scenario_id) if (apply_reported_filter and scenario_id) else frozenset()
    invalid_variations = sorted(
        {
            str(variation)
            for subgroup in scene_evaluations.values()
            if isinstance(subgroup, Mapping)
            for variation in subgroup
            if str(variation) != VALID_VARIATION
        }
    )
    results: dict[str, dict[str, Any]] = {}
    for variation in invalid_variations:
        if variation in excluded:
            continue
        subgroup_ratios: list[float] = []
        total_pairs = 0
        for subgroup in scene_evaluations.values():
            if not isinstance(subgroup, Mapping) or VALID_VARIATION not in subgroup or variation not in subgroup:
                continue
            computed = subgroup_misrank(
                _variation_losses(subgroup[VALID_VARIATION]),
                _variation_losses(subgroup[variation]),
            )
            if computed is None:
                continue
            ratio, pair_count = computed
            subgroup_ratios.append(ratio)
            total_pairs += pair_count
        ratio = _mean(subgroup_ratios)
        if ratio is None:
            continue
        results[variation] = {
            "misrank_rate": ratio,
            "subgroup_count": len(subgroup_ratios),
            "pair_count": total_pairs,
            "subgroup_misranks": subgroup_ratios,
        }
    return results


def scenario_misrank(
    scene_evaluations: Mapping[str, JsonValue],
    *,
    scenario_id: str | None = None,
    apply_reported_filter: bool = True,
) -> dict[str, Any]:
    """Aggregate one scenario into a variation table plus its scenario-level mean."""
    variations = misrank_by_variation(
        scene_evaluations,
        scenario_id=scenario_id,
        apply_reported_filter=apply_reported_filter,
    )
    rates = [payload["misrank_rate"] for payload in variations.values()]
    return {
        "scenario_id": scenario_id,
        "misrank_rate": _mean(rates),
        "variation_count": len(variations),
        "subgroup_count": max((payload["subgroup_count"] for payload in variations.values()), default=0),
        "pair_count": sum(payload["pair_count"] for payload in variations.values()),
        "variations": variations,
    }


def _scene_evaluations(payload: JsonValue) -> Mapping[str, JsonValue] | None:
    if not isinstance(payload, Mapping):
        return None
    block = payload.get("scene_evaluations")
    return block if isinstance(block, Mapping) else None


def parse_results_filename(path: Path) -> tuple[str | None, str | None]:
    """Return ``(scenario_id, model_key)`` inferred from a ``results_<model>.json`` path."""
    name = path.name
    model_key = None
    if name.startswith(RESULTS_FILENAME_PREFIX) and name.endswith(RESULTS_FILENAME_SUFFIX):
        model_key = name[len(RESULTS_FILENAME_PREFIX) : -len(RESULTS_FILENAME_SUFFIX)] or None
    scenario_id = path.parent.name if path.parent.name in SCENARIO_ORDER else None
    return scenario_id, model_key


def load_scenario_results(
    results_path: Path,
    *,
    model_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load LikePhys results into ``{scenario_id: {"payload": ..., "source_path": ...}}``.

    Accepts either one ``results_<model>.json`` file or an experiment directory laid out
    as ``<exp>/<scenario>/results_<model>.json``. When several probe backends are present
    under a directory, ``model_key`` selects one; otherwise the only backend found is used.

    Raises:
        FileNotFoundError: If ``results_path`` does not exist.
        ValueError: If no LikePhys result artifact could be resolved, or if a directory
            holds several backends and ``model_key`` was not supplied.
    """
    resolved = Path(results_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"LikePhys results path does not exist: {resolved}")

    candidates: list[Path]
    if resolved.is_file():
        candidates = [resolved]
    else:
        candidates = sorted(
            path
            for path in resolved.rglob(f"{RESULTS_FILENAME_PREFIX}*{RESULTS_FILENAME_SUFFIX}")
            if path.is_file()
        )
    if not candidates:
        raise ValueError(f"no LikePhys results_<model>.json artifacts found under {resolved}")

    discovered_models = {parse_results_filename(path)[1] for path in candidates}
    discovered_models.discard(None)
    if model_key is None and len(discovered_models) > 1:
        known = ", ".join(sorted(str(item) for item in discovered_models))
        raise ValueError(
            f"multiple LikePhys probe backends found under {resolved} ({known}); "
            "pass --probe-model to select one"
        )

    scenarios: dict[str, dict[str, Any]] = {}
    for path in candidates:
        scenario_id, file_model_key = parse_results_filename(path)
        if model_key is not None and file_model_key is not None and file_model_key != model_key:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = _scene_evaluations(payload)
        if block is None:
            continue
        inferred = scenario_id_for_variations(
            variation
            for subgroup in block.values()
            if isinstance(subgroup, Mapping)
            for variation in subgroup
        )
        key = scenario_id or inferred or str(payload.get("scenario") or payload.get("data") or path.parent.name)
        scenarios[key] = {
            "payload": payload,
            "scene_evaluations": block,
            "source_path": str(path.resolve()),
            "model_key": file_model_key,
        }
    if not scenarios:
        raise ValueError(f"no LikePhys scene_evaluations blocks found under {resolved}")
    return scenarios


def _pooled(values: Iterable[float]) -> float | None:
    collected = [float(value) for value in values]
    return _mean(collected)


def compute_likephys_metrics(
    *,
    scenario_results: Mapping[str, Mapping[str, Any]],
    apply_reported_filter: bool = True,
) -> dict[str, Any]:
    """Compute the LikePhys metric table from loaded per-scenario results.

    Args:
        scenario_results: Output of :func:`load_scenario_results`.
        apply_reported_filter: Apply ``read_exp_final.py``'s variation exclusions.
    """
    per_scenario: dict[str, dict[str, Any]] = {}
    for scenario_id in SCENARIO_ORDER:
        entry = scenario_results.get(scenario_id)
        if entry is None:
            continue
        summary = scenario_misrank(
            entry["scene_evaluations"],
            scenario_id=scenario_id,
            apply_reported_filter=apply_reported_filter,
        )
        summary["source_path"] = entry.get("source_path")
        per_scenario[scenario_id] = summary
    for scenario_id, entry in scenario_results.items():
        if scenario_id in per_scenario:
            continue
        summary = scenario_misrank(
            entry["scene_evaluations"],
            scenario_id=scenario_id if scenario_id in SCENARIO_ORDER else None,
            apply_reported_filter=apply_reported_filter,
        )
        summary["source_path"] = entry.get("source_path")
        per_scenario[scenario_id] = summary

    pooled_variation_rates: list[float] = []
    scenario_rates: list[float] = []
    for summary in per_scenario.values():
        pooled_variation_rates.extend(payload["misrank_rate"] for payload in summary["variations"].values())
        if summary["misrank_rate"] is not None:
            scenario_rates.append(summary["misrank_rate"])

    metrics: dict[str, float | None] = {metric_id: None for metric_id in METRIC_ORDER}
    metrics["likephys_misrank_rate"] = _pooled(pooled_variation_rates)
    metrics["likephys_dataset_weighted_misrank_rate"] = _pooled(scenario_rates)

    for domain, scenario_ids in DOMAIN_SCENARIOS.items():
        domain_rates = [
            payload["misrank_rate"]
            for scenario_id in scenario_ids
            if scenario_id in per_scenario
            for payload in per_scenario[scenario_id]["variations"].values()
        ]
        metrics[DOMAIN_METRIC_IDS[domain]] = _pooled(domain_rates)

    temporal_rates = [
        summary["variations"][TEMPORAL_DISORDER_VARIATION]["misrank_rate"]
        for summary in per_scenario.values()
        if TEMPORAL_DISORDER_VARIATION in summary["variations"]
    ]
    metrics["temporal_disorder_misrank_rate"] = _pooled(temporal_rates)

    model_keys = sorted({str(entry.get("model_key")) for entry in scenario_results.values() if entry.get("model_key")})
    return {
        "metrics": metrics,
        "per_scenario": per_scenario,
        "components": {
            "scenario_count": len(per_scenario),
            "scored_variation_count": len(pooled_variation_rates),
            "pair_count": sum(summary["pair_count"] for summary in per_scenario.values()),
            "aggregation": VARIATION_WEIGHTED,
            "reported_variation_filter_applied": apply_reported_filter,
            "probe_models": model_keys,
        },
    }


__all__ = [
    "DATASET_WEIGHTED",
    "DOMAIN_METRIC_IDS",
    "METRIC_ORDER",
    "METRIC_SPECS",
    "TEMPORAL_DISORDER_VARIATION",
    "VARIATION_WEIGHTED",
    "compute_likephys_metrics",
    "load_scenario_results",
    "misrank_by_variation",
    "parse_results_filename",
    "scenario_misrank",
    "subgroup_misrank",
    "video_loss",
]
