"""Official score schemas and result loading for both Physics-IQ protocols."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .physics_iq_runtime import score_raw_metrics_csv
from .protocols import ORIGINAL, PhysicsIQProtocolSpec

ORIGINAL_METRIC_KEYS: dict[str, str] = {
    "physics_iq_score": "final_score_orig",
    "physics_iq_stable_score": "final_score_stable",
    "physics_iq_spatiotemporal": "score_spatiotemporal",
    "physics_iq_spatial": "score_spatial",
    "physics_iq_weighted_spatial": "score_weighted_spatial",
    "physics_iq_mse_penalty": "score_mse",
}

VERIFIED_METRIC_KEYS: dict[str, str] = {
    "physics_iq_verified_score": "final_score_view",
    "physics_iq_verified_spatiotemporal": "score_spatiotemporal_view",
    "physics_iq_verified_spatial": "score_spatial_view",
    "physics_iq_verified_weighted_spatial": "score_weighted_spatial_view",
    "physics_iq_verified_mse": "score_mse_view",
}


def metric_key_map(spec: PhysicsIQProtocolSpec) -> dict[str, str]:
    return VERIFIED_METRIC_KEYS if spec.protocol == "verified" else ORIGINAL_METRIC_KEYS


def metric_specs(spec: PhysicsIQProtocolSpec) -> dict[str, dict[str, Any]]:
    prefix = "Physics-IQ Verified" if spec.protocol == "verified" else "Physics-IQ"
    result: dict[str, dict[str, Any]] = {}
    for metric_id in metric_key_map(spec):
        suffix = metric_id.removeprefix("physics_iq_verified_").removeprefix("physics_iq_")
        result[metric_id] = {
            "name": prefix if metric_id == spec.primary_metric_id else f"{prefix} {suffix.replace('_', ' ').title()}",
            "group": "aggregate" if metric_id == spec.primary_metric_id else "physics_iq_component",
            "higher_is_better": metric_id != "physics_iq_mse_penalty",
            "primary": metric_id == spec.primary_metric_id,
        }
    return result


def metric_values_from_scores(
    scores: Mapping[str, Any],
    spec: PhysicsIQProtocolSpec,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for metric_id, score_key in metric_key_map(spec).items():
        value = scores.get(score_key)
        try:
            values[metric_id] = None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            values[metric_id] = None
    return values


def _summary_scores(rows: list[dict[str, Any]], spec: PhysicsIQProtocolSpec) -> dict[str, float]:
    direct: dict[str, float] = {}
    reverse = {metric_id: score_key for metric_id, score_key in metric_key_map(spec).items()}
    for row in rows:
        key = str(row.get("metric_id") or row.get("metric") or row.get("Metric") or "").strip()
        value = row.get("score", row.get("value"))
        if not key or value in (None, ""):
            continue
        score_key = reverse.get(key, key)
        numeric = float(value)
        direct[score_key] = numeric / 100.0 if score_key.startswith("final_score") and numeric > 1 else numeric
    return direct


def load_official_results(
    path: Path,
    spec: PhysicsIQProtocolSpec = ORIGINAL,
    *,
    lazy_integrity: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Load official metrics JSON, a summary CSV, or the raw scenario table."""

    path = path.expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and isinstance(payload.get("scores"), Mapping):
            payload = payload["scores"]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Physics-IQ metrics JSON must be an object: {path}")
        return ({str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}, [])

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Physics-IQ results are empty: {path}")
    if "scenario" in rows[0]:
        return score_raw_metrics_csv(path, lazy_integrity=lazy_integrity), rows
    return _summary_scores(rows, spec), []


# Backward-compatible constants for callers that only know Physics-IQ Original.
METRIC_ORDER = tuple(ORIGINAL_METRIC_KEYS)
METRIC_SPECS = metric_specs(ORIGINAL)
