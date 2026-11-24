"""Validate official HarnessEval evidence and preserve its common-case aggregation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .runtime.io import read_json
from .runtime.protocols import FAMILIES
from .runtime.report import SCORING_POLICY, build_report, leaderboard_case_components

PRIMARY_METRIC = "overall_macro"
MACRO_METRICS = (
    PRIMARY_METRIC,
    "overall_core_macro",
    "overall_observation_macro",
    "observation_quality_macro",
    "observation_physical_plausibility_macro",
)
METRIC_IDS = (*MACRO_METRICS, *FAMILIES)


def unit_score(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite numeric score or null")
    if not 0 <= value <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return float(value)


def count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def normalize_results(path: Path, model_id: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    cases = []
    if path.is_dir():
        files = sorted(path.rglob("*.scores.json"))
        if not files:
            summary = path / "summary.json"
            if not summary.is_file():
                summary = path / "leaderboard_latest.json"
            return normalize_results(summary, model_id)
        payload = [read_json(file) for file in files]
    else:
        payload = read_json(path)
    if isinstance(payload, dict) and payload.get("schema_version") == "harnesseval.skill_eval":
        payload = [payload]
    if isinstance(payload, list):
        seen = set()
        for record in payload:
            if not isinstance(record, dict) or record.get("schema_version") != "harnesseval.skill_eval":
                raise ValueError("expected harnesseval.skill_eval case records")
            model, case = record.get("model_id"), record.get("case_id")
            family = (record.get("taxonomy") or {}).get("probe_family")
            if (
                not isinstance(model, str)
                or not model
                or not isinstance(case, str)
                or not case
                or family not in FAMILIES
            ):
                raise ValueError("case requires model_id, case_id and a supported probe_family")
            if (model, case) in seen:
                raise ValueError(f"duplicate case for model {model}: {case}")
            seen.add((model, case))
            unit_score((record.get("case_score") or {}).get("score"), f"{case}.core")
            for skill, value in ((record.get("observation_quality") or {}).get("dimensions") or {}).items():
                unit_score(value, f"{case}.{skill}")
            cases.append(record)
        report = build_report(cases)
        kind = "case_scores"
    elif isinstance(payload, dict) and payload.get("schema_version") == "harnesseval.report":
        report, kind = payload, "official_report"
    else:
        raise ValueError("expected harnesseval.report or harnesseval.skill_eval results")
    if report.get("scoring_policy") != SCORING_POLICY:
        raise ValueError("unsupported HarnessEval scoring policy")
    rows = report.get("leaderboard")
    if not isinstance(rows, list) or not rows:
        raise ValueError("no scored models found")
    ids = [row.get("model_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid or duplicate leaderboard model IDs")
    if model_id is None and len(rows) != 1:
        raise ValueError("multiple models found; select one with --model-id")
    selected = model_id or ids[0]
    if selected not in ids:
        raise ValueError(f"model not found: {selected}")
    common = report.get("common_case_counts") or {}
    for family in FAMILIES:
        count(common.get(family), f"common_case_counts.{family}")
    # Validate every model; selecting a model must not change the common-case cohort.
    for row in rows:
        family_scores = row.get("family_scores") or {}
        for family in FAMILIES:
            score = unit_score(family_scores.get(family), family)
            if (score is None) != (common[family] == 0):
                raise ValueError(f"score/coverage mismatch for {family}")
            coverage = count((row.get("coverage") or {}).get(family), f"coverage.{family}")
            if coverage < common[family]:
                raise ValueError(f"common cases exceed model coverage for {family}")
        for key in MACRO_METRICS:
            unit_score(row.get(key), key)
        expected = sum(family_scores.values()) / len(FAMILIES) if all(common.values()) else None
        actual = row.get(PRIMARY_METRIC)
        if (expected is None) != (actual is None) or (
            expected is not None and not math.isclose(expected, actual, abs_tol=1e-6)
        ):
            raise ValueError("overall_macro does not match the six-family macro average")
    row = rows[ids.index(selected)]
    metrics = {key: row.get(key) for key in MACRO_METRICS}
    metrics.update({family: row["family_scores"].get(family) for family in FAMILIES})
    return {
        "kind": kind,
        "model_id": selected,
        "scores": metrics,
        "case_count": sum(row["coverage"][family] for family in FAMILIES),
        "common_case_counts": common,
        "report": report,
        "case_records": [
            {**record, "published_components": leaderboard_case_components(record)}
            for record in cases
            if record["model_id"] == selected
        ],
    }
