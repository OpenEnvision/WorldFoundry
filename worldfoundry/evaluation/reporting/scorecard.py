"""Build the normalized WorldFoundry scorecard payload.

The scorecard consolidates run, benchmark, model, dataset, generation, and
metrics metadata into a single canonical JSON document that downstream
consumption (indexing, comparison, reporting).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.utils import write_json

from .comparison_identity import build_comparison_identity

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
_MISSING_EVIDENCE_REASON = "missing official/full-suite leaderboard evidence gate"
_EVIDENCE_CONTAINER_KEYS = frozenset(
    {
        "leaderboard_evidence",
        "leaderboard_eligibility",
        "leaderboard_eligibility_evidence",
        "fairness_gate",
    }
)
_DIRECT_EVIDENCE_KEYS = frozenset(
    {
        "official_full_suite_evidence",
        "official_leaderboard_evidence",
        "full_suite_evidence",
    }
)

# ── Internal helpers ─────────────────────────────────────────

def _worldfoundry_version() -> str:
    """Return the installed ``worldfoundry`` package version, or ``"unknown"``."""
    try:
        return metadata.version("worldfoundry")
    except metadata.PackageNotFoundError:
        return "unknown"


def _normalize_gate_kind(value: Any) -> str:
    """Normalise a gate kind string to a canonical underscore-separated form."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _mapping_evidence_reason(value: Mapping[str, Any]) -> str | None:
    """Determine the evidence reason from a mapping payload containing gate or eligibility fields."""
    for key in ("official_full_suite_evidence", "official_full_suite"):
        if value.get(key) is True:
            return key

    if value.get("official") is True and value.get("full_suite") is True:
        return "official_and_full_suite"

    gate_kind = _normalize_gate_kind(
        value.get("gate")
        or value.get("kind")
        or value.get("evidence")
        or value.get("evidence_kind")
    )
    if gate_kind in {"official_full_suite", "official_leaderboard", "full_suite"} and (
        value.get("passed") is True
        or value.get("present") is True
        or value.get("valid") is True
        or value.get("leaderboard_eligible") is True
    ):
        return f"{gate_kind}_gate"

    checks = value.get("checks")
    if (
        value.get("leaderboard_eligible") is True
        and isinstance(checks, Mapping)
        and checks.get("official_runtime_evidence") is True
    ):
        return "fairness_gate_official_runtime_evidence"

    return None


def _evidence_reason(value: Any, source_path: str) -> str | None:
    """Determine the evidence reason from a direct boolean or mapping payload."""
    source_key = source_path.rsplit(".", maxsplit=1)[-1]
    if value is True and source_key in _DIRECT_EVIDENCE_KEYS:
        return source_key
    if isinstance(value, Mapping):
        return _mapping_evidence_reason(value)
    return None


def _leaderboard_evidence_gate(
    *,
    leaderboard_evidence: Mapping[str, Any] | None,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    dataset: Mapping[str, Any],
    metrics_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the leaderboard evidence gate across all candidate payloads."""
    candidates: list[tuple[str, Any]] = []
    if leaderboard_evidence is not None:
        candidates.append(("leaderboard_evidence", leaderboard_evidence))

    for payload_name, payload in (
        ("run", run),
        ("benchmark", benchmark),
        ("dataset", dataset),
        ("metrics_summary", metrics_summary),
    ):
        for key in sorted(_EVIDENCE_CONTAINER_KEYS.union(_DIRECT_EVIDENCE_KEYS)):
            if key in payload:
                candidates.append((f"{payload_name}.{key}", payload[key]))

    sources = [
        {"path": source_path, "reason": reason}
        for source_path, value in candidates
        if (reason := _evidence_reason(value, source_path)) is not None
    ]
    return {
        "required": "official_full_suite",
        "present": bool(sources),
        "source_paths": [source["path"] for source in sources],
        "sources": sources,
    }


def build_scorecard(
    *,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    generation: Mapping[str, Any],
    metrics_summary: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    skipped: Mapping[str, Any] | None = None,
    leaderboard_evidence: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    evaluation_kind: str = "existing_results",
    comparison_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized scorecard for in-process scoring runs."""

    run_payload = dict(run)
    run_payload.setdefault("worldfoundry_version", _worldfoundry_version())
    benchmark_payload = dict(benchmark)
    dataset_payload = dict(dataset)
    metrics_summary_payload = dict(metrics_summary)

    leaderboard = dict(metrics_summary_payload.get("leaderboard") or {})
    per_metric = dict(metrics_summary_payload.get("per_metric") or {})
    generation_payload = dict(generation)
    failed_samples = int(metrics_summary_payload.get("failed_samples") or 0)
    skip_payload = dict(skipped or {})
    provenance_payload = dict(provenance or {})
    provenance_claim = provenance_payload.get("claim")
    provenance_claim = provenance_claim if isinstance(provenance_claim, Mapping) else {}
    provenance_candidate = (
        provenance_claim.get("leaderboard_candidate") is True
        if provenance_payload
        else True
    )
    evidence_gate = _leaderboard_evidence_gate(
        leaderboard_evidence=leaderboard_evidence,
        run=run_payload,
        benchmark=benchmark_payload,
        dataset=dataset_payload,
        metrics_summary=metrics_summary_payload,
    )

    eligibility_reasons: list[str] = []
    if failed_samples:
        eligibility_reasons.append(f"{failed_samples} sample(s) failed")
    if not evidence_gate["present"]:
        eligibility_reasons.append(_MISSING_EVIDENCE_REASON)
    if not provenance_candidate:
        eligibility_reasons.append("evaluation provenance is not leaderboard-comparable")
    score_valid = failed_samples == 0
    leaderboard_valid = score_valid and bool(evidence_gate["present"]) and provenance_candidate
    resolved_comparison_identity = build_comparison_identity(
        benchmark=benchmark_payload,
        dataset=dataset_payload,
        metrics={
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            **{
                key: metrics_summary_payload[key]
                for key in ("metric_revision", "metrics_revision", "metric_config_hash", "config_hash")
                if key in metrics_summary_payload
            },
        },
        provenance=provenance_payload,
        evaluation_kind=evaluation_kind,
        explicit=comparison_identity,
    )

    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": run_payload,
        "benchmark": benchmark_payload,
        "model": dict(model),
        "dataset": dataset_payload,
        "provenance": provenance_payload,
        "comparison_identity": resolved_comparison_identity,
        "eligibility": {
            "leaderboard_valid": leaderboard_valid,
            "leaderboard_eligible": leaderboard_valid,
            "score_valid": score_valid,
            "leaderboard_reason": (
                "official/full-suite evidence gate present"
                if leaderboard_valid
                else "; ".join(eligibility_reasons)
            ),
            "reasons": list(eligibility_reasons),
            "blocking_reasons": list(eligibility_reasons),
            "evidence_gate": evidence_gate,
        },
        "generation": generation_payload,
        "metrics": {
            "leaderboard": leaderboard,
            "groups": dict(metrics_summary_payload.get("groups") or {}),
            "per_metric": per_metric,
            "summary": dict(metrics_summary_payload),
        },
        "evaluation": {
            "available": True,
            "kind": evaluation_kind,
            "mode": resolved_comparison_identity["evaluation_mode"],
            "num_results": int(metrics_summary_payload.get("sample_count") or 0),
            "successful_samples": int(metrics_summary_payload.get("successful_samples") or 0),
            "errored_samples": failed_samples,
            "summary": dict(metrics_summary_payload),
            "leaderboard_metrics": leaderboard,
            "skip_count": int(skip_payload.get("count") or 0),
        },
        "comparisons": {},
        "judges": {},
        "skips": skip_payload,
        "samples_ref": "metrics/per_sample.jsonl",
        "artifacts": dict(artifacts),
    }


def write_scorecard(path: str | Path, **kwargs: Any) -> Path:
    """Build the scorecard payload and write it to *path* as JSON.

    Args:
        path: Destination file path for the scorecard JSON.
        **kwargs: Forwarded to :func:`build_scorecard`.

    Returns:
        The resolved path of the written scorecard file.
    """
    scorecard_path = Path(path)
    payload = build_scorecard(**kwargs)
    payload.setdefault("artifacts", {})
    payload["artifacts"]["scorecard"] = str(scorecard_path.resolve())
    write_json(scorecard_path, payload)
    return scorecard_path
