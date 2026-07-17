"""Protocol identity used to decide whether evaluation runs are comparable."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import stable_hash

COMPARISON_IDENTITY_SCHEMA_VERSION = "worldfoundry-comparison-identity-v1"

_STRICT_FIELDS = (
    "benchmark_id",
    "benchmark_revision",
    "protocol_id",
    "protocol_revision",
    "protocol_config_hash",
    "protocol_fidelity",
    "data_fidelity",
    "dataset_id",
    "dataset_revision",
    "dataset_hash",
    "split",
    "metric_revision",
    "metric_config_hash",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _metric_ids(metrics: Mapping[str, Any]) -> list[str]:
    per_metric = _mapping(metrics.get("per_metric"))
    leaderboard = _mapping(metrics.get("leaderboard"))
    return sorted(str(metric_id) for metric_id in (per_metric or leaderboard))


def _evaluation_mode(evaluation_kind: str, provenance: Mapping[str, Any]) -> str:
    normalized = str(evaluation_kind or "unknown").strip().lower().replace("-", "_")
    if normalized in {"reproduction", "new_model_evaluation", "adapted_protocol", "metric_only"}:
        return normalized

    # Provenance is stronger evidence than generic executor names such as
    # ``existing_results``: that executor is also used after model generation.
    fidelity = _mapping(provenance.get("fidelity"))
    evaluation_fidelity = str(fidelity.get("evaluation") or "unknown")
    if evaluation_fidelity == "metric_only":
        return "metric_only"
    if fidelity.get("data") in {"subset", "custom"}:
        return "adapted_protocol"
    if evaluation_fidelity == "modified":
        return "adapted_protocol"
    if evaluation_fidelity == "official" and fidelity.get("generation") == "pinned":
        return "reproduction"
    if evaluation_fidelity == "official":
        return "new_model_evaluation"

    aliases = {
        "reproduce": "reproduction",
        "official_reproduction": "reproduction",
        "model": "new_model_evaluation",
        "generate": "new_model_evaluation",
        "model_benchmark": "new_model_evaluation",
        "protocol_compliant": "new_model_evaluation",
        "modified": "adapted_protocol",
        "adapt": "adapted_protocol",
        "existing_results": "metric_only",
        "custom_results_metric_evaluation": "metric_only",
        "custom_dataset_metric_evaluation": "metric_only",
    }
    if normalized in aliases:
        return aliases[normalized]
    return normalized or "unknown"


def build_comparison_identity(
    *,
    benchmark: Mapping[str, Any],
    dataset: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    evaluation_kind: str = "unknown",
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable protocol/data identity without including model-specific settings."""

    benchmark_payload = dict(benchmark)
    dataset_payload = dict(dataset)
    metrics_payload = dict(metrics)
    metrics_summary = _mapping(metrics_payload.get("summary"))
    provenance_payload = dict(provenance or {})
    fidelity = _mapping(provenance_payload.get("fidelity"))
    explicit_payload = dict(explicit or {})

    derived = {
        "evaluation_mode": _evaluation_mode(evaluation_kind, provenance_payload),
        "benchmark_id": _text(_first(benchmark_payload, "benchmark_id", "id", "benchmark_name", "name")),
        "benchmark_revision": _text(
            _first(benchmark_payload, "benchmark_revision", "revision", "repo_revision", "upstream_revision")
        ),
        "protocol_id": _text(
            _first(benchmark_payload, "protocol_id", "evaluation_protocol", "protocol", "benchmark_id", "id")
            or _first(benchmark_payload, "benchmark_name", "name")
        ),
        "protocol_revision": _text(
            _first(benchmark_payload, "protocol_revision", "evaluator_revision", "scorer_revision")
        ),
        "protocol_config_hash": _text(
            _first(benchmark_payload, "protocol_config_hash", "evaluation_config_hash", "config_hash")
        ),
        "protocol_fidelity": _text(
            fidelity.get("evaluation") or _first(benchmark_payload, "protocol_fidelity", "evaluation_fidelity")
        ),
        "data_fidelity": _text(fidelity.get("data") or dataset_payload.get("data_fidelity")),
        "dataset_id": _text(_first(dataset_payload, "dataset_id", "id", "name")),
        "dataset_revision": _text(
            _first(dataset_payload, "dataset_revision", "revision", "version", "repo_revision")
        ),
        "dataset_hash": _text(
            _first(
                dataset_payload,
                "dataset_hash",
                "manifest_sha256",
                "requests_sha256",
                "content_sha256",
                "sha256",
            )
        ),
        "split": _text(dataset_payload.get("split")),
        "metric_ids": _metric_ids(metrics_payload),
        "metric_revision": _text(
            _first(metrics_payload, "metric_revision", "metrics_revision", "revision", "scorer_revision")
            or _first(metrics_summary, "metric_revision", "metrics_revision", "revision", "scorer_revision")
            or _first(benchmark_payload, "metric_revision", "metrics_revision")
        ),
        "metric_config_hash": _text(
            _first(metrics_payload, "metric_config_hash", "config_hash")
            or _first(metrics_summary, "metric_config_hash", "config_hash")
            or _first(benchmark_payload, "metric_config_hash")
        ),
    }
    for key, value in explicit_payload.items():
        if key in derived and value not in (None, ""):
            if key == "metric_ids":
                values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
                derived[key] = sorted(str(item) for item in values)
            else:
                derived[key] = _text(value)

    required = ("benchmark_id", "protocol_id", "dataset_id")
    recommended = (
        "benchmark_revision",
        "protocol_revision",
        "protocol_config_hash",
        "protocol_fidelity",
        "data_fidelity",
        "dataset_revision",
        "dataset_hash",
        "metric_revision",
        "metric_config_hash",
    )
    missing_required = [field for field in required if not derived.get(field)]
    missing_recommended = [field for field in recommended if not derived.get(field)]
    key_payload = {field: derived.get(field) for field in _STRICT_FIELDS}
    comparison_key = stable_hash(key_payload)
    return {
        "schema_version": COMPARISON_IDENTITY_SCHEMA_VERSION,
        **derived,
        "comparison_key": comparison_key,
        "status": "complete" if not missing_required and not missing_recommended else "incomplete",
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
    }


def comparison_identity_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Read a stored identity or derive a conservative identity for legacy artifacts."""

    stored = summary.get("comparison_identity")
    if isinstance(stored, Mapping) and stored.get("schema_version") == COMPARISON_IDENTITY_SCHEMA_VERSION:
        return dict(stored)
    evaluation = _mapping(summary.get("evaluation"))
    identity = build_comparison_identity(
        benchmark=_mapping(summary.get("benchmark")),
        dataset=_mapping(summary.get("dataset")),
        metrics=_mapping(summary.get("metrics")),
        provenance=_mapping(summary.get("provenance")),
        evaluation_kind=str(evaluation.get("mode") or evaluation.get("kind") or "unknown"),
    )
    identity["legacy_derived"] = True
    return identity


def compare_identities(identities: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Return hard incompatibilities and non-blocking identity warnings."""

    if not identities:
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    for left_index, left_value in enumerate(identities):
        left_identity = dict(left_value)
        for right_index, right_value in enumerate(identities[left_index + 1 :], start=left_index + 1):
            right_identity = dict(right_value)
            for field in _STRICT_FIELDS:
                left = left_identity.get(field)
                right = right_identity.get(field)
                if left in (None, "") or right in (None, ""):
                    if left != right:
                        warnings.append(
                            f"runs {left_index} and {right_index} cannot verify {field}: "
                            f"left={left!r}, right={right!r}"
                        )
                    continue
                if left != right:
                    errors.append(
                        f"runs {left_index} and {right_index} differ on {field}: "
                        f"left={left!r}, right={right!r}"
                    )
    for index, identity in enumerate(identities):
        missing = [*identity.get("missing_required", ()), *identity.get("missing_recommended", ())]
        if missing:
            warnings.append(f"run {index} has incomplete comparison identity: {', '.join(map(str, missing))}")
        if identity.get("legacy_derived"):
            warnings.append(f"run {index} uses a legacy-derived comparison identity")
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


__all__ = [
    "COMPARISON_IDENTITY_SCHEMA_VERSION",
    "build_comparison_identity",
    "compare_identities",
    "comparison_identity_from_summary",
]
