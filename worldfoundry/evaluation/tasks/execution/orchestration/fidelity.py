"""Protocol-fidelity policy for prepared evaluations.

This module deliberately contains no runner or catalog imports.  It classifies a
resolved evaluation along independent axes and derives the strongest claim that
the run may make.  Execution success and benchmark-specific eligibility remain
the responsibility of the existing runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

EVALUATION_PROVENANCE_SCHEMA_VERSION = "worldfoundry-evaluation-provenance-v1"

_PRODUCERS = frozenset({"catalog_model", "custom_model", "imported_artifacts", "imported_results"})
_GENERATION_FIDELITY = frozenset({"pinned", "custom", "not_applicable", "unknown"})
_DATA_FIDELITY = frozenset({"official", "subset", "custom", "unknown"})
_EVALUATION_FIDELITY = frozenset({"official", "modified", "metric_only", "unknown"})
_RUNTIME_FIDELITY = frozenset({"pinned", "compatible", "unknown"})
_REFERENCE_FIDELITY = frozenset({"pinned", "none", "unknown"})


def _validate(name: str, value: str, allowed: frozenset[str]) -> str:
    normalized = str(value).strip()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return normalized


def _dedupe_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(reason) for reason in reasons if str(reason).strip()))


@dataclass(frozen=True)
class EvaluationFidelity:
    """Orthogonal provenance dimensions for one resolved evaluation.

    ``claim_level`` is derived rather than supplied by callers, preventing a
    custom-data or metric-only run from self-declaring leaderboard eligibility.
    """

    producer: str
    generation: str
    data: str
    evaluation: str
    runtime: str = "unknown"
    reference: str = "none"
    reasons: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = EVALUATION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation provenance schema: {self.schema_version}")
        object.__setattr__(self, "producer", _validate("producer", self.producer, _PRODUCERS))
        object.__setattr__(
            self,
            "generation",
            _validate("generation fidelity", self.generation, _GENERATION_FIDELITY),
        )
        object.__setattr__(self, "data", _validate("data fidelity", self.data, _DATA_FIDELITY))
        object.__setattr__(
            self,
            "evaluation",
            _validate("evaluation fidelity", self.evaluation, _EVALUATION_FIDELITY),
        )
        object.__setattr__(self, "runtime", _validate("runtime fidelity", self.runtime, _RUNTIME_FIDELITY))
        object.__setattr__(
            self,
            "reference",
            _validate("reference fidelity", self.reference, _REFERENCE_FIDELITY),
        )
        object.__setattr__(self, "reasons", _dedupe_reasons(self.reasons))

    @property
    def claim_level(self) -> str:
        """Return the strongest claim supported by the declared provenance."""

        if (
            self.generation == "pinned"
            and self.data == "official"
            and self.evaluation == "official"
            and self.runtime == "pinned"
            and self.reference == "pinned"
        ):
            return "exact_reproduction"
        if self.data == "official" and self.evaluation == "official":
            return "benchmark_comparable"
        return "diagnostic"

    @property
    def leaderboard_candidate(self) -> bool:
        """Return whether downstream official evidence may establish eligibility."""

        return self.claim_level in {"exact_reproduction", "benchmark_comparable"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "fidelity": {
                "generation": self.generation,
                "data": self.data,
                "evaluation": self.evaluation,
                "runtime": self.runtime,
                "reference": self.reference,
            },
            "claim": {
                "level": self.claim_level,
                "leaderboard_candidate": self.leaderboard_candidate,
            },
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvaluationFidelity":
        dimensions = payload.get("fidelity")
        dimensions = dimensions if isinstance(dimensions, Mapping) else payload
        return cls(
            producer=str(payload.get("producer") or "imported_results"),
            generation=str(dimensions.get("generation") or "unknown"),
            data=str(dimensions.get("data") or "unknown"),
            evaluation=str(dimensions.get("evaluation") or "unknown"),
            runtime=str(dimensions.get("runtime") or "unknown"),
            reference=str(dimensions.get("reference") or "unknown"),
            reasons=tuple(payload.get("reasons") or ()),
            schema_version=str(payload.get("schema_version") or EVALUATION_PROVENANCE_SCHEMA_VERSION),
        )


def model_benchmark_fidelity(
    *,
    benchmark_mode: str,
    custom_data: bool,
    sample_limited: bool,
    benchmark_parameters: Mapping[str, Any] | None = None,
    producer: str = "catalog_model",
) -> EvaluationFidelity:
    """Classify a normal model × benchmark run conservatively."""

    reasons: list[str] = []
    if custom_data:
        data = "custom"
        reasons.append("custom benchmark requests or dataset")
    elif sample_limited or benchmark_mode == "official-validation":
        data = "subset"
        reasons.append("benchmark data coverage is a subset")
    else:
        data = "official"

    semantic_overrides = sorted(set(benchmark_parameters or {}).difference({"revision"}))
    if benchmark_mode != "official-run":
        evaluation = "modified"
        reasons.append(f"benchmark mode is {benchmark_mode}")
    elif semantic_overrides:
        evaluation = "modified"
        reasons.append("benchmark evaluation parameters were overridden: " + ", ".join(semantic_overrides))
    else:
        evaluation = "official"

    return EvaluationFidelity(
        producer=producer,
        generation="custom",
        data=data,
        evaluation=evaluation,
        runtime="compatible",
        reference="none",
        reasons=tuple(reasons),
    )


def reproduction_fidelity(
    *,
    benchmark_mode: str,
    benchmark_revision: bool,
    model_revision: bool,
    data: Mapping[str, Any],
    evaluation_parameters: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> EvaluationFidelity:
    """Classify an immutable reproduction recipe without trusting its label alone."""

    reasons: list[str] = []
    explicit_official_data = data.get("official") is True or data.get("source") == "protocol"
    custom_data_source = any(data.get(key) not in (None, "") for key in ("requests_path", "root", "task_name"))
    limited = data.get("num_samples") is not None or reference.get("full_suite") is False
    if explicit_official_data and not limited:
        data_fidelity = "official"
    elif limited or benchmark_mode == "official-validation":
        data_fidelity = "subset"
        reasons.append("reproduction recipe covers only a benchmark subset")
    elif custom_data_source:
        data_fidelity = "custom"
        reasons.append("reproduction recipe uses data not declared as official")
    else:
        data_fidelity = "official"

    protocol_declared_official = reference.get("protocol_fidelity") == "official"
    if benchmark_mode == "official-run" and benchmark_revision and (
        protocol_declared_official or not evaluation_parameters
    ):
        evaluation_fidelity = "official"
    else:
        evaluation_fidelity = "modified"
        if benchmark_mode != "official-run":
            reasons.append(f"reproduction benchmark mode is {benchmark_mode}")
        elif not benchmark_revision:
            reasons.append("benchmark revision is not pinned")
        else:
            reasons.append("recipe evaluation parameters are not declared official")

    generation_fidelity = "pinned" if model_revision else "unknown"
    if not model_revision:
        reasons.append("model revision is not pinned")
    runtime_fidelity = "pinned" if reference.get("runtime_fidelity") == "pinned" else "unknown"
    if runtime_fidelity != "pinned":
        reasons.append("runtime is not pinned")
    reference_fidelity = "pinned" if reference.get("evidence") or reference.get("score") is not None else "none"

    return EvaluationFidelity(
        producer="catalog_model",
        generation=generation_fidelity,
        data=data_fidelity,
        evaluation=evaluation_fidelity,
        runtime=runtime_fidelity,
        reference=reference_fidelity,
        reasons=tuple(reasons),
    )


def artifact_scoring_fidelity(*, benchmark_mode: str) -> EvaluationFidelity:
    evaluation = "official" if benchmark_mode == "official-run" else "modified"
    reasons = ["artifact data coverage is not verified as the official benchmark dataset"]
    if benchmark_mode != "official-run":
        reasons.append(f"benchmark mode is {benchmark_mode}")
    return EvaluationFidelity(
        producer="imported_artifacts",
        generation="not_applicable",
        data="unknown",
        evaluation=evaluation,
        runtime="compatible",
        reference="none",
        reasons=tuple(reasons),
    )


def metric_only_fidelity(*, producer: str, custom_data: bool = True) -> EvaluationFidelity:
    return EvaluationFidelity(
        producer=producer,
        generation="custom" if producer == "catalog_model" else "not_applicable",
        data="custom" if custom_data else "unknown",
        evaluation="metric_only",
        runtime="compatible" if producer == "catalog_model" else "unknown",
        reference="none",
        reasons=("metric-only results are not a complete benchmark protocol",),
    )


__all__ = [
    "EVALUATION_PROVENANCE_SCHEMA_VERSION",
    "EvaluationFidelity",
    "artifact_scoring_fidelity",
    "metric_only_fidelity",
    "model_benchmark_fidelity",
    "reproduction_fidelity",
]
