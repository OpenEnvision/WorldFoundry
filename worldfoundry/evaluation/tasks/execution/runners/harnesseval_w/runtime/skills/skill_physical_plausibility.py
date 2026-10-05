"""HarnessEval Physical Plausibility cache, PAVRM contract, and score."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import numeric, skill_result
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from .common import (
    MetricCacheContract,
    SkillBackendIdentity,
    normalized_metrics,
    skill_cache_path,
)


SKILL_ID = "physical_plausibility_inspector"
DEFINITION_VERSION = "harnesseval.physical_plausibility.pavrm"
CACHE_SCHEMA = "harnesseval.physical_plausibility.cache"
BACKEND_OUTPUT_SCHEMA = "harnesseval.physical_plausibility.backend_output"
COMPONENTS = ("visual_plausibility",)
SAMPLING = {"fps": 2.0, "max_pixels": 602112}
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.physical_plausibility.cache_contract",
    source_skill_ids=("render_physical_plausibility_inspector",),
    required_metrics=COMPONENTS,
    accepts_source_skill_score=False,
    compatibility_fields=(
        "video_fingerprint",
        "metric_id",
        "metric_version",
        "sampling_version",
        "normalization_version",
    ),
)

__all__ = [
    "SKILL_ID",
    "DEFINITION_VERSION",
    "CACHE_SCHEMA",
    "CACHE_CONTRACT",
    "BACKEND_OUTPUT_SCHEMA",
    "PhysicalPlausibilityBackend",
    "cache_path_for_video",
    "import_raw_score",
    "evaluate",
]


class PhysicalPlausibilityBackend(SkillBackendIdentity, Protocol):
    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _normalized_raw_score(raw_value: Any) -> tuple[float, float]:
    raw = None if isinstance(raw_value, bool) else numeric(raw_value)
    if raw is None or not 1.0 <= raw <= 5.0:
        raise ValueError("PAVRM raw_score must be a finite number in [1, 5]")
    return raw, raw / 5.0


def _normalize_backend_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if evidence.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("physical plausibility backend output schema mismatch")
    raw_metrics = evidence.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise RuntimeError("physical plausibility backend returned no metrics object")
    result = raw_metrics.get("pavrm")
    if not isinstance(result, Mapping):
        raise RuntimeError("PAVRM returned no result object")
    if result.get("error"):
        raise RuntimeError(f"PAVRM failed: {result['error']}")
    try:
        raw, score = _normalized_raw_score(result.get("raw_score"))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    reported = result.get("reported_score")
    if reported is not None:
        reported_score = None if isinstance(reported, bool) else numeric(reported)
        if reported_score is None or not math.isclose(
            reported_score, score, rel_tol=0.0, abs_tol=1e-4
        ):
            raise RuntimeError("PAVRM reported_score does not match raw_score / 5")
    return {
        "metrics": {"visual_plausibility": score},
        "details": {
            "pavrm": {
                "raw_score": raw,
                "score": score,
                "sampling": dict(SAMPLING),
            }
        },
    }


def score_components(metrics: Mapping[str, Any]) -> float | None:
    values, missing, invalid = normalized_metrics(metrics, COMPONENTS)
    return None if missing or invalid else values["visual_plausibility"]


def result_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values, missing, invalid = normalized_metrics(metrics, COMPONENTS)
    score = None if missing or invalid else values["visual_plausibility"]
    return skill_result(
        SKILL_ID,
        "ok" if score is not None else "invalid",
        score,
        metrics={"visual_plausibility": metrics.get("visual_plausibility")},
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "aggregation": "pavrm_raw_score_divided_by_5",
            "missing_components": missing,
            "invalid_components": invalid,
        },
    )


def cache_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return skill_cache_path(video_path, "physical_plausibility", cache_root)


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ValueError(f"physical plausibility backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "local":
        raise ValueError("physical plausibility backend execution_mode must be local")
    return identity


def _input_digest(backend: Mapping[str, str], video: Mapping[str, Any]) -> str:
    return value_digest(
        {
            "cache_schema": CACHE_SCHEMA,
            "skill_id": SKILL_ID,
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": dict(video),
            "backend": dict(backend),
            "sampling": SAMPLING,
        }
    )


def _load_cache(path: Path, input_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    provenance = payload.get("provenance") or {}
    if (
        payload.get("schema_version") != CACHE_SCHEMA
        or payload.get("skill_id") != SKILL_ID
        or payload.get("definition_version") != DEFINITION_VERSION
        or provenance.get("cache_contract_version") != CACHE_CONTRACT.version
        or provenance.get("input_digest") != input_digest
    ):
        return None
    return payload if result_from_metrics(payload.get("metrics") or {})["status"] == "ok" else None


def _response(path: Path, payload: Mapping[str, Any], cache_hit: bool) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "cache_path": str(path),
        "result": result_from_metrics(payload.get("metrics") or {}),
        "details": payload.get("details") or {},
        "provenance": payload.get("provenance") or {},
    }


def _write_cache(
    path: Path,
    metrics: Mapping[str, Any],
    details: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    result = result_from_metrics(metrics)
    if result["status"] != "ok":
        raise ValueError(f"cannot cache invalid physical metrics: {result['diagnostics']}")
    payload = {
        "schema_version": CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "metrics": dict(result["metrics"]),
        "details": dict(details),
        "provenance": dict(provenance),
    }
    atomic_write_json(path, payload)
    return _response(path, payload, False)


def import_raw_score(
    video_path: Path,
    raw_score: Any,
    backend: SkillBackendIdentity,
    cache_root: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Import one validated PAVRM output, never a wrapper score."""

    video_path = video_path.resolve()
    video = file_fingerprint(video_path)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video)
    path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)
    raw, score = _normalized_raw_score(raw_score)
    return _write_cache(
        path,
        {"visual_plausibility": score},
        {"pavrm": {"raw_score": raw, "score": score, "sampling": dict(SAMPLING)}},
        {
            "input_digest": digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "backend": identity,
            "execution_mode": "import",
            "source": dict(source),
        },
    )


def evaluate(
    video_path: Path,
    backend: PhysicalPlausibilityBackend,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one video, checking this skill's cache before PAVRM inference."""

    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    path = cache_path_for_video(video_path, cache_root)
    identity = _backend_identity(backend)
    video = file_fingerprint(video_path)
    digest = _input_digest(identity, video)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)

    evidence = _normalize_backend_evidence(backend.evaluate(video_path, video))
    return _write_cache(
        path,
        evidence["metrics"],
        evidence["details"],
        {
            "input_digest": digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "backend": identity,
            "execution_mode": str(backend.execution_mode),
        },
    )
