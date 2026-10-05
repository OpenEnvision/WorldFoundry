"""HarnessEval Appearance Consistency cache, backend contract, and score."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import clip01, numeric, skill_result
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from .common import (
    MetricCacheContract,
    SkillBackendIdentity,
    normalized_metrics,
    skill_cache_path,
)


SKILL_ID = "appearance_consistency_inspector"
DEFINITION_VERSION = "harnesseval.appearance_consistency"
CACHE_SCHEMA = "harnesseval.appearance_consistency.cache"
BACKEND_OUTPUT_SCHEMA = "harnesseval.appearance_consistency.backend_output"
COMPONENTS = ("background",)
BACKEND_METRICS = {"background": "background_consistency"}
SAMPLING = {"background": {"fps": 2.0, "short_video_fallback": "all_frames"}}
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.appearance_consistency.cache_contract",
    source_skill_ids=("render_quality_inspector",),
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
    "AppearanceConsistencyBackend",
    "cache_path_for_video",
    "import_metrics",
    "evaluate",
]


class AppearanceConsistencyBackend(SkillBackendIdentity, Protocol):
    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _normalize_backend_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if evidence.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("appearance consistency backend output schema mismatch")
    raw_metrics = evidence.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise RuntimeError("appearance consistency backend returned no metrics object")

    metric_name = BACKEND_METRICS["background"]
    result = raw_metrics.get(metric_name)
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{metric_name} returned no result object")
    raw_value = result.get("raw_score")
    raw = None if isinstance(raw_value, bool) else numeric(raw_value)
    if raw is None:
        raise RuntimeError(f"{metric_name} returned no finite numeric score")
    score = clip01(raw)
    return {
        "metrics": {"background": score},
        "details": {
            metric_name: {
                "component": "background",
                "raw_score": raw,
                "score": score,
                "sampling": dict(SAMPLING["background"]),
                "sampling_policy": result.get("sampling_policy"),
                "sampled_frames": result.get("sampled_frames"),
            }
        },
    }


def score_components(metrics: Mapping[str, Any]) -> float | None:
    values, missing, invalid = normalized_metrics(metrics, COMPONENTS)
    return None if missing or invalid else values["background"]


def result_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values, missing, invalid = normalized_metrics(metrics, COMPONENTS)
    score = None if missing or invalid else values["background"]
    return skill_result(
        SKILL_ID,
        "ok" if score is not None else "invalid",
        score,
        metrics={"background": metrics.get("background")},
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "aggregation": "single_required_component",
            "missing_components": missing,
            "invalid_components": invalid,
        },
    )


def cache_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return skill_cache_path(video_path, "appearance_consistency", cache_root)


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ValueError(
            f"appearance consistency backend identity missing: {', '.join(missing)}"
        )
    if str(backend.execution_mode).strip() != "local":
        raise ValueError(
            "appearance consistency backend execution_mode must be local"
        )
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
        raise ValueError(f"cannot cache invalid appearance metrics: {result['diagnostics']}")
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


def import_metrics(
    video_path: Path,
    metrics: Mapping[str, Any],
    backend: SkillBackendIdentity,
    cache_root: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Import compatible background evidence, never an aggregate score."""

    video_path = video_path.resolve()
    video = file_fingerprint(video_path)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video)
    path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)
    return _write_cache(
        path,
        metrics,
        {},
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
    backend: AppearanceConsistencyBackend,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one video, checking this skill's cache before backend inference."""

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
