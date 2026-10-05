"""HarnessEval Render Quality cache, backend contract, and score aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import clip01, mean_valid, numeric, skill_result
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from ..normalization import HPSV3_NORMALIZATION_VERSION, hpsv3_to_unit
from .common import (
    MetricCacheContract,
    SkillBackendIdentity,
    normalized_metrics,
    skill_cache_path,
)


SKILL_ID = "render_quality_inspector"
DEFINITION_VERSION = "harnesseval.render_quality"
CACHE_SCHEMA = "harnesseval.render_quality.cache"
BACKEND_OUTPUT_SCHEMA = "harnesseval.render_quality.backend_output"
COMPONENTS = ("aesthetic", "imaging", "hpsv3", "flickering")
BACKEND_METRICS = {
    "aesthetic": "aesthetic_quality",
    "imaging": "imaging_quality",
    "hpsv3": "hpsv3_quality",
    "flickering": "temporal_flickering",
}
SAMPLING = {
    "aesthetic": {"fps": 2.0},
    "imaging": {"fps": 2.0},
    "hpsv3": {"max_frames": 20, "method": "linspace_floor"},
    "flickering": {"fps": 10.0},
}
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.render_quality.cache_contract",
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
    "RenderQualityBackend",
    "cache_path_for_video",
    "import_metrics",
    "evaluate",
]


class RenderQualityBackend(SkillBackendIdentity, Protocol):
    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _normalize_backend_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw backend outputs according to the HarnessEval Render definition."""

    if evidence.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("render quality backend output schema mismatch")
    raw_metrics = evidence.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise RuntimeError("render quality backend returned no metrics object")
    metrics: dict[str, float] = {}
    details: dict[str, Any] = {}
    for component in COMPONENTS:
        metric_name = BACKEND_METRICS[component]
        result = raw_metrics.get(metric_name)
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{metric_name} returned no result object")
        raw_value = result.get("raw_score")
        raw = None if isinstance(raw_value, bool) else numeric(raw_value)
        if raw is None:
            raise RuntimeError(f"{metric_name} returned no finite numeric score")
        value = hpsv3_to_unit(raw) if component == "hpsv3" else clip01(raw)
        metrics[component] = value
        details[metric_name] = {
            "component": component,
            "raw_score": raw,
            "score": value,
            "sampling": dict(SAMPLING[component]),
            "sampled_frames": result.get("sampled_frames"),
        }
    return {"metrics": metrics, "details": details}


def score_components(metrics: Mapping[str, Any]) -> float | None:
    """Return the equal four-component mean, or None when evidence is incomplete."""

    values, missing, invalid = normalized_metrics(metrics, COMPONENTS)
    if missing or invalid:
        return None
    return mean_valid(values[key] for key in COMPONENTS)


def result_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build the standard score result from normalized component evidence."""

    values, missing, invalid = normalized_metrics(metrics, COMPONENTS)
    render_score = None
    if not missing and not invalid:
        render_score = mean_valid(values[key] for key in COMPONENTS)
    return skill_result(
        SKILL_ID,
        "ok" if render_score is not None else "invalid",
        render_score,
        metrics={key: metrics.get(key) for key in COMPONENTS},
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "aggregation": "equal_mean_required_components",
            "missing_components": missing,
            "invalid_components": invalid,
        },
    )


def cache_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return skill_cache_path(video_path, "render_quality", cache_root)


def _input_digest(
    backend_identity: Mapping[str, str],
    video: Mapping[str, Any],
) -> str:
    return value_digest(
        {
            "cache_schema": CACHE_SCHEMA,
            "skill_id": SKILL_ID,
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "normalization_version": HPSV3_NORMALIZATION_VERSION,
            "video": dict(video),
            "backend": dict(backend_identity),
            "sampling": SAMPLING,
        }
    )


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ValueError(f"render quality backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "local":
        raise ValueError("render quality backend execution_mode must be local")
    return identity


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
    result = result_from_metrics(payload.get("metrics") or {})
    return payload if result["status"] == "ok" else None


def _response(path: Path, payload: Mapping[str, Any], cache_hit: bool) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "cache_path": str(path),
        "result": result_from_metrics(payload.get("metrics") or {}),
        "details": payload.get("details") or {},
        "provenance": payload.get("provenance") or {},
    }


def import_metrics(
    video_path: Path,
    metrics: Mapping[str, Any],
    backend: SkillBackendIdentity,
    cache_root: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Import compatible component evidence into the formal skill cache."""

    video_path = video_path.resolve()
    video = file_fingerprint(video_path)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video)
    cache_path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(cache_path, digest)
    if cached is not None:
        return _response(cache_path, cached, True)
    result = result_from_metrics(metrics)
    if result["status"] != "ok":
        raise ValueError(f"cannot import invalid render metrics: {result['diagnostics']}")
    payload = {
        "schema_version": CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "metrics": dict(result["metrics"]),
        "details": {},
        "provenance": {
            "input_digest": digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "backend": identity,
            "execution_mode": "import",
            "source": dict(source),
        },
    }
    atomic_write_json(cache_path, payload)
    return _response(cache_path, payload, False)


def evaluate(
    video_path: Path,
    backend: RenderQualityBackend,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one complete video, using the deterministic per-skill cache path."""

    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cache_path = cache_path_for_video(video_path, cache_root)
    backend_identity = _backend_identity(backend)

    video = file_fingerprint(video_path)
    input_digest = _input_digest(backend_identity, video)
    cached = _load_cache(cache_path, input_digest)
    if cached is not None:
        return _response(cache_path, cached, True)

    evidence = _normalize_backend_evidence(backend.evaluate(video_path, video))
    result = result_from_metrics(evidence["metrics"])
    if result["status"] != "ok":
        raise RuntimeError(f"render quality produced invalid evidence: {result['diagnostics']}")
    payload = {
        "schema_version": CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "metrics": evidence["metrics"],
        "details": evidence["details"],
        "provenance": {
            "input_digest": input_digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "backend": backend_identity,
            "execution_mode": str(backend.execution_mode),
        },
    }
    atomic_write_json(cache_path, payload)
    return _response(cache_path, payload, False)
