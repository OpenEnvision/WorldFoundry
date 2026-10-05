"""HarnessEval Drift cache, chunk contract, and final.md score."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import clip01, mean_valid, numeric, skill_result, soft_threshold
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from .common import (
    MetricCacheContract,
    SkillBackendIdentity,
    normalized_metrics,
    skill_cache_path,
)
from .skill_appearance_consistency import (
    DEFINITION_VERSION as APPEARANCE_DEFINITION,
    score_components as appearance_score,
)
from .skill_motion_quality import (
    DEFINITION_VERSION as MOTION_DEFINITION,
    score_components as motion_score,
)
from .skill_physical_plausibility import (
    DEFINITION_VERSION as PHYSICAL_DEFINITION,
    score_components as physical_score,
)
from .skill_render_quality import (
    DEFINITION_VERSION as RENDER_DEFINITION,
    score_components as render_score,
)


SKILL_ID = "drift_degradation_analyzer"
DEFINITION_VERSION = "harnesseval.drift_degradation"
CACHE_SCHEMA = "harnesseval.drift_degradation.cache"
BACKEND_OUTPUT_SCHEMA = "harnesseval.drift_degradation.backend_output"
REQUIRED_COMPONENTS = (
    "aesthetic",
    "imaging",
    "hpsv3",
    "flickering",
    "dynamic",
    "smoothness",
    "background",
    "visual_plausibility",
)
OBSERVATION_DEFINITIONS = {
    "render_quality": RENDER_DEFINITION,
    "motion_quality": MOTION_DEFINITION,
    "appearance_quality": APPEARANCE_DEFINITION,
    "physical_quality": PHYSICAL_DEFINITION,
}
SAMPLING = {
    "observation_definitions": OBSERVATION_DEFINITIONS,
    "clip": {
        "model": "ViT-B/32",
        "fps": 2.0,
        "short_video_fallback": "all_frames",
        "aggregation": "normalize(mean(normalized_frame_embeddings))",
    },
}
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.drift_degradation.cache_contract",
    source_skill_ids=(
        "render_quality_inspector",
        "motion_quality_inspector",
        "appearance_consistency_inspector",
        "physical_plausibility_inspector",
    ),
    required_metrics=REQUIRED_COMPONENTS,
    accepts_source_skill_score=False,
    compatibility_fields=(
        "case_action_chunks",
        "chunk_boundaries",
        "video_fingerprints",
        "observation_definitions",
        "clip_model_fingerprint",
        "clip_sampling",
    ),
)

__all__ = [
    "SKILL_ID",
    "DEFINITION_VERSION",
    "CACHE_SCHEMA",
    "CACHE_CONTRACT",
    "BACKEND_OUTPUT_SCHEMA",
    "REQUIRED_COMPONENTS",
    "SAMPLING",
    "DriftDegradationBackend",
    "cache_path_for_video",
    "resolve_chunks",
    "result_from_chunks",
    "import_evidence",
    "evaluate",
]


class DriftDegradationBackend(SkillBackendIdentity, Protocol):
    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
        chunks: list[dict[str, Any]],
    ) -> Mapping[str, Any]: ...


def _video_stats(path: Path) -> tuple[int, float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video cannot be decoded: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"video has invalid metadata: {path}")
    return frame_count, fps


def _action_chunks(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    chunks = ((case.get("interaction") or {}).get("action") or {}).get("chunks")
    if not isinstance(chunks, list):
        return []
    return [dict(item) for item in chunks if isinstance(item, Mapping)]


def _chunk_ids(case: Mapping[str, Any]) -> list[str]:
    ids = [
        str(chunk.get("chunk_id") or f"c{index + 1:02d}")
        for index, chunk in enumerate(_action_chunks(case))
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("case action chunk ids must be unique")
    return ids


def _load_generation_metadata(video_path: Path) -> dict[str, Any]:
    path = video_path.parent / "metadata.json"
    if not path.is_file():
        return {}
    value = read_json(path)
    return dict(value) if isinstance(value, Mapping) else {}


def _explicit_segments(
    metadata: Mapping[str, Any], count: int, video_path: Path
) -> list[Path]:
    segments = metadata.get("segments")
    if segments is None or segments == []:
        return []
    if not isinstance(segments, list) or len(segments) != count:
        raise ValueError("generation metadata segments do not match case action chunks")
    output = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("generation metadata segment must be an object")
        raw = (
            segment.get("segment_video")
            or segment.get("output_video")
            or segment.get("output_video_path")
            or segment.get("video_path")
        )
        if not raw:
            raise ValueError("generation metadata segment has no video path")
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = video_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(path)
    return output


def resolve_chunks(
    video_path: Path,
    case: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve case-defined chunks to explicit segments or equal video intervals."""

    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    chunks = _action_chunks(case)
    if len(chunks) < 2:
        raise ValueError("HarnessEval Drift requires at least two case-defined action chunks")
    chunk_ids = _chunk_ids(case)
    metadata = (
        dict(metadata)
        if metadata is not None
        else _load_generation_metadata(video_path)
    )
    metadata_case = metadata.get("case_id")
    case_id = case.get("case_id")
    if metadata_case and case_id and str(metadata_case) != str(case_id):
        raise ValueError("generation metadata case_id does not match the case")

    explicit = _explicit_segments(metadata, len(chunks), video_path)
    frame_count = None if explicit else _video_stats(video_path)[0]
    output = []
    for position, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks)):
        if explicit:
            chunk_video = explicit[position]
            chunk_frames, _ = _video_stats(chunk_video)
            start, end, source = 0, chunk_frames, "explicit_segment_video"
        else:
            assert frame_count is not None
            chunk_video = video_path
            start = math.floor(position * frame_count / len(chunks))
            end = math.floor((position + 1) * frame_count / len(chunks))
            end = max(start + 1, min(frame_count, end))
            source = "equal_duration_fallback"
        output.append(
            {
                "chunk_id": chunk_id,
                "position": position,
                "action_chunk": chunk,
                "video_path": str(chunk_video),
                "video_fingerprint": file_fingerprint(chunk_video),
                "start_frame": start,
                "end_frame": end,
                "boundary_source": source,
            }
        )
    return output


def _normalized_embedding(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    numbers = []
    for item in value:
        number = None if isinstance(item, bool) else numeric(item)
        if number is None:
            return None
        numbers.append(number)
    norm = math.sqrt(sum(item * item for item in numbers))
    if norm <= 0:
        return None
    return [item / norm for item in numbers]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("CLIP embeddings must have the same non-zero dimension")
    return sum(a * b for a, b in zip(left, right))


def _linear_slope(values: list[float]) -> float:
    xs = [index / (len(values) - 1) for index in range(len(values))]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator


def _invalid(message: str, **diagnostics: Any) -> dict[str, Any]:
    return skill_result(
        SKILL_ID,
        "invalid",
        None,
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "error": message,
            **diagnostics,
        },
    )


def result_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the final.md Drift formula to ordered, normalized chunk evidence."""

    if len(chunks) < 2:
        return _invalid("HarnessEval Drift requires at least two case-defined chunks")
    chunk_quality: list[float] = []
    embeddings: list[list[float]] = []
    chunk_details = []
    for index, chunk in enumerate(chunks):
        metrics = chunk.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            return _invalid("chunk metrics must be an object", chunk_index=index)
        values, missing, invalid = normalized_metrics(metrics, REQUIRED_COMPONENTS)
        dimensions = {
            "render_quality": render_score(values),
            "motion_quality": motion_score(values),
            "appearance_quality": appearance_score(values),
            "physical_quality": physical_score(values),
        }
        if missing or invalid or any(value is None for value in dimensions.values()):
            return _invalid(
                "incomplete HarnessEval chunk quality",
                chunk_index=index,
                chunk_id=chunk.get("chunk_id"),
                missing_components=missing,
                invalid_components=invalid,
                dimensions=dimensions,
            )
        embedding = _normalized_embedding(chunk.get("clip_embedding"))
        if embedding is None:
            return _invalid(
                "missing or invalid per-chunk CLIP embedding",
                chunk_index=index,
                chunk_id=chunk.get("chunk_id"),
            )
        if embeddings and len(embedding) != len(embeddings[0]):
            return _invalid(
                "CLIP embedding dimensions differ between chunks",
                chunk_index=index,
                chunk_id=chunk.get("chunk_id"),
            )
        score = mean_valid(dimensions.values())
        assert score is not None
        chunk_quality.append(score)
        embeddings.append(embedding)
        chunk_details.append(
            {
                "chunk_id": chunk.get("chunk_id", str(index)),
                **dimensions,
                "chunk_quality": score,
            }
        )

    similarities = []
    for index in range(1, len(embeddings)):
        adjacent = max(0.0, _cosine(embeddings[index], embeddings[index - 1]))
        anchor = max(0.0, _cosine(embeddings[index], embeddings[0]))
        similarity = mean_valid((adjacent, anchor))
        assert similarity is not None
        similarities.append(similarity)
    mean_similarity = mean_valid(similarities)
    cross_chunk_consistency = mean_valid((mean_similarity, min(similarities)))
    assert cross_chunk_consistency is not None

    quality_slope = _linear_slope(chunk_quality)
    quality_trend = clip01(1.0 - soft_threshold(-quality_slope, 0.02, 0.15))
    quality_retention = clip01(chunk_quality[-1] / max(chunk_quality[0], 1e-6))
    worst_chunk_quality = min(chunk_quality)
    stability = clip01(1.0 - 2.0 * statistics.pstdev(chunk_quality))
    score = (
        0.35 * cross_chunk_consistency
        + 0.20 * quality_trend
        + 0.15 * quality_retention
        + 0.15 * worst_chunk_quality
        + 0.15 * stability
    )
    return skill_result(
        SKILL_ID,
        "ok",
        score,
        metrics={
            "cross_chunk_consistency": cross_chunk_consistency,
            "quality_trend": quality_trend,
            "quality_retention": quality_retention,
            "worst_chunk_quality": worst_chunk_quality,
            "chunk_quality_stability": stability,
            "quality_slope": quality_slope,
            "chunk_quality": chunk_quality,
            "chunk_appearance_similarity": similarities,
            "score_weights": {
                "cross_chunk_consistency": 0.35,
                "quality_trend": 0.20,
                "quality_retention": 0.15,
                "worst_chunk_quality": 0.15,
                "chunk_quality_stability": 0.15,
            },
        },
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "chunks": chunk_details,
            "segmentation": "case_defined_chunks",
        },
    )


def cache_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return skill_cache_path(video_path, "drift_degradation", cache_root)


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ValueError(f"Drift backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "local":
        raise ValueError("Drift backend execution_mode must be local")
    return identity


def _boundary_digest_value(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "action_chunk": chunk["action_chunk"],
            "video_fingerprint": chunk["video_fingerprint"],
            "start_frame": chunk["start_frame"],
            "end_frame": chunk["end_frame"],
            "boundary_source": chunk["boundary_source"],
        }
        for chunk in chunks
    ]


def _input_digest(
    backend: Mapping[str, str],
    video: Mapping[str, Any],
    chunks: list[dict[str, Any]],
) -> str:
    return value_digest(
        {
            "cache_schema": CACHE_SCHEMA,
            "skill_id": SKILL_ID,
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": dict(video),
            "chunks": _boundary_digest_value(chunks),
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
    result = result_from_chunks(payload.get("chunks") or [])
    return payload if result["status"] == "ok" else None


def _response(
    path: Path, payload: Mapping[str, Any], cache_hit: bool
) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "cache_path": str(path),
        "result": result_from_chunks(payload.get("chunks") or []),
        "details": payload.get("details") or {},
        "provenance": payload.get("provenance") or {},
    }


def _same_boundary(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    actual_path = Path(str(actual.get("video_path") or "")).expanduser().resolve()
    expected_path = Path(str(expected["video_path"])).resolve()
    actual_fingerprint = actual.get("video_fingerprint")
    expected_fingerprint = expected["video_fingerprint"]
    return (
        str(actual.get("chunk_id")) == str(expected["chunk_id"])
        and actual_path == expected_path
        and isinstance(actual_fingerprint, Mapping)
        and actual_fingerprint.get("size_bytes") == expected_fingerprint["size_bytes"]
        and actual_fingerprint.get("mtime_ns") == expected_fingerprint["mtime_ns"]
        and actual.get("start_frame") == expected["start_frame"]
        and actual.get("end_frame") == expected["end_frame"]
        and actual.get("boundary_source") == expected["boundary_source"]
    )


def _normalize_backend_evidence(
    evidence: Mapping[str, Any],
    boundaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evidence.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("Drift backend output schema mismatch")
    raw_chunks = evidence.get("chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) != len(boundaries):
        raise RuntimeError("Drift backend returned the wrong chunk count")
    output = []
    for index, (raw, expected) in enumerate(zip(raw_chunks, boundaries)):
        if not isinstance(raw, Mapping) or not _same_boundary(raw, expected):
            raise RuntimeError(
                f"Drift backend chunk boundary mismatch at index {index}"
            )
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(
                f"Drift backend chunk {expected['chunk_id']} has no metrics"
            )
        values, missing, invalid = normalized_metrics(metrics, REQUIRED_COMPONENTS)
        if missing or invalid:
            raise RuntimeError(
                f"Drift backend chunk {expected['chunk_id']} has invalid metrics: "
                f"missing={missing}, invalid={invalid}"
            )
        embedding = _normalized_embedding(raw.get("clip_embedding"))
        if embedding is None:
            raise RuntimeError(
                f"Drift backend chunk {expected['chunk_id']} has no valid CLIP embedding"
            )
        output.append(
            {
                "chunk_id": expected["chunk_id"],
                "metrics": {name: values[name] for name in REQUIRED_COMPONENTS},
                "clip_embedding": embedding,
                "boundary": {
                    key: expected[key]
                    for key in (
                        "video_path",
                        "video_fingerprint",
                        "start_frame",
                        "end_frame",
                        "boundary_source",
                    )
                },
            }
        )
    result = result_from_chunks(output)
    if result["status"] != "ok":
        raise RuntimeError(
            f"Drift backend produced invalid evidence: {result['diagnostics']}"
        )
    details = evidence.get("details")
    return output, dict(details) if isinstance(details, Mapping) else {}


def _write_cache(
    path: Path,
    chunks: list[dict[str, Any]],
    details: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "chunks": chunks,
        "details": dict(details),
        "provenance": dict(provenance),
    }
    atomic_write_json(path, payload)
    return _response(path, payload, False)


def import_evidence(
    video_path: Path,
    case: Mapping[str, Any],
    evidence: Mapping[str, Any],
    backend: SkillBackendIdentity,
    cache_root: Path,
    source: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Import component evidence only after validating its exact chunk boundaries."""

    video_path = video_path.expanduser().resolve()
    video = file_fingerprint(video_path)
    boundaries = resolve_chunks(video_path, case, metadata)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video, boundaries)
    path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)
    chunks, details = _normalize_backend_evidence(evidence, boundaries)
    return _write_cache(
        path,
        chunks,
        details,
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
    case: Mapping[str, Any],
    backend: DriftDegradationBackend,
    cache_root: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one video, checking the complete Drift cache before inference."""

    video_path = video_path.expanduser().resolve()
    video = file_fingerprint(video_path)
    boundaries = resolve_chunks(video_path, case, metadata)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video, boundaries)
    path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)

    evidence = backend.evaluate(video_path, video, boundaries)
    chunks, details = _normalize_backend_evidence(evidence, boundaries)
    return _write_cache(
        path,
        chunks,
        details,
        {
            "input_digest": digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "backend": identity,
            "execution_mode": str(backend.execution_mode),
        },
    )
