"""HarnessEval Return Consistency cache, evidence contract, and final.md score."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import clip01, numeric, skill_result, soft_threshold
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from .common import MetricCacheContract, SkillBackendIdentity, skill_cache_path


SKILL_ID = "return_consistency_verifier"
DEFINITION_VERSION = "harnesseval.return_consistency"
CACHE_SCHEMA = "harnesseval.return_consistency.cache"
BACKEND_OUTPUT_SCHEMA = "harnesseval.return_consistency.backend_output"
FRAME_COMPONENTS = (
    "ncc_score",
    "histogram_score",
    "edge_iou",
    "rgb_pixel_similarity",
    "clip_similarity",
)
SAMPLING = {
    "maximum_frames": 12,
    "uniform": "linspace_including_endpoints",
    "chunk_assignment": "equal_duration_sample_midpoint",
    "comparison_size": [192, 108],
    "clip_model": "ViT-B/32",
}
PAIR_WEIGHTS = {"best_similarity": 0.5, "mean_similarity": 0.5}
MULTI_PAIR_WEIGHTS = {"mean_pair_score": 0.7, "worst_pair_score": 0.3}
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.return_consistency.cache_contract",
    source_skill_ids=("return_consistency_verifier",),
    required_metrics=FRAME_COMPONENTS,
    accepts_source_skill_score=False,
    compatibility_fields=(
        "video_fingerprint",
        "case_action_chunks",
        "return_pairs",
        "sampled_frame_indices",
        "frame_components",
        "clip_model_fingerprint",
        "non_static_gate_inputs",
    ),
)

__all__ = [
    "SKILL_ID",
    "DEFINITION_VERSION",
    "CACHE_SCHEMA",
    "CACHE_CONTRACT",
    "BACKEND_OUTPUT_SCHEMA",
    "FRAME_COMPONENTS",
    "SAMPLING",
    "PAIR_WEIGHTS",
    "MULTI_PAIR_WEIGHTS",
    "ReturnConsistencyBackend",
    "cache_path_for_video",
    "frame_similarity",
    "resolve_sample_plan",
    "result_from_evidence",
    "validate_evidence",
    "import_evidence",
    "evaluate",
]


class ReturnConsistencyBackend(SkillBackendIdentity, Protocol):
    def evaluate(
        self,
        video_path: Path,
        video_fingerprint: Mapping[str, Any],
        sample_plan: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


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


def _return_pairs(case: Mapping[str, Any]) -> list[dict[str, str]]:
    non_model = case.get("non_model_facing") or {}
    raw = (
        non_model.get("return_windows")
        or non_model.get("closed_loop_pairs")
        or case.get("return_windows")
        or case.get("closed_loop_pairs")
    )
    if isinstance(raw, list):
        candidates = [item for item in raw if isinstance(item, Mapping)]
    else:
        single = non_model.get("return_window") or case.get("return_window")
        candidates = [single] if isinstance(single, Mapping) else []
    output = []
    for pair in candidates:
        reference = (
            pair.get("reference_chunk")
            or pair.get("start_chunk")
            or pair.get("source_chunk")
        )
        returned = (
            pair.get("return_chunk")
            or pair.get("end_chunk")
            or pair.get("target_chunk")
        )
        if reference and returned:
            output.append(
                {"reference_chunk": str(reference), "return_chunk": str(returned)}
            )
    return output


def _video_frame_count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video cannot be decoded: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count < 2:
        raise RuntimeError(f"Return requires at least two video frames: {path}")
    return count


def _uniform_indices(frame_count: int, maximum: int = 12) -> list[int]:
    count = min(maximum, frame_count)
    if count < 2:
        raise ValueError("Return requires at least two sampled frames")
    return sorted(
        {
            math.floor(position * (frame_count - 1) / (count - 1))
            for position in range(count)
        }
    )


def _chunk_position(chunk_id: str, chunk_ids: list[str]) -> int | None:
    if chunk_id in chunk_ids:
        return chunk_ids.index(chunk_id)
    lowered = str(chunk_id).strip().lower()
    if lowered.startswith("c") and lowered[1:].isdigit():
        position = int(lowered[1:]) - 1
        return position if 0 <= position < len(chunk_ids) else None
    return None


def _samples_for_chunk(position: int, chunk_count: int, sample_count: int) -> list[int]:
    start = position / chunk_count
    end = (position + 1) / chunk_count
    selected = [
        index
        for index in range(sample_count)
        if start <= (index + 0.5) / sample_count < end
    ]
    if selected:
        return selected
    center = (start + end) / 2.0
    return [
        min(
            range(sample_count),
            key=lambda index: abs((index + 0.5) / sample_count - center),
        )
    ]


def resolve_sample_plan(video_path: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve deterministic samples and all case-defined return loops."""

    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    chunk_ids = _chunk_ids(case)
    if not chunk_ids:
        raise ValueError("Return requires case-defined action chunks")
    sampled_frames = _uniform_indices(_video_frame_count(video_path))
    resolved = []
    for pair in _return_pairs(case):
        reference_position = _chunk_position(pair["reference_chunk"], chunk_ids)
        return_position = _chunk_position(pair["return_chunk"], chunk_ids)
        if reference_position is None or return_position is None:
            continue
        resolved.append(
            {
                **pair,
                "reference_sample_indices": _samples_for_chunk(
                    reference_position, len(chunk_ids), len(sampled_frames)
                ),
                "return_sample_indices": _samples_for_chunk(
                    return_position, len(chunk_ids), len(sampled_frames)
                ),
            }
        )
    fallback = not resolved
    if fallback:
        resolved = [
            {
                "reference_chunk": chunk_ids[0],
                "return_chunk": chunk_ids[-1],
                "reference_sample_indices": [0],
                "return_sample_indices": [len(sampled_frames) - 1],
            }
        ]
    return {
        "sampled_frame_indices": sampled_frames,
        "chunk_ids": chunk_ids,
        "pairs": resolved,
        "fallback_first_last": fallback,
    }


def _finite_unit(value: Any) -> float | None:
    number = None if isinstance(value, bool) else numeric(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def frame_similarity(components: Mapping[str, Any]) -> float:
    values = {name: _finite_unit(components.get(name)) for name in FRAME_COMPONENTS}
    if values["rgb_pixel_similarity"] is None:
        rgb_diff = _finite_unit(components.get("rgb_mean_abs_diff"))
        if rgb_diff is not None:
            values["rgb_pixel_similarity"] = clip01(
                1.0 - soft_threshold(rgb_diff, 0.05, 0.35)
            )
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"missing or invalid Return components: {', '.join(missing)}")
    gray_structure = statistics.mean(
        values[name] for name in ("ncc_score", "histogram_score", "edge_iou")
    )
    return clip01(
        0.25 * gray_structure
        + 0.25 * float(values["rgb_pixel_similarity"])
        + 0.50 * float(values["clip_similarity"])
    )


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


def result_from_evidence(
    evidence: Mapping[str, Any], *, fallback_first_last: bool = False
) -> dict[str, Any]:
    """Aggregate normalized frame evidence with the confirmed Return weights."""

    pairs = evidence.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        return _invalid("Return requires at least one resolved pair")
    pair_scores = []
    best_scores = []
    mean_scores = []
    pair_details = []
    try:
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise ValueError("Return pair evidence must be an object")
            similarities = [
                frame_similarity(item)
                for item in (pair.get("similarities") or [])
                if isinstance(item, Mapping)
            ]
            if not similarities:
                raise ValueError(
                    f"Return pair {pair.get('reference_chunk')} has no comparisons"
                )
            best = max(similarities)
            average = statistics.mean(similarities)
            pair_score = 0.5 * best + 0.5 * average
            best_scores.append(best)
            mean_scores.append(average)
            pair_scores.append(pair_score)
            pair_details.append(
                {
                    "reference_chunk": pair.get("reference_chunk"),
                    "return_chunk": pair.get("return_chunk"),
                    "best_similarity": best,
                    "mean_similarity": average,
                    "pair_score": pair_score,
                    "comparison_count": len(similarities),
                }
            )
    except (TypeError, ValueError) as exc:
        return _invalid(str(exc))

    mean_pair_score = statistics.mean(pair_scores)
    worst_pair_score = min(pair_scores)
    closed_loop = 0.7 * mean_pair_score + 0.3 * worst_pair_score
    mean_diff = _finite_unit(evidence.get("mean_diff"))
    peak_diff = _finite_unit(evidence.get("peak_diff"))
    active_ratio = _finite_unit(evidence.get("active_ratio"))
    if None in (mean_diff, peak_diff, active_ratio):
        return _invalid("missing or invalid non-static gate metrics")
    gate = statistics.mean(
        (
            soft_threshold(mean_diff, 0.004, 0.06),
            soft_threshold(peak_diff, 0.008, 0.12),
            active_ratio,
        )
    )
    return skill_result(
        SKILL_ID,
        "fallback_first_last" if fallback_first_last else "ok",
        closed_loop * gate,
        metrics={
            "closed_loop_pair_consistency": closed_loop,
            "best_pair_similarity": statistics.mean(best_scores),
            "mean_pair_similarity": statistics.mean(mean_scores),
            "mean_closed_loop_score": mean_pair_score,
            "worst_pair_similarity": worst_pair_score,
            "non_static_gate": gate,
            "motion_presence": gate,
            "mean_frame_diff": mean_diff,
            "peak_frame_diff": peak_diff,
            "active_frame_pair_ratio": active_ratio,
            "pair_weights": PAIR_WEIGHTS,
            "multi_pair_weights": MULTI_PAIR_WEIGHTS,
        },
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "pair_details": pair_details,
            "formal_gate_accepted": not fallback_first_last,
        },
    )


def cache_path_for_video(video_path: Path, cache_root: Path | None = None) -> Path:
    return skill_cache_path(video_path, "return_consistency", cache_root)


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise ValueError(f"Return backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "local":
        raise ValueError("Return backend execution_mode must be local")
    return identity


def _input_digest(
    backend: Mapping[str, str],
    video: Mapping[str, Any],
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    return value_digest(
        {
            "cache_schema": CACHE_SCHEMA,
            "skill_id": SKILL_ID,
            "definition_version": DEFINITION_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": dict(video),
            "case_id": case.get("case_id"),
            "action_chunks": _action_chunks(case),
            "sample_plan": dict(plan),
            "backend": dict(backend),
            "sampling": SAMPLING,
            "pair_weights": PAIR_WEIGHTS,
            "multi_pair_weights": MULTI_PAIR_WEIGHTS,
        }
    )


def _same_plan(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("sampled_frame_indices") == expected["sampled_frame_indices"]
        and actual.get("pairs") == expected["pairs"]
        and bool(actual.get("fallback_first_last"))
        == bool(expected["fallback_first_last"])
    )


def _normalize_backend_evidence(
    raw: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    if raw.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("Return backend output schema mismatch")
    if not _same_plan(raw, plan):
        raise RuntimeError("Return backend sample plan mismatch")
    clip_model = str(raw.get("clip_model") or "")
    clip_fingerprint = str(raw.get("clip_model_fingerprint") or "")
    if clip_model != SAMPLING["clip_model"] or not clip_fingerprint:
        raise RuntimeError("Return backend CLIP identity mismatch")
    expected_pairs = plan["pairs"]
    raw_pairs = raw.get("evidence_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(expected_pairs):
        raise RuntimeError("Return backend returned the wrong pair count")
    pairs = []
    for expected, returned in zip(expected_pairs, raw_pairs):
        if not isinstance(returned, Mapping):
            raise RuntimeError("Return backend pair must be an object")
        if (
            returned.get("reference_chunk") != expected["reference_chunk"]
            or returned.get("return_chunk") != expected["return_chunk"]
        ):
            raise RuntimeError("Return backend pair identity mismatch")
        comparisons = returned.get("similarities")
        if not isinstance(comparisons, list):
            raise RuntimeError("Return backend pair has no similarities")
        expected_cross_product = {
            (reference, returned_index)
            for reference in expected["reference_sample_indices"]
            for returned_index in expected["return_sample_indices"]
        }
        actual_cross_product = set()
        normalized_comparisons = []
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise RuntimeError("Return frame comparison must be an object")
            key = (comparison.get("reference_sample"), comparison.get("return_sample"))
            actual_cross_product.add(key)
            components = {
                name: _finite_unit(comparison.get(name)) for name in FRAME_COMPONENTS
            }
            if any(value is None for value in components.values()):
                raise RuntimeError(
                    f"Return backend has invalid frame components at {key}"
                )
            rgb_diff = _finite_unit(comparison.get("rgb_mean_abs_diff"))
            if rgb_diff is None:
                raise RuntimeError(
                    f"Return backend has invalid RGB difference at {key}"
                )
            expected_rgb = clip01(1.0 - soft_threshold(rgb_diff, 0.05, 0.35))
            if not math.isclose(
                components["rgb_pixel_similarity"],
                expected_rgb,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    f"Return RGB score does not match its raw difference at {key}"
                )
            normalized_comparisons.append(
                {
                    "reference_sample": key[0],
                    "return_sample": key[1],
                    **components,
                    "rgb_mean_abs_diff": rgb_diff,
                }
            )
        if actual_cross_product != expected_cross_product or len(comparisons) != len(
            expected_cross_product
        ):
            raise RuntimeError(
                "Return backend comparisons do not match the planned cross product"
            )
        pairs.append(
            {
                "reference_chunk": expected["reference_chunk"],
                "return_chunk": expected["return_chunk"],
                "similarities": normalized_comparisons,
            }
        )
    evidence = {
        "pairs": pairs,
        "mean_diff": raw.get("mean_diff"),
        "peak_diff": raw.get("peak_diff"),
        "active_ratio": raw.get("active_ratio"),
        "clip_model": clip_model,
        "clip_model_fingerprint": clip_fingerprint,
    }
    result = result_from_evidence(
        evidence, fallback_first_last=bool(plan["fallback_first_last"])
    )
    if result["status"] not in {"ok", "fallback_first_last"}:
        raise RuntimeError(
            f"Return backend produced invalid evidence: {result['diagnostics']}"
        )
    return evidence


def validate_evidence(
    raw: Mapping[str, Any], sample_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate raw backend/import evidence against an exact sample plan."""

    return _normalize_backend_evidence(raw, sample_plan)


def _load_cache(path: Path, digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    provenance = payload.get("provenance") or {}
    plan = payload.get("sample_plan") or {}
    evidence = payload.get("evidence") or {}
    if not all(isinstance(item, Mapping) for item in (provenance, plan, evidence)):
        return None
    if (
        payload.get("schema_version") != CACHE_SCHEMA
        or payload.get("skill_id") != SKILL_ID
        or payload.get("definition_version") != DEFINITION_VERSION
        or provenance.get("cache_contract_version") != CACHE_CONTRACT.version
        or provenance.get("input_digest") != digest
    ):
        return None
    raw = {
        "schema_version": BACKEND_OUTPUT_SCHEMA,
        "sampled_frame_indices": plan.get("sampled_frame_indices"),
        "pairs": plan.get("pairs"),
        "fallback_first_last": plan.get("fallback_first_last"),
        "evidence_pairs": evidence.get("pairs"),
        "mean_diff": evidence.get("mean_diff"),
        "peak_diff": evidence.get("peak_diff"),
        "active_ratio": evidence.get("active_ratio"),
        "clip_model": evidence.get("clip_model"),
        "clip_model_fingerprint": evidence.get("clip_model_fingerprint"),
    }
    try:
        normalized = _normalize_backend_evidence(raw, plan)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    result = result_from_evidence(
        normalized, fallback_first_last=bool(plan.get("fallback_first_last"))
    )
    payload = dict(payload)
    payload["evidence"] = normalized
    return payload if result["status"] in {"ok", "fallback_first_last"} else None


def _response(
    path: Path, payload: Mapping[str, Any], cache_hit: bool
) -> dict[str, Any]:
    plan = payload.get("sample_plan") or {}
    return {
        "cache_hit": cache_hit,
        "cache_path": str(path),
        "result": result_from_evidence(
            payload.get("evidence") or {},
            fallback_first_last=bool(plan.get("fallback_first_last")),
        ),
        "sample_plan": plan,
        "provenance": payload.get("provenance") or {},
    }


def _write_cache(
    path: Path,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "sample_plan": dict(plan),
        "evidence": dict(evidence),
        "provenance": dict(provenance),
    }
    atomic_write_json(path, payload)
    return _response(path, payload, False)


def import_evidence(
    video_path: Path,
    case: Mapping[str, Any],
    raw_evidence: Mapping[str, Any],
    backend: SkillBackendIdentity,
    cache_root: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Import raw components only after reproducing the exact sample plan."""

    video_path = video_path.expanduser().resolve()
    video = file_fingerprint(video_path)
    plan = resolve_sample_plan(video_path, case)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video, case, plan)
    path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)
    evidence = _normalize_backend_evidence(raw_evidence, plan)
    return _write_cache(
        path,
        plan,
        evidence,
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
    backend: ReturnConsistencyBackend,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one video, returning before decode/model work on a cache hit."""

    video_path = video_path.expanduser().resolve()
    video = file_fingerprint(video_path)
    plan = resolve_sample_plan(video_path, case)
    identity = _backend_identity(backend)
    digest = _input_digest(identity, video, case, plan)
    path = cache_path_for_video(video_path, cache_root)
    cached = _load_cache(path, digest)
    if cached is not None:
        return _response(path, cached, True)
    raw = backend.evaluate(video_path, video, plan)
    evidence = _normalize_backend_evidence(raw, plan)
    return _write_cache(
        path,
        plan,
        evidence,
        {
            "input_digest": digest,
            "cache_contract_version": CACHE_CONTRACT.version,
            "video": video,
            "backend": identity,
            "execution_mode": str(backend.execution_mode),
        },
    )
