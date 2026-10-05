"""Offscreen Evolution VLM evaluation and cache contract."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from ..aggregate import clean_float, mean_valid, numeric, skill_result
from ..io import atomic_write_json, file_fingerprint, read_json, value_digest
from .common import MetricCacheContract, SkillBackendIdentity, skill_cache_path


SKILL_ID = "offscreen_evolution_verifier"
DEFINITION_VERSION = "harnesseval.offscreen_evolution"
PROMPT_VERSION = "harnesseval.offscreen_evolution.prompt"
RESULT_CACHE_SCHEMA = "harnesseval.offscreen_evolution.result"
ANALYZE_AGENT_CACHE_SCHEMA = "harnesseval.offscreen_evolution.analyze_agent"
VERIFY_AGENT_CACHE_SCHEMA = "harnesseval.offscreen_evolution.verify_agent"
BACKEND_OUTPUT_SCHEMA = "harnesseval.offscreen_evolution.backend_output"
SAMPLING = {
    "method": "uniform_full_video",
    "max_frames": 12,
    "chunk_assignment": "normalized_temporal_midpoint",
}
Q_FIELDS = (
    "Q1_pre_visible",
    "Q2_offscreen_absent",
    "Q3_return_visible",
    "Q4_anchor_preservation",
    "Q5_no_reset",
    "Q6_state_change_visible",
    "Q7_evolution_plausible",
    "Q8_judgeable",
)
Q_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.offscreen_evolution.cache_contract",
    source_skill_ids=(SKILL_ID,),
    required_metrics=Q_FIELDS,
    accepts_source_skill_score=False,
    compatibility_fields=(
        "video_fingerprint",
        "initial_observation_sha256",
        "canonical_spec",
        "analyze_agent_prompt",
        "verify_agent_prompt",
        "sampled_frames",
        "frame_chunk_labels",
        "judge_backend",
    ),
)

__all__ = [
    "SKILL_ID",
    "DEFINITION_VERSION",
    "PROMPT_VERSION",
    "RESULT_CACHE_SCHEMA",
    "ANALYZE_AGENT_CACHE_SCHEMA",
    "VERIFY_AGENT_CACHE_SCHEMA",
    "BACKEND_OUTPUT_SCHEMA",
    "SAMPLING",
    "Q_FIELDS",
    "Q_LEVELS",
    "CACHE_CONTRACT",
    "OffscreenEvolutionBackend",
    "canonical_spec",
    "expected_spec_messages",
    "video_judge_messages",
    "normalize_q_scores",
    "aggregate_q_scores",
    "result_from_judgment",
    "is_compatible_result",
    "cache_paths",
    "evaluate",
]


class OffscreenEvolutionBackend(SkillBackendIdentity, Protocol):
    def infer(self, messages: list[dict[str, Any]]) -> Mapping[str, Any]: ...


def _chunk_text(chunk: Mapping[str, Any]) -> str | None:
    action = chunk.get("action")
    nested = action if isinstance(action, Mapping) else {}
    value = nested.get("text") if nested else None
    if value is None:
        value = chunk.get("text")
    return str(value) if value is not None else None


def _chunk_controls(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    action = chunk.get("action")
    nested = action if isinstance(action, Mapping) else {}
    controls = nested.get("controls") if nested else None
    if controls is None and nested.get("type") == "navigation":
        controls = [nested]
    if controls is None:
        controls = chunk.get("controls") or chunk.get("actions")
    if isinstance(controls, Mapping):
        return [dict(controls)]
    if isinstance(controls, list):
        return [dict(item) for item in controls if isinstance(item, Mapping)]
    return []


def _chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    values = [str(item.get("chunk_id") or "") for item in chunks]
    if not values or any(not value for value in values):
        raise ValueError("Offscreen requires non-empty action chunk ids")
    if len(values) != len(set(values)):
        raise ValueError("Offscreen action chunk ids must be unique")
    return values


def _chunk_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _visibility_groups(value: Any) -> dict[str, list[str]]:
    names = ("pre_visible", "offscreen", "return_visible")
    if isinstance(value, Mapping):
        if any(name in value for name in names):
            return {name: _chunk_list(value.get(name)) for name in names}
        groups = {name: [] for name in names}
        for chunk_id, flags in value.items():
            if not isinstance(flags, Mapping):
                continue
            resolved_chunk_id = str(chunk_id).strip()
            if not resolved_chunk_id:
                continue
            for name in names:
                if flags.get(name) is True:
                    groups[name].append(resolved_chunk_id)
        return groups
    groups = {name: [] for name in names}
    if not isinstance(value, (list, tuple)):
        return groups
    for item in value:
        if not isinstance(item, Mapping):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        for name in names:
            if item.get(name) is True:
                groups[name].append(chunk_id)
    return groups


def _visibility_from_window(
    non_model: Mapping[str, Any], chunk_ids: list[str]
) -> dict[str, list[str]]:
    window = non_model.get("offscreen_window") or {}
    if not isinstance(window, Mapping):
        return {}
    leave = str(window.get("leave_chunk") or "")
    returned = str(window.get("return_chunk") or "")
    if leave not in chunk_ids or returned not in chunk_ids:
        return {}
    leave_position = chunk_ids.index(leave)
    return_position = chunk_ids.index(returned)
    if return_position <= leave_position:
        return {}
    return {
        "pre_visible": [chunk_ids[max(0, leave_position - 1)]],
        "offscreen": chunk_ids[leave_position + 1 : return_position],
        "return_visible": [chunk_ids[return_position]],
    }


def canonical_spec(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build the authoritative offscreen-evolution specification from the case."""

    action = (case.get("interaction") or {}).get("action") or {}
    non_model = case.get("non_model_facing") or {}
    evolution = non_model.get("offscreen_evolution_spec") or {}
    chunks = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "text": _chunk_text(chunk),
            "controls": _chunk_controls(chunk),
        }
        for chunk in action.get("chunks") or []
        if isinstance(chunk, Mapping)
    ]
    chunk_ids = _chunk_ids(chunks)
    visibility = non_model.get("visibility_plan")
    if isinstance(visibility, Mapping):
        visibility = {
            name: _chunk_list(visibility.get(name))
            for name in ("pre_visible", "offscreen", "return_visible")
        }
    else:
        visibility = _visibility_from_window(non_model, chunk_ids)
    evolvable = (
        evolution.get("evolvable_elements")
        or non_model.get("evolvable_elements")
        or action.get("evolvable_elements")
        or action.get("evolvable_element")
        or []
    )
    if isinstance(evolvable, Mapping):
        evolvable = [dict(evolvable)]
    return clean_float(
        {
            "target_process": evolution.get("target_process"),
            "evolvable_elements": evolvable,
            "stable_anchors": non_model.get("stable_anchors")
            or non_model.get("protected_anchors")
            or [],
            "visibility_plan": visibility,
            "expected_evolution": evolution.get("expected_evolution") or [],
            "failure_modes": evolution.get("failure_modes") or [],
            "top_level_action_text": action.get("text") or "",
            "chunks": chunks,
            "return_window": non_model.get("return_window"),
            "offscreen_window": non_model.get("offscreen_window"),
        }
    )


def _expected_spec_is_resolved(expected_spec: Mapping[str, Any]) -> bool:
    visibility = _visibility_groups(expected_spec.get("visibility_timeline"))
    return bool(expected_spec.get("target_process")) and all(
        _chunk_list(visibility.get(name))
        for name in ("pre_visible", "offscreen", "return_visible")
    )


def _jpeg_data_url_from_rgb(frame: Any, max_side: int = 512, quality: int = 76) -> str:
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("failed to encode sampled Offscreen frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _jpeg_data_url_from_path(path: Path) -> str:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"initial observation cannot be decoded: {path}")
    return _jpeg_data_url_from_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _sample_video(video_path: Path) -> tuple[list[Any], dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video cannot be decoded: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"video has no frames: {video_path}")
    count = min(int(SAMPLING["max_frames"]), frame_count)
    indices = (
        [0]
        if count == 1
        else [int(index * (frame_count - 1) / (count - 1)) for index in range(count)]
    )
    frames = []
    wanted = set(indices)
    cursor = 0
    while wanted:
        ok, bgr = cap.read()
        if not ok:
            break
        if cursor in wanted:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            wanted.remove(cursor)
        cursor += 1
    cap.release()
    if wanted or len(frames) != len(indices):
        raise RuntimeError(f"video did not yield sampled frames: {sorted(wanted)}")
    return frames, {
        **SAMPLING,
        "source_frame_count": frame_count,
        "source_fps": fps,
        "sample_indices": indices,
    }


def expected_spec_messages(
    spec: Mapping[str, Any], initial_observation_path: Path
) -> list[dict[str, Any]]:
    action_view = {
        "top_level_action_text": spec.get("top_level_action_text"),
        "chunks": spec.get("chunks"),
    }
    content = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    "Infer the expected offscreen-evolution specification for this world-model case.",
                    "Use only the initial observation and model-facing action. Do not score a generated video.",
                    "Identify the target process, stable anchors, when the target should be visible or offscreen, and how it should continue evolving.",
                    "Return JSON only with keys: target_process, stable_anchors, visibility_timeline, expected_evolution, failure_modes.",
                    "visibility_timeline must be a JSON object whose keys are pre_visible, offscreen, and return_visible.",
                    'Each visibility_timeline value must be an array containing only chunk ids, for example: {"pre_visible":["c01"],"offscreen":["c03","c04"],"return_visible":["c06","c07"]}.',
                    "Do not put prose descriptions or boolean values in visibility_timeline.",
                    "",
                    json.dumps(action_view, ensure_ascii=False, indent=2),
                ]
            ),
        },
        {"type": "text", "text": "Initial observation image:"},
        {
            "type": "image_url",
            "image_url": _jpeg_data_url_from_path(initial_observation_path),
        },
    ]
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a careful benchmark case-specification assistant. Return valid JSON only.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def _frame_chunk_labels(
    spec: Mapping[str, Any], sampling: Mapping[str, Any]
) -> list[str]:
    chunk_ids = [
        str(item.get("chunk_id"))
        for item in spec.get("chunks") or []
        if isinstance(item, Mapping) and item.get("chunk_id")
    ]
    indices = sampling.get("sample_indices") or []
    frame_count = int(sampling.get("source_frame_count") or 0)
    if not chunk_ids or frame_count <= 0 or len(indices) == 0:
        raise ValueError(
            "Offscreen frame-to-chunk labeling requires chunks and samples"
        )
    return [
        chunk_ids[
            min(
                len(chunk_ids) - 1,
                int(((int(frame_index) + 0.5) / frame_count) * len(chunk_ids)),
            )
        ]
        for frame_index in indices
    ]


def video_judge_messages(
    spec: Mapping[str, Any],
    expected_spec: Mapping[str, Any],
    initial_observation_path: Path,
    frames: list[Any],
    sampling: Mapping[str, Any],
) -> list[dict[str, Any]]:
    content = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    "Evaluate this generated rollout against the canonical offscreen-evolution case specification.",
                    "Use the canonical specification as authoritative. The expected specification is only a case-audit reference.",
                    "Each question score must be exactly one of: 0, 0.25, 0.5, 0.75, 1.",
                    "Use 0 for absent or contradicted, 0.25 for weak evidence, 0.5 for partial or ambiguous evidence, 0.75 for mostly satisfied, and 1 for clearly and fully satisfied.",
                    "Judge visibility using the supplied approximate chunk labels. Do not infer successful offscreen evolution merely from generic motion or visual change.",
                    "",
                    "Questions:",
                    "Q1_pre_visible: the target process is clearly visible before the look-away phase.",
                    "Q2_offscreen_absent: the target process is out of view during intended offscreen chunks.",
                    "Q3_return_visible: the same target process returns afterward.",
                    "Q4_anchor_preservation: stable anchors preserve the identity of the world after return.",
                    "Q5_no_reset: the video avoids a hard cut, scene reset, or replacement world.",
                    "Q6_state_change_visible: the returned target state visibly differs from its pre-offscreen state.",
                    "Q7_evolution_plausible: the change matches continuous expected evolution rather than random deformation.",
                    "Q8_judgeable: the video is clear, non-static, and sufficiently observable to judge.",
                    "",
                    "Return JSON only:",
                    '{"q_scores":{"Q1_pre_visible":0|0.25|0.5|0.75|1,...},"reasons":{"Q1_pre_visible":"..."},"warnings":[],"summary":"..."}',
                    "",
                    "Canonical case specification:",
                    json.dumps(spec, ensure_ascii=False, indent=2),
                    "",
                    "Expected specification inferred before seeing the video, for audit only:",
                    json.dumps(expected_spec, ensure_ascii=False, indent=2),
                ]
            ),
        },
        {"type": "text", "text": "Initial observation image:"},
        {
            "type": "image_url",
            "image_url": _jpeg_data_url_from_path(initial_observation_path),
        },
    ]
    labels = _frame_chunk_labels(spec, sampling)
    indices = sampling.get("sample_indices") or []
    for position, (frame, chunk_id, frame_index) in enumerate(
        zip(frames, labels, indices)
    ):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Generated video sample {position:02d}, source frame "
                    f"{int(frame_index)}, approximate chunk {chunk_id}:"
                ),
            }
        )
        content.append(
            {"type": "image_url", "image_url": _jpeg_data_url_from_rgb(frame)}
        )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict VLM evaluator for offscreen world-state evolution. Return valid JSON only.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def _q_score(value: Any) -> float | None:
    score = None if isinstance(value, bool) else numeric(value)
    if score is None:
        return None
    if score < 0.125:
        return 0.0
    if score < 0.375:
        return 0.25
    if score < 0.625:
        return 0.5
    return 0.75 if score < 0.875 else 1.0


def normalize_q_scores(judgment: Mapping[str, Any]) -> dict[str, float | None]:
    raw = judgment.get("q_scores") or judgment.get("scores") or {}
    raw = raw if isinstance(raw, Mapping) else {}
    aliases = (
        ("Q1_pre_visible", "Q1", "pre_visible"),
        ("Q2_offscreen_absent", "Q2", "offscreen_absent"),
        ("Q3_return_visible", "Q3", "return_visible"),
        ("Q4_anchor_preservation", "Q4", "anchor_preservation"),
        ("Q5_no_reset", "Q5", "no_reset"),
        ("Q6_state_change_visible", "Q6", "state_change_visible"),
        ("Q7_evolution_plausible", "Q7", "evolution_plausible"),
        ("Q8_judgeable", "Q8", "judgeable"),
    )
    result = {}
    for canonical, *names in aliases:
        key = next((name for name in (canonical, *names) if name in raw), None)
        result[canonical] = _q_score(raw.get(key)) if key else None
    return result


def _weighted_mean(
    values: Mapping[str, Any], weights: Mapping[str, float]
) -> float | None:
    valid = [
        (float(values[key]), weight)
        for key, weight in weights.items()
        if numeric(values.get(key)) is not None
    ]
    total_weight = sum(weight for _, weight in valid)
    if not total_weight:
        return None
    return round(sum(value * weight for value, weight in valid) / total_weight, 6)


def aggregate_q_scores(q: Mapping[str, Any]) -> dict[str, float | None]:
    visibility = mean_valid(
        (
            q.get("Q1_pre_visible"),
            q.get("Q2_offscreen_absent"),
            q.get("Q3_return_visible"),
        )
    )
    validity = mean_valid((q.get("Q5_no_reset"), q.get("Q8_judgeable")))
    evolution = _weighted_mean(
        {
            "state_change": q.get("Q6_state_change_visible"),
            "plausible": q.get("Q7_evolution_plausible"),
        },
        {"state_change": 0.4, "plausible": 0.6},
    )
    core = _weighted_mean(
        {
            "anchor": q.get("Q4_anchor_preservation"),
            "evolution": evolution,
        },
        {"anchor": 0.35, "evolution": 0.65},
    )
    final = (
        round(float(visibility) * float(validity) * float(core), 6)
        if all(numeric(value) is not None for value in (visibility, validity, core))
        else None
    )
    return {
        "visibility_score": visibility,
        "validity_gate": validity,
        "anchor_score": q.get("Q4_anchor_preservation"),
        "evolution_score": evolution,
        "core_score": core,
        "final_score": final,
    }


def _text_tokens(value: Any) -> set[str]:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    elif isinstance(value, Mapping):
        value = json.dumps(value, ensure_ascii=False)
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", str(value or "").lower())
        if len(token) >= 4
    }


def _overlap(reference: Any, candidate: Any) -> float | None:
    reference_tokens = _text_tokens(reference)
    candidate_tokens = _text_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return None
    return round(len(reference_tokens & candidate_tokens) / len(reference_tokens), 6)


def case_audit(
    spec: Mapping[str, Any], expected_spec: Mapping[str, Any]
) -> dict[str, Any]:
    target = _overlap(
        [
            spec.get("target_process"),
            spec.get("evolvable_elements"),
            spec.get("expected_evolution"),
        ],
        [
            expected_spec.get("target_process"),
            expected_spec.get("expected_evolution"),
        ],
    )
    anchors = _overlap(spec.get("stable_anchors"), expected_spec.get("stable_anchors"))
    canonical_visibility = spec.get("visibility_plan") or {}
    expected_visibility = _visibility_groups(expected_spec.get("visibility_timeline"))
    visibility_fields = []
    for name in ("pre_visible", "offscreen", "return_visible"):
        reference = set(_chunk_list(canonical_visibility.get(name)))
        if reference:
            candidate = set(_chunk_list(expected_visibility.get(name)))
            visibility_fields.append(len(reference & candidate) / len(reference))
    visibility = mean_valid(visibility_fields)
    overall = mean_valid((target, anchors, visibility))
    warning = overall is None or overall < 0.4
    return {
        "status": "warning" if warning else "ok",
        "warning": warning,
        "expected_spec_alignment": overall,
        "target_alignment": target,
        "anchor_alignment": anchors,
        "visibility_alignment": visibility,
    }


def result_from_judgment(
    judgment: Mapping[str, Any],
    spec: Mapping[str, Any],
    expected_spec: Mapping[str, Any],
) -> dict[str, Any]:
    q_scores = normalize_q_scores(judgment)
    parts = aggregate_q_scores(q_scores)
    audit = case_audit(spec, expected_spec)
    score = parts["final_score"]
    status = "ok" if score is not None else "invalid"
    if audit["warning"]:
        status += "_case_audit_warning"
    return skill_result(
        SKILL_ID,
        status,
        score,
        metrics={
            **q_scores,
            **parts,
            "expected_spec_alignment": audit["expected_spec_alignment"],
        },
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "case_audit": audit,
        },
        notes=[
            "analyze_agent alignment is a case-audit warning only; scoring uses the canonical "
            "offscreen-evolution specification and generated video"
        ],
    )


def is_compatible_result(value: Mapping[str, Any] | None) -> bool:
    """Accept only independently versioned five-level Offscreen results."""

    if not isinstance(value, Mapping) or numeric(value.get("score")) is None:
        return False
    diagnostics = value.get("diagnostics") or {}
    metrics = value.get("metrics") or {}
    if not isinstance(diagnostics, Mapping) or not isinstance(metrics, Mapping):
        return False
    if (
        diagnostics.get("definition_version") != DEFINITION_VERSION
        or diagnostics.get("prompt_version") != PROMPT_VERSION
        or diagnostics.get("cache_contract_version") != CACHE_CONTRACT.version
    ):
        return False
    return all(
        numeric(metrics.get(name)) is not None and float(metrics[name]) in Q_LEVELS
        for name in Q_FIELDS
    )


def cache_paths(
    video_path: Path, case_id: str, cache_root: Path | None = None
) -> dict[str, Path]:
    result = skill_cache_path(video_path, "offscreen_evolution/results", cache_root)
    skill_root = result.parents[3]
    model_id, family = result.parents[1].name, result.parent.name
    return {
        "analyze_agent": skill_root / "analyze_agent" / f"{case_id}.json",
        "verify_agent": (
            skill_root / "verify_agent" / model_id / family / f"{case_id}.json"
        ),
        "result": result,
    }


def _backend_identity(backend: SkillBackendIdentity) -> dict[str, str]:
    identity = {
        "backend_id": str(backend.backend_id).strip(),
        "backend_version": str(backend.version).strip(),
        "config_digest": str(backend.config_digest).strip(),
    }
    missing = [key for key, value in identity.items() if not value]
    if missing:
        raise ValueError(f"Offscreen backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "api":
        raise ValueError("Offscreen backend execution_mode must be api")
    return identity


def _load_cache(path: Path, schema: str, input_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("schema_version") != schema
        or payload.get("skill_id") != SKILL_ID
        or payload.get("definition_version") != DEFINITION_VERSION
        or payload.get("prompt_version") != PROMPT_VERSION
        or payload.get("input_digest") != input_digest
    ):
        return None
    return dict(payload)


def _backend_output(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("Offscreen backend output schema mismatch")
    parsed = value.get("parsed")
    if not isinstance(parsed, Mapping):
        raise RuntimeError("Offscreen backend returned no parsed JSON object")
    return clean_float(
        {
            "parsed": dict(parsed),
            "raw_output_text": value.get("raw_output_text"),
            "raw_response": value.get("raw_response"),
            "response_metadata": value.get("response_metadata") or {},
            "provenance": value.get("provenance") or {},
        }
    )


def _agent_payload(
    *,
    schema: str,
    agent: str,
    case_id: str,
    input_digest: str,
    messages: list[dict[str, Any]],
    output: Mapping[str, Any],
    parsed_name: str,
    provenance: Mapping[str, Any],
    sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "agent": agent,
        "case_id": case_id,
        "input_digest": input_digest,
        parsed_name: output["parsed"],
        "conversation": messages,
        "raw_output_text": output.get("raw_output_text"),
        "raw_response": output.get("raw_response"),
        "response_metadata": output.get("response_metadata") or {},
        "provenance": {
            **dict(provenance),
            "backend_output": dict(output.get("provenance") or {}),
        },
    }
    if sampling is not None:
        payload["sampling"] = dict(sampling)
    payload["evidence_digest"] = value_digest(payload)
    return payload


def _result_response(
    paths: Mapping[str, Path], payload: Mapping[str, Any], cache_hit: bool
) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "cache_path": str(paths["result"]),
        "analyze_agent_cache_path": str(paths["analyze_agent"]),
        "verify_agent_cache_path": str(paths["verify_agent"]),
        "result": payload["result"],
        "provenance": payload.get("provenance") or {},
    }


def evaluate(
    video_path: Path,
    case: Mapping[str, Any],
    initial_observation_path: Path,
    backend: OffscreenEvolutionBackend,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one video; a valid result cache avoids decoding and inference."""

    video_path = video_path.expanduser().resolve()
    initial_observation_path = initial_observation_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not initial_observation_path.is_file():
        raise FileNotFoundError(initial_observation_path)
    case_id = str(case.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("Offscreen case_id is required")
    family = str(((case.get("taxonomy") or {}).get("probe_family") or ""))
    if family != "offscreen_evolution":
        raise ValueError(f"Offscreen skill does not accept family {family!r}")
    spec = canonical_spec(case)
    visibility = spec.get("visibility_plan") or {}
    if not spec.get("target_process") or not all(
        _chunk_list(visibility.get(name))
        for name in ("pre_visible", "offscreen", "return_visible")
    ):
        raise ValueError(
            "Offscreen canonical spec requires target_process and complete visibility_plan"
        )

    paths = cache_paths(video_path, case_id, cache_root)
    identity = _backend_identity(backend)
    video = file_fingerprint(video_path, include_sha256=True)
    initial = file_fingerprint(initial_observation_path, include_sha256=True)
    common_input = {
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "cache_contract_version": CACHE_CONTRACT.version,
        "case_id": case_id,
        "canonical_spec": spec,
        "initial_observation": initial,
        "backend": identity,
    }
    analyze_agent_digest = value_digest(
        {**common_input, "schema": ANALYZE_AGENT_CACHE_SCHEMA}
    )
    analyze_agent = _load_cache(
        paths["analyze_agent"], ANALYZE_AGENT_CACHE_SCHEMA, analyze_agent_digest
    )
    expected_spec = analyze_agent.get("expected_spec") if analyze_agent else None
    if not isinstance(expected_spec, Mapping) or not _expected_spec_is_resolved(
        expected_spec
    ):
        analyze_agent = None
    analyze_agent_hit = analyze_agent is not None
    if analyze_agent is None:
        messages = expected_spec_messages(spec, initial_observation_path)
        output = _backend_output(backend.infer(messages))
        analyze_agent = _agent_payload(
            schema=ANALYZE_AGENT_CACHE_SCHEMA,
            agent="analyze_agent",
            case_id=case_id,
            input_digest=analyze_agent_digest,
            messages=messages,
            output=output,
            parsed_name="expected_spec",
            provenance={"backend": identity, "execution_mode": backend.execution_mode},
        )
        expected_spec = analyze_agent.get("expected_spec")
        if not isinstance(expected_spec, Mapping):
            raise RuntimeError("Offscreen analyze_agent response has no expected_spec")
        if not _expected_spec_is_resolved(expected_spec):
            raise ValueError(
                "Offscreen analyze_agent did not resolve the target process and visibility plan"
            )
        atomic_write_json(paths["analyze_agent"], analyze_agent)
    expected_spec = analyze_agent.get("expected_spec")
    if not isinstance(expected_spec, Mapping):
        raise RuntimeError("Offscreen analyze_agent cache has no expected_spec")

    verify_agent_digest = value_digest(
        {
            **common_input,
            "schema": VERIFY_AGENT_CACHE_SCHEMA,
            "video": video,
            "sampling": SAMPLING,
            "analyze_agent_evidence_digest": analyze_agent.get("evidence_digest"),
        }
    )
    verify_agent = _load_cache(
        paths["verify_agent"], VERIFY_AGENT_CACHE_SCHEMA, verify_agent_digest
    )
    verify_agent_hit = verify_agent is not None
    if verify_agent is None:
        frames, sampling = _sample_video(video_path)
        messages = video_judge_messages(
            spec, expected_spec, initial_observation_path, frames, sampling
        )
        output = _backend_output(backend.infer(messages))
        verify_agent = _agent_payload(
            schema=VERIFY_AGENT_CACHE_SCHEMA,
            agent="verify_agent",
            case_id=case_id,
            input_digest=verify_agent_digest,
            messages=messages,
            output=output,
            parsed_name="judgment",
            sampling={
                **sampling,
                "frame_chunk_labels": _frame_chunk_labels(spec, sampling),
            },
            provenance={"backend": identity, "execution_mode": backend.execution_mode},
        )
        atomic_write_json(paths["verify_agent"], verify_agent)
    judgment = verify_agent.get("judgment")
    if not isinstance(judgment, Mapping):
        raise RuntimeError("Offscreen verify_agent cache has no judgment")

    result_digest = value_digest(
        {
            **common_input,
            "schema": RESULT_CACHE_SCHEMA,
            "video": video,
            "analyze_agent_evidence_digest": analyze_agent.get("evidence_digest"),
            "verify_agent_evidence_digest": verify_agent.get("evidence_digest"),
        }
    )
    cached_result = _load_cache(paths["result"], RESULT_CACHE_SCHEMA, result_digest)
    if cached_result is not None:
        return _result_response(paths, cached_result, True)

    result = result_from_judgment(judgment, spec, expected_spec)
    result["diagnostics"].update(
        {
            "analyze_agent_cache": str(paths["analyze_agent"]),
            "analyze_agent_evidence_digest": analyze_agent.get("evidence_digest"),
            "verify_agent_cache": str(paths["verify_agent"]),
            "verify_agent_evidence_digest": verify_agent.get("evidence_digest"),
        }
    )
    payload = {
        "schema_version": RESULT_CACHE_SCHEMA,
        "skill_id": SKILL_ID,
        "definition_version": DEFINITION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "input_digest": result_digest,
        "result": result,
        "provenance": {
            "backend": identity,
            "video": video,
            "initial_observation": initial,
            "analyze_agent_cache_hit": analyze_agent_hit,
            "verify_agent_cache_hit": verify_agent_hit,
        },
    }
    atomic_write_json(paths["result"], payload)
    return _result_response(paths, payload, False)
