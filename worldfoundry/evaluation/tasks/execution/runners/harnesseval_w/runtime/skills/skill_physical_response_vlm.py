"""HarnessEval Physical Response two-agent VLM evaluation and cache contract."""

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


SKILL_ID = "physical_response_verifier_vlm"
DEFINITION_VERSION = "harnesseval.physical_response_vlm"
PROMPT_VERSION = "harnesseval.physical_response.prompt"
RESULT_CACHE_SCHEMA = "harnesseval.physical_response.result"
ANALYZE_AGENT_CACHE_SCHEMA = "harnesseval.physical_response.analyze_agent"
VERIFY_AGENT_CACHE_SCHEMA = "harnesseval.physical_response.verify_agent"
BACKEND_OUTPUT_SCHEMA = "harnesseval.physical_response.backend_output"
SAMPLING = {"method": "uniform_full_video", "max_frames": 12}
Q_FIELDS = (
    "Q1_setup_visible",
    "Q2_intervention_visible",
    "Q3_response_presence",
    "Q4_condition_following",
    "Q5_causal_order",
    "Q6_anchor_preservation",
    "Q7_no_physical_artifacts",
    "Q8_judgeable",
)
Q_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
CACHE_CONTRACT = MetricCacheContract(
    version="harnesseval.physical_response.cache_contract",
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
    "PhysicalResponseBackend",
    "selected_variant",
    "canonical_spec",
    "expected_spec_messages",
    "video_judge_messages",
    "normalize_q_scores",
    "aggregate_q_scores",
    "result_from_judgment",
    "cache_paths",
    "evaluate",
]


class PhysicalResponseBackend(SkillBackendIdentity, Protocol):
    def infer(self, messages: list[dict[str, Any]]) -> Mapping[str, Any]: ...


def _runtime_action(runtime_metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    metadata = runtime_metadata or {}
    direct = metadata.get("interaction_action")
    if isinstance(direct, Mapping):
        return direct
    nested = (metadata.get("interaction") or {}).get("action")
    return nested if isinstance(nested, Mapping) else {}


def selected_variant(
    case: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any] | None = None,
    action_id: str | None = None,
) -> dict[str, Any]:
    action = ((case.get("interaction") or {}).get("action") or {})
    runtime_action = _runtime_action(runtime_metadata)
    selected = runtime_action.get("selected_variant") or action.get("selected_variant")
    if isinstance(selected, Mapping):
        return dict(selected)
    target_id = action_id or runtime_action.get("action_id") or action.get("action_id")
    for variants in (runtime_action.get("variants"), action.get("variants")):
        for variant in variants or []:
            if isinstance(variant, Mapping) and variant.get("action_id") == target_id:
                return dict(variant)
    return {}


def canonical_spec(
    case: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any] | None = None,
    action_id: str | None = None,
) -> dict[str, Any]:
    """Build the authoritative physical-transition specification."""

    action = ((case.get("interaction") or {}).get("action") or {})
    runtime_action = _runtime_action(runtime_metadata)
    non_model = case.get("non_model_facing") or {}
    selected = selected_variant(case, runtime_metadata, action_id)
    return clean_float(
        {
            "action_type": action.get("type") or runtime_action.get("type"),
            "action_id": (
                selected.get("action_id")
                or action_id
                or runtime_action.get("action_id")
                or action.get("action_id")
            ),
            "selected_variant_text": (
                selected.get("text")
                or action.get("text")
                or runtime_action.get("text")
                or ""
            ),
            "expected_outcome": non_model.get("expected_outcome") or [],
            "protected_anchors": non_model.get("protected_anchors") or [],
            "negative_constraints": non_model.get("negative_constraints") or [],
            "source_tags": (
                (case.get("world") or {}).get("source_tags")
                or runtime_metadata and runtime_metadata.get("source_tags")
                or {}
            ),
            "rollout_horizon": action.get("rollout_horizon")
            or runtime_action.get("rollout_horizon"),
        }
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
        raise RuntimeError("failed to encode sampled frame")
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
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if cursor in wanted:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cursor += 1
    cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(
            f"decoded {len(frames)} of {len(indices)} requested sample frames"
        )
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
        "selected_variant_text": spec.get("selected_variant_text"),
        "source_tags": spec.get("source_tags"),
        "rollout_horizon": spec.get("rollout_horizon"),
    }
    content = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    "Infer the expected physical-response specification for this world-model case.",
                    "Use only the initial observation and the selected model-facing physical action. Do not score any generated video.",
                    "Return JSON only with keys: target_system, intervention, controlled_variables, expected_response, causal_checks, protected_anchors, failure_modes.",
                    "causal_checks should be short physical cause-effect statements that can be judged from generated video frames.",
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


def video_judge_messages(
    spec: Mapping[str, Any],
    expected_spec: Mapping[str, Any],
    initial_observation_path: Path,
    frames: list[Any],
) -> list[dict[str, Any]]:
    content = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    "Evaluate this generated rollout against the canonical physical-transition case spec.",
                    "Use the canonical spec as authoritative. The expected spec is provided only as a case-audit reference.",
                    "Each question score must be exactly one of: 0, 0.25, 0.5, 0.75, 1.",
                    "Reward motion only when it is causally tied to the selected physical intervention and expected response.",
                    "",
                    "Questions:",
                    "Q1_setup_visible: the physical setup and target system are clearly visible.",
                    "Q2_intervention_visible: the specified physical intervention or condition is visibly instantiated.",
                    "Q3_response_presence: the target system responds with real physical motion or a state change.",
                    "Q4_condition_following: the response is consistent with the selected physical parameter or action.",
                    "Q5_causal_order: cause and effect occur in the correct order, with no effect before cause.",
                    "Q6_anchor_preservation: apparatus, camera, scale markers, and protected anchors remain fixed unless specified.",
                    "Q7_no_physical_artifacts: the video avoids teleporting, pass-through, hard cuts, replacement worlds, or extra forces.",
                    "Q8_judgeable: the video is clear, continuous, and sufficiently observable to judge.",
                    "",
                    "Return JSON only:",
                    '{"q_scores":{"Q1_setup_visible":0|0.25|0.5|0.75|1,...},"reasons":{"Q1_setup_visible":"..."},"warnings":[],"summary":"..."}',
                    "",
                    "Canonical case spec:",
                    json.dumps(spec, ensure_ascii=False, indent=2),
                    "",
                    "Expected spec inferred before seeing the video, for audit only:",
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
    for index, frame in enumerate(frames):
        content.append(
            {"type": "text", "text": f"Generated video sample {index:02d}:"}
        )
        content.append({"type": "image_url", "image_url": _jpeg_data_url_from_rgb(frame)})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict VLM evaluator for physical cause-effect response in generated worlds. Return valid JSON only.",
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
        ("Q1_setup_visible", "Q1", "setup_visible"),
        ("Q2_intervention_visible", "Q2", "intervention_visible"),
        ("Q3_response_presence", "Q3", "response_presence"),
        ("Q4_condition_following", "Q4", "condition_following"),
        ("Q5_causal_order", "Q5", "causal_order"),
        ("Q6_anchor_preservation", "Q6", "anchor_preservation"),
        ("Q7_no_physical_artifacts", "Q7", "no_physical_artifacts"),
        ("Q8_judgeable", "Q8", "judgeable"),
    )
    result = {}
    for canonical, *names in aliases:
        key = next((name for name in (canonical, *names) if name in raw), None)
        result[canonical] = _q_score(raw.get(key)) if key else None
    return result


def _weighted_mean(values: Mapping[str, Any], weights: Mapping[str, float]) -> float | None:
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
    intervention_grounding = mean_valid(
        (q.get("Q1_setup_visible"), q.get("Q2_intervention_visible"))
    )
    response_execution = _weighted_mean(
        {
            "response_presence": q.get("Q3_response_presence"),
            "condition_following": q.get("Q4_condition_following"),
            "causal_order": q.get("Q5_causal_order"),
        },
        {"response_presence": 0.45, "condition_following": 0.35, "causal_order": 0.20},
    )
    preservation = _weighted_mean(
        {
            "anchor_preservation": q.get("Q6_anchor_preservation"),
            "no_physical_artifacts": q.get("Q7_no_physical_artifacts"),
        },
        {"anchor_preservation": 0.55, "no_physical_artifacts": 0.45},
    )
    core = _weighted_mean(
        {
            "response_execution": response_execution,
            "preservation": preservation,
            "causal_order": q.get("Q5_causal_order"),
        },
        {
            "response_execution": 0.55,
            "preservation": 0.25,
            "causal_order": 0.20,
        },
    )
    judgeable = q.get("Q8_judgeable")
    final = (
        round(float(intervention_grounding) * float(judgeable) * float(core), 6)
        if all(
            numeric(value) is not None
            for value in (intervention_grounding, judgeable, core)
        )
        else None
    )
    return {
        "intervention_grounding": intervention_grounding,
        "response_execution": response_execution,
        "causal_order": q.get("Q5_causal_order"),
        "preservation": preservation,
        "validity_gate": judgeable,
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
        [spec.get("selected_variant_text"), spec.get("expected_outcome")],
        [expected_spec.get("target_system"), expected_spec.get("intervention")],
    )
    response = _overlap(
        [spec.get("selected_variant_text"), spec.get("expected_outcome")],
        [expected_spec.get("expected_response"), expected_spec.get("causal_checks")],
    )
    anchors = _overlap(
        spec.get("protected_anchors"), expected_spec.get("protected_anchors")
    )
    overall = mean_valid((target, response, anchors))
    warning = overall is None or overall < 0.35
    return {
        "status": "warning" if warning else "ok",
        "warning": warning,
        "expected_spec_alignment": overall,
        "target_alignment": target,
        "response_alignment": response,
        "anchor_alignment": anchors,
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
        metrics={**q_scores, **parts, "expected_spec_alignment": audit["expected_spec_alignment"]},
        diagnostics={
            "definition_version": DEFINITION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "cache_contract_version": CACHE_CONTRACT.version,
            "case_audit": audit,
        },
        notes=[
            "analyze_agent expected_spec_alignment is a case-audit warning only; "
            "the score uses the canonical physical spec and generated video"
        ],
    )


def cache_paths(
    video_path: Path, case_id: str, cache_root: Path | None = None
) -> dict[str, Path]:
    result = skill_cache_path(video_path, "physical_response/results", cache_root)
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
        raise ValueError(f"physical response backend identity missing: {', '.join(missing)}")
    if str(backend.execution_mode).strip() != "api":
        raise ValueError("physical response backend execution_mode must be api")
    return identity


def _load_cache(path: Path, schema: str, input_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != schema
        or payload.get("skill_id") != SKILL_ID
        or payload.get("definition_version") != DEFINITION_VERSION
        or payload.get("input_digest") != input_digest
    ):
        return None
    return payload


def _backend_output(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != BACKEND_OUTPUT_SCHEMA:
        raise RuntimeError("physical response backend output schema mismatch")
    parsed = value.get("parsed")
    if not isinstance(parsed, Mapping):
        raise RuntimeError("physical response backend returned no parsed JSON object")
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
    backend: PhysicalResponseBackend,
    cache_root: Path | None = None,
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
    action_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one video; a valid result cache avoids decoding and inference."""

    video_path = video_path.resolve()
    initial_observation_path = initial_observation_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not initial_observation_path.is_file():
        raise FileNotFoundError(initial_observation_path)
    case_id = str(case.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("physical response case_id is required")
    family = str(((case.get("taxonomy") or {}).get("probe_family") or ""))
    if family != "physical_transition":
        raise ValueError(f"physical response skill does not accept family {family!r}")
    spec = canonical_spec(case, runtime_metadata, action_id)
    if not spec.get("selected_variant_text"):
        raise ValueError("physical response selected_variant_text is required")

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
        atomic_write_json(paths["analyze_agent"], analyze_agent)
    expected_spec = analyze_agent.get("expected_spec")
    if not isinstance(expected_spec, Mapping):
        raise RuntimeError("physical response analyze_agent cache has no expected_spec")

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
            spec, expected_spec, initial_observation_path, frames
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
            sampling=sampling,
            provenance={"backend": identity, "execution_mode": backend.execution_mode},
        )
        atomic_write_json(paths["verify_agent"], verify_agent)
    judgment = verify_agent.get("judgment")
    if not isinstance(judgment, Mapping):
        raise RuntimeError("physical response verify_agent cache has no judgment")

    result_digest = value_digest(
        {
            **common_input,
            "schema": RESULT_CACHE_SCHEMA,
            "video": video,
            "sampling": SAMPLING,
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
