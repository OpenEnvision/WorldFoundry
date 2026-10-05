"""Scene-aware PAWEval label validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .loader import load_rubric

TRUST_STATUSES = {"TRUSTED", "QUESTIONABLE", "UNTRUSTED", "UNCERTAIN"}
BLOCK_STATUSES = {"PASS", "FAIL", "UNCLEAR", "NOT_APPLICABLE"}
YNUN_VALUES = {"YES", "NO", "UNCLEAR", "NOT_APPLICABLE"}
TRUST_BLOCKS = ("scene_grounding", "action_execution", "object_continuity", "physical_process")


@dataclass(frozen=True)
class ValidationResult:
    status: str
    payload: dict[str, Any] | None
    canonical_label: str | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return self.status == "valid"


def parse_response(raw: str | dict[str, Any]) -> ValidationResult:
    if isinstance(raw, dict):
        return ValidationResult(status="parsed", payload=raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ValidationResult(status="malformed_json", payload=None, errors=(str(exc),))
    if not isinstance(payload, dict):
        return ValidationResult(status="schema_invalid", payload=None, errors=("response must be a JSON object",))
    return ValidationResult(status="parsed", payload=payload)


def validate_outcome_response(raw: str | dict[str, Any], *, scene_id: str) -> ValidationResult:
    parsed = parse_response(raw)
    if parsed.payload is None:
        return parsed
    payload = parsed.payload
    errors: list[str] = []
    for key in ("scene_id", "outcome_readable", "outcome_in_schema"):
        if key not in payload:
            errors.append(f"missing required field: {key}")
    if not scene_id:
        errors.append("expected scene_id must be provided")
    elif payload.get("scene_id") != scene_id:
        errors.append(f"scene_id {payload.get('scene_id')!r} does not match expected scene_id {scene_id!r}")
    if errors:
        return ValidationResult(status="schema_invalid", payload=payload, errors=tuple(errors))

    rubric = load_rubric("outcome", scene_id)
    label = payload.get("outcome_label")
    outcome_in_schema = str(payload.get("outcome_in_schema") or "").upper()
    if payload.get("outcome_readable") is False or outcome_in_schema in {"NOT_READABLE", "UNREADABLE"}:
        return ValidationResult(status="unreadable", payload=payload, canonical_label=None)
    if outcome_in_schema == "OUT_OF_SCHEMA":
        return ValidationResult(status="out_of_schema", payload=payload, canonical_label=None)
    if not isinstance(label, str) or not label:
        return ValidationResult(status="schema_invalid", payload=payload, errors=("outcome_label must be a string",))

    canonical_labels = set(str(item) for item in rubric.get("canonical_labels", []))
    aliases = {str(key): str(value) for key, value in (rubric.get("label_aliases") or {}).items()}
    canonical = aliases.get(label, label)
    if canonical not in canonical_labels:
        return ValidationResult(
            status="rubric_invalid_label",
            payload=payload,
            canonical_label=None,
            errors=(f"label {label!r} is not canonical for scene {scene_id}",),
        )
    normalized = dict(payload)
    normalized["outcome_label"] = canonical
    return ValidationResult(status="valid", payload=normalized, canonical_label=canonical)


def validate_trustworthiness_response(raw: str | dict[str, Any]) -> ValidationResult:
    parsed = parse_response(raw)
    if parsed.payload is None:
        return parsed
    payload = parsed.payload
    errors: list[str] = []
    scene_id = payload.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        errors.append("scene_id must be a non-empty string")
    status = payload.get("status")
    if not isinstance(status, str):
        errors.append("status must be a string")
    elif status not in TRUST_STATUSES:
        errors.append(f"status {status!r} is not a valid trust status")

    for block_name in TRUST_BLOCKS:
        block = payload.get(block_name)
        if not isinstance(block, dict):
            errors.append(f"{block_name} must be an object")
            continue
        if block.get("status") not in BLOCK_STATUSES:
            errors.append(f"{block_name}.status {block.get('status')!r} is not valid")
        if block_name == "physical_process":
            failure_phase = block.get("failure_phase")
            if failure_phase is not None and not isinstance(failure_phase, str):
                errors.append("physical_process.failure_phase must be a string when provided")
            elif block.get("status") == "PASS" and failure_phase not in {None, "NONE"}:
                errors.append("physical_process.failure_phase must be NONE when physical_process.status is PASS")

    action = payload.get("action_execution")
    if isinstance(action, dict):
        for key in ("target_hit", "object_acquired", "action_spec_followed"):
            if action.get(key) not in YNUN_VALUES:
                errors.append(f"action_execution.{key} {action.get(key)!r} is not valid")
    failure_axes = payload.get("failure_axes")
    if not isinstance(failure_axes, list):
        errors.append("failure_axes must be a list")
    elif any(not isinstance(item, str) for item in failure_axes):
        errors.append("failure_axes must contain only strings")

    if errors:
        return ValidationResult(status="schema_invalid", payload=payload, errors=tuple(errors))
    return ValidationResult(status="valid", payload=payload)


def validate_axis_payload(*, axis: str, payload: dict[str, Any], scene_id: str) -> list[str]:
    """Validate a parsed PAWEval axis payload for one expected scene."""

    if axis == "outcome_readout":
        result = validate_outcome_response(payload, scene_id=scene_id)
        if result.status in {"valid", "unreadable", "out_of_schema"}:
            return []
        return list(result.errors) or [result.status]
    if axis == "trustworthiness_audit":
        result = validate_trustworthiness_response(payload)
        if result.status == "valid" and payload.get("scene_id") != scene_id:
            return [
                f"scene_id {payload.get('scene_id')!r} does not match expected scene_id {scene_id!r}"
            ]
        if result.status == "valid":
            return []
        return list(result.errors) or [result.status]
    return [f"unknown_axis:{axis}"]
