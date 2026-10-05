"""Judge response parsing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..rubrics.validate import validate_outcome_response, validate_trustworthiness_response
from .requests import AxisName


@dataclass(frozen=True)
class JudgeResponse:
    raw_text: str
    parsed_json: dict[str, Any] | None
    status: str
    error: str | None = None
    extraction_status: str | None = None
    schema_validation_status: str | None = None
    validation_errors: tuple[str, ...] = ()
    transport_status: str | None = None
    raw_response: dict[str, Any] | None = None
    response_format_used: bool | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"parsed", "extracted_json"} and self.schema_validation_status in {
            None,
            "not_validated",
            "valid",
            "unreadable",
            "out_of_schema",
        }

    @property
    def retryable(self) -> bool:
        return self.transport_status in {"timeout", "url_error", "response_format_rejected", "rate_limited", "server_error"}

    def to_record(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_json": self.parsed_json,
            "status": self.status,
            "error": self.error,
            "extraction_status": self.extraction_status or self.status,
            "schema_validation_status": self.schema_validation_status or "not_validated",
            "validation_errors": list(self.validation_errors),
            "transport_status": self.transport_status,
            "raw_response": self.raw_response,
            "response_format_used": self.response_format_used,
        }


def _json_candidates(raw_text: str) -> list[str]:
    candidates = [raw_text]
    lines = raw_text.splitlines()
    in_fence = False
    fence_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence and fence_lines:
                candidates.append("\n".join(fence_lines).strip())
                fence_lines = []
            in_fence = not in_fence
            continue
        if in_fence:
            fence_lines.append(line)
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(raw_text[index : index + end])
    first = raw_text.find("{")
    last = raw_text.rfind("}")
    if first != -1 and last > first:
        candidates.append(raw_text[first : last + 1])
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _validate_payload(axis: AxisName | None, payload: dict[str, Any], scene_id: str | None) -> tuple[str, tuple[str, ...]]:
    if axis == "outcome_readout":
        result = validate_outcome_response(payload, scene_id=scene_id or str(payload.get("scene_id") or ""))
    elif axis == "trustworthiness_audit":
        result = validate_trustworthiness_response(payload)
    else:
        return "not_validated", ()
    return result.status, tuple(result.errors)


def parse_judge_response(raw_text: str, *, axis: AxisName | None = None, scene_id: str | None = None) -> JudgeResponse:
    first_error: str | None = None
    first_policy_accepted: JudgeResponse | None = None
    first_schema_invalid: JudgeResponse | None = None
    for index, candidate in enumerate(_json_candidates(raw_text)):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            first_error = first_error or str(exc)
            continue
        if not isinstance(payload, dict):
            response = JudgeResponse(
                raw_text=raw_text,
                parsed_json=None,
                status="schema_invalid",
                error="expected JSON object",
                extraction_status="schema_invalid",
                schema_validation_status="missing_payload",
                validation_errors=("expected JSON object",),
            )
            first_schema_invalid = first_schema_invalid or response
            continue
        validation_status, validation_errors = _validate_payload(axis, payload, scene_id)
        status = "parsed" if index == 0 else "extracted_json"
        response = JudgeResponse(
            raw_text=raw_text,
            parsed_json=payload,
            status=status,
            error=first_error if status == "extracted_json" else None,
            extraction_status=status,
            schema_validation_status=validation_status,
            validation_errors=validation_errors,
        )
        if validation_status == "valid":
            return response
        if validation_status in {"not_validated", "unreadable", "out_of_schema"}:
            first_policy_accepted = first_policy_accepted or response
            continue
        first_schema_invalid = first_schema_invalid or response
    if first_policy_accepted is not None:
        return first_policy_accepted
    if first_schema_invalid is not None:
        return first_schema_invalid
    return JudgeResponse(
        raw_text=raw_text,
        parsed_json=None,
        status="malformed_json",
        error=first_error or "no JSON object found",
        extraction_status="malformed_json",
        schema_validation_status="missing_payload",
        validation_errors=("no parsed JSON object",),
    )
