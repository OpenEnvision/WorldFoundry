"""PAWEval judgment flow for the two rubric axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

import yaml

from .adapter import MediaTransportError, build_request_payload
from .evidence.package import EvidencePackage
from .judge.client import ProviderConfig, validate_provider_readiness
from .judge.requests import AxisName, JudgeRequest, RowIdentity
from .judge.responses import JudgeResponse
from .judge.retry import retry_call
from .rubrics.loader import load_rubric
from .rubrics.validate import validate_axis_payload

JudgmentStatus = Literal["outcome", "null_observation", "infrastructure_failure"]
AXES: tuple[AxisName, ...] = ("outcome_readout", "trustworthiness_audit")
TRACKS = frozenset({"calibration", "coverage"})

OUTCOME_PROMPT = """# PAWEval Outcome Readout

Scene: `{scene_id}`
Sample: `{sample_id}`

Use the provided evidence and scene rubric to return one outcome label.
Do not judge trustworthiness in this axis.
Return a single JSON object only. Do not use Markdown fences, explanations,
or any text before or after the JSON object. The response must begin with `{{`
and end with `}}`.

Return JSON with this exact shape and preserve the provided `scene_id`:

{{
  "scene_id": "{scene_id}",
  "outcome_readable": true,
  "outcome_in_schema": "IN_SCHEMA",
  "outcome_label": "<one rubric label or null>",
  "observed_result": "<short evidence-grounded observation>",
  "failure_axes": []
}}

If the result is not readable, set `outcome_readable` to false, set
`outcome_in_schema` to "NOT_READABLE", set `outcome_label` to null, and include
the reason in `failure_axes`. If the result is readable but outside the rubric,
set `outcome_in_schema` to "OUT_OF_SCHEMA" and `outcome_label` to null.

Evidence:
{evidence_summary}

Rubric:
{rubric_text}
"""

TRUSTWORTHINESS_PROMPT = """# PAWEval Trustworthiness Audit

Scene: `{scene_id}`
Sample: `{sample_id}`

Audit whether the evidence supports strict reporting. Do not choose the outcome
label in this axis.
Return a single JSON object only. Do not use Markdown fences, explanations,
or any text before or after the JSON object.

The top-level `status` field must be exactly one of:
`TRUSTED`, `QUESTIONABLE`, `UNTRUSTED`, `UNCERTAIN`.

Do not use `PASS` or `FAIL` as the top-level `status`. `PASS`, `FAIL`,
`UNCLEAR`, and `NOT_APPLICABLE` are allowed only inside nested audit blocks such
as `scene_grounding.status` and `action_execution.status`.

Critical schema rule: the top-level object is the final trust judgment, not a
nested audit block. A response like `{{"status": "FAIL", "notes": "..."}}` or
`{{"status": "PASS", "notes": "..."}}` is invalid. If all nested audit blocks
pass, set the top-level `status` to `TRUSTED`. If any nested audit block fails,
set the top-level `status` to `QUESTIONABLE` or `UNTRUSTED`. If evidence is not
decidable, set the top-level `status` to `UNCERTAIN`.

The first two fields of the returned JSON object must be exactly:

{{
  "scene_id": "{scene_id}",
  "status": "TRUSTED"
}}

Replace `TRUSTED` with `QUESTIONABLE`, `UNTRUSTED`, or `UNCERTAIN` only when
the audit judgment requires it. Never replace it with `PASS` or `FAIL`.

Before returning, verify that the top-level `scene_id` is the non-empty string
`{scene_id}` and that the top-level `status` is one of the four official trust
labels above. If any nested audit block is `FAIL`, choose `QUESTIONABLE` or
`UNTRUSTED` for the top-level `status`; never copy `FAIL` to the top level.

Return JSON with this exact shape and preserve the provided `scene_id`:

{{
  "scene_id": "{scene_id}",
  "status": "TRUSTED",
  "scene_grounding": {{"status": "PASS", "notes": ""}},
  "action_execution": {{
    "status": "PASS",
    "target_hit": "YES",
    "object_acquired": "YES",
    "action_spec_followed": "YES",
    "notes": ""
  }},
  "object_continuity": {{"status": "PASS", "notes": ""}},
  "physical_process": {{"status": "PASS", "notes": ""}},
  "failure_axes": []
}}

For `action_execution`, always include `target_hit`, `object_acquired`, and
`action_spec_followed`. Each must be exactly one of `YES`, `NO`, `UNCLEAR`, or
`NOT_APPLICABLE`.

Use:
- `TRUSTED` only when all required evidence supports the report.
- `QUESTIONABLE` when evidence is incomplete or ambiguous but not clearly false.
- `UNTRUSTED` when evidence contradicts the report.
- `UNCERTAIN` when the video/evidence cannot support a reliable audit.

Name concrete failure axes in `failure_axes` rather than adding prose outside
the JSON object. For this automated runner, keep every `notes` field as the
empty string `""`. Do not write prose, quotes, markdown, or natural-language
explanations inside `notes`. Put the audit decision in the enum fields and
`failure_axes` only.

Evidence:
{evidence_summary}

Rubric:
{rubric_text}
"""


class JudgeAdapter(Protocol):
    def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Return one response for a PAWEval axis request."""


class JudgmentPreflightError(ValueError):
    """A global condition prevented any PAWEval provider call."""


@dataclass(frozen=True)
class JudgmentConfig:
    provider_config: ProviderConfig | None = None
    attempts: int = 2
    retry_sleep_seconds: float = 0.0
    retry_backoff_factor: float = 2.0
    retry_max_sleep_seconds: float = 30.0


@dataclass(frozen=True)
class JudgmentSample:
    package: EvidencePackage
    track: Literal["calibration", "coverage"]
    row_identity: RowIdentity


@dataclass(frozen=True)
class Judgment:
    row_identity: RowIdentity
    track: Literal["calibration", "coverage"]
    status: JudgmentStatus
    outcome_label: str | None
    outcome_readout: Mapping[str, Any] | None
    trustworthiness_audit: Mapping[str, Any] | None
    failure_code: str | None = None

    def normalized_row(self) -> dict[str, Any]:
        """Return one metric-ready row without performing metric aggregation."""

        row = {
            "sample_id": _metric_sample_id(self.row_identity),
            "source_sample_id": self.row_identity.sample_id,
            "scene_id": self.row_identity.scene_id,
            "track": self.track,
            "model_or_lane": self.row_identity.model_or_lane,
            "repeat_index": self.row_identity.repeat_index,
            "observation": self.status,
        }
        if self.status == "outcome":
            if not self.outcome_label:
                raise ValueError("outcome judgment is missing outcome_label")
            return {
                **row,
                "outcome_label": self.outcome_label,
                "outcome_readout": self.outcome_readout,
                "trustworthiness_audit": self.trustworthiness_audit,
            }
        if self.status == "null_observation":
            return {
                **row,
                "outcome_readout": self.outcome_readout,
                "trustworthiness_audit": self.trustworthiness_audit,
            }
        if not self.failure_code:
            raise ValueError("infrastructure failure judgment is missing failure_code")
        return {**row, "failure_code": self.failure_code}


@dataclass(frozen=True)
class JudgmentBatch:
    judgments: tuple[Judgment, ...]

    def normalized_rows(self) -> list[dict[str, Any]]:
        """Return one normalized metric row for every judged identity."""

        return [item.normalized_row() for item in self.judgments]


def judge(
    *,
    samples: list[JudgmentSample],
    adapter: JudgeAdapter,
    config: JudgmentConfig | None = None,
) -> JudgmentBatch:
    """Judge every supplied identity without dropping local failures."""

    config = config or JudgmentConfig()
    _assert_unique_identities(samples)
    _validate_preflight(samples, adapter, config)
    judgments = [
        _judge_sample(sample, adapter, config)
        for sample in sorted(samples, key=lambda item: item.row_identity.sort_key())
    ]
    return JudgmentBatch(judgments=tuple(judgments))


def _assert_unique_identities(samples: list[JudgmentSample]) -> None:
    identities = [sample.row_identity.sort_key() for sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate PAWEval judgment identity")


def _metric_sample_id(identity: RowIdentity) -> str:
    """Make a source sample plus repeat a unique official-metric row identity."""

    return f"{identity.sample_id}::repeat={identity.repeat_index}"


def _validate_preflight(samples: list[JudgmentSample], adapter: JudgeAdapter, config: JudgmentConfig) -> None:
    if config.attempts < 1:
        raise JudgmentPreflightError("invalid_attempt_count")
    for sample in samples:
        if sample.track not in TRACKS:
            raise JudgmentPreflightError("invalid_track")
        identity = sample.row_identity
        if not all(isinstance(value, str) and value for value in (identity.sample_id, identity.scene_id, identity.model_or_lane)):
            raise JudgmentPreflightError("invalid_row_identity")
        if not isinstance(identity.repeat_index, int) or identity.repeat_index < 0:
            raise JudgmentPreflightError("invalid_row_identity")
        if identity.sample_id != sample.package.sample_id or identity.scene_id != sample.package.scene_id:
            raise JudgmentPreflightError("evidence_identity_mismatch")
    provider = config.provider_config or getattr(adapter, "config", None)
    if provider is None:
        return
    if not isinstance(provider, ProviderConfig):
        raise JudgmentPreflightError("invalid_provider_config")
    readiness = validate_provider_readiness(provider)
    if readiness["status"] != "pass":
        reasons = ",".join(str(item.get("reason") or "invalid_provider") for item in readiness["blocked"])
        raise JudgmentPreflightError(reasons)


def _judge_sample(sample: JudgmentSample, adapter: JudgeAdapter, config: JudgmentConfig) -> Judgment:
    if not sample.package.source_image.attached or not sample.package.frames:
        return Judgment(
            row_identity=sample.row_identity,
            track=sample.track,
            status="infrastructure_failure",
            outcome_label=None,
            outcome_readout=None,
            trustworthiness_audit=None,
            failure_code="evidence_unavailable",
        )

    payloads: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        try:
            request = _request_for(sample, axis)
        except MediaTransportError:
            return _media_transport_failure(sample)
        response = retry_call(
            lambda request=request: adapter.complete(request),
            attempts=config.attempts,
            retry_status=lambda item: item.retryable,
            sleep_seconds=config.retry_sleep_seconds,
            backoff_factor=config.retry_backoff_factor,
            max_sleep_seconds=config.retry_max_sleep_seconds,
        )
        payload = response.parsed_json
        if response.ok is not True or not isinstance(payload, dict):
            return _infrastructure_failure(sample, axis)
        errors = validate_axis_payload(axis=axis, payload=payload, scene_id=sample.row_identity.scene_id)
        if errors:
            return _infrastructure_failure(sample, axis)
        payloads[axis] = payload

    outcome = payloads["outcome_readout"]
    outcome_is_null = outcome.get("outcome_readable") is False or outcome.get("outcome_in_schema") in {
        "NOT_READABLE",
        "UNREADABLE",
        "OUT_OF_SCHEMA",
    }
    return Judgment(
        row_identity=sample.row_identity,
        track=sample.track,
        status="null_observation" if outcome_is_null else "outcome",
        outcome_label=None if outcome_is_null else str(outcome.get("outcome_label") or "") or None,
        outcome_readout=outcome,
        trustworthiness_audit=payloads["trustworthiness_audit"],
    )


def _infrastructure_failure(sample: JudgmentSample, axis: AxisName) -> Judgment:
    return Judgment(
        row_identity=sample.row_identity,
        track=sample.track,
        status="infrastructure_failure",
        outcome_label=None,
        outcome_readout=None,
        trustworthiness_audit=None,
        failure_code=f"malformed_{axis}",
    )


def _media_transport_failure(sample: JudgmentSample) -> Judgment:
    return Judgment(
        row_identity=sample.row_identity,
        track=sample.track,
        status="infrastructure_failure",
        outcome_label=None,
        outcome_readout=None,
        trustworthiness_audit=None,
        failure_code="media_transport_failed",
    )


def _request_for(sample: JudgmentSample, axis: AxisName) -> JudgeRequest:
    prompt = _render_axis_prompt(sample, axis)
    return JudgeRequest(
        row_identity=sample.row_identity,
        axis=axis,
        prompt=prompt,
        request_payload=build_request_payload(prompt=prompt, package=sample.package),
    )


def _render_axis_prompt(sample: JudgmentSample, axis: AxisName) -> str:
    rubric_axis = "outcome" if axis == "outcome_readout" else "trustworthiness"
    rubric = load_rubric(rubric_axis, sample.row_identity.scene_id)
    template = OUTCOME_PROMPT if axis == "outcome_readout" else TRUSTWORTHINESS_PROMPT
    return template.format(
        scene_id=sample.row_identity.scene_id,
        sample_id=sample.row_identity.sample_id,
        evidence_summary=_evidence_summary(sample),
        rubric_text=yaml.safe_dump(rubric, sort_keys=False, allow_unicode=False),
    )


def _evidence_summary(sample: JudgmentSample) -> str:
    package = sample.package
    lines = [
        f"sample_id: {package.sample_id}",
        f"scene_id: {package.scene_id}",
        f"source image attached: {package.source_image.attached}",
        "Evidence transport: source image identity plus sampled video frames.",
    ]
    for frame in package.frames:
        lines.append(f"- {frame.phase}: timestamp_s={frame.timestamp_s}")
    return "\n".join(lines)
