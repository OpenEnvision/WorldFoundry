# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Generate structured VLM answers from chronologically sampled video frames."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Literal

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from .checklist import CaseChecklist
from .protocol import LRHoldProtocol, PromptAnchors

DEFAULT_STRIDE = 8
DEFAULT_JPEG_QUALITY = 90


@dataclass(frozen=True)
class VLMConfig:
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "Qwen3.8-27B"
    max_tokens: int = 4096
    timeout_seconds: float = 600.0
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    sampling_stride_frames: int = DEFAULT_STRIDE
    backend: str = "local"
    api_key_env: str = "WORLDMEMBENCH_API_KEY"


@dataclass(frozen=True)
class QuestionAnswer:
    id: str
    answer: Literal["yes", "no"]
    evidence_image_numbers: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class EventStartedAnswer:
    answer: Literal["yes", "no"]
    evidence_image_numbers: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class VLMResponse:
    raw_response: str
    event_started: EventStartedAnswer
    answers: tuple[QuestionAnswer, ...]
    usage: dict = field(default_factory=dict)


class InvalidVLMResponse(EvaluationError):
    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def sample_indices(total_frames: int, *, stride: int = DEFAULT_STRIDE) -> list[int]:
    if total_frames <= 0:
        return []
    if stride <= 0:
        raise ValueError("sampling stride must be positive")
    indices = list(range(0, total_frames, stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def encode_jpegs(
    frames_rgb: list,
    *,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> list[bytes]:
    import cv2

    encoded: list[bytes] = []
    for frame in frames_rgb:
        ok, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise EvaluationError("failed to JPEG-encode a sampled frame")
        encoded.append(bytes(buffer))
    return encoded


def generate_answers(video_path, checklist, trajectory, config, *, reader, backend):
    """Stage one: sample the video and obtain answers in one VLM request."""
    if config.backend not in ("local", "api") or not config.base_url or not config.model:
        raise EvaluationError("VLM requires backend local/api, base_url and model")
    if config.max_tokens <= 0 or config.timeout_seconds <= 0 or not 1 <= config.jpeg_quality <= 100:
        raise EvaluationError("invalid VLM token, timeout or JPEG quality setting")
    info = reader.probe(video_path)
    indices = sample_indices(info.total_frames, stride=config.sampling_stride_frames)
    protocol = LRHoldProtocol.from_metadata(trajectory, actual_output_frames=info.total_frames)
    anchors = protocol.prompt_anchors(indices)
    images = encode_jpegs(
        [frame.rgb for frame in reader.read_indices(video_path, indices)], quality=config.jpeg_quality
    )
    initial = anchors.phase_ranges()["initial_hold"]
    response = backend.generate(
        prompt=build_prompt(checklist, anchors),
        jpeg_images=images,
        config=config,
        expected_ids=tuple(question.id for question in checklist.questions),
        event_image_numbers=tuple(range(initial[0], initial[1] + 1)),
    )
    return response, indices, protocol, anchors


def build_prompt(checklist: CaseChecklist, anchors: PromptAnchors) -> str:
    """Ask the VLM for visual evidence without disclosing expected answers."""

    questions = [
        {
            "id": question.id,
            "category": question.category,
            "evidence_window": question.evidence_window,
            "question": question.question,
        }
        for question in checklist.questions
    ]
    return "\n".join(
        [
            "You are judging one generated Dynamic Memory LR-hold video using eight case-specific questions.",
            "The attached images are sampled in chronological order.",
            "The target generation prompt and expected answers are deliberately hidden.",
            "Use only visible evidence; do not complete a plausible story from the question wording.",
            "Question wording describes the intended evaluation target, but it is not visual evidence.",
            "For a change claim, cite visible before-and-after images that show a concrete difference; otherwise answer no.",
            "If the required subject or object disappears, is replaced, or changes into a different scene, do not treat its absence as successful completion.",
            "In the reason, describe only concrete evidence visible in the cited images and do not invent an unseen state or event.",
            "The evaluator has assigned the images to the model's planned camera phases.",
            "Each range below is an inclusive pair of 1-based image numbers.",
            "Planned phase image ranges:",
            json.dumps(anchors.phase_ranges(), ensure_ascii=False, indent=2),
            "",
            "Useful comparison ranges:",
            json.dumps(anchors.compact_ranges(), ensure_ascii=False, indent=2),
            "",
            "Phase guidance:",
            "- returned_view means the returned_hold range, compared with the departure_baseline range when the question asks about change.",
            "- returned_hold_sequence means the progression inside the returned_hold range.",
            "- returned_and_later_visible_sequence means the returned_hold range through its final images.",
            "- final_stationary_view means the final_stationary range.",
            "- ordered_visible_sequence may use the full chronological image sequence.",
            "- For a comparison or temporal claim, use visible before/after evidence when available.",
            "- If an event never occurs, say that directly. Never invent a reset, reversal, worsening, disappearance, or change that is not visible.",
            "- Interpret extent words relative to the stated baseline, not by absolute impression.",
            "- Camera motion, framing, lighting, an action pose, or a replaced subject alone is not evidence of state progress.",
            "- If the exact state is not visibly established, answer no.",
            "- Cite only decisive 1-based image numbers.",
            "",
            "Event-start prerequisite:",
            "- First decide whether the central event described by the questions visibly starts during initial_hold, before the camera leaves the original view.",
            "- event_started is yes only when at least two initial_hold images show concrete temporal progress by the principal subject or object.",
            "- Mere pose, contact, camera motion, background motion, or a plausible intention is not an event start.",
            "- If no relevant action or state change is visibly underway by the end of initial_hold, event_started is no.",
            "- Even when event_started is no, answer all eight questions from visible evidence; the evaluator applies the zero-score gate.",
            "",
            "Questions (expected answers are intentionally hidden):",
            json.dumps(questions, ensure_ascii=False, indent=2),
            "",
            "Return one JSON object with exactly the keys event_started and answers.",
            "event_started must contain:",
            '- answer: "yes" or "no";',
            "- evidence_image_numbers: at least two decisive 1-based image numbers from initial_hold;",
            "- reason: one concise sentence describing the visible change, or the visible lack of change.",
            "answers must be a list of exactly eight objects in the same ID order.",
            "Each object must contain:",
            "- id: the question ID;",
            '- answer: "yes" or "no";',
            "- evidence_image_numbers: a short list of decisive 1-based image numbers;",
            "- reason: one concise sentence describing only the cited visible evidence.",
            "Return JSON only.",
        ]
    )


def parse_response(
    raw_text: str,
    expected_ids: tuple[str, ...],
    *,
    num_images: int,
    event_image_numbers: tuple[int, ...] | None = None,
) -> VLMResponse:
    text = raw_text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        # Accept only lossless repair of missing outer braces around the two
        # required top-level fields. Do not tolerate truncation or extra fields.
        try:
            wrapped = json.loads("{" + text + "}")
        except json.JSONDecodeError:
            raise InvalidVLMResponse("VLM response is not valid JSON") from error
        if not isinstance(wrapped, dict) or set(wrapped) != {
            "event_started",
            "answers",
        }:
            raise InvalidVLMResponse("VLM response is not valid JSON") from error
        value = wrapped
    if not isinstance(value, dict) or set(value) != {"event_started", "answers"}:
        raise InvalidVLMResponse("VLM response must contain exactly event_started and answers")
    event = _parse_event_started(
        value.get("event_started"),
        num_images=num_images,
        allowed_image_numbers=event_image_numbers,
    )
    rows = _answer_rows(value)
    if len(rows) != len(expected_ids):
        raise InvalidVLMResponse(f"expected {len(expected_ids)} answers, got {len(rows)}")
    answers: list[QuestionAnswer] = []
    for index, (answer, expected_id) in enumerate(zip(rows, expected_ids, strict=True)):
        if not isinstance(answer, dict):
            raise InvalidVLMResponse(f"answer {index} must be an object")
        if answer.get("id") != expected_id:
            raise InvalidVLMResponse(f"answer {index} ID must be {expected_id!r}")
        predicted = str(answer.get("answer", "")).strip().lower()
        if predicted not in {"yes", "no"}:
            raise InvalidVLMResponse(f"{expected_id}: answer must be yes or no")
        evidence = answer.get("evidence_image_numbers")
        if not isinstance(evidence, list) or not evidence:
            raise InvalidVLMResponse(f"{expected_id}: evidence_image_numbers must be a non-empty list")
        try:
            evidence_numbers = tuple(int(number) for number in evidence)
        except (TypeError, ValueError) as error:
            raise InvalidVLMResponse(f"{expected_id}: evidence image number must be an integer") from error
        if any(number < 1 or number > num_images for number in evidence_numbers):
            raise InvalidVLMResponse(f"{expected_id}: evidence image number is outside [1, {num_images}]")
        reason = str(answer.get("reason", "")).strip()
        if not reason:
            raise InvalidVLMResponse(f"{expected_id}: reason is required")
        answers.append(
            QuestionAnswer(
                id=expected_id,
                answer=predicted,  # type: ignore[arg-type]
                evidence_image_numbers=evidence_numbers,
                reason=reason,
            )
        )
    return VLMResponse(
        raw_response=raw_text,
        event_started=event,
        answers=tuple(answers),
    )


def _parse_event_started(
    value,
    *,
    num_images: int,
    allowed_image_numbers: tuple[int, ...] | None,
) -> EventStartedAnswer:
    if not isinstance(value, dict):
        raise InvalidVLMResponse("event_started must be an object")
    answer = str(value.get("answer", "")).strip().lower()
    if answer not in {"yes", "no"}:
        raise InvalidVLMResponse("event_started.answer must be yes or no")
    evidence = value.get("evidence_image_numbers")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise InvalidVLMResponse("event_started.evidence_image_numbers must contain at least two images")
    try:
        evidence_numbers = tuple(int(number) for number in evidence)
    except (TypeError, ValueError) as error:
        raise InvalidVLMResponse("event_started evidence image number must be an integer") from error
    if len(set(evidence_numbers)) < 2:
        raise InvalidVLMResponse("event_started evidence images must be distinct")
    if any(number < 1 or number > num_images for number in evidence_numbers):
        raise InvalidVLMResponse(f"event_started evidence image number is outside [1, {num_images}]")
    if allowed_image_numbers is not None:
        allowed = set(allowed_image_numbers)
        if any(number not in allowed for number in evidence_numbers):
            raise InvalidVLMResponse("event_started evidence images must come from initial_hold")
    reason = str(value.get("reason", "")).strip()
    if not reason:
        raise InvalidVLMResponse("event_started.reason is required")
    return EventStartedAnswer(
        answer=answer,  # type: ignore[arg-type]
        evidence_image_numbers=evidence_numbers,
        reason=reason,
    )


def _answer_rows(value) -> list:
    """Read the question answers from the structured VLM response."""

    rows = value.get("answers")
    if isinstance(rows, list):
        return rows
    raise InvalidVLMResponse("VLM response must contain answers")


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    path = urllib.parse.urlparse(base).path.rstrip("/")
    if path.endswith("/chat/completions"):
        return base
    if path.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def build_payload(
    config: VLMConfig,
    *,
    prompt: str,
    jpeg_images: list[bytes],
) -> dict:
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")},
        }
        for image in jpeg_images
    ]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }
    if config.backend == "api":
        payload.pop("chat_template_kwargs")
        payload["enable_thinking"] = False
    return payload


class HTTPVLMBackend:
    def generate(
        self,
        *,
        prompt: str,
        jpeg_images: list[bytes],
        config: VLMConfig,
        expected_ids: tuple[str, ...],
        event_image_numbers: tuple[int, ...],
    ) -> VLMResponse:
        headers = {"Content-Type": "application/json"}
        key = ""
        if config.backend == "api":
            key = os.environ.get(config.api_key_env, "")
            if not key:
                raise EvaluationError(f"set the API key environment variable {config.api_key_env}")
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(
            chat_completions_url(config.base_url),
            data=json.dumps(build_payload(config, prompt=prompt, jpeg_images=jpeg_images)).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            if key:
                detail = detail.replace(key, "[REDACTED]")
            raise EvaluationError(f"VLM HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise EvaluationError(f"VLM request failed: {error.reason}") from error
        try:
            raw_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise EvaluationError("VLM response has no chat content") from error
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise EvaluationError("VLM returned empty chat content")
        try:
            if body["choices"][0].get("finish_reason") == "length":
                raise InvalidVLMResponse("VLM output reached max_tokens")
            parsed = parse_response(
                raw_text,
                expected_ids,
                num_images=len(jpeg_images),
                event_image_numbers=event_image_numbers,
            )
            return replace(parsed, usage=body.get("usage") or {})
        except InvalidVLMResponse as error:
            raise InvalidVLMResponse(str(error), raw_response=raw_text) from error
