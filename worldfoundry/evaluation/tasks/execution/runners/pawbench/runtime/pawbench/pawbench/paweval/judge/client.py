"""Judge client protocol and provider adapters."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from .requests import JudgeRequest
from .responses import JudgeResponse, parse_judge_response

ResponseFormatMode = Literal["json_object", "none"]
_SUCCESS_RESPONSE_MAX_BYTES = 1024 * 1024
_ERROR_RESPONSE_MAX_BYTES = 64 * 1024


class JudgeClient(Protocol):
    def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Return one judge response for a request."""


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout: int = 120
    max_tokens: int = 2048
    response_format_mode: ResponseFormatMode = "json_object"
    response_format_fallback: bool = True

    def credential_status(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": _redacted_url(self.base_url),
            "api_key_env": self.api_key_env,
            "has_api_key": bool(os.environ.get(self.api_key_env)),
        }


def validate_provider_readiness(config: ProviderConfig) -> dict[str, Any]:
    """Validate adapter-level provider configuration without sending a request."""

    provider = config.credential_status()
    blocked: list[dict[str, Any]] = []
    if not provider["has_api_key"]:
        blocked.append({"reason": "missing_credentials", "api_key_env": config.api_key_env})
    endpoint_status = "pass"
    try:
        _completion_url(config.base_url)
    except ValueError as exc:
        endpoint_status = "fail"
        blocked.append({"reason": "invalid_endpoint", "detail": str(exc)})
    if endpoint_status != "pass":
        provider = {**provider, "base_url": "<redacted-provider-url>"}
    return {
        "status": "pass" if not blocked else "fail",
        "provider": provider,
        "endpoint_status": endpoint_status,
        "blocked": blocked,
    }


@dataclass
class StaticJudgeClient:
    responses_by_axis: dict[str, str]
    calls: list[JudgeRequest] | None = None

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        if self.calls is not None:
            self.calls.append(request)
        if request.axis not in self.responses_by_axis:
            return JudgeResponse(raw_text="", parsed_json=None, status="missing_response", error=request.axis)
        return parse_judge_response(self.responses_by_axis[request.axis], axis=request.axis, scene_id=request.scene_id)


@dataclass
class OpenAICompatibleJudgeClient:
    config: ProviderConfig

    def complete(self, request: JudgeRequest) -> JudgeResponse:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="blocked_missing_credentials",
                error=f"missing credentials in env var {self.config.api_key_env}",
                transport_status="missing_credentials",
                schema_validation_status="missing_payload",
            )
        use_response_format = self.config.response_format_mode == "json_object"
        response = self._complete_once(request, api_key=api_key, use_response_format=use_response_format)
        if (
            response.transport_status == "response_format_rejected"
            and use_response_format
            and self.config.response_format_fallback
        ):
            return self._complete_once(request, api_key=api_key, use_response_format=False)
        return response

    def _complete_once(self, request: JudgeRequest, *, api_key: str, use_response_format: bool) -> JudgeResponse:
        body: dict[str, Any] = {
            "model": request.model or self.config.model,
            "messages": request.request_payload.get("messages")
            or [
                {
                    "role": "system",
                    "content": "You are a strict PAWEval JSON judge. Return one valid JSON object and no prose.",
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
        }
        if use_response_format:
            body["response_format"] = {"type": "json_object"}
        try:
            completion_url = _completion_url(self.config.base_url)
        except ValueError as exc:
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="transport_failed",
                error=str(exc),
                transport_status="invalid_base_url",
                schema_validation_status="missing_payload",
                response_format_used=use_response_format,
            )
        http_request = urllib.request.Request(
            completion_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.config.timeout) as response:
                raw_body = _read_bounded(response, max_bytes=_SUCCESS_RESPONSE_MAX_BYTES)
                if raw_body is None:
                    return _response_too_large(use_response_format=use_response_format)
                raw_response = json.loads(raw_body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_stream = exc.fp if exc.fp is not None else exc
            try:
                error_body = _read_bounded(error_stream, max_bytes=_ERROR_RESPONSE_MAX_BYTES)
                if error_body is None:
                    return _response_too_large(use_response_format=use_response_format)
                detail = error_body.decode("utf-8", errors="replace")
                status = _transport_status_from_http(exc.code, detail)
                return JudgeResponse(
                    raw_text="",
                    parsed_json=None,
                    status="transport_failed",
                    error=f"HTTP {exc.code}",
                    transport_status=status,
                    schema_validation_status="missing_payload",
                    response_format_used=use_response_format,
                )
            finally:
                if error_stream is not exc:
                    error_stream.close()
                exc.close()
        except TimeoutError as exc:
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="transport_failed",
                error=repr(exc),
                transport_status="timeout",
                schema_validation_status="missing_payload",
                response_format_used=use_response_format,
            )
        except urllib.error.URLError as exc:
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="transport_failed",
                error=repr(exc),
                transport_status="url_error",
                schema_validation_status="missing_payload",
                response_format_used=use_response_format,
            )
        except (OSError, json.JSONDecodeError) as exc:
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="transport_failed",
                error=repr(exc),
                transport_status="client_error",
                schema_validation_status="missing_payload",
                response_format_used=use_response_format,
            )
        if not isinstance(raw_response, dict):
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="transport_failed",
                error="provider_response_not_object",
                transport_status="client_error",
                schema_validation_status="missing_payload",
                response_format_used=use_response_format,
            )
        raw_text = _response_content(raw_response)
        if raw_text is None:
            return JudgeResponse(
                raw_text="",
                parsed_json=None,
                status="transport_failed",
                error="provider_response_malformed",
                transport_status="client_error",
                schema_validation_status="missing_payload",
                response_format_used=use_response_format,
            )
        parsed = parse_judge_response(raw_text, axis=request.axis, scene_id=request.scene_id)
        return JudgeResponse(
            raw_text=parsed.raw_text,
            parsed_json=parsed.parsed_json,
            status=parsed.status,
            error=parsed.error,
            extraction_status=parsed.extraction_status,
            schema_validation_status=parsed.schema_validation_status,
            validation_errors=parsed.validation_errors,
            transport_status="ok",
            raw_response=_scrub_provider_response(raw_response),
            response_format_used=use_response_format,
        )


def _read_bounded(stream: Any, *, max_bytes: int) -> bytes | None:
    chunks: list[bytes] = []
    total = 0
    while total <= max_bytes:
        chunk = stream.read(max_bytes + 1 - total)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
    return None


def _response_content(raw_response: Mapping[str, Any]) -> str | None:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _response_too_large(*, use_response_format: bool) -> JudgeResponse:
    return JudgeResponse(
        raw_text="",
        parsed_json=None,
        status="transport_failed",
        error="response_too_large",
        transport_status="response_too_large",
        schema_validation_status="missing_payload",
        response_format_used=use_response_format,
    )


def _transport_status_from_http(status_code: int, detail: str) -> str:
    lowered = detail.lower()
    if status_code == 429:
        return "rate_limited"
    if 300 <= status_code < 400:
        return "redirect_rejected"
    if status_code >= 500:
        return "server_error"
    if "response_format" in lowered or "json_object" in lowered:
        return "response_format_rejected"
    return "http_error"


def _completion_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("PAWEval judge base_url must use http or https")
    if not parsed.hostname:
        raise ValueError("PAWEval judge base_url must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("PAWEval judge base_url must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("PAWEval judge base_url must not include query or fragment")
    base_path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, f"{base_path}/chat/completions", "", ""))


def _redacted_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return "<invalid>"
    hostname = parsed.hostname or "<invalid>"
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parsed.port
    except ValueError:
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _scrub_provider_response(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_provider_response(item) for key, item in value.items() if key != "reasoning_content"}
    if isinstance(value, list):
        return [_scrub_provider_response(item) for item in value]
    return value
