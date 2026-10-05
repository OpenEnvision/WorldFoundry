"""OpenAI-compatible inference backend for Offscreen Evolution."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ..io import value_digest
from ..skills.skill_offscreen_evolution import BACKEND_OUTPUT_SCHEMA


BACKEND_ID = "openai_compatible.offscreen_evolution_vlm"

__all__ = ["BACKEND_ID", "OpenAICompatibleBackend"]


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        role = "developer" if message.get("role") == "system" else message.get("role")
        content = []
        for item in message.get("content") or []:
            if item.get("type") == "text":
                content.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                content.append(
                    {"type": "input_image", "image_url": item.get("image_url")}
                )
        converted.append({"role": role, "content": content})
    return converted


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = []
        for item in message.get("content") or []:
            if item.get("type") == "text":
                content.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item.get("image_url")},
                    }
                )
        converted.append({"role": message.get("role"), "content": content})
    return converted


def _response_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str) and content.strip():
            return content
    parts = []
    for output in response.get("output") or []:
        if not isinstance(output, Mapping):
            continue
        for item in output.get("content") or []:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError("VLM response contains no output text")
    return text


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise RuntimeError("VLM response is not a JSON object") from None
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError("VLM response is not a JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeError("VLM response JSON must be an object")
    return value


class OpenAICompatibleBackend:
    """Send prepared messages to a local or remote model service."""

    backend_id = BACKEND_ID
    execution_mode = "api"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        wire_api: str = "responses",
        timeout: float = 90.0,
        retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.wire_api = wire_api.lower()
        self.timeout = timeout
        self.retries = retries
        if not self.base_url or not self.model:
            raise ValueError("base_url and model are required")
        if self.wire_api not in {
            "responses",
            "response",
            "chat",
            "chat_completions",
            "chat/completions",
        }:
            raise ValueError(f"unsupported VLM wire API: {wire_api}")
        self.version = self.model
        if self.wire_api in {"responses", "response"}:
            endpoint = (
                self.base_url
                if self.base_url.endswith("/responses")
                else f"{self.base_url}/responses"
            )
        else:
            endpoint = (
                self.base_url
                if self.base_url.endswith("/chat/completions")
                else f"{self.base_url}/chat/completions"
            )
        self.config_digest = value_digest(
            {
                "backend_id": BACKEND_ID,
                "endpoint": endpoint,
                "model": self.model,
                "wire_api": self.wire_api,
                "temperature": 0,
                "structured_output": "json_object",
            }
        )

    def _targets_and_body(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[str], dict[str, Any]]:
        if self.wire_api in {"responses", "response"}:
            urls = (
                [self.base_url]
                if self.base_url.endswith("/responses")
                else [f"{self.base_url}/responses", f"{self.base_url}/v1/responses"]
            )
            body = {
                "model": self.model,
                "input": _responses_input(messages),
                "temperature": 0,
                "text": {"format": {"type": "json_object"}},
            }
        else:
            urls = (
                [self.base_url]
                if self.base_url.endswith("/chat/completions")
                else [
                    f"{self.base_url}/chat/completions",
                    f"{self.base_url}/v1/chat/completions",
                ]
            )
            body = {
                "model": self.model,
                "messages": _chat_messages(messages),
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        return urls, body

    def _post(self, url: str, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(response.status)
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read().decode("utf-8", errors="replace")
        value = json.loads(raw)
        return status, value if isinstance(value, dict) else {"response": value}

    def infer(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        urls, body = self._targets_and_body(messages)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        attempts = []
        response = None
        request_url = None
        last_error = None
        retryable = {408, 409, 425, 429, 500, 502, 503, 504}
        for url in urls:
            for attempt in range(1, max(1, self.retries + 1) + 1):
                attempt_started = time.monotonic()
                try:
                    status, data = self._post(url, body)
                    error = None
                except (
                    TimeoutError,
                    urllib.error.URLError,
                    json.JSONDecodeError,
                ) as exc:
                    status, data, error = None, {}, repr(exc)
                attempts.append(
                    {
                        "attempt": attempt,
                        "url": url,
                        "status_code": status,
                        "error": error,
                        "elapsed_seconds": round(time.monotonic() - attempt_started, 6),
                    }
                )
                if status is not None and status < 400:
                    response, request_url = data, url
                    break
                last_error = error or json.dumps(data, ensure_ascii=False)[:1000]
                if status is not None and status not in retryable:
                    break
                if attempt <= self.retries:
                    time.sleep(
                        min(0.5 * (2 ** (attempt - 1)), 8.0) + random.uniform(0.0, 0.25)
                    )
            if response is not None:
                break
            if attempts and attempts[-1]["status_code"] not in {404, 405}:
                break
        if response is None:
            raise RuntimeError(f"Offscreen VLM request failed: {last_error}")

        raw_text = _response_text(response)
        return {
            "schema_version": BACKEND_OUTPUT_SCHEMA,
            "parsed": _parse_json(raw_text),
            "raw_output_text": raw_text,
            "raw_response": response,
            "response_metadata": {
                "id": response.get("id"),
                "model": response.get("model"),
                "created": response.get("created"),
                "usage": response.get("usage"),
                "request_url": request_url,
                "request_started_at": started_at,
                "response_received_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "attempt_count": len(attempts),
                "attempts": attempts,
            },
        }
