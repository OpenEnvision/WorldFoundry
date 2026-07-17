"""Provider-agnostic LLM client for prompt generation.

Supports OpenAI-compatible APIs and DashScope (Qwen). Requires optional
``wrbench[prompts]`` extra (httpx) and explicit provider configuration.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol


class LLMProvider(Protocol):
    def call_json(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]: ...


def _require_httpx():
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LLM prompt generation requires httpx. Install with: pip install 'wrbench[prompts]'"
        ) from exc


def _require_config_value(value: str | None, *, label: str) -> str:
    if value is not None and value.strip():
        return value.strip()
    raise RuntimeError(f"{label} required")


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError(f"LLM response is not a JSON object: {result!r}")
    return result


class OpenAICompatibleProvider:
    """Call any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        _require_httpx()
        import httpx

        self.api_key = _require_config_value(api_key, label="OpenAI-compatible LLM API key")
        self.base_url = _require_config_value(
            base_url,
            label="OpenAI-compatible LLM base URL",
        ).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def call_json(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }
        resp = self._client.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_json_response(content)


class DashScopeProvider:
    """Call DashScope compatible-mode chat completions (Qwen)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        _require_httpx()
        import httpx

        self.api_key = _require_config_value(api_key, label="DashScope API key")
        self.base_url = _require_config_value(
            base_url,
            label="DashScope compatible-mode base URL",
        ).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def call_json(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        resp = self._client.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_json_response(content)


def get_llm_provider(name: str | None = None, **kwargs: Any) -> LLMProvider:
    """Return an LLM provider by name (``openai``, ``dashscope``)."""
    provider = _require_config_value(name, label="LLM provider").lower()
    if provider in {"openai", "openai_compatible", "compatible"}:
        return OpenAICompatibleProvider(**kwargs)
    if provider in {"dashscope", "qwen"}:
        return DashScopeProvider(**kwargs)
    raise ValueError(f"Unknown LLM provider {provider!r}; expected openai or dashscope")


def call_llm_json(
    *,
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.2,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :func:`get_llm_provider`."""
    model_name = _require_config_value(model, label="LLM model")
    llm = get_llm_provider(provider, api_key=api_key, base_url=base_url)
    return llm.call_json(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model_name,
        temperature=temperature,
    )
