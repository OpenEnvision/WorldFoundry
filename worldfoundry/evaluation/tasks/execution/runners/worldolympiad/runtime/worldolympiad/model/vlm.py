from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Literal

import requests

from model.openrouter import (
    chat_openrouter_call,
    clean_json_content,
    get_openrouter_api_key,
    parse_json_content,
    text_openrouter_call,
    video_openrouter_call,
)
from model.qwenvl import (
    DEFAULT_QWEN_MODEL_NAME,
    chat_qwenvl_call,
    text_qwenvl_call,
    video_qwenvl_call,
)


VLMBackend = Literal["openrouter", "qwenvl", "qwenvl_server"]

DEFAULT_OPENROUTER_MODEL_NAME = "google/gemini-2.5-flash"
DEFAULT_VLM_BACKEND = "qwenvl_server"

_BACKEND_ALIASES = {
    "api": "openrouter",
    "openrouter": "openrouter",
    "remote": "openrouter",
    "cloud": "openrouter",
    "local": "qwenvl",
    "qwen": "qwenvl",
    "qwenvl": "qwenvl",
    "server": "qwenvl_server",
    "qwen_server": "qwenvl_server",
    "qwenvl_server": "qwenvl_server",
    "qwenvl-server": "qwenvl_server",
}


def resolve_vlm_backend(backend: str | None = None) -> VLMBackend:
    """
    Normalize backend aliases.

    Priority:
    1. explicit `backend`
    2. env `VLM_BACKEND`
    3. default `qwenvl_server`
    """
    raw_backend = (backend or os.getenv("VLM_BACKEND") or DEFAULT_VLM_BACKEND).strip().lower()
    normalized = _BACKEND_ALIASES.get(raw_backend)
    if normalized is None:
        supported = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"Unsupported VLM backend: {raw_backend}. Supported values: {supported}")
    return normalized  # type: ignore[return-value]


def default_model_name_for_backend(backend: str | None = None) -> str:
    normalized_backend = resolve_vlm_backend(backend)
    if normalized_backend in {"qwenvl", "qwenvl_server"}:
        local_path = os.getenv("QWENVL_MODEL_PATH", "").strip()
        if local_path:
            return local_path
        return os.getenv("QWENVL_MODEL_NAME", DEFAULT_QWEN_MODEL_NAME)
    return os.getenv("OPENROUTER_MODEL_NAME", DEFAULT_OPENROUTER_MODEL_NAME)


def resolve_model_name(model_name: str | None = None, backend: str | None = None) -> str:
    if model_name and model_name.strip():
        return model_name.strip()
    env_override = os.getenv("VLM_MODEL", "").strip()
    if env_override:
        return env_override
    return default_model_name_for_backend(backend)


def _get_text_callable(backend: str | None = None):
    normalized_backend = resolve_vlm_backend(backend)
    if normalized_backend == "qwenvl":
        return text_qwenvl_call
    if normalized_backend == "qwenvl_server":
        return text_qwenvl_server_call
    return text_openrouter_call


def _get_video_callable(backend: str | None = None):
    normalized_backend = resolve_vlm_backend(backend)
    if normalized_backend == "qwenvl":
        return video_qwenvl_call
    if normalized_backend == "qwenvl_server":
        return video_qwenvl_server_call
    return video_openrouter_call


def _get_chat_callable(backend: str | None = None):
    normalized_backend = resolve_vlm_backend(backend)
    if normalized_backend == "qwenvl":
        return chat_qwenvl_call
    if normalized_backend == "qwenvl_server":
        return chat_qwenvl_server_call
    return chat_openrouter_call


def qwenvl_server_url() -> str:
    base_url = os.getenv("QWENVL_SERVER_URL", "http://127.0.0.1:8008").strip().rstrip("/")
    return f"{base_url}/v1/chat/completions"


def chat_qwenvl_server_call(
    messages: List[Dict],
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
    timeout: int = 240,
    max_retries: int = 3,
) -> Dict:
    payload = {
        "model": model_name,
        "messages": messages,
        "timeout": timeout,
        "max_retries": 1,
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(qwenvl_server_url(), json=payload, timeout=timeout)
            try:
                response_json = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"QwenVL server response is not JSON. status={response.status_code}, "
                    f"body={response.text[:500]}"
                ) from exc
            if response.status_code >= 400:
                raise RuntimeError(f"QwenVL server HTTP {response.status_code}: {response_json}")
            if "choices" not in response_json or not response_json["choices"]:
                raise RuntimeError(f"QwenVL server response missing choices: {response_json}")
            return response_json
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
    raise RuntimeError(f"QwenVL server request failed after {max_retries} attempts: {last_error}") from last_error


def text_qwenvl_server_call(
    system_prompt: str,
    user_content: str,
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
) -> Dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return chat_qwenvl_server_call(messages=messages, model_name=model_name, timeout=180, max_retries=3)


def video_qwenvl_server_call(
    data_url: Any,
    system_prompt: str,
    user_content: str,
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
) -> Dict:
    if not isinstance(data_url, list):
        content: List[Any] = [{"type": "text", "text": user_content}, data_url]
    else:
        content = [{"type": "text", "text": user_content}]
        if data_url:
            content.append({"type": "text", "text": "These are the frames extracted from the video."})
            content.extend(data_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return chat_qwenvl_server_call(messages=messages, model_name=model_name, timeout=240, max_retries=3)


def text_vlm_call(
    system_prompt: str,
    user_content: str,
    model_name: str | None = None,
    backend: str | None = None,
) -> Dict:
    """
    Unified text-only VLM/LLM entrypoint.

    Example:
        text_vlm_call(..., backend="api")
        text_vlm_call(..., backend="local")
    """
    resolved_backend = resolve_vlm_backend(backend)
    resolved_model_name = resolve_model_name(model_name, resolved_backend)
    text_callable = _get_text_callable(resolved_backend)
    return text_callable(
        system_prompt=system_prompt,
        user_content=user_content,
        model_name=resolved_model_name,
    )


def video_vlm_call(
    data_url: Any,
    system_prompt: str,
    user_content: str,
    model_name: str | None = None,
    backend: str | None = None,
) -> Dict:
    """
    Unified multimodal entrypoint for single-video or frame-list inputs.
    """
    resolved_backend = resolve_vlm_backend(backend)
    resolved_model_name = resolve_model_name(model_name, resolved_backend)
    video_callable = _get_video_callable(resolved_backend)
    return video_callable(
        data_url=data_url,
        system_prompt=system_prompt,
        user_content=user_content,
        model_name=resolved_model_name,
    )


def chat_vlm_call(
    messages: List[Dict],
    model_name: str | None = None,
    backend: str | None = None,
    timeout: int = 240,
    max_retries: int = 3,
) -> Dict:
    """
    Unified generic chat entrypoint.
    """
    resolved_backend = resolve_vlm_backend(backend)
    resolved_model_name = resolve_model_name(model_name, resolved_backend)
    chat_callable = _get_chat_callable(resolved_backend)
    return chat_callable(
        messages=messages,
        model_name=resolved_model_name,
        timeout=timeout,
        max_retries=max_retries,
    )


__all__ = [
    "VLMBackend",
    "DEFAULT_OPENROUTER_MODEL_NAME",
    "DEFAULT_QWEN_MODEL_NAME",
    "DEFAULT_VLM_BACKEND",
    "clean_json_content",
    "parse_json_content",
    "get_openrouter_api_key",
    "resolve_vlm_backend",
    "default_model_name_for_backend",
    "resolve_model_name",
    "text_openrouter_call",
    "video_openrouter_call",
    "chat_openrouter_call",
    "text_qwenvl_call",
    "video_qwenvl_call",
    "chat_qwenvl_call",
    "qwenvl_server_url",
    "text_qwenvl_server_call",
    "video_qwenvl_server_call",
    "chat_qwenvl_server_call",
    "text_vlm_call",
    "video_vlm_call",
    "chat_vlm_call",
]
