from typing import Any, Dict, List
import os
import base64
import requests
import time
import json


OPENROUTER_API_KEY_ENV_NAMES = (
    "WORLDFOUNDRY_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "api_key",
)


def get_openrouter_api_key() -> str | None:
    """Resolve the OpenRouter key, retaining the legacy lowercase fallback."""

    return next((value for name in OPENROUTER_API_KEY_ENV_NAMES if (value := os.getenv(name))), None)


def _openrouter_headers() -> Dict[str, str]:
    api_key = get_openrouter_api_key()
    if not api_key:
        names = ", ".join(OPENROUTER_API_KEY_ENV_NAMES[:-1])
        raise RuntimeError(f"Missing OpenRouter API key; set {names}")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def clean_json_content(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def parse_json_content(resp: Dict) -> Any:
    """
    Parse the first choice message content as JSON after stripping fences.
    """
    if "choices" not in resp or not resp["choices"]:
        raise ValueError("VLM response missing choices")
    content = resp["choices"][0]["message"]["content"]
    content = clean_json_content(content)
    return json.loads(content)


def _post_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict,
    timeout: int = 180,
    max_retries: int = 3,
) -> Dict:
    """
    Call OpenRouter with simple retry on network / bad JSON / missing choices.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            try:
                resp_json = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    f"OpenRouter response not valid JSON. status={resp.status_code}, body={resp.text[:500]}"
                ) from exc

            if resp.status_code >= 400:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp_json}")
            if "choices" not in resp_json or not resp_json["choices"]:
                raise RuntimeError(f"OpenRouter response missing choices: {resp_json}")
            return resp_json
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {exc}") from exc

def text_openrouter_call(system_prompt: str, user_content: str, model_name: str) -> Dict:
    """Text-only OpenRouter helper (no video)."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    headers = _openrouter_headers()
    url = "https://openrouter.ai/api/v1/chat/completions"
    return _post_with_retry(url, headers, payload, timeout=180, max_retries=3)


def video_openrouter_call(data_url, system_prompt: str, user_content: str, model_name: str = "google/gemini-2.5-flash") -> Dict:
    """Helper function to encode video and call OpenRouter API."""
    if type(data_url) is not list:

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content,
                    },
                    data_url
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content,
                    }
                ]
            },{
                "role": "user",
                "content": [
                    "These are the frames extracted from the video"
                ]
            }
        ]

        for frame in data_url:
            messages[2]["content"].append(frame)

    
    payload = {
        "model": model_name,
        "messages": messages
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = _openrouter_headers()
    resp_json = _post_with_retry(url, headers, payload, timeout=180, max_retries=3)
    return resp_json


def chat_openrouter_call(
    messages: List[Dict],
    model_name: str,
    timeout: int = 240,
    max_retries: int = 3,
) -> Dict:
    """
    Generic chat call with retry (used for two-video compliance calls).
    """
    payload = {"model": model_name, "messages": messages}
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = _openrouter_headers()
    return _post_with_retry(url, headers, payload, timeout=timeout, max_retries=max_retries)
