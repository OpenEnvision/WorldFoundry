"""Scene (first-frame / T2I) prompt generation."""

from __future__ import annotations

import json
from typing import Any

from wrbench.datasets import prompt_templates_dir


def load_t2i_scene_system_prompt() -> str:
    path = prompt_templates_dir() / "t2i_scene_system_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


def build_scene_user_message(family: dict[str, Any]) -> str:
    """Build the user message payload for T2I scene generation."""
    payload = {
        "family_id": family["family_id"],
        "scene": family["scene"],
        "primary_object": family["primary_object"],
        "supporting_objects": family["supporting_objects"],
        "first_frame_guidance": family["first_frame_guidance"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def generate_t2i_scene(
    family: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    llm_call: Any | None = None,
) -> str:
    """Generate ``t2i_scene`` caption for a family record via LLM.

    Pass ``llm_call`` as a callable(system_prompt, user_message) -> dict with
    key ``t2i_scene`` for testing without network.
    """
    system_prompt = load_t2i_scene_system_prompt()
    user_message = build_scene_user_message(family)

    if llm_call is not None:
        result = llm_call(system_prompt, user_message)
    else:
        from wrbench.prompts.llm import call_llm_json

        result = call_llm_json(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            temperature=temperature,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )

    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        text = result.get("t2i_scene") or result.get("caption") or result.get("text")
        if text:
            return str(text).strip()
    raise ValueError(f"LLM response missing t2i_scene field: {result!r}")


def enrich_family_with_t2i_scene(
    family: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    llm_call: Any | None = None,
) -> dict[str, Any]:
    """Return a copy of *family* with ``t2i_scene`` populated."""
    out = dict(family)
    out["t2i_scene"] = generate_t2i_scene(
        family,
        model=model,
        temperature=temperature,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        llm_call=llm_call,
    )
    return out
