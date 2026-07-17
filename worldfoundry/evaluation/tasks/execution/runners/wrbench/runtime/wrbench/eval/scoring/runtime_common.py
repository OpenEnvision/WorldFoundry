"""Shared runtime helpers for local and DashScope evaluation runners."""

from __future__ import annotations

from typing import Any, Final


def require_section(mapping: dict[str, Any], key: str, scope: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing required mapping: {scope}.{key}")
    return value


def require_value(mapping: dict[str, Any], key: str, scope: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required value: {scope}.{key}")
    value = mapping[key]
    if value is None:
        raise ValueError(f"Required value is null: {scope}.{key}")
    return value


def require_text(mapping: dict[str, Any], key: str, scope: str) -> str:
    value = str(require_value(mapping, key, scope)).strip()
    if not value:
        raise ValueError(f"Required text is empty: {scope}.{key}")
    return value


def scoring_video_path(row: dict[str, Any], *, surface: str | None = None) -> str:
    scoring_surface = str(surface or row.get("scoring_video_surface") or "").strip()
    if scoring_surface in {"full_continuation", "source_generated_concat"}:
        for key in ("scoring_video_path", "original_concat_path", "path", "video_path"):
            value = row.get(key)
            if value:
                return str(value)
        return ""
    for key in ("eval_video_path", "path", "video_path"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


META_KEYS: Final[list[str]] = [
    "video_id",
    "prompt_id",
    "variant_id",
    "family_id",
    "reasoning_tier",
    "oov_gap",
    "event_delta",
    "divergence_id",
    "scenario_type",
    "visibility_gap_level",
    "protocol",
    "predictability_level",
    "event_count",
    "expected_state",
    "expected_visibility",
    "model",
    "camera_type",
    "world_state_prompt",
    "expected_behavior",
]
