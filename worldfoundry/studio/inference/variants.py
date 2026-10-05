from __future__ import annotations

import re

from worldfoundry.runtime.inference_catalog import (
    LINGBOT_VARIANT_BASE_ACT_PREVIEW,
    LINGBOT_VARIANT_BASE_CAM,
    LINGBOT_VARIANT_FAST,
    LINGBOT_WORLD_MODEL_ID,
)
from worldfoundry.studio.inference.catalog import CatalogEntry

LINGBOT_VARIANT_BASE_ACT = LINGBOT_VARIANT_BASE_ACT_PREVIEW


def normalize_cli_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def resolve_lingbot_variant_id(raw_variant: str | None) -> str | None:
    if raw_variant is None or not raw_variant.strip():
        return None
    normalized = normalize_cli_token(raw_variant)
    if normalized in {
        "default",
        "auto",
        "fast",
        "realtime",
        "fastrealtime",
        "lingbotfast",
        "lingbotfastrealtime",
        "lingbotworldfast",
    }:
        return LINGBOT_VARIANT_FAST
    if normalized in {"basecamera", "basecam", "camera", "cam", "lingbotbasecamera", "lingbotworldbasecam"}:
        return LINGBOT_VARIANT_BASE_CAM
    if normalized in {
        "baseact",
        "baseaction",
        "baseactpreview",
        "act",
        "action",
        "act2cam",
        "lingbotbaseact",
        "lingbotbaseaction",
        "lingbotworldbaseactpreview",
    }:
        return LINGBOT_VARIANT_BASE_ACT
    return None


def resolve_cli_variant_id(entry: CatalogEntry, raw_variant: str | None) -> str | None:
    if entry.model_id != LINGBOT_WORLD_MODEL_ID:
        if raw_variant and raw_variant.strip():
            raise ValueError(f"{entry.display_name} does not expose CLI variants.")
        return None
    resolved = resolve_lingbot_variant_id(raw_variant)
    if resolved is None and raw_variant and raw_variant.strip():
        raise ValueError(f"Unknown variant `{raw_variant}` for {entry.display_name}.")
    return resolved
