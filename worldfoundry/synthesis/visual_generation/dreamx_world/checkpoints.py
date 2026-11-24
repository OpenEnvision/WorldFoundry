"""Checkpoint resolution shared by DreamX-World inference sessions."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path


def enforce_offline_model_loading() -> None:
    """Reject network-enabled DreamX inference before any model loader runs."""

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
        value = os.environ.get(name)
        if value is not None and value.strip().lower() not in {"1", "true", "yes", "on"}:
            raise ValueError(f"DreamX-World inference requires offline model loading; {name}={value!r} is not allowed.")
        os.environ[name] = "1"


def resolve_checkpoint(
    source: str | Path | None,
    *,
    default_name: str,
    required: Sequence[str],
    label: str,
) -> Path:
    checkpoint = Path(source or checkpoint_root_path(default_name)).expanduser().resolve()
    missing = [relative for relative in required if not (checkpoint / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} checkpoint is incomplete at {checkpoint}; missing: {', '.join(missing)}")
    return checkpoint


__all__ = ["enforce_offline_model_loading", "resolve_checkpoint"]
