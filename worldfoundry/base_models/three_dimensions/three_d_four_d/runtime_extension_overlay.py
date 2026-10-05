"""Expose persistently built runtime extensions without modifying site-packages."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]


def add_runtime_extension_overlay(
    name: str,
    *,
    environment_variable: str | None = None,
) -> Path | None:
    configured = os.environ.get(environment_variable) if environment_variable else None
    overlay = (
        Path(configured).expanduser().resolve()
        if configured
        else _REPO_ROOT / "artifacts" / "runtime_extensions" / name
    )
    if not overlay.is_dir():
        return None
    overlay_text = str(overlay)
    if overlay_text not in sys.path:
        sys.path.insert(0, overlay_text)
    return overlay

