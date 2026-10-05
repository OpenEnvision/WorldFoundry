"""Reusable safety guardrail interfaces.

Exports resolve lazily (same pattern as ``core.model_loading.checkpoints``/``core.utils``)
so that importing the guardrail Protocols does not pull the video-io stack
(imageio/numpy) or, transitively, torch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy facade — Protocols stay importable without imageio / numpy / torch
# ──────────────────────────────────────────────────────────────────────────

_EXPORT_MODULES = {
    "ContentSafetyGuardrail": "worldfoundry.core.safety.guardrails",
    "GuardrailRunner": "worldfoundry.core.safety.guardrails",
    "PostprocessingGuardrail": "worldfoundry.core.safety.guardrails",
    "VideoData": "worldfoundry.core.safety.video_io",
    "get_video_filepaths": "worldfoundry.core.safety.video_io",
    "read_video": "worldfoundry.core.safety.video_io",
    "save_video": "worldfoundry.core.safety.video_io",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access and cache it."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports to completion tools without importing video I/O."""

    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
