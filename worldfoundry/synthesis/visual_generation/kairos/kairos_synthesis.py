"""Synthesis facade over the official Kairos Sensenova runtime."""

from __future__ import annotations

from typing import Any

from ..runtime_facade import RuntimeAdapterSynthesis
from .runtime import KairosRuntime


class KairosSynthesis(RuntimeAdapterSynthesis):
    """Thin synthesis facade over the official Kairos Sensenova runtime checkout."""

    RUNTIME_CLS = KairosRuntime
    MODEL_ID = KairosRuntime.MODEL_ID
    DISPLAY_NAME = KairosRuntime.DISPLAY_NAME

    def runtime_plan(self) -> dict[str, Any]:
        """Return the runtime's execution plan without running it."""
        return self.runtime.runtime_plan()


__all__ = ["KairosSynthesis"]
