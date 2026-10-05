"""Synthesis facade over the official AC3D runtime."""

from __future__ import annotations

from typing import Any

from ..runtime_facade import RuntimeAdapterSynthesis
from .runtime import AC3DRuntime


class AC3DSynthesis(RuntimeAdapterSynthesis):
    """Thin synthesis facade over the official AC3D runtime."""

    RUNTIME_CLS = AC3DRuntime
    MODEL_ID = AC3DRuntime.MODEL_ID
    DISPLAY_NAME = AC3DRuntime.DISPLAY_NAME

    def runtime_plan(self) -> dict[str, Any]:
        """Return the runtime's execution plan without running it."""
        return self.runtime.runtime_plan()


__all__ = ["AC3DSynthesis"]
