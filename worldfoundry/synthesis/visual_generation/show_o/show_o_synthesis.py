from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import ShowORuntime


class ShowOSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the Show-O runtime."""

    RUNTIME_CLS = ShowORuntime
    MODEL_ID = ShowORuntime.MODEL_ID
    DISPLAY_NAME = ShowORuntime.DISPLAY_NAME


__all__ = ["ShowOSynthesis"]
