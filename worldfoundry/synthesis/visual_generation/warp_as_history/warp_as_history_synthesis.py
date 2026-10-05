"""Synthesis facade for the Warp-as-History runtime."""

from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import WarpAsHistoryRuntime


class WarpAsHistorySynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the Warp-as-History runtime."""

    RUNTIME_CLS = WarpAsHistoryRuntime
    MODEL_ID = WarpAsHistoryRuntime.MODEL_ID
    DISPLAY_NAME = WarpAsHistoryRuntime.DISPLAY_NAME


__all__ = ["WarpAsHistorySynthesis"]
