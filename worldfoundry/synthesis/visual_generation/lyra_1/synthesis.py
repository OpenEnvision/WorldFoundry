"""Synthesis facade for the Lyra-1 runtime."""

from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import Lyra1Runtime


class Lyra1Synthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the Lyra-1 runtime."""

    RUNTIME_CLS = Lyra1Runtime
    MODEL_ID = Lyra1Runtime.MODEL_ID
    DISPLAY_NAME = Lyra1Runtime.DISPLAY_NAME
    BLOCKED_REASONS = Lyra1Runtime.BLOCKED_REASONS
    MULTI_TRAJECTORY_INDEX = Lyra1Runtime.MULTI_TRAJECTORY_INDEX


__all__ = ["Lyra1Synthesis"]
