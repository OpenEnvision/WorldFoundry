"""Synthesis facade for the Open-MAGVIT2 runtime."""

from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import OpenMAGVIT2Runtime


class OpenMAGVIT2Synthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the Open-MAGVIT2 runtime."""

    RUNTIME_CLS = OpenMAGVIT2Runtime
    MODEL_ID = OpenMAGVIT2Runtime.MODEL_ID
    DISPLAY_NAME = OpenMAGVIT2Runtime.DISPLAY_NAME


__all__ = ["OpenMAGVIT2Synthesis"]
