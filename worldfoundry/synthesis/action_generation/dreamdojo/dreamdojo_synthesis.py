"""Synthesis facade for the DreamDojo runtime."""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.runtime_facade import RuntimeFacadeSynthesis

from .worldfoundry_runtime import DreamDojoRuntime


class DreamDojoSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the DreamDojo runtime."""

    RUNTIME_CLS = DreamDojoRuntime
    MODEL_ID = DreamDojoRuntime.MODEL_ID
    DISPLAY_NAME = DreamDojoRuntime.DISPLAY_NAME


__all__ = ["DreamDojoSynthesis"]
