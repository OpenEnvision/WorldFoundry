from __future__ import annotations

from worldfoundry.base_models.three_dimensions.point_clouds.pixelsplat import PixelSplatRuntime

from ..runtime_facade import RuntimeFacadeSynthesis


class PixelSplatSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the base-model pixelSplat runtime."""

    RUNTIME_CLS = PixelSplatRuntime
    MODEL_ID = PixelSplatRuntime.MODEL_ID
    DISPLAY_NAME = PixelSplatRuntime.DISPLAY_NAME


__all__ = ["PixelSplatSynthesis"]
