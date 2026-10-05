"""Shared LTX PixelNorm / RMSNorm used by video and audio VAEs.

Tiny subpackage so video (:mod:`~..video`) and audio
(:mod:`~..audio`) import the same checkpoint-compatible
normalization primitives.

:func:`~.normalization.build_normalization_layer` selects
:class:`~.normalization.PixelNorm` or RMSNorm from
:class:`~.normalization.NormType`.  No encode/decode logic lives
here.

This is shared LTX graph math, not a runner Protocol adapter.
"""

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.common.normalization import (
    NormType,
    PixelNorm,
    build_normalization_layer,
)

__all__ = [
    "NormType",
    "PixelNorm",
    "build_normalization_layer",
]
