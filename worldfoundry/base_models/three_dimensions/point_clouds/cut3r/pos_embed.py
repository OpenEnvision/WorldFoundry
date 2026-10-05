"""CUT3R positional-embedding adapters."""

from worldfoundry.base_models.perception_core.general_perception.uniception.uniception.models.libs.croco.pos_embed import (
    get_2d_sincos_pos_embed,
)
from worldfoundry.core.attention.rotary.rope_2d import RotaryPositionEmbedding2D


class RoPE2D(RotaryPositionEmbedding2D):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__(frequency=freq, scaling_factor=F0)


__all__ = ["RoPE2D", "get_2d_sincos_pos_embed"]
