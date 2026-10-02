"""Pi3 uses the shared axial 2D rotary embedding and patch position cache."""

from worldfoundry.core.attention.rotary.rope_2d import PositionGetter, RotaryPositionEmbedding2D


class RoPE2D(RotaryPositionEmbedding2D):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__(frequency=freq, scaling_factor=F0)


__all__ = ["RoPE2D", "PositionGetter"]
