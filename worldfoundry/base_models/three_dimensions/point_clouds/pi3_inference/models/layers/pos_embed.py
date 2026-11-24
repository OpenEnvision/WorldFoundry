"""Pi3 uses the shared axial 2D rotary embedding and patch position cache."""

from worldfoundry.core.attention.rotary.rope_2d import (
    CrocoRotaryPositionEmbedding2D as RoPE2D,
)
from worldfoundry.core.attention.rotary.rope_2d import PositionGetter

__all__ = ["PositionGetter", "RoPE2D"]
