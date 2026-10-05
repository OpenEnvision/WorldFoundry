"""MoVerse normalization, preserving its checkpoint's activation dtype."""

from torch import nn

from ...reference_21 import WanLayerNorm as _WanLayerNorm


class WanLayerNorm(_WanLayerNorm):
    """Use MoVerse's native-dtype LayerNorm for both causal streams."""

    def forward(self, x):
        return nn.LayerNorm.forward(self, x).type_as(x)
