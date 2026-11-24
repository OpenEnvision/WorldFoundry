# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from torch import Tensor

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    Attention as VGGTAttention,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    MemEffAttention as VGGTMemEffAttention,
)
from worldfoundry.core.attention import (
    scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention,
)

XFORMERS_AVAILABLE = False


class Attention(VGGTAttention):
    def _apply_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return _worldfoundry_scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)


class MemEffAttention(VGGTMemEffAttention, Attention):
    """VGGT tensor-only interface with the family attention policy."""
