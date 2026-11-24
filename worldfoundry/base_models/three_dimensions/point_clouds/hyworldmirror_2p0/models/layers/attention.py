# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch
from torch import Tensor

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    Attention as VGGTAttention,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    MemEffAttention as VGGTMemEffAttention,
)

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    _USE_FLASH_ATTN_V3 = True
except ImportError:
    flash_attn_func_v3 = None
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_func as flash_attn_func_v2,
        )
    except ImportError:
        flash_attn_func_v2 = None
    _USE_FLASH_ATTN_V3 = False
from worldfoundry.core.attention import (
    scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention,
)

from ...comm.communication import _All2All
from ...comm.padding import depad_by_length, pad_by_length


class Attention(VGGTAttention):
    def _compute_qkv(self, x: Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        return q, k, v, B, N, C

    def _apply_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if (q.dtype == torch.bfloat16 or q.dtype == torch.float16) and (
            flash_attn_func_v3 is not None or flash_attn_func_v2 is not None
        ):
            q = q.transpose(1, 2) if q.is_contiguous() else q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2) if k.is_contiguous() else k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2) if v.is_contiguous() else v.transpose(1, 2).contiguous()
            if _USE_FLASH_ATTN_V3:
                x = flash_attn_func_v3(q, k, v)
            else:
                x = flash_attn_func_v2(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            # FlashAttention 3 may return ``(output, softmax_lse, ...)`` while
            # FlashAttention 2 returns the output tensor directly.
            if isinstance(x, tuple):
                if not x:
                    raise RuntimeError("FlashAttention returned an empty result tuple.")
                x = x[0]
            if not isinstance(x, Tensor):
                raise TypeError(f"FlashAttention returned {type(x)!r}, expected a tensor.")
            x = x.transpose(1, 2) if x.is_contiguous() else x.transpose(1, 2).contiguous()
        else:
            x = _worldfoundry_scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        return x

    def _project_output(self, x: Tensor, B: int, N: int, C: int) -> Tensor:
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, pos=None) -> Tensor:
        q, k, v, B, N, C = self._compute_qkv(x)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = self._apply_attention(q, k, v)
        return self._project_output(x, B, N, C)


class DistAttention(Attention):
    def forward(self, x: Tensor, pos=None, sp_size=1, sp_group=None, padding_tokens=0) -> Tensor:
        q, k, v, B, N, C = self._compute_qkv(x)

        if sp_size > 1:
            q = _All2All.apply(q, 1, 2, sp_group, False)
            k = _All2All.apply(k, 1, 2, sp_group, False)
            v = _All2All.apply(v, 1, 2, sp_group, False)
            q = depad_by_length(q, padding_tokens, 2)
            k = depad_by_length(k, padding_tokens, 2)
            v = depad_by_length(v, padding_tokens, 2)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = self._apply_attention(q, k, v)

        if sp_size > 1:
            x = pad_by_length(x, padding_tokens, 2, 0)
            x = _All2All.apply(x, 2, 1, sp_group, False)

        return self._project_output(x, B, N, C)


class MemEffAttention(VGGTMemEffAttention, Attention):
    """VGGT tensor-only interface with the family attention policy."""
