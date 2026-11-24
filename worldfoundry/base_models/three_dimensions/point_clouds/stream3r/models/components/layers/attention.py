# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch
import torch.nn.functional as F
from torch import Tensor

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    Attention as VGGTAttention,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    MemEffAttention as VGGTMemEffAttention,
)

XFORMERS_AVAILABLE = False


class Attention(VGGTAttention):
    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            kv_cache = [k, v]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            if attn_mask is not None:
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        if kv_cache is not None:
            return x, kv_cache

        return x


class MemEffAttention(VGGTMemEffAttention, Attention):
    """VGGT tensor-only interface with the family attention policy."""
