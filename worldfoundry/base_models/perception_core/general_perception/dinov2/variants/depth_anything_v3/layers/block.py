# flake8: noqa: F821
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

"""Module for base_models -> perception_core -> general_perception -> dinov2 -> variants -> depth_anything_v3 -> layers -> block.py functionality."""

import logging
from typing import Callable

from torch import nn

from .attention import Attention
from worldfoundry.core.nn.blocks.layers import Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import drop_add_residual_stochastic_depth, get_branges_scales
from worldfoundry.core.nn.transformer.vit import RopePreNormTransformerBlock

logger = logging.getLogger("dinov2")
XFORMERS_AVAILABLE = True


class Block(RopePreNormTransformerBlock):
    """Depth Anything V3 pre-norm transformer block with RoPE-aware attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            norm_eps=ln_eps,
            attn_kwargs={"qk_norm": qk_norm, "rope": rope},
        )
