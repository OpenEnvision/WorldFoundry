# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

"""Module for base_models -> perception_core -> general_perception -> dinov2 -> variants -> vipe_dav2 -> layers -> block.py functionality."""

import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import Tensor, nn

from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.xformers import index_select_cat

from .attention import Attention, MemEffAttention
from worldfoundry.core.nn.blocks.layers import LayerScale, Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import (
    add_residual,
    drop_add_residual_stochastic_depth,
    get_branges_scales,
)
from worldfoundry.core.nn.transformer.vit import PreNormTransformerBlock


fmha = None
XFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


class Block(PreNormTransformerBlock):
    """ViPE-DAV2 pre-norm transformer block."""

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
        )


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """Index-select, concatenate nested tensors, and return cached attention bias."""
    if fmha is None:
        raise NotImplementedError("Nested tensor attention requires xFormers, which is disabled in ViPE")

    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    """Stochastic depth for nested tensor lists (requires xFormers)."""
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    """Nested tensor block implementation."""

    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list

        def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

        def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        attn_bias, x = get_attn_bias_and_cat(x_list)
        x = x + attn_residual_func(x, attn_bias=attn_bias)
        x = x + ffn_residual_func(x)
        return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            raise NotImplementedError("Nested tensor attention requires xFormers, which is disabled in ViPE")
        else:
            raise AssertionError
