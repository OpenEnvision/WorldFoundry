# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from collections.abc import Callable

from timm.layers import DropPath as DropPath
from torch import Tensor, nn

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    Block as VGGTBlock,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    NestedTensorBlock as VGGTNestedTensorBlock,
)
from worldfoundry.core.nn.blocks.layers import Mlp as Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import add_residual as add_residual
from worldfoundry.core.nn.blocks.stochastic_depth import (
    drop_add_residual_stochastic_depth as drop_add_residual_stochastic_depth,
)
from worldfoundry.core.nn.blocks.stochastic_depth import (
    get_branges_scales as get_branges_scales,
)

from .attention import Attention

XFORMERS_AVAILABLE = False


class Block(VGGTBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
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
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
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
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, attn_mask=None, kv_cache=None) -> Tensor:
            if kv_cache is not None:
                x, kv_cache = self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask, kv_cache=kv_cache)
                return self.ls1(x), kv_cache
            elif attn_mask is not None:
                return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
            else:
                return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            assert attn_mask is None and kv_cache is None, (
                "attn_mask and kv_cache are not supported for drop_add_residual_stochastic_depth yet"
            )
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            if kv_cache is not None:
                delta_x, kv_cache = attn_residual_func(x, pos=pos, attn_mask=attn_mask, kv_cache=kv_cache)
            else:
                delta_x = attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + self.drop_path1(delta_x)
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            if kv_cache is not None:
                delta_x, kv_cache = attn_residual_func(x, pos=pos, attn_mask=attn_mask, kv_cache=kv_cache)
            else:
                delta_x = attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + delta_x
            x = x + ffn_residual_func(x)

        if kv_cache is not None:
            return x, kv_cache
        else:
            return x


class NestedTensorBlock(VGGTNestedTensorBlock, Block):
    """VGGT nested interface with the family block constructor and forward."""
