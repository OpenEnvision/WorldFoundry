# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import os
from collections.abc import Callable

from timm.layers import DropPath as DropPath
from torch import Tensor, nn

from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.block import (
    get_attn_bias_and_cat as get_attn_bias_and_cat,
)
from worldfoundry.core.nn.blocks.layers import Mlp as Mlp

from .attention import Attention, MemEffAttention
from .layer_scale import LayerScale

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import (  # noqa: F401 - Import checks optional dependency availability.
            fmha,
            index_select_cat,
            scaled_index_add,
        )

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Block)")
    else:
        # warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class Block(nn.Module):
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
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        x: Tensor,
        pos=None,
        attn_mask=None,
        segment_lengths=None,
        use_cache=False,
        past_key_value=None,
        cache_budget=None,
        tokens_per_frame=None,
    ):
        current_kv = None
        scores = None

        def attn_residual_func(x: Tensor, pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths) -> Tensor:
            nonlocal current_kv, scores
            if use_cache:
                attn_out, current_kv, scores = self.attn(
                    self.norm1(x),
                    pos=pos,
                    use_cache=True,
                    past_key_value=past_key_value,
                    cache_budget=cache_budget,
                    tokens_per_frame=tokens_per_frame,
                )
                return self.ls1(attn_out)
            else:
                return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask, segment_lengths=segment_lengths)
        x = x + ffn_residual_func(x)
        return (x, current_kv, scores) if use_cache else x


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: list[Tensor]) -> list[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

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
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
