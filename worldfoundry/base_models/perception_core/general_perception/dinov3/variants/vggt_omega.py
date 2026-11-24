# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software is distributed under the DINOv3 License Agreement.
"""VGGT-Omega bindings for DINOv3 query/key normalization and list-aware FFNs."""

from collections.abc import Callable
from functools import partial

from torch import nn

from ..layers.attention import SelfAttention as DinoSelfAttention
from ..layers.block import SelfAttentionBlock as DinoSelfAttentionBlock
from ..layers.ffn_layers import ListMlp


class SelfAttention(DinoSelfAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__(
            dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop, mask_k_bias, device, use_qk_norm=use_qk_norm
        )


class SelfAttentionBlock(DinoSelfAttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = ListMlp,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            ffn_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            partial(attn_class, use_qk_norm=use_qk_norm),
            ffn_layer,
            mask_k_bias,
            device,
        )
