# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module for base_models -> three_dimensions -> general_3d -> lagernvs -> lagernvs_runtime -> models -> layers -> renderer_blocks.py functionality."""

import torch.nn as nn
from models.layers.attention import Attention
from worldfoundry.core.nn.blocks.layers import Mlp


#################################################################################
#                             Renderer Block Classes                            #
#################################################################################


class FullAttentionBlock(nn.Module):
    """
    A block with full self-attention (all tokens attend to all tokens).
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        """Init.

        Args:
            hidden_dim: The hidden dim.
            num_heads: The num heads.
            ln_bias: The ln bias.
            attn_qkv_bias: The attn qkv bias.
            attn_dropout: The attn dropout.
            attn_fc_bias: The attn fc bias.
            attn_fc_dropout: The attn fc dropout.
            mlp_ratio: The mlp ratio.
            mlp_bias: The mlp bias.
            mlp_dropout: The mlp dropout.
            use_qk_norm: The use qk norm.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            bias=mlp_bias,
            drop=mlp_dropout,
        )

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    A block with cross-attention from target tokens to conditioning tokens.
    Based on CDiT block from Navigation World Models https://arxiv.org/pdf/2412.03572.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        """Init.

        Args:
            hidden_dim: The hidden dim.
            num_heads: The num heads.
            ln_bias: The ln bias.
            attn_qkv_bias: The attn qkv bias.
            attn_dropout: The attn dropout.
            attn_fc_bias: The attn fc bias.
            attn_fc_dropout: The attn fc dropout.
            mlp_ratio: The mlp ratio.
            mlp_bias: The mlp bias.
            mlp_dropout: The mlp dropout.
            use_qk_norm: The use qk norm.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.self_attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.norm2_kv = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.cross_attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.norm_ffn = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            bias=mlp_bias,
            drop=mlp_dropout,
        )

    def forward(self, x, cond_tokens):
        """Forward.

        Args:
            x: The x.
            cond_tokens: The cond tokens.
        """
        # x: (B v_target) x P x C
        # cond_tokens: (B v_target) x (P v_input) x C
        assert (
            x.ndim == 3 and cond_tokens.ndim == 3
        ), f"Unexpected number of dimensions, {x.ndim}, {cond_tokens.ndim}"

        # self-attention
        y = self.norm1(x)
        y = self.self_attn(y, kv=None)
        x = x + y

        # cross-attention
        y = self.norm2(x)
        y_kv = self.norm2_kv(cond_tokens)
        y = self.cross_attn(y, kv=y_kv)
        x = x + y

        # feedforward
        y = self.norm_ffn(x)
        y = self.mlp(y)
        x = x + y
        return x


class BidirectionalCrossAttentionBlock(nn.Module):
    """
    A block with bidirectional cross-attention between target and conditioning tokens.
    Based on CDiT block from Navigation World Models https://arxiv.org/pdf/2412.03572.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        """Init.

        Args:
            hidden_dim: The hidden dim.
            num_heads: The num heads.
            ln_bias: The ln bias.
            attn_qkv_bias: The attn qkv bias.
            attn_dropout: The attn dropout.
            attn_fc_bias: The attn fc bias.
            attn_fc_dropout: The attn fc dropout.
            mlp_ratio: The mlp ratio.
            mlp_bias: The mlp bias.
            mlp_dropout: The mlp dropout.
            use_qk_norm: The use qk norm.
        """
        super().__init__()

        self.norm1_x = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.self_attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.norm2_x = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.cross_attn_x = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.norm3_x = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.mlp_x = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            bias=mlp_bias,
            drop=mlp_dropout,
        )

        self.norm1_rec = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.cross_attn_rec = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2_rec = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.mlp_rec = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            bias=mlp_bias,
            drop=mlp_dropout,
        )

    def forward(self, x, cond_tokens):
        """Forward.

        Args:
            x: The x.
            cond_tokens: The cond tokens.
        """
        # x: (B v_target) x P x C
        # cond_tokens: (B v_target) x (P v_input) x C
        assert (
            x.ndim == 3 and cond_tokens.ndim == 3
        ), f"Unexpected number of dimensions, {x.ndim}, {cond_tokens.ndim}"

        # self-attention
        x = x + self.self_attn(self.norm1_x(x), kv=None)

        # cross-attention
        x_norm = self.norm2_x(x)
        rec_norm = self.norm1_rec(cond_tokens)
        # usual cross-attention
        x = x + self.cross_attn_x(x_norm, kv=rec_norm)
        # reverse cross-attention
        cond_tokens = cond_tokens + self.cross_attn_rec(rec_norm, kv=x_norm)

        # feedforward
        x = x + self.mlp_x(self.norm3_x(x))
        cond_tokens = cond_tokens + self.mlp_rec(self.norm2_rec(cond_tokens))
        return x, cond_tokens
