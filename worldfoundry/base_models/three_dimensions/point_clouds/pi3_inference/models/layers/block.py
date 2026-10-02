# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable

from torch import Tensor, nn

from worldfoundry.core.nn.blocks.layers import DropPath, LayerScale, Mlp

from ...utils.geometry import se3_inverse
from .attention import Attention, CrossAttentionRope, PRopeFlashAttention


class BlockRope(nn.Module):
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
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
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
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, xpos=None) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), xpos=xpos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
        return x


class CrossBlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        cross_attn_class: Callable[..., nn.Module] = CrossAttentionRope,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        init_values=None,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, rope=rope, qk_norm=qk_norm
        )

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_y = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, rope=rope, qk_norm=qk_norm
        )

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )

    def forward(self, x: Tensor, y: Tensor, xpos=None, ypos=None) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), xpos=xpos))

        def cross_attn_residual_func(x: Tensor, y: Tensor) -> Tensor:
            return self.ls_y(self.cross_attn(self.norm2(x), y, y, qpos=xpos, kpos=ypos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm3(x)))

        x = x + attn_residual_func(x)
        y_ = self.norm_y(y)
        x = x + cross_attn_residual_func(x, y_)
        x = x + ffn_residual_func(x)

        return x


class PoseInjectBlock(nn.Module):
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
        attn_class: Callable[..., nn.Module] = PRopeFlashAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
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
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, poses, H, W, patch_h, patch_w, K=None, connect=False, attn_mask=None) -> Tensor:
        extrinsics = se3_inverse(poses)

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), extrinsics, H, W, patch_h, patch_w, K=K, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if connect:
            return x + attn_residual_func(x) + ffn_residual_func(x)
        return attn_residual_func(x) + ffn_residual_func(x)


class CrossOnlyBlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        # attn_class was removed because it is no longer used
        cross_attn_class: Callable[..., nn.Module] = CrossAttentionRope,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        init_values=None,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")

        # ---------------------------------------------------

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_y = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, rope=rope, qk_norm=qk_norm
        )

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )

    def forward(self, x: Tensor, y: Tensor, xpos=None, ypos=None) -> Tensor:
        # ---------------------------

        def cross_attn_residual_func(x: Tensor, y: Tensor) -> Tensor:
            # NOTE: self.norm2(x) is x after pre-normalization
            return self.ls_y(self.cross_attn(self.norm2(x), y, y, qpos=xpos, kpos=ypos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm3(x)))

        # x = x + attn_residual_func(x)

        y_ = self.norm_y(y)
        x = x + cross_attn_residual_func(x, y_)
        x = x + ffn_residual_func(x)

        return x
