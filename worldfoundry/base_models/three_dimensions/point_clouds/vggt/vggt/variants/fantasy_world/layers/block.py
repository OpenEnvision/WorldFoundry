# Copyright Alibaba Inc. All Rights Reserved.

from collections.abc import Callable

import torch
from torch import Tensor, nn

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import (
    Attention,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    Block as VGGTBlock,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    NestedTensorBlock as VGGTNestedTensorBlock,
)
from worldfoundry.core.nn.blocks.layers import DropPath, LayerScale, Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import add_residual as add_residual
from worldfoundry.core.nn.blocks.stochastic_depth import (
    get_branges_scales as get_branges_scales,
)

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
        fused_attn: bool = True,  # use _worldfoundry_scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        nn.Module.__init__(self)

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

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

    def attn_residual_func(self, x: Tensor, pos=None, e=None) -> Tensor:
        if e is None:
            return self.ls1(self.attn(self.norm1(x), pos=pos))
        return self.ls1(self.attn(self.norm1(x) * (1 + e[1]) + e[0], pos=pos))

    def ffn_residual_func(self, x: Tensor, e=None) -> Tensor:
        if e is None:
            return self.ls2(self.mlp(self.norm2(x)))
        return self.ls2(self.mlp(self.norm2(x)) * (1 + e[4]) + e[3]) * e[5]

    def forward(
        self,
        x: Tensor,
        pos=None,
        e0=None,
        return_partial: bool = False,
        run_remaining: bool = False,
        modifiers: tuple | None = None,  # e_mod
    ) -> Tensor | tuple[Tensor, tuple]:
        if run_remaining:
            assert modifiers is not None, "run_remaining need modifiers"
            e_mod = modifiers

            x = x + self.ffn_residual_func(x, e=e_mod)
            return x

        if e0 is not None:
            B = e0.shape[0]
            if x.shape[0] != B:
                e0 = e0.unsqueeze(1).repeat(1, x.shape[0] // B, 1, 1)
                e0 = e0.reshape(x.shape[0], 6, -1)

        e_mod = (self.modulation + e0).chunk(6, dim=1) if e0 is not None else None

        x = x + self.attn_residual_func(x, pos=pos, e=e_mod)

        if return_partial:
            return x, e_mod

        if modifiers is not None:
            e_mod = modifiers

        x = x + self.ffn_residual_func(x, e=e_mod)
        return x

    def forward_partial(self, *args, **kwargs):
        return self.forward(*args, **kwargs, return_partial=True)

    def forward_remaining(self, x: Tensor, e=None):
        return self.forward(x, run_remaining=True, modifiers=e)


class CamTokenProjector(nn.Module):
    def __init__(self, out_dim: int, hidden: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(nn.Linear(36, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        B, V_in, _ = cam.shape
        repeat = 3
        pad = cam[:, :1, :].repeat(1, repeat, 1)
        cam = torch.cat([cam, pad], dim=1)
        V_pad = cam.size(1)
        cam = cam.view(B, V_pad // 4, 4, 9).reshape(B, V_pad // 4, 36)

        cam = cam.flatten(0, 1)
        cam = self.mlp(cam).view(-1, 1, self.out_dim)

        return cam


class NestedTensorBlock(VGGTNestedTensorBlock, Block):
    """VGGT nested interface with the family block constructor and forward."""
