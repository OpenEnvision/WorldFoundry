"""UniDepth inference components.

Author: Luigi Piccinelli. CC-BY-NC-4.0.
Adapted from UniDepth 8d8cfe4c7ee15297099983607febf0d4f32eb3d6.
"""

from functools import partial
from math import pi
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils.misc import default


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2, bias=use_bias)
        self.q = nn.Linear(dim, dim, bias=use_bias)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=use_bias)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if pos_embed is not None:
            pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
            q = q + pos_embed
        if pos_embed_context is not None:
            pos_embed_context = rearrange(pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads)
            k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, attn_mask=attn_bias)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = (
            self.ls1(
                self.attn(
                    x,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.mlp(x)) + x
        return x


class ConvUpsampleShuffleResidual(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        expansion: int = 4,
        layer_scale: float = 1.0,
        kernel_size: int = 7,
        padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                )
            )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_dim // 4,
                hidden_dim // 4,
                kernel_size=7,
                padding=3,
                padding_mode=padding_mode,
                groups=hidden_dim // 4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim // 4,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x) + self.residual(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class CvnxtBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        layer_scale=1.0,
        expansion=4,
        dilation=1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            groups=dim,
            dilation=dilation,
            padding_mode=padding_mode,
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones((dim))) if layer_scale > 0.0 else 1.0

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.gamma * x
        x = input + x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = default(output_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        from xformers.components.attention import NystromAttention

        self.attention_fn = NystromAttention(num_landmarks=128, num_heads=num_heads, dropout=dropout)

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(pos_embed, "b n (h d) -> b n h d", h=self.num_heads)
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads)
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)
