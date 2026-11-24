"""Attention blocks required by LAQ latent-action extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        values, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * values


def feed_forward(dim: int, mult: int = 4, dropout: float = 0.0) -> nn.Sequential:
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PEG(nn.Module):
    def __init__(self, dim: int, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    def forward(self, x: Tensor, shape: tuple[int, int, int, int] | None = None) -> Tensor:
        needs_shape = x.ndim == 3
        if needs_shape and shape is None:
            raise ValueError("Flattened PEG input requires its video shape")

        original_shape = x.shape
        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding))
        x = self.dsconv(x)
        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")
        return x.reshape(original_shape)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.scale = 8
        inner_dim = dim_head * heads

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        kv_input = x
        query = self.to_q(self.norm(x))
        key, value = self.to_kv(kv_input).chunk(2, dim=-1)
        query, key, value = (rearrange(item, "b n (h d) -> b h n d", h=self.heads) for item in (query, key, value))

        query = F.normalize(query, dim=-1) * self.q_scale
        key = F.normalize(key, dim=-1) * self.k_scale
        similarity = einsum("b h i d, b h j d -> b h i j", query, key) * self.scale
        if attn_bias is not None:
            similarity = similarity + attn_bias

        attention = self.attn_dropout(similarity.softmax(dim=-1))
        output = einsum("b h i j, b h j d -> b h i d", attention, value)
        return self.to_out(rearrange(output, "b h n d -> b n (h d)"))


class ContinuousPositionBias(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        num_dims: int = 2,
        layers: int = 2,
        log_dist: bool = True,
    ) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        modules: list[nn.Module] = [nn.Sequential(nn.Linear(num_dims, dim), nn.LeakyReLU(0.1))]
        modules.extend(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1)) for _ in range(layers - 1))
        modules.append(nn.Linear(dim, heads))
        self.net = nn.ModuleList(modules)
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions: int, device: torch.device | str = "cpu") -> Tensor:
        positions = [torch.arange(dimension, device=device) for dimension in dimensions]
        grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
        grid = rearrange(grid, "c ... -> (...) c")
        relative_position = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
        if self.log_dist:
            relative_position = torch.sign(relative_position) * torch.log(relative_position.abs() + 1)
        self.rel_pos = relative_position

        bias = relative_position.float()
        for layer in self.net:
            bias = layer(bias)
        return rearrange(bias, "i j h -> h i j")


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        peg: bool = False,
        peg_causal: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                        ),
                        None,
                        feed_forward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        video_shape: tuple[int, int, int, int] | None = None,
        attn_bias: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        for peg, self_attention, _, feedforward in self.layers:
            if peg is not None:
                x = peg(x, shape=video_shape) + x
            x = self_attention(x, attn_bias=attn_bias) + x
            x = feedforward(x) + x
        return self.norm_out(x)
