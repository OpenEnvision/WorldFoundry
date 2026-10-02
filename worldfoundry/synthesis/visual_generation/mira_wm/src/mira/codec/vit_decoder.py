"""ViT video decoder for the RAEv2 codec (ViTok-v2-style space-time transformer).

The decoder is a deep ViT operating on a patchified latent. Each :class:`SpaceTimeBlock` factorises
attention into bidirectional spatial self-attention within a frame and causal temporal self-attention
across frames, so the codec stays causal in time while behaving like a ViT in space.

It reuses :class:`mira.ml.SelfAttention` (per-head QK-norm, GQA and RoPE) and adds the
pieces ViTok needs on top: patch unembedding, axial 2D + causal 1D RoPE frequencies, LayerScale, a
SwiGLU MLP and a strided ``ConvTranspose2d`` that lifts the latent grid up to the ViT grid.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor, nn

from mira.codec.config import ViTDecoderConfig
from mira.ml import SelfAttention, SelfAttentionConfig, init_weights


def _rope_cos_sin(positions: Tensor, dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """1D RoPE cos/sin for a set of positions.

    Returns ``(cos, sin)`` each of shape ``(len(positions), dim)``, with consecutive pairs
    repeated to match :func:`mira.ml.apply_rotary_emb`.
    """
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    k = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    inv_freq = theta ** (-2.0 * k / dim)  # (dim // 2,)
    freqs = positions.float()[:, None] * inv_freq[None, :]  # (N, dim // 2)
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # (N, dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def temporal_rope(num_frames: int, head_dim: int, theta: float, device: torch.device):
    """Causal 1D RoPE over the time axis. Shape ``(T, head_dim)``."""
    positions = torch.arange(num_frames, device=device)
    return _rope_cos_sin(positions, head_dim, theta)


def spatial_rope(height: int, width: int, head_dim: int, theta: float, device: torch.device):
    """Axial 2D RoPE over a ``height x width`` grid. Shape ``(height * width, head_dim)``.

    Half of ``head_dim`` encodes the row position, the other half the column position.
    """
    assert head_dim % 4 == 0, f"2D RoPE needs head_dim divisible by 4, got {head_dim}"
    half = head_dim // 2
    rows = torch.arange(height, device=device).repeat_interleave(width)  # (h*w,)
    cols = torch.arange(width, device=device).repeat(height)  # (h*w,)
    cos_h, sin_h = _rope_cos_sin(rows, half, theta)
    cos_w, sin_w = _rope_cos_sin(cols, half, theta)
    cos = torch.cat([cos_h, cos_w], dim=-1)  # (h*w, head_dim)
    sin = torch.cat([sin_h, sin_w], dim=-1)
    return cos, sin


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SwiGLU(nn.Module):
    """SwiGLU MLP (~2.67x expansion at dim_multiplier=4), matching the world-model FeedForward."""

    def __init__(self, dim: int, dim_multiplier: int = 4, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * dim_multiplier * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.swish_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.output_linear = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_linear(torch.nn.functional.silu(self.swish_linear(x)) * self.gate_linear(x))


class SpaceTimeBlock(nn.Module):
    """Pre-norm transformer block: spatial (bidirectional) attn, temporal (causal) attn, MLP.

    Operates on tokens shaped ``(B, T, H, W, C)``. Spatial attention is over the ``H*W`` tokens
    within each frame; temporal attention is over the ``T`` frames (causal when ``causal=True``).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None,
        causal: bool,
        mlp_dim_multiplier: int,
        layerscale_init: float,
        eps: float,
        qk_norm: Literal["rmsnorm", "layernorm"] = "rmsnorm",
    ):
        super().__init__()
        attn_config = SelfAttentionConfig(
            embed_dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, qk_norm=qk_norm
        )
        self.space_norm = nn.LayerNorm(dim, eps=eps)
        self.space_attn = SelfAttention(attn_config, causal=False)
        self.space_ls = LayerScale(dim, layerscale_init)

        self.time_norm = nn.LayerNorm(dim, eps=eps)
        self.time_attn = SelfAttention(attn_config, causal=causal)
        self.time_ls = LayerScale(dim, layerscale_init)

        self.mlp_norm = nn.LayerNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim, dim_multiplier=mlp_dim_multiplier)
        self.mlp_ls = LayerScale(dim, layerscale_init)

    def forward(
        self,
        x: Tensor,
        rope_spatial: tuple[Tensor, Tensor],
        rope_temporal: tuple[Tensor, Tensor],
    ) -> Tensor:
        b, t, h, w, _ = x.shape

        xs = rearrange(x, "b t h w c -> (b t) (h w) c")
        xs = xs + self.space_ls(self.space_attn(self.space_norm(xs), rotary_emb=rope_spatial))

        xt = rearrange(xs, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
        xt = xt + self.time_ls(self.time_attn(self.time_norm(xt), rotary_emb=rope_temporal))

        x = rearrange(xt, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
        x = x + self.mlp_ls(self.mlp(self.mlp_norm(x)))
        return x


class PatchUnembed(nn.Module):
    """Project tokens to per-patch pixels and fold back into frames.

    ``(B, T, H//p, W//p, width) -> (B, T * patch_size_t, out_channels, H, W)``.
    """

    def __init__(self, out_channels: int, patch_size: int, width: int, patch_size_t: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.proj = nn.Linear(width, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return rearrange(
            x,
            "b t h w (c pt p1 p2) -> b (t pt) c (h p1) (w p2)",
            c=self.out_channels,
            pt=self.patch_size_t,
            p1=self.patch_size,
            p2=self.patch_size,
        )


def _build_vit_blocks(
    width: int,
    depth: int,
    num_heads: int,
    num_kv_heads: int | None,
    is_causal: bool,
    mlp_dim_multiplier: int,
    layerscale_init: float,
    eps: float,
    qk_norm: Literal["rmsnorm", "layernorm"],
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            SpaceTimeBlock(
                dim=width,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                causal=is_causal,
                mlp_dim_multiplier=mlp_dim_multiplier,
                layerscale_init=layerscale_init,
                eps=eps,
                qk_norm=qk_norm,
            )
            for _ in range(depth)
        ]
    )


def _apply_vit_blocks(
    blocks: nn.ModuleList,
    x: Tensor,
    rope_spatial: tuple[Tensor, Tensor],
    rope_temporal: tuple[Tensor, Tensor],
    activation_checkpointing: bool,
    training: bool,
) -> Tensor:
    for block in blocks:
        if training and activation_checkpointing:
            x = torch.utils.checkpoint.checkpoint(block, x, rope_spatial, rope_temporal, use_reentrant=False)  # type: ignore[assignment]  # checkpoint() is typed Optional but returns the block output
        else:
            x = block(x, rope_spatial, rope_temporal)
    return x


class ViTVideoDecoder(nn.Module):
    """Causal-video ViT decoder for the RAEv2 codec.

    A kxk strided ``ConvTranspose2d`` lifts the latent grid to the ViT grid (e.g. a /32 latent up
    to a /16 ViT grid with ``stride=2``), followed by the space-time ViT blocks and patch unembed.
    """

    def __init__(self, config: ViTDecoderConfig) -> None:
        super().__init__()
        self.config = config
        width = config.vit_width
        self.head_dim = width // config.vit_num_heads

        self.from_latent = nn.ConvTranspose2d(
            config.latent_dim,
            width,
            kernel_size=config.bottleneck.stride,
            stride=config.bottleneck.stride,
            bias=True,
        )
        self.blocks = _build_vit_blocks(
            width,
            config.vit_depth,
            config.vit_num_heads,
            config.vit_num_kv_heads,
            config.is_causal,
            config.mlp_dim_multiplier,
            config.layerscale_init,
            config.eps,
            config.qk_norm,
        )
        self.norm_out = nn.LayerNorm(width, eps=config.eps)
        self.patch_unembed = PatchUnembed(
            config.out_channels, config.patch_size, width, patch_size_t=config.patch_size_t
        )

        self.apply(init_weights)

    @property
    def last_layer_weight(self) -> Tensor:
        return self.patch_unembed.proj.weight

    def forward(self, z: Tensor) -> Tensor:
        # ConvTranspose2d: lift via channels-first, then back to channels-last for ViT blocks.
        b, t = z.shape[:2]
        x = rearrange(z, "b t c h w -> (b t) c h w")
        x = self.from_latent(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t)

        _, t, h, w, _ = x.shape
        rope_spatial = spatial_rope(h, w, self.head_dim, self.config.rope_theta_spatial, x.device)
        rope_temporal = temporal_rope(t, self.head_dim, self.config.rope_theta_temporal, x.device)
        x = _apply_vit_blocks(
            self.blocks,
            x,
            rope_spatial,
            rope_temporal,
            self.config.activation_checkpointing,
            self.training,
        )
        x = self.norm_out(x)
        return torch.tanh(self.patch_unembed(x))
