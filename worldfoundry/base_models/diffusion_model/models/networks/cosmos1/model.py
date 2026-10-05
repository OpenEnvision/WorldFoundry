"""Inference-only Cosmos Predict1 / GEN3C diffusion transformer.

Checkpoint-shaped forward math only.  Sampling, loading, offload, text
encoding, video tokenization, and camera preparation live in the shared
diffusion infrastructure (``denoisers/cosmos1.py`` + recipe).

Architecture
------------
The 7B Predict1 DiT reuses Cosmos 2.5 block primitives
(:class:`Cosmos25TransformerBlock`, AdaLN, RoPE, patch embed) and adds a
factorized learnable absolute position table (T/H/W) that is RMS-normalized
and added before every block.  Latents are ``B C T H W``; ``in_channels=81``
is the GEN3C packed video+camera layout.  When ``concat_padding_mask`` is
set, a nearest-resized spatial mask is broadcast over time and concatenated
as an extra channel before patchify.

This module must stay numerically compatible with the pinned Hub checkpoint.
Do not introduce runner or offload logic here.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..cosmos2p5.model import (
    Cosmos25AdaLayerNorm,
    Cosmos25PatchEmbed,
    Cosmos25RotaryPosEmbed,
    Cosmos25TimeEmbed,
    Cosmos25TransformerBlock,
)


def _rms_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize the last dim in fp32, then cast the scale back to ``value.dtype``."""
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True, dtype=torch.float32)
    norm = norm / value.shape[-1] ** 0.5 + eps
    return value / norm.to(value.dtype)


class Cosmos1LearnablePosition(nn.Module):
    """Factorized absolute position embedding added before every DiT block."""

    def __init__(self, hidden_size: int, max_size: tuple[int, int, int]) -> None:
        super().__init__()
        frames, height, width = max_size
        self.pos_emb_h = nn.Parameter(torch.empty(height, hidden_size))
        self.pos_emb_w = nn.Parameter(torch.empty(width, hidden_size))
        self.pos_emb_t = nn.Parameter(torch.empty(frames, hidden_size))
        nn.init.trunc_normal_(self.pos_emb_h, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_w, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_t, std=0.02)

    def forward(self, grid: tuple[int, int, int], batch: int) -> torch.Tensor:
        """Return ``[B, T*H*W, C]`` positions for ``grid``, or raise if it exceeds the table."""
        frames, height, width = grid
        if frames > len(self.pos_emb_t) or height > len(self.pos_emb_h) or width > len(self.pos_emb_w):
            raise ValueError(f"Cosmos1 latent grid {grid} exceeds the checkpoint position table")
        value = (
            self.pos_emb_t[:frames, None, None]
            + self.pos_emb_h[None, :height, None]
            + self.pos_emb_w[None, None, :width]
        )
        return _rms_normalize(value).reshape(1, frames * height * width, -1).expand(batch, -1, -1)


class Cosmos1Transformer3DModel(nn.Module):
    """Checkpoint-compatible 7B Cosmos Predict1 DiT used by GEN3C."""

    def __init__(
        self,
        in_channels: int = 81,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            concat_padding_mask=concat_padding_mask,
        )
        patch_channels = in_channels + int(concat_padding_mask)
        self.patch_embed = Cosmos25PatchEmbed(patch_channels, hidden_size, patch_size)
        self.rope = Cosmos25RotaryPosEmbed(attention_head_dim, max_size, patch_size, rope_scale)
        token_grid = tuple(size // patch for size, patch in zip(max_size, patch_size, strict=True))
        self.extra_position = Cosmos1LearnablePosition(hidden_size, token_grid)
        self.time_embed = Cosmos25TimeEmbed(hidden_size)
        self.time_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.transformer_blocks = nn.ModuleList(
            Cosmos25TransformerBlock(
                hidden_size,
                num_attention_heads,
                attention_head_dim,
                text_embed_dim,
                mlp_ratio,
                adaln_lora_dim,
            )
            for _ in range(num_layers)
        )
        self.norm_out = Cosmos25AdaLayerNorm(hidden_size, adaln_lora_dim)
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(hidden_size, patch_volume * out_channels, bias=False)

    def _patchify(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
        fps: float | torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int, int]]:
        """Pack ``B C T H W`` latents into ``B (T'H'W') C`` tokens and 3D RoPE.

        Optionally concatenates a nearest-resized spatial padding mask as an
        extra channel (broadcast over time) so the patch embed sees the GEN3C
        packed layout.  ``grid`` is the post-patch token lattice ``(T', H', W')``.
        """
        batch, _, frames, height, width = hidden_states.shape
        if self.config.concat_padding_mask:
            if padding_mask is None:
                padding_mask = hidden_states.new_zeros((batch, 1, height, width))
            padding_mask = functional.interpolate(padding_mask.float(), (height, width), mode="nearest")
            padding_mask = padding_mask.unsqueeze(2).expand(-1, -1, frames, -1, -1)
            hidden_states = torch.cat((hidden_states, padding_mask.type_as(hidden_states)), dim=1)
        pt, ph, pw = self.config.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError("Cosmos1 latent dimensions must be divisible by the patch size")
        grid = (frames // pt, height // ph, width // pw)
        tokens = self.patch_embed(hidden_states).flatten(1, 3)
        return tokens, self.rope(hidden_states, fps), grid

    def _unpatchify(self, hidden_states: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """Fold ``B (T'H'W') (pt·ph·pw·C)`` tokens back to ``B C T H W`` video latents."""
        from einops import rearrange

        frames, height, width = grid
        pt, ph, pw = self.config.patch_size
        return rearrange(
            hidden_states,
            "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=frames,
            h=height,
            w=width,
            pt=pt,
            ph=ph,
            pw=pw,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        fps: float | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise packed video+camera latents; return ``B C T H W`` residual.

        ``encoder_hidden_states`` is the text (or other) cross-attn context.
        The factorized absolute table is added *before every block*, matching
        the Predict1 checkpoint (not a one-shot positional add at the stem).
        """
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None].to(dtype=torch.bool)
        hidden_states, rotary_emb, grid = self._patchify(hidden_states, padding_mask, fps)
        batch = hidden_states.shape[0]
        timestep = timestep.reshape(batch)
        projected, temb = self.time_embed(hidden_states, timestep)
        projected = self.time_norm(projected)
        extra_position = self.extra_position(grid, batch).type_as(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = hidden_states + extra_position
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                projected,
                temb,
                rotary_emb,
                attention_mask,
            )
        hidden_states = self.proj_out(self.norm_out(hidden_states, projected, temb))
        return self._unpatchify(hidden_states, grid)


__all__ = ["Cosmos1Transformer3DModel"]
