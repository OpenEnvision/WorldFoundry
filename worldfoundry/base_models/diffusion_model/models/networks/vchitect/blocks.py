"""Vchitect joint DiT blocks (AdaLN + spatial/temporal/text attention).

Hidden states are per-frame patches ``(B·F) S C``.  Context is text
tokens repeated over frames.  The last block is ``context_pre_only``:
it updates video tokens but drops the text residual (SD3-style).
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import Attention


class GELU(nn.Module):
    """Linear + tanh-GELU used as the FFN expansion."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand ``[..., dim_in]`` to ``[..., dim_out]``."""
        return torch.nn.functional.gelu(self.proj(hidden_states), approximate="tanh")


class FeedForward(nn.Module):
    """GELU MLP with a no-op Dropout slot so state-dict keys stay ``net.0/1/2``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList((GELU(dim, dim * 4), nn.Dropout(0.0), nn.Linear(dim * 4, dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map ``(B·F) S C`` tokens through the three-module MLP."""
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaLayerNormZero(nn.Module):
    """AdaLN-Zero: emit MSA/MLP ``(shift, scale, gate)`` from the timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, embedding: torch.Tensor):
        """Return ``(normed_for_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)``."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(
            self.silu(embedding)
        ).chunk(6, dim=1)
        normalized = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    """Final / context-pre-only AdaLN (shift/scale only, no residual gate)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """``norm(x) * (1+scale) + shift``; ``embedding`` is ``[B, C]`` (broadcast over S)."""
        scale, shift = self.linear(self.silu(embedding)).chunk(2, dim=1)
        return self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]


class JointTransformerBlock(nn.Module):
    """One Vchitect layer: AdaLN → joint attn → gated FFN (video and optional text)."""

    def __init__(self, dim: int, heads: int, *, context_pre_only: bool) -> None:
        super().__init__()
        self.context_pre_only = bool(context_pre_only)
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormContinuous(dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, context_pre_only=context_pre_only)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim)
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        *,
        frequencies: torch.Tensor,
        frames: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return ``(updated_context or None, updated_video)`` on ``(B·F) S C``."""
        normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, embedding)
        if self.context_pre_only:
            normalized_context = self.norm1_context(encoder_hidden_states, embedding)
        else:
            normalized_context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, embedding
            )
        attention, context_attention = self.attn(
            normalized,
            normalized_context,
            frequencies=frequencies,
            frames=frames,
        )
        hidden_states = hidden_states + gate_msa[:, None] * attention
        normalized = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp[:, None] * self.ff(normalized)
        if self.context_pre_only:
            return None, hidden_states
        encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None] * context_attention
        normalized_context = self.norm2_context(encoder_hidden_states)
        normalized_context = normalized_context * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp[:, None] * self.ff_context(normalized_context)
        return encoder_hidden_states, hidden_states


__all__ = [
    "AdaLayerNormContinuous",
    "AdaLayerNormZero",
    "FeedForward",
    "GELU",
    "JointTransformerBlock",
]
