# SPDX-License-Identifier: MIT
# Derived from FastWAM official source revision 45d8e1458921d83f8ad6cf9ce993d371208dabd0.
"""Shared WAM blocks with family-bound attention and native-dtype modulation."""

from collections.abc import Callable

import torch
from torch import nn

from worldfoundry.core.kernels import residual_gate_add
from worldfoundry.core.nn import RMSNorm
from worldfoundry.core.nn import scale_shift as modulate

from ..model import Head


class NativeDtypeHead(Head):
    """Keep the WAM timestep/modulation arithmetic in its native dtype."""

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class TorchGateModule(nn.Module):
    """Preserve unfused residual math in the WAM checkpoint graph."""

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual_gate_add(x, residual, gate, backend="torch")


class WamSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        *,
        attention_fn: Callable[..., torch.Tensor],
        rope_fn: Callable[..., torch.Tensor],
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        self.attention_fn = attention_fn
        self.rope_fn = rope_fn

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, self_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = self.rope_fn(q, freqs, self.num_heads)
        k = self.rope_fn(k, freqs, self.num_heads)
        x = self.attention_fn(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class WamCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        *,
        attention_fn: Callable[..., torch.Tensor],
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        self.attention_fn = attention_fn

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attention_fn(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class WamDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        attention_fn: Callable[..., torch.Tensor],
        rope_fn: Callable[..., torch.Tensor],
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = WamSelfAttention(
            hidden_dim, attn_head_dim, num_heads, eps, attention_fn=attention_fn, rope_fn=rope_fn
        )
        self.cross_attn = WamCrossAttention(hidden_dim, attn_head_dim, num_heads, eps, attention_fn=attention_fn)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, hidden_dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = TorchGateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)  # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x
