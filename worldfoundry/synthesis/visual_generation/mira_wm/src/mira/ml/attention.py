"""Self-attention building blocks: GQA self-attention with QK-norm, RoPE, and adaptive LayerNorm."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor
from torch.nn import functional as F


def local_causal_mask(q_len: int, k_len: int, context: int | None, device: torch.device) -> Tensor:
    """Create a local causal mask for attention.

    Args:
        q_len: Length of the query sequence.
        k_len: Length of the key sequence.
        context: Size of the local context window, or None for an unbounded causal mask.
        device: Device to create the mask on.

    Returns:
        A boolean mask of shape (q_len, k_len) where True indicates allowed attention.
    """
    tensor = torch.full(
        (q_len, k_len),
        fill_value=1,
        dtype=torch.bool,
        device=device,
    )

    shift = k_len - q_len

    mask = torch.tril(tensor, diagonal=shift)
    if context is not None:
        mask = torch.triu(mask, diagonal=shift - context + 1)
    return mask


class SelfAttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embed_dim: int
    num_heads: int
    # Can be set to None by the config, in which case it is set to num_heads by the validator.
    num_kv_heads: int | None
    gating: bool = False
    qk_norm: Literal["layernorm", "rmsnorm"] = "layernorm"

    @model_validator(mode="after")
    def validate_heads(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        assert self.embed_dim % self.num_heads == 0, (
            f"{self.embed_dim=} must be divisible by {self.num_heads=}"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"{self.num_heads=} must be divisible by {self.num_kv_heads=}"
        )
        return self

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def total_head_dim(self) -> int:
        return (self.num_heads + 2 * self.num_kv_heads) * self.head_dim  # type: ignore

    @property
    def total_kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim  # type: ignore


class SelfAttention(nn.Module):
    def __init__(
        self,
        config: SelfAttentionConfig,
        causal: bool = False,
        context_window_length: int | None = None,
    ):
        super().__init__()
        self.causal = causal
        if context_window_length is not None:
            assert causal, "Context can only be used with causal attention"
        self.context_window_length = context_window_length
        self.gating = config.gating
        self.dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # GQA: fewer KV heads than Q heads. The validator fills num_kv_heads (defaulting to
        # num_heads), so it is always set by the time the module is built.
        assert config.num_kv_heads is not None
        self.num_kv_heads = config.num_kv_heads
        self.gqa_enabled = self.num_kv_heads != self.num_heads

        q_out = self.dim
        kv_out = self.num_kv_heads * self.head_dim
        gate_out = self.dim if self.gating else 0

        self.wqkv = nn.Linear(self.dim, q_out + 2 * kv_out + gate_out, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        qk_norm_cls = QKRMSNorm if config.qk_norm == "rmsnorm" else QKLayerNorm
        self.q_ln = qk_norm_cls((self.num_heads, self.head_dim))
        self.k_ln = qk_norm_cls((self.num_kv_heads, self.head_dim))

    def forward(
        self,
        x: Tensor,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        return_kv: bool = False,
        # (k_ctx, v_ctx) if given, each shape (B, T_ctx, num_kv_heads, head_dim)
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ):
        bsz, seqlen, _ = x.shape

        q_out = self.dim
        kv_out = self.num_kv_heads * self.head_dim
        if self.gating:
            q, k, v, gating = self.wqkv(x).split([q_out, kv_out, kv_out, self.dim], dim=-1)
        else:
            q, k, v = self.wqkv(x).split([q_out, kv_out, kv_out], dim=-1)
            gating = None

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_ln(q)
        k = self.k_ln(k)

        to_cache = None
        if return_kv:
            to_cache = (k.clone(), v.clone())
            if self.context_window_length is not None and seqlen > self.context_window_length:
                to_cache = (
                    k[:, -self.context_window_length :],
                    v[:, -self.context_window_length :],
                )

        if kv_cache is not None:
            k_ctx, v_ctx = kv_cache
            k = torch.cat([k_ctx, k], dim=1)
            v = torch.cat([v_ctx, v], dim=1)

        if rotary_emb is not None:
            if kv_cache is not None:
                rotary_emb_q = (rotary_emb[0][-1:], rotary_emb[1][-1:])
                q = apply_rotary_emb(q, rotary_emb_q)
            else:
                q = apply_rotary_emb(q, rotary_emb)
            k = apply_rotary_emb(k, rotary_emb)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        attn_mask = None
        if self.causal:
            attn_mask = local_causal_mask(
                q_len=q.shape[-2],
                k_len=k.shape[-2],
                context=self.context_window_length,
                device=x.device,
            )
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=self.gqa_enabled)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        if self.gating:
            assert gating is not None
            y = y * torch.sigmoid(gating)

        y = self.wo(y)

        if return_kv:
            return y, to_cache
        return y


def apply_rotary_emb(x: Tensor, freqs: tuple[Tensor, Tensor]):
    x = rearrange(x, "b t n_head c -> b n_head t c")
    # assume the channel dimension c contains c/2 imaginary numbers, each represented by two
    # consecutive values
    cos, sin = freqs  # shape (t c)
    # both cos and sin are repeat_interleaved, so two consecutive values are identical, matching
    # the imaginary numbers above
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)  # both has shape (b, n_head, t, c // 2)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # (b, n_head, t, c)
    # apply the rotation matrix
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    out = rearrange(out, "b n_head t c -> b t n_head c")
    return out


class QKLayerNorm(nn.Module):
    def __init__(self, input_shape: tuple, eps: float = 1e-5):
        super().__init__()
        self.qk_scale = nn.Parameter(torch.ones(input_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = self.qk_scale.to(torch.float32) * x

        x = x.to(input_dtype)

        return x


class QKRMSNorm(nn.Module):
    """Per-head RMSNorm for Q/K. Same parameter shape as QKLayerNorm so checkpoints are interchangeable."""

    def __init__(self, input_shape: tuple, eps: float = 1e-5):
        super().__init__()
        self.qk_scale = nn.Parameter(torch.ones(input_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.qk_scale.to(torch.float32) * x
        return x.to(input_dtype)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.gamma_beta = nn.Linear(cond_dim, 2 * embed_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        normalized_x = self.layer_norm(x)
        gamma_beta = self.gamma_beta(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + gamma) * normalized_x + beta
