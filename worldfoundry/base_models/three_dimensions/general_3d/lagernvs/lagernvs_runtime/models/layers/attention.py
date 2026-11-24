# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def _get_flash_attention_ops():
    """Automatically detect GPU and return appropriate flash attention ops.

    Returns Flash Attention 3 ops for H100 (compute capability >= 9.0),
    otherwise returns Flash Attention 2 ops.
    """
    if not torch.cuda.is_available():
        return None
    if xops is None:
        return None

    # Get compute capability of current device
    major, _ = torch.cuda.get_device_capability()

    # H100 has compute capability 9.0
    if major >= 9:
        # Use Flash Attention 3 for H100 and newer
        try:
            return (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)
        except AttributeError:
            # Fall back to flash2 if flash3 not available
            print("Flash Attention 3 not available, falling back to Flash Attention 2")
            return (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
    else:
        # Use Flash Attention 2 for older GPUs
        return (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias=False,
        fc_bias=False,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        self.attn_dropout = attn_dropout

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Get appropriate flash attention ops based on GPU
        self.flash_attn_ops = _get_flash_attention_ops()

    def forward(self, q: torch.Tensor, kv=None) -> torch.Tensor:
        # attention block that supports non-query keys and values
        if kv is None:
            kv = q
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q, k, v = (einops.rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        dropout_p = self.attn_dropout if self.training else 0.0
        if xops is not None:
            try:
                x = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    p=dropout_p,
                    op=self.flash_attn_ops,
                )
            except (NotImplementedError, RuntimeError, ValueError):
                x = None
        else:
            x = None

        if x is None:
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
            ).transpose(1, 2)

        x = einops.rearrange(x, "b n h d -> b n (h d)")

        x = self.attn_fc_dropout(self.proj(x))
        return x
