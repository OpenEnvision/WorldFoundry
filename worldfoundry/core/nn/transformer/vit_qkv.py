"""ViT-style QKV self-attention blocks built on core attention primitives.

Layering rules
--------------
``worldfoundry.core.attention``
    All attention math and kernels (SDPA, RoPE, context-parallel, varlen,
    backends) **and** reusable ViT-style ``nn.Module`` blocks that compose
    those primitives (this module).

``worldfoundry.core.nn``
    Generic building blocks (Mlp, PatchEmbed, DropPath), ``vit_block``,
    stochastic depth, patching. Re-exports attention symbols from
    ``core.attention`` for convenience only — not a second home for attention
    logic.

``worldfoundry.base_models``
    Model-specific ``Attention`` subclasses (e.g. ``MemEffAttention``,
    ``NestedTensorBlock``) that extend or wrap the shared blocks here.

Not this module:
    Attention kernels and RoPE tables live in sibling modules. Generic
    MLP / PatchEmbed live in :mod:`worldfoundry.core.nn`.

Public surface:

- :class:`QKVSelfAttention` — linear QKV + exact SDPA + out-proj.
- :class:`QKNormRopeSelfAttention` — optional Q/K norm and 2D RoPE.
"""

from __future__ import annotations

from typing import Any, Optional

from torch import Tensor, nn

from worldfoundry.core.attention.backends.native import scaled_dot_product_attention


# ──────────────────────────────────────────────────────────────────────────
# Shared ViT blocks — projections stay here; kernels stay in native SDPA
# ──────────────────────────────────────────────────────────────────────────


class QKVSelfAttention(nn.Module):
    """Linear QKV projection, scaled dot-product attention, and output projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build fused QKV and output projections; dropout is training-only."""

        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias: Any = None) -> Tensor:
        """Self-attend ``[B, N, C]`` tokens; ``attn_bias`` is the SDPA mask slot."""

        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        x = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.permute(0, 2, 1, 3).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QKNormRopeSelfAttention(QKVSelfAttention):
    """Self-attention with optional Q/K layer-norm and 2D rotary position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope: Optional[nn.Module] = None,
    ) -> None:
        """Optionally attach per-head Q/K norm and a 2D RoPE module."""

        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        head_dim = dim // num_heads
        self.fused_attn = fused_attn
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.rope = rope

    def forward(self, x: Tensor, pos: Any = None, attn_mask: Any = None) -> Tensor:
        """Self-attend with optional RoPE; ``fused_attn=False`` keeps the matmul path.

        The mask is repeated over heads because callers pass a 2-D token
        mask, not a 4-D score tensor.
        """

        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=((attn_mask)[:, None].repeat(1, self.num_heads, 1, 1) if attn_mask is not None else None),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        """Legacy manual-attention path kept for compatibility."""

        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


__all__ = [
    "QKVSelfAttention",
    "QKNormRopeSelfAttention",
]
