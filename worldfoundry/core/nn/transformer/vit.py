"""Shared pre-norm ViT transformer block implementations.

Residual + LayerScale + DropPath around core attention. RoPE variants
inject frequencies before the attention callable so ViT and DiT share
the same block shell.

Not this module:
    Attention kernels live in :mod:`worldfoundry.core.attention`.
    Subset-batch stochastic depth lives in
    :mod:`worldfoundry.core.nn.blocks.stochastic_depth`. Pluggable fused
    pre/post hooks live in :mod:`worldfoundry.core.nn.transformer.ops`.

Public surface:

- :func:`apply_prenorm_transformer_residuals` — drop-path / SD schedule.
- :class:`PreNormTransformerBlock` — norm → attn → ls → FFN.
- :class:`RopePreNormTransformerBlock` — same shell, ``pos`` / ``attn_mask``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from torch import Tensor, nn

from worldfoundry.core.attention import QKVSelfAttention
from worldfoundry.core.nn.blocks.layers import DropPath, LayerScale, Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import drop_add_residual_stochastic_depth


# ──────────────────────────────────────────────────────────────────────────
# Residual schedule — full-batch DropPath vs subset-batch stochastic depth
# ──────────────────────────────────────────────────────────────────────────


def apply_prenorm_transformer_residuals(
    x: Tensor,
    *,
    attn_residual: Callable[..., Tensor],
    ffn_residual: Callable[[Tensor], Tensor],
    sample_drop_ratio: float,
    drop_path1: nn.Module,
    drop_path2: nn.Module | None = None,
    training: bool = False,
    stochastic_depth_pos: Optional[Tensor] = None,
    attn_residual_kwargs: Optional[dict[str, Any]] = None,
) -> Tensor:
    """Run attention and FFN sub-blocks with drop-path / stochastic-depth scheduling."""

    attn_kwargs = dict(attn_residual_kwargs or {})
    sd_pos = stochastic_depth_pos

    if training and sample_drop_ratio > 0.1:
        # High drop rates waste a full-batch mask; subset-batch SD keeps the
        # residual expectation while skipping most of the attention/FFN compute.
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=attn_residual,
            sample_drop_ratio=sample_drop_ratio,
            pos=sd_pos,
        )
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=ffn_residual,
            sample_drop_ratio=sample_drop_ratio,
        )
    elif training and sample_drop_ratio > 0.0:
        drop_path = drop_path1
        if sd_pos is not None or attn_kwargs:
            x = x + drop_path(attn_residual(x, pos=sd_pos, **attn_kwargs))
        else:
            x = x + drop_path(attn_residual(x))
        x = x + drop_path(ffn_residual(x))  # FIXME: drop_path2
    else:
        if sd_pos is not None or attn_kwargs:
            x = x + attn_residual(x, pos=sd_pos, **attn_kwargs)
        else:
            x = x + attn_residual(x)
        x = x + ffn_residual(x)
    return x


# ──────────────────────────────────────────────────────────────────────────
# Pre-norm blocks — LayerScale + DropPath around a swappable attn/FFN pair
# ──────────────────────────────────────────────────────────────────────────


class PreNormTransformerBlock(nn.Module):
    """Pre-norm transformer block: norm → attn → ls → drop_path → norm → ffn."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Any = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = QKVSelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        *,
        norm_eps: float | None = None,
        attn_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Wire LayerScale only when ``init_values`` is set (CaiT-style residuals).

        ``attn_class`` / ``ffn_layer`` stay factories so RoPE or SwiGLU can
        replace the defaults without a second block type. ``drop_path`` is
        stored as ``sample_drop_ratio`` for the subset-batch SD threshold.
        """

        super().__init__()
        norm_kwargs = {"eps": norm_eps} if norm_eps is not None else {}
        extra_attn_kwargs = dict(attn_kwargs or {})

        self.norm1 = norm_layer(dim, **norm_kwargs)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            **extra_attn_kwargs,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, **norm_kwargs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N, dim]`` pre-norm residual; eval is a plain residual add."""

        return apply_prenorm_transformer_residuals(
            x,
            attn_residual=lambda xs: self.ls1(self.attn(self.norm1(xs))),
            ffn_residual=lambda xs: self.ls2(self.mlp(self.norm2(xs))),
            sample_drop_ratio=self.sample_drop_ratio,
            drop_path1=self.drop_path1,
            drop_path2=self.drop_path2,
            training=self.training,
        )


class RopePreNormTransformerBlock(PreNormTransformerBlock):
    """Pre-norm block whose attention accepts positional embeddings and optional masks."""

    def forward(self, x: Tensor, pos: Any = None, attn_mask: Any = None) -> Tensor:
        """Same residual schedule as the base block, forwarding ``pos`` / ``attn_mask``.

        ``pos`` is also passed into subset-batch SD so dropped rows keep
        matching rotary frequencies.
        """

        return apply_prenorm_transformer_residuals(
            x,
            attn_residual=lambda xs, pos=None, attn_mask=None: self.ls1(
                self.attn(self.norm1(xs), pos=pos, attn_mask=attn_mask)
            ),
            ffn_residual=lambda xs: self.ls2(self.mlp(self.norm2(xs))),
            sample_drop_ratio=self.sample_drop_ratio,
            drop_path1=self.drop_path1,
            drop_path2=self.drop_path2,
            training=self.training,
            stochastic_depth_pos=pos,
            attn_residual_kwargs={"attn_mask": attn_mask},
        )


__all__ = [
    "PreNormTransformerBlock",
    "RopePreNormTransformerBlock",
    "apply_prenorm_transformer_residuals",
]
