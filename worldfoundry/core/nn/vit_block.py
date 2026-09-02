"""Shared pre-norm ViT transformer block implementations."""

from __future__ import annotations

from typing import Any, Callable, Optional

from torch import Tensor, nn

from worldfoundry.core.attention import QKVSelfAttention
from worldfoundry.core.nn.layers import DropPath, LayerScale, Mlp
from worldfoundry.core.nn.stochastic_depth import drop_add_residual_stochastic_depth


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
    # Keep the two residual branches independently configurable.  ``drop_path2``
    # is optional for callers of this helper that predate the second path, so
    # retain the historical behavior by falling back to ``drop_path1``.
    ffn_drop_path = drop_path1 if drop_path2 is None else drop_path2

    if training and sample_drop_ratio > 0.1:
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
        if sd_pos is not None or attn_kwargs:
            x = x + drop_path1(attn_residual(x, pos=sd_pos, **attn_kwargs))
        else:
            x = x + drop_path1(attn_residual(x))
        x = x + ffn_drop_path(ffn_residual(x))
    else:
        if sd_pos is not None or attn_kwargs:
            x = x + attn_residual(x, pos=sd_pos, **attn_kwargs)
        else:
            x = x + attn_residual(x)
        x = x + ffn_residual(x)
    return x


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
