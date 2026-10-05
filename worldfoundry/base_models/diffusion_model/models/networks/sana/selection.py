"""Explicit SANA block selection without a second framework registry.

The released SANA and SANA-Video checkpoints store the block implementation
name in their model configuration (LiteLA, GDN, UCPE CamCtrl, Triton fused
variants, GLUMBConv / Linear FFN).  Resolving those names belongs to the
checkpoint-compatible graph, but registering them globally does not.  The
small lazy mapping here keeps optional GDN/Triton imports out of ordinary SANA
image inference and makes every supported checkpoint architecture visible in
one place.
"""

from __future__ import annotations

from importlib import import_module


_ATTENTION_BLOCKS = {
    "GDN": ("sana_gdn_blocks", "GDN"),
    "BidirectionalGDN": ("sana_gdn_blocks", "BidirectionalGDN"),
    "BidirectionalGDNTriton": ("sana_gdn_blocks_triton", "BidirectionalGDNTriton"),
    "BidirectionalGDNUCPESinglePathLiteLATriton": (
        "sana_gdn_blocks_triton",
        "BidirectionalGDNUCPESinglePathLiteLATriton",
    ),
    "BidirectionalGDNUCPESinglePathLiteLABothTriton": (
        "sana_gdn_blocks_triton",
        "BidirectionalGDNUCPESinglePathLiteLABothTriton",
    ),
    "V2VBiGDNAttention": ("sana_v2v_attn_blocks", "V2VBiGDNAttention"),
    "V2VStateCachedBiGDNAttention": (
        "sana_v2v_attn_blocks",
        "V2VStateCachedBiGDNAttention",
    ),
    "V2VAfterRoPEGatedSoftmaxAttention": (
        "sana_v2v_attn_blocks",
        "V2VAfterRoPEGatedSoftmaxAttention",
    ),
    "V2VGatedSoftmaxAttention": (
        "sana_v2v_attn_blocks",
        "V2VGatedSoftmaxAttention",
    ),
}

_FFN_BLOCKS = {
    "CachedGLUMBConvTemp": ("basic_modules", "CachedGLUMBConvTemp"),
}


def _resolve(name: str | None, table: dict[str, tuple[str, str]], *, kind: str):
    if not name:
        return None
    try:
        module_name, attribute = table[name]
    except KeyError as error:
        supported = ", ".join(sorted(table))
        raise ValueError(f"unknown Sana {kind} {name!r}; supported: {supported}") from error
    module = import_module(f"{__package__}.{module_name}")
    return getattr(module, attribute)


def resolve_attention_block(name: str | None):
    """Resolve a released Sana attention block by checkpoint config name."""

    return _resolve(name, _ATTENTION_BLOCKS, kind="attention block")


def resolve_ffn_block(name: str | None):
    """Resolve a released Sana feed-forward block by checkpoint config name."""

    return _resolve(name, _FFN_BLOCKS, kind="feed-forward block")


__all__ = ["resolve_attention_block", "resolve_ffn_block"]
