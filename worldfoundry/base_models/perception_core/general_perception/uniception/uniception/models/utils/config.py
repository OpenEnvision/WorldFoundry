"""
Model Utils Config
"""

import os
import warnings

import torch

__all__ = ["set_fused_attn", "use_fused_attn"]

# Use torch.scaled_dot_product_attention where possible
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, "scaled_dot_product_attention")
if "UNICEPTION_FUSED_ATTN" in os.environ:  # noqa: SIM108 - Branches carry per-case comments.
    _USE_FUSED_ATTN = int(os.environ["UNICEPTION_FUSED_ATTN"])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on


def use_fused_attn() -> bool:
    "Return whether to use torch.nn.functional.scaled_dot_product_attention"
    return _USE_FUSED_ATTN > 0


def set_fused_attn(enable: bool = True):
    "Set whether to use torch.nn.functional.scaled_dot_product_attention"
    global _USE_FUSED_ATTN
    if not _HAS_FUSED_ATTN:
        warnings.warn(
            "This version of pytorch does not have F.scaled_dot_product_attention, fused_attn flag ignored.",
            stacklevel=1,
        )
        return
    _USE_FUSED_ATTN = 1 if enable else 0
