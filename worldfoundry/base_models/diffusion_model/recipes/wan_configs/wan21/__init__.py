"""Wan 2.1 architecture EasyDicts (T2V 1.3B / 14B and I2V 14B).

Package index for the official Wan 2.1 release configs.  Each identity
is an EasyDict: UMT5-XXL text, Wan2.1 VAE stride ``(4, 8, 8)``,
patch ``(1, 2, 2)``, and the 1.3B vs 14B DiT widths.  ``WAN_CONFIGS``
maps the short names used by runners.

Shared T5 / dtype / negative-prompt defaults live in :mod:`.shared`.
I2V additionally loads OpenCLIP XLM-RoBERTa.
"""

from .i2v_14b import i2v_14B
from .t2v_14b import t2v_14B
from .t2v_1p3b import t2v_1_3B

WAN_CONFIGS = {
    "t2v-1.3B": t2v_1_3B,
    "t2v-14B": t2v_14B,
    "i2v-14B": i2v_14B,
}

__all__ = ["WAN_CONFIGS", "i2v_14B", "t2v_14B", "t2v_1_3B"]
