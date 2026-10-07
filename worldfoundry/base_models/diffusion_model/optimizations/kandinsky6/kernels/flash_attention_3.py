"""FlashAttention 3, when the package is installed."""

from __future__ import annotations

try:
    from flash_attn_interface import flash_attn_func as flash_attention_3
except Exception:
    flash_attention_3 = None
