"""FlashAttention 2, when the package is installed."""

from __future__ import annotations

try:
    from flash_attn import flash_attn_func as flash_attention_2
except Exception:
    flash_attention_2 = None
