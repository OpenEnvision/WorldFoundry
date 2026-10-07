"""SageAttention, when the package is installed."""

from __future__ import annotations

try:
    import sageattention as _sage_mod

    def sage(q, k, v, **_):
        return _sage_mod.sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

except Exception:
    sage = None
