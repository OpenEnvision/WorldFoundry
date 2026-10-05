"""StepVideo attention processors on ``B S H D`` QKV.

``torch`` uses shared SDPA.  ``parallel`` is intentionally not implemented
in this graph: sequence-parallel attention is installed by the runtime.
"""

import torch
import torch.nn as nn
from einops import rearrange
from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention

class Attention(nn.Module):
    """Mixin that selects the SDPA processor used by self/cross-attn blocks."""

    def __init__(self):
        super().__init__()
    
    def attn_processor(self, attn_type):
        """Return the callable for ``attn_type`` (``torch`` only in this graph)."""
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            raise ValueError("StepVideo parallel attention is configured by the shared runtime, not the model graph")
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):
        """SDPA on ``B S H D``; a 3-D mask is broadcast across heads."""

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = _worldfoundry_scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        """Placeholder: sequence-parallel attention is wired by the runtime, not here."""
        raise RuntimeError("parallel attention must be installed through a WorldFoundry runtime extension")
