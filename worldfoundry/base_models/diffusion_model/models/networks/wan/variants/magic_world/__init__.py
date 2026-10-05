"""MagicWorld: causal Wan 2.x with persistent KV cache.

Submodules
----------
- :mod:`.model` — bidirectional official-layout stem.
- :mod:`.causal` — causal self-attention and cache-aware blocks.

Adds causal / KV-cache hooks.  No VACE, action encoder, or linear attn.
"""
