"""Gamma-World / Cosmos-Predict video DiT family.

:class:`MinimalDiT` (in ``dit.py``) is the full-attention backbone:
``B C T H W`` latents → 3D patch tokens ``B (T'H'W') C`` with 3D RoPE
or learnable axis embeddings.  ``causal.py`` adds chunk-causal self-attn
and KV cache.  ``multiview_dit.py`` injects multi-agent action tokens.
Context-parallel and neighborhood attention live in sibling modules.
"""
