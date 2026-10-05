"""Causal Wan graphs shared by Self-Forcing / Causal-Forcing / Krea.

Submodules
----------
- :mod:`.model` — bidirectional Wan stem used by forcing recipes.
- :mod:`.self_forcing` / :mod:`.causal_forcing` / :mod:`.long_video`
  — causal self-attn + KV cache for teacher-forcing / long rollout.
- :mod:`.krea` / :mod:`.krea_model` / :mod:`.krea_attention` /
  :mod:`.krea_sage` — Krea realtime causal Wan + optional SageAttention.

Adds causal attention and (Krea) optional linear/Sage kernels.  No VACE
or camera ControlNet in this package.
"""
