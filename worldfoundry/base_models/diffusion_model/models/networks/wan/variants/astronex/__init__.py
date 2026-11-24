# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Astronex-World: Wan2.2 TI2V-5B with PRoPE camera and action conditioning.

Submodules
----------
- :mod:`.attention` — FlashAttention 3/2 or SDPA selection for every attention call.
- :mod:`.action` — per-frame action embedder added to the timestep modulation.
- :mod:`.model` — bidirectional denoiser (the released teacher sampling form).
- :mod:`.causal` — causal denoiser with sink + sliding-window KV caches and a
  chunk-anchored PRoPE cache (the released student sampling form).

Both denoisers load the released ``model-*.safetensors`` (``model.``-prefixed Wan2.2
keys plus ``self_attn.prope_o`` and ``action_embedder``). PRoPE math comes from
:mod:`..minwm.prope`; Wan norms, RoPE and the causal head come from :mod:`..forcing`.
"""
