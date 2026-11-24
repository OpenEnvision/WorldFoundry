# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Attention backend selection shared by the Astronex-World denoisers."""

from __future__ import annotations

import torch

from worldfoundry.core.attention.backends.probe import probe_attention_backends
from worldfoundry.core.attention.sequence.varlen import flash_attention


def flash_attention_version(device: torch.device) -> int | None:
    """Return 3 for FlashAttention 3 on Hopper, 2 for FlashAttention 2, or ``None`` for PyTorch SDPA."""
    capabilities = probe_attention_backends(device)
    if capabilities["flash_attention_3"].usable:
        return 3
    if capabilities["flash_attention_2"].usable:
        return 2
    return None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_lens: torch.Tensor | None = None) -> torch.Tensor:
    """``[B, L, N, D]`` attention on the fastest installed FlashAttention kernel."""
    return flash_attention(q, k, v, k_lens=k_lens, version=flash_attention_version(q.device))
