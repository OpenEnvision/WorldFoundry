"""Inference-only probe hooks.

The upstream project includes optional activation logging used for training and
debugging. WorldFoundry only needs inference, so the public hook is kept as a
no-op to avoid importing the heavy probing stack.
"""

from __future__ import annotations

import torch


def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:
    return x
