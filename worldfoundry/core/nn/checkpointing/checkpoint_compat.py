"""Compatibility helpers required when loading released checkpoints.

Older graphs wrap blocks in ``CheckpointFunction``. This module provides
no-op / identity wrappers so inference can load those state dicts
without enabling training checkpointing.

Not this module:
    Training SAC policy lives in
    :mod:`worldfoundry.core.nn.checkpointing.activation_checkpointing`. Do not turn
    that on just to satisfy a missing buffer name.

Public surface:

- :class:`InferenceCheckpointModule` — registers the accumulators some
  released checkpoints still list in their state dict.
"""

from __future__ import annotations

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────────
# Inference-only checkpoint buffers — load released graphs without SAC
# ──────────────────────────────────────────────────────────────────────────


class InferenceCheckpointModule(nn.Module):
    """Preserve legacy checkpoint metadata buffers without training utilities."""

    def __init__(self) -> None:
        """Register the integer/float accumulators older trainers persisted.

        Names and dtypes must match the released keys; values stay at zero
        because inference never updates them.
        """

        super().__init__()
        buffers = {
            "accum_video_sample_counter": torch.tensor(0, dtype=torch.int64),
            "accum_image_sample_counter": torch.tensor(0, dtype=torch.int64),
            "accum_iteration": torch.tensor(0, dtype=torch.int64),
            "accum_train_in_hours": torch.tensor(0.0, dtype=torch.float32),
        }
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)


__all__ = ["InferenceCheckpointModule"]
