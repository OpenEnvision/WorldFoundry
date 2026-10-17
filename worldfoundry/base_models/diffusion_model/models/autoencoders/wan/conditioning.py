# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Request-owned reuse of a Wan2.1 encoder's settled zero-input condition."""

from __future__ import annotations

import torch
from torch import nn


def wan21_encoder_history_frames(encoder: nn.Module) -> int:
    """Return the temporal input history of the canonical streaming encoder.

    Residual shortcuts have temporal width one, while normalization and
    attention operate independently on each frame. Each temporal convolution
    extends the history by its dilated width at the current sampling stride.
    """
    from .reference_21_streaming import Encoder3d

    if type(encoder) is not Encoder3d:
        raise TypeError("zero-condition reuse requires the canonical Wan2.1 streaming encoder")
    history, stride = 0, 1
    for layer in encoder.modules():
        if isinstance(layer, nn.Conv3d):
            history += (layer.kernel_size[0] - 1) * layer.dilation[0] * stride
            stride *= layer.stride[0]
    if stride != 4:
        raise ValueError("streaming Wan2.1 conditioning requires four-frame temporal compression")
    return history


class WanZeroConditionCache:
    """Cache complete, original-shape conditions after a fixed zero-input tail.

    The owner supplies one seed image followed exclusively by zero pixels.
    A grouped encode is reusable only if its *first* frame is past the encoder
    history. Group sizes retain separate entries, preserving the output
    projection's original geometry. Conditions are read-only; changed image,
    geometry, weights or precision require a new cache.
    """

    def __init__(self, encoder: nn.Module) -> None:
        self.history_frames = wan21_encoder_history_frames(encoder)
        self.encoder = encoder
        self.encoded_zero_frames = 0
        self.reuse_calls = 0
        self._conditions: dict[int, torch.Tensor] = {}

    def _require_inference(self) -> None:
        if self.encoder.training or torch.is_grad_enabled():
            raise RuntimeError("zero-condition reuse requires eval no-grad inference")

    def get(self, pixel_frames: int) -> torch.Tensor | None:
        """Return a settled condition for this exact grouped encode size."""
        self._require_inference()
        condition = self._conditions.get(pixel_frames)
        if condition is not None:
            self.reuse_calls += 1
        return condition

    def record(self, condition: torch.Tensor, pixel_frames: int, *, first: bool) -> None:
        """Observe an actual encode without changing its arithmetic or state."""
        self._require_inference()
        if first:
            self._conditions.clear()
            self.reuse_calls = 0
            self.encoded_zero_frames = pixel_frames - 1
        else:
            if self.encoded_zero_frames >= self.history_frames:
                self._conditions[pixel_frames] = condition
            self.encoded_zero_frames += pixel_frames

    @property
    def nbytes(self) -> int:
        """Storage retained for the complete grouped conditions."""
        return sum(value.numel() * value.element_size() for value in self._conditions.values())
