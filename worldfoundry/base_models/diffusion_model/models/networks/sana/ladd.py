# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Trainable latent-adversarial heads used by SANA-Sprint SCM-LADD.

Adapted from NVlabs/Sana's ``ladd_blocks.py``.  The frozen SANA transformer
remains a separate role in WorldFoundry; this module owns only the parameters
that the discriminator optimizer updates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil, sqrt

import torch
from torch import nn
from torch.nn.utils.spectral_norm import SpectralNorm


class LADDResidualBlock(nn.Module):
    """Residual wrapper around one LADD spectral-conv / norm stage."""

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        if not isinstance(function, nn.Module):
            raise TypeError("LADD residual function must be an nn.Module")
        self.function = function

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.function(inputs) + inputs) / sqrt(2.0)


class LADDSpectralConv1d(nn.Conv1d):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class LADDBatchNormLocal(nn.Module):
    """Per-virtual-batch normalization from the official LADD head."""

    def __init__(
        self,
        num_features: int,
        *,
        affine: bool = True,
        virtual_batch_size: int = 8,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if isinstance(num_features, bool) or int(num_features) <= 0:
            raise ValueError("num_features must be a positive integer")
        if isinstance(virtual_batch_size, bool) or int(virtual_batch_size) <= 0:
            raise ValueError("virtual_batch_size must be a positive integer")
        if float(epsilon) <= 0:
            raise ValueError("epsilon must be positive")
        self.virtual_batch_size = int(virtual_batch_size)
        self.epsilon = float(epsilon)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(int(num_features)))
            self.bias = nn.Parameter(torch.zeros(int(num_features)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"LADD local normalization expects [B,C,N], got {tuple(inputs.shape)}")
        original_shape = inputs.shape
        groups = ceil(inputs.shape[0] / self.virtual_batch_size)
        if inputs.shape[0] % groups:
            raise ValueError(
                "LADD virtual batches must divide the local batch exactly; "
                f"got batch={inputs.shape[0]}, virtual_batch_size={self.virtual_batch_size}"
            )
        grouped = inputs.view(groups, -1, inputs.shape[-2], inputs.shape[-1])
        mean = grouped.mean((1, 3), keepdim=True)
        variance = grouped.var((1, 3), keepdim=True, unbiased=False)
        normalized = (grouped - mean) / torch.sqrt(variance + self.epsilon)
        if self.affine:
            normalized = normalized * self.weight[None, None, :, None]
            normalized = normalized + self.bias[None, None, :, None]
        return normalized.view(original_shape)


def _make_ladd_block(channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        LADDSpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        LADDBatchNormLocal(channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SANAFeatureDiscriminatorHead(nn.Module):
    """One official unconditional DiscHead over a SANA block feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.main = nn.Sequential(
            _make_ladd_block(self.channels, kernel_size=1),
            LADDResidualBlock(_make_ladd_block(self.channels, kernel_size=9)),
        )
        self.classifier = LADDSpectralConv1d(self.channels, 1, kernel_size=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.main(inputs))


class SANAFeatureDiscriminatorHeads(nn.Module):
    """Independent trainable head role for selected frozen SANA block outputs."""

    def __init__(self, *, hidden_size: int, block_ids: Sequence[int]) -> None:
        super().__init__()
        resolved = tuple(int(value) for value in block_ids)
        if not resolved or any(value < 0 for value in resolved):
            raise ValueError("block_ids must contain non-negative indices")
        if len(set(resolved)) != len(resolved):
            raise ValueError("block_ids cannot contain duplicates")
        self.block_ids = resolved
        self.heads = nn.ModuleList(SANAFeatureDiscriminatorHead(hidden_size) for _ in resolved)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        values = tuple(features)
        if len(values) != len(self.heads):
            raise ValueError(f"expected {len(self.heads)} SANA features, got {len(values)}")
        logits: list[torch.Tensor] = []
        for feature, head in zip(values, self.heads, strict=True):
            if feature.ndim != 3 or int(feature.shape[-1]) != head.channels:
                raise ValueError(
                    f"SANA discriminator feature must be [B,N,{head.channels}], got {tuple(feature.shape)}"
                )
            channels_first = feature.transpose(1, 2)
            logits.append(head(channels_first).reshape(channels_first.shape[0], -1))
        return torch.cat(logits, dim=1)


__all__ = [
    "LADDBatchNormLocal",
    "LADDResidualBlock",
    "LADDSpectralConv1d",
    "SANAFeatureDiscriminatorHead",
    "SANAFeatureDiscriminatorHeads",
]
