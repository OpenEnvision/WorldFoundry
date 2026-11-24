# SPDX-License-Identifier: Apache-2.0
"""Alias-free upsample filters for the MiniMax H3 audio VAE.

SGLang (Apache-2.0) port of Kaiser-sinc low-pass filters used
by BigVGAN-style activations.  :class:`UpSample1d` /
:class:`DownSample1d` wrap :class:`LowPassFilter1d`.

Tensors are 1-D waveforms or vocoder feature maps
(``N C T``).  Parallel LTX helpers live in
:mod:`~..ltx.audio.vocoder`.

Used by :mod:`.bigvgan` and :mod:`.audio_vae`; not a runner
Protocol adapter.
"""

# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.alias_free import DownSample1d as DownSample1d
from worldfoundry.base_models.diffusion_model.models.autoencoders.alias_free import (
    LowPassFilter1d as LowPassFilter1d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.alias_free import (
    kaiser_sinc_filter1d as kaiser_sinc_filter1d,
)

# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html


class UpSample1d(nn.Module):
    """Alias-free 1-D upsample used by MiniMax H3 vocoder activations."""

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter)

    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x.mul_(self.ratio)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class Activation1d(nn.Module):
    """Periodic activation wrapped with alias-free up/down sampling."""

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x
