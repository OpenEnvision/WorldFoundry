# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Open-Sora and Open-Oasis; sinusoidal features originate in OpenAI GLIDE.
"""PixArt conditioning with CPU-built frequencies and device-side trigonometry."""

import math

import torch

from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import SizeEmbedder, TimestepEmbedder


class CpuFrequencyTimestepEmbedder(TimestepEmbedder):
    """Keep CPU FP32 frequency construction and the caller's projection dtype."""

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class CpuFrequencySizeEmbedder(SizeEmbedder):
    """Sana's size projection with the CPU-frequency convention of Open-Sora."""

    timestep_embedding = staticmethod(CpuFrequencyTimestepEmbedder.timestep_embedding)
