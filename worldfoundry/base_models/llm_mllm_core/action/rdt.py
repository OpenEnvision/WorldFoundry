# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from RDT-1B and Go1; the shared sinusoid/MLP is owned by SANA.
"""RDT/Go1 timestep embedding with a checkpoint-selected basis dtype."""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import (
    TimestepEmbedder as SanaTimestepEmbedder,
)


class TimestepEmbedder(SanaTimestepEmbedder):
    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__(hidden_size, frequency_embedding_size)
        self._worldfoundry_execution_dtype = dtype

    @SanaTimestepEmbedder.dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        self._worldfoundry_execution_dtype = value

    def timestep_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        return super().timestep_embedding(t, dim, max_period).to(self.dtype)
