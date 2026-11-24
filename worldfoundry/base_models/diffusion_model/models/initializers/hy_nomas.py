"""Hy-Nomas uses ceil(T/4), ceil(H/32), ceil(W/32), including 720p padding."""

from __future__ import annotations

import math

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class HyNomasLatentInitializer:
    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            request.batch_size,
            32,
            math.ceil(request.num_frames / 4),
            math.ceil(request.height / 32),
            math.ceil(request.width / 32),
            device=device,
            dtype=dtype,
            generator=generator,
        )


def build_hy_nomas_initializer(context: ComponentBuildContext) -> HyNomasLatentInitializer:
    return HyNomasLatentInitializer()
