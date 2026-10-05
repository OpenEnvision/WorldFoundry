"""StepVideo :class:`~...contracts.LatentInitializer`.

StepVideo's VAE packs every 17 pixel frames into 3 latent frames and uses
16× spatial compression with 64 latent channels.  Noise layout is
``[B, T_lat, 64, H/16, W/16]`` (channel-last-after-time, matching the
native DiT).  ``num_frames`` must be a positive multiple of 17.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class StepVideoLatentInitializer:
    """Draw StepVideo-shaped Gaussian noise from :class:`DiffusionRequest` geometry."""

    def initialize(self, request, *, generator, device, dtype):
        """Return ``[B, (num_frames//17)*3, 64, H/16, W/16]`` noise."""
        if request.height % 16 or request.width % 16:
            raise ValueError("StepVideo height and width must be divisible by 16")
        if request.num_frames < 17 or request.num_frames % 17:
            raise ValueError("StepVideo num_frames must be a positive multiple of 17")
        latent_frames = request.num_frames // 17 * 3
        return torch.randn(
            request.batch_size,
            latent_frames,
            64,
            request.height // 16,
            request.width // 16,
            generator=generator,
            device=device,
            dtype=dtype,
        )


def build_step_video_latent_initializer(context: ComponentBuildContext) -> StepVideoLatentInitializer:
    """Return a stateless StepVideo initializer (no component options)."""
    del context
    return StepVideoLatentInitializer()


__all__ = ["StepVideoLatentInitializer", "build_step_video_latent_initializer"]
