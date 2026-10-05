"""Vchitect-2B :class:`~...contracts.LatentInitializer`.

Draws SD3-style 16-channel noise with layout
``[B, num_frames, C, H/8, W/8]``.  Vchitect's transformer expects time
before channels (unlike Wan's ``[B, C, T, H, W]``).
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class VchitectLatentInitializer:
    """Create per-frame SD3 latent noise for Vchitect recipes."""

    def __init__(self, *, channels: int = 16, spatial_compression: int = 8) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[B, num_frames, C, H/s, W/s]`` Gaussian noise."""
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError("Vchitect height and width must be divisible by the VAE compression")
        # The released pipeline draws noise using the concatenated prompt embedding
        # dtype. Its CLIP/T5 sequence embedding is float32 even though the
        # transformer runs in bfloat16, so the seeded noise must be float32 too.
        del dtype
        return torch.randn(
            request.batch_size,
            request.num_frames,
            self.channels,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )


def build_vchitect_latent_initializer(context: ComponentBuildContext) -> VchitectLatentInitializer:
    """Build from optional ``channels`` / ``spatial_compression`` options."""
    return VchitectLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
    )


__all__ = ["VchitectLatentInitializer", "build_vchitect_latent_initializer"]
