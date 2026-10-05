"""LVDM text-to-video :class:`~...contracts.LatentInitializer`.

Draws Gaussian noise with shape ``[B, C, T, H/8, W/8]``.  LVDM codecs
are frame-wise (no temporal VAE compression), so ``T == request.num_frames``.
Default ``C=4`` matches the SD-family AutoencoderKL used by LVDM / T2V-Turbo.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class LVDMTextToVideoLatentInitializer:
    """Create per-frame SD-latent noise for LVDM / T2V-Turbo recipes."""

    def __init__(self, *, channels: int = 4, spatial_compression: int = 8) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)
        if self.channels <= 0 or self.spatial_compression <= 0:
            raise ValueError("LVDM latent dimensions must be positive")

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[B, C, num_frames, H/s, W/s]`` Gaussian noise."""
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError(
                f"LVDM height and width must be divisible by {self.spatial_compression}: "
                f"got {request.height}x{request.width}"
            )
        return torch.randn(
            request.batch_size,
            self.channels,
            request.num_frames,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
            generator=generator,
            device=device,
            dtype=dtype,
        )


def build_lvdm_t2v_latent_initializer(context: ComponentBuildContext) -> LVDMTextToVideoLatentInitializer:
    """Build from optional ``channels`` / ``spatial_compression`` options."""
    return LVDMTextToVideoLatentInitializer(
        channels=int(context.component_options.get("channels", 4)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
    )


__all__ = ["LVDMTextToVideoLatentInitializer", "build_lvdm_t2v_latent_initializer"]
