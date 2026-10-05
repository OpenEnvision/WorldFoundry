"""Cosmos Predict2 Video2World :class:`~...contracts.EncodedLatentInitializer`.

``initialize`` always raises: this role requires the recipe-bound
:class:`~...contracts.LatentEncoder`.  ``initialize_with_encoder`` encodes
the observation (image or video tail via
:func:`.video_conditioning.prepare_video_conditioning_pixels`), draws
σ-max Gaussian noise, and returns :class:`~...contracts.LatentInitialization`
whose extra tensors (``condition_latents``, ``condition_mask``,
``condition_indicator``, ``padding_mask``, ``fps``) become
``DenoiserInput.conditioning`` for :class:`~..denoisers.cosmos2.Cosmos2Denoiser`.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest, LatentEncoder, LatentInitialization
from .video_conditioning import prepare_video_conditioning_pixels


class Cosmos2Video2WorldInitializer:
    """Encode the observation and create the sigma-max latent state."""

    def __init__(
        self,
        *,
        sigma_max: float = 80.0,
        spatial_compression: int = 8,
        temporal_compression: int = 4,
    ) -> None:
        self.sigma_max = float(sigma_max)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Rejected: Predict2 must use :meth:`initialize_with_encoder`."""
        del request, generator, device, dtype
        raise RuntimeError("Cosmos Predict2 requires its latent_encoder binding")

    @torch.no_grad()
    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        if request.height % 16 or request.width % 16:
            raise ValueError("Cosmos Predict2 height and width must be divisible by 16")
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError(
                "Cosmos Predict2 num_frames must satisfy "
                f"(num_frames - 1) % {self.temporal_compression} == 0"
            )
        # Released Video2World weights use a 93-frame sampling window. Shorter
        # latent sequences produce late-frame breakup. The shared decoder trims
        # to the original request length after sampling this complete window.
        request = replace(request, num_frames=max(93, request.num_frames))
        pixels, conditioned_frames = prepare_video_conditioning_pixels(
            request,
            device=device,
            dtype=dtype,
            temporal_compression=self.temporal_compression,
            owner="Cosmos Predict2 Video2World",
        )
        if pixels is None:
            raise ValueError("Cosmos Predict2 Video2World requires an image or video input")

        latent_frames = (request.num_frames - 1) // self.temporal_compression + 1
        shape = (
            request.batch_size,
            16,
            latent_frames,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
        )
        noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * self.sigma_max
        condition_latents = latent_encoder.encode(pixels).to(device=device, dtype=dtype)
        if condition_latents.shape != noise.shape:
            raise ValueError(
                "Cosmos Predict2 condition latent geometry does not match denoiser noise: "
                f"{tuple(condition_latents.shape)} vs {tuple(noise.shape)}"
            )
        condition_count = (conditioned_frames - 1) // self.temporal_compression + 1
        indicator = torch.zeros(
            (request.batch_size, 1, latent_frames, 1, 1),
            device=device,
            dtype=dtype,
        )
        indicator[:, :, :condition_count] = 1.0
        condition_mask = indicator.expand(-1, -1, -1, shape[-2], shape[-1])
        return LatentInitialization(
            latents=noise,
            conditioning={
                "condition_latents": condition_latents,
                "condition_mask": condition_mask,
                "condition_indicator": indicator,
                "padding_mask": torch.zeros(
                    (request.batch_size, 1, request.height, request.width),
                    device=device,
                    dtype=dtype,
                ),
                "fps": float(request.inputs.get("fps", 16.0)),
            },
        )


def build_cosmos2_video2world_initializer(
    context: ComponentBuildContext,
) -> Cosmos2Video2WorldInitializer:
    options = context.component_options
    return Cosmos2Video2WorldInitializer(
        sigma_max=float(options.get("sigma_max", 80.0)),
        spatial_compression=int(options.get("spatial_compression", 8)),
        temporal_compression=int(options.get("temporal_compression", 4)),
    )


__all__ = ["Cosmos2Video2WorldInitializer", "build_cosmos2_video2world_initializer"]
