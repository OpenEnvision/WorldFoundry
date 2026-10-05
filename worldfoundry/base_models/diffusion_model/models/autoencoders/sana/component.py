"""Sana DC-AE :class:`~...contracts.LatentEncoder` + :class:`~...contracts.LatentDecoder`.

Wraps the f32c32 Deep-Compression Autoencoder.  ``encode`` scales latents
by the official ``scaling_factor`` (~0.414).  ``decode`` inverts that
scale (and optional ``decoder_input_scale``) then clamps pixels to
``[-1, 1]``.  Image-only: BCHW RGB in, BCHW out.  ``return_latent`` skips
decode.
"""

from __future__ import annotations

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from .dc_ae import DCAE, dc_ae_f32c32


class SanaDCAutoencoder:
    """Encode and decode Sana image latents with explicit scale semantics."""

    spatial_compression_factor = 32
    temporal_compression_factor = 1
    latent_ch = 32

    def __init__(
        self,
        model: DCAE,
        *,
        scaling_factor: float = 0.41407,
        decoder_input_scale: float = 1.0,
        spatial_tile_size: int | None = None,
        spatial_overlap: int = 256,
    ) -> None:
        self.model = model
        self.scaling_factor = float(scaling_factor)
        self.decoder_input_scale = float(decoder_input_scale)
        if self.decoder_input_scale <= 0:
            raise ValueError("Sana decoder_input_scale must be positive")
        self.spatial_tile_size = spatial_tile_size
        self.spatial_overlap = int(spatial_overlap)
        if spatial_tile_size is not None and (
            spatial_tile_size <= 0 or spatial_tile_size % self.spatial_compression_factor
            or not 0 <= self.spatial_overlap < spatial_tile_size
            or self.spatial_overlap % self.spatial_compression_factor
        ):
            raise ValueError("Sana decode tile size and overlap must be multiples of 32 with 0 <= overlap < tile size")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Sana DC-AE expects BCHW RGB images, got {tuple(images.shape)}")
        parameter = next(self.model.parameters())
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        return self.model.encode(images) * self.scaling_factor

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        if request is not None and bool(request.inputs.get("return_latent", False)):
            return latents
        parameter = next(self.model.parameters())
        latents = latents.to(device=parameter.device, dtype=parameter.dtype)
        latents = latents / self.decoder_input_scale
        latents = latents / self.scaling_factor
        if self.spatial_tile_size is not None and max(latents.shape[-2:]) * 32 > self.spatial_tile_size:
            return self._decode_tiled(latents).clamp_(-1.0, 1.0)
        return self.model.decode(latents).clamp_(-1.0, 1.0)

    def _decode_tiled(self, latents: torch.Tensor) -> torch.Tensor:
        """Blend overlapping image tiles; attention context is local to each tile."""
        scale = self.spatial_compression_factor
        tile_size = self.spatial_tile_size // scale
        stride = tile_size - self.spatial_overlap // scale
        height, width = latents.shape[-2:]
        pixels = torch.zeros((latents.shape[0], 3, height * scale, width * scale), device=latents.device, dtype=torch.float32)
        weights = torch.zeros_like(pixels[:1, :1])
        for top in range(0, height, stride):
            for left in range(0, width, stride):
                tile = self.model.decode(latents[..., top:top + tile_size, left:left + tile_size])
                th, tw = tile.shape[-2:]
                wy = torch.ones(th, device=tile.device, dtype=torch.float32)
                wx = torch.ones(tw, device=tile.device, dtype=torch.float32)
                for weight, start, total in ((wy, top * scale, height * scale), (wx, left * scale, width * scale)):
                    overlap = min(self.spatial_overlap, weight.numel())
                    if overlap:
                        ramp = torch.arange(1, overlap + 1, device=tile.device, dtype=torch.float32) / (overlap + 1)
                        if start:
                            weight[:overlap] *= ramp
                        if start + weight.numel() < total:
                            weight[-overlap:] *= ramp.flip(0)
                weight = wy[:, None] * wx[None, :]
                region = (..., slice(top * scale, top * scale + th), slice(left * scale, left * scale + tw))
                pixels[region].add_(tile.float() * weight)
                weights[region].add_(weight)
        return (pixels / weights).to(latents.dtype)


def build_sana_dc_autoencoder(context: ComponentBuildContext) -> SanaDCAutoencoder:
    """Load the original DC-AE graph through the shared module loader."""

    config = dc_ae_f32c32("dc-ae-f32c32-sana-1.1", None)
    model = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=DCAE, config={"cfg": config}),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, DCAE):
        raise TypeError(f"expected DCAE, got {type(model).__name__}")
    return SanaDCAutoencoder(
        model,
        scaling_factor=config.scaling_factor or 0.41407,
        decoder_input_scale=float(context.component_options.get("decoder_input_scale", 1.0)),
        spatial_tile_size=(int(context.component_options.get("spatial_tile_size", 1024)) if context.component_options.get("tiled", False) else None),
        spatial_overlap=int(context.component_options.get("spatial_overlap", 256)),
    )


__all__ = ["SanaDCAutoencoder", "build_sana_dc_autoencoder"]
