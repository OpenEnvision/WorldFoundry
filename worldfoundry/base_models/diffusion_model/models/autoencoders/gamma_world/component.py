"""Gamma-World multi-view :class:`~...contracts.LatentEncoder` / decoder.

Reuses :class:`~..wan.component.WanVideoDecoder`.  Encode is a passthrough.
Decode splits the packed time axis ``(V*T)`` into per-player clips,
decodes each with Wan, then tiles views horizontally
(``B C T H (V W)``).  ``n_players`` comes from ``request.inputs``.
"""

from __future__ import annotations

import torch
from einops import rearrange

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ..wan.component import WanVideoDecoder, build_wan_video_decoder


class GammaWorldVideoCodec:
    """Reuse one Wan codec while keeping player timelines independent."""

    def __init__(self, codec: WanVideoDecoder) -> None:
        self.codec = codec

    @property
    def spatial_compression_factor(self) -> int:
        return self.codec.spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self.codec.temporal_compression_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode BCTHW pixels with the shared Wan VAE."""
        return self.codec.encode(images)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Decode packed multi-view latents and tile views along width."""
        if bool(request.inputs.get("return_latent", False)):
            return latents
        n_views = int(request.inputs.get("n_players", 2))
        if latents.shape[2] % n_views:
            raise ValueError(
                f"Gamma latent time dimension {latents.shape[2]} is not divisible by n_players={n_views}"
            )
        per_view = rearrange(latents, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded = self.codec.decode(per_view)
        decoded = decoded[:, :, : request.num_frames]
        return rearrange(decoded, "(B V) C T H W -> B C T H (V W)", V=n_views)


def build_gamma_world_video_codec(context: ComponentBuildContext) -> GammaWorldVideoCodec:
    """Build the Wan decoder, then wrap it with the multi-view layout."""
    return GammaWorldVideoCodec(build_wan_video_decoder(context))


__all__ = ["GammaWorldVideoCodec", "build_gamma_world_video_codec"]
