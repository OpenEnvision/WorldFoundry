"""Path-loaded Wan 2.1 VAE wrapper for DreamX / resident pipelines.

Not the runner :class:`~...contracts.LatentDecoder`.  Constructs the
camera-21 CNN from a ``.pth`` path and applies official mean/std scale.
``encode_to_latent`` / ``decode_to_pixel`` use ``[B, T, C, H, W]``
(time-before-channel), the resident layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .variants.camera_21 import _video_vae

WAN_2P1_VAE_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
WAN_2P1_VAE_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


class WanVAEWrapper(nn.Module):
    """Compatibility facade around a resident checkpoint-backed Wan codec."""

    def __init__(
        self,
        model_root: str | Path | None = None,
        *,
        vae_path: str | Path | None = None,
        mean: Sequence[float] = WAN_2P1_VAE_MEAN,
        std: Sequence[float] = WAN_2P1_VAE_STD,
        z_dim: int = 16,
        temporal_downsample: Sequence[bool] | None = None,
        model_factory=None,
    ) -> None:
        super().__init__()
        if vae_path is None:
            if model_root is None:
                raise ValueError("WanVAEWrapper requires model_root or vae_path")
            vae_path = Path(model_root) / "Wan2.1_VAE.pth"
        self.register_buffer("latent_mean", torch.tensor(tuple(mean)), persistent=False)
        self.register_buffer(
            "latent_inverse_std",
            1.0 / torch.tensor(tuple(std)),
            persistent=False,
        )
        options = {"pretrained_path": str(vae_path), "z_dim": int(z_dim)}
        if temporal_downsample is not None:
            options["temperal_downsample"] = list(temporal_downsample)
        self.model = (model_factory or _video_vae)(**options).eval().requires_grad_(False)

    def _scale(self, value: torch.Tensor) -> list[torch.Tensor]:
        return [self.latent_mean.to(value), self.latent_inverse_std.to(value)]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        encoded = torch.stack(
            [self.model.encode(item.unsqueeze(0), self._scale(pixel)).float().squeeze(0) for item in pixel]
        )
        return encoded.permute(0, 2, 1, 3, 4)

    def encode_to_latent_cached(self, pixel: torch.Tensor) -> torch.Tensor:
        return self.model.cached_encode(pixel, self._scale(pixel)).float().permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache and latent.shape[0] != 1:
            raise ValueError("cached Wan VAE decoding requires batch size 1")
        decode = self.model.cached_decode if use_cache else self.model.decode
        decoded = torch.stack(
            [
                decode(item.unsqueeze(0), self._scale(latent)).float().clamp_(-1, 1).squeeze(0)
                for item in latent.permute(0, 2, 1, 3, 4)
            ]
        )
        return decoded.permute(0, 2, 1, 3, 4)


__all__ = ["WAN_2P1_VAE_MEAN", "WAN_2P1_VAE_STD", "WanVAEWrapper"]
