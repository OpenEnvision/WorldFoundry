"""DreamX-World adapter around the canonical Wan 2.2 48-channel VAE.

:class:`AutoencoderKLWan3_8` mixes
:class:`~...adapter.WanAutoencoderAdapterMixin` with the
reference Wan 2.2 CNN from :mod:`~...reference_22`.

Latents are 48-channel BCTHW, 4× temporal / 8× spatial,
official Wan 2.2 mean/std.  This is a loading / naming
adapter, not a new residual graph.

Used by DreamX-World recipes; runner VAE38 path is
:func:`~...component.build_wan_video_vae38_decoder`.
"""

from __future__ import annotations

import inspect
from typing import Sequence

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.adapter import (
    WanAutoencoderAdapterMixin,
    WanAutoencoderConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import (
    WAN_2P2_VAE_MEAN,
    WAN_2P2_VAE_STD,
    _video_vae,
)
from worldfoundry.core.model_loading import load_model

class AutoencoderKLWan3_8(WanAutoencoderAdapterMixin, nn.Module):
    """DreamX-World wrapper around the Wan 2.2 48-channel VAE38."""
    def __init__(
        self,
        latent_channels: int = 48,
        c_dim: int = 160,
        vae_pth: str | None = None,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        temperal_downsample: Sequence[bool] = (False, True, True),
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.config = WanAutoencoderConfig(
            latent_channels=int(latent_channels),
            temporal_compression_ratio=int(temporal_compression_ratio),
            spatial_compression_ratio=int(spatial_compression_ratio),
        )
        self.register_buffer(
            "latent_mean", torch.tensor(WAN_2P2_VAE_MEAN), persistent=False
        )
        self.register_buffer(
            "latent_inverse_std",
            1.0 / torch.tensor(WAN_2P2_VAE_STD),
            persistent=False,
        )
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=latent_channels,
            dim=c_dim,
            dim_mult=list(dim_mult),
            temperal_downsample=list(temperal_downsample),
        ).eval().requires_grad_(False)

    @property
    def scale(self) -> list[torch.Tensor]:
        return [self.latent_mean, self.latent_inverse_std]
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        additional_kwargs: dict | None = None,
    ) -> "AutoencoderKLWan3_8":
        parameters = set(inspect.signature(cls.__init__).parameters) - {"self", "cls"}
        config = {
            key: value
            for key, value in (additional_kwargs or {}).items()
            if key in parameters
        }
        return load_model(
            cls,
            pretrained_model_path,
            config=config,
            torch_dtype=torch.float32,
            device="cpu",
            state_dict_converter=lambda state: {
                f"model.{key}": value for key, value in state.items()
            },
        )
