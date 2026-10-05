"""Diffusers-shaped encode/decode façade over the native Wan VAE.

:class:`WanAutoencoderAdapterMixin` returns
:class:`~worldfoundry.core.nn.AutoencoderKLOutput` /
:class:`DecoderOutput` so variant VAEs can drop into older pipelines
without importing Diffusers.  The runner uses :class:`WanVideoDecoder`
instead.  Encode is deterministic (zero log-variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldfoundry.core.nn import (
    AutoencoderKLOutput,
    DecoderOutput,
    DiagonalGaussianDistribution,
    ModuleDeviceDtypeMixin,
)


@dataclass(frozen=True, slots=True)
class WanAutoencoderConfig:
    """Runtime-visible latent geometry used by existing Wan consumers."""

    latent_channels: int
    temporal_compression_ratio: int
    spatial_compression_ratio: int


class WanAutoencoderAdapterMixin(ModuleDeviceDtypeMixin):
    """Expose canonical Wan encode/decode results without backend types."""

    @property
    def latent_channels(self) -> int:
        """Expose Diffusers-compatible latent geometry on the module."""

        return int(self.config.latent_channels)

    @property
    def temporal_compression_ratio(self) -> int:
        """Expose the temporal scale expected by existing Wan pipelines."""

        return int(self.config.temporal_compression_ratio)

    @property
    def spatial_compression_ratio(self) -> int:
        """Expose the spatial scale expected by existing Wan pipelines."""

        return int(self.config.spatial_compression_ratio)

    def encode(self, value: torch.Tensor, return_dict: bool = True):
        """Encode a batch; wrap the mean as a deterministic diagonal Gaussian."""
        self._prepare_codec_offload()
        encoded = torch.stack(
            [
                self.model.encode(item.unsqueeze(0), self.scale).squeeze(0)
                for item in value
            ]
        )
        moments = torch.cat((encoded, torch.zeros_like(encoded)), dim=1)
        posterior = DiagonalGaussianDistribution(moments, deterministic=True)
        if return_dict:
            return AutoencoderKLOutput(latent_dist=posterior)
        return (posterior,)

    def decode(self, value: torch.Tensor, return_dict: bool = True):
        """Decode a latent batch and clamp pixels to ``[-1, 1]``."""
        self._prepare_codec_offload()
        decoded = torch.stack(
            [
                self.model.decode(item.unsqueeze(0), self.scale)
                .clamp_(-1, 1)
                .squeeze(0)
                for item in value
            ]
        )
        if return_dict:
            return DecoderOutput(sample=decoded)
        return (decoded,)

    def _prepare_codec_offload(self) -> None:
        """Activate an installed offload hook for methods bypassing forward."""
        hook = getattr(self, "_hf_hook", None)
        pre_forward = getattr(hook, "pre_forward", None)
        if pre_forward is not None:
            pre_forward(self)


__all__ = ["WanAutoencoderAdapterMixin", "WanAutoencoderConfig"]
