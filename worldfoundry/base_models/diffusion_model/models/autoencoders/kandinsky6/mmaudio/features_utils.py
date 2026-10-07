# Audio feature / VAE facade.
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder import AutoEncoderModule
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder.distributions import DiagonalGaussianDistribution
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.mel_converter import get_mel_converter


class FeaturesUtils(nn.Module):
    """Mel autoencoder. Waveform synthesis lives on the separate vocoder module."""

    def __init__(
        self,
        *,
        tod_vae_ckpt: str | None = None,
        mode: Literal["44k"] = "44k",
        need_vae_encoder: bool = True,
        need_vae_decoder: bool = True,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(
            vae_ckpt_path=tod_vae_ckpt,
            mode=mode,
            need_vae_encoder=need_vae_encoder,
            need_vae_decoder=need_vae_decoder,
        )
        self.scaling_factor = scaling_factor
        self.downsample_factor = 1024

    def compile(self):
        """Compile latent decoding with ``torch.compile``."""
        self.decode = torch.compile(self.decode)

    def train(self, mode: bool = True) -> FeaturesUtils:
        """Keep the inference-only audio component in evaluation mode."""
        return super().train(False)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        """Encode a waveform into an audio latent distribution."""
        assert self.tod is not None, "VAE is not loaded"
        mel = self.mel_converter(x)
        return self.tod.encode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode audio latents into a mel-spectrogram."""
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def wrapped_decode(self, z):
        """Decode latents into a mel spectrogram for the separate vocoder."""
        return self.decode(z.to(dtype=self.dtype))

    @torch.no_grad()
    def wrapped_encode(self, audio):
        """Encode audio and return the mean latent through the forward hook."""
        dist = self.encode_audio(audio.to(dtype=self.dtype))
        return dist.mean
