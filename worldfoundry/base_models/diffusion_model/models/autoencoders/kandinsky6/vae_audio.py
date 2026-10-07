"""Audio VAE and BigVGAN vocoder as separate modules."""
from __future__ import annotations

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.ext.autoencoder.autoencoder import load_bigvgan_v2
from worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6.mmaudio.features_utils import FeaturesUtils


def build_audio_vae(
    *,
    tod_vae_ckpt: str,
    mode: str = "44k",
    need_vae_encoder: bool = True,
    need_vae_decoder: bool = True,
    scaling_factor: float = 1.0,
    device: str | torch.device = "cuda",
) -> FeaturesUtils:
    """Load the mel autoencoder and move it to ``device``."""
    audio_vae = FeaturesUtils(
        tod_vae_ckpt=tod_vae_ckpt,
        mode=mode,  # type: ignore[arg-type]
        need_vae_encoder=need_vae_encoder,
        need_vae_decoder=need_vae_decoder,
        scaling_factor=scaling_factor,
    )
    return audio_vae.to(device).eval()


def build_vocoder(
    *,
    ckpt: str,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Load BigVGAN and move it to ``device``.

    The vocoder turns the mel spectrogram from ``build_audio_vae`` into a waveform.
    Its checkpoint is not part of the audio VAE weights.
    """
    vocoder = load_bigvgan_v2(ckpt).eval()
    for parameter in vocoder.parameters():
        parameter.requires_grad = False
    return vocoder.to(device)
