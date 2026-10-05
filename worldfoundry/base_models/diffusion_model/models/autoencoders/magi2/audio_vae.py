# Ported from SandAI MAGI-2-preview (Apache-2.0): Stable-Audio-Open-1.0 Oobleck audio VAE.
"""Stable Audio Open VAE used by MAGI-2 audio recipes.

Self-contained encoder/decoder definitions. Public ``encode`` / ``decode`` operate on waveform batches. Large CNN stacks are implementation detail."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


__all__ = ["AudioAutoencoder", "create_model_from_config"]


# ---------------------------------------------------------------------------
#  Activations
# ---------------------------------------------------------------------------

def _snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True,
                 alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha, beta = torch.exp(alpha), torch.exp(beta)
        return _snake_beta(x, alpha, beta)


def _get_activation(activation: Literal["elu", "snake", "none"], channels: int | None = None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


# ---------------------------------------------------------------------------
#  Weight-normed convolutions
# ---------------------------------------------------------------------------

def wn_conv1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# ---------------------------------------------------------------------------
#  Bottleneck
# ---------------------------------------------------------------------------

class VAEBottleneck(nn.Module):
    def encode(self, x: torch.Tensor, return_info: bool = False, **kwargs):
        mean, scale = x.chunk(2, dim=1)
        stdev = F.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        if return_info:
            return latents, {"kl": kl}
        return latents

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
#  Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, use_snake: bool = False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            _get_activation(act, channels=out_channels),
            wn_conv1d(in_channels, out_channels, kernel_size=7, dilation=dilation, padding=padding),
            _get_activation(act, channels=out_channels),
            wn_conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool = False):
        super().__init__()
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, 3, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, 9, use_snake=use_snake),
            _get_activation(act, channels=in_channels),
            wn_conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride,
                     padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 use_snake: bool = False, use_nearest_upsample: bool = False):
        super().__init__()
        act = "snake" if use_snake else "elu"
        if use_nearest_upsample:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                wn_conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=1,
                         bias=False, padding="same"),
            )
        else:
            upsample = wn_conv_transpose1d(in_channels, out_channels, kernel_size=2 * stride,
                                         stride=stride, padding=math.ceil(stride / 2))
        self.layers = nn.Sequential(
            _get_activation(act, channels=in_channels),
            upsample,
            ResidualUnit(out_channels, out_channels, 1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 9, use_snake=use_snake),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
#  Encoder / Decoder
# ---------------------------------------------------------------------------

class OobleckEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, channels: int = 128, latent_dim: int = 32,
                 c_mults: list[int] = [1, 2, 4, 8], strides: list[int] = [2, 4, 8, 8],
                 use_snake: bool = False, **_kwargs):
        super().__init__()
        c_mults = [1] + list(c_mults)
        layers: list[nn.Module] = [wn_conv1d(in_channels, c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels,
                                       strides[i], use_snake=use_snake))
        act = "snake" if use_snake else "elu"
        layers.extend([
            _get_activation(act, channels=c_mults[-1] * channels),
            wn_conv1d(c_mults[-1] * channels, latent_dim, kernel_size=3, padding=1),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels: int = 2, channels: int = 128, latent_dim: int = 32,
                 c_mults: list[int] = [1, 2, 4, 8], strides: list[int] = [2, 4, 8, 8],
                 use_snake: bool = False, use_nearest_upsample: bool = False,
                 final_tanh: bool = True, **_kwargs):
        super().__init__()
        c_mults = [1] + list(c_mults)
        layers: list[nn.Module] = [wn_conv1d(latent_dim, c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(DecoderBlock(c_mults[i] * channels, c_mults[i - 1] * channels,
                                       strides[i - 1], use_snake=use_snake,
                                       use_nearest_upsample=use_nearest_upsample))
        act = "snake" if use_snake else "elu"
        layers.extend([
            _get_activation(act, channels=c_mults[0] * channels),
            wn_conv1d(c_mults[0] * channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity(),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
#  AudioAutoencoder
# ---------------------------------------------------------------------------

class AudioAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int,
                 downsampling_ratio: int, sample_rate: int, io_channels: int = 2,
                 bottleneck: nn.Module | None = None, in_channels: int | None = None,
                 out_channels: int | None = None, soft_clip: bool = False):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels or io_channels
        self.out_channels = out_channels or io_channels
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.soft_clip = soft_clip

    def encode(self, audio: torch.Tensor, skip_bottleneck: bool = False,
               return_info: bool = False, **kwargs) -> torch.Tensor:
        info: dict[str, Any] = {}
        latents = self.encoder(audio)
        info["pre_bottleneck_latents"] = latents
        if self.bottleneck is not None and not skip_bottleneck:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)
        if return_info:
            return latents, info
        return latents

    def decode(self, latents: torch.Tensor, skip_bottleneck: bool = False, **kwargs) -> torch.Tensor:
        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
        if self.soft_clip:
            decoded = torch.tanh(decoded)
        return decoded


# Alias matching the destination package naming convention.
Magi2AudioVAE = AudioAutoencoder


# ---------------------------------------------------------------------------
#  Factory helpers
# ---------------------------------------------------------------------------

def _create_encoder(cfg: dict) -> nn.Module:
    cfg_type = cfg.get("type")
    assert cfg_type == "oobleck", f"Only oobleck encoder supported, got {cfg_type}"
    enc = OobleckEncoder(**cfg["config"])
    if not cfg.get("requires_grad", True):
        for p in enc.parameters():
            p.requires_grad = False
    return enc


def _create_decoder(cfg: dict) -> nn.Module:
    cfg_type = cfg.get("type")
    assert cfg_type == "oobleck", f"Only oobleck decoder supported, got {cfg_type}"
    dec = OobleckDecoder(**cfg["config"])
    if not cfg.get("requires_grad", True):
        for p in dec.parameters():
            p.requires_grad = False
    return dec


def _create_bottleneck(cfg: dict) -> nn.Module:
    cfg_type = cfg.get("type")
    assert cfg_type == "vae", f"Only vae bottleneck supported, got {cfg_type}"
    bn = VAEBottleneck()
    if not cfg.get("requires_grad", True):
        for p in bn.parameters():
            p.requires_grad = False
    return bn


def create_model_from_config(config: dict) -> AudioAutoencoder:
    """Build an AudioAutoencoder from a Stable Audio Open config dict.

    The published ``stable-audio-open-1.0/model_config.json`` describes the
    complete diffusion model and nests the VAE at
    ``model.pretransform.config``.  Accept that on-disk format directly in
    addition to the smaller standalone-autoencoder format used by unit tests.
    """
    if config.get("model_type") != "autoencoder":
        pretransform = config.get("model", {}).get("pretransform", {})
        if pretransform.get("type") != "autoencoder":
            raise ValueError("Stable Audio config does not contain an autoencoder pretransform")
        config = {
            "model_type": "autoencoder",
            "sample_rate": config["sample_rate"],
            "model": pretransform["config"],
        }
    assert config.get("model_type") == "autoencoder"
    ae = config["model"]
    encoder = _create_encoder(ae["encoder"])
    decoder = _create_decoder(ae["decoder"])
    bottleneck = _create_bottleneck(ae["bottleneck"]) if ae.get("bottleneck") else None
    return AudioAutoencoder(
        encoder=encoder, decoder=decoder,
        latent_dim=ae["latent_dim"], downsampling_ratio=ae["downsampling_ratio"],
        sample_rate=config["sample_rate"], io_channels=ae["io_channels"],
        bottleneck=bottleneck,
        in_channels=ae.get("in_channels"), out_channels=ae.get("out_channels"),
        soft_clip=ae["decoder"].get("soft_clip", False),
    )
