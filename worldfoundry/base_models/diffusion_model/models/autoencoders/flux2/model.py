"""FLUX.2 2-D autoencoder network (encode/decode stills).

Checkpoint-compatible CNN. :func:`~.component.encode_video_batch_refs` folds video frames through this module."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class Flux2AutoencoderConfig:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 32


def _swish(values: Tensor) -> Tensor:
    return values * torch.sigmoid(values)


class AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm(values)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)

        batch, channels, height, width = query.shape
        query = query.flatten(2).transpose(1, 2).unsqueeze(1)
        key = key.flatten(2).transpose(1, 2).unsqueeze(1)
        value = value.flatten(2).transpose(1, 2).unsqueeze(1)
        output = nn.functional.scaled_dot_product_attention(query, key, value)
        output = output.squeeze(1).transpose(1, 2).reshape(batch, channels, height, width)
        return values + self.proj_out(output)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, values: Tensor) -> Tensor:
        residual = self.conv1(_swish(self.norm1(values)))
        residual = self.conv2(_swish(self.norm2(residual)))
        if self.in_channels != self.out_channels:
            values = self.nin_shortcut(values)
        return values + residual


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(nn.functional.pad(values, (0, 1, 0, 1)))


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for level in range(self.num_resolutions):
            blocks = nn.ModuleList()
            attention = nn.ModuleList()
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out

            down = nn.Module()
            down.block = blocks
            down.attn = attention
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                resolution //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, padding=1)

    def forward(self, values: Tensor) -> Tensor:
        hidden_states = [self.conv_in(values)]
        for level in range(self.num_resolutions):
            for block in self.down[level].block:
                hidden_states.append(block(hidden_states[-1]))
            if level != self.num_resolutions - 1:
                hidden_states.append(self.down[level].downsample(hidden_states[-1]))

        values = self.mid.block_1(hidden_states[-1])
        values = self.mid.attn_1(values)
        values = self.mid.block_2(values)
        values = self.conv_out(_swish(self.norm_out(values)))
        return self.quant_conv(values)


class Flux2Autoencoder(nn.Module):
    """FLUX.2 2-D AE with official encode/decode and latent scale."""
    def __init__(self, params: Flux2AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.ps = (2, 2)
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=1e-4,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def encode(self, images: Tensor) -> Tensor:
        mean = torch.chunk(self.encoder(images), 2, dim=1)[0]
        latents = nn.functional.pixel_unshuffle(mean, downscale_factor=self.ps[0])
        self.bn.eval()
        return self.bn(latents)


__all__ = ["Flux2Autoencoder", "Flux2AutoencoderConfig"]
