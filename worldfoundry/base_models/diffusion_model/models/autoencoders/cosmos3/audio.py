"""Cosmos3 AVAE sound decoder (inference-only).

Decodes 64-channel sound latents to waveform. Used by :class:`~.component.Cosmos3MediaDecoder`."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Snake1d(nn.Module):
    """Learned periodic activation used by the AVAE vocoder."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)


class Cosmos3AudioResidualUnit(nn.Module):
    def __init__(self, dimension: int, dilation: int) -> None:
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=padding))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        padding = (hidden_states.shape[-1] - output.shape[-1]) // 2
        if padding > 0:
            hidden_states = hidden_states[..., padding:-padding]
        return hidden_states + output


class Cosmos3AudioDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            )
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(output_dim, 1)
        self.res_unit2 = Cosmos3AudioResidualUnit(output_dim, 3)
        self.res_unit3 = Cosmos3AudioResidualUnit(output_dim, 9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class Cosmos3AudioDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: tuple[int, ...],
        channel_multiples: tuple[int, ...],
    ) -> None:
        super().__init__()
        multiples = (1, *channel_multiples)
        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * multiples[-1], kernel_size=7, padding=3))
        self.block = nn.ModuleList(
            Cosmos3AudioDecoderBlock(
                input_dim=channels * multiples[len(upsampling_ratios) - index],
                output_dim=channels * multiples[len(upsampling_ratios) - index - 1],
                stride=stride,
            )
            for index, stride in enumerate(upsampling_ratios)
        )
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return self.conv2(self.snake1(hidden_states))


class Cosmos3AVAEAudioDecoder(nn.Module):
    """Checkpoint-shaped AVAE module containing only the shipped decoder weights."""

    def __init__(
        self,
        vocoder_input_dim: int = 64,
        dec_dim: int = 320,
        dec_c_mults: tuple[int, ...] = (1, 2, 4, 8, 16),
        dec_strides: tuple[int, ...] = (2, 4, 5, 6, 8),
        dec_out_channels: int = 2,
        sampling_rate: int = 48000,
        **unused_config,
    ) -> None:
        super().__init__()
        del unused_config
        self.sampling_rate = int(sampling_rate)
        self.decoder = Cosmos3AudioDecoder(
            channels=int(dec_dim),
            input_channels=int(vocoder_input_dim),
            audio_channels=int(dec_out_channels),
            upsampling_ratios=tuple(reversed(dec_strides)),
            channel_multiples=tuple(dec_c_mults),
        )

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        waveform = self.decoder(latents).clamp(-1.0, 1.0)
        return waveform.squeeze(0) if squeeze else waveform


__all__ = ["Cosmos3AVAEAudioDecoder"]
