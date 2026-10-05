"""Inference-only streaming TAEW2.2 decoder for Wan 2.2 latents.

:class:`TAEW22StreamingDecoder` is a tiny autoencoder that
previews 48-channel Wan 2.2 latents with cached causal
state.  It is not a training-quality VAE38.

Input latents are BCTHW (Wan 2.2 layout).  Output is a
fast RGB preview, not the official decoder.

Use for interactive preview; final decode stays on
:class:`~..model.WanVideoVAE38`.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_WorkItem = namedtuple("_WorkItem", ("tensor", "block_index"))


def _conv(in_channels: int, out_channels: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value / 3) * 3


class _MemoryBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(torch.cat([value, past], dim=1)) + self.skip(value))


class _TemporalGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        count, channels, height, width = value.shape
        value = self.conv(value)
        return value.reshape(count * self.stride, channels, height, width)


class TAEW22StreamingDecoder(nn.Module):
    """Decode three Wan2.2 latents into one causal RGB frame block."""

    latent_channels = 48
    image_channels = 3
    patch_size = 2
    frames_to_trim = 3

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        widths = (256, 128, 64, 64)
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(self.latent_channels, widths[0]),
            nn.ReLU(inplace=True),
            _MemoryBlock(widths[0], widths[0]),
            _MemoryBlock(widths[0], widths[0]),
            _MemoryBlock(widths[0], widths[0]),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(widths[0], 1),
            _conv(widths[0], widths[1], bias=False),
            _MemoryBlock(widths[1], widths[1]),
            _MemoryBlock(widths[1], widths[1]),
            _MemoryBlock(widths[1], widths[1]),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(widths[1], 2),
            _conv(widths[1], widths[2], bias=False),
            _MemoryBlock(widths[2], widths[2]),
            _MemoryBlock(widths[2], widths[2]),
            _MemoryBlock(widths[2], widths[2]),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(widths[2], 2),
            _conv(widths[2], widths[3], bias=False),
            nn.ReLU(inplace=True),
            _conv(widths[3], self.image_channels * self.patch_size**2),
        )
        self._load_decoder(checkpoint_path)
        self.reset()

    def _load_decoder(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path).expanduser()
        if path.suffix.casefold() == ".safetensors":
            from safetensors.torch import load_file

            raw = load_file(str(path), device="cpu")
        else:
            raw = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        if not isinstance(raw, dict):
            raise ValueError("TAEW2.2 checkpoint must contain a state dictionary")
        state = {}
        for name, value in raw.items():
            normalized = name.removeprefix("module.")
            if normalized.startswith("decoder."):
                state[normalized.removeprefix("decoder.")] = value
        expected = self.decoder.state_dict()
        for index, layer in enumerate(self.decoder):
            if not isinstance(layer, _TemporalGrow):
                continue
            name = f"{index}.conv.weight"
            if name in state and state[name].shape[0] > expected[name].shape[0]:
                state[name] = state[name][-expected[name].shape[0] :]
        self.decoder.load_state_dict(state, strict=True)

    def reset(self) -> None:
        self._queue: list[_WorkItem] = []
        self._memory: list[torch.Tensor | None] = [None] * len(self.decoder)
        self._decoded_frames = 0

    def _advance(self) -> torch.Tensor | None:
        while self._queue:
            value, index = self._queue.pop(0)
            if index == len(self.decoder):
                return value.unsqueeze(1)
            layer = self.decoder[index]
            if isinstance(layer, _MemoryBlock):
                past = value * 0 if self._memory[index] is None else self._memory[index]
                output = layer(value, past)
                self._memory[index] = value
                self._queue.insert(0, _WorkItem(output, index + 1))
            elif isinstance(layer, _TemporalGrow):
                output = layer(value)
                count, channels, height, width = output.shape
                chunks = output.view(
                    count // layer.stride,
                    layer.stride * channels,
                    height,
                    width,
                ).chunk(layer.stride, dim=1)
                for chunk in reversed(chunks):
                    self._queue.insert(0, _WorkItem(chunk, index + 1))
            else:
                self._queue.insert(0, _WorkItem(layer(value), index + 1))
        return None

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[2] != self.latent_channels:
            raise ValueError(
                "TAEW2.2 latents must have [B,T,48,H,W] layout, "
                f"got {tuple(latents.shape)}"
            )
        first_block = self._decoded_frames == 0
        self._queue.extend(_WorkItem(value, 0) for value in latents.unbind(1))
        frames = []
        while self._queue:
            output = self._advance()
            if output is not None:
                if self.patch_size > 1:
                    output = F.pixel_shuffle(output, self.patch_size)
                frames.append(output.clamp_(0, 1))
                self._decoded_frames += 1
        if not frames:
            raise RuntimeError("TAEW2.2 decoder produced no frames")
        output = torch.cat(frames, dim=1)
        return output[:, self.frames_to_trim :] if first_block else output


__all__ = ["TAEW22StreamingDecoder"]
