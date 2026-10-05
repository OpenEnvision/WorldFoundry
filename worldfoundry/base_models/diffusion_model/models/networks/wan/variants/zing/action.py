# Adapted from seedleap/zing-world-model (Apache-2.0).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ActionConfig

PRIMITIVE_BIT_WIDTH = 8


def action_sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("action embedding dimension must be even")
    half = dim // 2
    values = position.to(torch.float64).mul(1000.0)
    frequencies = torch.pow(10000, -torch.arange(half, device=values.device).to(values).div(half))
    sinusoid = torch.outer(values, frequencies)
    return torch.cat((torch.cos(sinusoid) - 1, torch.sin(sinusoid)), dim=1)


class CausalActionTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.norm = nn.LayerNorm(out_channels, bias=bias)
        self.causal_pad = kernel_size - 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (self.causal_pad, 0))
        value = self.conv(value).transpose(1, 2)
        value = self.norm(value)
        return F.silu(value).transpose(1, 2)


class ActionConditioner(nn.Module):
    def __init__(self, config: ActionConfig, dim: int):
        super().__init__()
        if config.embed_dim % 8:
            raise ValueError("action.embed_dim must be divisible by 8")
        hidden_bias = bool(config.non_proj_bias)
        self.embed_dim = config.embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim * 2, bias=hidden_bias),
            nn.SiLU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim * 2, bias=hidden_bias),
        )
        self.encode_1 = CausalActionTemporalBlock(
            config.embed_dim * 2, config.hidden_dim, config.kernel_size, hidden_bias
        )
        self.encode_2 = CausalActionTemporalBlock(config.hidden_dim, config.hidden_dim, config.kernel_size, hidden_bias)
        self.proj = nn.Linear(config.hidden_dim, dim, bias=True)
        if not self.proj.weight.is_meta:
            nn.init.normal_(self.fuse[0].weight, std=0.02)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def frame_states(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 4 or action.shape[-1] != PRIMITIVE_BIT_WIDTH:
            raise ValueError("action must have shape [B, F, S, 8]")
        weights = action.to(device=self.proj.weight.device, dtype=torch.float32)
        if not torch.isfinite(weights).all() or torch.any((weights < 0) | (weights > 1)):
            raise ValueError("action values must be finite and in [0, 1]")
        weights = weights.mean(dim=2)
        hidden = action_sinusoidal_embedding_1d(self.embed_dim // 4, weights.reshape(-1))
        hidden = hidden.to(weights.dtype).reshape(*weights.shape[:-1], self.embed_dim * 2)
        hidden = hidden.to(self.fuse[0].weight.dtype)
        hidden = hidden + self.fuse(hidden)
        hidden = self.encode_1(hidden.transpose(1, 2))
        hidden = self.encode_2(hidden).transpose(1, 2)
        return self.proj(hidden.to(self.proj.weight.dtype))

    def add_action_to_tokens(
        self, tokens: torch.Tensor, action: torch.Tensor, grid_sizes: torch.Tensor, align_to_end: bool = True
    ) -> torch.Tensor:
        states = self.frame_states(action).to(tokens.dtype)
        residuals = []
        for sample_states, grid in zip(states, grid_sizes.tolist()):
            frames, height, width = (int(value) for value in grid)
            if sample_states.shape[0] < frames:
                raise ValueError("action history is shorter than the latent block")
            selected = sample_states[-frames:] if align_to_end else sample_states[:frames]
            residual = selected[:, None].expand(-1, height * width, -1).reshape(1, frames * height * width, -1)
            if residual.shape[1] != tokens.shape[1]:
                raise ValueError("action token count does not match the latent token count")
            residuals.append(residual)
        return tokens + torch.cat(residuals, dim=0)
