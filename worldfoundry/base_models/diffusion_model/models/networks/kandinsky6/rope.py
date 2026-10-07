from __future__ import annotations

import torch
from torch import nn, Tensor

from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.tensors import get_freqs


class RoPE1D(nn.Module):
    """1-D Rotary Position Embedding — used for text and audio sequences."""

    def __init__(
        self,
        dim: int,
        max_pos: int = 2048,
        max_period: float = 10000.0,
        freqs_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        self.freqs_scaling = freqs_scaling
        freq = get_freqs(dim // 2, max_period) * freqs_scaling
        self.register_buffer("args", torch.outer(torch.arange(max_pos, dtype=freq.dtype), freq), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        # RoPE tables are fp32; keep trig in fp32.
        args = self.args[pos]  # (seq_len, dim//2)
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)

    def reset_parameters(self) -> None:
        freq = get_freqs(self.dim // 2, self.max_period).to(self.args.device) * self.freqs_scaling
        self.args = torch.outer(torch.arange(self.max_pos, dtype=freq.dtype, device=freq.device), freq)


class RoPE3D(nn.Module):
    """3-D Rotary Position Embedding — used for video spatial-temporal tokens (T, H, W)."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        for i, (d, mp) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(d // 2, max_period)
            self.register_buffer(f"args_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype), freq), persistent=False)

    def forward(
        self,
        shape: tuple,
        pos: list[Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        T, H, W = shape
        args_t = getattr(self, "args_0")[pos[0]] / scale_factor[0]  # (T, d//2)
        args_h = getattr(self, "args_1")[pos[1]] / scale_factor[1]  # (H, d//2)
        args_w = getattr(self, "args_2")[pos[2]] / scale_factor[2]  # (W, d//2)

        args = torch.cat(
            [
                args_t.view(T, 1, 1, -1).expand(T, H, W, -1),
                args_h.view(1, H, 1, -1).expand(T, H, W, -1),
                args_w.view(1, 1, W, -1).expand(T, H, W, -1),
            ],
            dim=-1,
        )
        cos, sin = torch.cos(args), torch.sin(args)
        rope = torch.stack([cos, -sin, sin, cos], dim=-1)  # (T, H, W, total_dim, 4)
        rope = rope.view(*rope.shape[:-1], 2, 2)           # (T, H, W, total_dim, 2, 2)
        return rope.unsqueeze(-4)                           # (T, H, W, 1, total_dim, 2, 2)

    def reset_parameters(self) -> None:
        for i, (d, mp) in enumerate(zip(self.axes_dims, self.max_pos)):
            freq = get_freqs(d // 2, self.max_period).to(getattr(self, f"args_{i}").device)
            setattr(self, f"args_{i}", torch.outer(torch.arange(mp, dtype=freq.dtype, device=freq.device), freq))
