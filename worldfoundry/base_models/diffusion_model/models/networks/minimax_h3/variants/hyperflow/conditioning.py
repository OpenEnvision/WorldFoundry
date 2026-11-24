# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
# Adapted from Video-Rebirth/hyperflow at 1dd2f342aba5ab51da02b62885939655e8e268da.
"""Released two-time conditioning and unmerged LoRA projection arithmetic."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

from ...dit import MiniMaxH3TimeEmbedder


class HyperFlowTwoTimeEmbedder(nn.Module):
    """Blend canonical H3 embeddings of a step's start and endpoint in FP32."""

    def __init__(self, base: MiniMaxH3TimeEmbedder, gate: float) -> None:
        super().__init__()
        self.base = base
        self.endpoint = copy.deepcopy(base)
        self.gate = float(gate)
        self._endpoint_timesteps: torch.Tensor | None = None

    @contextmanager
    def endpoint_context(self, endpoint: torch.Tensor) -> Iterator[None]:
        self._endpoint_timesteps = endpoint
        try:
            yield
        finally:
            self._endpoint_timesteps = None

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if self._endpoint_timesteps is None:
            raise RuntimeError("HyperFlow requires endpoint timesteps for each denoiser forward.")
        if self._endpoint_timesteps.shape != timestep.shape:
            raise ValueError("Every HyperFlow timestep must have one endpoint.")
        current = self.base(timestep)
        endpoint = self.endpoint(self._endpoint_timesteps.to(timestep.device))
        return current + self.gate * (endpoint - current)


class HyperFlowLinear(nn.Module):
    """PEFT's inference arithmetic without merging or changing adapter precision."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.scaling = float(alpha) / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(value)
        update = self.lora_B(self.lora_A(value.to(self.lora_A.weight.dtype)))
        return (result + update * self.scaling).to(result.dtype)


class HyperFlowQKVProjection(nn.Module):
    """Keep the release's separate Q/K/V matmuls inside canonical H3 attention."""

    def __init__(self, fused: nn.Linear, *, head_dim: int | None = None) -> None:
        super().__init__()
        self.head_dim = head_dim
        inner = fused.out_features // 3
        for index, name in enumerate(("q", "k", "v")):
            projection = nn.Linear(
                fused.in_features,
                inner,
                bias=fused.bias is not None,
                device=fused.weight.device,
                dtype=fused.weight.dtype,
            )
            weight = (
                fused.weight.chunk(3)[index]
                if head_dim is None
                else fused.weight.view(-1, 3, head_dim, fused.in_features)[:, index].reshape(inner, -1)
            )
            projection.weight = nn.Parameter(weight, requires_grad=False)
            if fused.bias is not None:
                bias = (
                    fused.bias.chunk(3)[index]
                    if head_dim is None
                    else fused.bias.view(-1, 3, head_dim)[:, index].reshape(inner)
                )
                projection.bias = nn.Parameter(bias, requires_grad=False)
            setattr(self, name, projection)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query, key, values = self.q(value), self.k(value), self.v(value)
        if self.head_dim is None:
            return torch.cat((query, key, values), dim=-1)
        return torch.cat(
            [tensor.unflatten(-1, (-1, self.head_dim)) for tensor in (query, key, values)], dim=-1
        ).flatten(-2)


class HyperFlowSwiGLUProjection(nn.Module):
    """The release projects up|gate; canonical H3 consumes gate|up activations."""

    def __init__(self, projection: nn.Linear) -> None:
        super().__init__()
        self.proj = projection

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        up, gate = self.proj(value).chunk(2, dim=-1)
        return torch.cat((gate, up), dim=-1)
