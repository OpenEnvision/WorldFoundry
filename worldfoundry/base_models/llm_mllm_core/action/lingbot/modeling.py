# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Robbyant LingBot-VLA (Apache-2.0) and OpenFlamingo (MIT).
"""Conditioned norms and task-token heads with the released LingBot-VLA parameter hierarchy."""

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.encoders.lvdm.resampler import (
    FeedForward,
    PerceiverAttention,
)
from worldfoundry.core.nn.transformer.dit import _rms_norm_reference


class TaskTokenResampler(nn.Module):
    def __init__(
        self,
        dim_in: int = 768,
        dim_mid: int = 1024,
        dim_head: int = 64,
        dim_out: int = 1024,
        num_layers: int = 8,
        num_queries: int = 8,
        num_heads: int = 16,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class TaskTokenDepthHead(nn.Module):
    def __init__(
        self,
        proj_config: dict[str, int],
        llm_hidden_size: int = 4096,
        use_intermediate_depth: bool = False,
    ) -> None:
        super().__init__()

        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries = self.projector(llm_feats, queries)
        return queries


class AdaRMSNorm(nn.Module):
    """Float32 RMS and FiLM arithmetic, with the cast back after conditioning."""

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = _rms_norm_reference(hidden_states.float(), None, self.variance_epsilon, "fp32")
        hidden_states = self.weight * hidden_states
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        hidden_states = (1 + gamma.float()) * hidden_states + beta.float()
        return hidden_states.to(input_dtype)
