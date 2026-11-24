# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Action repeat is inspired by CogACT.
"""Inference-only flow-matching action head for Mem-0."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.lvdm.blocks import nonlinearity as swish
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamzero.action_encoder import (
    SinusoidalPositionalEncoding,
)
from worldfoundry.core.nn.blocks.layers import TwoLayerReLU as MLP

from .cross_attention_dit import DiT


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("Mem-0 timesteps must have shape (batch_size,)")
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)
        hidden = swish(self.layer2(torch.cat([action_embedding, time_embedding], dim=-1)))
        return self.layer3(hidden)


class FlowmatchingActionHead(nn.Module):
    """Checkpoint-compatible Euler flow sampler."""

    def __init__(self, config: Any, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        if int(config.hidden_size) != hidden_size:
            raise ValueError(f"Mem-0 action hidden_size {config.hidden_size} does not match backbone {hidden_size}")
        action_model_config = {
            "input_embedding_dim": int(config.input_embedding_dim),
            "attention_head_dim": int(config.attention_head_dim),
            "num_attention_heads": int(config.num_attention_heads),
        }
        self.input_embedding_dim = action_model_config["input_embedding_dim"]
        diffusion_model_config = {**action_model_config, **dict(config.diffusion_model_cfg)}
        self.model = DiT(**diffusion_model_config)
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.num_inference_timesteps = int(config.num_inference_timesteps)
        if self.num_inference_timesteps < 1:
            raise ValueError("Mem-0 num_inference_timesteps must be positive")

        self.state_encoder = (
            MLP(
                input_dim=int(config.state_dim),
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                int(config.max_seq_len),
                self.input_embedding_dim,
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.num_timestep_buckets = int(config.num_timestep_buckets)
        self.config = config

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        step_size = 1.0 / self.num_inference_timesteps
        state_features = self.state_encoder(state) if state is not None else None
        for step in range(self.num_inference_timesteps):
            continuous_time = step / float(self.num_inference_timesteps)
            discretized_time = int(continuous_time * self.num_timestep_buckets)
            timesteps = torch.full(
                (batch_size,),
                fill_value=discretized_time,
                device=device,
            )
            action_features = self.action_encoder(actions, timesteps)
            if self.config.add_pos_embed:
                position_ids = torch.arange(action_features.shape[1], device=device)
                action_features = action_features + self.position_embedding(position_ids).unsqueeze(0)
            state_action_features = (
                torch.cat((state_features, action_features), dim=1) if state_features is not None else action_features
            )
            model_output = self.model(
                hidden_states=state_action_features,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
            )
            predicted_velocity = self.action_decoder(model_output)[:, -self.action_horizon :]
            actions = actions + step_size * predicted_velocity
        return actions

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype


__all__ = ["FlowmatchingActionHead"]
