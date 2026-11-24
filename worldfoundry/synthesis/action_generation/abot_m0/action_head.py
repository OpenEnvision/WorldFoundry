# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML flow-matching action expert used by ABot-M0 checkpoints.

The module keeps the released parameter names while removing all training-only
code.  The DiT implementation is intentionally local so loading a policy never
imports an adjacent source checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

from worldfoundry.core.nn.blocks.layers import TwoLayerReLU as MLP


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 2:
            raise ValueError(f"timesteps must be [B, T], got {tuple(timesteps.shape)}")
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            raise ValueError("embedding_dim must be at least two")
        exponent = -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
        frequencies = timesteps.float().unsqueeze(-1) * exponent.exp()
        encoded = torch.cat((frequencies.sin(), frequencies.cos()), dim=-1)
        if encoded.shape[-1] < self.embedding_dim:
            encoded = F.pad(encoded, (0, self.embedding_dim - encoded.shape[-1]))
        return encoded


class ActionEncoder(nn.Module):
    """Released three-layer action/time encoder (names are checkpoint ABI)."""

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch:
            raise ValueError(f"timesteps must be [B], got {tuple(timesteps.shape)} for B={batch}")
        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(timesteps[:, None].expand(-1, horizon)).to(action_embedding.dtype)
        return self.layer3(F.silu(self.layer2(torch.cat((action_embedding, time_embedding), dim=-1))))


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype or torch.float32
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # ``Module.to(dtype=...)`` casts the embedder parameters but not this
        # plain Python dtype attribute.  Derive the runtime dtype from the
        # loaded checkpoint weights so bf16/fp16 inference cannot feed fp32
        # activations into a lower-precision Linear layer.
        weight = self.timestep_embedder.linear_1.weight
        projected = self.time_proj(timesteps).to(device=weight.device, dtype=weight.dtype)
        return self.timestep_embedder(projected)


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, norm_elementwise_affine: bool = False, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.chunk_dim = 0
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(self, value: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(value) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type
        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError("num_positional_embeddings is required with positional embeddings")
        if positional_embeddings == "sinusoidal":
            from diffusers.models.embeddings import SinusoidalPositionalEmbedding

            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None
        self.norm1 = (
            AdaLayerNorm(dim, norm_elementwise_affine=False, norm_eps=norm_eps)
            if norm_type == "ada_norm"
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=True,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout, bias=True)
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden_states, temb) if self.norm_type == "ada_norm" else self.norm1(hidden_states)
        if self.pos_embed is not None:
            normalized = self.pos_embed(normalized)
        attention = self.attn1(
            normalized,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if self.final_dropout is not None:
            attention = self.final_dropout(attention)
        hidden_states = hidden_states + attention
        return hidden_states + self.ff(self.norm3(hidden_states))


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False
        self.timestep_encoder = TimestepEncoder(self.inner_dim, compute_dtype=compute_dtype)
        blocks = []
        for index in range(num_layers):
            self_attention = index % 2 == 1 and interleave_self_attention
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if self_attention else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        time_embedding = self.timestep_encoder(timestep).to(hidden_states.dtype)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for index, block in enumerate(self.transformer_blocks):
            self_attention = index % 2 == 1 and bool(self.config.interleave_self_attention)
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None if self_attention else encoder_hidden_states,
                encoder_attention_mask=None if self_attention else encoder_attention_mask,
                temb=time_embedding,
            )
            all_hidden_states.append(hidden_states)
        shift, scale = self.proj_out_1(F.silu(time_embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)
        return (output, all_hidden_states) if return_all_hidden_states else output


@dataclass(frozen=True)
class ActionHeadConfig:
    action_model_type: str = "DiT-B"
    hidden_size: int = 1024
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    action_dim: int = 14
    state_dim: int = 14
    future_action_window_size: int = 49
    num_inference_timesteps: int = 4
    num_timestep_buckets: int = 1000
    num_target_vision_tokens: int = 32
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    t_eps: float = 0.05
    diffusion_model_cfg: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActionHeadConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})

    @property
    def action_horizon(self) -> int:
        return self.future_action_window_size + 1


_DIT_VARIANTS = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowmatchingActionHead(nn.Module):
    """Inference half of the released AML sample-prediction flow head."""

    def __init__(self, config: ActionHeadConfig) -> None:
        super().__init__()
        if config.action_model_type not in _DIT_VARIANTS:
            raise ValueError(f"unsupported action expert {config.action_model_type!r}")
        variant = _DIT_VARIANTS[config.action_model_type]
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = int(variant["input_embedding_dim"])
        model_config = {**variant, **dict(config.diffusion_model_cfg or {})}
        self.model = DiT(**model_config)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_encoder = (
            MLP(config.state_dim, config.hidden_size, self.input_embedding_dim) if config.state_dim else None
        )
        self.action_encoder = ActionEncoder(config.action_dim, self.input_embedding_dim)
        self.action_decoder = MLP(self.model.config.output_dim, config.hidden_size, config.action_dim)
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None
        self.num_timestep_buckets = config.num_timestep_buckets
        self.t_eps = config.t_eps
        self.config = config

    @torch.inference_mode()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            device=vl_embs.device,
            dtype=vl_embs.dtype,
            generator=generator,
        )
        steps = int(num_inference_steps or self.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        state_features = self.state_encoder(state) if state is not None and self.state_encoder is not None else None
        for step in range(steps):
            time_value = step / float(steps)
            discrete_time = int(time_value * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), discrete_time, dtype=torch.long, device=actions.device)
            action_features = self.action_encoder(actions, timesteps)
            if self.position_embedding is not None:
                positions = torch.arange(action_features.shape[1], device=actions.device)
                action_features = action_features + self.position_embedding(positions)[None]
            future = self.future_tokens.weight[None].expand(batch_size, -1, -1)
            policy_tokens = torch.cat(
                (state_features, future, action_features) if state_features is not None else (future, action_features),
                dim=1,
            )
            hidden = self.model(
                hidden_states=policy_tokens,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
            )
            predicted_actions = self.action_decoder(hidden)[:, -self.action_horizon :]
            velocity = (predicted_actions - actions) / max(1.0 - time_value, self.t_eps)
            actions = actions + velocity / steps
        return actions


__all__ = ["ActionHeadConfig", "DiT", "FlowmatchingActionHead"]
