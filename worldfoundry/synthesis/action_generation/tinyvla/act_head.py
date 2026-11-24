"""Inference-only ACT/CVAE action head used by TinyVLA checkpoints."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from worldfoundry.base_models.llm_mllm_core.action.transformer import TransformerEncoder

from .transformer import Transformer, TransformerEncoderLayer


def _sinusoid_encoding_table(n_position: int, hidden_dim: int) -> torch.Tensor:
    def row(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (index // 2) / hidden_dim) for index in range(hidden_dim)]

    table = np.asarray([row(position) for position in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


def _position_encoding_1d(source: torch.Tensor) -> torch.Tensor:
    sequence_length, hidden_dim = source.shape[1:]
    device = source.device
    compute_dtype = torch.float32
    position = torch.arange(sequence_length, dtype=compute_dtype, device=device).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=compute_dtype, device=device) * (-math.log(10000.0) / hidden_dim)
    )
    encoding = torch.zeros((sequence_length, hidden_dim), dtype=compute_dtype, device=device)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding.to(dtype=source.dtype)


class DETRVAEHead(nn.Module):
    """ACT decoder whose module names match released TinyVLA checkpoints."""

    def __init__(
        self,
        transformer: Transformer,
        encoder: TransformerEncoder,
        state_dim: int,
        num_queries: int,
        camera_names: list[str],
        action_dim: int,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq = False
        self.vq_class = 0
        self.vq_dim = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.input_proj_env_state = nn.Linear(state_dim, hidden_dim)
        self.pos = nn.Embedding(2, hidden_dim)
        self.backbones = None

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("pos_table", _sinusoid_encoding_table(2 + num_queries, hidden_dim))
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        del vq_sample
        batch_size = qpos.shape[0]
        if actions is None:
            latent = torch.zeros((batch_size, self.latent_dim), dtype=qpos.dtype, device=qpos.device)
            return self.latent_out_proj(latent), None, None, None, None

        action_embed = self.encoder_action_proj(actions)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        encoder_input = torch.cat((cls_embed, qpos_embed, action_embed), dim=1).permute(1, 0, 2)
        prefix_mask = torch.zeros((batch_size, 2), dtype=torch.bool, device=qpos.device)
        padding_mask = torch.cat((prefix_mask, is_pad), dim=1)
        position = self.pos_table.detach().permute(1, 0, 2)
        encoded = self.encoder(encoder_input, pos=position, src_key_padding_mask=padding_mask)[0]
        latent_info = self.latent_proj(encoded)
        mean, log_variance = latent_info.chunk(2, dim=-1)
        standard_deviation = (log_variance / 2).exp()
        sample = mean + standard_deviation * torch.randn_like(standard_deviation)
        return self.latent_out_proj(sample), None, None, mean, log_variance

    def forward(self, qpos=None, env_state=None, hidden_states=None, actions=None, is_pad=None, vq_sample=None):
        del env_state
        latent_input, probabilities, binaries, mean, log_variance = self.encode(qpos, actions, is_pad, vq_sample)
        proprio_input = self.input_proj_robot_state(qpos)
        position = _position_encoding_1d(hidden_states)
        decoded = self.transformer(
            hidden_states,
            None,
            self.query_embed.weight,
            position,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
        )[-1]
        actions_hat = self.action_head(decoded)
        padding_hat = self.is_pad_head(decoded)
        return actions_hat, padding_hat, [mean, log_variance], probabilities, binaries


def build_act_head(spec: dict, *, state_dim: int) -> DETRVAEHead:
    hidden_dim = int(spec["hidden_dim"])
    dropout = float(spec["dropout"])
    heads = int(spec["nheads"])
    feedforward = int(spec["dim_feedforward"])
    encoder_layers = int(spec["enc_layers"])
    decoder_layers = int(spec["dec_layers"])
    pre_norm = bool(spec["pre_norm"])
    encoder_layer = TransformerEncoderLayer(hidden_dim, heads, feedforward, dropout, "relu", pre_norm)
    encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
    encoder = TransformerEncoder(encoder_layer, encoder_layers, encoder_norm)
    transformer = Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=heads,
        dim_feedforward=feedforward,
        num_encoder_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )
    return DETRVAEHead(
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=int(spec["chunk_size"]),
        camera_names=list(spec.get("camera_names") or []),
        action_dim=int(spec["action_dim"]),
    )


__all__ = ["DETRVAEHead", "build_act_head"]
