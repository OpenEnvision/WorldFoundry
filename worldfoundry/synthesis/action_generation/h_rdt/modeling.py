# SPDX-License-Identifier: MPL-2.0
"""Inference-only H-RDT network.

This module is derived from the official H-RDT implementation at commit
8be02cb5e631898dfab12f9b6bbb77e32999547a.  Training losses, hub mixins, and
dataset code are intentionally excluded.  Parameter/module names remain
checkpoint-compatible with the official release.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.attention import native_sdpa_priority, scaled_dot_product_attention
from worldfoundry.core.utils.python.misc_utils import env_is_true


def _one_dimensional_embedding(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("H-RDT positional embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    values = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(values), np.cos(values)], axis=1)


def _nd_embedding(embed_dim: int, grid_sizes: tuple[int, ...]) -> np.ndarray:
    embedding = np.zeros(grid_sizes + (embed_dim,))
    for axis, size in enumerate(grid_sizes):
        if size <= 1:
            continue
        shape = [1] * len(grid_sizes) + [embed_dim]
        shape[axis] = -1
        embedding += _one_dimensional_embedding(embed_dim, np.arange(size)).reshape(shape)
    return embedding


def multimodal_position_embedding(
    embed_dim: int,
    lengths: Iterable[tuple[str, int | tuple[int, ...] | list[int]]],
) -> np.ndarray:
    """Reproduce the official multimodal sinusoidal position initialization."""

    modalities = OrderedDict(lengths)
    total = 0
    for name, length in modalities.items():
        if name == "image" and isinstance(length, (tuple, list)):
            total += int(np.prod([abs(int(item)) for item in length]))
        else:
            total += abs(int(length))

    modality_embedding = None
    if len(modalities) > 1:
        modality_embedding = _one_dimensional_embedding(embed_dim, np.arange(len(modalities)))

    result = np.zeros((total, embed_dim))
    start = 0
    for index, (name, length) in enumerate(modalities.items()):
        if name == "image" and isinstance(length, (tuple, list)):
            full_shape = tuple(abs(int(item)) for item in length)
            embedded_shape = tuple(int(item) if int(item) > 0 else 1 for item in length)
            current = np.zeros(full_shape + (embed_dim,))
            current += _nd_embedding(embed_dim, embedded_shape)
            current = current.reshape(-1, embed_dim)
        else:
            value = int(length)
            current = np.zeros((abs(value), embed_dim))
            if value > 1:
                current += _one_dimensional_embedding(embed_dim, np.arange(value))
        if modality_embedding is not None:
            current += modality_embedding[index]
        result[start : start + len(current)] = current
        start += len(current)
    return result


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(values.dtype) * self.weight


def _repeat_kv(values: torch.Tensor, repeats: int) -> torch.Tensor:
    batch, sequence, heads, head_dim = values.shape
    if repeats == 1:
        return values
    return (
        values[:, :, :, None, :]
        .expand(batch, sequence, heads, repeats, head_dim)
        .reshape(batch, sequence, heads * repeats, head_dim)
    )


def _attention_backends(requested: str | None, device: torch.device, *, has_mask: bool) -> tuple[str, ...] | None:
    normalized = str(requested or "auto").strip().lower().replace("_attention", "")
    if normalized in {"", "auto", "sdpa", "torch"}:
        preferred = native_sdpa_priority(device, has_mask=has_mask)
        return preferred or None
    aliases = {"mem_efficient": "efficient", "memory_efficient": "efficient"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"math", "efficient", "cudnn", "flash"}:
        raise ValueError(f"Unsupported H-RDT attention backend: {requested!r}")
    return (normalized,)


def _grouped_key_value(key, value, repeats, backends, mask=None, *, enabled=False):
    if not enabled:
        return _repeat_kv(key, repeats).transpose(1, 2), _repeat_kv(value, repeats).transpose(1, 2), False
    # cuDNN-only/mixed cuDNN+efficient requests keep the expanded-head
    # contract: these providers need not accept native GQA. Efficient alone
    # has the core's grouped-view compatibility path.
    legacy_provider = backends and backends != ("efficient",) and not {"flash", "math"}.intersection(backends)
    # Per-example masks already require head expansion in that compatibility
    # path, so retain the original preparation without an extra small V copy.
    batch_mask = backends == ("efficient",) and mask is not None and mask.ndim > 2
    if repeats > 1 and (legacy_provider or batch_mask):
        return _repeat_kv(key, repeats).transpose(1, 2), _repeat_kv(value, repeats).transpose(1, 2), False
    if repeats > 1:
        # wkv interleaves K and V in the last dimension. Materialize only the
        # small KV-head tensor so fused SDPA gets unit-stride feature channels.
        value = value.contiguous()
    return key.transpose(1, 2), value.transpose(1, 2), repeats > 1


class Attention(nn.Module):
    """Official grouped-query self-attention using WorldFoundry core SDPA."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        attention_backend: str = "auto",
        enable_gqa_optimization: bool | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = int(config["num_heads"])
        configured_kv = config.get("num_kv_heads")
        self.n_kv_heads = self.n_heads if configured_kv is None else int(configured_kv)
        if self.n_heads % self.n_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = int(config["hidden_size"])
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_size = self.hidden_size // self.n_heads
        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.hidden_size, bias=False)
        eps = float(config["norm_eps"])
        self.norm_q = RMSNorm(self.head_size, eps=eps)
        self.norm_k = RMSNorm(self.head_size, eps=eps)
        self.attn_scale = 1.0 / math.sqrt(self.head_size)
        self.attention_backend = attention_backend
        # Snapshot at construction; explicit False overrides a process opt-in.
        self.enable_gqa_optimization = (
            env_is_true("WORLDFOUNDRY_HRDT_GQA_OPTIMIZATION")
            if enable_gqa_optimization is None
            else enable_gqa_optimization
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        query = self.wq(values).view(batch, sequence, self.n_heads, self.head_size)
        key_value = self.wkv(values).view(batch, sequence, self.n_kv_heads, self.head_size, 2)
        key, value = key_value.unbind(-1)
        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query.transpose(1, 2)
        backends = _attention_backends(self.attention_backend, query.device, has_mask=False)
        key, value, enable_gqa = _grouped_key_value(
            key, value, self.n_rep, backends, enabled=self.enable_gqa_optimization
        )
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=self.attn_scale,
            enable_gqa=enable_gqa,
            backends=backends,
        )
        return self.wo(output.transpose(1, 2).contiguous().view(batch, sequence, -1))


class CrossAttention(nn.Module):
    """Official grouped-query cross-attention using WorldFoundry core SDPA."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        attention_backend: str = "auto",
        enable_gqa_optimization: bool | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = int(config["num_heads"])
        configured_kv = config.get("num_kv_heads")
        self.n_kv_heads = self.n_heads if configured_kv is None else int(configured_kv)
        if self.n_heads % self.n_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = int(config["hidden_size"])
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_size = self.hidden_size // self.n_heads
        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_size, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_size * 2, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, self.hidden_size, bias=False)
        eps = float(config["norm_eps"])
        self.norm_q = RMSNorm(self.head_size, eps=eps)
        self.norm_k = RMSNorm(self.head_size, eps=eps)
        self.attn_scale = 1.0 / math.sqrt(self.head_size)
        self.attention_backend = attention_backend
        self.enable_gqa_optimization = (
            env_is_true("WORLDFOUNDRY_HRDT_GQA_OPTIMIZATION")
            if enable_gqa_optimization is None
            else enable_gqa_optimization
        )

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = values.shape
        condition_length = int(condition.shape[1])
        query = self.wq(values).view(batch, sequence, self.n_heads, self.head_size)
        key_value = self.wkv(condition).view(
            batch,
            condition_length,
            self.n_kv_heads,
            self.head_size,
            2,
        )
        key, value = key_value.unbind(-1)
        query = self.norm_q(query).transpose(1, 2)
        backends = _attention_backends(self.attention_backend, query.device, has_mask=mask is not None)
        key = self.norm_k(key)
        attention_mask = None
        if mask is not None:
            attention_mask = mask.to(torch.bool).reshape(batch, 1, 1, condition_length)
            attention_mask = attention_mask.expand(-1, -1, sequence, -1)
            if self.enable_gqa_optimization and batch == 1:
                # A single example's padding mask is shared by all heads.
                # Keep that contract visible to the core grouped-view path.
                attention_mask = attention_mask[0, 0]
        key, value, enable_gqa = _grouped_key_value(
            key, value, self.n_rep, backends, attention_mask, enabled=self.enable_gqa_optimization
        )
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.attn_scale,
            enable_gqa=enable_gqa,
            backends=backends,
        )
        return self.wo(output.transpose(1, 2).contiguous().view(batch, sequence, -1))


def _adapter(kind: str, in_features: int, out_features: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_features, out_features)
    match = re.match(r"^mlp(\d+)x_silu$", kind)
    if not match:
        raise ValueError(f"Unknown H-RDT adapter type: {kind}")
    layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
    for _ in range(1, int(match.group(1))):
        layers.extend((nn.SiLU(), nn.Linear(out_features, out_features)))
    return nn.Sequential(*layers)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        values = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(values), torch.sin(values)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding.to(self.dtype))


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(values)) * self.w3(values))


def _modulate(values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return values * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class HRDTBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        config: Mapping[str, Any],
        *,
        training_mode: str = "lang",
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if training_mode != "lang":
            raise ValueError(f"H-RDT only supports training_mode='lang', got {training_mode!r}")
        self.layer_idx = layer_idx
        hidden_size = int(config["hidden_size"])
        eps = float(config["norm_eps"])
        self.attn_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.attn = Attention(config, attention_backend=attention_backend)
        self.img_cross_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.img_cond_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.img_cross_attn = CrossAttention(config, attention_backend=attention_backend)
        self.lang_cross_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.lang_cond_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.lang_cross_attn = CrossAttention(config, attention_backend=attention_backend)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.ffn = FeedForward(
            dim=hidden_size,
            hidden_dim=4 * hidden_size,
            multiple_of=int(config["multiple_of"]),
            ffn_dim_multiplier=config.get("ffn_dim_multiplier"),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        contexts: Mapping[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        contexts = contexts or {}
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(timestep).chunk(9, dim=1)
        hidden = values + gate_attn.unsqueeze(1) * self.attn(_modulate(self.attn_norm(values), shift_attn, scale_attn))
        image = contexts.get("img_c")
        if image is not None:
            hidden = hidden + gate_cross.unsqueeze(1) * self.img_cross_attn(
                _modulate(self.img_cross_norm(hidden), shift_cross, scale_cross),
                self.img_cond_norm(image),
            )
        language = contexts.get("lang_c")
        if language is not None:
            hidden = hidden + self.lang_cross_attn(
                self.lang_cross_norm(hidden),
                self.lang_cond_norm(language),
                contexts.get("lang_attn_mask"),
            )
        return hidden + gate_mlp.unsqueeze(1) * self.ffn(_modulate(self.ffn_norm(hidden), shift_mlp, scale_mlp))


class _OutputMLP(nn.Module):
    """State-dict-compatible inference subset of timm.models.Mlp."""

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_size * 4, output_size)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.drop1(self.act(self.fc1(values)))
        return self.drop2(self.fc2(self.norm(values)))


class ActionDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=float(config["norm_eps"]))
        self.ffn = _OutputMLP(hidden_size, int(config["output_size"]))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, values: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(timestep).chunk(2, dim=1)
        return self.ffn(_modulate(self.ffn_norm(values), shift, scale))


class HRDT(nn.Module):
    def __init__(
        self,
        *,
        horizon: int,
        config: Mapping[str, Any],
        act_pos_emb_config: Iterable[tuple[str, Any]],
        img_pos_emb_config: Iterable[tuple[str, Any]],
        lang_pos_emb_config: Iterable[tuple[str, Any]],
        max_img_len: int,
        max_lang_len: int,
        training_mode: str,
        dtype: torch.dtype,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        hidden_size = int(config["hidden_size"])
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                HRDTBlock(
                    index,
                    config,
                    training_mode=training_mode,
                    attention_backend=attention_backend,
                )
                for index in range(int(config["depth"]))
            ]
        )
        self.action_decoder = ActionDecoder(config)
        self.x_pos_emb = nn.Parameter(torch.zeros(1, 1 + self.horizon, hidden_size))
        self.lang_pos_emb = nn.Parameter(torch.zeros(1, int(max_lang_len), hidden_size))
        self.img_pos_emb = nn.Parameter(torch.zeros(1, int(max_img_len), hidden_size))
        if self.x_pos_emb.device.type != "meta":
            self.x_pos_emb.data.copy_(
                torch.from_numpy(multimodal_position_embedding(hidden_size, act_pos_emb_config)).float().unsqueeze(0)
            )
            self.lang_pos_emb.data.copy_(
                torch.from_numpy(multimodal_position_embedding(hidden_size, lang_pos_emb_config)).float().unsqueeze(0)
            )
            self.img_pos_emb.data.copy_(
                torch.from_numpy(multimodal_position_embedding(hidden_size, img_pos_emb_config)).float().unsqueeze(0)
            )

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        *,
        img_c: torch.Tensor | None,
        lang_c: torch.Tensor | None,
        lang_attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        timestep_embedding = self.t_embedder(timestep)
        if timestep_embedding.shape[0] == 1:
            timestep_embedding = timestep_embedding.expand(values.shape[0], -1)
        values = values + self.x_pos_emb
        if img_c is not None:
            if img_c.shape[1] > self.img_pos_emb.shape[1]:
                raise ValueError("image token sequence exceeds the checkpoint position table")
            img_c = img_c + self.img_pos_emb[:, : img_c.shape[1]]
        if lang_c is not None:
            if lang_c.shape[1] > self.lang_pos_emb.shape[1]:
                raise ValueError("language token sequence exceeds the checkpoint position table")
            lang_c = lang_c + self.lang_pos_emb[:, : lang_c.shape[1]]
        contexts = {
            "img_c": img_c,
            "lang_c": lang_c,
            "lang_attn_mask": lang_attn_mask,
        }
        for block in self.blocks:
            values = block(values, timestep_embedding, contexts)
        return self.action_decoder(values, timestep_embedding)[:, -self.horizon :]


class ActionEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.state_adaptor = _adapter(str(config["st_adaptor"]), state_dim, hidden_size)
        self.action_adaptor = _adapter(str(config["act_adaptor"]), action_dim, hidden_size)

    def encode_state(self, values: torch.Tensor) -> torch.Tensor:
        return self.state_adaptor(values)

    def encode_action(self, values: torch.Tensor) -> torch.Tensor:
        return self.action_adaptor(values)


class HRDTRunner(nn.Module):
    """Checkpoint-compatible H-RDT inference runner."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        state_dim = int(config["state_dim"])
        action_dim = int(config["action_dim"])
        horizon = int(config["pred_horizon"])
        hidden_size = int(config["hrdt"]["hidden_size"])
        self.model = HRDT(
            horizon=horizon,
            config=config["hrdt"],
            act_pos_emb_config=config["act_pos_emb_config"],
            img_pos_emb_config=config["img_pos_emb_config"],
            lang_pos_emb_config=config["lang_pos_emb_config"],
            max_img_len=int(config["max_img_len"]),
            max_lang_len=int(config["max_lang_len"]),
            training_mode=str(config.get("training_mode", "lang")),
            dtype=dtype,
            attention_backend=attention_backend,
        )
        self.img_adapter = _adapter(
            str(config.get("img_adapter", "mlp2x_silu")),
            int(config.get("vision", {}).get("feature_dim", 2176)),
            hidden_size,
        )
        self.action_encoder = ActionEncoder(state_dim, action_dim, hidden_size, config)
        self.lang_adapter = _adapter(
            str(config.get("lang_adapter", "mlp2x_silu")),
            int(config.get("text", {}).get("feature_dim", 4096)),
            hidden_size,
        )
        # The released human-pretraining snapshot retains a legacy video
        # adapter even though the official image+language ``predict_action``
        # path does not invoke it.  Materialize the declared module so the
        # checkpoint remains strictly restorable instead of silently dropping
        # four published tensors.
        if config.get("video_adapter"):
            self.video_adapter = _adapter(
                str(config["video_adapter"]),
                int(config.get("video", {}).get("feature_dim", 2048)),
                hidden_size,
            )
        scheduler = config["noise_scheduler"]
        self.num_inference_timesteps = int(scheduler["num_inference_timesteps"])
        self.pred_horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_feature_dim = int(config.get("vision", {}).get("feature_dim", 2176))
        self.language_feature_dim = int(config.get("text", {}).get("feature_dim", 4096))
        self.max_img_len = int(config["max_img_len"])
        self.max_lang_len = int(config["max_lang_len"])

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        state_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        lang_tokens: torch.Tensor | None = None,
        lang_attn_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_inference_timesteps: int | None = None,
    ) -> torch.Tensor:
        if state_tokens.ndim != 3 or state_tokens.shape[1] != 1:
            raise ValueError("H-RDT state_tokens must have shape [batch, 1, state_dim]")
        if state_tokens.shape[-1] != self.state_dim:
            raise ValueError(f"H-RDT expects {self.state_dim} state values, got {state_tokens.shape[-1]}")
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != self.image_feature_dim:
            raise ValueError(f"H-RDT image_tokens must have shape [batch, tokens, {self.image_feature_dim}]")
        if lang_tokens is not None and lang_tokens.shape[-1] != self.language_feature_dim:
            raise ValueError(
                f"H-RDT language tokens must end in {self.language_feature_dim}, got {lang_tokens.shape[-1]}"
            )
        image_condition = self.img_adapter(image_tokens)
        language_condition = self.lang_adapter(lang_tokens) if lang_tokens is not None else None
        state_trajectory = self.action_encoder.encode_state(state_tokens)
        steps = int(num_inference_timesteps or self.num_inference_timesteps)
        if steps <= 0:
            raise ValueError("num_inference_timesteps must be positive")
        noisy_action = torch.randn(
            (image_tokens.shape[0], self.pred_horizon, self.action_dim),
            dtype=image_tokens.dtype,
            device=image_tokens.device,
            generator=generator,
        )
        timestep = torch.tensor([0.0], dtype=image_tokens.dtype, device=image_tokens.device)
        step_size = 1.0 / steps
        for _ in range(steps):
            action_trajectory = self.action_encoder.encode_action(noisy_action)
            prediction = self.model(
                torch.cat([state_trajectory, action_trajectory], dim=1),
                timestep,
                img_c=image_condition,
                lang_c=language_condition,
                lang_attn_mask=lang_attn_mask,
            )
            noisy_action = prediction * step_size + noisy_action
            timestep = timestep + step_size
        return noisy_action


__all__ = ["HRDT", "HRDTRunner", "multimodal_position_embedding"]
