# Copyright (C) 2026 Xiaomi Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only Xiaomi-Robotics-1 model graph.

The multimodal backbone reuses the checkpoint-compatible Qwen3-VL primitives
already kept in tree for Xiaomi-Robotics-0.  This module adds the policy token
embeddings and the rectified-flow DiT required by the XR1 checkpoint without
executing checkpoint-side Python.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.synthesis.action_generation.xiaomi_robotics_0.modeling_mibot import (
    Qwen3VLModel as _Qwen3VLModel,
)
from worldfoundry.synthesis.action_generation.xiaomi_robotics_0.modeling_mibot import (
    Qwen3VLPreTrainedModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb,
)


@dataclass
class XR1BackboneOutput:
    """VLM state consumed by the policy head."""

    past_key_values: Any
    position_ids: torch.Tensor
    attention_mask: torch.Tensor


class XR1Qwen3VLModel(_Qwen3VLModel):
    """Qwen3-VL backbone with state, action, and score placeholders."""

    def __init__(
        self,
        config: Any,
        *,
        score_token_id: int,
        state_token_id: int,
        action_token_count: int,
    ) -> None:
        super().__init__(config)
        self.score_token_id = int(score_token_id)
        self.state_token_id = int(state_token_id)
        self.action_token_count = int(action_token_count)
        hidden_size = int(config.text_config.hidden_size)
        self.action_embed = nn.Embedding(self.action_token_count, hidden_size)
        self.score_embed = nn.Embedding(1, hidden_size)
        self.action_embed.apply(self._init_weights)
        self.score_embed.apply(self._init_weights)

    @staticmethod
    def _replace_embeddings(
        inputs_embeds: torch.Tensor,
        mask: torch.Tensor,
        replacements: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        count = int(mask.sum().item())
        replacements = replacements.reshape(-1, inputs_embeds.shape[-1]).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if replacements.shape[0] != count:
            raise ValueError(
                f"{name} embedding count mismatch: prompt has {count}, runtime supplied {replacements.shape[0]}"
            )
        expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(expanded, replacements)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        state_embeds: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None:
            raise ValueError("XR1 requires token IDs so policy placeholders can be resolved")
        if inputs_embeds is not None:
            raise ValueError("XR1 does not accept caller-provided input embeddings")

        inputs_embeds = self.get_input_embeddings()(input_ids)
        state_mask = input_ids == self.state_token_id
        if bool(state_mask.any()):
            if state_embeds is None:
                raise ValueError("state embeddings are required when the prompt contains <state>")
            inputs_embeds = self._replace_embeddings(
                inputs_embeds,
                state_mask,
                state_embeds,
                name="state",
            )

        action_low = self.state_token_id + 1
        action_high = action_low + self.action_token_count
        action_mask = (input_ids >= action_low) & (input_ids < action_high)
        if bool(action_mask.any()):
            action_indices = input_ids[action_mask] - action_low
            action_embeds = self.action_embed(action_indices)
            inputs_embeds = self._replace_embeddings(
                inputs_embeds,
                action_mask,
                action_embeds,
                name="action",
            )

        score_mask = input_ids == self.score_token_id
        if bool(score_mask.any()):
            score_embeds = self.score_embed(
                torch.zeros(int(score_mask.sum().item()), dtype=torch.long, device=input_ids.device)
            )
            inputs_embeds = self._replace_embeddings(
                inputs_embeds,
                score_mask,
                score_embeds,
                name="score",
            )

        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )


class XR1Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel):
    """Checkpoint-compatible Qwen wrapper reduced to policy inference outputs."""

    _checkpoint_conversion_mapping: dict[str, str] = {}  # noqa: RUF012 - Shared class-level value, as upstream.
    _tied_weights_keys: ClassVar[dict] = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False

    def __init__(
        self,
        config: Any,
        *,
        score_token_id: int,
        state_token_id: int,
        action_token_count: int,
    ) -> None:
        super().__init__(config)
        self.model = XR1Qwen3VLModel(
            config,
            score_token_id=score_token_id,
            state_token_id=state_token_id,
            action_token_count=action_token_count,
        )
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @property
    def language_model(self) -> nn.Module:
        return self.model.language_model

    @property
    def visual(self) -> nn.Module:
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        state_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> XR1BackboneOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_embeds=state_embeds,
            **kwargs,
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return XR1BackboneOutput(
            past_key_values=outputs.past_key_values,
            position_ids=outputs.position_ids,
            attention_mask=attention_mask,
        )


class MLPProjector(nn.Module):
    """Checkpoint-compatible GELU MLP projector."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        inter_dim: int | None = None,
        num_layers: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if num_layers > 1 and (inter_dim is None or inter_dim <= 0):
            raise ValueError("inter_dim must be positive for a multi-layer projector")
        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, int(inter_dim), bias=bias))
            for _ in range(1, num_layers - 1):
                layers.extend(
                    [
                        nn.GELU(approximate="tanh"),
                        nn.Linear(int(inter_dim), int(inter_dim), bias=bias),
                    ]
                )
            layers.extend(
                [
                    nn.GELU(approximate="tanh"),
                    nn.Linear(int(inter_dim), output_dim, bias=bias),
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class TimestepEmbedder(nn.Module):
    """Sinusoidal rectified-flow timestep embedding."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        arguments = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding.to(self.mlp[0].weight.dtype)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(timesteps))[:, None]


def _repeat_batch(value: torch.Tensor, target_batch_size: int) -> torch.Tensor:
    current = int(value.shape[0])
    if current == target_batch_size:
        return value
    if target_batch_size % current:
        raise ValueError(f"cannot repeat cache batch {current} to {target_batch_size}")
    return value.repeat_interleave(target_batch_size // current, dim=0)


class DiTAttention(nn.Module):
    """Exact policy attention routed through WorldFoundry core SDPA."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        kv_heads: int,
        dropout: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        if hidden_size % head_dim:
            raise ValueError("hidden_size must be divisible by head_dim")
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.num_heads = self.hidden_size // self.head_dim
        if self.num_heads % kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        self.kv_group = self.num_heads // int(kv_heads)
        self.dropout = float(dropout)
        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor],
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(
            batch_size,
            query_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cosine, sine = position_embeds
        if cosine.ndim == 4:
            cosine, sine = cosine[0], sine[0]
        query, key = apply_rotary_pos_emb(query, key, cosine, sine)

        cached_key, cached_value = past_key_values
        cached_key = _repeat_batch(cached_key, batch_size).repeat_interleave(self.kv_group, dim=1)
        cached_value = _repeat_batch(cached_value, batch_size).repeat_interleave(self.kv_group, dim=1)
        key = torch.cat((cached_key, key), dim=-2)
        value = torch.cat((cached_value, value), dim=-2)
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch_size, query_length, -1))


class DiTMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        intermediate = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(value)) * self.up_proj(value))


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1 + scale) + shift


class DiTDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.attn = DiTAttention(
            hidden_size=hidden_size,
            head_dim=int(config["head_dim"]),
            kv_heads=int(config["kv_heads"]),
            dropout=float(config["dropout"]),
            attention_bias=bool(config["attention_bias"]),
        )
        self.mlp = DiTMLP(hidden_size)
        self.input_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor],
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        timestep_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_table[None] + timestep_embeds
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = _modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        hidden_states = residual + gate_msa * self.attn(
            hidden_states,
            past_key_values,
            position_embeds,
            attention_mask,
        )
        residual = hidden_states
        hidden_states = _modulate(self.post_layernorm(hidden_states), shift_mlp, scale_mlp)
        return residual + gate_mlp * self.mlp(hidden_states)


class DiT(nn.Module):
    """Checkpoint-compatible action diffusion transformer."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.is_causal = bool(config["is_causal"])
        self.layers = nn.ModuleList([DiTDecoderLayer(config) for _ in range(int(config["num_layers"]))])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, Qwen3VLTextRMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        timestep_embeds: torch.Tensor,
    ) -> torch.Tensor:
        start_index = max(0, len(past_key_values) - len(self.layers))
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start_index + index],
                position_embeds,
                timestep_embeds,
                attention_mask,
            )
        return hidden_states


def _legacy_cache_tensors(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [(layer.keys, layer.values) for layer in layers]
    return [(layer[0], layer[1]) for layer in cache]


class XR1(nn.Module):
    """Qwen3-VL conditioned rectified-flow action policy."""

    def __init__(
        self,
        *,
        vlm_config: Any,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        n_choices: int,
        dit_config: Mapping[str, Any],
        num_steps: int,
        knowledge_insulation: bool,
        score_token_id: int,
        state_token_id: int,
        action_token_count: int,
        timestep_frequency_size: int,
    ) -> None:
        super().__init__()
        self.state_shape = tuple(int(value) for value in state_shape)
        self.action_shape = tuple(int(value) for value in action_shape)
        self.n_choices = int(n_choices)
        self.num_steps = int(num_steps)
        self.knowledge_insulation = bool(knowledge_insulation)
        self.vlm = XR1Qwen3VLForConditionalGeneration(
            vlm_config,
            score_token_id=score_token_id,
            state_token_id=state_token_id,
            action_token_count=action_token_count,
        )
        hidden_size = int(vlm_config.text_config.hidden_size)
        dit_hidden_size = int(dit_config["hidden_size"])
        self.state_projector_choice = MLPProjector(
            input_dim=self.state_shape[-1],
            inter_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.action_projector_choice = nn.Sequential(
            MLPProjector(
                input_dim=hidden_size,
                inter_dim=hidden_size,
                output_dim=hidden_size,
                num_layers=4,
            ),
            MLPProjector(
                input_dim=hidden_size,
                output_dim=self.action_shape[-1] * self.n_choices,
                num_layers=1,
            ),
        )
        self.score_projector_choice = nn.Sequential(
            MLPProjector(
                input_dim=hidden_size,
                inter_dim=hidden_size,
                output_dim=hidden_size,
                num_layers=4,
            ),
            MLPProjector(
                input_dim=hidden_size,
                output_dim=self.n_choices,
                num_layers=1,
            ),
        )
        self.dit = DiT(dit_config)
        self.state_projector = MLPProjector(
            input_dim=self.state_shape[-1],
            inter_dim=dit_hidden_size,
            output_dim=dit_hidden_size,
            num_layers=2,
        )
        self.action_projector = MLPProjector(
            input_dim=self.action_shape[-1],
            inter_dim=dit_hidden_size,
            output_dim=dit_hidden_size,
            num_layers=2,
        )
        self.action_output_layer = MLPProjector(
            input_dim=dit_hidden_size,
            inter_dim=dit_hidden_size,
            output_dim=self.action_shape[-1],
            num_layers=2,
        )
        self.t_embedder = TimestepEmbedder(dit_hidden_size, timestep_frequency_size)
        self.t_projector = MLPProjector(
            input_dim=dit_hidden_size,
            output_dim=6 * dit_hidden_size,
            bias=True,
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(vlm_config.text_config)
        self.sink = nn.Embedding(1, dit_hidden_size)

    def refresh_nonpersistent_rotary_buffers(self, *, device: str = "cpu") -> None:
        """Recreate non-persistent RoPE buffers after meta-device assembly."""

        text_config = self.vlm.config.text_config
        vision_config = self.vlm.config.vision_config
        self.vlm.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(
            text_config,
            device=device,
        )
        self.vlm.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            # The vision model builds 2-D RoPE by looking up row/column
            # frequencies and concatenating the two axes.  Its constructor
            # therefore receives half a head, exactly as in
            # ``Qwen3VLVisionModel.__init__``.  Recreating the meta-device
            # buffer at a full head silently doubles cos/sin to 128 values for
            # the released 64-D vision heads and only fails on a real forward.
            int(vision_config.hidden_size // vision_config.num_heads // 2)
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config, device=device)

    @staticmethod
    def _unpad_cache(
        cache: Any,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segments: torch.Tensor | None,
        target_batch: int,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        layers = _legacy_cache_tensors(cache)
        if segments is None:
            return layers, position_ids, attention_mask
        if len(segments) != target_batch:
            raise ValueError(f"expected {target_batch} conditioning segments, got {len(segments)}")
        keys = torch.stack([layer[0] for layer in layers])
        values = torch.stack([layer[1] for layer in layers])
        lengths = segments[:, 1] - segments[:, 0]
        maximum = int(lengths.max().item())
        new_keys = keys.new_zeros((keys.shape[0], target_batch, keys.shape[2], maximum, keys.shape[4]))
        new_values = values.new_zeros(new_keys.shape)
        new_positions = position_ids.new_zeros((3, target_batch, maximum))
        new_mask = attention_mask.new_zeros((target_batch, maximum))
        for index in range(target_batch):
            start = int(segments[index, 0].item())
            length = int(lengths[index].item())
            end = start + length
            source_batch = index if keys.shape[1] > 1 else 0
            new_keys[:, index, :, :length] = keys[:, source_batch, :, start:end]
            new_values[:, index, :, :length] = values[:, source_batch, :, start:end]
            new_positions[:, index, :length] = position_ids[:, source_batch, start:end]
            new_mask[index, :length] = 1
        return list(zip(new_keys, new_values, strict=True)), new_positions, new_mask

    def _dit_velocity(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        *,
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        timestep_embeds = self.t_embedder(timestep[:, 0, 0] * 1000)
        timestep_embeds = self.t_projector(timestep_embeds).view(
            timestep_embeds.shape[0],
            6,
            -1,
        )
        noisy_action = self.action_projector(noisy_action * action_mask)
        sink = self.sink.weight[None].expand(state_embed.shape[0], -1, -1)
        hidden_states = torch.cat((sink, state_embed, noisy_action), dim=1).contiguous()
        hidden_states = self.dit(
            hidden_states,
            past_key_values,
            attention_mask,
            position_embeds,
            timestep_embeds,
        )
        return self.action_output_layer(hidden_states[:, -noisy_action.shape[1] :])

    @torch.inference_mode()
    def generate(self, batch: Mapping[str, Any], *, seed: int) -> torch.Tensor:
        if self.training:
            raise RuntimeError("XR1 inference requires eval mode")
        values = dict(batch)
        segments_key = "action_vlm_condition_segments" if self.knowledge_insulation else "action_segments"
        segments = values.pop(segments_key, None)
        values.pop("vlm_action_actual_length", None)
        values.pop("vlm_action_target", None)
        values.pop("vlm_action_mask", None)
        action = values.pop("action", None)
        action_mask = values.pop("action_mask", None)
        state = values.pop("state", None)
        if state is None:
            raise ValueError("XR1 requires a robot state tensor")
        parameter = next(self.state_projector_choice.parameters())
        state = state.to(device=parameter.device, dtype=parameter.dtype)
        if action is None:
            action = torch.zeros(
                (state.shape[0], *self.action_shape),
                device=state.device,
                dtype=state.dtype,
            )
        else:
            action = action.to(device=state.device, dtype=state.dtype)
        if action_mask is None:
            action_mask = torch.ones_like(action)
        else:
            action_mask = action_mask.to(device=state.device, dtype=state.dtype)
        values["state_embeds"] = self.state_projector_choice(state.flatten(0, 1))
        outputs = self.vlm(**values, use_cache=True)

        batch_size, action_length, _ = action.shape
        query_length = action_length + state.shape[1] + 1
        cache, cache_positions, cache_attention = self._unpad_cache(
            outputs.past_key_values,
            outputs.position_ids,
            outputs.attention_mask,
            segments,
            batch_size,
        )
        position_ids = (
            torch.arange(query_length, device=action.device).view(1, 1, -1).expand(3, batch_size, -1)
            + cache_positions.max(dim=-1).values[..., None]
            + 1
        )
        position_embeds = self.rotary_emb(action, position_ids)
        cache_mask = cache_attention[:, None, :].expand(-1, query_length, -1).bool()
        if self.dit.is_causal:
            query_mask = torch.ones(
                (batch_size, query_length, query_length),
                device=action.device,
                dtype=torch.bool,
            ).tril()
        else:
            query_mask = torch.ones(
                (batch_size, query_length, query_length),
                device=action.device,
                dtype=torch.bool,
            )
        attention_mask = torch.cat((cache_mask, query_mask), dim=-1)[:, None]
        state_embed = self.state_projector(state)
        generator = torch.Generator(device=action.device)
        generator.manual_seed(int(seed))
        sample = torch.randn(
            action.shape,
            device=action.device,
            dtype=action.dtype,
            generator=generator,
        )
        step_size = 1.0 / self.num_steps
        for step in range(self.num_steps):
            timestep = torch.full(
                (sample.shape[0], 1, 1),
                step / self.num_steps,
                device=sample.device,
                dtype=sample.dtype,
            )
            sample = (
                sample
                + self._dit_velocity(
                    sample,
                    timestep,
                    action_mask=action_mask,
                    state_embed=state_embed,
                    position_embeds=position_embeds,
                    past_key_values=cache,
                    attention_mask=attention_mask,
                )
                * step_size
            )
        return sample


__all__ = [
    "XR1",
    "DiT",
    "MLPProjector",
    "TimestepEmbedder",
    "XR1Qwen3VLForConditionalGeneration",
]
