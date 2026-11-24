# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 PRIME-RL
#
# Adapted from LaST-R1 (https://github.com/CHEN-H01/LaST-R1, revision
# ba4e939632427e066f734bf9f7abfa11cc4911f6, MIT License):
# transformers/integrations/sdpa_attention.py and
# transformers/models/qwen3_vl/modeling_qwen3_vl.py, which modify Hugging Face
# Transformers 4.57.0 (Apache License 2.0). Only the inference path used by the
# offline policy step is kept: the value head and the causal attention mode with
# a bidirectional action-token block.
"""LaST-R1 Qwen3-VL policy model on top of the installed Transformers Qwen3-VL."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

VALUE_HEAD_HIDDEN_SIZES = (512, 256, 128)


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run Transformers' SDPA attention with LaST-R1's action-token mask.

    When ``action_length`` is passed, ``attn_mode="causal"`` keeps the prompt
    and latent tokens causal and lets the trailing ``action_length`` tokens
    attend to each other in both directions. Without ``action_length`` this is
    the stock Transformers SDPA path.
    """
    sdpa_kwargs: dict[str, bool] = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if "action_length" in kwargs:
        if kwargs.get("attn_mode") != "causal":
            raise ValueError(f"LaST-R1 inference supports attn_mode='causal', got {kwargs.get('attn_mode')!r}")
        action_length = int(kwargs["action_length"])
        if attention_mask is None:
            q_length, kv_length = query.shape[-2], key.shape[-2]
            query_positions = torch.arange(kv_length - q_length, kv_length, device=query.device).view(1, 1, -1, 1)
            key_positions = torch.arange(kv_length, device=query.device).view(1, 1, 1, -1)
            attention_mask = query_positions >= key_positions
        if action_length > 0:
            attention_mask[:, :, -action_length:, -action_length:] = True

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    return attn_output.transpose(1, 2).contiguous(), None


class LastR1TextAttention(Qwen3VLTextAttention):
    """Qwen3-VL text attention that routes SDPA through the action-token mask."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        implementation = self.config._attn_implementation
        if implementation == "sdpa":
            attention_interface = sdpa_attention_forward
        elif implementation == "eager":
            attention_interface = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class ValueHead(nn.Module):
    """ReLU MLP that scores the hidden state at the chosen latent end."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = VALUE_HEAD_HIDDEN_SIZES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(in_dim, width), nn.ReLU()))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class LastR1Qwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with LaST-R1's value head and action-token text attention."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            layer.self_attn = LastR1TextAttention(config.text_config, layer_idx)
        self.value_head = ValueHead(config.text_config.hidden_size)
        self.post_init()


def decode_action_tokens(token_ids: Any, *, action_vocab_end: int, bins: int = 256) -> Any:
    """Map action token ids to normalized actions in ``[-1, 1]``.

    Matches LaST-R1's ``ActionTokenizer.decode_token_ids_to_actions``: ids count
    down from ``action_vocab_end`` and select the centre of one of ``bins - 1``
    uniform intervals.
    """
    import numpy as np

    edges = np.linspace(-1, 1, bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    indices = np.clip(action_vocab_end - np.asarray(token_ids) - 1, a_min=0, a_max=centers.shape[0] - 1)
    return centers[indices]
