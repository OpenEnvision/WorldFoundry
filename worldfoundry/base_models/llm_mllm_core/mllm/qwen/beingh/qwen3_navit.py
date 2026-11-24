# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import glob
import re
from functools import partial
from typing import ClassVar

import torch
from safetensors import safe_open
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from worldfoundry.core.attention import scaled_dot_product_attention

from .qwen3.configuration_qwen3 import Qwen3Config as _Qwen3Config
from .qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

# isort: split
from .qwen2_navit import get_layer_mapping_strategy, interpolate_layer_params

# isort: split
# isort: split
from .qwen2_navit import BaseNavitOutputWithPast, NaiveCache, pad_sequence

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)


class Qwen3Config(_Qwen3Config):
    """Qwen config implementation."""

    model_type = "qwen3"
    keys_to_ignore_at_inference: ClassVar[list] = ["past_key_values"]

    def __init__(self, qk_norm=True, layer_module="Qwen3DecoderLayer", **kwargs):
        super().__init__(**kwargs)
        self.qk_norm = qk_norm
        self.layer_module = layer_module


class PackedAttention(Qwen3Attention):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask: list[torch.Tensor],
        packed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
    ):
        total_seq_len = packed_sequence.shape[0]
        dtype, device = packed_sequence.dtype, packed_sequence.device
        packed_query_states = torch.zeros((total_seq_len, self.num_heads * self.head_dim), dtype=dtype, device=device)
        packed_key_states = torch.zeros(
            (total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device
        )
        packed_value_states = torch.zeros(
            (total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device
        )

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence).to(dtype)
        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence).to(dtype)
        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence).to(dtype)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, list):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask, strict=False
            ):
                attn_output = scaled_dot_product_attention(
                    query_states.to(torch.bfloat16).unsqueeze(0),
                    key_states.to(torch.bfloat16).unsqueeze(0),
                    value_states.to(torch.bfloat16).unsqueeze(0),
                    attn_mask=attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    backend="efficient",
                )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0),  # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output[packed_und_token_indexes])

        return packed_attn_output


class PackedAttentionMoT(Qwen3Attention):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config, layer_idx)

        if self.config.qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=config.expert_config.rms_norm_eps)
            self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=config.expert_config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_mot_gen = nn.Identity()
            self.k_norm_mot_gen = nn.Identity()

        # if llm and expert are the same, then use the pretrained weight, else reinitializaiton
        mot_gen_hidden_size = config.expert_config.hidden_size

        self.q_proj_mot_gen = nn.Linear(mot_gen_hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_mot_gen = nn.Linear(
            mot_gen_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_mot_gen = nn.Linear(
            mot_gen_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_mot_gen = nn.Linear(self.num_heads * self.head_dim, mot_gen_hidden_size, bias=config.attention_bias)

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        total_seq_len = packed_sequence_und.shape[0] + packed_sequence_gen.shape[0]
        dtype, device = packed_sequence_und.dtype, packed_sequence_und.device

        packed_query_states = torch.zeros((total_seq_len, self.num_heads * self.head_dim), dtype=dtype, device=device)
        packed_key_states = torch.zeros(
            (total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device
        )
        packed_value_states = torch.zeros(
            (total_seq_len, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device
        )

        # 0123 6789 45 1011         <-> 0123 6789 45 1011
        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_mot_gen(packed_sequence_gen)

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_mot_gen(packed_sequence_gen)

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_mot_gen(packed_sequence_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_mot_gen(
            packed_query_states[packed_gen_token_indexes]
        )

        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_mot_gen(packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, list):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask, strict=False
            ):
                # breakpoint()
                attn_output = scaled_dot_product_attention(
                    query_states.to(torch.bfloat16).unsqueeze(0),
                    key_states.to(torch.bfloat16).unsqueeze(0),
                    value_states.to(torch.bfloat16).unsqueeze(0),
                    attn_mask=attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    backend="efficient",
                )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0),  # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            # breakpoint()
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_und = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_gen = self.o_proj_mot_gen(packed_attn_output[packed_gen_token_indexes])

        return packed_attn_output_und, packed_attn_output_gen


class Qwen3DecoderLayer(nn.Module):
    """Qwen decoder layer implementation."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        # [seq_len, dim]
        residual = packed_sequence_und
        packed_sequence = self.input_layernorm(packed_sequence_und)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
        )

        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence, packed_sequence_gen


class Qwen3MoTDecoderLayer(nn.Module):
    """Qwen mo t decoder layer implementation."""

    def __init__(
        self,
        config,
        # gen_config
        layer_idx: int | None = None,
        attn_module: Qwen3Attention | None = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(config.expert_config)

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(
            config.expert_config.hidden_size, eps=config.expert_config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(
            config.expert_config.hidden_size, eps=config.expert_config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        residual_und = packed_sequence_und
        residual_gen = packed_sequence_gen

        packed_sequence_und_ = self.input_layernorm(packed_sequence_und)
        packed_sequence_gen_ = self.input_layernorm_mot_gen(packed_sequence_gen)

        # Self Attention
        packed_sequence_und_, packed_sequence_gen_ = self.self_attn(
            packed_sequence_und=packed_sequence_und_,
            packed_sequence_gen=packed_sequence_gen_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )

        packed_sequence_und = residual_und + packed_sequence_und_
        packed_sequence_gen = residual_gen + packed_sequence_gen_

        # Fully Connected
        residual_und = packed_sequence_und
        residual_gen = packed_sequence_gen

        packed_sequence_und_ = packed_sequence_und.new_zeros(packed_sequence_und.shape)
        packed_sequence_gen_ = packed_sequence_gen.new_zeros(packed_sequence_gen.shape)

        packed_sequence_und_ = self.mlp(self.post_attention_layernorm(packed_sequence_und))
        packed_sequence_gen_ = self.mlp_mot_gen(self.post_attention_layernorm_mot_gen(packed_sequence_gen))

        packed_sequence_und = residual_und + packed_sequence_und_
        packed_sequence_gen = residual_gen + packed_sequence_gen_

        return packed_sequence_und, packed_sequence_gen

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_mot_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_mot_gen(packed_vae_query_sequence).to(
                torch.bfloat16
            )

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_mot_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen3DecoderLayer": Qwen3DecoderLayer,
    "Qwen3MoTDecoderLayer": partial(Qwen3MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen3Model(Qwen3PreTrainedModel):
    """Qwen model implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_mot = config.use_mot
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]

        self.layers = nn.ModuleList([layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_mot:
            self.norm_mot_gen = Qwen3RMSNorm(config.expert_config.hidden_size, eps=config.expert_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.post_init()

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor | None = None,
        packed_gen_token_indexes: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_mot:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
        else:
            extra_inputs.update(packed_und_token_indexes=packed_und_token_indexes)

        for decoder_layer in self.layers:
            packed_sequence_und, packed_sequence_gen = decoder_layer(
                packed_sequence_und=packed_sequence_und,
                packed_sequence_gen=packed_sequence_gen,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs,
            )

        if self.use_mot:
            # packed_sequence_ = torch.zeros((packed_sequence_und.shape[0]+packed_sequence_gen.shape[0], packed_sequence_und.shape[1]), dtype=dtype, device=device)
            packed_sequence_und_ = torch.zeros_like(packed_sequence_und)
            packed_sequence_gen_ = torch.zeros_like(packed_sequence_gen)
            # packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence_und)
            packed_sequence_und_ = self.norm(packed_sequence_und)
            packed_sequence_gen_ = self.norm_mot_gen(packed_sequence_gen)
            return packed_sequence_und_, packed_sequence_gen_
        else:
            assert packed_sequence_gen.shape[0] == 0
            return self.norm(packed_sequence_und), packed_sequence_gen


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    """Qwen for causal lm implementation."""

    _tied_weights_keys: ClassVar[list] = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.llm_layers = config.num_hidden_layers
        self.use_mot = config.use_mot
        if self.use_mot:
            self.expert_layers = config.expert_config.num_hidden_layers
            self.llm_identical = config.hidden_size == config.expert_config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def custom_init_pretrained(self, mllm_state_dict, logger):
        logger.info("Initializing LLM decoder from MLLM weights")
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                continue

            original_name = (
                "language_model." + name.replace("_und", "") if "norm_und" in name else "language_model." + name
            )
            try:
                param.data.copy_(mllm_state_dict.pop(original_name).data)
            except BaseException:
                logger.info(f"'{name}' not initialized")
        return mllm_state_dict

    def init_pretrained(self, mllm_state_dict, enable=True):
        if not enable:
            print("Initializing LLM decoder randomly")
            return mllm_state_dict

        print("Initializing LLM decoder from MLLM weights")
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                continue

            original_name = (
                "language_model." + name.replace("_und", "") if "norm_und" in name else "language_model." + name
            )
            try:
                param.data.copy_(mllm_state_dict.pop(original_name).data)
            except BaseException:
                print(f"'{name}' not initialized")
        return mllm_state_dict

    def init_expert(self, expert_path, from_scratch=True):
        """initialize parameters of action expert using the pretrained llm ckpt"""

        if from_scratch:
            print("INFO: `from_scratch=True`. Expert (`_mot_gen`) parameters will be randomly initialized.")
            return
        assert 1 == 0
        safetensor_files = glob.glob(f"{expert_path}/*.safetensors")
        expert_state_dict = {}
        for file_path in safetensor_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
                    expert_state_dict[key] = f.get_tensor(key)

        layer_mapping = get_layer_mapping_strategy(self.llm_layers, self.expert_layers)

        for name, param in self.named_parameters():
            if "mot_gen" not in name:
                continue

            original_name = name.replace("_mot_gen", "")

            if any(n in name for n in ["q_norm_mot_gen", "k_norm_mot_gen"]):
                param.data.copy_(self.state_dict()[original_name].data)
            elif any(proj in name for proj in ["q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen"]):
                if self.llm_identical and not expert_path:
                    param.data.copy_(expert_state_dict[original_name].data)
            else:
                layer_match = re.search(r"layers\.(\d+)\.", name)
                if layer_match:
                    curr_layer_idx = int(layer_match.group(1))
                    if curr_layer_idx < len(layer_mapping):
                        expert_pos = layer_mapping[curr_layer_idx]
                        original_name_template = name.replace("_mot_gen", "").replace(
                            f"layers.{curr_layer_idx}.", "layers.{}."
                        )  # Build parameter name template

                        interpolated_param = interpolate_layer_params(
                            expert_state_dict, expert_pos, original_name_template, num_expert_layers=self.expert_layers
                        )

                        if interpolated_param is not None:
                            param.data.copy_(interpolated_param)
                        else:
                            fallback_idx = min(int(expert_pos), self.expert_layers - 1)
                            fallback_name = original_name_template.format(fallback_idx)
                            if fallback_name in expert_state_dict:
                                param.data.copy_(expert_state_dict[fallback_name])
                                print(f"Layer {curr_layer_idx}: fallback to expert layer {fallback_idx}")
                else:
                    if original_name in expert_state_dict:
                        param.data.copy_(expert_state_dict[original_name].data)

    def init_mot(self):
        """initialize the action expert by direct param copy, requiring llm and expert sharing the same arch"""
        for name, param in self.named_parameters():
            if "mot_gen" in name:
                original_name = name.replace("_mot_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence_und: torch.Tensor,
        packed_sequence_gen: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor | None = None,
        packed_gen_token_indexes: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(
            packed_sequence_und=packed_sequence_und,
            packed_sequence_gen=packed_sequence_gen,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs
