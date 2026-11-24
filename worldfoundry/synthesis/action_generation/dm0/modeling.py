"""Inference-only DM0 policy architecture.

The language/vision base and Perception Encoder are shared with the existing
in-tree Dexbotic implementation; this module only carries DM0's flow-matching
action expert and merged-attention sampler.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CONFIG_MAPPING, DynamicCache, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3

from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.synthesis.action_generation.db_cogact.modeling.base import (
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)

from .utils import (
    make_attn_mask_2d,
    make_attn_mask_4d,
    make_suffix_attn_mask_2d,
    posemb_sincos,
)


class DM0Config(DexboticConfig):
    """Configuration embedded in released DM0 checkpoints."""

    model_type = "dexbotic_dm0"
    action_dim = 32
    chunk_size = 50
    bf16 = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        action_config = kwargs.get("action_config")
        if isinstance(action_config, dict):
            payload = dict(action_config)
            model_type = payload.pop("model_type")
            self.action_config = CONFIG_MAPPING[model_type](**payload)
        elif isinstance(action_config, str):
            raise TypeError("DM0 action_config must be embedded in the local checkpoint config")
        llm_config = kwargs.get("llm_config")
        if isinstance(llm_config, dict):
            payload = dict(llm_config)
            model_type = payload.pop("model_type")
            self.llm_config = CONFIG_MAPPING[model_type](**payload)
        elif isinstance(llm_config, str):
            raise TypeError("DM0 llm_config must be embedded in the local checkpoint config")


class DM0Model(DexboticVLMModel):
    """Vision-language prefix plus a separate Qwen3 action expert."""

    def __init__(self, config: DM0Config) -> None:
        super().__init__(config)
        self.action_expert = Qwen3ForCausalLM(config.action_config)
        self.action_expert.model.embed_tokens = None
        action_hidden = config.action_config.hidden_size
        self.action_in_proj = nn.Linear(config.action_dim, action_hidden)
        self.action_time_mlp_in = nn.Linear(2 * action_hidden, action_hidden)
        self.action_time_mlp_out = nn.Linear(action_hidden, action_hidden)
        self.action_out_proj = nn.Linear(action_hidden, config.action_dim)

    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mm_vision_module(images)
        return self.mm_projector_module(features)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.llm.embed_tokens(tokens)


class DM0ForCausalLM(DexboticForCausalLM):
    """Checkpoint-compatible DM0 model with Euler flow sampling only."""

    config_class = DM0Config
    _tied_weights_keys: ClassVar[dict] = {"lm_head.weight": "model.llm.embed_tokens.weight"}

    def _real_init(self, config: DM0Config) -> None:
        self.model = DM0Model(config)
        self.lm_head = nn.Linear(
            config.llm_config.hidden_size,
            config.llm_config.vocab_size,
            bias=False,
        )
        self.post_init()

    @staticmethod
    def _cached_layer(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one KV layer across Transformers cache API generations."""

        layers = getattr(cache, "layers", None)
        if layers is not None:
            layer = layers[layer_idx]
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
                return keys, values
        # Transformers <=4.57 exposed DynamicCache as a subscriptable sequence.
        try:
            cached = cache[layer_idx]  # type: ignore[index]
        except (IndexError, TypeError) as exc:
            raise RuntimeError(f"DM0 cache layer {layer_idx} is unavailable") from exc
        if not (
            isinstance(cached, (tuple, list))
            and len(cached) >= 2
            and isinstance(cached[0], torch.Tensor)
            and isinstance(cached[1], torch.Tensor)
        ):
            raise TypeError(f"DM0 cache layer {layer_idx} does not contain key/value tensors")
        return cached[0], cached[1]

    def _compute_merged_layer(
        self,
        layer_idx: int,
        modules: list[nn.Module],
        embeddings: list[torch.Tensor | None],
        position_ids: torch.LongTensor,
        cache: DynamicCache | None,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> list[torch.Tensor | None]:
        queries: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        sequence_lengths: list[int] = []
        layers = [module.layers[layer_idx] for module in modules]
        batch_size = 0

        for layer, hidden in zip(layers, embeddings, strict=True):
            if hidden is None:
                sequence_lengths.append(0)
                continue
            normalized = layer.input_layernorm(hidden)
            batch_size, sequence_length, _ = normalized.shape
            sequence_lengths.append(sequence_length)
            projection_dtype = layer.self_attn.q_proj.weight.dtype
            normalized = normalized.to(dtype=projection_dtype)
            query = layer.self_attn.q_norm(
                layer.self_attn.q_proj(normalized).view(
                    batch_size,
                    sequence_length,
                    -1,
                    layer.self_attn.head_dim,
                )
            ).transpose(1, 2)
            key = layer.self_attn.k_norm(
                layer.self_attn.k_proj(normalized).view(
                    batch_size,
                    sequence_length,
                    -1,
                    layer.self_attn.head_dim,
                )
            ).transpose(1, 2)
            value = (
                layer.self_attn.v_proj(normalized)
                .view(
                    batch_size,
                    sequence_length,
                    -1,
                    layer.self_attn.head_dim,
                )
                .transpose(1, 2)
            )
            queries.append(query)
            keys.append(key)
            values.append(value)

        query_states = torch.cat(queries, dim=2)
        key_states = torch.cat(keys, dim=2)
        value_states = torch.cat(values, dim=2)
        rotary = self.model.llm.rotary_emb
        dummy = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = rotary(dummy, position_ids)
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if cache is not None:
            if use_cache:
                key_states, value_states = cache.update(
                    key_states,
                    value_states,
                    layer_idx,
                )
            elif len(cache) > layer_idx:
                cached_keys, cached_values = self._cached_layer(cache, layer_idx)
                key_states = torch.cat((cached_keys, key_states), dim=-2)
                value_states = torch.cat((cached_values, value_states), dim=-2)

        attention_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=layers[0].self_attn.scaling,
            enable_gqa=query_states.shape[1] != key_states.shape[1],
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, sum(sequence_lengths), -1)
        outputs: list[torch.Tensor | None] = []
        offset = 0
        for layer, hidden, length in zip(layers, embeddings, sequence_lengths, strict=True):
            if hidden is None:
                outputs.append(None)
                continue
            attended = attention_output[:, offset : offset + length]
            offset += length
            attended = layer.self_attn.o_proj(attended)
            residual = hidden + attended
            normalized = layer.post_attention_layernorm(residual)
            normalized = normalized.to(dtype=layer.mlp.gate_proj.weight.dtype)
            outputs.append(residual + layer.mlp(normalized))
        return outputs

    def _merged_attention_forward(
        self,
        modules: list[nn.Module],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        cache: DynamicCache | None,
        embeddings: list[torch.Tensor | None],
        use_cache: bool,
    ) -> tuple[list[torch.Tensor | None], DynamicCache | None]:
        for layer_idx in range(len(modules[0].layers)):
            embeddings = self._compute_merged_layer(
                layer_idx,
                modules,
                embeddings,
                position_ids,
                cache,
                attention_mask,
                use_cache,
            )
        outputs = [
            module.norm(hidden) if hidden is not None else None
            for module, hidden in zip(modules, embeddings, strict=True)
        ]
        return outputs, cache

    def get_prefix_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        images: torch.FloatTensor,
        image_masks: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states: list[torch.Tensor] = []
        padding_masks: list[torch.Tensor] = []
        attention_groups: list[int] = []
        for view, view_mask in zip(
            images.transpose(0, 1),
            image_masks.transpose(0, 1),
            strict=True,
        ):
            image_hidden = self.model.embed_image(view)
            batch_size, token_count = image_hidden.shape[:2]
            hidden_states.append(image_hidden)
            padding_masks.append(view_mask[:, None].expand(batch_size, token_count))
            attention_groups.extend([1] * token_count)

        text_hidden = self.model.embed_language_tokens(input_ids)
        hidden_states.append(text_hidden)
        padding_masks.append(attention_mask.bool())
        attention_groups.extend([1] * text_hidden.shape[1])
        hidden = torch.cat(hidden_states, dim=1)
        padding = torch.cat(padding_masks, dim=1)
        groups = torch.tensor(
            attention_groups,
            dtype=torch.int32,
            device=hidden.device,
        )[None].expand(padding.shape[0], -1)
        return hidden, padding, groups

    def get_suffix_hidden_states(
        self,
        noisy_actions: torch.FloatTensor,
        time: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_embedding = posemb_sincos(
            time,
            self.model.action_in_proj.out_features,
        ).to(dtype=noisy_actions.dtype)
        action_embedding = self.model.action_in_proj(noisy_actions)
        expanded_time = time_embedding[:, None, :].expand_as(action_embedding)
        hidden = self.model.action_time_mlp_out(
            F.silu(self.model.action_time_mlp_in(torch.cat((action_embedding, expanded_time), dim=2)))
        )
        batch_size, action_length = hidden.shape[:2]
        padding = torch.ones(
            batch_size,
            action_length,
            device=hidden.device,
            dtype=torch.bool,
        )
        groups = torch.tensor(
            [1] + [0] * (action_length - 1),
            device=hidden.device,
            dtype=torch.int32,
        )[None].expand(batch_size, -1)
        return hidden, padding, groups

    @torch.inference_mode()
    def inference_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        states: torch.FloatTensor,
        images: torch.FloatTensor,
        image_masks: torch.BoolTensor,
        diffusion_steps: int = 10,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        batch_size = states.shape[0]
        dt = -1.0 / diffusion_steps
        noise = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.config.action_dim,
            device=states.device,
            dtype=states.dtype,
            generator=generator,
        )
        time = torch.tensor(1.0, device=states.device, dtype=states.dtype)
        prefix, prefix_padding, prefix_groups = self.get_prefix_hidden_states(
            input_ids,
            attention_mask,
            images,
            image_masks,
        )
        prefix_mask = make_attn_mask_4d(
            make_attn_mask_2d(prefix_padding, prefix_groups),
            prefix.dtype,
        )
        positions = torch.cumsum(prefix_padding, dim=1) - 1
        modules = [self.model.llm, self.model.action_expert.model]
        _, cache = self._merged_attention_forward(
            modules,
            prefix_mask,
            positions,
            DynamicCache(),
            [prefix, None],
            True,
        )
        for _ in range(diffusion_steps):
            noise, time = self._denoise_step(
                noise,
                time,
                dt,
                batch_size,
                prefix_padding,
                prefix_groups,
                modules,
                cache,
            )
        return noise

    def _denoise_step(
        self,
        actions: torch.Tensor,
        time: torch.Tensor,
        dt: float,
        batch_size: int,
        prefix_padding: torch.Tensor,
        prefix_groups: torch.Tensor,
        modules: list[nn.Module],
        cache: DynamicCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix, suffix_padding, suffix_groups = self.get_suffix_hidden_states(
            actions,
            time.broadcast_to(batch_size),
        )
        attention_mask = make_attn_mask_4d(
            make_suffix_attn_mask_2d(
                suffix_padding,
                suffix_groups,
                prefix_padding,
                prefix_groups,
            ),
            suffix.dtype,
        )
        offsets = torch.sum(prefix_padding, dim=-1)[:, None]
        positions = offsets + torch.cumsum(suffix_padding, dim=1) - 1
        outputs, _ = self._merged_attention_forward(
            modules,
            attention_mask,
            positions,
            cache,
            [None, suffix],
            False,
        )
        suffix_output = outputs[1]
        if suffix_output is None:
            raise RuntimeError("DM0 action expert produced no suffix output")
        velocity = self.model.action_out_proj(suffix_output[:, -self.config.chunk_size :])
        return actions + velocity * dt, time + dt
