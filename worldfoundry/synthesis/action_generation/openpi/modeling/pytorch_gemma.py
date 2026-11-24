from typing import Literal

import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING, AutoModel

from . import hf_gemma as modeling_gemma
from .hf_gemma_config import GemmaConfig


class _PaliGemmaProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


class _PaliGemmaCore(nn.Module):
    """Minimal PaliGemma backbone with checkpoint-compatible module names."""

    def __init__(self, config, text_config: GemmaConfig):
        super().__init__()
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = _PaliGemmaProjector(config)
        self.language_model = modeling_gemma.GemmaModel(text_config)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(image_outputs.last_hidden_state)


class _PaliGemmaForInference(nn.Module):
    """PaliGemma container matching official OpenPI safetensor keys."""

    def __init__(self, config, text_config: GemmaConfig):
        super().__init__()
        self.config = config
        self.config.text_config = text_config
        self.model = _PaliGemmaCore(config, text_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

    @property
    def language_model(self):
        return self.model.language_model


class _GemmaExpertForInference(nn.Module):
    """Expert container matching GemmaForCausalLM checkpoint keys."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.model = modeling_gemma.GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        vlm_text_config = GemmaConfig(
            head_dim=vlm_config.head_dim,
            hidden_size=vlm_config.width,
            intermediate_size=vlm_config.mlp_dim,
            num_attention_heads=vlm_config.num_heads,
            num_hidden_layers=vlm_config.depth,
            num_key_value_heads=vlm_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[0],
            adarms_cond_dim=vlm_config.width if use_adarms[0] else None,
        )
        action_expert_config_hf = GemmaConfig(
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = _PaliGemmaForInference(vlm_config_hf, vlm_text_config)
        self.gemma_expert = _GemmaExpertForInference(action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self.requires_grad_(False)
        self.eval()

    def to_bfloat16_for_selected_params(
        self,
        precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float16":
            self.to(dtype=torch.float16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            for layer_idx in range(num_layers):
                query_states = []
                key_states = []
                value_states = []
                gates = []
                for model, hidden_states, cond in zip(models, inputs_embeds, adarms_cond, strict=True):
                    layer = model.layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=cond)
                    gates.append(gate)
                    hidden_shape = (
                        *hidden_states.shape[:-1],
                        -1,
                        layer.self_attn.head_dim,
                    )
                    query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
                    key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
                    value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)
                rotary_input = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = models[0].rotary_emb(rotary_input, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )
                attention_output, _ = modeling_gemma.eager_attention_forward(
                    models[0].layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    models[0].layers[layer_idx].self_attn.scaling,
                )
                attention_output = attention_output.reshape(
                    query_states.shape[0],
                    -1,
                    query_states.shape[1] * query_states.shape[-1],
                )

                next_embeds = []
                start_pos = 0
                for model, hidden_states, gate, cond in zip(models, inputs_embeds, gates, adarms_cond, strict=True):
                    layer = model.layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]
                    layer_attention = attention_output[:, start_pos:end_pos]
                    layer_attention = layer_attention.to(layer.self_attn.o_proj.weight.dtype)
                    residual = modeling_gemma._gated_residual(
                        hidden_states,
                        layer.self_attn.o_proj(layer_attention),
                        gate,
                    )
                    normalized, gate = layer.post_attention_layernorm(residual, cond=cond)
                    normalized = normalized.to(layer.mlp.up_proj.weight.dtype)
                    next_embeds.append(
                        modeling_gemma._gated_residual(
                            residual,
                            layer.mlp(normalized),
                            gate,
                        )
                    )
                    start_pos = end_pos
                inputs_embeds = next_embeds

            outputs_embeds = [
                model.norm(hidden_states, cond=cond)[0]
                for model, hidden_states, cond in zip(models, inputs_embeds, adarms_cond, strict=True)
            ]
            prefix_output, suffix_output = outputs_embeds
            prefix_past_key_values = None
        return [prefix_output, suffix_output], prefix_past_key_values
