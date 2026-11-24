# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hy-VLA dual-tower module.

Modified by WorldFoundry from the upstream inference implementation: imports
and the documentation example are package-relative, training-only freeze /
unfreeze helpers are omitted, and precision follows the selected inference
dtype. The official BF16 path retains its original numerical dtype.

``HyDualTower`` pairs a VLM with an action-expert decoder under a
shared-attention forward. Both slots are architecture-neutral
``nn.Module`` attributes; any HuggingFace-compatible CausalLM that
exposes the standard transformers decoder-layer contract
(``layers[i].{input_layernorm, self_attn, post_attention_layernorm,
mlp}``, plus the MoT extensions ``input_layernorm_v`` /
``post_attention_layernorm_v`` / ``self_attn.{q,k,v,o}_proj_v`` and
``mlp_v`` for the modality-aware variant) can be dropped in.
"""

from typing import ClassVar

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache

from worldfoundry.core.attention.rotary.embedding import apply_rotary_pos_emb

from .modeling_hunyuan_vl_mot import (
    HunYuanVLMoTForConditionalGeneration,
    _HunYuanVLMoTTextForCausalLM,
)


def mask_apply(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    text_funcs,
    vision_funcs,
    out_dims=None,
):
    """Batch-flattened modality routing for the MoT dual-tower forward.

    Args:
        hidden_states: ``(B, S, D)`` token features.
        mask: ``(B, S)`` bool / int. ``True`` (or ``1``) -> vision token,
            ``False`` (or ``0``) -> text token.
        text_funcs: callables applied to text tokens (one per output).
        vision_funcs: callables applied to vision tokens (one per output).
        out_dims: optional list of per-output last-dim sizes. ``None``
            means each output keeps the input ``D``.

    Returns:
        ``list[Tensor]`` with shape ``(B, S, out_dim_i)``; entries the
        functions did not write are zeros (``torch.empty`` slots that
        were never indexed are explicitly zero-initialised when neither
        modality covers the full sequence, see below).
    """
    B, S, D = hidden_states.size()
    flat = hidden_states.reshape(B * S, D)
    mask_flat = mask.reshape(B * S).bool()

    if out_dims is None:
        out_flat = [torch.zeros(B * S, D, device=flat.device, dtype=flat.dtype) for _ in text_funcs]
    else:
        out_flat = [torch.zeros(B * S, od, device=flat.device, dtype=flat.dtype) for od in out_dims]

    text_idx = ~mask_flat
    if text_idx.any():
        hs_t = flat[text_idx]
        for i, fn in enumerate(text_funcs):
            out_flat[i][text_idx] = fn(hs_t)

    vis_idx = mask_flat
    if vis_idx.any():
        hs_v = flat[vis_idx]
        for i, fn in enumerate(vision_funcs):
            out_flat[i][vis_idx] = fn(hs_v)

    return [o.view(B, S, -1) for o in out_flat]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class HyDualTowerConfig(PretrainedConfig):
    """Config for :class:`HyDualTower`.

    Both ``vlm_config`` and ``expert_config`` are full ``PretrainedConfig``
    instances and are passed in directly.
    """

    model_type = "hy_dual_tower"
    sub_configs: ClassVar[dict] = {"vlm_config": AutoConfig, "expert_config": AutoConfig}

    def __init__(
        self,
        vlm_config: PretrainedConfig | None = None,
        expert_config: PretrainedConfig | None = None,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.vlm_config = vlm_config
        self.expert_config = expert_config

        self.attention_implementation = attention_implementation

        # Optional reference to the outer ``HyVLAConfig`` (used by
        # HyVLAFlowMatching to keep its proj_width in sync with the
        # expert tower's hidden_size).
        self.config = kwargs.get("config")

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` "
                f"({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HyDualTower(PreTrainedModel):
    """Plug-in dual-tower container: VLM + action expert with shared attention.

    The two slot attributes ``self.vlm`` and ``self.expert`` are
    architecture-neutral. The default factory uses
    :class:`HunYuanVLMoTForConditionalGeneration` and
    :class:`HunYuanDenseV1MoTForCausalLM`; any HuggingFace-style decoder
    that satisfies the modality-aware MoT layer contract can be plugged
    in via :meth:`from_components`.

    State-dict layout:
        ``vlm.<rest>``     -- VLM weights
        ``expert.<rest>``  -- expert weights
    """

    config_class = HyDualTowerConfig

    def __init__(
        self,
        config: HyDualTowerConfig,
        *,
        inference_dtype: torch.dtype | None = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.vlm = HunYuanVLMoTForConditionalGeneration(config=config.vlm_config)
        # Action-expert: use the decoder wrapper to preserve
        # ``expert.model.{layers,norm,rotary_emb}``, then drop the tokenizer
        # input/output modules exactly as the official checkpoint build does
        # in ``train.py``. Inference consumes the expert hidden states through
        # ``action_out_proj`` and never reads an expert language-model head.
        self.expert = _HunYuanVLMoTTextForCausalLM(config=config.expert_config)
        self.expert.model.embed_tokens = None
        self.expert.lm_head = None

        self.to_inference_dtype(inference_dtype or torch.bfloat16)

    # ------------------------------------------------------------------
    # Component-level factory (the "plug-in" contract).
    # ------------------------------------------------------------------
    @classmethod
    def from_components(
        cls,
        *,
        vlm: PreTrainedModel,
        expert: PreTrainedModel,
        attention_implementation: str = "eager",
        outer_config: PretrainedConfig | None = None,
    ) -> "HyDualTower":
        """Build a ``HyDualTower`` from pre-instantiated VLM / expert modules.

        The returned tower assumes ownership of both modules; the caller
        should not keep separate references. Useful for swapping in
        custom backbones without subclassing.

        Example::

            from worldfoundry.synthesis.action_generation.hy_embodied_vla import HyDualTower

            tower = HyDualTower.from_components(
                vlm=MyCustomVLM.from_pretrained("..."),
                expert=MyCustomExpert.from_pretrained("..."),
            )
        """
        cfg = HyDualTowerConfig(
            vlm_config=vlm.config,
            expert_config=expert.config,
            attention_implementation=attention_implementation,
            config=outer_config,
        )
        instance = cls.__new__(cls)
        PreTrainedModel.__init__(instance, config=cfg)
        instance.config = cfg
        instance.vlm = vlm
        instance.expert = expert
        if instance.expert.model.embed_tokens is not None:
            instance.expert.model.embed_tokens = None
        if instance.expert.lm_head is not None:
            instance.expert.lm_head = None
        instance.to_inference_dtype(next(vlm.parameters()).dtype)
        return instance

    def to_inference_dtype(self, dtype: torch.dtype):
        """Apply the upstream mixed-precision layout using ``dtype``.

        The official CUDA path passes ``torch.bfloat16``. WorldFoundry may
        pass ``torch.float16`` on GPUs without BF16 support; parameter names
        and the checkpoint layout are unchanged.
        """
        if not dtype.is_floating_point:
            raise TypeError(f"HyDualTower inference dtype must be floating point, got {dtype}")
        self.vlm = self.vlm.to(dtype=dtype)

        params_to_change_dtype = [
            "language_model.model.layers",
            "expert.model.layers",
            "visual",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Visual + language token embedders (called by the outer policy)
    # ------------------------------------------------------------------
    def embed_image(self, image: torch.Tensor):
        """Encode RGB inputs through the VLM vision tower.

        Args:
            image: ``(C, H, W)`` (single frame), ``(B, C, H, W)`` (batch
                of single frames), or ``(B, K, C, H, W)`` (MEM video
                stack -- routed to the SpaceTime-augmented ViT).

        Returns:
            ``(B, N, D)`` patch features, where ``N`` is the per-frame
            (or per-stack) token count and ``D`` is the visual hidden
            dim.
        """
        if image.dim() == 5:
            # Wrapper returns [(B*N, C)] (batch flattened); restore to (B, N, C).
            B = image.shape[0]
            feat = self.vlm.visual(image)[0]
            return feat.view(B, -1, feat.shape[-1]).contiguous()

        image_list = list(image.unsqueeze(1) if image.dim() == 3 else image.split(1, dim=0))
        # image_list: list of (1, 3, h, w)
        image_features = self.vlm.visual(image_list)  # list of (num_tokens, 2048)
        image_features = torch.stack(image_features, dim=0)
        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Look up text-token embeddings via the VLM language tower."""
        return self.vlm.language_model.model.embed_tokens(tokens)

    # ------------------------------------------------------------------
    # Shared-attention dual-tower forward
    # ------------------------------------------------------------------
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        modality_masks: list[torch.FloatTensor] | None = None,
    ):
        models = [self.vlm.language_model.model, self.expert.model]
        att_vis_output = []
        prefix_emb_layer_outputs = []
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.vlm.config.num_hidden_layers

        # ``position_embeddings`` are constant across layers; compute once.
        # The first arg picks output dtype/device, so we pass a float
        # tensor (not the int64 ``position_ids``).
        _dtype_ref = next(h for h in inputs_embeds if h is not None)
        position_embeddings = models[0].rotary_emb(_dtype_ref.float(), position_ids)

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []

            # Per-tower sequence length (used to slice the concatenated
            # q/k tensors before the per-tower q/k layernorm).
            seq_len_list = []

            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                modality_mask = modality_masks[i]

                hidden_states = mask_apply(
                    hidden_states,
                    modality_mask,
                    [lambda x: layer.input_layernorm(x)],  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                    [lambda x: layer.input_layernorm_v(x)],  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                )[0]

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                # Batch-flattened modality routing (see ``mask_apply``
                # docstring). The dual-tower forward always supplies a
                # non-None ``modality_mask`` so we bypass the vendor's
                # per-sample fallback path entirely.
                query_state, key_state, value_state = mask_apply(
                    hidden_states,
                    modality_mask,
                    [
                        lambda x: layer.self_attn.q_proj(x),  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        lambda x: layer.self_attn.k_proj(x),  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        lambda x: layer.self_attn.v_proj(x),  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                    ],
                    [
                        lambda x: layer.self_attn.q_proj_v(x),  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        lambda x: layer.self_attn.k_proj_v(x),  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        lambda x: layer.self_attn.v_proj_v(x),  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                    ],
                    out_dims=[
                        self.config.vlm_config.num_attention_heads * layer.self_attn.head_dim,
                        self.config.vlm_config.num_key_value_heads * layer.self_attn.head_dim,
                        self.config.vlm_config.num_key_value_heads * layer.self_attn.head_dim,
                    ],
                )

                # (batch_size, num_heads, seq_len, head_dim)
                query_state = query_state.view(hidden_shape).transpose(1, 2)
                key_state = key_state.view(hidden_shape).transpose(1, 2)
                value_state = value_state.view(hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
                seq_len_list.append(hidden_states.shape[1])

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            q_parts = query_states.split(seq_len_list, dim=2)
            k_parts = key_states.split(seq_len_list, dim=2)
            q_normed = []
            k_normed = []

            vlm_layer = models[0].layers[layer_idx]
            for q_part, k_part in zip(q_parts, k_parts, strict=False):
                q_normed.append(vlm_layer.self_attn.query_layernorm(q_part))
                k_normed.append(vlm_layer.self_attn.key_layernorm(k_part))
            query_states = torch.cat(q_normed, dim=2)
            key_states = torch.cat(k_normed, dim=2)

            # (batch_size, seq_len, num_heads, head_dim)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
                    past_key_values[layer_idx]["key_states"] = key_states
                    past_key_values[layer_idx]["value_states"] = value_states

            attention_interface = self.get_attention_interface()
            att_output, probs = attention_interface(
                attention_mask,
                batch_size,
                layer.self_attn.head_dim,
                query_states,
                key_states,
                value_states,
            )

            att_output = att_output.to(dtype=value_states.dtype)  # (b, seq_vlm, ...)
            att_vis_output.append(probs)  # probs (b, 8, seq, seq)

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                modality_mask = modality_masks[i]
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

                    out_emb = mask_apply(
                        att_output[:, start:end],
                        modality_mask,
                        [lambda x: layer.self_attn.o_proj(x)],  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        [lambda x: layer.self_attn.o_proj_v(x)],  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        out_dims=[models[i].config.hidden_size],
                    )[0]

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()
                    out_emb = mask_apply(
                        out_emb,
                        modality_mask,
                        [lambda x: layer.mlp(layer.post_attention_layernorm(x))],  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                        [lambda x: layer.mlp_v(layer.post_attention_layernorm_v(x))],  # noqa: B023 - Keeps the upstream late binding of the loop variable.
                    )[0]

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            prefix_emb_layer_outputs.append(outputs_embeds[0])
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values, att_vis_output, prefix_emb_layer_outputs

    # ------------------------------------------------------------------
    # Attention backends
    # ------------------------------------------------------------------
    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            return self.flash_attention_forward
        return self.eager_attention_forward

    def flash_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        num_att_heads = self.config.vlm_config.num_attention_heads
        num_key_value_heads = self.config.vlm_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # bf16 -inf approximation

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output, probs


__all__ = ["HyDualTower", "HyDualTowerConfig"]
