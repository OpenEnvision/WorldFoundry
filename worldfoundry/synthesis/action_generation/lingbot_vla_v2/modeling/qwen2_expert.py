"""Qwen2 action expert and routed-MoE inference layers."""

from typing import ClassVar

import torch
import torch.nn.functional as F
import transformers.models.qwen2.modeling_qwen2 as hf_qwen2
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    PreTrainedModel,
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as _Qwen2ForCausalLM,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model as _Qwen2Model,
)
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, logging

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenRMSNorm as FixQwen2RMSNorm

logger = logging.get_logger(__name__)


class Qwen2FusedExperts(nn.Module):
    """Checkpoint-compatible packed expert weights with inference dispatch.

    Shape convention matches nn.Linear(in, out).weight = [out, in]:
      gate_proj: [E, intermediate_size, hidden_size]
      up_proj:   [E, intermediate_size, hidden_size]
      down_proj:  [E, hidden_size, intermediate_size]

    The fast inference path is selected by :class:`Qwen2TokenMoeBlock`; this
    module provides the portable PyTorch implementation for unsupported GPUs
    or Triton build failures without changing checkpoint parameter names.
    """

    def __init__(self, num_experts, hidden_size, intermediate_size, initializer_range=0.02):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.register_buffer("_gate_up_proj_cache", None, persistent=False)
        self._gate_up_proj_cache_key = None
        self._moe_workspace = None
        self._moe_workspace_key = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.up_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=self.initializer_range)
        self.clear_inference_cache()

    def clear_inference_cache(self):
        self._gate_up_proj_cache = None
        self._gate_up_proj_cache_key = None
        self._moe_workspace = None
        self._moe_workspace_key = None

    def _get_moe_workspace(self, hidden_states, top_k):
        if self.training or torch.is_grad_enabled() or not hidden_states.is_cuda:
            return None
        num_tokens, hidden_size = hidden_states.shape
        key = (
            num_tokens,
            int(top_k),
            self.num_experts,
            hidden_size,
            self.intermediate_size,
            hidden_states.dtype,
            hidden_states.device,
        )
        if self._moe_workspace is None or self._moe_workspace_key != key:
            max_routes = num_tokens * int(top_k)
            self._moe_workspace = {
                "counts": torch.empty((self.num_experts,), device=hidden_states.device, dtype=torch.int32),
                "rows": torch.empty((self.num_experts, max_routes), device=hidden_states.device, dtype=torch.int32),
                "slots": torch.empty((self.num_experts, max_routes), device=hidden_states.device, dtype=torch.int32),
                "inter": torch.empty(
                    (num_tokens, int(top_k), self.intermediate_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                "out": torch.empty((num_tokens, hidden_size), device=hidden_states.device, dtype=torch.float32),
            }
            self._moe_workspace_key = key
        return self._moe_workspace

    def forward(self, module, num_experts, routing_weights, selected_experts, hidden_states):
        del module
        if num_experts != self.num_experts:
            raise ValueError(f"expected {self.num_experts} experts, got {num_experts}")
        from worldfoundry.core.kernels import routed_swiglu_moe

        allow_triton = not self.training and not torch.is_grad_enabled()
        return routed_swiglu_moe(
            hidden_states,
            routing_weights,
            selected_experts,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            workspace=lambda: self._get_moe_workspace(hidden_states, selected_experts.shape[1]),
            allow_triton=allow_triton,
        )


class Qwen2TokenMoeBlock(nn.Module):
    """Token-level routing MoE block with all-to-all computation for torch.compile compatibility."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Checkpoint-resident loss-free routing bias used during inference.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts),
            persistent=True,
        )

        # gating (per-token)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # EP/fused support: choose expert storage based on moe_implementation
        self._moe_implementation = getattr(config, "_moe_implementation", None) or "eager"
        if self._moe_implementation == "fused":
            self.experts = Qwen2FusedExperts(
                self.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
                initializer_range=getattr(config, "initializer_range", 0.02),
            )
        else:
            self.experts = nn.ModuleList(
                [QwenMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
            )

        self.shared_expert = QwenMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self._router_activation = getattr(config, "router_activation", "softmax")
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self._use_shared_expert_gate = getattr(config, "use_shared_expert_gate", True)
        if self._use_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Token-level routing with all-to-all computation for torch.compile compatibility."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        _num_tokens = batch_size * sequence_length

        # Token-level routing: each token individually
        hidden_flat = hidden_states.reshape(-1, hidden_dim)  # (B*T, D)
        # Gate in true fp32 (autocast disabled): bf16 gate logits can flip top-k
        # selection on near-equal scores -> routing jitter / rotating dead experts.
        # cf. VideoPretrain lumos/moe/router.py TokenChoiceTopKRouter.
        with torch.amp.autocast(hidden_flat.device.type, enabled=False):
            router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())  # (B*T, num_experts)

        if self._router_activation == "sigmoid":
            routing_scores = router_logits.sigmoid()
        else:
            routing_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

        scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor != 1.0:
            routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Packed experts use the shared cached Triton/PyTorch dispatcher.
        if self._moe_implementation == "fused":
            final_hidden_states = self.experts(
                module=self,
                num_experts=self.num_experts,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                hidden_states=hidden_flat,
            )
        else:
            # Original eager path: every expert processes all tokens
            expert_outputs = torch.stack(
                [expert(hidden_flat) for expert in self.experts], dim=0
            )  # (num_experts, B*T, D)
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()  # (B*T, top_k, num_experts)
            weights = (
                (expert_mask * routing_weights.unsqueeze(-1).float()).sum(dim=1).to(hidden_states.dtype)
            )  # (B*T, num_experts)
            final_hidden_states = torch.einsum("ebd,be->bd", expert_outputs, weights)  # (B*T, D)

        # Shared expert: applied to all tokens (fixed shape)
        if final_hidden_states.dtype != hidden_flat.dtype:
            final_hidden_states = final_hidden_states.to(hidden_flat.dtype)
        shared_expert_output = self.shared_expert(hidden_flat)
        if self._use_shared_expert_gate:
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_flat)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor | None = None,
        start: int | None = 0,
        end: int | None = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        ada_cond: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # Ensure input dtypes match weight dtype (needed for gradient checkpointing
        # recomputation where autocast context is lost)
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)
        if ada_cond is not None:
            ada_cond = ada_cond.to(param_dtype)

        if compute_kqv:
            if ada_cond is not None:
                hidden_states = self.input_layernorm(hidden_states, ada_cond)
            else:
                hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

            query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)

            return query_state, key_state, value_state

        elif output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])

            # first residual
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            if ada_cond is not None:
                out_emb = self.post_attention_layernorm(out_emb, ada_cond)
            else:
                out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            # Handle MoE block returning (hidden_states, router_logits)
            router_logits = None
            if isinstance(out_emb, tuple):
                out_emb, router_logits = out_emb
            # second residual
            out_emb += after_first_residual

            return out_emb, router_logits

        else:
            raise ValueError(
                f"Invaild Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2DecoderLayer in LingBot-VLA"
            )


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]  # noqa: RUF012 - Shared class-level value, as upstream.
    _skip_keys_device_placement = ["past_key_values"]  # noqa: RUF012 - Shared class-level value, as upstream.
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {  # noqa: RUF012 - Shared class-level value, as upstream.
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2FusedExperts):
            module.initializer_range = std
            module.reset_parameters()


class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    get_input_embeddings = _Qwen2Model.get_input_embeddings
    set_input_embeddings = _Qwen2Model.set_input_embeddings
    forward = _Qwen2Model.forward

    def __init__(self, config: Qwen2Config, eval=False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = FixQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        if eval:
            self._init_weights = lambda module: None
        self.post_init()


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys: ClassVar[list] = ["lm_head.weight"]
    _tp_plan: ClassVar[dict] = {"lm_head": "colwise_rep"}
    _pp_plan: ClassVar[dict] = {"lm_head": (["hidden_states"], ["logits"])}
    get_input_embeddings = _Qwen2ForCausalLM.get_input_embeddings
    set_input_embeddings = _Qwen2ForCausalLM.set_input_embeddings
    get_output_embeddings = _Qwen2ForCausalLM.get_output_embeddings
    set_output_embeddings = _Qwen2ForCausalLM.set_output_embeddings
    forward = _Qwen2ForCausalLM.forward
    set_decoder = _Qwen2ForCausalLM.set_decoder
    get_decoder = _Qwen2ForCausalLM.get_decoder

    def __init__(self, config, eval):
        super().__init__(config)
        self.model = Qwen2Model(config, eval)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


def apply_lingbot_qwen2_patch():
    hf_qwen2.Qwen2DecoderLayer = Qwen2DecoderLayer
    hf_qwen2.Qwen2PreTrainedModel = Qwen2PreTrainedModel
    hf_qwen2.Qwen2Model = Qwen2Model
    hf_qwen2.Qwen2ForCausalLM = Qwen2ForCausalLM
