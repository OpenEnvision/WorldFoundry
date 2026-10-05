"""MAGI-2-preview DiT top model (single-GPU PyTorch port).

Ported faithfully from SandAI's Apache-2.0 ``model/magi2_preview.py``. This
module holds the composite modules:

* :class:`Attention` (pre-norm -> gate + QKV -> q/k RMSNorm -> RoPE ->
  ``ops.attention_with_sink`` -> scalar sigmoid gate -> proj),
* :class:`MLP` (dense SwiGLU7, per-modality),
* :class:`CoreMultiHeadMoE` + :class:`MultiHeadMoELayer` (12-head x 256-expert
  top-6 routed MoE via ``ops.flash_mh_moe_*`` + two dense shared-expert paths),
* :class:`PreAdapter` / :class:`PostAdapter` (fp32 embedders / output heads),
* :class:`TransformerLayer` (4-stream MHC hyper-connection wrapping attn+ffn),
* :class:`TransformerBlock` and :class:`Transformer` (top model).

Single-GPU collapse (cp = ep = dp = 1): Ulysses dispatch/undispatch removed
(``cp_split_sizes = [total_tokens]``), CP scatter/gather in attention removed
(``ops.attention_with_sink`` called directly), EP dispatch/undispatch in MoE
removed (padded_num_heads == num_heads, all experts local / unsharded). The
upstream compile / custom-op decorators, the FA3 + fused-attention-with-sink
imports and the Triton MoE kernel import are all dropped in favour of the
pure-torch fallbacks in :mod:`ops`. Mixed dtype (fp32 embedders / adapters /
heads / sinks / norms, bf16 blocks) and every checkpoint-facing parameter name
are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .config import Magi2PreviewConfig, MoEConfig
from .ops import (
    attention_with_sink,
    compute_topk_probs_and_indices,
    flash_mh_moe_fwd,
    flash_mh_moe_global_sort,
    swiglu7_split,
)
from .preview_layers import (
    ElementWiseFourierEmbed,
    MHCHandler,
    MLPActivationType,
    Modality,
    ModalityDispatcher,
    MultiModalityRMSNorm,
    VarlenHandler,
    apply_rotary_emb_torch,
    create_activation_func,
    create_linear,
)

_FP32 = torch.float32
_BF16 = torch.bfloat16


# ============================================================
# Attention
# ============================================================


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    num_modality: int
    num_layers: int
    attn_softcap: float = -1.0
    sink_token_num: int = 1


class Attention(nn.Module):
    """Preview self-attention: pre-norm, QKV + per-head sink, RoPE, gated proj.

    Tokens stay modality-sorted for the grouped linears, then are un-permuted so
    RoPE and ``attention_with_sink`` see the packed-document order described by
    ``cu_seqlens``.  A learned per-head sink logit is mixed into the softmax
    normalizer (FA3 path) or the score matrix (PyTorch reference).
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, eps=1e-6, num_modality=config.num_modality
        )

        self.head_dim = config.head_dim
        self.num_heads_q = config.num_heads_q
        self.num_heads_kv = config.num_heads_kv
        self.q_size = self.num_heads_q * self.head_dim
        self.kv_size = self.num_heads_kv * self.head_dim
        self.gating_total_size = self.num_heads_q

        out_dim = self.q_size + self.kv_size * 2
        self.linear_g = create_linear(
            config.hidden_size,
            self.gating_total_size,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.linear_qkv = create_linear(
            config.hidden_size,
            out_dim,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.linear_proj = create_linear(
            self.q_size,
            config.hidden_size,
            bias=False,
            num_experts=config.num_modality,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        # Per-head sink logit(s); fp32. Shape (max(1, sink_token_num), num_heads_q).
        self.sinks = nn.Parameter(
            torch.empty(max(1, config.sink_token_num), config.num_heads_q, dtype=_FP32)
        )

        self.q_norm = MultiModalityRMSNorm(
            config.head_dim, num_modality=config.num_modality, out_dtype=_FP32
        )
        self.k_norm = MultiModalityRMSNorm(
            config.head_dim, num_modality=config.num_modality, out_dtype=_FP32
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        t = hidden_states.size(0)
        x = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher)
        g = self.linear_g(x, modality_dispatcher=modality_dispatcher)
        qkv = self.linear_qkv(x, modality_dispatcher=modality_dispatcher)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=1)

        q = q.view(t, self.num_heads_q, self.head_dim).clone()
        k = k.view(t, self.num_heads_kv, self.head_dim).clone()
        v = v.view(t, self.num_heads_kv, self.head_dim)
        g = g.view(t, self.num_heads_q, 1).clone()

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        # Undo the modality permutation so RoPE / attention see original packing
        # order (cu_seqlens describes doc boundaries in that order).
        q = modality_dispatcher._inv_permute(q).unsqueeze(0)
        k = modality_dispatcher._inv_permute(k).unsqueeze(0)
        v = modality_dispatcher._inv_permute(v).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

        # Single-GPU: no CP scatter/gather -> call the attention core directly.
        q = q.squeeze(0).to(_BF16)
        k = k.squeeze(0).to(_BF16)
        v = v.squeeze(0).to(_BF16)
        out = attention_with_sink(
            q, k, v, self.sinks, cu_seqlens=varlen_handler.cu_seqlens_q
        )

        # Back to modality order for the (modality-order) gate + projection.
        out = modality_dispatcher._permute(out)
        out = out * torch.sigmoid(g)
        out = out.reshape(-1, self.q_size).to(_BF16)
        return self.linear_proj(out, modality_dispatcher=modality_dispatcher)


# ============================================================
# dense MLP (non-MoE layers)
# ============================================================


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(nn.Module):
    """Dense SwiGLU7 feed-forward used on non-MoE preview layers."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, num_modality=config.num_modality
        )
        intermediate_size_up = (
            config.intermediate_size * 2 if config.gated_act else config.intermediate_size
        )
        self.up_gate_proj = create_linear(
            config.hidden_size,
            intermediate_size_up,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=config.num_modality,
        )
        self.down_proj = create_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=config.num_modality,
        )
        self.activation_func = create_activation_func(config.activation_type)

    def forward(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher)
        x = self.activation_func(x)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher)
        return x


# ============================================================
# multi-head MoE
# ============================================================


@dataclass
class CoreMultiHeadMoEConfig:
    hidden_size: int
    num_heads: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    num_layers: int
    params_dtype: torch.dtype
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0


class CoreMultiHeadMoE(nn.Module):
    """12-head x 256-expert top-6 MoE. Single-GPU: ep_size == 1 (no EP split).

    Every head's experts are held locally / unsharded, so
    ``local_flatten_num_experts == num_heads * num_experts`` and there is no
    ``ep_dispatch`` / ``ep_undispatch`` / head padding.
    """

    def __init__(self, config: CoreMultiHeadMoEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.config = config
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.topk = config.top_k
        self.d_head = config.hidden_size // config.num_heads
        self.d_expert = config.expert_intermediate_size
        self.flatten_num_experts = self.num_heads * self.num_experts

        # Single-GPU EP collapse: ep_size == 1.
        self.ep_size = 1
        self.padded_num_heads = self.num_heads
        self.ep_pad_heads = 0
        self.has_real_moe_heads = True
        self.local_num_heads = self.num_heads
        self.local_flatten_num_experts = self.local_num_heads * self.num_experts

        self.gate = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_head, dtype=_FP32)
        )
        self.W_gate = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts,
                self.d_head,
                self.d_expert,
                dtype=config.params_dtype,
            )
        )
        self.W_up = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts,
                self.d_head,
                self.d_expert,
                dtype=config.params_dtype,
            )
        )
        self.W_down = nn.Parameter(
            torch.empty(
                self.local_flatten_num_experts,
                self.d_expert,
                self.d_head,
                dtype=config.params_dtype,
            )
        )
        self.router = nn.Module()
        self.router.register_buffer(
            "expert_bias", torch.zeros(self.local_flatten_num_experts, dtype=_FP32)
        )
        self.router.register_buffer(
            "expert_bias_ema", torch.zeros(self.local_flatten_num_experts, dtype=_FP32)
        )

    def _route(self, x_heads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_heads = x_heads.size(1)
        gate = self.gate.view(num_heads, self.num_experts, self.d_head).float()
        router_logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
        expert_bias = self.router.expert_bias.view(num_heads, self.num_experts)
        topk_probs, topk_indices = compute_topk_probs_and_indices(
            router_logits=router_logits,
            top_k=self.topk,
            score_func=self.config.score_func,
            expert_bias=expert_bias,
            route_norm=self.config.route_norm,
        )
        topk_probs = topk_probs * self.config.route_scale
        return topk_probs, topk_indices

    def _flash_forward(
        self,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        gather_ids, probs, expert_offsets = flash_mh_moe_global_sort(
            topk_probs=topk_probs,
            topk_indices=topk_indices,
            num_experts=self.num_experts,
        )
        return flash_mh_moe_fwd(
            x=x_heads,
            gather_ids=gather_ids,
            probs=probs,
            expert_offsets=expert_offsets,
            W_gate=self.W_gate,
            W_up=self.W_up,
            W_down=self.W_down,
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_heads = x.view(-1, self.num_heads, self.d_head)
        topk_probs, topk_indices = self._route(x_heads)
        out = self._flash_forward(x_heads, topk_probs, topk_indices)
        return out.reshape(-1, self.num_heads * self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


@dataclass
class MultiHeadMoELayerConfig:
    hidden_size: int
    num_modality: int
    num_layers: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    shared_expert_intermediate_sizes: int
    modality_specific_expert_intermediate_sizes: int
    shared_expert_params_dtype: torch.dtype
    split_merge_intermediate_size: Optional[int]
    moe_config: MoEConfig


class MultiHeadMoELayer(nn.Module):
    """Routed multi-head MoE plus two dense shared-expert SwiGLU7 paths.

    ``moe_mlp`` is the 12-head × 256-expert top-6 router.  A modality-shared
    expert and a per-modality expert run in parallel and are added to the
    routed output.  Optional ``split_linear`` / ``merge_linear`` resize the
    MoE hidden width without changing the surrounding residual stream.
    """

    def __init__(self, config: MultiHeadMoELayerConfig) -> None:
        super().__init__()
        self.config = config
        moe_hidden_size = config.split_merge_intermediate_size or config.hidden_size
        self.moe_mlp = CoreMultiHeadMoE(
            CoreMultiHeadMoEConfig(
                hidden_size=moe_hidden_size,
                num_heads=config.moe_config.num_heads,
                num_experts=config.moe_config.num_experts,
                top_k=config.moe_config.top_k,
                expert_intermediate_size=config.moe_config.expert_intermediate_size,
                num_layers=config.num_layers,
                params_dtype=config.params_dtype,
                score_func=config.moe_config.score_func,
                route_norm=config.moe_config.route_norm,
                route_scale=config.moe_config.route_scale,
            )
        )
        self.is_swiglu = config.activation_type in [MLPActivationType.SWIGLU7]
        self.activation_func = create_activation_func(config.activation_type)
        self.shared_expert_fc1, self.shared_expert_fc2 = self._create_shared_expert_pair(
            config, config.shared_expert_intermediate_sizes, num_modality=1
        )
        (
            self.modality_specific_shared_expert_fc1,
            self.modality_specific_shared_expert_fc2,
        ) = self._create_shared_expert_pair(
            config,
            config.modality_specific_expert_intermediate_sizes,
            num_modality=config.num_modality,
        )
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, num_modality=config.num_modality
        )
        mid = config.split_merge_intermediate_size or config.hidden_size
        self.split_linear = create_linear(
            config.hidden_size,
            mid,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            bias=False,
        )
        self.merge_linear = create_linear(
            mid,
            config.hidden_size,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            bias=False,
        )

    @staticmethod
    def _create_shared_expert_pair(
        config: MultiHeadMoELayerConfig, intermediate_size: int, num_modality: int = 1
    ) -> tuple[nn.Module, nn.Module]:
        factor = 2 if config.activation_type in [MLPActivationType.SWIGLU7] else 1
        fc1 = create_linear(
            config.hidden_size,
            intermediate_size * factor,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            num_experts=num_modality,
            bias=False,
        )
        fc2 = create_linear(
            intermediate_size,
            config.hidden_size,
            num_layers=config.num_layers,
            dtype=config.shared_expert_params_dtype,
            num_experts=num_modality,
            bias=False,
        )
        return fc1, fc2

    def _shared_expert_forward(
        self, norm_output: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        x1 = self.shared_expert_fc1(norm_output)
        x2 = self.modality_specific_shared_expert_fc1(
            norm_output, modality_dispatcher=modality_dispatcher
        )
        x = torch.concat([x1, x2], dim=-1)
        x = self.activation_func(x)
        x1, x2 = x.split(
            [
                self.config.shared_expert_intermediate_sizes,
                self.config.modality_specific_expert_intermediate_sizes,
            ],
            dim=-1,
        )
        x1 = self.shared_expert_fc2(x1)
        x2 = self.modality_specific_shared_expert_fc2(
            x2.contiguous(), modality_dispatcher=modality_dispatcher
        )
        return x1 + x2

    def forward(
        self, hidden_states: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        norm_output = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher)
        moe_input = self.split_linear(norm_output)
        moe_out = self.moe_mlp(moe_input)
        moe_out = self.merge_linear(moe_out)
        return moe_out + self._shared_expert_forward(norm_output, modality_dispatcher)


# ============================================================
# adapters (fp32 embedders / output heads)
# ============================================================


@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    virtual_width_factor: int = 1


class PreAdapter(nn.Module):
    """fp32 per-modality embedders plus element-wise Fourier RoPE coords.

    Video / audio / text rows are scattered into a shared adapter-width buffer
    (hidden × MHC streams).  ``coords_mapping`` is encoded independently and
    returned as the RoPE table for every layer.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter_dim = config.hidden_size * config.virtual_width_factor
        self.video_embedder = nn.Linear(
            config.video_in_channels, self.adapter_dim, bias=True, dtype=_FP32
        )
        self.text_embedder = nn.Linear(
            config.text_in_channels, self.adapter_dim, bias=True, dtype=_FP32
        )
        self.audio_embedder = nn.Linear(
            config.audio_in_channels, self.adapter_dim, bias=True, dtype=_FP32
        )
        self.rope = ElementWiseFourierEmbed(
            config.hidden_size // config.num_attention_heads,
            in_pixels=False,
            learnable=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_idx: torch.Tensor,
        audio_idx: torch.Tensor,
        text_idx: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.rope(coords_mapping)
        output_x = torch.zeros(
            x.shape[0], self.adapter_dim, device=x.device, dtype=_FP32
        )
        if text_idx.numel() > 0:
            output_x.index_copy_(
                0,
                text_idx,
                self.text_embedder(
                    x.index_select(0, text_idx)[:, : self.config.text_in_channels]
                ),
            )
        if audio_idx.numel() > 0:
            output_x.index_copy_(
                0,
                audio_idx,
                self.audio_embedder(
                    x.index_select(0, audio_idx)[:, : self.config.audio_in_channels]
                ),
            )
        if video_idx.numel() > 0:
            output_x.index_copy_(
                0,
                video_idx,
                self.video_embedder(
                    x.index_select(0, video_idx)[:, : self.config.video_in_channels]
                ),
            )
        return output_x, rope


class PostAdapter(nn.Module):
    """fp32 per-modality RMSNorm + linear heads back to video/audio channels."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter_dim = config.hidden_size * config.virtual_width_factor
        self.final_norm_video = MultiModalityRMSNorm(self.adapter_dim)
        self.final_norm_audio = MultiModalityRMSNorm(self.adapter_dim)
        self.final_linear_video = nn.Linear(
            self.adapter_dim, config.video_in_channels, bias=False, dtype=_FP32
        )
        self.final_linear_audio = nn.Linear(
            self.adapter_dim, config.audio_in_channels, bias=False, dtype=_FP32
        )
        self.final_out_dim = max(config.video_in_channels, config.audio_in_channels)

    def forward(
        self, x: torch.Tensor, video_idx: torch.Tensor, audio_idx: torch.Tensor
    ) -> torch.Tensor:
        x_out = torch.zeros(
            x.shape[0], self.final_out_dim, device=x.device, dtype=_FP32
        )
        if video_idx.numel() > 0:
            x_video = self.final_norm_video(
                x.index_select(0, video_idx).to(self.final_norm_video.weight.dtype)
            )
            x_video = self.final_linear_video(x_video).to(_FP32)
            x_out[:, : self.config.video_in_channels].index_copy_(0, video_idx, x_video)
        if audio_idx.numel() > 0:
            x_audio = self.final_norm_audio(
                x.index_select(0, audio_idx).to(self.final_norm_audio.weight.dtype)
            )
            x_audio = self.final_linear_audio(x_audio).to(_FP32)
            x_out[:, : self.config.audio_in_channels].index_copy_(0, audio_idx, x_audio)
        return x_out


# ============================================================
# transformer layer (MHC-wrapped attn + ffn)
# ============================================================


class TransformerLayer(nn.Module):
    """One preview layer: 4-stream MHC wrapping attention then MLP or MoE.

    Layers in ``moe_config.moe_layers`` use :class:`MultiHeadMoELayer`; the
    multi-modal boundary layers (``mm_layers``) keep three grouped-linear
    experts so video / audio / text do not share the same QKV/FFN weights.
    """

    def __init__(self, model_config: Magi2PreviewConfig, layer_idx: int) -> None:
        super().__init__()
        num_modality = 3 if layer_idx in model_config.mm_layers else 1
        self.config_mhc = model_config.mhc_config
        self.attention = Attention(
            AttentionConfig(
                hidden_size=model_config.hidden_size,
                num_heads_q=model_config.num_heads_q,
                num_heads_kv=model_config.num_heads_kv,
                head_dim=model_config.head_dim,
                params_dtype=model_config.params_dtype,
                num_modality=num_modality,
                num_layers=model_config.num_layers,
                attn_softcap=model_config.attn_softcap,
                sink_token_num=model_config.attn_sinks.sink_token_num,
            )
        )
        activation_type = MLPActivationType(model_config.activation_type)
        gated_act = activation_type in [MLPActivationType.SWIGLU7]
        if layer_idx in model_config.moe_config.moe_layers:
            moe_cfg = model_config.moe_config
            self.mlp = MultiHeadMoELayer(
                MultiHeadMoELayerConfig(
                    hidden_size=model_config.hidden_size,
                    num_modality=3,
                    num_layers=model_config.num_layers,
                    activation_type=activation_type,
                    params_dtype=model_config.params_dtype,
                    shared_expert_intermediate_sizes=moe_cfg.shared_expert_intermediate_size,
                    modality_specific_expert_intermediate_sizes=moe_cfg.modality_specific_expert_intermediate_size,
                    shared_expert_params_dtype=_BF16,
                    split_merge_intermediate_size=moe_cfg.split_merge_intermediate_size,
                    moe_config=moe_cfg,
                )
            )
        else:
            intermediate_size = (
                int(model_config.hidden_size * model_config.intermediate_factor * 2 / 3)
                // 128
                * 128
                if gated_act
                else int(model_config.hidden_size * model_config.intermediate_factor)
            )
            self.mlp = MLP(
                MLPConfig(
                    hidden_size=model_config.hidden_size,
                    intermediate_size=intermediate_size,
                    activation_type=activation_type,
                    params_dtype=model_config.params_dtype,
                    num_modality=num_modality,
                    num_layers=model_config.num_layers,
                    gated_act=gated_act,
                )
            )
        self._init_mhc(model_config, num_modality)

    def _init_mhc(self, model_config: Magi2PreviewConfig, num_modality: int) -> None:
        n = self.config_mhc.num_stream
        c = model_config.hidden_size
        dtype = _FP32
        self.mhc_alpha_pre_attn = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_post_attn = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_res_attn = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_pre_mlp = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_post_mlp = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_alpha_res_mlp = nn.Parameter(
            torch.full((1,), float(self.config_mhc.alpha_init), dtype=dtype)
        )
        self.mhc_bias_pre_attn = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_post_attn = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_pre_mlp = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_post_mlp = nn.Parameter(torch.empty(n, dtype=dtype))
        self.mhc_bias_res_attn = nn.Parameter(torch.empty(n, n, dtype=dtype))
        self.mhc_bias_res_mlp = nn.Parameter(torch.empty(n, n, dtype=dtype))
        self.mhc_phi_fused_attn = nn.Parameter(
            torch.empty(n * c, n + n + (n * n), dtype=dtype)
        )
        self.mhc_phi_fused_mlp = nn.Parameter(
            torch.empty(n * c, n + n + (n * n), dtype=dtype)
        )
        self.mhc_norm = MultiModalityRMSNorm(
            c * n, num_modality=num_modality, out_dtype=dtype
        )
        self.mhc_handler = MHCHandler(
            n_stream=n, hidden_size=c, num_sk_iters=20, sk_eps=1e-12, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        n = self.config_mhc.num_stream
        s = hidden_states.reshape(hidden_states.shape[0], n, -1)
        assert s.is_contiguous()

        h_post_attn_, h_res_attn_, attn_in = self._forward_mhc_pre_attn(
            s, modality_dispatcher
        )
        attn_out = self.attention(
            attn_in, rope, varlen_handler, modality_dispatcher, cp_split_sizes
        )

        s_, h_post_mlp_, h_res_mlp_, mlp_in = self._forward_mhc_attn_to_mlp(
            s, attn_out, h_post_attn_, h_res_attn_, modality_dispatcher
        )
        mlp_out = self.mlp(mlp_in, modality_dispatcher)

        s = self._forward_mhc_post_mlp(s_, mlp_out, h_post_mlp_, h_res_mlp_)
        return s.reshape(s.shape[0], -1)

    def _forward_mhc_pre_attn(
        self, s: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_flat = self.mhc_handler.flatten_mstream(s)
        h_pre_attn_, h_post_attn_, h_res_attn_ = self.mhc_handler.apply_norm_and_compute_h_(
            x=s_flat,
            norm_fn=partial(self.mhc_norm, modality_dispatcher=modality_dispatcher),
            phi_fused=self.mhc_phi_fused_attn,
        )
        attn_in = self.mhc_handler.compute_and_apply_hpre(
            x=s,
            abh_pre=(self.mhc_alpha_pre_attn, self.mhc_bias_pre_attn, h_pre_attn_),
            out_dtype=s.dtype,
        )
        return h_post_attn_, h_res_attn_, attn_in

    def _forward_mhc_attn_to_mlp(
        self,
        s: torch.Tensor,
        attn_out: torch.Tensor,
        h_post_attn_: torch.Tensor,
        h_res_attn_: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_post_attn, h_res_attn = self.mhc_handler.compute_hpost_and_hres(
            abh_post=(self.mhc_alpha_post_attn, self.mhc_bias_post_attn, h_post_attn_),
            abh_res=(self.mhc_alpha_res_attn, self.mhc_bias_res_attn, h_res_attn_),
            out_dtype=s.dtype,
        )
        s_ = self.mhc_handler.apply_hpost_and_hres(
            res=s, out=attn_out, h_post=h_post_attn, h_res=h_res_attn
        )
        s_flat = self.mhc_handler.flatten_mstream(s_)
        h_pre_mlp_, h_post_mlp_, h_res_mlp_ = self.mhc_handler.apply_norm_and_compute_h_(
            x=s_flat,
            norm_fn=partial(self.mhc_norm, modality_dispatcher=modality_dispatcher),
            phi_fused=self.mhc_phi_fused_mlp,
        )
        mlp_in = self.mhc_handler.compute_and_apply_hpre(
            x=s_,
            abh_pre=(self.mhc_alpha_pre_mlp, self.mhc_bias_pre_mlp, h_pre_mlp_),
            out_dtype=s.dtype,
        )
        return s_, h_post_mlp_, h_res_mlp_, mlp_in

    def _forward_mhc_post_mlp(
        self,
        s_: torch.Tensor,
        mlp_out: torch.Tensor,
        h_post_mlp_: torch.Tensor,
        h_res_mlp_: torch.Tensor,
    ) -> torch.Tensor:
        h_post_mlp, h_res_mlp = self.mhc_handler.compute_hpost_and_hres(
            abh_post=(self.mhc_alpha_post_mlp, self.mhc_bias_post_mlp, h_post_mlp_),
            abh_res=(self.mhc_alpha_res_mlp, self.mhc_bias_res_mlp, h_res_mlp_),
            out_dtype=s_.dtype,
        )
        return self.mhc_handler.apply_hpost_and_hres(
            res=s_, out=mlp_out, h_post=h_post_mlp, h_res=h_res_mlp
        )


class TransformerBlock(nn.Module):
    """Stack of preview layers with optional CPU layer-stream offload."""

    def __init__(self, model_config: Magi2PreviewConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(model_config, i) for i in range(model_config.num_layers)]
        )
        # Layer-by-layer CPU offload: when set to a compute device, each layer is
        # streamed CPU->device just before it runs and moved back after, so the
        # 228GB preview MoE never has more than one layer resident on the GPU.
        # Weights otherwise live in CPU RAM. Off by default (all-resident path).
        self.layer_offload_device: torch.device | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(_BF16)
        x = modality_dispatcher._permute(x)
        offload = self.layer_offload_device
        for layer in self.layers:
            if offload is not None:
                layer.to(offload)
            x = layer(x, rope, varlen_handler, modality_dispatcher, cp_split_sizes)
            if offload is not None:
                layer.to("cpu")
        x = modality_dispatcher._inv_permute(x)
        return x


# ============================================================
# top model
# ============================================================


def _validate_inference_model_config(model_config: Magi2PreviewConfig) -> None:
    if not model_config.mhc_config.enable:
        raise ValueError("MAGI-2 inference expects mhc_config.enable=true.")
    if not model_config.attn_gating.enable:
        raise ValueError("MAGI-2 inference expects scalar attention gating to be enabled.")
    if not model_config.attn_sinks.enable:
        raise ValueError("MAGI-2 inference expects sh-layout attention sinks.")


class Transformer(nn.Module):
    """MAGI-2-preview DiT: PreAdapter -> N x TransformerLayer -> PostAdapter.

    Single-GPU: cp = ep = dp = 1. Ulysses dispatch/undispatch removed;
    ``cp_split_sizes = [total_tokens]``.
    """

    def __init__(self, model_config: Magi2PreviewConfig, ep_size: int = 1) -> None:
        super().__init__()
        _validate_inference_model_config(model_config)
        self.config = model_config
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            virtual_width_factor=model_config.mhc_config.num_stream,
        )
        self.pre_adapter = PreAdapter(adapter_config)
        self.post_adapter = PostAdapter(adapter_config)
        self.block = TransformerBlock(model_config)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        time_token_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Single-GPU: no Ulysses dispatch; the whole packed stream is one CP shard.
        total_tokens = x.shape[0]
        cp_split_sizes_t = torch.tensor([total_tokens], device=x.device, dtype=torch.long)

        time_mask = modality_mapping == Modality.TIME
        modality_mapping = modality_mapping.clone()
        modality_mapping[time_mask] = Modality.TEXT
        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        video_idx = (modality_mapping == Modality.VIDEO).nonzero().flatten()
        audio_idx = (modality_mapping == Modality.AUDIO).nonzero().flatten()
        text_idx = (modality_mapping == Modality.TEXT).nonzero().flatten()
        time_idx = time_mask.nonzero().flatten()

        x, rope = self.pre_adapter(
            x, coords_mapping, video_idx, audio_idx, text_idx, time_idx
        )
        if time_token_sequence is not None and time_token_sequence.shape[-1] > 0:
            x[:, : time_token_sequence.shape[-1]] = time_token_sequence.to(x.dtype)

        x = self.block(
            x,
            rope,
            varlen_handler=varlen_handler,
            modality_dispatcher=modality_dispatcher,
            cp_split_sizes=cp_split_sizes_t,
        )
        x_out = self.post_adapter(x, video_idx, audio_idx)
        return x_out


__all__ = [
    "AdapterConfig",
    "Attention",
    "AttentionConfig",
    "CoreMultiHeadMoE",
    "CoreMultiHeadMoEConfig",
    "MLP",
    "MLPConfig",
    "MultiHeadMoELayer",
    "MultiHeadMoELayerConfig",
    "PostAdapter",
    "PreAdapter",
    "Transformer",
    "TransformerBlock",
    "TransformerLayer",
]


def _smoke() -> None:
    """Tiny-config CPU forward smoke: assert output shape + finiteness."""
    from dataclasses import replace

    from .config import (
        AttentionSinksConfig,
        MHCConfig,
        Magi2PreviewConfig,
        MoEConfig,
        MAGI2_MODALITY_AUDIO,
        MAGI2_MODALITY_TEXT,
        MAGI2_MODALITY_VIDEO,
    )
    from .preview_layers import get_coords, seqlens2cu_seqlens

    torch.manual_seed(0)
    moe_cfg = MoEConfig(
        num_experts=8,
        top_k=2,
        moe_layers=(2, 3),
        expert_intermediate_size=64,
        shared_expert_intermediate_size=64,
        modality_specific_expert_intermediate_size=64,
        num_heads=2,
        route_scale=1.0,
    )
    cfg = Magi2PreviewConfig(
        num_layers=4,
        hidden_size=256,
        head_dim=128,
        num_query_groups=2,
        mm_layers=(0, 1, 2, 3),
        mhc_config=MHCConfig(num_stream=2),
        moe_config=moe_cfg,
        attn_sinks=AttentionSinksConfig(enable=True, sink_token_num=1),
    )
    # moe_layers=(2,3): layers 2,3 are MoE; layers 0,1 dense MLP. All are mm.

    model = Transformer(cfg).eval()
    # Deterministic finite init for the empty()-initialised params.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)

    # Build a packed stream: 8 video, 3 audio, 2 text tokens.
    n_video, n_audio, n_text = 8, 3, 2
    total = n_video + n_audio + n_text
    in_ch = max(cfg.video_in_channels, cfg.audio_in_channels, cfg.text_in_channels)
    x = torch.randn(total, in_ch, dtype=_FP32)

    modality = torch.tensor(
        [MAGI2_MODALITY_VIDEO] * n_video
        + [MAGI2_MODALITY_AUDIO] * n_audio
        + [MAGI2_MODALITY_TEXT] * n_text,
        dtype=torch.long,
    )

    # Coords [L, 9]; use ref == size so the Fourier scale is exactly 1 (no nan).
    vid_coords = get_coords((2, 2, 2), (2, 2, 2))  # 8 video tokens
    aud_coords = get_coords((3, 1, 1), (3, 1, 1))  # 3 audio tokens
    txt_coords = get_coords((2, 1, 1), (2, 1, 1))  # 2 text tokens
    coords = torch.cat([vid_coords, aud_coords, txt_coords], dim=0)
    assert coords.shape == (total, 9), coords.shape

    # Single packed doc => cu_seqlens = [0, total].
    cu = seqlens2cu_seqlens(torch.tensor([total], dtype=torch.int32)).to(torch.int32)
    vh = VarlenHandler(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=total, max_seqlen_k=total)

    with torch.no_grad():
        out = model(x, coords, modality, vh)

    assert out.shape == (total, cfg.audio_in_channels), out.shape
    assert out.dtype == _FP32, out.dtype
    assert torch.isfinite(out).all(), "output has non-finite values"
    print(f"[smoke] OK  out.shape={tuple(out.shape)}  dtype={out.dtype}  finite=True")


if __name__ == "__main__":
    _smoke()
