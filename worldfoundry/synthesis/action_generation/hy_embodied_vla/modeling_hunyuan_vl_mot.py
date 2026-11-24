# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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

"""HunYuanVL-MoT (Mixture of Transformers) multimodal model.

WorldFoundry modifications retain the checkpoint structure and inference /
generation paths while removing language-model loss and label handling. Local
identifier cleanup and dtype propagation make the retained path lint-clean and
compatible with both BF16 and FP16 inference.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import (
    ClassVar,
    Final,
    Literal,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import (
    QwenRMSNorm as HunYuanVLMoTRMSNorm,
)
from worldfoundry.core.attention.rotary.embedding import (
    apply_rotary_pos_emb as _apply_rotary_pos_emb,
)
from worldfoundry.core.nn.blocks.layers import LayerScale as _ViTLayerScale

try:
    from timm.layers import (
        DropPath,
        LayerType,
        Mlp,
        PatchDropout,
        PatchEmbed,
        resample_abs_pos_embed,
    )

    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

import itertools

from .configuration_hunyuan_vl_mot import HunYuanVLMoTConfig, HunYuanVLMoTTextConfig

logger = logging.getLogger(__name__)

HY_VL_MOT_IMAGE_TOKEN_ID = 120687
HY_VL_MOT_VIDEO_TOKEN_ID = 120688
HY_VL_MOT_LATENT_TOKEN_ID = 120690


# ============================================================================
# Attention backend selection
# ============================================================================

_FLASH_ATTN_FUNCTIONS = None
_FLASH_ATTN_IMPORT_ATTEMPTED = False
_FLASH_ATTN_RUNTIME_DISABLED = False
_LAST_MOT_ATTENTION_BACKEND = "uninitialized"
_LAST_VISION_ATTENTION_BACKEND = "uninitialized"


def _load_flash_attn_functions():
    """Import FA2 lazily; missing or ABI-incompatible wheels are optional."""
    global _FLASH_ATTN_FUNCTIONS, _FLASH_ATTN_IMPORT_ATTEMPTED
    if not _FLASH_ATTN_IMPORT_ATTEMPTED:
        _FLASH_ATTN_IMPORT_ATTEMPTED = True
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except (ImportError, OSError, RuntimeError):
            _FLASH_ATTN_FUNCTIONS = None
        else:
            _FLASH_ATTN_FUNCTIONS = (flash_attn_func, flash_attn_varlen_func)
    return _FLASH_ATTN_FUNCTIONS


def _can_use_external_fa2(tensor: torch.Tensor) -> bool:
    """Use external FA2 only on its audited Ampere/Ada/Hopper range."""
    if _FLASH_ATTN_RUNTIME_DISABLED:
        return False
    if tensor.device.type != "cuda" or tensor.dtype not in {torch.float16, torch.bfloat16}:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(tensor.device)
    except (AssertionError, RuntimeError):
        return False
    # SM80-SM90: Ampere, Ada, Hopper. SM75 and SM100+ use PyTorch SDPA.
    return major in {8, 9} and _load_flash_attn_functions() is not None


def get_hunyuan_attention_backend() -> str:
    """Return the backend used by the most recent MoT attention call."""
    return _LAST_MOT_ATTENTION_BACKEND


def get_hunyuan_vision_attention_backend() -> str:
    """Return the backend used by the most recent HYViT2 spatial call."""
    return _LAST_VISION_ATTENTION_BACKEND


# ============================================================================
# Output dataclass
# ============================================================================


@dataclass
@auto_docstring(custom_intro="Base class for HunYuanVLMoT outputs.")
class HunYuanVLMoTModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states for fast sequential decoding.
    """

    logits: torch.FloatTensor | None = None
    last_hidden_state: Optional[torch.FloatTensor] = None  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
    past_key_values: Cache | None = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.


# ============================================================================
# Utility helpers
# ============================================================================


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ============================================================================
# RMSNorm
# ============================================================================


# ============================================================================
# Rotary Embedding
# ============================================================================


class HunYuanVLMoTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: HunYuanVLMoTConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_scaling.get("type", "default") if config.rope_scaling else "default"

        if self.rope_type == "dynamic" and config.rope_scaling and config.rope_scaling.get("alpha"):
            alpha = config.rope_scaling["alpha"]
            base = config.rope_theta * alpha ** (config.head_dim / (config.head_dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, config.head_dim, 2).float().to(device) / config.head_dim))
            self.attention_scaling = 1.0
        else:
            if self.rope_type in ROPE_INIT_FUNCTIONS:
                inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](config, device)
            else:
                base = config.rope_theta
                dim = config.head_dim
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim))
                self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq.clone()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ============================================================================
# MoT (Mixture of Transformers) helpers
# ============================================================================


def _mask_apply(
    hidden_states: torch.Tensor, mask: torch.Tensor, text_funcs, vision_funcs, out_dims=None, padding_mask=None
):
    """
    Routes tokens to modality-specific functions.
    hidden_states: (B, S, D), mask: (B, S) bool — True = vision token.
    padding_mask: (B, S), dtype=int or None — 1=valid, 0=padding. If provided,
                  padding tokens are excluded from GEMM to avoid bf16 rounding diffs.

    Each sample in the batch is processed independently to avoid bf16 GEMM
    rounding differences caused by different matrix dimensions when B varies.
    """
    B, S, D = hidden_states.size()

    placeholder = hidden_states[0:1, 0:1, :]
    zero_feature = 0

    # Determine output dims
    if out_dims is None:
        num_outputs = len(text_funcs)
        # We'll collect per-sample results and stack
        per_sample_results = [[] for _ in range(num_outputs)]
    else:
        num_outputs = len(out_dims)
        per_sample_results = [[] for _ in range(num_outputs)]

    for b in range(B):
        hs_b = hidden_states[b]  # (S, D)
        mask_b = mask[b].bool()  # (S,)

        if padding_mask is not None:  # noqa: SIM108 - Branches carry per-case comments.
            valid_b = padding_mask[b].bool()  # (S,)
        else:
            valid_b = None

        # Prepare output buffer for this sample
        if out_dims is None:
            out_b = [torch.zeros(S, D, device=hs_b.device, dtype=hs_b.dtype) for _ in range(num_outputs)]
        else:
            out_b = [torch.zeros(S, od, device=hs_b.device, dtype=hs_b.dtype) for od in out_dims]

        # Text tokens: mask == 0 (and valid if padding_mask provided)
        text_idx = ~mask_b & valid_b if valid_b is not None else ~mask_b

        if text_idx.any():
            hs_t = hs_b[text_idx]  # (N_t, D)
            for i, fn in enumerate(text_funcs):
                out_b[i][text_idx] = fn(hs_t)
        else:
            for fn in text_funcs:
                zero_feature = zero_feature + fn(placeholder).mean() * 0

        # Vision tokens: mask == 1 (and valid if padding_mask provided)
        vis_idx = mask_b & valid_b if valid_b is not None else mask_b

        if vis_idx.any():
            hs_v = hs_b[vis_idx]  # (N_v, D)
            for i, fn in enumerate(vision_funcs):
                out_b[i][vis_idx] = fn(hs_v)
        else:
            for fn in vision_funcs:
                zero_feature = zero_feature + fn(placeholder).mean() * 0

        for i in range(num_outputs):
            per_sample_results[i].append(out_b[i])

    # Stack per-sample results into (B, S, od)
    result = [torch.stack(per_sample_results[i], dim=0) for i in range(num_outputs)]
    result[0] = result[0] + zero_feature
    return result


def _flash_attention_forward_mot_fast(
    query,
    key,
    value,
    attention_mask,
    scaling=None,
    *,
    flash_attn_varlen_func,
):
    """Official FA2 varlen path for supported CUDA devices."""
    # Transpose from (B, heads, S, D) -> (B, S, heads, D)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    B, S_q, H, D = query.shape
    S_k = key.shape[1]

    v_seqlens = attention_mask["v_seqlens"]
    padding_mask = attention_mask.get("padding_mask", None)  # (B, S_k), 1=valid, 0=padding

    # ---- Step 1: Causal attention via flash_attn_varlen_func ----
    if padding_mask is not None and B > 1 and S_q == S_k:
        # Prefill with left-padding: unpad -> varlen attn -> repad
        pad_bool = padding_mask.bool()  # (B, S_k)
        seqlens = padding_mask.sum(dim=-1).to(torch.int32)  # (B,)
        cu_seqlens = torch.zeros(B + 1, device=query.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
        max_seqlen = seqlens.max().item()

        # Unpad: boolean indexing extracts all valid tokens in row-major order
        q_unpad = query[pad_bool]  # (total_valid, H, D)
        k_unpad = key[pad_bool]
        v_unpad = value[pad_bool]

        output_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scaling,
            causal=True,
        )

        # Repad: scatter valid tokens back into (B, S_q, H, D)
        attn_output = query.new_zeros(B, S_q, H, D)
        attn_output[pad_bool] = output_unpad
    else:
        # B==1, no padding, or decode step (S_q != S_k) -- simple reshape path
        _, _, H_kv, _ = key.shape
        q_flat = query.contiguous().view(B * S_q, H, D)
        k_flat = key.contiguous().view(B * S_k, H_kv, D)
        v_flat = value.contiguous().view(B * S_k, H_kv, D)
        cu_q_t = torch.arange(0, B + 1, dtype=torch.int32, device=query.device) * S_q
        cu_k_t = torch.arange(0, B + 1, dtype=torch.int32, device=query.device) * S_k

        attn_flat = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q=cu_q_t,
            cu_seqlens_k=cu_k_t,
            max_seqlen_q=S_q,
            max_seqlen_k=S_k,
            softmax_scale=scaling,
            causal=(S_q == S_k),
        )
        attn_output = attn_flat.reshape(B, S_q, H, D)

    # ---- Step 2: Visual bidirectional attention override ----
    has_visual = False
    if isinstance(v_seqlens, list):
        for segs in v_seqlens:
            if segs.numel() > 0 and not (segs == 0).all():
                has_visual = True
                break
    else:
        if v_seqlens.numel() > 0 and not (v_seqlens == 0).all():
            has_visual = True

    if has_visual:
        # Normalize v_seqlens to a list of tensors, one per batch element
        if isinstance(v_seqlens, list):  # noqa: SIM108 - Branches carry per-case comments.
            v_seglens_list = v_seqlens
        else:
            v_seglens_list = [v_seqlens]  # B==1

        visual_query_parts = []
        visual_key_parts = []
        visual_value_parts = []
        cu_v_seqlen = [0]
        max_v_seqlen = 0
        visual_write_back = []
        fake_visual = True

        for b_idx, segs in enumerate(v_seglens_list):
            if segs.numel() == 0 or (segs == 0).all():
                continue
            for seg_idx in range(segs.shape[0]):
                s, e = segs[seg_idx, 0].item(), segs[seg_idx, 1].item()
                seg_len = e - s
                if seg_len <= 0:
                    continue
                fake_visual = False
                visual_query_parts.append(query[b_idx, s:e])
                visual_key_parts.append(key[b_idx, s:e])
                visual_value_parts.append(value[b_idx, s:e])
                cu_v_seqlen.append(cu_v_seqlen[-1] + seg_len)
                if seg_len > max_v_seqlen:
                    max_v_seqlen = seg_len
                visual_write_back.append((b_idx, s, e))

        if fake_visual:
            dummy = query[:1, :1].sum() * 0
            attn_output = attn_output + dummy
        else:
            visual_query_cat = torch.cat(visual_query_parts, dim=0)
            visual_key_cat = torch.cat(visual_key_parts, dim=0)
            visual_value_cat = torch.cat(visual_value_parts, dim=0)
            cu_v_seqlens_t = torch.tensor(cu_v_seqlen, device=query.device, dtype=torch.int32)

            visual_attn_output = flash_attn_varlen_func(
                visual_query_cat,
                visual_key_cat,
                visual_value_cat,
                cu_seqlens_q=cu_v_seqlens_t,
                cu_seqlens_k=cu_v_seqlens_t,
                max_seqlen_q=max_v_seqlen,
                max_seqlen_k=max_v_seqlen,
                softmax_scale=scaling,
                causal=False,
            )

            # Write back: override causal attention output with bidirectional visual output
            attn_output = attn_output.clone()
            offset = 0
            for b_idx, s, e in visual_write_back:
                seg_len = e - s
                attn_output[b_idx, s:e] = visual_attn_output[offset : offset + seg_len]
                offset += seg_len

    return attn_output, None


def _visual_segment_list(v_seqlens, batch_size: int):
    """Normalize the vendor segment representation to one tensor per sample."""
    if isinstance(v_seqlens, list):
        if len(v_seqlens) != batch_size:
            raise ValueError(f"Expected {batch_size} visual-segment tensors, got {len(v_seqlens)}")
        return v_seqlens
    if batch_size != 1:
        raise ValueError("A tensor visual-segment value is valid only for batch size 1")
    return [v_seqlens]


def _build_mot_sdpa_mask(
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    v_seqlens,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build the MoT inference mask used by the correctness-first SDPA path.

    Text queries use causal attention during prefill. Each visual segment
    replaces those rows with an isolated bidirectional block, matching the
    official FA2 write-back override. Decode calls (``Q != K``) attend to all
    cached keys. Padding keys are always hidden and padding query rows are
    zeroed after SDPA.
    """
    if query_length == key_length:
        allowed = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=device,
        ).tril()
        allowed = allowed.unsqueeze(0).expand(batch_size, -1, -1).clone()
    else:
        allowed = torch.ones(
            batch_size,
            query_length,
            key_length,
            dtype=torch.bool,
            device=device,
        )

    for batch_index, segments in enumerate(_visual_segment_list(v_seqlens, batch_size)):
        for segment in segments:
            start, end = (int(segment[0].item()), int(segment[1].item()))
            query_start, query_end = max(start, 0), min(end, query_length)
            key_start, key_end = max(start, 0), min(end, key_length)
            if query_end <= query_start or key_end <= key_start:
                continue
            allowed[batch_index, query_start:query_end, :] = False
            allowed[
                batch_index,
                query_start:query_end,
                key_start:key_end,
            ] = True

    query_valid = None
    if padding_mask is not None:
        if padding_mask.ndim != 2 or padding_mask.shape != (batch_size, key_length):
            raise ValueError(
                f"MoT padding_mask must have shape ({batch_size}, {key_length}), got {tuple(padding_mask.shape)}"
            )
        key_valid = padding_mask.to(device=device, dtype=torch.bool)
        allowed &= key_valid[:, None, :]
        if query_length == key_length:
            query_valid = key_valid
            allowed &= query_valid[:, :, None]
    return allowed[:, None, :, :], query_valid


def _sdpa_attention_forward_mot(
    query,
    key,
    value,
    attention_mask,
    scaling=None,
):
    """Pure-PyTorch MoT attention with causal/padding/visual semantics."""
    batch_size, query_heads, query_length, _ = query.shape
    key_length = key.shape[2]
    key_heads = key.shape[1]
    if query_heads % key_heads != 0:
        raise ValueError(f"MoT GQA requires query heads divisible by KV heads; got {query_heads}/{key_heads}")
    if query_heads != key_heads:
        repeat = query_heads // key_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    mask, query_valid = _build_mot_sdpa_mask(
        batch_size=batch_size,
        query_length=query_length,
        key_length=key_length,
        device=query.device,
        v_seqlens=attention_mask["v_seqlens"],
        padding_mask=attention_mask.get("padding_mask"),
    )
    output_dtype = query.dtype
    if query.device.type == "cpu" and query.dtype in {torch.float16, torch.bfloat16}:
        query, key, value = query.float(), key.float(), value.float()
    sdpa_kwargs = {}
    if scaling is not None:
        sdpa_kwargs["scale"] = scaling
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        **sdpa_kwargs,
    ).to(dtype=output_dtype)
    if query_valid is not None:
        output = output * query_valid[:, None, :, None]
    return output.transpose(1, 2).contiguous(), None


def _attention_forward_mot(query, key, value, attention_mask, scaling=None):
    """Dispatch to official FA2 where supported, otherwise PyTorch SDPA."""
    global _FLASH_ATTN_RUNTIME_DISABLED, _LAST_MOT_ATTENTION_BACKEND
    if _can_use_external_fa2(query):
        try:
            result = _flash_attention_forward_mot_fast(
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                flash_attn_varlen_func=_load_flash_attn_functions()[1],
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            _FLASH_ATTN_RUNTIME_DISABLED = True
            logger.warning(
                "External FlashAttention failed; using PyTorch SDPA for this process: %s",
                exc,
            )
        else:
            _LAST_MOT_ATTENTION_BACKEND = "flash_attention_2"
            return result
    _LAST_MOT_ATTENTION_BACKEND = "torch_sdpa"
    return _sdpa_attention_forward_mot(
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
    )


def _sdpa_spatial_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Pure-PyTorch bidirectional spatial attention, including varlen isolation.

    Inputs use the HYViT2 layout ``(B, H, N, D)``. A varlen call represents
    concatenated images in a single batch row; each ``cu_seqlens`` interval is
    therefore evaluated independently so tokens never cross image boundaries.
    """
    batch_size, _, sequence_length, _ = query.shape
    output_dtype = query.dtype
    if query.device.type == "cpu" and query.dtype in {torch.float16, torch.bfloat16}:
        query, key, value = query.float(), key.float(), value.float()
    sdpa_kwargs = {"scale": scaling} if scaling is not None else {}

    if cu_seqlens is None:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            **sdpa_kwargs,
        )
    else:
        if batch_size != 1:
            raise ValueError(f"HYViT2 varlen attention expects one concatenated batch row; got batch size {batch_size}")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("HYViT2 cu_seqlens must be a one-dimensional boundary tensor")
        boundaries = [int(index) for index in cu_seqlens.detach().cpu().tolist()]
        if boundaries[0] != 0 or boundaries[-1] != sequence_length:
            raise ValueError(
                "HYViT2 cu_seqlens must span the concatenated sequence: "
                f"expected [0, ..., {sequence_length}], got {boundaries}"
            )
        output = torch.empty_like(query)
        for start, end in itertools.pairwise(boundaries):
            if end < start:
                raise ValueError(f"HYViT2 cu_seqlens are not monotonic: {boundaries}")
            if end == start:
                continue
            output[:, :, start:end] = F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                dropout_p=0.0,
                is_causal=False,
                **sdpa_kwargs,
            )
    return output.to(dtype=output_dtype).transpose(1, 2).contiguous()


def _flash_spatial_attention_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float | None,
    cu_seqlens: torch.Tensor | None,
    flash_attn_func,
    flash_attn_varlen_func,
) -> torch.Tensor:
    """Official FA2 HYViT2 spatial path for supported CUDA devices."""
    batch_size, _, sequence_length, _ = query.shape
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    if cu_seqlens is None:
        return flash_attn_func(
            query,
            key,
            value,
            softmax_scale=scaling,
            causal=False,
        )
    if batch_size != 1:
        raise ValueError(f"HYViT2 varlen attention expects one concatenated batch row; got batch size {batch_size}")
    segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(segment_lengths.max().item()) if segment_lengths.numel() else 0
    output = flash_attn_varlen_func(
        query.squeeze(0),
        key.squeeze(0),
        value.squeeze(0),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scaling,
        causal=False,
    )
    return output.reshape(batch_size, sequence_length, query.shape[2], query.shape[3])


def _spatial_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Dispatch HYViT2 and SpaceTime spatial attention to FA2 or SDPA."""
    global _FLASH_ATTN_RUNTIME_DISABLED, _LAST_VISION_ATTENTION_BACKEND
    if _can_use_external_fa2(query):
        flash_functions = _load_flash_attn_functions()
        try:
            result = _flash_spatial_attention_fast(
                query,
                key,
                value,
                scaling=scaling,
                cu_seqlens=cu_seqlens,
                flash_attn_func=flash_functions[0],
                flash_attn_varlen_func=flash_functions[1],
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            _FLASH_ATTN_RUNTIME_DISABLED = True
            logger.warning(
                "External FlashAttention failed; using PyTorch SDPA for this process: %s",
                exc,
            )
        else:
            _LAST_VISION_ATTENTION_BACKEND = "flash_attention_2"
            return result
    _LAST_VISION_ATTENTION_BACKEND = "torch_sdpa"
    return _sdpa_spatial_attention(
        query,
        key,
        value,
        scaling=scaling,
        cu_seqlens=cu_seqlens,
    )


def _modality_mask_to_segments_single(mask_1d):
    """
    Process a single 1D mask to find modality segments.

    mask_1d: Tensor of shape (slen,), dtype=int64
    Returns: Tensor of shape (num_segments, 2) with [start, end) pairs
    """
    slen = mask_1d.numel()

    is_zero = (mask_1d == 0).to(torch.int64)
    padded = torch.cat(
        [
            torch.tensor([0], device=mask_1d.device),
            is_zero,
            torch.tensor([0], device=mask_1d.device),
        ]
    )

    diff = padded[1:] - padded[:-1]
    zero_run_starts = (diff == 1).nonzero(as_tuple=True)[0]
    zero_run_ends = (diff == -1).nonzero(as_tuple=True)[0] - 1

    separators = []
    for s, e in zip(zero_run_starts, zero_run_ends, strict=False):
        if (e - s + 1) >= 2:
            separators.append((s, e))

    segments = []
    seg_start = 0
    for s, e in separators:
        seg_end = s - 1
        if seg_end >= seg_start:
            segments.append([seg_start, seg_end])
        seg_start = e + 1

    if seg_start < slen:
        segments.append([seg_start, slen - 1])

    for i in range(len(segments)):
        segments[i][1] = segments[i][1] + 2  # make end exclusive

    if segments:
        return torch.tensor(segments, device=mask_1d.device)
    return torch.zeros((0, 2), dtype=torch.long, device=mask_1d.device)


def _modality_mask_to_segments(mask: torch.Tensor):
    """
    Convert a boolean modality mask to (start, end) visual segment pairs.
    Supports batch size >= 1.

    mask: (B, slen) or (slen,)
    Returns:
        - If B == 1: Tensor of shape (num_segments, 2) [backward compatible]
        - If B > 1: list of Tensors, each of shape (num_segments_i, 2)
    """
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    B, slen = mask.shape

    if slen == 1:
        # KV cache decode step
        single = torch.tensor([[0, 0]], device=mask.device)
        if B == 1:
            return single
        return [single.clone() for _ in range(B)]

    if B == 1:
        return _modality_mask_to_segments_single(mask[0].to(torch.int64))

    # Multi-batch: return a list of segment tensors
    results = []
    for b in range(B):
        results.append(_modality_mask_to_segments_single(mask[b].to(torch.int64)))
    return results


# ============================================================================
# MoT Attention
# ============================================================================


class HunYuanVLMoTAttention(nn.Module):
    """Multi-headed attention with per-modality text/vision projection paths."""

    def __init__(self, config: HunYuanVLMoTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        # Text projections
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.query_layernorm = HunYuanVLMoTRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanVLMoTRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Vision projections (separate path, _v suffix matches checkpoint keys)
        self.q_proj_v = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_v = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_v = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_v = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def _mask_apply(self, hidden_states, modality_mask, text_funcs, vision_funcs, out_dims=None, padding_mask=None):
        if modality_mask is None:
            return [text_funcs[0](hidden_states)]
        return _mask_apply(hidden_states, modality_mask, text_funcs, vision_funcs, out_dims, padding_mask=padding_mask)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        modality_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Extract padding_mask for excluding padding from GEMM
        pm = None
        if isinstance(attention_mask, dict):
            padding_mask = attention_mask.get("padding_mask", None)
            if padding_mask is not None and hidden_states.shape[1] == padding_mask.shape[1]:
                pm = padding_mask

        query_states, key_states, value_states = self._mask_apply(
            hidden_states,
            modality_mask,
            [self.q_proj, self.k_proj, self.v_proj],
            [self.q_proj_v, self.k_proj_v, self.v_proj_v],
            out_dims=[
                self.config.num_attention_heads * self.head_dim,
                self.config.num_key_value_heads * self.head_dim,
                self.config.num_key_value_heads * self.head_dim,
            ],
            padding_mask=pm,
        )

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _attention_forward_mot(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self._mask_apply(
            attn_output,
            modality_mask,
            [self.o_proj],
            [self.o_proj_v],
            padding_mask=pm,
        )[0]
        return attn_output, attn_weights


# ============================================================================
# Decoder Layer
# ============================================================================


class HunYuanVLMoTDecoderLayer(nn.Module):
    """A single transformer decoder layer with per-modality norm and MLP paths."""

    def __init__(self, config: HunYuanVLMoTTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVLMoTAttention(config=config, layer_idx=layer_idx)

        # Text paths
        self.mlp = QwenMLP(config, bias=config.mlp_bias)
        self.input_layernorm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Vision paths (_v suffix matches checkpoint keys)
        self.mlp_v = QwenMLP(config, bias=config.mlp_bias)
        self.input_layernorm_v = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_v = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        modality_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Extract padding_mask for excluding padding from GEMM
        padding_mask = None
        if isinstance(attention_mask, dict):
            padding_mask = attention_mask.get("padding_mask", None)
        # Only use padding_mask during prefill (seq dims match)
        pm = padding_mask if (padding_mask is not None and hidden_states.shape[1] == padding_mask.shape[1]) else None

        residual = hidden_states

        hidden_states = _mask_apply(
            hidden_states,
            modality_mask,
            [self.input_layernorm],
            [self.input_layernorm_v],
            padding_mask=pm,
        )[0]

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            modality_mask=modality_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = _mask_apply(
            hidden_states,
            modality_mask,
            [lambda x: self.mlp(self.post_attention_layernorm(x))],
            [lambda x: self.mlp_v(self.post_attention_layernorm_v(x))],
            padding_mask=pm,
        )[0]
        hidden_states = residual + hidden_states

        # Zero out padding positions to prevent non-zero residuals from accumulating.
        # Without this, Linear bias at padding positions propagates through residual
        # connections, changing GEMM matrix dimensions in mask_apply and causing
        # bf16 rounding differences for valid tokens.
        if padding_mask is not None and hidden_states.shape[1] == padding_mask.shape[1]:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1)

        return hidden_states


# ============================================================================
# Inner language model (text-only decoder with MoT)
# ============================================================================


class _HunYuanVLMoTInnerPreTrainedModel(PreTrainedModel):
    """Internal base for the text decoder that lives inside HunYuanVLMoTModel."""

    config_class = HunYuanVLMoTConfig
    base_model_prefix = "model"
    _no_split_modules: ClassVar[list] = ["HunYuanVLMoTDecoderLayer"]
    _skip_keys_device_placement: ClassVar[list] = ["past_key_values"]
    # Our MoT attention selects external FA2 or PyTorch SDPA internally.
    # Keep Transformers from imposing a package-level attention backend.
    _supports_sdpa = False

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


class _HunYuanVLMoTTextModel(_HunYuanVLMoTInnerPreTrainedModel):
    """Pure text decoder (embed_tokens + layers + norm + rotary)."""

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunYuanVLMoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HunYuanVLMoTRotaryEmbedding(config=config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        modality_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            if attention_mask is not None and attention_mask.shape[0] > 1:
                # Multi-batch with left-padding: compute per-sample position_ids.
                # For each row, valid tokens (mask==1) get positions 0,1,2,...
                # and padding tokens (mask==0) get position 0 (don't matter, will be ignored).
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.clamp(min=0)
                # During decode, inputs_embeds is (B, 1, D) but attention_mask is (B, past+1).
                # Only keep the position_ids corresponding to the current input tokens.
                seq_len = inputs_embeds.shape[1]
                if position_ids.shape[1] > seq_len:
                    position_ids = position_ids[:, -seq_len:]
            else:
                position_ids = cache_position.unsqueeze(0)
        text_position_ids = position_ids

        if modality_mask is None:
            modality_mask = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)

        visual_segs = _modality_mask_to_segments(modality_mask)

        # During decode, modality_mask may be full prefill length but inputs_embeds is (B, 1, D).
        # Truncate modality_mask to match current input length.
        seq_len = inputs_embeds.shape[1]
        if modality_mask is not None and modality_mask.shape[1] > seq_len:
            modality_mask = modality_mask[:, -seq_len:]

        causal_mask = {
            "v_seqlens": visual_segs,
            "padding_mask": attention_mask,  # (B, seqlen), 1=valid, 0=padding; None if no padding
        }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, text_position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                modality_mask=modality_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class _HunYuanVLMoTTextForCausalLM(_HunYuanVLMoTInnerPreTrainedModel, GenerationMixin):
    """Text decoder + lm_head for generation (inner component)."""

    _tied_weights_keys: ClassVar[dict] = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.model = _HunYuanVLMoTTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        modality_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            modality_mask=modality_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
        )


# ============================================================================
# HYViT2 — Vision Transformer with any-resolution support
# ============================================================================


# ViT Attention / Block


class _ViTAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        seperate_qv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

        if seperate_qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        B, N, C = x.shape
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = _spatial_attention_forward(
            q,
            k,
            v,
            scaling=self.scale,
            cu_seqlens=cu_slens,
        ).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        if mlp_layer is None:
            mlp_layer = Mlp
        self.norm1 = norm_layer(dim)
        self.attn = _ViTAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = _ViTLayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = _ViTLayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# HYViT2 VisionTransformer


class _HYViT2VisionTransformer(nn.Module):
    """Vision Transformer with variable-resolution support."""

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: bool | None = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Callable | None = None,
        norm_layer: LayerType | None = None,
        act_layer: LayerType | None = None,
        strict_img_size: bool = False,
        block_fn: type[nn.Module] | None = None,
        mlp_layer: type[nn.Module] | None = None,
        ignore_head: bool = False,
    ) -> None:
        super().__init__()
        if embed_layer is None:
            embed_layer = PatchEmbed
        if block_fn is None:
            block_fn = _ViTBlock
        if mlp_layer is None:
            mlp_layer = Mlp

        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.ignore_head = ignore_head

        embed_args = {}
        if dynamic_img_size:
            embed_args.update({"strict_img_size": False, "output_fmt": "NHWC"})
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
            dynamic_img_pad=dynamic_img_pad,
            strict_img_size=strict_img_size,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = None
        self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        pos_embed_shape = int(self.pos_embed.shape[1] ** 0.5)
        if (h, w) == (pos_embed_shape, pos_embed_shape):
            return self.pos_embed
        rescaled = self.pos_embed.new_zeros(1, h * w, self.pos_embed.shape[2])
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        if torch.__version__ == "2.0.0":
            dtype = pe_2d.dtype
            pe_2d = (
                F.interpolate(pe_2d.float(), out_size, mode="bilinear", align_corners=False).to(dtype).view(-1, h * w)
            )
        else:
            pe_2d = F.interpolate(pe_2d, out_size, mode="bilinear", align_corners=False).view(-1, h * w)
        rescaled[0] = pe_2d.T.contiguous()
        return rescaled

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
            x = x + pos_embed

        return self.pos_drop(x)

    def sample_positional_embedding(self, grid):
        pos_embed_shape = int(self.pos_embed.shape[1] ** 0.5)
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        n, _ = grid.shape
        grid = grid.view(1, n, 1, 2)
        output_dtype = pe_2d.dtype
        pos_embedding = F.grid_sample(
            pe_2d.float(), grid.float(), mode="bilinear", align_corners=False, padding_mode="border"
        )
        return pos_embedding.view(1, -1, n).to(dtype=output_dtype).transpose(1, 2)

    def forward_get_embedding_list(self, x_list, use_grid_sampling):
        x_all = []
        image_sizes = []
        slen = []

        if use_grid_sampling:
            image_grids = []
            image_patches = []
            for x in x_list:
                _, _, h, w = x.shape
                pad_h = (
                    self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]
                ) % self.patch_embed.patch_size[0]
                pad_w = (
                    self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]
                ) % self.patch_embed.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
                _, _, h, w = x.shape
                h = h // self.patch_embed.patch_size[0]
                w = w // self.patch_embed.patch_size[1]
                x = (
                    x.view(1, 3, h, self.patch_embed.patch_size[0], w, self.patch_embed.patch_size[1])
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(-1, 3, self.patch_embed.patch_size[0], self.patch_embed.patch_size[1])
                )
                margin_h = 1.0 / h
                margin_w = 1.0 / w
                dh = torch.linspace(-1 + margin_h, 1 - margin_h, steps=h, device=x.device)
                dw = torch.linspace(-1 + margin_w, 1 - margin_w, steps=w, device=x.device)
                meshx, meshy = torch.meshgrid((dh, dw))
                grid = torch.stack((meshy, meshx), 2).reshape(-1, 2)
                image_patches.append(x)
                image_grids.append(grid)
                image_sizes.append((h, w))
                slen.append(h * w)

            image_patches = torch.cat(image_patches, dim=0)
            image_grids = torch.cat(image_grids, dim=0)
            x = self.patch_embed(image_patches)
            pos_embedding = self.sample_positional_embedding(image_grids)
            c = pos_embedding.size(-1)
            x = x.reshape(1, -1, c) + pos_embedding
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        else:
            for x in x_list:
                _, _, h, w = x.shape
                pad_h = (
                    self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]
                ) % self.patch_embed.patch_size[0]
                pad_w = (
                    self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]
                ) % self.patch_embed.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
                _, _, h, w = x.shape
                h = h // self.patch_embed.patch_size[0]
                w = w // self.patch_embed.patch_size[1]
                x = self.patch_embed(x)
                x = x + self.rescale_positional_embedding(out_size=(h, w))
                x = self.patch_drop(x)
                x = self.norm_pre(x)
                x_all.append(x)
                image_sizes.append((h, w))
            slen = [xi.size(1) for xi in x_all]
            x = torch.cat(x_all, dim=1)

        cu_indices = [0]
        for i in slen:
            cu_indices.append(cu_indices[-1] + i)
        cu_slens = torch.tensor(cu_indices, dtype=torch.int32, device=x.device)
        return x, cu_slens, slen, image_sizes

    def forward_features_list(self, x_list, use_grid_sampling=True):
        x, cu_slens, slen, image_sizes = self.forward_get_embedding_list(x_list, use_grid_sampling=use_grid_sampling)
        for blk in self.blocks:
            x = blk(x, cu_slens=cu_slens)
        x_return = x.split(slen, dim=1)
        return x_return, image_sizes

    def forward_features(self, x: torch.Tensor):
        _, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        return x, (h, w)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, cal_attn_pool=False):
        if isinstance(x, list):
            x, image_sizes = self.forward_features_list(x)
            if not cal_attn_pool:
                return x, image_sizes, None
            cls_tokens = torch.cat([self.forward_head(cur_x) for cur_x in x], dim=0)
            return x, image_sizes, cls_tokens
        else:
            x, image_sizes = self.forward_features(x)
            if not cal_attn_pool:
                return x, image_sizes, None
            return x, image_sizes, self.forward_head(x)


# HYViT2 model config


@dataclass
class _HYViT2VisionCfg:
    width: int = 1152
    layers: tuple[int, int, int, int] | int = 27
    heads: int = 16
    patch_size: int = 16
    image_size: tuple[int, int] | int = 384
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0


def _create_hyvit2() -> _HYViT2VisionTransformer:
    """Construct the exact vision architecture populated by the VLA checkpoint."""
    vision_cfg = _HYViT2VisionCfg()
    return _HYViT2VisionTransformer(
        img_size=2048,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=vision_cfg.layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        dynamic_img_pad=False,
        strict_img_size=False,
        ignore_head=False,
        num_classes=0,
    )


# HYViT2 Projection / pooling


class _NormalizedDwPooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, forward_type="2x"):
        B, H, W, C = x.shape
        if forward_type == "2x":
            new_x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2, W // 2, 4, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
        elif forward_type == "1x":
            new_x = x.reshape(B, H, W, 1, C)
            fused_x = torch.cat([new_x, new_x], dim=-1)
            score = self.predictor(fused_x)
            return (new_x * F.softmax(score, dim=-2)).sum(dim=-2)
        elif forward_type == "4x":
            new_x = x.reshape(B, H // 4, 4, W // 4, 4, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 4, W // 4, 16, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 16, -1)
        else:
            raise ValueError(f"Unknown forward_type: {forward_type}")
        fused_x = torch.cat([new_x, pooled_x], dim=-1)
        score = self.predictor(fused_x)
        return (new_x * F.softmax(score, dim=-2)).sum(dim=-2)


class _HYViT2MLPProjector(nn.Module):
    def __init__(self, in_channels, out_channels, twoview=False):
        super().__init__()
        self.proj1 = nn.Linear(in_channels, out_channels)
        self.proj2 = nn.Linear(out_channels, out_channels)
        self.act = nn.GELU()
        self.pooler = _NormalizedDwPooler(out_channels)
        self.out_channels = out_channels
        embed_std = 1 / math.sqrt(out_channels)
        if twoview:
            self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

    def _forward_list(self, x, size):
        split_lens = [h // 2 * w // 2 for h, w in size]
        all_x = []
        for i, (h, w) in enumerate(size):
            now_x = (
                x[i].reshape(1, h // 2, 2, w // 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(h // 2 * w // 2, 2, 2, -1)
            )
            all_x.append(now_x)
        x = torch.cat(all_x, dim=0)
        x = self.proj1(x)
        x = self.pooler(x, forward_type="2x")
        x = self.act(x)
        x = self.proj2(x)
        c = x.shape[-1]
        x = torch.split(x, split_lens, dim=0)
        xs = []
        for i, (h, w) in enumerate(size):
            now_x = x[i].reshape(1, h // 2, w // 2, -1)
            now_x = now_x.reshape(1, -1, c)
            xs.append(now_x)
        return xs

    def forward(self, x, size=(16, 16), x2=None, size2=(16, 16)):
        if isinstance(x, list):
            xs = self._forward_list(x, size)
            if x2 is not None:
                xs2 = self._forward_list(x2, size2)
                dtype = xs[0].dtype
                sep = self.image_sep.reshape(1, 1, -1).expand(1, 1, self.out_channels).to(dtype)
                xs = [torch.cat([xi, sep, x2i], dim=1) for xi, x2i in zip(xs, xs2, strict=False)]
            return xs
        else:
            h, w = size
            x = x.reshape(x.shape[0], h, w, -1)
            x = self.proj1(x)
            x = self.pooler(x, forward_type="2x")
            x = self.act(x)
            x = self.proj2(x)
            b, h, w, c = x.shape
            x = x.reshape(b, -1, c)
            return x


# HYViT2_400MAnyRes — Any-resolution ViT wrapper


class HYViT2_400MAnyRes(nn.Module):
    """
    Any-resolution HYViT2 wrapper.
    Loads the vision tower and projects features to the language model dimension.
    """

    def __init__(self, vision_tower: str, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.merger = _HYViT2MLPProjector(in_channels=1152, out_channels=2048, twoview=False)
        self.select_layer = -1
        self.load_model()

    def load_model(self, device_map=None):
        # Image preprocessing (normalization with mean/std=[0.5,0.5,0.5]) is handled
        # by HunYuanVLMoTProcessor before the model forward pass.
        self.vision_tower = _create_hyvit2()
        self.vision_tower.eval()
        self.is_loaded = True

    def _forward_func(self, images, cal_attn_pool=False):
        if isinstance(images, list):
            xs = [x.to(self.dtype) for x in images]
            image_features, img_size, cls_token = self.vision_tower(xs, cal_attn_pool=cal_attn_pool)
        else:
            image_forward_outs, img_size, cls_token = self.vision_tower(
                images.to(self.dtype), cal_attn_pool=cal_attn_pool
            )
            image_features = image_forward_outs.to(images.dtype)
        return image_features, img_size, cls_token

    def forward(self, images, cal_attn_pool=False):
        with torch.no_grad():
            image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)

        if isinstance(images, list):
            image_features = [self.merger(x, s).squeeze(0) for x, s in zip(image_features, img_size, strict=False)]
        else:
            image_features = self.merger(image_features, img_size)
            C = image_features.shape[-1]
            image_features = [image_features.reshape(-1, C)]

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.pos_embed.dtype

    @property
    def device(self):
        return self.vision_tower.pos_embed.device

    @property
    def hidden_size(self):
        return 1152

    @property
    def config(self):
        return type("HYViT2ConfigWrapper", (), {"patch_size": 16})()


# ============================================================================
# Vision Encoder — wraps HYViT2_400MAnyRes
# ============================================================================


class _HunYuanVLMoTVisionEncoder(nn.Module):
    """
    Thin wrapper that loads the HYViT2_400MAnyRes vision encoder.
    """

    def __init__(self, vit_config_name: str = "hyvit2_anyres"):
        super().__init__()
        self._encoder = HYViT2_400MAnyRes(vit_config_name)

    @property
    def dtype(self):
        return next(self._encoder.parameters()).dtype

    def forward(self, images):
        return self._encoder(images)


# ============================================================================
# Top-level PreTrainedModel
# ============================================================================


@auto_docstring
class HunYuanVLMoTPreTrainedModel(PreTrainedModel):
    config_class = HunYuanVLMoTConfig
    base_model_prefix = "model"
    _no_split_modules = ["HunYuanVLMoTDecoderLayer"]  # noqa: RUF012 - Shared class-level value, as upstream.
    _skip_keys_device_placement = "past_key_values"
    # Attention dispatch is internal: optional FA2 on SM80-SM90, SDPA elsewhere.
    _supports_sdpa = False
    _can_compile_fullgraph = False

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


# ============================================================================
# HunYuanVLMoTModel — core model combining vision + language
# ============================================================================


@auto_docstring
class HunYuanVLMoTModel(HunYuanVLMoTPreTrainedModel):
    """
    The HunYuanVL-MoT model: HYViT2 vision encoder + MoT language decoder.
    Token slots marked with HY_VL_MOT_IMAGE_TOKEN_ID / HY_VL_MOT_VIDEO_TOKEN_ID are replaced
    by the corresponding visual embeddings before being passed to the decoder.
    """

    base_model_prefix = ""
    config: HunYuanVLMoTConfig
    _checkpoint_conversion_mapping = {}  # noqa: RUF012 - Shared class-level value, as upstream.
    accepts_loss_kwargs = False

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.language_model = _HunYuanVLMoTTextForCausalLM._from_config(config)
        # Use HYViT2_400MAnyRes directly so weight keys
        # match the checkpoint: model.visual.vision_tower.*, model.visual.merger.*
        self.visual = HYViT2_400MAnyRes("hyvit2_anyres")
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    # ------------------------------------------------------------------
    # Vision feature extraction helpers
    # ------------------------------------------------------------------

    def _reshape_pixel_values(self, pixel_values: torch.Tensor, grid_thw: torch.LongTensor):
        """Reshape flat patch tensor back into (T, H*patch, W*patch, C) images."""
        pixel_values = pixel_values.reshape(-1, 3, 16, 16)
        num_patches = grid_thw.prod(dim=-1).tolist()
        patches_list = torch.split(pixel_values, num_patches, dim=0)
        images = []
        for idx, pv in enumerate(patches_list):
            T, H, W = grid_thw[idx][0].item(), grid_thw[idx][1].item(), grid_thw[idx][2].item()
            pv = (
                pv.reshape(T, H // 2, W // 2, 2, 2, 3, 16, 16)
                .permute(0, 1, 3, 2, 4, 6, 7, 5)
                .reshape(T, H, W, 16, 16, 3)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(T, H * 16, W * 16, 3)
                .permute(0, 3, 1, 2)
            )
            images.append(pv)
        return images

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        images = self._reshape_pixel_values(pixel_values, image_grid_thw)
        return self.visual(images)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor | None = None
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        frames = []
        for img_list in self._reshape_pixel_values(pixel_values_videos, video_grid_thw):
            # Each video's T frames are treated as independent images
            frames.extend(img_list.split(1, dim=0))
        return self.visual(frames)

    def get_image_video_features(
        self, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, device, dtype
    ):
        recon_images, recon_videos = [], []
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            recon_images = self._reshape_pixel_values(pixel_values, image_grid_thw)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            for pv in self._reshape_pixel_values(pixel_values_videos, video_grid_thw):
                recon_videos.extend(pv.split(1, dim=0))

        fake_image = torch.zeros(1, 3, 64, 64, dtype=dtype, device=device)
        all_embeds = self.visual([fake_image] + recon_images + recon_videos)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        split_point = len(recon_images) + 1
        image_embeds = all_embeds[1:split_point]
        video_embeds = all_embeds[split_point:]
        zero_feature = all_embeds[0].mean() * 0
        return image_embeds, video_embeds, zero_feature

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        """Find where HY_VL_MOT_IMAGE_TOKEN_ID / HY_VL_MOT_VIDEO_TOKEN_ID appear and validate counts."""
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            img_embed = embed_fn(torch.tensor(HY_VL_MOT_IMAGE_TOKEN_ID, dtype=torch.long, device=inputs_embeds.device))
            vid_embed = embed_fn(torch.tensor(HY_VL_MOT_VIDEO_TOKEN_ID, dtype=torch.long, device=inputs_embeds.device))
            special_image_mask = (inputs_embeds == img_embed).all(-1)
            special_video_mask = (inputs_embeds == vid_embed).all(-1)
        else:
            special_image_mask = input_ids == HY_VL_MOT_IMAGE_TOKEN_ID
            special_video_mask = input_ids == HY_VL_MOT_VIDEO_TOKEN_ID

        union_mask = special_image_mask | special_video_mask
        n_img = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(f"Image feature/token count mismatch: tokens={n_img}, features={image_features.shape[0]}")
        n_vid = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(f"Video feature/token count mismatch: tokens={n_vid}, features={video_features.shape[0]}")
        return special_image_mask, special_video_mask, union_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        position_ids: Optional[torch.LongTensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        past_key_values: Optional[Cache] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        inputs_embeds: Optional[torch.FloatTensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        pixel_values: Optional[torch.Tensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        pixel_values_videos: Optional[torch.FloatTensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        image_grid_thw: Optional[torch.LongTensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        video_grid_thw: Optional[torch.LongTensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        cache_position: Optional[torch.LongTensor] = None,  # noqa: UP045 - Transformers auto_docstring formats typing.Optional annotations.
        logits_to_keep: Union[int, torch.Tensor] = 0,  # noqa: UP007 - Transformers auto_docstring formats typing.Union annotations.
        **kwargs: Unpack[TransformersKwargs],
    ) -> HunYuanVLMoTModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width of each image's patch grid.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width of each video's patch grid.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        union_mask = None
        if pixel_values is None and pixel_values_videos is None:
            pass  # inference with KV cache — skip vision encoding
        else:
            image_embeds, video_embeds, zero_feature = self.get_image_video_features(
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            if len(image_embeds) > 0:
                image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _, union_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds, image_features=image_embeds_cat
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
            if len(video_embeds) > 0:
                video_embeds_cat = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask, union_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds, video_features=video_embeds_cat
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)
            inputs_embeds = inputs_embeds + zero_feature

        if union_mask is not None:
            kwargs["modality_mask"] = union_mask
        else:
            kwargs["modality_mask"] = torch.zeros(
                inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return HunYuanVLMoTModelOutputWithPast(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# ============================================================================
# HunYuanVLMoTForConditionalGeneration — public generation entry point
# ============================================================================


@auto_docstring
class HunYuanVLMoTForConditionalGeneration(HunYuanVLMoTPreTrainedModel, GenerationMixin):
    """
    HunYuanVL-MoT with a language modelling head for multimodal conditional generation.

    Supports images, videos, and text input with Mixture-of-Transformers attention.
    """

    _checkpoint_conversion_mapping = {}  # noqa: RUF012 - Shared class-level value, as upstream.
    _tied_weights_keys = {"model.language_model.lm_head.weight": "model.language_model.model.embed_tokens.weight"}  # noqa: RUF012 - Shared class-level value, as upstream.
    accepts_loss_kwargs = False
    config: HunYuanVLMoTConfig

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.model = HunYuanVLMoTModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> HunYuanVLMoTModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width of each image patch grid.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width of each video patch grid.
        """
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        # Position IDs are generated from rope_deltas inside forward
        model_inputs["position_ids"] = None
        # Only pass pixel_values on the first forward pass.
        # Compatible with both transformers 4.x (cache_position based)
        # and transformers 5.x (is_first_iteration based).
        _cp = model_inputs.get("cache_position")
        is_subsequent = (_cp is not None and len(_cp) > 0 and _cp[0] != 0) or (
            _cp is None and not is_first_iteration and use_cache
        )
        if is_subsequent:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["video_grid_thw"] = None

        return model_inputs


__all__ = [
    "HunYuanVLMoTForConditionalGeneration",
    "HunYuanVLMoTModel",
    "HunYuanVLMoTPreTrainedModel",
]
