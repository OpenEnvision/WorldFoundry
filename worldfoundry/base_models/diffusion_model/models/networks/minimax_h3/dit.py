"""MiniMax H3 packed-token audio-video DiT (single-GPU PyTorch).

Ported from SGLang's Apache-2.0
``runtime/models/dits/minimax_h3.py``. This is a faithful single-GPU port:

* ``ColumnParallelLinear`` / ``RowParallelLinear`` / ``MergedColumnParallelLinear``
  collapse to plain ``nn.Linear`` (world_size = 1).
* Ulysses sequence-parallel all-to-all, tensor-parallel all-gather, breakable
  CUDA graphs, and the layerwise-offload mixin are removed — every path here is
  rank-local.
* Dense variable-length non-causal attention is computed with
  ``F.scaled_dot_product_attention`` per packed document (``cu_seqlens``).

The numerics and the mixed fp32/bf16 parameter split are preserved exactly:
patch projections, the timestep embedder, the output heads, and
``rope.inv_freq`` stay fp32; transformer blocks are bf16. The fused CUDA/Triton
operators are represented by the portable fallbacks in :mod:`.ops` (Stage 8
swaps in the fused kernels behind the ``can_use_fused_*`` probes).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from .ops import (
    apply_qk_norm,
    apply_rope_qk,
    indexed_gate,
    indexed_scale_shift,
    make_rms_norm,
    qk_norm_rope,
    rope_cos_sin_cache,
    silu_mul,
)

_BF16 = torch.bfloat16
_FP32 = torch.float32

_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER = (
    "video_patch_proj.weight",
    "video_patch_proj.bias",
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
)
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
    }
)


# --------------------------------------------------------------------------- #
# Grouped-QKV checkpoint reorder (dense MHA: per-head [q,k,v] -> [q,k,v] all).
# --------------------------------------------------------------------------- #
def reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Reorder a grouped ``[q,k,v]`` checkpoint tensor to ``[q_all,k_all,v_all]``."""

    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )
    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1)
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


# --------------------------------------------------------------------------- #
# Variable-length non-causal attention over a packed sequence.
# --------------------------------------------------------------------------- #
def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Dense non-causal attention per packed document.

    ``q``/``k``/``v``: ``[T, num_heads, head_dim]`` packed rows.
    ``cu_seqlens``: cumulative sequence boundaries ``[num_docs + 1]``.
    Returns ``[T, num_heads, head_dim]``.
    """

    total, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        length = int(stop) - int(start)
        if length <= 0:
            continue
        # [L, nh, hd] -> [1, nh, L, hd] for SDPA.
        q_seg = q[start:stop].transpose(0, 1).unsqueeze(0)
        k_seg = k[start:stop].transpose(0, 1).unsqueeze(0)
        v_seg = v[start:stop].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            q_seg, k_seg, v_seg, is_causal=False, scale=softmax_scale
        )
        out[start:stop] = attended.squeeze(0).transpose(0, 1)
    return out


# --------------------------------------------------------------------------- #
# Timestep embedder (fp32 sinusoidal, cos-before-sin).
# --------------------------------------------------------------------------- #
class MiniMaxH3TimeEmbedder(nn.Module):
    """fp32 sinusoidal timestep MLP; concatenates ``cos`` before ``sin``."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = nn.Linear(arch.timestep_input_dim, arch.time_embed_hidden_size, bias=True, dtype=_FP32)
        self.proj_out = nn.Linear(arch.time_embed_hidden_size, arch.time_embed_dim, bias=True, dtype=_FP32)
        self.register_buffer("_frequency_cache", None, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32 (cos concatenated before sin)."""

        half = self.frequency_embedding_size // 2
        freqs = self._frequency_cache
        if freqs is None or freqs.device != t.device:
            freqs = torch.exp(
                -math.log(10000.0) * torch.arange(half, dtype=_FP32, device=t.device) / half
            )
            self._frequency_cache = freqs
        args = t.to(_FP32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden = self.proj_in(t_freq)
        hidden = F.silu(hidden)
        return self.proj_out(hidden)


# --------------------------------------------------------------------------- #
# 3D RoPE frequency builder.
# --------------------------------------------------------------------------- #
class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary 0.75)."""

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(inv_freq_len, dtype=_FP32), persistent=True)

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""

        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(_FP32)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


# --------------------------------------------------------------------------- #
# Attention (fused qkv -> per-head qk RMSNorm -> RoPE -> varlen SDPA -> out).
# --------------------------------------------------------------------------- #
class MiniMaxH3Attention(nn.Module):
    """Fused-QKV self-attention with per-head RMSNorm, 3D RoPE, and varlen SDPA."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        # Fused qkv: three logical matrices stored as one tensor.
        self.qkv_proj = nn.Linear(arch.hidden_size, 3 * self.inner_dim, bias=False, dtype=_BF16)
        self.q_norm = make_rms_norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = make_rms_norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, arch.hidden_size, bias=False, dtype=_BF16)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        attention_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed rows -> [T, hidden]."""

        total = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        if rope_cache is None:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        else:
            cos_sin_cache, positions = rope_cache
            q, k = qk_norm_rope(q, k, self.q_norm, self.k_norm, cos_sin_cache, positions)
        if attention_groups is None:
            out = _varlen_attention(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
        else:
            # Variants can expose global media with document-local text without
            # materializing a quadratic attention mask or duplicating projections.
            out = torch.zeros_like(q)
            for queries, keys in attention_groups:
                attended = F.scaled_dot_product_attention(
                    q.index_select(0, queries).transpose(0, 1).unsqueeze(0),
                    k.index_select(0, keys).transpose(0, 1).unsqueeze(0),
                    v.index_select(0, keys).transpose(0, 1).unsqueeze(0),
                    scale=self.softmax_scale,
                )
                out.index_copy_(0, queries, attended.squeeze(0).transpose(0, 1))
        out = out.reshape(total, self.num_heads * self.head_dim)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# SwiGLU MLP.
# --------------------------------------------------------------------------- #
class MiniMaxH3MLP(nn.Module):
    """SwiGLU MLP with fused ``gate|up`` projection (``fc1``) and ``fc2`` down-proj."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        # gate and up fused into fc1.
        self.fc1 = nn.Linear(arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False, dtype=_BF16)
        self.fc2 = nn.Linear(arch.ffn_hidden_size, arch.hidden_size, bias=False, dtype=_BF16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = silu_mul(hidden)
        return self.fc2(hidden)


# --------------------------------------------------------------------------- #
# AdaLN projection (SiLU + zero-init linear over unique condition embeddings).
# --------------------------------------------------------------------------- #
class MiniMaxH3AdalnProj(nn.Module):
    """[M, t_dim] -> expand_ratio tensors of [M * modality_num, H]."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        *,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != "
                f"{expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = nn.Linear(arch.time_embed_dim, out_features, bias=True, dtype=_BF16)

    def project_local(self, adaln_input: torch.Tensor) -> torch.Tensor:
        return self.linear(adaln_input)

    def split_output(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.split_output(self.project_local(adaln_input))


# --------------------------------------------------------------------------- #
# Token refiner (pre-norm transformer, no AdaLN/RoPE).
# --------------------------------------------------------------------------- #
class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Pre-norm transformer block used only on text tokens (no AdaLN / RoPE)."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm1 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch)
        self.mlp = MiniMaxH3MLP(arch)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cache=None, cu_seqlens=cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    """Shallow text refiner run once per request before packing into the DiT stream."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(arch) for _ in range(arch.token_refiner_num_layers)]
        )
        self.final_norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# DiT block (AdaLN-modulated attention + MLP with gated residuals).
# --------------------------------------------------------------------------- #
class MiniMaxH3DiTBlock(nn.Module):
    """AdaLN-modulated attention + MLP with per-row gated residuals.

    ``combined_indices`` select among ``num_unique_timesteps × 3`` modality
    vectors (video / audio / text) produced by :class:`MiniMaxH3AdalnProj`.
    """

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm1 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch)
        self.mlp = MiniMaxH3MLP(arch)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
        attention_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        ffn_conditioner: Any = None,
    ) -> torch.Tensor:
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = self.norm1(x)
        h = indexed_scale_shift(h, shift_msa, scale_msa, combined_indices, dtype=_BF16)
        if attention_groups is None:
            h = self.attn(h, rope_cache=rope_cache, cu_seqlens=cu_seqlens)
        else:
            h = self.attn(h, rope_cache=rope_cache, cu_seqlens=cu_seqlens, attention_groups=attention_groups)
        x = indexed_gate(residual, gate_msa, h, combined_indices, dtype=_BF16)

        residual = x
        if ffn_conditioner is not None:
            x = ffn_conditioner(x)
        h = self.norm2(x)
        h = indexed_scale_shift(h, shift_mlp, scale_mlp, combined_indices, dtype=_BF16)
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices, dtype=_BF16)


# --------------------------------------------------------------------------- #
# Final layer (single-modality AdaLN -> fp32 -> dual video/audio heads).
# --------------------------------------------------------------------------- #
class MiniMaxH3FinalLayer(nn.Module):
    """Final AdaLN (timestep-only) then fp32 video-patch and audio heads."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = nn.Linear(arch.hidden_size, video_patch_dim, bias=True, dtype=_FP32)
        self.audio_out = nn.Linear(arch.hidden_size, arch.audio_latents_dim, bias=True, dtype=_FP32)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = self.norm(x)
        h = indexed_scale_shift(h, shift, scale, inverse_indices, dtype=_BF16)
        h = h.to(_FP32)  # full precision through both output projections
        return self.video_out(h), self.audio_out(h)


# --------------------------------------------------------------------------- #
# Full model.
# --------------------------------------------------------------------------- #
class MiniMaxH3DiTModel(nn.Module):
    """Single-GPU MiniMax H3 packed-token audio-video DiT."""

    def __init__(self, config: MiniMaxH3DiTArchConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        arch = config if config is not None else MiniMaxH3DiTArchConfig(**overrides)
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim

        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.video_patch_proj = nn.Linear(video_patch_dim, arch.hidden_size, bias=True, dtype=_FP32)
        self.audio_patch_proj = nn.Linear(arch.audio_latents_dim, arch.hidden_size, bias=True, dtype=_FP32)
        self.condition_proj = nn.Linear(arch.text_dim, arch.hidden_size, bias=True, dtype=_BF16)
        self.time_embedder = MiniMaxH3TimeEmbedder(arch)
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(arch)
        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(arch) for _ in range(arch.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(arch)

    # -- weight-dtype guard ------------------------------------------------- #
    def post_load_weights(self) -> None:
        """Assert the mixed fp32/bf16 parameter split survived weight loading."""

        for name in _MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER:
            param = self.get_parameter(name)
            if param.dtype != _FP32:
                raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")
        if self.rope.inv_freq.dtype != _FP32:
            raise ValueError(f"rope.inv_freq must stay fp32 after load, got {self.rope.inv_freq.dtype}.")

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        ids = pos_info.get("position_ids") if isinstance(pos_info, dict) else getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        value = psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    @staticmethod
    def _psp_optional_field(psp: Any, field: str) -> Any:
        return psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Project and refine request-static text conditioning once."""

        text_len = int(refiner_cu_seqlens[1].item())
        if text_len <= 0 or text_len > int(prompt_embeds.shape[0]):
            raise ValueError(
                f"refiner cu_seqlens live text length must be in [1, {int(prompt_embeds.shape[0])}], got {text_len}"
            )
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=_BF16)
        # Refiner attends the single text document [0, text_len].
        refiner_cu = torch.tensor([0, text_len], device=device, dtype=torch.int32)
        text_embed = self.condition_proj(text_rows)
        return self.token_refiner(text_embed, cu_seqlens=refiner_cu)

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the request-static RoPE cos/sin cache and position index."""

        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        seq_len = int(img_position_ids.shape[1])
        rope_freqs = self.rope(img_position_ids).to(device)
        return (
            rope_cos_sin_cache(rope_freqs, dtype=_BF16),
            torch.arange(seq_len, device=device, dtype=torch.long),
        )

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        device: torch.device,
        refined_prompt_embeds_length: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the packed embedding sequence [S, H] bf16 and t_emb [M, t_dim] fp32."""

        if refined_prompt_embeds_length is None:
            text_len = int(refiner_cu_seqlens[1].item())
        elif torch.is_tensor(refined_prompt_embeds_length):
            text_len = int(refined_prompt_embeds_length.item())
        else:
            text_len = int(refined_prompt_embeds_length)
        if text_len <= 0 or text_len > int(text_embeddings_selected.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(text_embeddings_selected.shape[0])}], got {text_len}"
            )
        text_pos = text_pos[:text_len]
        if refined_prompt_embeds_length is not None:
            text_embed = text_embeddings_selected[:text_len].to(device=device, dtype=_BF16)
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    "refined prompt embeddings must have hidden width "
                    f"{self.hidden_size}, got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(text_embeddings_selected, refiner_cu_seqlens, device=device)

        seq_len = int(x.shape[1])
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=_BF16)

        if text_pos.numel():
            embeddings.index_add_(0, text_pos, text_embed[:text_len].to(_BF16))

        # Latent embedders stay fp32; only scatter into their packed positions.
        if img_pos.numel():
            x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(_FP32)
            video_embed = self.video_patch_proj(x_rows)
            embeddings.index_add_(0, img_pos, video_embed.to(_BF16))

        if audio_pos.numel():
            audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(_FP32)
            audio_embed = self.audio_patch_proj(audio_rows)
            embeddings.index_add_(0, audio_pos, audio_embed.to(_BF16))

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward returning ``(video_logits, audio_logits)``."""

        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: {sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        update_mask = _required_kwarg(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")
        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(_required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info")
        text_pos = self._pos_ids(_required_kwarg(kwargs, "text_pos_info"), "text_pos_info")
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"), "img_pos_for_infer_output_info"
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device

        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids).to(device)
            rope_cache = (
                rope_cos_sin_cache(rope_freqs, dtype=_BF16),
                torch.arange(seq_len, device=device, dtype=torch.long),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
        )
        adaln_input = F.silu(t_emb).to(_BF16)
        inverse_indices = inverse_indices.to(device)
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags.clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != seq_len:
                raise ValueError(
                    f"block_token_tags must cover the packed sequence ({seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(block_token_tags, inverse_indices, alpha=MINIMAX_H3_ADALN_MODALITY_NUM)

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=block_combined,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
            )
        video_logits, audio_logits = self.final_layer(
            hidden, adaln_input=adaln_input, inverse_indices=inverse_indices
        )

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "reorder_grouped_qkv_to_qkv",
]
