"""MAGI-2 refiner (super-resolution) DiT top model (single-GPU PyTorch port).

Ported faithfully from SandAI's Apache-2.0 ``model/magi2_refiner.py``. The
refiner is a dense sibling of the preview DiT: 30 layers, hidden 4096, GQA
(32 query heads : 8 kv heads), element-wise Fourier RoPE, per-modality grouped
linears on the multi-modal boundary layers, scalar attention gating, and
frame-local windowed attention on every layer.

Differences from the preview DiT (all reproduced here):

* **Dense, not MoE** -- plain SwiGLU7 :class:`MLP` on every layer (no router /
  experts / shared experts). Upstream ``MoEConfig`` / ``NativeMoELinear`` are
  dead code and are skipped.
* **Plain residuals, no MHC hyper-connections** -- ``x = x + attn(x)`` then
  ``x = x + mlp(x)`` (upstream L2045 / L2053). Adapter / hidden width is a flat
  4096; there is no 4-stream widening.
* **No attention sinks** -- attention is a plain non-causal (masked) softmax.
  Scalar attention gating (``out * sigmoid(g)``) *is* kept (upstream L1750).
* **GQA 32:8** -- 32 query heads share 8 kv heads; kv heads are broadcast to the
  query-head count via ``repeat_interleave`` before the score matmul.
* **Frame-local windowed attention** on all 30 layers -- a video query at frame
  ``f`` attends video keys in frames ``[f - R, f + R]`` (``R = 11``) plus, densely,
  every audio + text key; audio / text queries attend every key densely. This is
  the exact union encoded by ``calc_local_qk_range`` (upstream L1348) which the
  upstream FA3 ``flex_flash_attn_with_cp`` kernel consumes; here it is a boolean
  mask over the full (single-GPU) sequence fed to a pure-torch masked attention.

Single-GPU collapse (cp = ep = dp = 1): the Ulysses ``dispatch`` / ``undispatch``
around ``forward`` and the CP scatter / gather inside ``flash_attn_with_cp`` /
``flex_flash_attn_with_cp`` reduce to identity, so ``cp_split_sizes`` is dropped
and attention is computed directly on the full packed stream. The
``magi_compile`` / ``magi_register_custom_op`` / ``flash_attn_interface`` /
``magi_attn`` / Triton imports are all dropped in favour of pure-torch math.
The fp32 (embedders / heads / norms) vs bf16 (blocks) dtype split and every
checkpoint-facing parameter name are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .config import (
    MAGI2_MODALITY_AUDIO,
    MAGI2_MODALITY_TEXT,
    MAGI2_MODALITY_VIDEO,
    Magi2RefinerConfig,
)
from .preview_layers import (
    ElementWiseFourierEmbed,
    MLPActivationType,
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
# frame-local windowed attention ranges + mask
# ============================================================


@dataclass
class LocalAttnHandler:
    """Query / key ranges for frame-local windowed attention (upstream ``FFAHandler``).

    Each ``(q_range, k_range)`` row means "these query rows may attend these key
    rows". A query attends the *union* of the key ranges of every row whose query
    range covers it -- exactly the online-softmax merge the upstream FA3 kernel
    performs. ``attn_type_map == 0`` everywhere means full / non-causal blocks.
    """

    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: Optional[float] = None


def calc_local_qk_range(
    num_video_tokens: int,
    num_audio_and_txt_tokens: int,
    num_frames: int,
    frame_receptive_field: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce upstream ``calc_local_qk_range`` (L1348) without CUDA pinning.

    Builds three families of ``(q_range, k_range)`` pairs:

    * per-frame local pairs -- video queries of frame ``i`` attend video keys in
      frames ``[i - R, i + R]`` (key range clamped to ``[0, num_video_tokens]``),
    * one video->(audio+text) pair -- every video query attends every a/t key,
    * one (audio+text)->all pair -- every a/t query attends every key.
    """
    token_per_frame = num_video_tokens // num_frames
    total_tokens = num_video_tokens + num_audio_and_txt_tokens

    q_range_list = []
    k_range_list = []
    for i in range(num_frames):
        q_range_list.append(
            torch.tensor([i * token_per_frame, (i + 1) * token_per_frame])
        )
        k_range_list.append(
            torch.tensor(
                [
                    (i - frame_receptive_field) * token_per_frame,
                    (i + frame_receptive_field + 1) * token_per_frame,
                ]
            )
        )
    local_q_range = torch.stack(q_range_list, dim=0)
    local_k_range = torch.stack(k_range_list, dim=0)

    local_k_range[local_k_range < 0] = 0
    local_k_range[local_k_range > num_video_tokens] = num_video_tokens

    video_q_range = torch.tensor([[0, num_video_tokens]])
    video_k_range = torch.tensor(
        [[num_video_tokens, num_video_tokens + num_audio_and_txt_tokens]]
    )
    at_q_ranges = torch.tensor([[num_video_tokens, total_tokens]])
    at_k_ranges = torch.tensor([[0, total_tokens]])

    q_ranges = torch.cat([local_q_range, video_q_range, at_q_ranges], dim=0).to(
        device=device, dtype=torch.int32
    )
    k_ranges = torch.cat([local_k_range, video_k_range, at_k_ranges], dim=0).to(
        device=device, dtype=torch.int32
    )
    return q_ranges, k_ranges


def calc_local_attn_handler(
    num_video_tokens: int,
    num_audio_and_txt_tokens: int,
    num_frames: int,
    frame_receptive_field: int,
    device: torch.device = torch.device("cpu"),
) -> LocalAttnHandler:
    """Build a :class:`LocalAttnHandler` (upstream ``calc_local_attn_ffa_handler``)."""
    q_ranges, k_ranges = calc_local_qk_range(
        num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field, device
    )
    seqlen = num_video_tokens + num_audio_and_txt_tokens
    return LocalAttnHandler(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        attn_type_map=torch.zeros(q_ranges.shape[0], device=device, dtype=torch.int32),
        softmax_scale=None,
    )


def build_range_mask(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean ``[Tq, Tk]`` mask (True == attend) = OR of every ``(q,k)`` block.

    OR-ing the blocks reproduces the FA3 union-over-covering-ranges semantics: a
    query row's attended keys are the union of the key ranges of all rows whose
    query range covers it. Disjoint key ranges make the union a plain concat, so
    a single masked softmax equals the upstream online-softmax merge.
    """
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
    for (qs, qe), (ks, ke) in zip(q_ranges.tolist(), k_ranges.tolist()):
        if qe > qs and ke > ks:
            mask[qs:qe, ks:ke] = True
    return mask


def build_block_diag_mask(cu_seqlens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Block-diagonal (per packed-doc) full-attention mask for non-local layers."""
    bounds = cu_seqlens.tolist()
    total = int(bounds[-1])
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        mask[int(start) : int(stop), int(start) : int(stop)] = True
    return mask


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Plain non-causal masked attention with GQA kv-head broadcast (fp32 math).

    ``q``: ``[T, num_heads_q, head_dim]``; ``k`` / ``v``: ``[T, num_heads_kv, head_dim]``;
    ``attn_mask``: ``[Tq, Tk]`` bool (True == attend). Each kv head is broadcast to
    ``num_heads_q // num_heads_kv`` query heads (query head ``h`` uses kv head
    ``h // group``), matching flash-attn's native GQA. Returns ``[T, num_heads_q,
    head_dim]`` in bf16 (mirrors the upstream FA3 bf16 output).
    """
    _, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    group = num_heads_q // num_heads_kv
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    qf = q.float().transpose(0, 1)  # [nh, Tq, d]
    kf = k.float().transpose(0, 1)  # [nh, Tk, d]
    vf = v.float().transpose(0, 1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [nh, Tq, Tk]
    scores = scores.masked_fill(~attn_mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)  # [nh, Tq, d]
    return out.transpose(0, 1).to(_BF16)  # [T, nh, d]


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
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(nn.Module):
    """GQA self-attention: pre-norm -> QKV (+ gate) -> q/k RMSNorm -> RoPE ->
    masked attention -> scalar sigmoid gate -> output proj. No sinks."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, eps=1e-6, num_modality=config.num_modality
        )
        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

        self.linear_qkv = create_linear(
            config.hidden_size,
            self.q_size + self.kv_size * 2,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        if config.enable_attn_gating:
            self.linear_g = create_linear(
                config.hidden_size,
                config.num_heads_q,
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
        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        local_mask: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        hidden_states = self.pre_norm(
            hidden_states, modality_dispatcher=modality_dispatcher
        ).to(_BF16)
        qkv = self.linear_qkv(hidden_states, modality_dispatcher=modality_dispatcher).to(_FP32)

        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=1)
        q = q.view(-1, cfg.num_heads_q, cfg.head_dim)
        k = k.view(-1, cfg.num_heads_kv, cfg.head_dim)
        v = v.view(-1, cfg.num_heads_kv, cfg.head_dim)

        if cfg.enable_attn_gating:
            g = self.linear_g(hidden_states, modality_dispatcher=modality_dispatcher).to(_FP32)
            g = g.unsqueeze(-1)  # (T, num_heads_q, 1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        # Undo the modality permutation so RoPE / attention see the original
        # packing order (video first) that the coords + local ranges describe.
        q = modality_dispatcher._inv_permute(q).unsqueeze(0)
        k = modality_dispatcher._inv_permute(k).unsqueeze(0)
        v = modality_dispatcher._inv_permute(v).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb).squeeze(0)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb).squeeze(0)
        v = v.squeeze(0)

        attn_mask = local_mask if cfg.use_local_attn else full_mask
        self_attn_out = attention_varlen(q, k, v, attn_mask)

        # Back to modality order for the (modality-order) gate + projection.
        self_attn_out = modality_dispatcher._permute(self_attn_out)
        if cfg.enable_attn_gating:
            self_attn_out = self_attn_out * torch.sigmoid(g)
        self_attn_out = self_attn_out.reshape(-1, self.q_size).to(_BF16)
        return self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)


# ============================================================
# dense MLP (SwiGLU7)
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
    """Dense SwiGLU7 feed-forward used on every refiner layer."""

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
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(_BF16)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).to(_FP32)
        x = self.activation_func(x).to(_BF16)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).to(_FP32)
        return x


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
    params_dtype: torch.dtype


class Adapter(nn.Module):
    """Per-modality input embedders (fp32) + element-wise Fourier RoPE."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(
            config.video_in_channels, config.hidden_size, bias=True, dtype=_FP32
        )
        self.text_embedder = nn.Linear(
            config.text_in_channels, config.hidden_size, bias=True, dtype=_FP32
        )
        self.audio_embedder = nn.Linear(
            config.audio_in_channels, config.hidden_size, bias=True, dtype=_FP32
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
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.rope(coords_mapping)
        output_x = torch.zeros(
            x.shape[0], self.config.hidden_size, device=x.device, dtype=x.dtype
        )
        output_x[text_mask] = self.text_embedder(
            x[text_mask, : self.config.text_in_channels]
        )
        output_x[audio_mask] = self.audio_embedder(
            x[audio_mask, : self.config.audio_in_channels]
        )
        output_x[video_mask] = self.video_embedder(
            x[video_mask, : self.config.video_in_channels]
        )
        return output_x, rope


@dataclass
class TransformerConfig:
    hidden_size: int
    video_in_channels: int
    audio_in_channels: int
    text_in_channels: int
    params_dtype: torch.dtype
    post_process_dtype: torch.dtype


class PostAdapter(nn.Module):
    """Final per-modality normalization + projection back to latent channels."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.final_norm_video = MultiModalityRMSNorm(config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(config.hidden_size)
        self.final_linear_video = nn.Linear(
            config.hidden_size, config.video_in_channels, bias=False, dtype=_FP32
        )
        self.final_linear_audio = nn.Linear(
            config.hidden_size, config.audio_in_channels, bias=False, dtype=_FP32
        )

    def forward(
        self, x: torch.Tensor, video_mask: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        x_out = torch.zeros(
            x.shape[0],
            max(self.config.video_in_channels, self.config.audio_in_channels),
            device=x.device,
            dtype=x.dtype,
        )
        x_out[video_mask, : self.config.video_in_channels] = x_video
        x_out[audio_mask, : self.config.audio_in_channels] = x_audio
        return x_out


# ============================================================
# transformer layer (plain residuals; no MHC)
# ============================================================


class TransformerLayer(nn.Module):
    """One refiner layer: plain residual attention then dense SwiGLU7 MLP.

    Unlike the preview DiT there is no MHC widening and no MoE.  Layers listed
    in ``local_attn_layers`` use the frame-local window mask; the rest use a
    per-document block-diagonal full mask.
    """

    def __init__(self, model_config: Magi2RefinerConfig, layer_idx: int) -> None:
        super().__init__()
        num_modality = 3 if layer_idx in model_config.mm_layers else 1
        use_local_attn = layer_idx in model_config.local_attn_layers
        self.post_norm = layer_idx in getattr(model_config, "post_norm_layers", ())

        self.attention = Attention(
            AttentionConfig(
                hidden_size=model_config.hidden_size,
                num_heads_q=model_config.num_heads_q,
                num_heads_kv=model_config.num_heads_kv,
                head_dim=model_config.head_dim,
                params_dtype=model_config.params_dtype,
                num_modality=num_modality,
                num_layers=model_config.num_layers,
                use_local_attn=use_local_attn,
                enable_attn_gating=model_config.enable_attn_gating,
            )
        )

        activation_type = MLPActivationType(model_config.activation_type)
        gated_act = activation_type in [MLPActivationType.SWIGLU7]
        if gated_act:
            intermediate_size = int(model_config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            intermediate_size = model_config.hidden_size * 4
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
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(
                model_config.hidden_size, num_modality=num_modality
            )
            self.mlp_post_norm = MultiModalityRMSNorm(
                model_config.hidden_size, num_modality=num_modality
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        local_mask: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attention(
            hidden_states, rope, modality_dispatcher, local_mask, full_mask
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out

        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + mlp_out
        return hidden_states


class TransformerBlock(nn.Module):
    """Sequential stack of dense refiner layers (no layer offload)."""

    def __init__(self, model_config: Magi2RefinerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(model_config, i) for i in range(model_config.num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        local_mask: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, rope, modality_dispatcher, local_mask, full_mask)
        return x


# ============================================================
# top model
# ============================================================


class Transformer(nn.Module):
    """MAGI-2 refiner DiT: Adapter -> 30 x TransformerLayer -> PostAdapter.

    Single-GPU: cp = ep = dp = 1. Ulysses dispatch/undispatch removed; local
    windowed attention becomes one masked attention over the full packed stream.
    """

    config: TransformerConfig

    def __init__(self, model_config: Magi2RefinerConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.config = TransformerConfig(
            hidden_size=model_config.hidden_size,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            text_in_channels=model_config.text_in_channels,
            params_dtype=model_config.params_dtype,
            post_process_dtype=_FP32,
        )
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            params_dtype=_FP32,
        )
        self.pre_adapter = Adapter(adapter_config)
        self.block = TransformerBlock(model_config)
        self.post_adapter = PostAdapter(self.config)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: LocalAttnHandler,
    ) -> torch.Tensor:
        total_tokens = x.shape[0]
        device = x.device

        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        video_mask = modality_mapping == MAGI2_MODALITY_VIDEO
        audio_mask = modality_mapping == MAGI2_MODALITY_AUDIO
        text_mask = modality_mapping == MAGI2_MODALITY_TEXT

        # Frame-local windowed mask (in original packing order) + a plain
        # per-doc block-diagonal mask for any non-local layer.
        local_mask = build_range_mask(
            local_attn_handler.q_ranges,
            local_attn_handler.k_ranges,
            total_tokens,
            device,
        )
        full_mask = build_block_diag_mask(varlen_handler.cu_seqlens_q, device)

        x, rope = self.pre_adapter(x, coords_mapping, video_mask, audio_mask, text_mask)

        # Only x moves to params_dtype; rope stays fp32.
        x = x.to(self.config.params_dtype)
        x = modality_dispatcher._permute(x)
        x = self.block(x, rope, modality_dispatcher, local_mask, full_mask)
        x = modality_dispatcher._inv_permute(x)

        return self.post_adapter(x, video_mask, audio_mask)


__all__ = [
    "Adapter",
    "AdapterConfig",
    "Attention",
    "AttentionConfig",
    "LocalAttnHandler",
    "MLP",
    "MLPConfig",
    "PostAdapter",
    "Transformer",
    "TransformerBlock",
    "TransformerConfig",
    "TransformerLayer",
    "attention_varlen",
    "build_block_diag_mask",
    "build_range_mask",
    "calc_local_attn_handler",
    "calc_local_qk_range",
]


def _smoke() -> None:
    """Tiny-config CPU forward smoke: assert output shape + finiteness."""
    from .config import Magi2RefinerConfig
    from .preview_layers import get_coords, seqlens2cu_seqlens

    torch.manual_seed(0)
    cfg = Magi2RefinerConfig(
        num_layers=4,
        hidden_size=256,
        head_dim=128,          # -> num_heads_q = 2
        num_query_groups=1,    # -> num_heads_kv = 1  (GQA 2:1)
        mm_layers=(0, 1, 2, 3),
        local_attn_layers=tuple(range(4)),
        frame_receptive_field=1,
    )

    model = Transformer(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)

    n_video, n_audio, n_text = 8, 3, 2
    num_frames = 4  # token_per_frame = 8 // 4 = 2
    total = n_video + n_audio + n_text
    in_ch = max(cfg.video_in_channels, cfg.audio_in_channels, cfg.text_in_channels)
    x = torch.randn(total, in_ch, dtype=_FP32)

    modality = torch.tensor(
        [MAGI2_MODALITY_VIDEO] * n_video
        + [MAGI2_MODALITY_AUDIO] * n_audio
        + [MAGI2_MODALITY_TEXT] * n_text,
        dtype=torch.long,
    )

    # ref == size so the Fourier scale is exactly 1 (no nan). Video: 4 frames x 2 x 1.
    vid_coords = get_coords((4, 2, 1), (4, 2, 1))
    aud_coords = get_coords((n_audio, 1, 1), (n_audio, 1, 1))
    txt_coords = get_coords((n_text, 1, 1), (n_text, 1, 1))
    coords = torch.cat([vid_coords, aud_coords, txt_coords], dim=0)
    assert coords.shape == (total, 9), coords.shape

    cu = seqlens2cu_seqlens(torch.tensor([total], dtype=torch.int32)).to(torch.int32)
    vh = VarlenHandler(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=total, max_seqlen_k=total)
    lah = calc_local_attn_handler(
        num_video_tokens=n_video,
        num_audio_and_txt_tokens=n_audio + n_text,
        num_frames=num_frames,
        frame_receptive_field=cfg.frame_receptive_field,
    )

    with torch.no_grad():
        out = model(x, coords, modality, vh, lah)

    assert out.shape == (total, cfg.audio_in_channels), out.shape
    assert torch.isfinite(out).all(), "output has non-finite values"
    print(
        f"[smoke] OK  out.shape={tuple(out.shape)}  dtype={out.dtype}  "
        f"num_heads_q={cfg.num_heads_q} num_heads_kv={cfg.num_heads_kv}  finite=True"
    )


if __name__ == "__main__":
    _smoke()
