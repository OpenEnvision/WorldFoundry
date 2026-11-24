# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hansheng Chen.
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3:
#   lakonlab/models/architectures/diffusers/minimax_h3.py
# Pruned AdaLN table adapted from MiniMax & Hugging Face MiniMax-H3-Pruned (Apache-2.0).
"""Pruned MiniMax-H3 DiT variant with optional Sol-H3 DWD pixel head for pi-H3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ...config import MINIMAX_H3_ADALN_MODALITY_NUM, MiniMaxH3DiTArchConfig
from ...dit import (
    _FORWARD_SUPPORTED_KWARGS,
    MiniMaxH3DiTModel,
    _compute_prefix_sink_tokens,
    _normalize_host_bounds,
    _required_kwarg,
)
from ...ops import make_rms_norm, rms_norm_indexed_scale_shift, rope_cos_sin_cache
from ..hyperflow.conditioning import HyperFlowQKVProjection, HyperFlowSwiGLUProjection
from .sol_dwd_head import BLOCK, FRAMES_TO_TRIM, OUT_CH, PATCH, PATCH_T, SolH3DWDHead

_PI_H3_FORWARD_KWARGS = _FORWARD_SUPPORTED_KWARGS | frozenset({"return_features"})

PI_H3_DEFAULT_DWD_HEAD_CONFIG: dict[str, Any] = {
    "pixel_patch_size": (PATCH_T, 2 * PATCH, 2 * PATCH),
    "first_frame_pixel_patch_size": (PATCH_T - FRAMES_TO_TRIM, 2 * PATCH, 2 * PATCH),
    "first_frame_period": BLOCK,
    "pixel_channels": OUT_CH,
    "linear_head": False,
    "pixel_refiner": {
        "type": "SolH3DWDHead",
        "num_layers": 8,
    },
}


class PiH3PrunedTimeEmbedder(nn.Module):
    """Exact linear interpolation lookup on ``t in [0, 1]`` into the pruned AdaLN table."""

    table: torch.Tensor

    def __init__(self, table_size: int = 1025, adaln_rank: int = 8) -> None:
        super().__init__()
        if table_size < 2:
            raise ValueError(f"time_table_size must be >= 2, got {table_size}.")
        if adaln_rank <= 0:
            raise ValueError(f"adaln_rank must be positive, got {adaln_rank}.")
        self.register_buffer("table", torch.zeros(table_size, adaln_rank, dtype=torch.float32))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Interpolate ``table`` at clamped ``timestep in [0, 1]`` in float32."""

        steps = self.table.shape[0] - 1
        position = timestep.float().flatten().clamp(0.0, 1.0) * steps
        lower = position.floor().clamp(max=steps - 1).long()
        return torch.lerp(
            self.table[lower].float(),
            self.table[lower + 1].float(),
            (position - lower).unsqueeze(-1),
        )


class PiH3PrunedAdalnProj(nn.Module):
    """Rank-``adaln_rank`` AdaLN linear projection without extra SiLU.

    The pruned timestep table already stores coordinates of ``silu(t_emb)``, so
    this projection applies ``self.linear(temb)`` directly.
    """

    def __init__(
        self,
        adaln_rank: int = 8,
        hidden_size: int = 5376,
        *,
        expand_ratio: int = 6,
        modality_num: int = MINIMAX_H3_ADALN_MODALITY_NUM,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.adaln_rank = int(adaln_rank)
        self.hidden_size = int(hidden_size)
        self.expand_ratio = int(expand_ratio)
        self.modality_num = int(modality_num)
        out_features = self.expand_ratio * self.hidden_size * self.modality_num
        self.linear = nn.Linear(self.adaln_rank, out_features, bias=True, dtype=dtype)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        projected = self.linear(temb.to(self.linear.weight.dtype))
        return tuple(projected.reshape(-1, self.expand_ratio * self.hidden_size).chunk(self.expand_ratio, dim=-1))


class PiH3PrunedFinalLayer(nn.Module):
    """Pruned final RMSNorm + ``(shift, scale)`` AdaLN projection and video/audio output heads."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        adaln_rank: int = 8,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        video_patch_dim = arch.video_patch_output_dim
        self.norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps, dtype=dtype)
        self.adaln_proj = PiH3PrunedAdalnProj(
            adaln_rank=adaln_rank,
            hidden_size=arch.hidden_size,
            expand_ratio=2,
            modality_num=1,
            dtype=dtype,
        )
        self.video_out = nn.Linear(arch.hidden_size, video_patch_dim, bias=True, dtype=dtype)
        self.audio_out = nn.Linear(arch.hidden_size, arch.audio_latents_dim, bias=True, dtype=dtype)

    def modulate(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaln_proj(adaln_input)
        x_in = x if x.dtype == self.norm.weight.dtype else x.to(self.norm.weight.dtype)
        return rms_norm_indexed_scale_shift(x_in, self.norm, shift, scale, inverse_indices, dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.modulate(x, adaln_input=adaln_input, inverse_indices=inverse_indices)
        return self.video_out(h.to(self.video_out.weight.dtype)), self.audio_out(h.to(self.audio_out.weight.dtype))


class PiH3DWDAdaLayerNormOut(nn.Module):
    """DWD output AdaLayerNorm matching Diffusers' ``AdaLayerNormContinuous``.

    Uses non-affine ``nn.LayerNorm`` and applies ``SiLU`` before ``linear``,
    splitting outputs as ``(scale, shift)`` (scale first, shift second).
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_embedding_dim: int,
        *,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, 2 * hidden_size, bias=True, dtype=dtype)
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear(self.silu(temb).to(self.linear.weight.dtype))
        scale, shift = emb.chunk(2, dim=-1)
        normed = self.norm(hidden_states)
        out = normed * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)
        return out.to(hidden_states.dtype)


def _build_sol_dwd_head(hidden_size: int, dwd_head_config: Mapping[str, Any]) -> SolH3DWDHead:
    """Instantiate :class:`SolH3DWDHead` from a LakonLab ``dwd_head_config`` mapping."""

    if dwd_head_config.get("linear_head", False):
        raise ValueError("pi-H3 SolH3DWDHead requires linear_head=False.")
    refiner_cfg = dict(dwd_head_config.get("pixel_refiner") or {})
    refiner_type = refiner_cfg.pop("type", "SolH3DWDHead")
    if refiner_type != "SolH3DWDHead":
        raise ValueError(f"Unsupported pi-H3 DWD pixel_refiner type: {refiner_type!r}.")
    # Drop training-only config keys; inference weights load strictly via denoisers/pi_h3.py.
    refiner_cfg.pop("pretrained", None)
    refiner_cfg.pop("checkpointing", None)
    return SolH3DWDHead(
        in_channels=hidden_size,
        src_patch_size=dwd_head_config.get("pixel_patch_size", (PATCH_T, 2 * PATCH, 2 * PATCH)),
        first_frame_src_patch_size=dwd_head_config.get(
            "first_frame_pixel_patch_size",
            (PATCH_T - FRAMES_TO_TRIM, 2 * PATCH, 2 * PATCH),
        ),
        first_frame_period=int(dwd_head_config.get("first_frame_period", BLOCK)),
        residual=False,
        pixel_channels=int(dwd_head_config.get("pixel_channels", OUT_CH)),
        **refiner_cfg,
    )


class PiH3DiTModel(MiniMaxH3DiTModel):
    """Pruned MiniMax-H3 DiT with optional Sol-H3 DWD pixel head.

    Reuses canonical :class:`MiniMaxH3DiTModel` attention, SwiGLU MLP, text
    refiner, and 3D RoPE while replacing the sinusoidal timestep MLP and
    full-rank AdaLN projections with ``MiniMax-H3-Pruned``'s rank-``adaln_rank``
    lookup table and folded linear projections.
    """

    adaln_basis: torch.Tensor
    adaln_mean: torch.Tensor
    time_embedder: PiH3PrunedTimeEmbedder
    final_layer: PiH3PrunedFinalLayer

    def __init__(
        self,
        config: MiniMaxH3DiTArchConfig | None = None,
        *,
        adaln_rank: int = 8,
        time_table_size: int = 1025,
        rope_theta: float = 10000.0,
        dwd_head_config: Mapping[str, Any] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **overrides: Any,
    ) -> None:
        arch = config if config is not None else MiniMaxH3DiTArchConfig(**overrides)
        super().__init__(arch)
        self.adaln_rank = int(adaln_rank)
        self.time_table_size = int(time_table_size)
        self.rope_theta = float(rope_theta)
        self.dwd_head_config = None if dwd_head_config is None else dict(dwd_head_config)

        # Match Diffusers/LakonLab uniform dtype on input projections.
        self.video_patch_proj = nn.Linear(arch.video_patch_output_dim, arch.hidden_size, bias=True, dtype=dtype)
        self.audio_patch_proj = nn.Linear(arch.audio_latents_dim, arch.hidden_size, bias=True, dtype=dtype)
        self.condition_proj = nn.Linear(arch.text_dim, arch.hidden_size, bias=True, dtype=dtype)

        # Pruned timestep lookup table and PCA metadata buffers stored in MiniMax-H3-Pruned.
        self.time_embedder = PiH3PrunedTimeEmbedder(table_size=self.time_table_size, adaln_rank=self.adaln_rank)
        self.register_buffer("adaln_basis", torch.zeros(self.adaln_rank, arch.time_embed_dim, dtype=dtype))
        self.register_buffer("adaln_mean", torch.zeros(arch.time_embed_dim, dtype=dtype))

        # Diffusers-format H3 checkpoints omit deterministic RoPE inv_freq; keep it fp32 and non-persistent.
        self.rope.register_buffer(
            "inv_freq",
            1.0
            / (
                self.rope_theta
                ** (
                    torch.arange(0, 2 * arch.rope_inv_freq_len, 2, device="cpu", dtype=torch.float32)
                    / (2 * arch.rope_inv_freq_len)
                )
            ),
            persistent=False,
        )

        # Retain Diffusers unfused Q/K/V and [up|gate] SwiGLU parameter layouts on canonical blocks.
        for block in [*self.token_refiner.blocks, *self.blocks]:
            attn: Any = block.attn
            mlp: Any = block.mlp
            attn.qkv_proj = HyperFlowQKVProjection(attn.qkv_proj)
            mlp.fc1 = HyperFlowSwiGLUProjection(mlp.fc1)
        for block in self.blocks:
            block.adaln_proj = PiH3PrunedAdalnProj(
                adaln_rank=self.adaln_rank,
                hidden_size=arch.hidden_size,
                expand_ratio=6,
                modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
                dtype=dtype,
            )

        self.final_layer = PiH3PrunedFinalLayer(arch, adaln_rank=self.adaln_rank, dtype=dtype)

        if dwd_head_config is not None:
            self.dwd_norm_out: PiH3DWDAdaLayerNormOut | None = PiH3DWDAdaLayerNormOut(
                arch.hidden_size,
                self.adaln_rank,
                eps=arch.final_norm_eps,
                dtype=dtype,
            )
            self.dwd_pixel_refiner: SolH3DWDHead | None = _build_sol_dwd_head(arch.hidden_size, dwd_head_config).to(
                dtype=dtype
            )
        else:
            self.dwd_norm_out = None
            self.dwd_pixel_refiner = None

        # Canonical MiniMaxH3DiTModel initializes refiner/DiT blocks in bf16; align to requested dtype
        # while keeping self.rope.inv_freq strictly in fp32.
        self.token_refiner.to(dtype=dtype)
        self.blocks.to(dtype=dtype)

    @property
    def has_dwd_head(self) -> bool:
        """Return whether this transformer includes a Sol-H3 DWD pixel refiner head."""

        return self.dwd_pixel_refiner is not None

    def post_load_weights(self) -> None:
        """Verify non-persistent RoPE frequencies stayed in fp32 after weight loading."""

        if self.rope.inv_freq.dtype != torch.float32:
            raise ValueError(f"rope.inv_freq must stay fp32 after load, got {self.rope.inv_freq.dtype}.")

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
        refiner_cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed packed text/video/audio tokens and look up pruned timestep coordinates."""

        if refined_prompt_embeds_length is None:
            text_len = (
                int(refiner_cu_seqlens_host[-1])
                if refiner_cu_seqlens_host is not None
                else int(refiner_cu_seqlens[-1].item())
            )
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
        dtype = self.condition_proj.weight.dtype
        if refined_prompt_embeds_length is not None:
            text_embed = text_embeddings_selected[:text_len].to(device=device, dtype=dtype)
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    f"refined prompt embeddings must have hidden width {self.hidden_size}, got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(
                text_embeddings_selected,
                refiner_cu_seqlens,
                device=device,
                refiner_cu_seqlens_host=refiner_cu_seqlens_host,
            )

        seq_len = int(x.shape[1])
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
        if text_pos.numel():
            embeddings.index_add_(0, text_pos, text_embed[:text_len].to(dtype))
        if img_pos.numel():
            x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(self.video_patch_proj.weight.dtype)
            embeddings.index_add_(0, img_pos, self.video_patch_proj(x_rows).to(dtype))
        if audio_pos.numel():
            audio_rows = (
                audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(self.audio_patch_proj.weight.dtype)
            )
            embeddings.index_add_(0, audio_pos, self.audio_patch_proj(audio_rows).to(dtype))

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> Any:
        """Run packed DiT inference; returns ``(video, audio)`` or ``((video, audio), features)``."""

        unexpected = sorted(set(kwargs) - _PI_H3_FORWARD_KWARGS)
        if unexpected:
            raise TypeError(
                f"PiH3DiTModel.forward received unexpected kwargs: {unexpected}; "
                f"supported kwargs: {sorted(_PI_H3_FORWARD_KWARGS)}"
            )

        return_features = bool(kwargs.get("return_features", False))
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
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(torch.int32)
        cu_seqlens_host = _normalize_host_bounds(self._psp_optional_field(psp, "cu_seqlens_q_host"))
        raw_max_seqlen = self._psp_optional_field(psp, "max_seqlen_q")
        max_seqlen = None if raw_max_seqlen is None else int(raw_max_seqlen)
        ignore_trailing_pad = bool(self._psp_optional_field(psp, "ignore_trailing_pad"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_cu_host = _normalize_host_bounds(self._psp_optional_field(refiner_psp, "cu_seqlens_q_host"))

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")

        device = x.device
        dtype = self.condition_proj.weight.dtype
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids).to(device)
            rope_cache = (
                rope_cos_sin_cache(rope_freqs, dtype=dtype),
                torch.arange(seq_len, device=device, dtype=torch.long),
            )

        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)
        infer_out_pos = infer_out_pos.to(device)

        hidden, t_emb = self._embed(
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
            refiner_cu_seqlens_host=refiner_cu_host,
        )
        # Pruned table entries already encode silu(t_emb); do NOT apply an extra SiLU here.
        temb = t_emb.to(dtype)
        inverse_indices = inverse_indices.to(device)
        if block_token_tags is None:
            assert token_tags is not None
            block_token_tags = token_tags.to(device).clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != seq_len:
                raise ValueError(
                    f"block_token_tags must cover the packed sequence ({seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(block_token_tags, inverse_indices, alpha=MINIMAX_H3_ADALN_MODALITY_NUM)
        cu_seqlens = cu_seqlens.to(device)

        prefix_sink_tokens = _compute_prefix_sink_tokens(
            text_pos=text_pos,
            img_pos=img_pos,
            infer_out_pos=infer_out_pos,
            audio_pos=audio_pos,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            refiner_cu_seqlens_host=refiner_cu_host,
        )
        force_dense_attention = bool(kwargs.get("force_dense_attention", False))
        hidden = self._run_transformer_blocks(
            hidden,
            adaln_input=temb,
            combined_indices=block_combined,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ignore_trailing_pad=ignore_trailing_pad,
            prefix_sink_tokens=prefix_sink_tokens,
            force_dense_attention=force_dense_attention,
        )

        norm_hidden = self.final_layer.modulate(hidden, adaln_input=temb, inverse_indices=inverse_indices)
        video_logits = self.final_layer.video_out(
            norm_hidden.index_select(0, infer_out_pos).to(self.final_layer.video_out.weight.dtype)
        )
        audio_logits = self.final_layer.audio_out(
            norm_hidden.index_select(0, audio_pos).to(self.final_layer.audio_out.weight.dtype)
        )
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)

        output = (video_logits, audio_logits)
        if not return_features:
            return output

        features = self.dwd_norm_out(hidden, temb, inverse_indices) if self.dwd_norm_out is not None else norm_hidden
        feature_dict = {
            "image": features.index_select(0, infer_out_pos),
            "audio": features.index_select(0, audio_pos),
        }
        return output, feature_dict

    def forward_dwd_head(
        self,
        features: torch.Tensor,
        token_grid_size: tuple[int, int, int],
        *,
        force_padded: bool = False,
    ) -> torch.Tensor:
        """Project final-layer video features directly to normalized RGB ``(B, 3, F, H_pix, W_pix)``."""

        if self.dwd_pixel_refiner is None:
            raise RuntimeError("This PiH3DiTModel was constructed without a SolH3DWDHead.")
        if features.ndim == 2:
            features = features.unsqueeze(0)
        pixels = self.dwd_pixel_refiner(None, features, token_grid_size, force_padded=force_padded)
        return pixels.transpose(1, 2)


__all__ = [
    "PI_H3_DEFAULT_DWD_HEAD_CONFIG",
    "PiH3DWDAdaLayerNormOut",
    "PiH3DiTModel",
    "PiH3PrunedAdalnProj",
    "PiH3PrunedFinalLayer",
    "PiH3PrunedTimeEmbedder",
]
