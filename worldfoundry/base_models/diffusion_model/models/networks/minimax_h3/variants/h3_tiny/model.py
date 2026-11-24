# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from fuyi25/H3-Tiny at a3f17f0ef85e9af5d07b2007c4b2dbd6d035e69c (Apache-2.0).
"""H3-Tiny pruned and MeanFlow few-step DiT built on the canonical MiniMax-H3 architecture."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...config import MINIMAX_H3_ADALN_MODALITY_NUM, MiniMaxH3DiTArchConfig
from ...dit import (
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
    MiniMaxH3TimeEmbedder,
    MiniMaxH3TokenRefiner,
    MiniMaxH3TokenRefinerBlock,
)
from ..hyperflow.conditioning import HyperFlowQKVProjection, HyperFlowSwiGLUProjection

H3_TINY_ENDPOINT_MODES: tuple[str, ...] = ("meanflow", "r_equals_t", "none")

_R_TIME_EMBEDDER_FP32_PARAM_NAMES = (
    "time_embedder.r_time_embedder.proj_in.weight",
    "time_embedder.r_time_embedder.proj_in.bias",
    "time_embedder.r_time_embedder.proj_out.weight",
    "time_embedder.r_time_embedder.proj_out.bias",
)


class LowRankAdalnProj(nn.Module):
    """Low-rank AdaLN modulation through a rank bottleneck: ``linear_2(linear_1(x))``.

    Matches Diffusers' ``LowRankModulation`` (`silu(temb) -> linear_1 -> linear_2`).
    Inside :class:`MiniMaxH3DiTModel`, ``F.silu(t_emb)`` is applied once before
    dispatching ``adaln_input`` to each block, so ``apply_silu`` defaults to
    ``False`` when installed in canonical blocks and can be set to ``True`` for
    standalone evaluation on pre-activation ``temb``.
    """

    def __init__(
        self,
        time_embed_dim: int,
        hidden_size: int,
        rank: int,
        *,
        expand_ratio: int = 6,
        modality_num: int = MINIMAX_H3_ADALN_MODALITY_NUM,
        apply_silu: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"adaln_rank must be a positive integer, got {rank!r}")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.expand_ratio = int(expand_ratio)
        self.modality_num = int(modality_num)
        self.apply_silu = bool(apply_silu)
        out_features = self.expand_ratio * self.hidden_size * self.modality_num
        self.linear_1 = nn.Linear(int(time_embed_dim), self.rank, bias=True, dtype=dtype)
        self.linear_2 = nn.Linear(self.rank, out_features, bias=True, dtype=dtype)

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = F.silu(adaln_input) if self.apply_silu else adaln_input
        projected = self.linear_2(self.linear_1(hidden.to(self.linear_1.weight.dtype)))
        m = projected.shape[0]
        return tuple(
            projected.view(m * self.modality_num, self.expand_ratio * self.hidden_size).chunk(self.expand_ratio, dim=-1)
        )


class H3TinyMeanFlowTimeEmbedder(MiniMaxH3TimeEmbedder):
    """Canonical FP32 timestep embedder with optional MeanFlow interval branch.

    Computes ``temb(t, r) = emb_t(t) + emb_r(r - t)`` in FP32 on the ``t = 1 - sigma``
    axis. When ``endpoint_mode == "r_equals_t"``, ``r`` is pinned to ``t`` so the
    interval branch evaluates at ``0``; when ``endpoint_mode == "none"``, no interval
    branch is instantiated.
    """

    def __init__(self, arch: MiniMaxH3DiTArchConfig, *, endpoint_mode: str = "meanflow") -> None:
        if endpoint_mode not in H3_TINY_ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {H3_TINY_ENDPOINT_MODES}, got {endpoint_mode!r}")
        super().__init__(arch)
        self.endpoint_mode = endpoint_mode
        self.r_time_embedder: MiniMaxH3TimeEmbedder | None = (
            MiniMaxH3TimeEmbedder(arch) if endpoint_mode != "none" else None
        )
        self._endpoint_timestep: torch.Tensor | None = None

    @contextmanager
    def endpoint_context(self, endpoint_timestep: torch.Tensor | None) -> Generator[None, None, None]:
        previous = self._endpoint_timestep
        self._endpoint_timestep = endpoint_timestep
        try:
            yield
        finally:
            self._endpoint_timestep = previous

    def forward_with_endpoint(
        self,
        timestep: torch.Tensor,
        endpoint_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_fp32 = timestep.to(torch.float32)
        temb = super().forward(t_fp32)
        if self.endpoint_mode == "none":
            return temb
        if self.r_time_embedder is None:
            raise RuntimeError("H3TinyMeanFlowTimeEmbedder missing r_time_embedder for active endpoint mode.")
        if self.endpoint_mode == "r_equals_t" or endpoint_timestep is None:
            endpoint = t_fp32
        else:
            endpoint = endpoint_timestep.to(device=t_fp32.device, dtype=torch.float32)
        if endpoint.shape != t_fp32.shape:
            raise ValueError(f"endpoint_timestep {list(endpoint.shape)} != timestep {list(t_fp32.shape)}")
        if (
            not getattr(self, "_skip_host_validation", False)
            and not (t_fp32.is_cuda and torch.cuda.is_current_stream_capturing())
            and bool((endpoint < t_fp32).any())
        ):
            raise ValueError("MeanFlow endpoints must be dataward of their timesteps (r >= t)")
        interval = endpoint - t_fp32
        return temb + self.r_time_embedder(interval)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward_with_endpoint(t, self._endpoint_timestep)


class H3TinyDiTModel(MiniMaxH3DiTModel):
    """Structured-pruned MiniMax-H3 DiT with low-rank AdaLN and optional MeanFlow conditioning.

    Reuses the canonical :class:`MiniMaxH3DiTModel` blocks, token refiner, RoPE,
    patch projections, and output heads while supporting:
    - pruned transformer depth (``num_layers``, e.g. 42) and narrowed block FFN width
      (``ffn_hidden_size``, e.g. 9792) independent of ``refiner_ffn_hidden_size`` (14336);
    - low-rank per-block AdaLN projections (``adaln_rank``, e.g. 8);
    - MeanFlow interval conditioning (``endpoint_mode`` in ``("meanflow", "r_equals_t", "none")``);
    - Diffusers-order separate Q/K/V and ``[up, gate]`` SwiGLU projections for exact weight loading.
    """

    def __init__(
        self,
        config: MiniMaxH3DiTArchConfig,
        *,
        refiner_ffn_hidden_size: int | None = None,
        adaln_rank: int | None = None,
        endpoint_mode: str = "meanflow",
        rope_theta: float = 10000.0,
    ) -> None:
        if endpoint_mode not in H3_TINY_ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {H3_TINY_ENDPOINT_MODES}, got {endpoint_mode!r}")
        super().__init__(config)
        self.refiner_ffn_hidden_size = (
            config.ffn_hidden_size if refiner_ffn_hidden_size is None else int(refiner_ffn_hidden_size)
        )
        self.adaln_rank = None if adaln_rank is None else int(adaln_rank)
        self.endpoint_mode = endpoint_mode
        self.rope_theta = float(rope_theta)

        if self.refiner_ffn_hidden_size != config.ffn_hidden_size:
            refiner_arch = replace(config, ffn_hidden_size=self.refiner_ffn_hidden_size)
            self.token_refiner = MiniMaxH3TokenRefiner(refiner_arch)

        for block in (*self.token_refiner.blocks, *self.blocks):
            if not isinstance(block, (MiniMaxH3DiTBlock, MiniMaxH3TokenRefinerBlock)):
                raise TypeError(f"Expected MiniMax-H3 block, got {type(block).__name__}")
            if not isinstance(block.attn.qkv_proj, nn.Linear) or not isinstance(block.mlp.fc1, nn.Linear):
                raise TypeError("Expected canonical nn.Linear qkv_proj and fc1 before H3-Tiny projection wrapping.")
            block.attn.add_module("qkv_proj", HyperFlowQKVProjection(block.attn.qkv_proj))
            block.mlp.add_module("fc1", HyperFlowSwiGLUProjection(block.mlp.fc1))

        if self.adaln_rank is not None:
            for block in self.blocks:
                block.adaln_proj = LowRankAdalnProj(
                    config.time_embed_dim,
                    config.hidden_size,
                    self.adaln_rank,
                    apply_silu=False,
                )

        self.time_embedder: H3TinyMeanFlowTimeEmbedder = H3TinyMeanFlowTimeEmbedder(config, endpoint_mode=endpoint_mode)
        self.rebuild_rope(device="cpu")

    def rebuild_rope(self, device: str | torch.device = "cpu") -> None:
        """Deterministically rebuild non-persistent FP32 RoPE inverse frequencies."""
        dim = self.arch.rope_inv_freq_len
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, 2 * dim, 2, device="cpu", dtype=torch.float32) / (2 * dim))
        )
        self.rope.register_buffer("inv_freq", inv_freq.to(device=device, dtype=torch.float32), persistent=False)

    def post_load_weights(self) -> None:
        """Verify canonical FP32/BF16 parameter split, including ``r_time_embedder`` when active."""
        super().post_load_weights()
        if self.time_embedder.r_time_embedder is not None:
            for name in _R_TIME_EMBEDDER_FP32_PARAM_NAMES:
                param = self.get_parameter(name)
                if param.dtype != torch.float32:
                    raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")

    def time_embedding(
        self,
        timestep: torch.Tensor,
        endpoint_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate ``temb(t, r) = emb_t(t) + emb_r(r - t)`` directly in FP32."""
        return self.time_embedder.forward_with_endpoint(timestep, endpoint_timestep)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor | None = None,
        audio_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        timestep_indices: torch.Tensor | None = None,
        token_tags: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        video_indices: torch.Tensor | None = None,
        audio_indices: torch.Tensor | None = None,
        text_indices: torch.Tensor | None = None,
        endpoint_timestep: torch.Tensor | None = None,
        **extra: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict video/audio velocities for one unpadded, batch-one packed layout."""
        if hidden_states is None and "x" in extra:
            if token_tags is not None:
                extra["token_tags"] = token_tags
            with self.time_embedder.endpoint_context(endpoint_timestep):
                return super().forward(**extra)
        if extra:
            raise TypeError(f"Unsupported H3TinyDiTModel.forward arguments: {sorted(extra)}")
        if (
            hidden_states is None
            or audio_hidden_states is None
            or encoder_hidden_states is None
            or timestep is None
            or timestep_indices is None
            or token_tags is None
            or position_ids is None
            or video_indices is None
            or audio_indices is None
            or text_indices is None
        ):
            raise TypeError("H3TinyDiTModel.forward requires all packed video/audio/text tensor inputs.")
        length = int(position_ids.shape[0])
        if tuple(token_tags.shape) != (length,) or tuple(timestep_indices.shape) != (length,):
            raise ValueError("`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors.")
        device = hidden_states.device
        video = hidden_states.new_zeros((1, length, hidden_states.shape[-1]))
        audio = audio_hidden_states.new_zeros((1, length, audio_hidden_states.shape[-1]))
        video.index_copy_(1, video_indices, hidden_states)
        audio.index_copy_(1, audio_indices, audio_hidden_states)
        with self.time_embedder.endpoint_context(endpoint_timestep):
            video_out, audio_out = super().forward(
                x=video,
                audio_x=audio,
                prompt_embeds=encoder_hidden_states[0],
                unique_timesteps=timestep,
                inverse_indices=timestep_indices,
                token_tags=token_tags,
                img_position_ids=position_ids[None],
                img_pos_info={"position_ids": video_indices},
                audio_pos_info={"position_ids": audio_indices},
                text_pos_info={"position_ids": text_indices},
                img_pos_for_infer_output_info={"position_ids": video_indices},
                update_mask=hidden_states.new_ones(video_indices.shape),
                skip_mask_out_condition=True,
                packed_seq_params={
                    "cu_seqlens_q": torch.tensor([0, length], device=device, dtype=torch.int32),
                    "cu_seqlens_q_host": (0, length),
                },
                refiner_packed_seq_params={
                    "cu_seqlens_q": torch.tensor([0, text_indices.numel()], device=device, dtype=torch.int32),
                    "cu_seqlens_q_host": (0, int(text_indices.numel())),
                },
            )
        return video_out[None], audio_out[None]


__all__ = [
    "H3_TINY_ENDPOINT_MODES",
    "H3TinyDiTModel",
    "H3TinyMeanFlowTimeEmbedder",
    "LowRankAdalnProj",
]
