# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Released HyperFlow projections and two-time inputs on the canonical H3 DiT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from worldfoundry.core.distributed.collectives.tensor import all_gather_concat

from ...config import MiniMaxH3DiTArchConfig
from ...dit import MiniMaxH3DiTModel, _coerce_context_parallel_config
from .attention import HyperFlowAttention, HyperFlowAttentionState
from .conditioning import HyperFlowQKVProjection, HyperFlowSwiGLUProjection, HyperFlowTwoTimeEmbedder


class HyperFlowDiTModel(MiniMaxH3DiTModel):
    """Reuse H3 embedding, refiner, attention, AdaLN, MLP, and output-head math."""

    def __init__(
        self,
        config: MiniMaxH3DiTArchConfig,
        *,
        attention_backend: str = "native",
        sol_attn: bool = False,
        degree: int = 1,
        rank: int = 0,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__(config)
        self.attention_state = HyperFlowAttentionState(degree=degree, rank=rank)
        # The Diffusers-format release omits this deterministic buffer.
        self.rope.register_buffer(
            "inv_freq",
            1.0
            / (
                rope_theta
                ** (
                    torch.arange(0, 2 * config.rope_inv_freq_len, 2, device="cpu").float()
                    / (2 * config.rope_inv_freq_len)
                )
            ),
            persistent=False,
        )
        for layer, block in [(None, block) for block in self.token_refiner.blocks] + list(enumerate(self.blocks)):
            block.attn.qkv_proj = HyperFlowQKVProjection(block.attn.qkv_proj)
            block.mlp.fc1 = HyperFlowSwiGLUProjection(block.mlp.fc1)
            block.attn.attention_fn = HyperFlowAttention(
                self.attention_state, layer=layer, backend=attention_backend, sol_attn=sol_attn
            )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        endpoint: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict video/audio velocities for one unpadded, batch-one packed layout."""
        if not isinstance(self.time_embedder, HyperFlowTwoTimeEmbedder):
            raise RuntimeError("Load the HyperFlow adapter before denoising.")
        length = position_ids.shape[0]
        device = hidden_states.device
        video = hidden_states.new_zeros((1, length, hidden_states.shape[-1]))
        audio = audio_hidden_states.new_zeros((1, length, audio_hidden_states.shape[-1]))
        video.index_copy_(1, video_indices, hidden_states)
        audio.index_copy_(1, audio_indices, audio_hidden_states)
        self.attention_state.step = step
        try:
            with self.time_embedder.endpoint_context(endpoint):
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
                    packed_seq_params={"cu_seqlens_q": torch.tensor([0, length], device=device, dtype=torch.int32)},
                    refiner_packed_seq_params={
                        "cu_seqlens_q": torch.tensor([0, text_indices.numel()], device=device, dtype=torch.int32)
                    },
                )
        finally:
            self.attention_state.step = None
        return video_out[None], audio_out[None]

    def set_context_parallel(self, config: Any) -> dict[str, Any]:
        parsed = _coerce_context_parallel_config(config)
        if self.attention_state.degree > 1 and parsed is not None and parsed.sp_size > 1:
            raise ValueError(
                "HyperFlowDiTModel with degree > 1 cannot be combined with Ulysses context_parallel > 1 "
                "(would double-shard the sequence)."
            )
        return super().set_context_parallel(config)

    def forward_blocks(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        inverse_indices: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        step_cache: object | None = None,
        step_index: int | None = None,
        context_parallel: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.attention_state
        state.sequence_length = hidden.shape[0]
        if state.degree > 1:
            eff_cp = self._context_parallel if context_parallel is None else context_parallel
            if eff_cp not in (None, False, 1):
                parsed_cp = _coerce_context_parallel_config(eff_cp)  # type: ignore[arg-type]
                if parsed_cp is not None and parsed_cp.sp_size > 1:
                    raise ValueError(
                        "HyperFlowDiTModel with degree > 1 cannot be combined with Ulysses context_parallel > 1 "
                        "(would double-shard the sequence)."
                    )
            chunk = (state.sequence_length + state.degree - 1) // state.degree
            padding = chunk * state.degree - state.sequence_length

            def split_rows(value: torch.Tensor) -> torch.Tensor:
                pad: tuple[int, ...] = (0, 0) * (value.ndim - 1) + (0, padding)
                return F.pad(value, pad)[state.rank * chunk : (state.rank + 1) * chunk].contiguous()

            hidden = split_rows(hidden)
            combined_indices = split_rows(combined_indices)
            inverse_indices = split_rows(inverse_indices)
            rope_cache = tuple(split_rows(value) for value in rope_cache)
            cu_seqlens = torch.tensor([0, chunk], device=hidden.device, dtype=torch.int32)
            cu_seqlens_host = (0, chunk)
            max_seqlen = chunk
            ignore_trailing_pad = False
        outputs = super().forward_blocks(
            hidden,
            adaln_input=adaln_input,
            combined_indices=combined_indices,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            inverse_indices=inverse_indices,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ignore_trailing_pad=ignore_trailing_pad,
            prefix_sink_tokens=prefix_sink_tokens,
            force_dense_attention=force_dense_attention,
            step_cache=step_cache,
            step_index=step_index,
            context_parallel=context_parallel,
        )
        if state.degree > 1:
            outputs = tuple(all_gather_concat(value, dim=0)[: state.sequence_length] for value in outputs)
        return outputs
