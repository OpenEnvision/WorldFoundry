# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Causal DreamDojo student on the canonical action-conditioned Cosmos DiT.

Adapted from NVIDIA/DreamDojo at 02f119b759d5c7f84a399fdeea3c6e82e7ed6cff,
cosmos_predict2/_src/predict2/interactive/networks/dit_action_causal.py.
"""

from __future__ import annotations

import torch
from einops import repeat

from worldfoundry.core.attention.cache.kv_cache_policy import SlidingWindowPolicy
from worldfoundry.core.attention.cache.kvcache import CompactingKVCache

from .action import ActionChunkCosmosTransformer


class CausalActionCosmosTransformer(ActionChunkCosmosTransformer):
    """Predict a single latent frame with caller-owned, three-frame KV history.

    Checkpoint parameters are identical to the bidirectional action DiT. Only
    clean prefill calls commit keys; noisy sampling calls leave history intact.
    Positions are absolute within each re-encoded 13-pixel-frame window.
    """

    def forward_frame(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        action: torch.Tensor | None,
        frame_index: int,
        kv_cache: dict[int, CompactingKVCache],
        store_kv: bool = False,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate ``[B,C,1,H,W]`` using four actions, or no action for frame 0."""
        batch, _, frames, height, width = hidden_states.shape
        if frames != 1 or frame_index < 0:
            raise ValueError("DreamDojo student requires one latent frame and a nonnegative frame index")
        if action is None and frame_index != 0:
            raise ValueError("DreamDojo student requires actions after the first observation")
        if action is not None and tuple(action.shape) != (batch, self.temporal_compression, self.action_dim):
            raise ValueError("DreamDojo student requires four normalized 384-D actions per latent frame")
        tokens, _, grid = self._prepare_tokens(
            hidden_states,
            embedder=self.patch_embed,
            fps=None,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
        )
        rope_input = hidden_states.expand(batch, -1, frame_index + 1, height, width)
        spatial_tokens = grid[1] * grid[2]
        rotary = tuple(value[-spatial_tokens:] for value in self.rope(rope_input, None))
        context = self.text_embed(encoder_hidden_states)
        projected, temb = self.time_embed(tokens, timestep.reshape(batch))
        projected = projected.unsqueeze(1)
        temb = temb.unsqueeze(1)
        if action is not None:
            grouped = action.reshape(batch, 1, -1).to(tokens)
            projected = projected + self.action_embedder_B_D(grouped)
            temb = temb + self.action_embedder_B_3D(grouped)
        projected = self.time_norm(projected)
        projected, temb = (repeat(value, "b t c -> b (t h w) c", h=grid[1], w=grid[2]) for value in (projected, temb))
        for index, block in enumerate(self.transformer_blocks):
            if index not in kv_cache:
                kv_cache[index] = CompactingKVCache(
                    frame_tokens=spatial_tokens,
                    policy=SlidingWindowPolicy(window_blocks=3),
                    seq_dim=2,
                )
            tokens = block(
                tokens,
                context,
                projected,
                temb,
                rotary,
                None,
                kv_cache=kv_cache[index],
                store_kv=store_kv,
            )
        return self._unpatchify(self.proj_out(self.norm_out(tokens, projected, temb)), grid)
