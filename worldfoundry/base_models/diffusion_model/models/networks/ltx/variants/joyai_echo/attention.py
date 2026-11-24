# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Echo's raw-key, text and rebased causal caches on native LTX attention.

Adapted from JoyAI-Echo and Lightricks LTX-2; see THIRD-PARTY-NOTICES.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import apply_rotary_emb

if TYPE_CHECKING:
    from worldfoundry.core.attention.cache.context import ContextAttentionCache


def _slice_rope(pe: tuple[torch.Tensor, torch.Tensor], start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    return pe[0][..., start:end, :], pe[1][..., start:end, :]


def update_kv_cache(
    cache: dict,
    start: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert one global token range and return the active sink+FIFO window.

    The cache is deliberately inference-only. Repeated denoising forwards for
    the same block replace that block in place; the clean refresh forward then
    replaces it once more before the next block starts.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("causal KV caches are inference-only")
    if k.shape != v.shape or k.ndim != 3:
        raise ValueError("KV tensors must have matching [batch, tokens, dim] shapes")

    length = int(cache["length"])
    old_positions = cache["positions"][:length]
    old_k = cache["k"][:, :length]
    old_v = cache["v"][:, :length]
    end = start + k.shape[1]

    # A denoising step is a transaction over [start, end): discard a previous
    # noisy version of that range while retaining earlier committed history.
    keep_old = old_positions < start
    positions = torch.cat([old_positions[keep_old], torch.arange(start, end, device=k.device)], dim=0)
    merged_k = torch.cat([old_k[:, keep_old], k], dim=1)
    merged_v = torch.cat([old_v[:, keep_old], v], dim=1)

    local = int(cache.get("local_attn_size", -1))
    sink = int(cache.get("sink_tokens", 0))
    if local >= 0 and positions.numel() > local:
        if not 0 <= sink < local:
            raise ValueError(f"expected 0 <= sink_tokens < local_attn_size, got {sink}/{local}")
        sink_mask = positions < sink
        recent_budget = local - int(sink_mask.sum())
        recent_start = max(sink, end - recent_budget)
        keep = sink_mask | (positions >= recent_start)
        positions = positions[keep]
        merged_k = merged_k[:, keep]
        merged_v = merged_v[:, keep]

    active = positions.numel()
    if active > cache["k"].shape[1]:
        raise ValueError(f"KV cache overflow: {active} active tokens exceed capacity {cache['k'].shape[1]}")
    cache["k"][:, :active].copy_(merged_k)
    cache["v"][:, :active].copy_(merged_v)
    cache["positions"][:active].copy_(positions)
    cache["length"] = active
    return cache["k"][:, :active].clone(), cache["v"][:, :active].clone()


class EchoCachedAttention(Attention):
    """Share native attention parameters while retaining Echo cache ownership and rebasing."""

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        camera_control: object | None = None,
        cache: ContextAttentionCache | None = None,
        *,
        kv_cache: dict | None = None,
        kv_cache_start: int = 0,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        """Use the native path without Echo caches; retain raw-key rebasing for causal inference."""
        if kv_cache is None and crossattn_cache is None:
            return super().forward(x, context, mask, pe, k_pe, perturbation_mask, all_perturbed, camera_control, cache)
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.q_norm(self.to_q(x))
            if crossattn_cache is not None:
                if not crossattn_cache["is_init"]:
                    cached_k = self.k_norm(self.to_k(context))
                    size = cached_k.shape[1]
                    crossattn_cache["k"][:, :size].copy_(cached_k)
                    crossattn_cache["v"][:, :size].copy_(v)
                    crossattn_cache["length"] = size
                    crossattn_cache["is_init"] = True
                size = int(crossattn_cache["length"])
                k = crossattn_cache["k"][:, :size]
                v = crossattn_cache["v"][:, :size]
                if pe is not None:
                    q = self.apply_rotary_embedding(q, pe)
            else:
                k = self.k_norm(self.to_k(context))
                local_pe = kv_cache.get("local_rope_pe") if kv_cache is not None else None
                local_q_pe = kv_cache.get("local_cross_q_rope_pe") if kv_cache is not None else None
                local_k_pe = kv_cache.get("local_cross_k_rope_pe") if kv_cache is not None else None
                if kv_cache is not None and local_pe is not None:
                    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
                    active = k.shape[1]
                    q_len = q.shape[1]
                    q = apply_rotary_emb(q, _slice_rope(local_pe, active - q_len, active), self.rope_type)
                    k = apply_rotary_emb(k, _slice_rope(local_pe, 0, active), self.rope_type)
                elif kv_cache is not None and (local_q_pe is not None or local_k_pe is not None):
                    if local_q_pe is None or local_k_pe is None:
                        raise ValueError("cross-modal RoPE rebase requires both query and key templates")
                    new_keys = k.shape[1]
                    k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)
                    query_slice = kv_cache["local_cross_q_slices"].get((kv_cache_start, kv_cache_start + new_keys))
                    if query_slice is None:
                        raise ValueError("missing local cross-modal query RoPE slice")
                    q = apply_rotary_emb(q, _slice_rope(local_q_pe, *query_slice), self.rope_type)
                    k = apply_rotary_emb(k, _slice_rope(local_k_pe, 0, k.shape[1]), self.rope_type)
                else:
                    if pe is not None:
                        q = self.apply_rotary_embedding(q, pe)
                        k = self.apply_rotary_embedding(k, pe if k_pe is None else k_pe)
                    if kv_cache is not None:
                        k, v = update_kv_cache(kv_cache, kv_cache_start, k, v)

            out = (
                self.attention_function(q, k, v, self.heads)
                if mask is None
                else self.masked_attention_function(q, k, v, self.heads, mask)
            )

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)

        return self.to_out(out)
