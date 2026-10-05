"""Native spatial, temporal, and text-cross attention for Vchitect-2.

Video tokens are packed as a frame-batch ``(B·T) S C`` so spatial
attention is joint over video + context tokens on each frame.
Temporal attention rearranges to ``(B·S) T C``, applies 1-D RoPE
along T, then returns to frame-batch layout.  Text cross-attention
uses the first-frame context keys/values against queries flattened
over ``S·T``.

Returns ``(video, context)`` residuals.  ``context_pre_only`` skips
the extra context output projection on early blocks, matching the
released Vchitect-2 checkpoint.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from worldfoundry.core.attention import scaled_dot_product_attention


class Attention(nn.Module):
    """Checkpoint-compatible Vchitect attention projections and processor."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        context_pre_only: bool,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.context_pre_only = bool(context_pre_only)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_q_cross = nn.Linear(dim, dim)
        self.to_q_temp = nn.Linear(dim, dim)
        self.to_k_temp = nn.Linear(dim, dim)
        self.to_v_temp = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList((nn.Linear(dim, dim), nn.Dropout(0.0)))
        self.to_out_temporal = nn.Linear(dim, dim)
        if not self.context_pre_only:
            self.to_add_out = nn.Linear(dim, dim)
        self.to_add_out_temporal = nn.Linear(dim, dim)
        self.to_out_context = nn.Linear(dim, dim)

    @staticmethod
    def _rotary(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        """Apply complex 1D RoPE along the temporal axis of ``(..., T, H, D)``."""
        complex_values = torch.view_as_complex(values.float().reshape(*values.shape[:-1], -1, 2))
        frequencies = frequencies[: values.shape[1]].view(1, values.shape[1], 1, -1)
        return torch.view_as_real(complex_values * frequencies).flatten(3).to(values.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        frequencies: torch.Tensor,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatial joint attn + temporal RoPE attn + text cross; return ``(video, context)``."""
        residual_length = hidden_states.shape[1]
        frame_batch = hidden_states.shape[0]
        if frame_batch % frames:
            raise ValueError("Vchitect frame-batch dimension must be divisible by frames")
        batch = frame_batch // frames
        head_dim = hidden_states.shape[-1] // self.heads

        context_q = self.add_q_proj(encoder_hidden_states)
        context_k = self.add_k_proj(encoder_hidden_states)
        context_v = self.add_v_proj(encoder_hidden_states)

        query = torch.cat((self.to_q(hidden_states), context_q), dim=1)
        key = torch.cat((self.to_k(hidden_states), context_k), dim=1)
        value = torch.cat((self.to_v(hidden_states), context_v), dim=1)
        temporal_q = torch.cat((self.to_q_temp(hidden_states), context_q), dim=1)
        temporal_k = torch.cat((self.to_k_temp(hidden_states), context_k), dim=1)
        temporal_v = torch.cat((self.to_v_temp(hidden_states), context_v), dim=1)
        cross_q = torch.cat((self.to_q_cross(hidden_states), context_q), dim=1)

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(frame_batch, -1, self.heads, head_dim)

        query, key, value = heads(query), heads(key), heads(value)
        temporal_q, temporal_k, temporal_v = heads(temporal_q), heads(temporal_k), heads(temporal_v)

        spatial = scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2).reshape(frame_batch, -1, self.heads * head_dim)

        temporal_q = rearrange(temporal_q, "(b t) s h d -> (b s) t h d", b=batch, t=frames)
        temporal_k = rearrange(temporal_k, "(b t) s h d -> (b s) t h d", b=batch, t=frames)
        temporal_v = rearrange(temporal_v, "(b t) s h d -> (b s) t h d", b=batch, t=frames)
        temporal_q = self._rotary(temporal_q, frequencies)
        temporal_k = self._rotary(temporal_k, frequencies)
        temporal = scaled_dot_product_attention(
            temporal_q.transpose(1, 2),
            temporal_k.transpose(1, 2),
            temporal_v.transpose(1, 2),
        ).transpose(1, 2).reshape(temporal_v.shape[0], frames, self.heads * head_dim)
        temporal = rearrange(temporal, "(b s) t d -> (b t) s d", b=batch, t=frames)

        cross_q = heads(cross_q)
        cross_q = rearrange(cross_q, "(b t) s h d -> b h (s t) d", b=batch, t=frames)
        key_y = heads(context_k)[::frames].transpose(1, 2)
        value_y = heads(context_v)[::frames].transpose(1, 2)
        cross = scaled_dot_product_attention(cross_q, key_y, value_y)
        cross = rearrange(
            cross.transpose(1, 2).reshape(batch, -1, self.heads * head_dim),
            "b (s t) d -> (b t) s d",
            b=batch,
            t=frames,
        )
        spatial = spatial * 1.1 + self.to_out_context(cross)

        spatial_hidden, spatial_context = spatial[:, :residual_length], spatial[:, residual_length:]
        temporal_hidden, temporal_context = temporal[:, :residual_length], temporal[:, residual_length:]
        hidden_output = self.to_out[1](self.to_out[0](spatial_hidden))
        if frames > 1:
            hidden_output = hidden_output + self.to_out_temporal(temporal_hidden)

        context_output = spatial_context
        if not self.context_pre_only:
            context_output = self.to_add_out(context_output)
        if frames > 1:
            context_output = context_output + self.to_add_out_temporal(temporal_context)
        return hidden_output, context_output


__all__ = ["Attention"]
