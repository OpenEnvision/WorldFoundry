"""Native Wan residual blocks used by Echo-Memory checkpoint recipes.

Echo-Memory extends the packed Wan DiT with optional action embeddings
and temporal memory slots declared by an immutable recipe.  Video tokens
stay ``B (F·HW) C``; 12-D relative-RT actions are ``B F 12`` and are
broadcast over each spatial grid.  Spatial-grid recipes pool context
frames into a fixed token set and inject them via text concat or gated
cross-attention.

State-space variants scan along latent time independently per spatial
token (block-wise SSM) or apply a depthwise temporal convolution
(VideoSSM hybrid).  Modules are selected by
:class:`~.schema.EchoMemoryRecipe`, not by runtime flags.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.core.attention import packed_sequence_attention
from worldfoundry.core.nn import RMSNorm
from worldfoundry.core.nn import scale_shift as modulate

from ..wan.model import DiTBlock, _SELF_ATTENTION_INTERNAL_KWARGS
from .architecture import EchoCheckpointArchitecture
from .schema import EchoMemoryMechanism, EchoMemoryRecipe, SpatialInjection


class CameraPoseEmbedding(nn.Module):
    """Zero-initialized 12D relative-RT projection used by released weights."""

    def __init__(self, dim: int, pose_dim: int = 12) -> None:
        super().__init__()
        self.proj = nn.Linear(int(pose_dim), int(dim))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Project ``[B, F, 12]`` relative-RT actions to ``[B, F, dim]``."""
        return self.proj(actions)


class EchoTemporalSelfAttention(nn.Module):
    """Checkpoint-compatible attention used by the optional action branch."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q = nn.Linear(self.dim, self.dim)
        self.k = nn.Linear(self.dim, self.dim)
        self.v = nn.Linear(self.dim, self.dim)
        self.o = nn.Linear(self.dim, self.dim)
        self.norm_q = RMSNorm(self.dim, eps=eps)
        self.norm_k = RMSNorm(self.dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attend a temporal sequence ``[B·HW, F, C]`` (no RoPE)."""
        batch, sequence = x.shape[:2]
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        return self.o(packed_sequence_attention(q=q, k=k, v=v, num_heads=self.num_heads))


class BlockWiseStateSpaceMemory(nn.Module):
    """Per-spatial-token recurrent state update along latent time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.in_proj = nn.Linear(self.dim, self.dim * 2)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.decay_logit = nn.Parameter(torch.zeros(self.dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, *, frames: int) -> torch.Tensor:
        """Scan ``B (F·HW) C`` along F independently per spatial token."""
        batch, tokens, dim = x.shape
        if frames <= 1 or dim != self.dim or tokens % frames:
            return x
        spatial = tokens // frames
        sequence = x.reshape(batch, frames, spatial, dim).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch * spatial, frames, dim)
        update, update_gate = self.in_proj(sequence).chunk(2, dim=-1)
        update = torch.tanh(update)
        update_gate = torch.sigmoid(update_gate)
        decay = torch.sigmoid(self.decay_logit).to(x).view(1, dim)
        state = torch.zeros(sequence.shape[0], dim, dtype=x.dtype, device=x.device)
        states: list[torch.Tensor] = []
        for index in range(frames):
            state = decay * state + (1.0 - decay) * update[:, index]
            states.append(state * update_gate[:, index])
        output = self.out_proj(torch.stack(states, dim=1))
        output = output.reshape(batch, spatial, frames, dim).permute(0, 2, 1, 3)
        output = output.reshape(batch, tokens, dim)
        return x + torch.tanh(self.gate) * output


class VideoSSMHybridMemory(nn.Module):
    """Legacy depthwise temporal-convolution state-space hybrid."""

    def __init__(self, dim: int, kernel_size: int = 3, expand: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        hidden = self.dim * max(int(expand), 1)
        self.in_proj = nn.Linear(self.dim, hidden)
        self.dw = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=self.kernel_size,
            groups=hidden,
            padding=self.kernel_size - 1,
        )
        self.out_proj = nn.Linear(hidden, self.dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, *, frames: int) -> torch.Tensor:
        """Depthwise temporal conv over the F axis of ``B (F·HW) C`` tokens."""
        batch, tokens, dim = x.shape
        if frames <= 1 or dim != self.dim or tokens % frames:
            return x
        spatial = tokens // frames
        sequence = x.reshape(batch, frames, spatial, dim).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch * spatial, frames, dim)
        output = self.in_proj(sequence).transpose(1, 2)
        output = self.dw(output)[..., :frames].transpose(1, 2)
        output = self.out_proj(output)
        output = output.reshape(batch, spatial, frames, dim).permute(0, 2, 1, 3)
        output = output.reshape(batch, tokens, dim)
        return x + torch.tanh(self.gate) * output


class SpatialGridMemory(nn.Module):
    """Pool context features to a learned, fixed-size spatial token grid."""

    def __init__(self, dim: int, *, grid_size: int = 8, num_tokens: int = 64) -> None:
        super().__init__()
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.num_tokens = int(num_tokens)
        self.spatial_to_tokens = nn.Parameter(torch.empty(self.grid_size * self.grid_size, self.num_tokens))
        nn.init.normal_(self.spatial_to_tokens, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        *,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Pool context tokens to ``[B, num_tokens, C]`` via a learned spatial grid."""
        batch, tokens, dim = context_tokens.shape
        if dim != self.dim:
            raise ValueError(f"spatial memory dim mismatch: input={dim}, module={self.dim}")
        spatial = int(height) * int(width)
        if tokens == int(frames) * spatial:
            values = context_tokens.reshape(batch, int(frames), spatial, dim).mean(dim=1)
        else:
            values = context_tokens
        grid_cells = self.grid_size * self.grid_size
        pooled = F.adaptive_avg_pool1d(values.transpose(1, 2), grid_cells).transpose(1, 2)
        weights = torch.softmax(self.spatial_to_tokens, dim=0)
        return torch.einsum("bgd,gm->bmd", pooled, weights)


class SpatialCrossAttentionReadout(nn.Module):
    """Gated target readout used by the released cross-attention ablation."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(int(dim), int(num_heads), batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Gated cross-attn: target queries attend to spatial memory tokens."""
        delta, _ = self.attn(target, memory, memory, need_weights=False)
        return target + torch.tanh(self.gate) * delta


class EchoWanAttentionBlock(DiTBlock):
    """Canonical Wan block plus checkpoint-declared Echo memory slots."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        action_attention: bool = False,
        block_state_space: bool = False,
        video_ssm_hybrid: bool = False,
    ) -> None:
        super().__init__(
            has_image_input,
            dim,
            num_heads,
            ffn_dim,
            eps,
        )
        self.action_mlp = CameraPoseEmbedding(dim)
        if action_attention:
            self.self_attn_with_action = EchoTemporalSelfAttention(dim, num_heads, eps)
            nn.init.zeros_(self.self_attn_with_action.o.weight)
            nn.init.zeros_(self.self_attn_with_action.o.bias)
        if block_state_space:
            self.block_wise_ssm = BlockWiseStateSpaceMemory(dim)
        if video_ssm_hybrid:
            self.videossm_hybrid = VideoSSMHybridMemory(dim)

    def _inject_actions(
        self,
        x: torch.Tensor,
        memory_context: Mapping[str, Any],
    ) -> torch.Tensor:
        """Add per-frame 12D action embeddings, broadcast over ``H·W`` tokens."""
        frames, height, width = _memory_grid_size(memory_context)
        actions = memory_context.get("actions")
        if actions is None:
            raise ValueError("Echo-Memory blocks require memory_context['actions']")
        actions = actions if torch.is_tensor(actions) else torch.as_tensor(actions)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3 or actions.shape[0] != x.shape[0] or actions.shape[-1] != 12:
            raise ValueError(
                "Echo actions must be [B,F,12] and match the latent batch; "
                f"got {tuple(actions.shape)} for batch {x.shape[0]}"
            )
        if int(actions.shape[1]) != frames:
            raise ValueError(f"Echo action length {actions.shape[1]} does not match latent frames {frames}")
        if int(x.shape[1]) != frames * height * width:
            raise ValueError("Echo token length does not match the declared latent grid")
        embedded = self.action_mlp(actions.to(device=x.device, dtype=self.action_mlp.proj.weight.dtype)).to(
            dtype=x.dtype
        )
        return x + embedded.repeat_interleave(height * width, dim=1)

    def _action_attention(
        self,
        action_x: torch.Tensor,
        original_x: torch.Tensor,
        memory_context: Mapping[str, Any],
    ) -> torch.Tensor:
        """Optional temporal self-attn over F after reshaping ``B (F·HW) C``."""
        module = getattr(self, "self_attn_with_action", None)
        if module is None:
            return action_x
        frames, height, width = _memory_grid_size(memory_context)
        batch = int(action_x.shape[0])
        spatial = height * width
        sequence = action_x.reshape(batch, frames, spatial, self.dim)
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * spatial, frames, self.dim)
        output = module(sequence)
        output = output.reshape(batch, spatial, frames, self.dim).permute(0, 2, 1, 3)
        return original_x + output.reshape(batch, frames * spatial, self.dim)

    def _apply_temporal_memory(
        self,
        x: torch.Tensor,
        memory_context: Mapping[str, Any],
    ) -> torch.Tensor:
        """Run the recipe's SSM / VideoSSM branch when those modules exist."""
        block_ssm = getattr(self, "block_wise_ssm", None)
        video_ssm = getattr(self, "videossm_hybrid", None)
        if block_ssm is None and video_ssm is None:
            return x

        frames, _, _ = _memory_grid_size(memory_context)
        if block_ssm is not None:
            x = block_ssm(x, frames=frames)
        if video_ssm is not None:
            x = video_ssm(x, frames=frames)
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        memory_context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Wan residual block plus Echo action / SSM slots on ``B (F·HW) C``."""
        if memory_context is None:
            raise ValueError("EchoWanAttentionBlock requires memory_context")
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        modulation = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            modulation = tuple(value.squeeze(2) for value in modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        original_x = x
        x = self._inject_actions(x, memory_context)
        x = self._action_attention(x, original_x, memory_context)
        self_attention_kwargs = {
            key: kwargs.pop(key)
            for key in _SELF_ATTENTION_INTERNAL_KWARGS
            if key in kwargs
        }
        kwargs.pop("_worldfoundry_inplace_residual", None)
        y = self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            freqs,
            **self_attention_kwargs,
        )
        x = self.gate(x, gate_msa, y)
        x = self._apply_temporal_memory(x, memory_context)
        x = x + self.cross_attn(self.norm3(x), context, **kwargs)
        y = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.gate(x, gate_mlp, y)
        return x


def _memory_grid_size(memory_context: Mapping[str, Any]) -> tuple[int, int, int]:
    """Read the declared ``(frames, height, width)`` latent grid from memory_context."""
    values = tuple(int(value) for value in memory_context.get("grid_size", ()))
    if len(values) != 3 or min(values) <= 0:
        raise ValueError("Echo memory_context requires a positive (frames, height, width) grid_size")
    return values


class EchoWanMemoryAdapter(nn.Module):
    """Model-side spatial memory adapter selected by an immutable recipe."""

    def __init__(
        self,
        *,
        dim: int,
        recipe: EchoMemoryRecipe,
        architecture: EchoCheckpointArchitecture,
    ) -> None:
        super().__init__()
        self.recipe = recipe
        self.spatial_memory_module: SpatialGridMemory | None = None
        self.spatial_memory_readout_module: SpatialCrossAttentionReadout | None = None
        if recipe.mechanism is EchoMemoryMechanism.SPATIAL_GRID:
            if architecture.spatial_grid_shape is None:
                raise ValueError("spatial recipe has no checkpoint grid shape")
            cells, tokens = architecture.spatial_grid_shape
            grid_size = int(round(math.sqrt(cells)))
            self.spatial_memory_module = SpatialGridMemory(
                dim,
                grid_size=grid_size,
                num_tokens=tokens,
            )
            if recipe.spatial_injection is SpatialInjection.CROSS_ATTENTION_READOUT:
                self.spatial_memory_readout_module = SpatialCrossAttentionReadout(dim)

    def prepare_inputs(
        self,
        *,
        x: torch.Tensor,
        context: torch.Tensor,
        grid_size: tuple[int, int, int],
        memory_context: Mapping[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any] | None]:
        """Optionally pool context frames into spatial memory and inject into text/video."""
        if memory_context is None:
            raise ValueError("Echo recipes require memory_context")
        frames, height, width = (int(value) for value in grid_size)
        enriched_context = dict(memory_context)
        enriched_context["grid_size"] = (frames, height, width)
        if self.spatial_memory_module is None:
            return x, context, enriched_context
        context_frames = int(memory_context.get("num_context_frames", 0))
        if context_frames <= 0:
            raise ValueError("spatial Echo recipe requires positive num_context_frames")

        if context_frames >= frames:
            raise ValueError(f"context frames {context_frames} must be smaller than total latent frames {frames}")
        spatial = height * width
        target_length = (frames - context_frames) * spatial
        context_tokens = x[:, target_length:]
        memory_tokens = self.spatial_memory_module(
            context_tokens,
            frames=context_frames,
            height=height,
            width=width,
        )
        if self.spatial_memory_readout_module is not None:
            target = self.spatial_memory_readout_module(x[:, :target_length], memory_tokens)
            x = torch.cat([target, x[:, target_length:]], dim=1)
        if self.recipe.spatial_injection is SpatialInjection.CONCAT_TEXT:
            context = torch.cat([context, memory_tokens], dim=1)
        enriched_context["spatial_memory_tokens"] = memory_tokens
        return x, context, enriched_context


__all__ = [
    "BlockWiseStateSpaceMemory",
    "CameraPoseEmbedding",
    "EchoTemporalSelfAttention",
    "EchoWanAttentionBlock",
    "EchoWanMemoryAdapter",
    "SpatialCrossAttentionReadout",
    "SpatialGridMemory",
    "VideoSSMHybridMemory",
]
