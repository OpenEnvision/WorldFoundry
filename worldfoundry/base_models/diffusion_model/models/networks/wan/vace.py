"""Wan 2.1 VACE (Video All-in-one Creation and Editing) on the native graph.

VACE adds a parallel condition transformer that consumes extra video
latents (masks, references, edits) and injects residual hints into a
subset of the main DiT blocks.  This file is the packed-tensor native
implementation used by WorldFoundry recipes.

Control flow
------------
1. Patchify noisy latents with the shared Wan stem.
2. Patchify ``vace_context`` with ``vace_patch_embedding``.
3. Run ``vace_blocks`` (must include layer 0).  Block 0 mixes the first
   condition tokens with the main hidden states; later blocks refine the
   condition stream and emit an ``after_proj`` hint.
4. Run the main ``blocks``.  Each :class:`BaseWanAttentionBlock` that
   maps to a VACE layer adds ``hint * vace_context_scale``.

No action, camera, causal, linear-attention, or TeaCache hooks live here.
The DiffSynth-shaped factory that stacks hints as a tensor is
:mod:`.vace_core`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange

from worldfoundry.core.nn import sinusoidal_embedding_1d

from .model import DiTBlock, WanModel


class VaceWanAttentionBlock(DiTBlock):
    """Condition branch used to produce residual hints for selected Wan blocks."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        block_id: int = 0,
    ) -> None:
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps)
        self.block_id = int(block_id)
        if self.block_id == 0:
            # First VACE layer is the only one allowed to see the main stream.
            self.before_proj = torch.nn.Linear(dim, dim)
        self.after_proj = torch.nn.Linear(dim, dim)

    def forward(
        self,
        condition: torch.Tensor,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep_modulation: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.block_id == 0:
            condition = self.before_proj(condition) + hidden_states
        condition = super().forward(
            condition,
            context,
            timestep_modulation,
            frequencies,
        )
        return condition, self.after_proj(condition)


class BaseWanAttentionBlock(DiTBlock):
    """Canonical Wan block with an optional VACE residual injection point."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        hint_index: int | None = None,
    ) -> None:
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps)
        self.hint_index = hint_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        timestep_modulation: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        hints: Sequence[torch.Tensor],
        context_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the Wan block, then add the mapped VACE hint when this layer is selected."""
        hidden_states = super().forward(
            hidden_states,
            context,
            timestep_modulation,
            frequencies,
        )
        if self.hint_index is not None:
            hidden_states = hidden_states + hints[self.hint_index] * float(context_scale)
        return hidden_states


class VaceWanModel(WanModel):
    """Wan2.1 transformer with VACE conditioning integrated into the shared graph.

    The parameter names intentionally match the official ``VaceWanModel``
    checkpoint. Loading and placement remain owned by the framework loader.
    """

    def __init__(
        self,
        vace_layers: Sequence[int] | None = None,
        vace_in_dim: int | None = None,
        model_type: str = "vace",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        del model_type, text_len, window_size, qk_norm, cross_attn_norm
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=False,
        )
        selected_layers = tuple(range(0, num_layers, 2)) if vace_layers is None else tuple(vace_layers)
        if not selected_layers or selected_layers[0] != 0:
            raise ValueError("VACE layers must be non-empty and start at block zero")
        if len(set(selected_layers)) != len(selected_layers):
            raise ValueError("VACE layers must be unique")
        if min(selected_layers) < 0 or max(selected_layers) >= num_layers:
            raise ValueError("VACE layer index is outside the Wan transformer")

        self.vace_layers = selected_layers
        self.vace_in_dim = int(vace_in_dim if vace_in_dim is not None else in_dim)
        self.vace_layers_mapping = {layer: index for index, layer in enumerate(selected_layers)}
        self.blocks = torch.nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    False,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    hint_index=self.vace_layers_mapping.get(index),
                )
                for index in range(num_layers)
            ]
        )
        self.vace_blocks = torch.nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    False,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    block_id=layer,
                )
                for layer in selected_layers
            ]
        )
        self.vace_patch_embedding = torch.nn.Conv3d(
            self.vace_in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def _frequencies(
        self,
        frames: int,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            torch.cat(
                (
                    self.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                    self.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                    self.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
                ),
                dim=-1,
            )
            .reshape(frames * height * width, 1, -1)
            .to(device)
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        vace_context: torch.Tensor,
        vace_context_scale: float = 1.0,
        **_: object,
    ) -> torch.Tensor:
        """Denoise ``x`` while injecting VACE hints computed from ``vace_context``."""
        if x.ndim != 5:
            raise ValueError(f"Wan VACE latents must be BCTHW, got {tuple(x.shape)}")
        if vace_context.ndim != 5:
            raise ValueError(f"Wan VACE context must be BCTHW, got {tuple(vace_context.shape)}")
        if vace_context.shape[0] != x.shape[0] or vace_context.shape[1] != self.vace_in_dim:
            raise ValueError(
                "Wan VACE context batch/channels do not match the network: "
                f"{tuple(vace_context.shape)}"
            )
        timestep = timestep.reshape(-1)
        if timestep.numel() == 1 and x.shape[0] != 1:
            timestep = timestep.expand(x.shape[0])
        if timestep.numel() != x.shape[0]:
            raise ValueError("Wan VACE timestep must be scalar or have one value per sample")

        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype)
        )
        timestep_modulation = self.time_projection(time_embedding).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        hidden_states, (frames, height, width) = self.patchify(x)
        condition = self.vace_patch_embedding(vace_context)
        if condition.shape[2:] != (frames, height, width):
            raise ValueError(
                "Wan VACE context geometry must match latent patch geometry: "
                f"{tuple(condition.shape[2:])} != {(frames, height, width)}"
            )
        condition = rearrange(condition, "b c f h w -> b (f h w) c").contiguous()
        frequencies = self._frequencies(frames, height, width, device=hidden_states.device)

        sequence_parallel_state = getattr(
            self,
            "_worldfoundry_sequence_parallel",
            None,
        )
        sequence_parallel_length = hidden_states.shape[1]
        if sequence_parallel_state is not None:
            from worldfoundry.core.distributed.sequence_parallel.runtime import (
                sequence_parallel_chunk,
            )

            degree = int(sequence_parallel_state.sp_degree)
            padding = (-sequence_parallel_length) % degree
            if padding:
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
                condition = torch.nn.functional.pad(condition, (0, 0, 0, padding))
                frequencies = torch.cat(
                    [
                        frequencies,
                        torch.ones(
                            padding,
                            *frequencies.shape[1:],
                            dtype=frequencies.dtype,
                            device=frequencies.device,
                        ),
                    ],
                    dim=0,
                )
            hidden_states = sequence_parallel_chunk(hidden_states, dim=1)
            condition = sequence_parallel_chunk(condition, dim=1)

        hints: list[torch.Tensor] = []
        for block in self.vace_blocks:
            condition, hint = block(
                condition,
                hidden_states,
                context,
                timestep_modulation,
                frequencies,
            )
            hints.append(hint)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                context,
                timestep_modulation,
                frequencies,
                hints=hints,
                context_scale=vace_context_scale,
            )
        if sequence_parallel_state is not None:
            from worldfoundry.core.distributed.sequence_parallel.runtime import (
                sequence_parallel_all_gather,
            )

            hidden_states = sequence_parallel_all_gather(hidden_states, dim=1)
            hidden_states = hidden_states[:, :sequence_parallel_length]
        hidden_states = self.head(hidden_states, time_embedding)
        return self.unpatchify(hidden_states, (frames, height, width))


__all__ = ["BaseWanAttentionBlock", "VaceWanAttentionBlock", "VaceWanModel"]
