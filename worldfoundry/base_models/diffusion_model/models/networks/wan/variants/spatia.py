"""Spatia (Wan 2.2 TI2V): reference-frame RoPE plus optional VACE hints.

Spatia prepends reference video frames to the latent grid.  Those tokens
keep a shifted 3D RoPE (time taken from the end of the frequency table,
height offset by ``H``) so they do not collide with target-frame
positions.  When a DiffSynth-shaped VACE trunk is supplied, hints are
computed on the *target* tokens only and injected via
:meth:`WanModel.after_transformer_block`.

No action, camera, causal, or linear-attention logic lives here.
"""

from dataclasses import dataclass
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.vace_core import (
    build_vace_wan_classes,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    DiTBlock,
    WanModel,
)
from worldfoundry.core.model_loading import hash_state_dict_keys

VaceWanAttentionBlock, VaceWanModel, VaceWanModelDictConverter = build_vace_wan_classes(
    DiTBlock,
    hash_state_dict_keys,
    module_name=__name__,
)


@dataclass(slots=True)
class _SpatiaTokenState:
    """Per-forward bookkeeping for reference-token count and VACE hints."""

    reference_tokens: int
    vace_hints: tuple[torch.Tensor, ...] | list[torch.Tensor] | None
    vace: torch.nn.Module | None
    vace_scale: float


class SpatiaWanModel(WanModel):
    """Wan2.2 TI2V with Spatia reference RoPE and VACE residual injection."""

    def prepare_token_sequence(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        t: torch.Tensor,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ):
        """Shift RoPE for reference frames and optionally run the VACE trunk."""
        f_all, h, w = grid_size
        num_ref_frames = int(kwargs.get("num_ref_frames", 0) or 0)
        if not 0 <= num_ref_frames < f_all:
            raise ValueError(f"Spatia num_ref_frames must be in [0, {f_all}); got {num_ref_frames}")
        reference_tokens = num_ref_frames * h * w
        if num_ref_frames:
            target_f = f_all - num_ref_frames
            reference_freqs = torch.cat(
                [
                    self.freqs[0][-num_ref_frames:]
                    .view(num_ref_frames, 1, 1, -1)
                    .expand(num_ref_frames, h, w, -1),
                    self.freqs[1][h : 2 * h]
                    .view(1, h, 1, -1)
                    .expand(num_ref_frames, h, w, -1),
                    self.freqs[2][:w]
                    .view(1, 1, w, -1)
                    .expand(num_ref_frames, h, w, -1),
                ],
                dim=-1,
            ).reshape(reference_tokens, 1, -1).to(x.device)
            freqs = torch.cat(
                [reference_freqs, self.rotary_frequencies((target_f, h, w), device=x.device)],
                dim=0,
            )

        vace = kwargs.get("vace")
        vace_context = kwargs.get("vace_context")
        hints = None
        if vace is not None and vace_context is not None:
            context = kwargs["context"]
            target_t_mod = t_mod[:, reference_tokens:] if t_mod.ndim == 4 else t_mod
            hints = vace(
                x[:, reference_tokens:],
                vace_context,
                context,
                target_t_mod,
                freqs[reference_tokens:],
            )
        state = _SpatiaTokenState(
            reference_tokens=reference_tokens,
            vace_hints=hints,
            vace=vace,
            vace_scale=float(kwargs.get("vace_scale", 1.0)),
        )
        return x, freqs, t_mod, t, state

    def after_transformer_block(
        self,
        x: torch.Tensor,
        block_id: int,
        token_state: _SpatiaTokenState,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Add the mapped VACE hint onto target tokens; leave reference tokens intact."""
        del kwargs
        if token_state.vace_hints is None or token_state.vace is None:
            return x
        mapping = token_state.vace.vace_layers_mapping
        if block_id not in mapping:
            return x
        start = token_state.reference_tokens
        target = x[:, start:] + token_state.vace_hints[mapping[block_id]] * token_state.vace_scale
        return torch.cat([x[:, :start], target], dim=1) if start else target

__all__ = [
    "SpatiaWanModel",
    "VaceWanAttentionBlock",
    "VaceWanModel",
    "VaceWanModelDictConverter",
]
