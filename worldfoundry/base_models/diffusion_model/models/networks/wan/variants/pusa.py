"""PUSA (Wan 2.2) per-frame timestep on the native packed Wan graph.

PUSA checkpoints keep the standard Wan parameter graph.  The only added
behavior is expanding one diffusion timestep per latent frame across the
spatial patch tokens so AdaLN can vary along time.  Attention, VACE,
action, camera, causal, and linear-attention paths are unchanged.
"""

from __future__ import annotations

import torch

from ..model import WanModel


class PusaWanModel(WanModel):
    """Wan2.2 DiT accepting one diffusion timestep per latent frame.

    PUSA checkpoints keep the standard Wan parameter graph.  The only graph
    semantic is expanding frame timesteps across spatial patch tokens, so the
    implementation stays on the canonical Wan attention and block stack.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Force the host Wan graph to accept ``[B, L]`` timesteps."""
        kwargs["per_token_timestep"] = True
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Expand per-frame timesteps to one value per patch token, then call Wan."""
        if timestep.ndim == 2:
            temporal_tokens = (x.shape[2] - self.patch_size[0]) // self.patch_size[0] + 1
            spatial_tokens = (
                ((x.shape[3] - self.patch_size[1]) // self.patch_size[1] + 1)
                * ((x.shape[4] - self.patch_size[2]) // self.patch_size[2] + 1)
            )
            total_tokens = temporal_tokens * spatial_tokens
            if timestep.shape[1] == temporal_tokens:
                timestep = timestep.repeat_interleave(spatial_tokens, dim=1)
            elif timestep.shape[1] != total_tokens:
                raise ValueError(
                    "PUSA timestep must contain one value per latent frame "
                    f"({temporal_tokens}) or patch token ({total_tokens}); got {timestep.shape[1]}"
                )
        return super().forward(x=x, timestep=timestep, context=context, **kwargs)


__all__ = ["PusaWanModel"]
