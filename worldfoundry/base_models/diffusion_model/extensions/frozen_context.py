"""Generic frozen-context lifecycle for suffix-conditioned diffusion.

Installed by the ``frozen-context`` runner strategy.  A clean context suffix
from ``DiffusionRequest.inputs["frozen_context_latents"]`` is written back
onto the tail of the latent sequence after every scheduler update so the
denoiser can attend to a known past without the solver drifting it.

The suffix must be shorter than the full latent time axis.  Shape changes
during the run are treated as a contract violation.
"""

from __future__ import annotations

import torch

from .base import DiffusionExtension, DiffusionRunContext


class FrozenContextSuffixExtension(DiffusionExtension):
    """Restore a clean context suffix after every scheduler update.

    ``on_run_start`` caches a ``[B, C, T, H, W]`` tensor (unsqueezing 4-D
    input).  ``after_step`` copies that suffix onto ``latents[:, :, -T:]``.

    Raises:
        TypeError: ``frozen_context_latents`` is missing or not a tensor.
        ValueError: Rank / empty time axis; channel or spatial mismatch;
            suffix not shorter than the full sequence.
        RuntimeError: ``after_step`` runs before ``on_run_start``.
    """

    extension_id = "frozen-context-suffix"

    def on_run_start(self, context: DiffusionRunContext) -> None:
        values = context.request.inputs.get("frozen_context_latents")
        if not isinstance(values, torch.Tensor):
            raise TypeError("frozen-context execution requires frozen_context_latents")
        if values.ndim == 4:
            values = values.unsqueeze(0)
        if values.ndim != 5 or int(values.shape[2]) <= 0:
            raise ValueError("frozen_context_latents must be a non-empty [B,C,T,H,W] tensor")
        context.state[self.extension_id] = values

    def after_step(
        self,
        context: DiffusionRunContext,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        clean = context.state[self.extension_id]
        if not isinstance(clean, torch.Tensor):
            raise RuntimeError("frozen context state was not initialized")
        clean = clean.to(device=latents.device, dtype=latents.dtype)
        if clean.shape[0] == 1 and latents.shape[0] > 1:
            clean = clean.expand(latents.shape[0], -1, -1, -1, -1)
        if clean.shape[:2] != latents.shape[:2] or clean.shape[-2:] != latents.shape[-2:]:
            raise ValueError("frozen context shape changed during denoising")
        if int(clean.shape[2]) >= int(latents.shape[2]):
            raise ValueError("frozen context must be shorter than the full latent sequence")
        result = latents.clone()
        result[:, :, -int(clean.shape[2]) :] = clean
        return result


__all__ = ["FrozenContextSuffixExtension"]
