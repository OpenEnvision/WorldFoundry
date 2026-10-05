"""Framework-owned latent projection for partially frozen diffusion inputs.

Installed by the ``masked-latent`` runner strategy.  After every scheduler
``step``, clean regions selected by ``denoise_mask`` are restored from
``clean_latents`` in ``Conditioning.shared``.  The denoiser still predicts a
full tensor; this hook is the only place the freeze is applied.

Requires tensor ``clean_latents`` and ``denoise_mask``.  Missing or
non-broadcastable conditions fail the run rather than silently skip.
"""

from __future__ import annotations

import torch

from .base import DiffusionExtension, DiffusionRunContext

_WAN_DENOISE_MASK_IS_ALL_ONES = "_worldfoundry_wan_denoise_mask_is_all_ones"


class FrozenLatentMaskExtension(DiffusionExtension):
    """Restore clean latent regions selected by a continuous denoise mask.

    ``after_step`` computes ``latents * mask + clean * (1 - mask)``.

    Raises:
        TypeError: ``clean_latents`` or ``denoise_mask`` is missing or not a
            tensor.
        ValueError: The tensors do not broadcast to the generated latent
            shape (wrapped from the underlying ``RuntimeError``).
    """

    extension_id = "frozen-latent-mask"

    def after_step(
        self,
        context: DiffusionRunContext,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        all_ones_semantics = context.conditioning.shared.get(
            _WAN_DENOISE_MASK_IS_ALL_ONES,
            False,
        )
        if not isinstance(all_ones_semantics, bool):
            raise TypeError(
                f"{_WAN_DENOISE_MASK_IS_ALL_ONES} must be a bool when provided"
            )
        if all_ones_semantics:
            # Pure TI2V text-to-video has no frozen latent region.  The
            # initializer supplies this CPU-known proof so every scheduler
            # step can skip two full activation pointwise operations without
            # synchronizing on ``mask.all()``.
            return latents
        clean = context.conditioning.shared.get("clean_latents")
        mask = context.conditioning.shared.get("denoise_mask")
        if not isinstance(clean, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError(
                "masked-latent execution requires tensor clean_latents and denoise_mask conditions"
            )
        clean = clean.to(device=latents.device, dtype=latents.dtype)
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        try:
            return latents * mask + clean * (1.0 - mask)
        except RuntimeError as error:
            raise ValueError(
                "clean_latents and denoise_mask must broadcast to the generated latent shape"
            ) from error


__all__ = ["FrozenLatentMaskExtension"]
