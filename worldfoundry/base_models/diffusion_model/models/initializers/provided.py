"""Framework-neutral :class:`~...contracts.LatentInitializer` from caller tensors.

Implements the initializer Protocol by reading ``request.inputs[input_key]``
(default ``\"latents\"``) and moving it onto the run device/dtype.  Used when
a recipe or evaluation harness already owns the starting latent (img2img,
 inversion, or a pre-encoded trajectory).  Does not load a VAE.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


class ProvidedLatentInitializer:
    """Use a normalized request input as the initial diffusion state."""

    def __init__(self, input_key: str = "latents") -> None:
        self.input_key = str(input_key)
        if not self.input_key:
            raise ValueError("provided latent input key cannot be empty")

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``request.inputs[input_key]`` as the initial latent batch."""
        del generator
        value = request.inputs.get(self.input_key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"request.inputs[{self.input_key!r}] must be a tensor")
        latents = value.to(device=device, dtype=dtype)
        if latents.ndim < 4:
            raise ValueError(f"provided latents must have at least four dimensions, got {tuple(latents.shape)}")
        if int(latents.shape[0]) != request.batch_size:
            raise ValueError(
                f"provided latent batch must match prompt batch: {int(latents.shape[0])} != {request.batch_size}"
            )
        return latents


def build_provided_latent_initializer(context: ComponentBuildContext) -> ProvidedLatentInitializer:
    """Build from ``component_options[\"input_key\"]`` (default ``latents``)."""
    return ProvidedLatentInitializer(
        input_key=str(context.component_options.get("input_key", "latents")),
    )


__all__ = ["ProvidedLatentInitializer", "build_provided_latent_initializer"]
