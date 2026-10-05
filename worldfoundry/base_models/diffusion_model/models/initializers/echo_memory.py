"""Echo-Memory :class:`~...contracts.LatentInitializer`.

Builds a concatenated latent ``[target_noise | frozen_context]`` along
time.  Target frames follow Wan 4n+1 geometry
(``T_lat = (num_frames-1)//4 + 1``).  The suffix comes from
``request.inputs[\"frozen_context_latents\"]`` (``[C,T,H,W]`` or
``[B,C,T,H,W]``), already encoded by the caller.  The denoiser reads
``num_context_frames`` from conditioning separately; this initializer
only owns the starting tensor.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest


def _context_latents(request: DiffusionRequest, *, device, dtype) -> torch.Tensor:
    values = request.inputs.get("frozen_context_latents")
    if not isinstance(values, torch.Tensor):
        raise TypeError("Echo-Memory requires tensor request.inputs['frozen_context_latents']")
    if values.ndim == 4:
        values = values.unsqueeze(0)
    if values.ndim != 5:
        raise ValueError("frozen_context_latents must be [C,T,H,W] or [B,C,T,H,W]")
    if values.shape[0] == 1 and request.batch_size > 1:
        values = values.expand(request.batch_size, -1, -1, -1, -1)
    if values.shape[0] != request.batch_size:
        raise ValueError("frozen context batch must match the prompt batch")
    return values.to(device=device, dtype=dtype)


class EchoMemoryLatentInitializer:
    """Create target noise followed by a clean, frozen context suffix."""

    def __init__(
        self,
        *,
        channels: int = 16,
        spatial_compression: int = 8,
        temporal_compression: int = 4,
    ) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``cat([randn_target, frozen_context], dim=2)``."""
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError("Echo-Memory height and width must match Wan latent compression")
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError("Echo-Memory num_frames must have form 4n+1")
        target_frames = (request.num_frames - 1) // self.temporal_compression + 1
        context = _context_latents(request, device=device, dtype=dtype)
        expected = (
            self.channels,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
        )
        if (int(context.shape[1]), int(context.shape[3]), int(context.shape[4])) != expected:
            raise ValueError(
                "Echo context channel/spatial shape does not match the requested video: "
                f"got {tuple(context.shape)}, expected (*,{expected[0]},T,{expected[1]},{expected[2]})"
            )
        target = torch.randn(
            request.batch_size,
            self.channels,
            target_frames,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return torch.cat([target, context], dim=2)


def build_echo_memory_latent_initializer(
    context: ComponentBuildContext,
) -> EchoMemoryLatentInitializer:
    """Build with Wan-default 16-ch / 8× / 4× compression unless overridden."""
    return EchoMemoryLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
    )


__all__ = [
    "EchoMemoryLatentInitializer",
    "build_echo_memory_latent_initializer",
]
