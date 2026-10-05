"""Wan 2.1 / 2.2 spatial control adapter (camera / ControlNet residual).

``SimpleAdapter`` is the official camera-control stem used when a Wan
checkpoint publishes an overlay ``control_adapter`` that is added onto
patch embeddings.  It pixel-unshuffles each frame, projects to DiT width,
and applies a small residual stack.  ``process_camera_coordinates`` builds
Plücker features from a planar camera trajectory so callers do not need
to reimplement that geometry.

This module does not own VACE, action, or causal attention; those live in
:mod:`.vace` and :mod:`.variants`.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two-convolution residual used after the adapter's spatial projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` plus a 3x3 residual, keeping ``B C H W`` layout."""
        return values + self.conv2(self.relu(self.conv1(values)))


class SimpleAdapter(nn.Module):
    """Spatial control adapter used by Wan camera-conditioned checkpoints.

    Input is ``B C F H W`` camera / control latents.  Each frame is
    independently unshuffled and convolved so the output grid matches the
    DiT patch embedding and can be added in :meth:`WanModel.patchify`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        downscale_factor: int = 8,
        num_residual_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(
            in_dim * downscale_factor * downscale_factor,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(out_dim) for _ in range(num_residual_blocks)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Project ``B C F H W`` control latents onto the DiT patch grid."""
        batch, channels, frames, height, width = values.shape
        values = values.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            channels,
            height,
            width,
        )
        values = self.residual_blocks(self.conv(self.pixel_unshuffle(values)))
        values = values.view(
            batch,
            frames,
            values.size(1),
            values.size(2),
            values.size(3),
        )
        return values.permute(0, 2, 1, 3, 4)

    def process_camera_coordinates(
        self,
        direction: str,
        length: int,
        height: int,
        width: int,
        speed: float = 1 / 54,
        origin=None,
    ) -> torch.Tensor:
        """Build camera-control Plücker features through shared core geometry."""

        from worldfoundry.core.geometry.trajectory import (
            generate_planar_camera_coordinates,
            wan_camera_coordinates_to_plucker,
        )

        options = {} if origin is None else {"origin": origin}
        rows = generate_planar_camera_coordinates(
            direction,
            length,
            speed=speed,
            **options,
        )
        return wan_camera_coordinates_to_plucker(
            rows,
            width=width,
            height=height,
        )


__all__ = ["ResidualBlock", "SimpleAdapter"]
