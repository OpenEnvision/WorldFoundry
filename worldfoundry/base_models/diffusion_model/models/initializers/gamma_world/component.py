"""Gamma-World multi-view :class:`~...contracts.EncodedLatentInitializer`.

Requires the recipe-bound Wan :class:`~...contracts.LatentEncoder`.
``request.inputs[\"images\"]`` is one still per player, or a single
horizontally concatenated strip that
:func:`~worldfoundry.core.utils.split_horizontal_views` splits.
``initialize`` without an encoder always raises.

Returned :class:`~...contracts.LatentInitialization` carries per-view
first-frame latents and masks for the Gamma-World denoiser.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentEncoder, LatentInitialization


class GammaWorldLatentInitializer:
    """Encode one first frame per player and create a shared multi-view noise state."""

    def __init__(
        self,
        *,
        channels: int = 16,
        spatial_compression: int = 8,
        temporal_compression: int = 4,
        num_players: int = 2,
        required_num_frames: int | None = None,
    ) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)
        self.num_players = int(num_players)
        self.required_num_frames = required_num_frames
        if min(self.channels, self.spatial_compression, self.temporal_compression, self.num_players) <= 0:
            raise ValueError("Gamma latent dimensions and num_players must be positive")

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Rejected: Gamma-World must use :meth:`initialize_with_encoder`."""
        del request, generator, device, dtype
        raise RuntimeError("Gamma-World requires its shared Wan latent_encoder binding")

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, np.ndarray, torch.Tensor))
            and not hasattr(value, "mode")
        )

    def _images(self, request: DiffusionRequest) -> list[object]:
        value = request.inputs.get("images", request.inputs.get("image"))
        if value is None:
            raise ValueError("Gamma-World requires one initial image per player")
        images = list(value) if self._is_sequence(value) else [value]
        requested_players = int(request.inputs.get("n_players", self.num_players))
        if len(images) == 1 and requested_players > 1:
            from worldfoundry.core.utils import split_horizontal_views

            images = list(
                split_horizontal_views(
                    images[0],
                    num_views=requested_players,
                    target_size=(request.width, request.height),
                )
            )
        if len(images) != requested_players:
            raise ValueError(
                f"Gamma-World received {len(images)} images for n_players={requested_players}"
            )
        if requested_players != self.num_players:
            raise ValueError(
                f"the released Gamma checkpoint supports {self.num_players} players, "
                f"got {requested_players}"
            )
        return images

    @torch.no_grad()
    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        if request.batch_size != 1:
            raise ValueError("Gamma-World currently supports one multi-player rollout per request")
        if self.required_num_frames is not None and request.num_frames != self.required_num_frames:
            raise ValueError(
                "Gamma-World released checkpoints require "
                f"num_frames={self.required_num_frames} per view "
                f"(official state_t=48 latent frames; received {request.num_frames!r})"
            )
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError(
                f"Gamma height and width must be divisible by {self.spatial_compression}"
            )
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError("Gamma num_frames must satisfy (num_frames - 1) % 4 == 0")

        from worldfoundry.core import load_pil_image

        images = self._images(request)
        first_frame_latents = []
        for image_value in images:
            image = load_pil_image(image_value, first_sequence_item=False)
            array = np.asarray(image, dtype=np.float32)
            pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            pixels = torch.nn.functional.interpolate(
                pixels.unsqueeze(0),
                size=(request.height, request.width),
                mode="bicubic",
                align_corners=False,
            )
            pixels = pixels.div_(127.5).sub_(1.0).unsqueeze(2).to(device=device, dtype=dtype)
            encoded = latent_encoder.encode(pixels).to(device=device, dtype=dtype)
            if encoded.shape[2] != 1:
                encoded = encoded[:, :, :1]
            first_frame_latents.append(encoded)

        latent_frames = (request.num_frames - 1) // self.temporal_compression + 1
        latent_height = request.height // self.spatial_compression
        latent_width = request.width // self.spatial_compression
        noise = torch.randn(
            request.batch_size,
            self.channels,
            self.num_players * latent_frames,
            latent_height,
            latent_width,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        clean = torch.zeros_like(noise, dtype=dtype)
        mask = torch.zeros(
            request.batch_size,
            1,
            self.num_players * latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )
        for player, encoded in enumerate(first_frame_latents):
            if encoded.shape[:2] != clean.shape[:2] or encoded.shape[-2:] != clean.shape[-2:]:
                raise ValueError(
                    "Gamma encoded image geometry does not match the denoiser: "
                    f"{tuple(encoded.shape)} vs {tuple(clean.shape)}"
                )
            offset = player * latent_frames
            clean[:, :, offset : offset + 1] = encoded
            mask[:, :, offset : offset + 1] = 1

        view_indices = torch.arange(self.num_players, device=device, dtype=torch.long)
        view_indices = view_indices.repeat_interleave(latent_frames).unsqueeze(0)
        return LatentInitialization(
            latents=noise,
            conditioning={
                "gt_frames": clean,
                "condition_video_input_mask_B_C_T_H_W": mask,
                "use_video_condition": True,
                "view_indices_B_T": view_indices,
                "padding_mask": torch.zeros(
                    request.batch_size,
                    1,
                    request.height,
                    request.width,
                    device=device,
                    dtype=dtype,
                ),
                "initial_noise": noise.clone(),
                "n_views": self.num_players,
                "latent_frames_per_view": latent_frames,
            },
        )


def build_gamma_world_latent_initializer(
    context: ComponentBuildContext,
) -> GammaWorldLatentInitializer:
    options = context.component_options
    return GammaWorldLatentInitializer(
        channels=int(context.recipe_options.get("latent_channels", 16)),
        spatial_compression=int(context.recipe_options.get("spatial_compression", 8)),
        temporal_compression=int(context.recipe_options.get("temporal_compression", 4)),
        num_players=int(options.get("num_players", 2)),
        required_num_frames=context.recipe_options.get("required_num_frames"),
    )


__all__ = ["GammaWorldLatentInitializer", "build_gamma_world_latent_initializer"]
