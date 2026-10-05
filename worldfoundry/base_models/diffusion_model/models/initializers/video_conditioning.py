"""Shared image / video pixel prep for Video2World-style initializers.

Normalizes one still or the tail of one video to ``BCTHW`` in ``[-1, 1]``.
The returned integer is the number of *physical* conditioning frames; the
remaining slots repeat the last frame by default. Cosmos Predict2.5 uses
zero padding for still images to match NVIDIA's image input preparation.
The model-owned latent mask (built by the initializer) decides which encoded
frames stay frozen during denoising.

Used by Cosmos Predict2 / Predict1 and similar encoded initializers.
Accepts ``request.inputs[\"image\"]`` / ``images`` or ``video`` / ``videos``,
not both.  Batch size must be 1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from worldfoundry.core.media.codecs.video import coerce_video_frames
from worldfoundry.core.utils import load_pil_image

from ...contracts import DiffusionRequest


def prepare_video_conditioning_pixels(
    request: DiffusionRequest,
    *,
    device: torch.device,
    dtype: torch.dtype,
    temporal_compression: int,
    allowed_latent_frames: tuple[int, ...] = (1, 2),
    owner: str = "video diffusion model",
    zero_pad_image: bool = False,
) -> tuple[torch.Tensor | None, int]:
    """Normalize one image or the tail of one video to ``BCTHW`` pixels.

    The returned integer is the number of physical conditioning frames.  The
    remaining frames repeat the last observation by default.  Callers that
    follow an image-to-video reference implementation can instead zero-pad
    still images; video input always repeats its last observed frame.
    """

    image = request.inputs.get("image", request.inputs.get("images"))
    video = request.inputs.get("video", request.inputs.get("videos"))
    if image is not None and video is not None:
        raise ValueError(f"{owner} accepts either image or video conditioning, not both")
    if image is None and video is None:
        return None, 0
    if request.batch_size != 1:
        raise ValueError(f"{owner} media conditioning currently requires batch size 1")

    if image is not None:
        array = np.asarray(load_pil_image(image), dtype=np.float32).copy()
        frames = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        physical_frames = 1
    else:
        latent_frames = int(request.inputs.get("num_latent_conditional_frames", 1))
        if latent_frames not in allowed_latent_frames:
            raise ValueError(
                "num_latent_conditional_frames must be one of "
                f"{list(allowed_latent_frames)}, got {latent_frames}"
            )
        physical_frames = temporal_compression * (latent_frames - 1) + 1
        frames = torch.from_numpy(
            coerce_video_frames(video, max_frames=physical_frames, from_end=True)
        ).permute(0, 3, 1, 2)
        if len(frames) < physical_frames:
            raise ValueError(
                f"{owner} video conditioning requires at least {physical_frames} input frames"
            )
        frames = frames[-physical_frames:]

    frames = functional.interpolate(
        frames.float(),
        (request.height, request.width),
        mode="bilinear",
        align_corners=False,
    )
    if len(frames) < request.num_frames:
        padding_frames = request.num_frames - len(frames)
        padding = (
            torch.zeros((padding_frames, *frames.shape[1:]), dtype=frames.dtype, device=frames.device)
            if image is not None and zero_pad_image
            else frames[-1:].expand(padding_frames, -1, -1, -1)
        )
        frames = torch.cat(
            (frames, padding)
        )
    frames = frames[: request.num_frames]
    pixels = frames.permute(1, 0, 2, 3).unsqueeze(0).div(127.5).sub(1.0)
    return pixels.to(device=device, dtype=dtype), physical_frames


__all__ = ["prepare_video_conditioning_pixels"]
