from __future__ import annotations

import torch
from torch import Tensor

from worldfoundry.base_models.diffusion_model.kandinsky6_types import LatentBundle


@torch.no_grad()
def postprocess_video(
    bundle: LatentBundle,
    vae,
    bs: int,
) -> Tensor:
    """Decode video latents → (bs, 3, T, H, W) uint8 in [0, 255].

    Input layout: packed ``(sum_T, H_lat, W_lat, C)`` or batched
    ``(bs, T, H_lat, W_lat, C)``.
    """
    video = bundle.video
    assert video is not None

    if video.ndim == 4:
        frames = video.reshape(bs, -1, video.shape[-3], video.shape[-2], video.shape[-1])
    elif video.ndim == 5:
        if video.shape[0] != bs:
            raise ValueError(f"batched video has batch size {video.shape[0]}, expected {bs}")
        frames = video
    else:
        raise ValueError(f"video must have rank 4 or 5, got {video.ndim}")
    # (bs, T, H, W, C) → (bs, C, T, H, W) for VAE input
    frames = (frames / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
    # Hunyuan VAE loads as fp16; DiT latents are bf16 — match weight dtype.
    vae_dtype = next(vae.parameters()).dtype
    frames = vae.decode(frames.to(dtype=vae_dtype)).sample

    return ((frames.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
