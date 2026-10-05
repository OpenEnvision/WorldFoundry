"""Small media and geometry adapters required by Matrix inference."""

from __future__ import annotations

import os

import numpy as np
import torch

from .cleanup import _rank_local_cuda_device


def _parse_query_hits(result, *, return_revgrid, return_view_change):
    """Unpack the FrustumHandler result while discarding auxiliary IDs."""
    if not isinstance(result, tuple):
        return result, None, None
    output = result[0]
    extra_index = 3  # output, candidate ids, keyframe candidate ids
    revgrid = None
    view_change = None
    if return_revgrid:
        revgrid = result[extra_index]
        extra_index += 1
    if return_view_change:
        view_change = result[extra_index]
    return output, revgrid, view_change


def _prope_camera_kwargs(camera_data):
    return {
        "clean_latent_indices_prope_intrinsic": camera_data["clean_latent_indices_prope_intrinsic"],
        "clean_latent_indices_prope_extrinsic": camera_data["clean_latent_indices_prope_extrinsic"],
        "noisy_latent_indices_prope_intrinsic": camera_data["noisy_latent_indices_prope_intrinsic"],
        "noisy_latent_indices_prope_extrinsic": camera_data["noisy_latent_indices_prope_extrinsic"],
    }


def _get_frustum_handler_cls():
    from ..frustum.frustum_handler import FrustumHandler

    return FrustumHandler


_DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def _init_da3_depth_estimator(device=None):
    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
        DepthAnything3,
    )

    model_ref = os.environ.get("DA3_MODEL_PATH") or os.environ.get("DA3_MODEL_ID") or _DA3_MODEL_ID
    if device is not None and str(device) == "cuda":
        device = _rank_local_cuda_device() or device
    print(f"[inference] loading depth estimator: {model_ref}")
    estimator = DepthAnything3.from_pretrained(model_ref).eval()
    if device is not None:
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.set_device(device)
        estimator = estimator.to(device)
    return estimator


def _decode_latents_to_numpy_frames(pipe, latents, device, tiled=False):
    """Decode BCTHW latents into uint8 THWC frames."""
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    pipe.load_models_to_device(["vae"])
    decode_kwargs = {"device": device}
    if tiled:
        decode_kwargs.update(
            {
                "tiled": True,
                "tile_size": (30, 52),
                "tile_stride": (15, 26),
            }
        )
    else:
        decode_kwargs["tiled"] = False
    decoded = pipe.vae.decode(
        latents.to(
            device=device,
            dtype=getattr(pipe, "vae_dtype", pipe.torch_dtype),
        ),
        **decode_kwargs,
    )
    pipe.load_models_to_device([])
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if decoded.ndim == 5:
        decoded = decoded[0]
    decoded = decoded.float().permute(1, 2, 3, 0)
    return ((decoded + 1.0) * 127.5).clamp(0, 255).to("cpu").numpy().astype(np.uint8)


def _encode_frames_per_frame(pipe, frames, device, tiled=False):
    """Encode each RGB frame into one independently addressable memory latent."""
    frame_array = np.asarray(frames)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (N,H,W,3), got {tuple(frame_array.shape)}")
    pipe.load_models_to_device(["vae"])
    codec_dtype = getattr(pipe, "vae_dtype", pipe.torch_dtype)
    videos = []
    for frame in frame_array:
        tensor = torch.from_numpy(frame).to(device=device, dtype=codec_dtype)
        videos.append(tensor.permute(2, 0, 1).unsqueeze(1) / 127.5 - 1.0)
    encoded = pipe.vae.encode(videos, device=device, tiled=tiled)
    pipe.load_models_to_device([])
    if encoded.ndim != 5 or encoded.shape[2] != 1:
        raise RuntimeError(f"Matrix memory VAE encode must return (N,C,1,H,W), got {tuple(encoded.shape)}")
    return encoded.squeeze(2).permute(1, 0, 2, 3).unsqueeze(0).contiguous().detach().to("cpu", dtype=torch.float32)
