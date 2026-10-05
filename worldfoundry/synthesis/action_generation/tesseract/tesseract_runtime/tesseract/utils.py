import gc
import cv2
import numpy as np

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed


logger = get_logger(__name__)


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")


def modify_transformer(model, add_in_channels, add_out_channels=0):
    device = model.device
    dtype = model.dtype

    model.config.in_channels += add_in_channels
    old_proj = model.patch_embed.proj
    old_in_channels = old_proj.in_channels
    old_out_channels = old_proj.out_channels

    if type(old_proj) == nn.Conv2d:
        new_proj = nn.Conv2d(
            old_in_channels + add_in_channels,
            old_out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        ).to(device, dtype)
    elif type(old_proj) == nn.Linear:
        new_proj = nn.Linear(old_in_channels + add_in_channels, old_out_channels).to(device, dtype)

    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, : old_proj.weight.shape[1]].copy_(old_proj.weight)
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
    model.patch_embed.proj = new_proj

    if add_out_channels > 0:
        old_proj = model.proj_out
        patch_size = model.config.patch_size
        out_channels = model.config.out_channels
        new_proj = nn.Linear(
            old_proj.in_features, old_proj.out_features + add_out_channels * patch_size * patch_size
        ).to(device, dtype)
        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[: old_proj.weight.shape[0]].copy_(old_proj.weight)
            if old_proj.bias is not None:
                new_proj.bias[: old_proj.bias.shape[0]].copy_(old_proj.bias)
        model.proj_out = new_proj


def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
    target_height, target_width = target_size
    original_height, original_width = frames[0].shape[:2]
    if original_height == target_height and original_width == target_width:
        return [frame for frame in frames]

    # ==== interpolation method ====
    if interpolation == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
        logger.warning(f"Unsupported interpolation: {interpolation}. Using bilinear instead.")

    processed_frames = []
    for frame in frames:
        original_height, original_width = frame.shape[:2]
        aspect_ratio_target = target_width / target_height
        aspect_ratio_original = original_width / original_height

        if aspect_ratio_original > aspect_ratio_target:
            new_width = int(aspect_ratio_target * original_height)
            start_x = (original_width - new_width) // 2
            cropped_frame = frame[:, start_x : start_x + new_width]
        else:
            new_height = int(original_width / aspect_ratio_target)
            start_y = (original_height - new_height) // 2
            cropped_frame = frame[start_y : start_y + new_height, :]
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=interpolation)
        processed_frames.append(resized_frame)

    return processed_frames


def read_video_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"No frames available in video: {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
