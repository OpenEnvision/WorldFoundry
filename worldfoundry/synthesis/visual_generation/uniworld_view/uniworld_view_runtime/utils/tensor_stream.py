# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor


def _preprocess_frames_for_stream3r(frames_np: np.ndarray, mode: str = "crop") -> torch.Tensor:
    """
    Mirror STream3R's load_and_preprocess_images but for in-memory video frames.
    - Width resized to 518 (divisible by 14). Height resized proportionally and rounded to /14.
    - For crop mode: if height > 518 after resize, center crop to 518.
    - If shapes still differ across frames, pad to max(H, W) with white (1.0).

    Args:
        frames_np: np.ndarray of shape (T, H, W, 3), values in [0,1]
        mode: 'crop' or 'pad' (keep 'crop' to match default)

    Returns:
        Tensor of shape (T, 3, Hs, Ws) in [0,1]
    """
    assert frames_np.ndim == 4 and frames_np.shape[-1] == 3
    T = frames_np.shape[0]
    target_size = 518
    images: list[torch.Tensor] = []
    shapes = set()
    to_tensor = ToTensor()

    for i in range(T):
        # Convert to PIL in [0,255]
        img_np = np.clip(frames_np[i] * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np).convert("RGB")
        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # crop mode
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img_t = to_tensor(img)  # (3,H,W) in [0,1]

        # Center-crop height if necessary (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_t = img_t[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img_t.shape[1]
            w_padding = target_size - img_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_t = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

        shapes.add((img_t.shape[1], img_t.shape[2]))
        images.append(img_t)

    # If different shapes, pad to the max H and W (white background)
    if len(shapes) > 1:
        max_height = max(s[0] for s in shapes)
        max_width = max(s[1] for s in shapes)
        padded_images: list[torch.Tensor] = []
        for img_t in images:
            h_padding = max_height - img_t.shape[1]
            w_padding = max_width - img_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_t = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img_t)
        images = padded_images

    return torch.stack(images, dim=0)
