# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DepthAnything v3 depth/feature visualization and export helpers."""

import os
import subprocess

import cv2
import imageio
import matplotlib
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.specs import Prediction
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.logger import logger
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.pca_utils import (
    PCARGBVisualizer,
)
from worldfoundry.core.execution.parallel_execution import async_call


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """Visualize a depth map using a colormap."""
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    return img_colored_np


def vis_depth_map_tensor(
    result: torch.Tensor,
    color_map: str = "Spectral",
) -> torch.Tensor:
    """Color-map the depth map."""
    far = result.reshape(-1)[:16_000_000].float().quantile(0.99).log().to(result)
    try:
        near = result[result > 0][:16_000_000].float().quantile(0.01).log().to(result)
    except (RuntimeError, ValueError) as e:
        logger.error(f"No valid depth values found. Reason: {e}")
        near = torch.zeros_like(far)
    result = result.log()
    result = (result - near) / (far - near)
    return apply_color_map_to_image(result, color_map)


def apply_color_map(
    x: torch.Tensor,
    color_map: str = "inferno",
) -> torch.Tensor:
    cmap = matplotlib.cm.get_cmap(color_map)
    mapped = cmap(x.float().detach().clip(min=0, max=1).cpu().numpy())[..., :3]
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: torch.Tensor,
    color_map: str = "inferno",
) -> torch.Tensor:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def export_to_depth_vis(
    prediction: Prediction,
    export_dir: str,
):
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images

    os.makedirs(os.path.join(export_dir, "depth_vis"), exist_ok=True)
    for idx in range(prediction.depth.shape[0]):
        depth_vis = visualize_depth(prediction.depth[idx])
        image_vis = images_u8[idx]
        depth_vis = depth_vis.astype(np.uint8)
        image_vis = image_vis.astype(np.uint8)
        vis_image = np.concatenate([image_vis, depth_vis], axis=1)
        save_path = os.path.join(export_dir, f"depth_vis/{idx:04d}.jpg")
        imageio.imwrite(save_path, vis_image, quality=95)


@async_call
def export_to_feat_vis(
    prediction,
    export_dir,
    fps=15,
):
    """Export feature visualization with PCA."""
    out_dir = os.path.join(export_dir, "feat_vis")
    os.makedirs(out_dir, exist_ok=True)

    images = prediction.processed_images
    for k, v in prediction.aux.items():
        if not k.startswith("feat_layer_"):
            continue
        os.makedirs(os.path.join(out_dir, k), exist_ok=True)
        viz = PCARGBVisualizer(basis_mode="fixed", percentile_mode="global", clip_percent=10.0)
        viz.fit_reference(v)
        feats_vis = viz.transform_video(v)
        for idx in tqdm(range(len(feats_vis))):
            img = images[idx]
            feat_vis = (feats_vis[idx] * 255).astype(np.uint8)
            feat_vis = cv2.resize(
                feat_vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            save_path = os.path.join(out_dir, f"{k}/{idx:06d}.jpg")
            save = np.concatenate([img, feat_vis], axis=1)
            imageio.imwrite(save_path, save, quality=95)
        input_pattern = os.path.join(out_dir, k, "%06d.jpg")
        output_path = os.path.join(out_dir, f"{k}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-hide_banner",
                "-y",
                "-framerate",
                str(fps),
                "-start_number",
                "0",
                "-i",
                input_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            check=True,
        )
