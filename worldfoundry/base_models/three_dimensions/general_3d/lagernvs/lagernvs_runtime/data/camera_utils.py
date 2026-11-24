# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure camera math utilities: intrinsic matrices, Plucker rays, crop geometry."""

import einops
import torch


def get_full_res_crop_dims_constant_ar(orig_hw, tgt_hw):
    """Computes aspect-ratio maintaining crop dimensions.

    Given original and target (height, width), returns the crop dimensions at
    original resolution such that resizing to tgt_hw preserves the aspect ratio.
    """
    orig_h, orig_w = orig_hw
    aspect_ratio_tgt_h_div_w = tgt_hw[0] / tgt_hw[1]
    aspect_ratio_src_h_div_w = orig_hw[0] / orig_hw[1]
    if aspect_ratio_tgt_h_div_w > aspect_ratio_src_h_div_w:
        crop_h = orig_h
        crop_w = int(crop_h / aspect_ratio_tgt_h_div_w)
    else:
        crop_w = orig_w
        crop_h = int(crop_w * aspect_ratio_tgt_h_div_w)
    return (crop_h, crop_w)


def get_K_matrices(fxfycxcy):
    """Convert [V, 4] fxfycxcy tensor to [V, 3, 3] intrinsic matrices."""
    Ks = []
    for fxfycxcy_inst in fxfycxcy:
        fx, fy, cx, cy = fxfycxcy_inst
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks.append(K)
    return torch.stack(Ks)


def get_uv_hom(raymap_hw):
    """Create homogeneous pixel coordinates grid.

    Args:
        raymap_hw: (height, width) tuple
    Returns:
        uv_hom: [H, W, 3] tensor of homogeneous pixel coordinates
    """
    height, width = raymap_hw

    pixel_centers_x = torch.linspace(0.5, width - 0.5, width)[None, :]
    pixel_centers_y = torch.linspace(0.5, height - 0.5, height)[:, None]

    pixel_centers_x = pixel_centers_x.expand((height, -1))
    pixel_centers_y = pixel_centers_y.expand((-1, width))
    ones = torch.ones_like(pixel_centers_x, dtype=pixel_centers_x.dtype)

    uv_hom = torch.stack([pixel_centers_x, pixel_centers_y, ones], axis=2).float()

    return uv_hom


def get_ray_dirs_local(uv_hom, raymap_Ks):
    """Compute local ray directions from homogeneous coords and intrinsics.

    Args:
        uv_hom: [V, H*W, 3]
        raymap_Ks: [V, 3, 3]
    Returns:
        ray_dirs_local: [V, H*W, 3]
    """
    uv_hom_T = uv_hom.transpose(-2, -1)  # v x 3 x hw
    K_inv = torch.linalg.inv(raymap_Ks).float()  # v x 3 x 3
    ray_dirs_local = torch.bmm(K_inv, uv_hom_T).transpose(-2, -1)  # v x hw x 3
    ray_dirs_local = ray_dirs_local / torch.linalg.norm(ray_dirs_local, dim=-1, keepdim=True)
    return ray_dirs_local


def get_ray_dirs_global(c2w, ray_dirs_local):
    """Rotate local ray directions to world frame using c2w rotation.

    Args:
        c2w: [V, 4, 4]
        ray_dirs_local: [V, H*W, 3]
    Returns:
        ray_dirs_global: [V, H*W, 3]
    """
    ray_dirs_global = torch.bmm(c2w[:, :3, :3], ray_dirs_local.transpose(-2, -1)).transpose(-2, -1)  # b x hw x 3
    return ray_dirs_global


def compute_plucker_rays(c2w, raymap_Ks, raymap_hw):
    """Compute Plucker ray coordinates from camera poses and intrinsics.

    Args:
        c2w: [V, 4, 4] camera-to-world matrices
        raymap_Ks: [V, 3, 3] intrinsic matrices
        raymap_hw: (H, W)
    Returns:
        plucker_rays: [V, 6, H, W]
    """
    uv = get_uv_hom(raymap_hw)
    uv = uv[None, ...].expand(c2w.shape[0], *uv.shape)  # v x h x w x 3
    _, h, w, _ = uv.shape
    uv = uv.view(c2w.shape[0], -1, 3)  # v x h*w x 3
    dirs_local = get_ray_dirs_local(uv, raymap_Ks)
    dirs_global = get_ray_dirs_global(c2w, dirs_local)
    dirs_global = dirs_global.view(c2w.shape[0], h, w, 3)  # v x h x w x 3
    ray_o = c2w[:, :3, 3][:, None, None, :]  # v x 1 x 1 x 3
    ray_o = ray_o.expand_as(dirs_global)  # v x h x w x 3
    moment = torch.cross(ray_o, dirs_global, dim=-1)
    plucker_rays = torch.cat([moment, dirs_global], dim=-1)
    plucker_rays = einops.rearrange(plucker_rays, "v h w c -> v c h w")
    return plucker_rays


def adjust_intrinsics_for_crop_and_resize(fxfycxcy_orig, im_hw_orig, crop_hw_in_orig, tgt_hw):
    """Adjust intrinsics for center-crop at original resolution then resize to tgt_hw.

    Assumes crop_hw_in_orig preserves the target aspect ratio.
    """
    fx_orig, fy_orig, cx_orig, cy_orig = fxfycxcy_orig
    cx, cy = (
        (cx_orig - (im_hw_orig[1] - crop_hw_in_orig[1]) // 2) * tgt_hw[1] / crop_hw_in_orig[1],
        (cy_orig - (im_hw_orig[0] - crop_hw_in_orig[0]) // 2) * tgt_hw[0] / crop_hw_in_orig[0],
    )
    fx = fx_orig * tgt_hw[1] / crop_hw_in_orig[1]
    fy = fy_orig * tgt_hw[0] / crop_hw_in_orig[0]
    return fx, fy, cx, cy
