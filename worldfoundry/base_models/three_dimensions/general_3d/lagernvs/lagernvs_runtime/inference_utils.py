# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import math
import os
from pathlib import Path

import av
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.load_fn import (
    load_and_preprocess_images as _vggt_load_and_preprocess_images,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _round_to_multiple(value, multiple):
    return max(multiple, int(round(value / multiple) * multiple))


def _load_lagernvs_images(image_names, *, mode="resize", target_size=512, patch_size=8):
    if len(image_names) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    for image_name in image_names:
        image = Image.open(image_name)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size

        if mode == "square_crop":
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            new_width = new_height = _round_to_multiple(target_size, patch_size)
        else:
            scale = target_size / max(width, height)
            new_width = _round_to_multiple(width * scale, patch_size)
            new_height = _round_to_multiple(height * scale, patch_size)

        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = to_tensor(image)
        images.append(tensor)
        shapes.add((tensor.shape[1], tensor.shape[2]))

    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded = []
        for tensor in images:
            h_padding = max_height - tensor.shape[1]
            w_padding = max_width - tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                tensor = torch.nn.functional.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded.append(tensor)
        images = padded

    return torch.stack(images)


def load_and_preprocess_images_compat(image_names, *, mode="resize", target_size=512, patch_size=8):
    try:
        return _vggt_load_and_preprocess_images(
            image_names,
            mode=mode,
            target_size=target_size,
            patch_size=patch_size,
        )
    except TypeError:
        if target_size == 518 and patch_size == 14:
            return _vggt_load_and_preprocess_images(
                image_names,
                mode="pad" if mode == "resize" else "crop",
            )
        return _load_lagernvs_images(
            image_names,
            mode=mode,
            target_size=target_size,
            patch_size=patch_size,
        )


def save_video(video_tensor, output_path, fps=25):
    video = einops.rearrange(video_tensor, "v c h w -> v h w c")
    video = video.detach().cpu().numpy()
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)

    with io.BytesIO() as buffer:
        with av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.height, stream.width = video.shape[1], video.shape[2]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18"}
            for frame_np in video:
                frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        buffer.seek(0)
        with open(output_path, "wb") as f_out:
            f_out.write(buffer.getvalue())


def render_chunked(
    model,
    inputs,
    view_chunk_size=16,
    num_cond_views=2,
    device=None,
):
    """Chunked rendering for when number of total views is large.

    Useful mostly for evaluation when number of views is large.

    Args:
        model: The viewgen model.
        inputs: Tuple of (cond_images, rays, cam_tokens) where:
            - cond_images: (B, num_cond_views, C, H, W) conditioning images
            - rays: (B, num_cond_views + video_length, 6, H, W) Plucker rays
            - cam_tokens: (B, num_cond_views + video_length, 11) camera tokens
        view_chunk_size: Number of target views per chunk.
        num_cond_views: Number of conditioning views.
        device: Device to move chunks to.
    """
    cond_images, rays_plucker, cam_token = inputs

    cond_plucker = rays_plucker[:, :num_cond_views, ...]
    cond_tokens = cam_token[:, :num_cond_views, ...]

    tgt_plucker = rays_plucker[:, num_cond_views:, ...]
    tgt_tokens = cam_token[:, num_cond_views:, ...]
    video_length = tgt_plucker.shape[1]

    # Create black padding for target views (model ignores these pixels)
    B, _, C, H, W = cond_images.shape
    tgt_images = torch.zeros(B, video_length, C, H, W, device=cond_images.device)

    video_out = []
    if device is None:
        device = cond_images.device
    num_chunks = math.ceil(video_length / view_chunk_size)
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * view_chunk_size
        end_idx = min((chunk_idx + 1) * view_chunk_size, video_length)
        chunk_tgt_images = tgt_images[:, start_idx:end_idx, ...]
        chunk_tgt_plucker = tgt_plucker[:, start_idx:end_idx, ...]
        chunk_tgt_tokens = tgt_tokens[:, start_idx:end_idx, ...]

        chunk_images = torch.concat([cond_images, chunk_tgt_images], dim=1)
        chunk_plucker = torch.concat([cond_plucker, chunk_tgt_plucker], dim=1)
        chunk_tokens = torch.concat([cond_tokens, chunk_tgt_tokens], dim=1)

        chunk_images = chunk_images.to(device)
        chunk_tokens = chunk_tokens.to(device)
        chunk_plucker = chunk_plucker.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk_out = model(
                chunk_images,
                chunk_plucker,
                chunk_tokens,
                num_cond_views=num_cond_views,
            )
        video_out.append(chunk_out[:, num_cond_views:, :3, ...])
    video_out = torch.cat(video_out, dim=1)
    return video_out


def create_360_camera_trajectory_from_c2w_and_intrinsics(
    c2w_poses, intrinsics, num_frames_traj, num_cond, bounds=(0, math.pi * 2)
):
    """Create 360 camera trajectory by fitting a circle to existing camera positions.

    The circular path's position is determined by fitting a circle to all input
    camera c2ws, and the look-at point is determined by checking the look-at point
    of all input cameras.

    Args:
        c2w_poses: Camera-to-world poses, shape B x V x 4 x 4
        intrinsics: Camera intrinsics matrices, shape B x V x 3 x 3
        num_frames_traj: Number of frames in the trajectory
        num_cond: Number of conditioning views

    Returns:
        Tuple of (cond_c2w, new_c2w, new_fxfycxcy) where:
        - cond_c2w: B x num_cond x 4 x 4 (conditioning camera c2w)
        - new_c2w: B x num_frames_traj x 4 x 4 (new trajectory c2w)
        - new_fxfycxcy: B x num_frames_traj x 4 (new intrinsics as [fx, fy, cx, cy])
    """
    B = c2w_poses.shape[0]
    num_input_views = c2w_poses.shape[1]
    device = c2w_poses.device
    cond_extrinsics = c2w_poses[:, :num_cond, :3, :4]
    cond_intrinsics = intrinsics[:, :num_cond]

    # Convert to c2w (camera to world) format
    cond_c2w = torch.zeros(B, num_cond, 4, 4, device=device)
    cond_c2w[:, :, :3, :] = cond_extrinsics
    cond_c2w[:, :, 3, 3] = 1.0

    # Use all input cameras for plane fitting and geometry estimation
    all_c2w = torch.zeros(B, num_input_views, 4, 4, device=device)
    all_c2w[:, :, :3, :] = c2w_poses[:, :, :3, :4]
    all_c2w[:, :, 3, 3] = 1.0

    # Compute center of all input cameras (average position)
    center = all_c2w[:, :, :3, 3].mean(dim=1)  # B x 3

    # Compute look-at point (minimizes distance to all input camera look-at rays)
    cam_positions = all_c2w[:, :, :3, 3]  # B x num_input_views x 3
    cam_forward = all_c2w[:, :, :3, 2]  # B x num_input_views x 3 (z-axis)

    # For each ray with origin p and direction d (normalized),
    # the closest point on the ray to a point q is: p + d * dot(q - p, d)
    # We want to minimize sum_i ||q - (p_i + d_i * dot(q - p_i, d_i))||^2
    # Solution: (I - sum_i d_i d_i^T)^{-1} * sum_i (I - d_i d_i^T) p_i

    # Compute projection matrices for each ray: I - d_i d_i^T
    identity = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 3 x 3
    identity = identity.expand(B, num_input_views, -1, -1)  # B x num_input_views x 3 x 3

    d = cam_forward  # B x num_input_views x 3
    ddT = torch.einsum("bvi,bvj->bvij", d, d)  # B x num_input_views x 3 x 3
    proj = identity - ddT  # B x num_input_views x 3 x 3

    # Sum projection matrices
    proj_sum = proj.sum(dim=1)  # B x 3 x 3

    # Compute sum of projected positions
    proj_p = torch.einsum("bvij,bvj->bvi", proj, cam_positions)  # B x num_input_views x 3
    proj_p_sum = proj_p.sum(dim=1)  # B x 3

    # Solve for look-at point
    look_at = torch.linalg.solve(proj_sum, proj_p_sum.unsqueeze(-1)).squeeze(-1)  # B x 3

    # Fit a plane to the camera positions using PCA
    # Center the positions
    centered_positions = cam_positions - center.unsqueeze(1)  # B x num_input_views x 3

    # Compute covariance matrix for each batch
    # cov = (1/n) * X^T X where X is centered_positions
    plane_basis_1 = torch.zeros(B, 3, device=device)
    plane_basis_2 = torch.zeros(B, 3, device=device)
    plane_normal = torch.zeros(B, 3, device=device)

    for b in range(B):
        # Compute covariance matrix: 3 x 3
        cov = torch.matmul(centered_positions[b].T, centered_positions[b]) / num_input_views  # 3 x 3

        # Perform SVD to get principal components
        # U contains the eigenvectors (principal components)
        # S contains the singular values (related to eigenvalues)
        U, S, _ = torch.svd(cov)

        # The two largest singular values correspond to the plane basis
        # The smallest singular value corresponds to the plane normal
        plane_basis_1[b] = U[:, 0]  # First principal component (largest variance)
        plane_basis_2[b] = U[:, 1]  # Second principal component
        plane_normal[b] = U[:, 2]  # Third principal component (smallest variance, normal to plane)

    # Ensure plane normal points in the general "up" direction
    # In OpenCV coordinate system, y-axis points down, so "up" is negative y-axis
    cam_down = all_c2w[:, :, :3, 1]  # B x num_input_views x 3 (y-axis, points down)
    avg_up = -cam_down.mean(dim=1)  # B x 3 (negate to get "up" direction)
    for b in range(B):
        if torch.dot(plane_normal[b], avg_up[b]) < 0:
            plane_normal[b] = -plane_normal[b]

    # Compute radius as average distance from all input cameras to center (projected on plane)
    # Project positions onto the plane defined by plane_basis_1 and plane_basis_2
    projected_offsets = torch.zeros(B, num_input_views, 2, device=device)
    for b in range(B):
        for v in range(num_input_views):
            offset = cam_positions[b, v] - center[b]  # 3
            projected_offsets[b, v, 0] = torch.dot(offset, plane_basis_1[b])
            projected_offsets[b, v, 1] = torch.dot(offset, plane_basis_2[b])

    # Compute radius as average distance in the plane
    radius = torch.norm(projected_offsets, dim=2).mean(dim=1)  # B

    # Generate new camera positions on circular path in the fitted plane
    angles = torch.linspace(bounds[0], bounds[1], num_frames_traj + 1, device=device)[:-1]

    # Create positions on circle for each batch using the fitted plane basis vectors
    new_positions = torch.zeros(B, num_frames_traj, 3, device=device)
    for b in range(B):
        for v in range(num_frames_traj):
            angle = angles[v]
            # Use plane basis vectors to create circular path in the fitted plane
            offset = radius[b] * (torch.cos(angle) * plane_basis_1[b] + torch.sin(angle) * plane_basis_2[b])
            new_positions[b, v] = center[b] + offset

    # Compute rotation matrices to look at the target point
    new_c2w = torch.zeros(B, num_frames_traj, 4, 4, device=device)

    for b in range(B):
        for v in range(num_frames_traj):
            pos = new_positions[b, v]  # 3
            target = look_at[b]  # 3

            # Forward direction (z-axis in OpenCV): from camera to target
            forward = F.normalize(target - pos, dim=-1)

            # Right direction (x-axis): perpendicular to forward and plane_normal
            right = F.normalize(torch.cross(forward, plane_normal[b], dim=-1), dim=-1)

            # Down direction (y-axis): perpendicular to forward and right
            down = F.normalize(torch.cross(forward, right, dim=-1), dim=-1)

            # Build rotation matrix [right | down | forward]
            R = torch.stack([right, down, forward], dim=-1)  # 3 x 3

            new_c2w[b, v, :3, :3] = R
            new_c2w[b, v, :3, 3] = pos
            new_c2w[b, v, 3, 3] = 1.0

    # Use intrinsics from conditioning camera 0 for all new views
    ref_intrinsics = cond_intrinsics[:, 0:1, :, :]  # B x 1 x 3 x 3
    new_intrinsics = ref_intrinsics.expand(B, num_frames_traj, 3, 3)

    # Extract fxfycxcy from intrinsics
    new_fxfycxcy = torch.stack(
        [
            new_intrinsics[:, :, 0, 0],  # fx
            new_intrinsics[:, :, 1, 1],  # fy
            new_intrinsics[:, :, 0, 2],  # cx
            new_intrinsics[:, :, 1, 2],  # cy
        ],
        dim=-1,
    )  # B x num_frames_traj x 4

    return cond_c2w, new_c2w, new_fxfycxcy


def compute_plucker_coordinates(c2w, fxfycxcy, image_size_hw):
    """Compute plucker coordinates from camera parameters.

    Args:
        c2w: Camera-to-world matrices, shape B x V x 4 x 4
        fxfycxcy: Camera intrinsics [fx, fy, cx, cy], shape B x V x 4
        image_size_hw: Tuple of (height, width) for image

    Returns:
        Plucker coordinates, shape B x V x 6 x H x W
        Format: [o x d, d] where o is ray origin and d is ray direction
    """
    B, V = c2w.shape[:2]
    h, w = image_size_hw
    device = c2w.device

    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    x = x[None, None, :, :].expand(B, V, -1, -1).reshape(B, V, -1)  # B x V x (h*w)
    y = y[None, None, :, :].expand(B, V, -1, -1).reshape(B, V, -1)  # B x V x (h*w)

    # Convert pixel coordinates to normalized camera coordinates
    fx = fxfycxcy[:, :, 0:1]  # B x V x 1
    fy = fxfycxcy[:, :, 1:2]  # B x V x 1
    cx = fxfycxcy[:, :, 2:3]  # B x V x 1
    cy = fxfycxcy[:, :, 3:4]  # B x V x 1

    x = (x.float() + 0.5 - cx) / fx
    y = (y.float() + 0.5 - cy) / fy
    z = torch.ones_like(x)

    # Ray directions in camera space
    ray_d = torch.stack([x, y, z], dim=3).float()  # B x V x (h*w) x 3

    # Transform to world space
    R = c2w[:, :, :3, :3]  # B x V x 3 x 3
    ray_d = torch.einsum("bvij,bvpj->bvpi", R, ray_d)  # B x V x (h*w) x 3
    ray_d = ray_d / torch.norm(ray_d, dim=3, keepdim=True)  # Normalize

    # Ray origins (camera positions)
    ray_o = c2w[:, :, :3, 3][:, :, None, :].expand_as(ray_d)  # B x V x (h*w) x 3

    # Reshape to image dimensions
    ray_o = einops.rearrange(ray_o, "b v (h w) c -> b v c h w", h=h, w=w)
    ray_d = einops.rearrange(ray_d, "b v (h w) c -> b v c h w", h=h, w=w)

    # Compute plucker coordinates: [o x d, d]
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    plucker = torch.cat([o_cross_d, ray_d], dim=2)  # B x V x 6 x h x w

    return plucker


def create_bspline_interp(
    c2w_poses,
    intrinsics,
    num_frames_traj,
    num_cond,
    ease_in_out=False,
    double_to_repeat=False,
):
    """Create smooth camera trajectory using cubic B-spline interpolation.

    Uses cubic B-splines with C2 continuity (smooth second derivatives) for
    very smooth camera paths. The curve approximates the conditioning camera
    poses and samples num_frames_traj points with constant speed along the
    curve using arc-length parameterization.

    Args:
        c2w_poses: Camera-to-world poses, shape B x V x 4 x 4
        intrinsics: Camera intrinsics matrices, shape B x V x 3 x 3
        num_frames_traj: Number of frames in the trajectory
        num_cond: Number of conditioning views to interpolate through
        ease_in_out: If True, applies ease-in/ease-out using smoothstep curve,
            making camera speed 0 at start and end of path. If False, uses
            constant speed (default behavior).
        double_to_repeat: If True, creates a path that goes forward then backward

    Returns:
        Tuple of (cond_c2w, new_c2w, new_fxfycxcy) where:
        - cond_c2w: B x num_cond x 4 x 4 (conditioning camera c2w)
        - new_c2w: B x num_frames_traj x 4 x 4 (new trajectory c2w)
        - new_fxfycxcy: B x num_frames_traj x 4 (new intrinsics)
    """
    B = c2w_poses.shape[0]
    device = c2w_poses.device

    # Extract conditioning poses
    if num_cond == 1:
        cond_extrinsics = c2w_poses[:, :, :3, :4]
        cond_intrinsics = intrinsics
    else:
        cond_extrinsics = c2w_poses[:, :num_cond, :3, :4]
        cond_intrinsics = intrinsics[:, :num_cond]

    # Convert to c2w format
    cond_c2w = torch.zeros(B, cond_extrinsics.shape[1], 4, 4, device=device)
    cond_c2w[:, :, :3, :] = cond_extrinsics
    cond_c2w[:, :, 3, 3] = 1.0

    # Extract positions and rotations from conditioning poses
    cond_positions = cond_c2w[:, :, :3, 3]  # B x num_cond x 3
    cond_rotations = cond_c2w[:, :, :3, :3]  # B x num_cond x 3 x 3

    # Create interpolated trajectory
    num_frames_traj_total = num_frames_traj if not double_to_repeat else num_frames_traj * 2
    new_c2w = torch.zeros(B, num_frames_traj_total, 4, 4, device=device)

    for b in range(B):
        # Convert rotation matrices to quaternions for smooth interpolation
        cond_quaternions = _rotation_matrices_to_quaternions(cond_rotations[b])

        # Step 1: Oversample the curve to compute arc length
        # Use fine sampling to accurately measure curve length
        num_fine_samples = max(cond_extrinsics.shape[1] * 50, 500)
        t_fine = torch.linspace(0, 1, num_fine_samples, device=device)

        # Interpolate positions at fine resolution using cubic B-spline
        positions_fine = []
        quaternions_fine = []
        for t in t_fine:
            pos = _cubic_bspline_interpolate_points(cond_positions[b], t)
            positions_fine.append(pos)
            # Interpolate quaternions using B-spline as well for C2 continuity
            quat = _cubic_bspline_interpolate_quaternions(cond_quaternions, t)
            quaternions_fine.append(quat)

        positions_fine = torch.stack(positions_fine)  # num_fine_samples x 3
        quaternions_fine = torch.stack(quaternions_fine)  # num_fine_samples x 4

        # Step 2: Compute cumulative arc length
        segments = positions_fine[1:] - positions_fine[:-1]
        segment_lengths = torch.norm(segments, dim=-1)
        arc_lengths = torch.cat([torch.zeros(1, device=device), torch.cumsum(segment_lengths, dim=0)])
        total_length = arc_lengths[-1]

        # Step 3: Sample in arc length space (uniform or eased)
        if ease_in_out:
            # Apply smoothstep easing: speed is 0 at start and end
            # Use smoothstep function: 3t^2 - 2t^3 for t in [0, 1]
            # This gives zero derivative at t=0 and t=1
            t_linear = torch.linspace(0, 1, num_frames_traj, device=device)
            t_eased = t_linear * t_linear * (3.0 - 2.0 * t_linear)
            target_arc_lengths = t_eased * total_length
        else:
            # Uniform sampling (constant speed)
            target_arc_lengths = torch.linspace(0, total_length, num_frames_traj, device=device)
        if double_to_repeat:
            target_arc_lengths = torch.cat([target_arc_lengths, torch.flip(target_arc_lengths, [0])], dim=0)

        new_positions = []
        new_quaternions = []

        for target_length in target_arc_lengths:
            # Find the fine sample index corresponding to this arc length
            idx = torch.searchsorted(arc_lengths, target_length).item() - 1
            idx = max(0, min(idx, len(arc_lengths) - 2))

            # Interpolate within segment to get exact arc length
            alpha = (target_length - arc_lengths[idx]) / segment_lengths[idx] if segment_lengths[idx] > 1e-08 else 0.0
            alpha = max(0.0, min(1.0, float(alpha)))

            # Get position at this arc length
            pos = positions_fine[idx] + alpha * (positions_fine[idx + 1] - positions_fine[idx])
            new_positions.append(pos)

            # Get quaternion at this arc length using SLERP between fine samples
            # This ensures smooth rotation even within the fine sampling
            quat = _slerp_quaternions(quaternions_fine[idx], quaternions_fine[idx + 1], alpha)
            new_quaternions.append(quat)

        # Stack into tensors
        new_positions_tensor = torch.stack(new_positions)  # num_frames_traj x 3
        new_quaternions_tensor = torch.stack(new_quaternions)  # num_frames_traj x 4

        # Convert quaternions back to rotation matrices
        new_rotations_tensor = _quaternions_to_rotation_matrices(new_quaternions_tensor)

        # Build c2w matrices
        new_c2w[b, :, :3, :3] = new_rotations_tensor
        new_c2w[b, :, :3, 3] = new_positions_tensor
        new_c2w[b, :, 3, 3] = 1.0

    # Use intrinsics from conditioning camera 0 for all new views
    ref_intrinsics = cond_intrinsics[:, 0:1, :, :]  # B x 1 x 3 x 3
    new_intrinsics = ref_intrinsics.expand(B, num_frames_traj_total, 3, 3)

    # Extract fxfycxcy from intrinsics
    new_fxfycxcy = torch.stack(
        [
            new_intrinsics[:, :, 0, 0],  # fx
            new_intrinsics[:, :, 1, 1],  # fy
            new_intrinsics[:, :, 0, 2],  # cx
            new_intrinsics[:, :, 1, 2],  # cy
        ],
        dim=-1,
    )  # B x num_frames_traj x 4

    return cond_c2w, new_c2w, new_fxfycxcy


def _catmull_rom_interpolate_points(points, t_control, t):
    """Interpolate points using Catmull-Rom spline.

    Catmull-Rom splines pass through all control points and have C1 continuity
    (smooth gradients at control points).

    Args:
        points: Control points, shape (num_points, 3)
        t_control: Parameter values for control points, shape (num_points,)
        t: Target parameter value to interpolate at (scalar)

    Returns:
        Interpolated position, shape (3,)
    """
    num_points = len(points)

    # Handle boundary cases
    if t <= t_control[0]:
        return points[0]
    elif t >= t_control[-1]:
        return points[-1]

    # Find which segment we're in
    idx = torch.searchsorted(t_control, t).item() - 1
    idx = max(0, min(idx, num_points - 2))

    # Normalize t to [0, 1] within the segment
    t_local = (t - t_control[idx]) / (t_control[idx + 1] - t_control[idx])
    t_local = float(torch.clamp(t_local, 0, 1))

    # Get the four control points for Catmull-Rom spline
    # p0 and p3 are used to determine tangents at p1 and p2
    p0 = points[max(0, idx - 1)]
    p1 = points[idx]
    p2 = points[idx + 1]
    p3 = points[min(num_points - 1, idx + 2)]

    # Catmull-Rom spline formula
    # This ensures the curve passes through p1 and p2 with smooth tangents
    t2 = t_local * t_local
    t3 = t2 * t_local

    result = 0.5 * (
        (2 * p1) + (-p0 + p2) * t_local + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )

    return result


def _cubic_bspline_interpolate_points(points, t):
    """Interpolate points using uniform cubic B-spline with clamped endpoints.

    Cubic B-splines provide C2 continuity (smooth second derivatives) for
    very smooth camera paths. The curve approximates the control points
    rather than passing through them exactly.

    Uses clamped (open) B-spline so the curve starts and ends at the
    first and last control points.

    Args:
        points: Control points, shape (num_points, 3)
        t: Target parameter value in [0, 1] to interpolate at (scalar)

    Returns:
        Interpolated position, shape (3,)
    """
    num_points = len(points)
    device = points.device

    # Need at least 2 control points
    if num_points < 2:
        return points[0] if num_points == 1 else torch.zeros(3, device=device)

    # For clamped cubic B-splines, we augment the control points
    # by repeating the first and last points 3 times each
    # This ensures the curve starts and ends at the endpoints
    augmented_points = torch.cat(
        [
            points[0:1].expand(3, -1),  # Repeat first point 3 times
            points,  # Original control points
            points[-1:].expand(3, -1),  # Repeat last point 3 times
        ],
        dim=0,
    )  # (num_points + 6) x 3

    num_augmented = len(augmented_points)
    degree = 3  # Cubic B-spline

    # Create uniform knot vector for clamped B-spline
    # For clamped spline with n control points and degree p:
    # knot vector has n + p + 1 knots
    num_knots = num_augmented + degree + 1

    # Clamped knot vector: [0,0,0,0, ..., 1,1,1,1] with uniform spacing in between
    knots = torch.zeros(num_knots, device=device)
    knots[: degree + 1] = 0.0  # First (degree+1) knots are 0
    knots[-(degree + 1) :] = 1.0  # Last (degree+1) knots are 1

    # Uniform spacing in the middle
    num_internal = num_knots - 2 * (degree + 1)
    if num_internal > 0:
        knots[degree + 1 : -(degree + 1)] = torch.linspace(0, 1, num_internal + 2, device=device)[1:-1]

    # Clamp t to [0, 1]
    t = float(torch.clamp(t, 0.0, 1.0))

    # Find the knot span index for t using binary search
    # The span is the index i where knots[i] <= t < knots[i+1]
    span = _find_knot_span(t, degree, knots)

    # Compute the basis functions for this span
    basis = _compute_bspline_basis(span, t, degree, knots)

    # Compute the interpolated point as weighted sum of control points
    # For cubic B-spline, we use 4 control points (degree + 1)
    result = torch.zeros(3, device=device)
    for i in range(degree + 1):
        control_idx = span - degree + i
        if 0 <= control_idx < num_augmented:
            result += basis[i] * augmented_points[control_idx]

    return result


def _find_knot_span(t, degree, knots):
    """Find the knot span index for parameter t.

    Returns the index i such that knots[i] <= t < knots[i+1].

    Args:
        t: Parameter value in [0, 1]
        degree: Degree of the B-spline
        knots: Knot vector

    Returns:
        Knot span index
    """
    num_knots = len(knots)
    n = num_knots - degree - 2  # Number of control points - 1

    # Special case for t at the end
    if t >= knots[n + 1]:
        return n

    # Binary search for the span
    low = degree
    high = n + 1

    mid = (low + high) // 2
    while t < knots[mid] or t >= knots[mid + 1]:
        if t < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def _compute_bspline_basis(span, t, degree, knots):
    """Compute B-spline basis functions using Cox-de Boor recursion.

    Args:
        span: Knot span index
        t: Parameter value
        degree: Degree of B-spline
        knots: Knot vector

    Returns:
        Tensor of basis function values, shape (degree + 1,)
    """
    device = knots.device

    # Initialize basis functions
    basis = torch.zeros(degree + 1, device=device)
    left = torch.zeros(degree + 1, device=device)
    right = torch.zeros(degree + 1, device=device)

    basis[0] = 1.0

    # Cox-de Boor recursion
    for j in range(1, degree + 1):
        left[j] = t - knots[span + 1 - j]
        right[j] = knots[span + j] - t

        saved = 0.0
        for r in range(j):
            temp = basis[r] / (right[r + 1] + left[j - r])
            basis[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        basis[j] = saved

    return basis


def _slerp_rotation_matrices(rotations, t):
    """Interpolate rotation matrices using SLERP across all control rotations.

    Uses spherical linear interpolation (SLERP) to smoothly interpolate
    between rotation matrices. For multiple control points, performs
    sequential SLERP operations.

    Args:
        rotations: Rotation matrices, shape (num_rotations, 3, 3)
        t: Target parameter value in [0, 1] to interpolate at (scalar)

    Returns:
        Interpolated rotation matrix, shape (3, 3)
    """
    num_rotations = len(rotations)

    # Handle boundary cases
    if t <= 0.0:
        return rotations[0]
    elif t >= 1.0:
        return rotations[-1]

    # Scale t to segment space [0, num_rotations-1]
    t_scaled = t * (num_rotations - 1)

    # Find which segment we're in
    idx = int(torch.floor(torch.tensor(t_scaled)).item())
    idx = max(0, min(idx, num_rotations - 2))

    # Local parameter within the segment [0, 1]
    t_local = t_scaled - idx

    # Get the two rotation matrices to interpolate between
    R1 = rotations[idx]
    R2 = rotations[idx + 1]

    # Perform SLERP between R1 and R2
    return _slerp_two_rotations(R1, R2, t_local)


def _slerp_two_rotations(R1, R2, t):
    """Perform SLERP between two rotation matrices.

    Args:
        R1: First rotation matrix, shape (3, 3)
        R2: Second rotation matrix, shape (3, 3)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated rotation matrix, shape (3, 3)
    """
    # Compute relative rotation: R_rel = R1^T * R2
    R_rel = torch.matmul(R1.T, R2)

    # Convert relative rotation to axis-angle representation
    trace = torch.trace(R_rel)

    # Handle numerical issues
    if torch.abs(trace - 3.0) < 1e-6:
        # Rotations are identical
        return R1

    # Compute rotation angle
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    if torch.abs(angle) < 1e-6:
        # Very small rotation, use linear interpolation
        R_interp = (1 - t) * R1 + t * R2
        # Re-orthogonalize
        U, _, Vt = torch.linalg.svd(R_interp)
        R_ortho = torch.matmul(U, Vt)
        # Ensure proper rotation (det = +1)
        if torch.det(R_ortho) < 0:
            Vt[-1, :] *= -1
            R_ortho = torch.matmul(U, Vt)
        return R_ortho

    # Extract rotation axis
    axis = torch.stack(
        [
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ]
    ) / (2 * torch.sin(angle))

    # Interpolate angle
    angle_interp = angle * t

    # Compute interpolated relative rotation using Rodrigues' formula
    K = torch.zeros(3, 3, device=R1.device, dtype=R1.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    identity_matrix = torch.eye(3, device=R1.device, dtype=R1.dtype)
    R_rel_interp = identity_matrix + torch.sin(angle_interp) * K + (1 - torch.cos(angle_interp)) * torch.matmul(K, K)

    # Apply interpolated relative rotation to R1
    R_result = torch.matmul(R1, R_rel_interp)

    return R_result


def _rotation_matrices_to_quaternions(rotation_matrices):
    """Convert rotation matrices to quaternions.

    Args:
        rotation_matrices: Rotation matrices, shape (num_rotations, 3, 3)

    Returns:
        Quaternions in [w, x, y, z] format, shape (num_rotations, 4)
    """
    num_rotations = rotation_matrices.shape[0]
    device = rotation_matrices.device
    quaternions = torch.zeros(num_rotations, 4, device=device)

    for i in range(num_rotations):
        R = rotation_matrices[i]
        trace = torch.trace(R)

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            quaternions[i, 0] = 0.25 / s
            quaternions[i, 1] = (R[2, 1] - R[1, 2]) * s
            quaternions[i, 2] = (R[0, 2] - R[2, 0]) * s
            quaternions[i, 3] = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            quaternions[i, 0] = (R[2, 1] - R[1, 2]) / s
            quaternions[i, 1] = 0.25 * s
            quaternions[i, 2] = (R[0, 1] + R[1, 0]) / s
            quaternions[i, 3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            quaternions[i, 0] = (R[0, 2] - R[2, 0]) / s
            quaternions[i, 1] = (R[0, 1] + R[1, 0]) / s
            quaternions[i, 2] = 0.25 * s
            quaternions[i, 3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            quaternions[i, 0] = (R[1, 0] - R[0, 1]) / s
            quaternions[i, 1] = (R[0, 2] + R[2, 0]) / s
            quaternions[i, 2] = (R[1, 2] + R[2, 1]) / s
            quaternions[i, 3] = 0.25 * s

    return quaternions


def _quaternions_to_rotation_matrices(quaternions):
    """Convert quaternions to rotation matrices.

    Args:
        quaternions: Quaternions in [w, x, y, z] format, shape (num_quaternions, 4)

    Returns:
        Rotation matrices, shape (num_quaternions, 3, 3)
    """
    num_quaternions = quaternions.shape[0]
    device = quaternions.device
    rotation_matrices = torch.zeros(num_quaternions, 3, 3, device=device)

    for i in range(num_quaternions):
        w, x, y, z = quaternions[i]

        # Normalize quaternion
        norm = torch.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # Compute rotation matrix
        rotation_matrices[i, 0, 0] = 1 - 2 * (y * y + z * z)
        rotation_matrices[i, 0, 1] = 2 * (x * y - w * z)
        rotation_matrices[i, 0, 2] = 2 * (x * z + w * y)
        rotation_matrices[i, 1, 0] = 2 * (x * y + w * z)
        rotation_matrices[i, 1, 1] = 1 - 2 * (x * x + z * z)
        rotation_matrices[i, 1, 2] = 2 * (y * z - w * x)
        rotation_matrices[i, 2, 0] = 2 * (x * z - w * y)
        rotation_matrices[i, 2, 1] = 2 * (y * z + w * x)
        rotation_matrices[i, 2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrices


def _cubic_bspline_interpolate_quaternions(quaternions, t):
    """Interpolate quaternions using cubic B-spline with clamped endpoints.

    Uses the same B-spline approach as positions, but operates on quaternions
    with proper normalization and sign handling for smooth interpolation.

    Args:
        quaternions: Quaternions in [w, x, y, z] format, shape (num_quaternions, 4)
        t: Target parameter value in [0, 1] to interpolate at (scalar)

    Returns:
        Interpolated quaternion, shape (4,)
    """
    num_quaternions = len(quaternions)
    device = quaternions.device

    if num_quaternions < 2:
        return quaternions[0] if num_quaternions == 1 else torch.zeros(4, device=device)

    # Ensure quaternion continuity: flip signs to ensure shortest path
    aligned_quaternions = torch.zeros_like(quaternions)
    aligned_quaternions[0] = quaternions[0]

    for i in range(1, num_quaternions):
        # Check if we should flip the sign to ensure shortest path
        dot_product = torch.dot(aligned_quaternions[i - 1], quaternions[i])
        if dot_product < 0:
            aligned_quaternions[i] = -quaternions[i]
        else:
            aligned_quaternions[i] = quaternions[i]

    # Augment control points for clamped B-spline
    augmented_quaternions = torch.cat(
        [
            aligned_quaternions[0:1].expand(3, -1),
            aligned_quaternions,
            aligned_quaternions[-1:].expand(3, -1),
        ],
        dim=0,
    )

    num_augmented = len(augmented_quaternions)
    degree = 3

    # Create uniform knot vector for clamped B-spline
    num_knots = num_augmented + degree + 1
    knots = torch.zeros(num_knots, device=device)
    knots[: degree + 1] = 0.0
    knots[-(degree + 1) :] = 1.0

    num_internal = num_knots - 2 * (degree + 1)
    if num_internal > 0:
        knots[degree + 1 : -(degree + 1)] = torch.linspace(0, 1, num_internal + 2, device=device)[1:-1]

    t = float(torch.clamp(t, 0.0, 1.0))

    # Find knot span and compute basis functions
    span = _find_knot_span(t, degree, knots)
    basis = _compute_bspline_basis(span, t, degree, knots)

    # Compute interpolated quaternion as weighted sum
    result = torch.zeros(4, device=device)
    for i in range(degree + 1):
        control_idx = span - degree + i
        if 0 <= control_idx < num_augmented:
            result += basis[i] * augmented_quaternions[control_idx]

    # Normalize the result quaternion
    result = result / torch.norm(result)

    return result


def _slerp_quaternions(q1, q2, t):
    """Perform spherical linear interpolation (SLERP) between two quaternions.

    Args:
        q1: First quaternion [w, x, y, z], shape (4,)
        q2: Second quaternion [w, x, y, z], shape (4,)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated quaternion, shape (4,)
    """
    # Normalize quaternions
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)

    # Compute dot product
    dot = torch.dot(q1, q2)

    # If dot product is negative, flip q2 to ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Clamp dot product to avoid numerical issues with acos
    dot = torch.clamp(dot, -1.0, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / torch.norm(result)

    # Compute angle between quaternions
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Compute SLERP
    w1 = torch.sin((1 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    result = w1 * q1 + w2 * q2

    return result / torch.norm(result)


def create_target_camera_path(image_names, video_length, num_cond_views, image_size_hw, device, dtype, mode="resize"):
    """Create a target camera trajectory for rendering novel views.

    LagerNVS does not require input camera poses — it only needs a target
    camera path specifying where to render from (as Plucker rays). This function
    automatically constructs a smooth target trajectory by using VGGT to infer
    approximate input view positions, then interpolating a path through them.

    For multi-view (num_cond_views >= 2): Interpolates a B-spline path through
    the inferred view positions with camera-based scene normalization.
    For single-view (num_cond_views == 1): Uses world-based normalization and
    creates a forward dolly by translating +0.3 along the camera z-axis.

    Args:
        image_names: List of image file paths (loaded internally at 518px for VGGT)
        mode: Preprocessing mode for VGGT input images ("resize" or "square_crop")
        video_length: Number of target video frames to generate
        num_cond_views: Number of conditioning views
        image_size_hw: Tuple (H, W) for the target Plucker ray resolution
        device: Torch device
        dtype: Torch dtype for autocast (e.g. torch.bfloat16)

    Returns:
        Tuple of:
        - rays: Plucker rays tensor of shape (1, num_cond_views + video_length, 6, H, W)
        - cam_tokens: Camera tokens tensor of shape (1, num_cond_views + video_length, 11)
    """

    # Load images at 518px for VGGT pose estimation
    images = load_and_preprocess_images_compat(image_names, mode=mode, target_size=518, patch_size=14).to(device)

    # Use VGGT to infer approximate input view positions (for trajectory planning only)
    vggt_model = VGGT(pred_cameras=True)
    vggt_ckpt_env = os.environ.get("WORLDFOUNDRY_VGGT_CKPT")
    local_vggt_ckpt = Path(
        vggt_ckpt_env
        or Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "~/.cache/worldfoundry/checkpoints")).expanduser()
        / "VGGT-1B"
        / "model.pt"
    ).expanduser()
    if local_vggt_ckpt.is_file():
        vggt_pretrained_state = torch.load(local_vggt_ckpt, map_location="cpu")
    else:
        vggt_pretrained_state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location="cpu",
        )
    vggt_model.load_state_dict(vggt_pretrained_state, strict=False)
    vggt_model.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pose_enc = vggt_model(images)

    # Free VGGT memory (it is not used by the viewgen model itself)
    del vggt_model, vggt_pretrained_state
    torch.cuda.empty_cache()

    H, W = image_size_hw

    # Decode inferred poses to extrinsics (we only use camera positions/rotations
    # from VGGT, not its intrinsics — see default intrinsics construction below).
    if pose_enc.dim() == 2:
        pose_enc = pose_enc.unsqueeze(0)
    extrinsics_w2c, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_size_hw)

    # Use default intrinsics instead of VGGT estimates. VGGT's intrinsics are
    # approximate and can be noisy for few-view inputs. A standard pinhole
    # assumption (fx = fy = image_width, principal point at center) is
    # sufficient here for target views
    B, S = extrinsics_w2c.shape[:2]
    intrinsics = torch.zeros(B, S, 3, 3, device=extrinsics_w2c.device, dtype=extrinsics_w2c.dtype)
    intrinsics[:, :, 0, 0] = float(W)  # fx
    intrinsics[:, :, 1, 1] = float(W)  # fy (same as fx for square pixels)
    intrinsics[:, :, 0, 2] = float(W) / 2.0  # cx
    intrinsics[:, :, 1, 2] = float(H) / 2.0  # cy
    intrinsics[:, :, 2, 2] = 1.0

    # Invert w2c -> c2w for trajectory interpolation
    R_w2c = extrinsics_w2c[:, :, :3, :3]
    t_w2c = extrinsics_w2c[:, :, :3, 3:]
    R_c2w = R_w2c.transpose(-1, -2)
    t_c2w = -R_c2w @ t_w2c

    c2w = torch.zeros(B, S, 4, 4, device=extrinsics_w2c.device)
    c2w[:, :, :3, :3] = R_c2w
    c2w[:, :, :3, 3:] = t_c2w
    c2w[:, :, 3, 3] = 1.0

    # Normalize scene scale
    # Express all poses relative to the original first camera
    first_cam_inv = torch.linalg.inv(c2w[:, 0:1, :, :])  # (B, 1, 4, 4)
    c2w = first_cam_inv @ c2w

    # Sort cameras so the B-spline traverses them in a reasonable order.
    # Find the translation axis with the largest range and argsort along it.
    # Sorting happens after normalization so the reference frame stays fixed
    # to the original first input image.
    if num_cond_views >= 2:
        positions = c2w[0, :num_cond_views, :3, 3]  # (num_cond, 3)
        ranges = positions.max(dim=0).values - positions.min(dim=0).values
        sort_axis = ranges.argmax().item()
        sort_order = positions[:, sort_axis].argsort()
        c2w[:, :num_cond_views] = c2w[:, sort_order]
        intrinsics[:, :num_cond_views] = intrinsics[:, sort_order]

    total_views = num_cond_views + video_length

    # Build fxfycxcy from the default intrinsics matrix for ray computation
    default_fxfycxcy = torch.stack(
        [
            intrinsics[:, :, 0, 0],
            intrinsics[:, :, 1, 1],
            intrinsics[:, :, 0, 2],
            intrinsics[:, :, 1, 2],
        ],
        dim=-1,
    )  # (B, S, 4)

    if num_cond_views >= 2:
        # Camera-based normalization (base_dataset.py:288-299)
        scene_scale = 1.35 * torch.max(torch.norm(c2w[:, :num_cond_views, :3, 3], dim=-1))
        scene_scale = torch.clamp(scene_scale, min=1e-6)
        c2w[:, :, :3, 3] /= scene_scale
        camera_scale = torch.max(torch.norm(c2w[:, :num_cond_views, :3, 3], dim=-1)).item()

        cam_tokens = torch.zeros(1, total_views, 11)
        cam_tokens[:, :, 9] = camera_scale
        cam_tokens[:, :, 10] = 0.0

        # Interpolate smooth B-spline trajectory through inferred positions.
        # Use double_to_repeat to create a forth-and-back path.
        half_length = video_length // 2
        _, new_c2w, new_fxfycxcy = create_bspline_interp(
            c2w,
            intrinsics,
            num_frames_traj=half_length,
            num_cond=num_cond_views,
            double_to_repeat=True,
        )

        # Compute Plucker rays for target trajectory
        target_rays = compute_plucker_coordinates(new_c2w, new_fxfycxcy, image_size_hw)
    else:
        # Single-view: world-based normalization (scene_scale = 1.0, camera at origin)
        cam_tokens = torch.zeros(1, total_views, 11)
        cam_tokens[:, :, 9] = 0.0
        cam_tokens[:, :, 10] = 1.0

        # Create target camera that moves forward (+0.3z) then back.
        # Interpolate smoothly: first half goes 0 → +0.3z, second half returns.
        half_length = video_length // 2
        t_forward = torch.linspace(0, 1, half_length, device=device)
        t_back = torch.linspace(1, 0, video_length - half_length, device=device)
        t_all = torch.cat([t_forward, t_back])  # (video_length,)

        origin_c2w = c2w[:, 0:1, :, :].clone()  # (B, 1, 4, 4) - identity
        forward = origin_c2w[:, :, :3, 2]  # (B, 1, 3)
        origin_pos = origin_c2w[:, :, :3, 3]  # (B, 1, 3)

        # Build target c2w for each frame
        target_c2w = origin_c2w.expand(B, video_length, 4, 4).clone()
        for i, t in enumerate(t_all):
            target_c2w[:, i, :3, 3] = origin_pos + 0.3 * t * forward

        # Use default intrinsics for target ray computation
        ref_fxfycxcy = default_fxfycxcy[:, 0, :]  # (B, 4)
        target_fxfycxcy = ref_fxfycxcy.unsqueeze(1).expand(B, video_length, 4)

        target_rays = compute_plucker_coordinates(target_c2w, target_fxfycxcy, image_size_hw)

    # Conditioning views get zero Plucker rays (model does not use input camera poses)
    cond_rays = torch.zeros(B, num_cond_views, 6, H, W, device=target_rays.device)
    rays = torch.cat([cond_rays, target_rays], dim=1)

    return rays, cam_tokens
