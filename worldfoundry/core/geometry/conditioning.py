# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera-conditioning geometry shared by SANA-WM and other pose-aware DiTs.

SANA's camera-control path (and any recipe that packs poses into AdaLN /
Plücker tokens) needs the same operations: sample random camera-to-world
matrices, invert SE(3) poses cheaply, unproject pixels into a 6-D raymap,
and pack pixel-rate poses into latent-rate tensors.

This module is the core home for that plumbing so
``base_models/.../cam_utils.py`` does not own a second copy:

- :func:`generate_random_c2w_poses` / :func:`random_rotation_matrix_quaternion`
  — uniform SO(3) via unit quaternions plus bounded translation.
- :func:`get_pose_inverse` — batched ``[R|t]`` inverse using ``R^T``.
- :func:`compute_raymap` — OpenCV unprojection to Plücker ``(d, m)`` or
  ``(o, d)`` at an arbitrary resolution.
- :func:`pack_spatiotemporal_camera_conditioning` — relative-to-first-frame
  poses plus a dense chunked Plücker volume at latent stride.

Pixel-center convention here is integer pixel indices (no ``+0.5``),
matching the original SANA cam_utils. For RealEstate10K-style Plücker
with half-pixel centers see :mod:`worldfoundry.core.geometry.pose`.
"""

import torch

# ──────────────────────────────────────────────────────────────────────────
# Random SE(3) — unit-quaternion SO(3) plus bounded translation
# ──────────────────────────────────────────────────────────────────────────


def generate_random_c2w_poses(N, max_translation_range=10.0, dtype=torch.float32, device="cpu"):
    """
    Generates N random 4x4 Camera-to-World (c2w) homogeneous transformation matrices.

    The rotation (R) is generated using unit quaternions for uniform sampling
    of the 3D rotation space.
    The translation (t) is generated randomly within a specified range.

    Args:
        N (int): The number of poses to generate.
        max_translation_range (float): The maximum absolute value for the
                                       x, y, and z translation components.
        dtype (torch.dtype): Data type of the output tensor.
        device (torch.device or str): Device for the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) containing the c2w poses.
    """

    # 1. Generate N random unit quaternions for Rotation (R)

    # Generate N random quaternion components (N, 4)
    q = torch.randn(N, 4, dtype=dtype, device=device)

    # Normalize to get N unit quaternions (q / ||q||)
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)

    # Extract components
    a, b, c, d = q.unbind(dim=1)  # a, b, c, d are now (N,) tensors

    # Pre-calculate squared terms
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d

    # Pre-calculate double products
    bc, bd, cd = b * c, b * d, c * d
    ad, ac, ab = a * d, a * c, a * b

    # Construct the (N, 3, 3) rotation matrix batch from quaternions
    #
    R_batch = torch.stack(
        [
            torch.stack([a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac)], dim=1),
            torch.stack([2 * (bc + ad), a2 - b2 + c2 - d2, 2 * (cd - ab)], dim=1),
            torch.stack([2 * (bd - ac), 2 * (cd + ab), a2 - b2 - c2 + d2], dim=1),
        ],
        dim=1,
    )  # (N, 3, 3)

    # 2. Generate N random translation vectors (t)

    # Generate N random numbers for t_x, t_y, t_z in [-range, +range]
    # torch.rand(N, 3) generates uniform random numbers in [0, 1)
    t_batch = (torch.rand(N, 3, dtype=dtype, device=device) * 2 * max_translation_range) - max_translation_range
    # t_batch is now (N, 3)

    # 3. Assemble the (N, 4, 4) homogeneous poses

    # Create the base (N, 4, 4) tensor (identity matrix padded)
    poses = torch.eye(4, dtype=dtype, device=device).repeat(N, 1, 1)

    # Insert the rotation R_batch
    poses[:, :3, :3] = R_batch

    # Insert the translation t_batch
    poses[:, :3, 3] = t_batch

    return poses


def random_rotation_matrix_quaternion(dtype=torch.float32, device="cpu"):
    """Return one uniform random 3x3 rotation from a unit quaternion.

    Sampling four i.i.d. Gaussians and normalizing yields a uniform
    distribution on SO(3). Prefer
    :func:`generate_random_c2w_poses` when a batched 4x4 pose is needed.
    """
    # 1. Generate four random numbers (components of a random quaternion)
    q = torch.randn(4, dtype=dtype, device=device)

    # 2. Normalize to get a unit quaternion
    q = q / torch.linalg.norm(q)

    # Extract components
    a, b, c, d = q[0], q[1], q[2], q[3]

    # 3. Convert unit quaternion to 3x3 rotation matrix
    # Based on the standard quaternion-to-matrix formula
    R = torch.tensor(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ],
        dtype=dtype,
        device=device,
    )

    # The sum of squared components (a*a + b*b + c*c + d*d) is 1.0,
    # so we don't need to divide the matrix by the norm, R is already correct.

    return R


# ──────────────────────────────────────────────────────────────────────────
# Pose inverse, Plücker raymap, latent-rate packing
# ──────────────────────────────────────────────────────────────────────────


def get_pose_inverse(T):
    """Invert batched 4x4 camera poses using ``R^{-1} = R^T``.

    Faster and more stable than ``torch.linalg.inv`` for rigid transforms.
    Leading batch dimensions are preserved.

    Args:
        T: ``(..., 4, 4)`` homogeneous matrices with orthonormal ``R``.

    Returns:
        ``(..., 4, 4)`` inverses on the same device/dtype as *T*.
    """
    # Extract R and t
    R = T[..., :3, :3]  # (..., 3, 3)
    t = T[..., :3, 3]  # (..., 3)

    # Compute R_inv = R.T
    R_inv = R.transpose(-1, -2)  # (..., 3, 3)

    # Compute t_inv = -R_inv @ t
    # torch.matmul handles the batch dimension (...)
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # Construct the inverse matrix T_inv
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device).repeat(T.shape[:-2] + (1, 1))
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv

    return T_inv


def compute_raymap(intrinsics, poses, H, W, use_plucker=True):
    """
    Computes a geometry raymap (directions/moments or origins/directions).

    Args:
        intrinsics: (T, 4) tensor [fx, fy, cx, cy]
        poses: (T, 4, 4) tensor [Camera-to-World]
        H, W: int, spatial resolution of the raymap
        use_plucker: bool, if True returns Plucker coords (d, m),
                     else returns (o, d).
    Returns:
        raymap: (T, H, W, 6) tensor
    """
    T = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype

    # 1. Create Pixel Grid (T, H, W)
    # indexing='ij' -> y (rows), x (cols)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x_grid = x_grid[None, ...].expand(T, -1, -1)
    y_grid = y_grid[None, ...].expand(T, -1, -1)

    # 2. Parse Intrinsics (T, 1, 1)
    fx = intrinsics[:, 0].view(T, 1, 1)
    fy = intrinsics[:, 1].view(T, 1, 1)
    cx = intrinsics[:, 2].view(T, 1, 1)
    cy = intrinsics[:, 3].view(T, 1, 1)

    # 3. Unproject to Camera Frame Directions
    # OpenCV convention: +Z forward, +X right, +Y down
    x_cam = (x_grid - cx) / fx
    y_cam = (y_grid - cy) / fy
    z_cam = torch.ones_like(x_cam)

    # Stack to (T, H, W, 3)
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # 4. Transform to World Frame
    # R: (T, 3, 3), t: (T, 3)
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]

    # Rotate: d_world = R @ d_cam
    # einsum: t=batch, i=row, j=col, h=height, w=width
    dirs_world = torch.einsum("tij,thwj->thwi", R, dirs_cam)

    # Normalize Direction vectors
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

    # 5. Prepare Origins
    # Expand translation t to (T, H, W, 3)
    origins = t.view(T, 1, 1, 3).expand_as(dirs_world)

    if use_plucker:
        # Plucker Moments: m = o x d
        moments = torch.cross(origins, dirs_world, dim=-1)
        # Return (Direction, Moment) -> 6 channels
        return torch.cat([dirs_world, moments], dim=-1)
    else:
        # Standard Ray: (Origin, Direction) -> 6 channels
        return torch.cat([origins, dirs_world], dim=-1)


def _normalize_poses_identity_unit_distance(
    in_c2ws: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
):
    """Rewrite poses so *ref0* is identity and *ref1* is unit distance away.

    Used as a numerically-stable canonicalization before packing camera
    tokens. Distances below ``1e-2`` are left unscaled so two coincident
    cameras do not explode the translation.
    """

    ref0_c2w = in_c2ws[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(ref0_c2w), in_c2ws)

    ref1_c2w = c2ws[ref1_idx]
    dist = torch.linalg.norm(ref1_c2w[:3, 3] - ref0_c2w[:3, 3])
    if dist > 1e-2:  # numerically stable
        c2ws[:, :3, 3] /= dist

    return c2ws


def pack_spatiotemporal_camera_conditioning(
    camera_to_world,
    intrinsics,
    *,
    image_height: int,
    image_width: int,
    temporal_stride: int = 8,
    spatial_stride: int = 32,
):
    """Pack pixel-rate poses into latent-rate camera and Pluecker tensors.

    The function is model-independent geometry plumbing.  It emits a compact
    ``[F_latent, 20]`` pose/intrinsics tensor plus a dense
    ``[6 * temporal_stride, F_latent, H_latent, W_latent]`` ray tensor.
    """

    poses = torch.as_tensor(camera_to_world, dtype=torch.float32)
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError("camera conditioning currently requires batch size 1")
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must be [F,4,4], got {tuple(poses.shape)}")
    values = torch.as_tensor(intrinsics, dtype=torch.float32)
    if values.ndim == 3 and values.shape[-2:] == (3, 3):
        values = torch.stack(
            (values[:, 0, 0], values[:, 1, 1], values[:, 0, 2], values[:, 1, 2]),
            dim=-1,
        )
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError("camera intrinsics currently require batch size 1")
        values = values[0]
    if values.ndim == 1 and values.numel() == 4:
        values = values[None].expand(poses.shape[0], -1).clone()
    if values.shape != (poses.shape[0], 4):
        raise ValueError(
            f"intrinsics must be [F,4] or [F,3,3], got {tuple(values.shape)} for {poses.shape[0]} frames"
        )
    if min(image_height, image_width, temporal_stride, spatial_stride) <= 0:
        raise ValueError("camera conditioning geometry must be positive")

    first_inverse = get_pose_inverse(poses[:1])[0]
    poses = torch.matmul(first_inverse, poses)
    latent_height = int(image_height) // int(spatial_stride)
    latent_width = int(image_width) // int(spatial_stride)
    latent_intrinsics = values.clone()
    latent_intrinsics[:, (0, 2)] *= latent_width / float(image_width)
    latent_intrinsics[:, (1, 3)] *= latent_height / float(image_height)
    indices = torch.arange(0, poses.shape[0], int(temporal_stride))
    raymap = torch.cat((poses[indices].reshape(len(indices), 16), latent_intrinsics[indices]), dim=-1)

    chunks = []
    for index in indices:
        start = max(0, int(index) - (int(temporal_stride) - 1))
        stop = start + int(temporal_stride)
        chunk_poses = poses[start:stop]
        chunk_intrinsics = latent_intrinsics[start:stop]
        if len(chunk_poses) < temporal_stride:
            missing = int(temporal_stride) - len(chunk_poses)
            chunk_poses = torch.cat((chunk_poses, chunk_poses[-1:].expand(missing, -1, -1)))
            chunk_intrinsics = torch.cat(
                (chunk_intrinsics, chunk_intrinsics[-1:].expand(missing, -1))
            )
        plucker = compute_raymap(
            chunk_intrinsics,
            chunk_poses,
            latent_height,
            latent_width,
            use_plucker=True,
        )
        chunks.append(plucker.permute(0, 3, 1, 2).reshape(-1, latent_height, latent_width))
    return {
        "camera_conditions": raymap,
        "chunk_plucker": torch.stack(chunks).permute(1, 0, 2, 3),
    }
