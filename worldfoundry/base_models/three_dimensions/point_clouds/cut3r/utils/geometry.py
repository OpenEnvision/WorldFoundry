# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

"""Module for base_models -> three_dimensions -> point_clouds -> cut3r -> utils -> geometry.py functionality."""

import numpy as np
import torch


def xy_grid(
    W,
    H,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:

        def arange(*a, **kw):
            return torch.arange(*a, device=device, **kw)

        meshgrid, stack = torch.meshgrid, torch.stack

        def ones(*a):
            return torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    if isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and Trf.ndim == 3 and pts.ndim == 4:
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def get_med_dist_between_poses(poses):
    """Median pairwise distance between camera centers (DUSt3R convention)."""
    from scipy.spatial.distance import pdist

    centers = [
        pose[:3, 3].detach().cpu().numpy()
        if isinstance(pose, torch.Tensor)
        else np.asarray(pose[:3, 3])
        for pose in poses
    ]
    return np.median(pdist(centers))


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K
