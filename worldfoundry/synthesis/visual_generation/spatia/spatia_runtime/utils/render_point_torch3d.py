import logging

import torch
try:
    from pytorch3d.ops import sample_farthest_points
    from pytorch3d.renderer import (
        AlphaCompositor,
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
    )
    from pytorch3d.structures import Pointclouds
except (ImportError, OSError) as exc:
    # Some otherwise valid environments ship a PyTorch3D extension compiled
    # against a different torch ABI. The basic point splat below remains usable.
    sample_farthest_points = None
    PointsRenderer = None
    _pytorch3d_import_error = exc


logger = logging.getLogger(__name__)


def _render_torch_zbuffer(
    points_xyz_rgb: torch.Tensor,
    w2c_matrices: torch.Tensor,
    intrinsics: list | tuple,
    image_size: tuple[int, int],
    radius: float,
    max_points: int | None,
) -> torch.Tensor:
    """Render OpenCV-camera point splats when the PyTorch3D extension cannot load."""
    height, width = image_size
    points = points_xyz_rgb[:, :3]
    colors = points_xyz_rgb[:, 3:6].clamp(0, 1)
    if max_points is not None and max_points < len(points):
        keep = torch.linspace(0, len(points) - 1, max_points, device=points.device).long()
        points, colors = points[keep], colors[keep]
    points_h = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    splat_radius = max(1, int(round(float(radius) * min(height, width) / 2)))
    offset = torch.arange(-splat_radius, splat_radius + 1, device=points.device)
    delta_y, delta_x = torch.meshgrid(offset, offset, indexing="ij")
    disk = delta_x.square() + delta_y.square() <= splat_radius * splat_radius
    delta_x, delta_y = delta_x[disk], delta_y[disk]
    outputs = []
    for camera in w2c_matrices:
        cam = points_h @ camera[:3].T
        depth = cam[:, 2]
        visible = torch.isfinite(cam).all(dim=1) & (depth > 1e-4)
        if not bool(visible.any()):
            outputs.append(torch.zeros((height, width, 3), device=points.device, dtype=colors.dtype))
            continue
        cam, depth = cam[visible], depth[visible]
        visible_colors = colors[visible]
        x = torch.round(fx * cam[:, 0] / depth + cx).long()
        y = torch.round(fy * cam[:, 1] / depth + cy).long()
        x = x[:, None] + delta_x[None, :]
        y = y[:, None] + delta_y[None, :]
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        pixel = (y * width + x)[inside]
        if not pixel.numel():
            outputs.append(torch.zeros((height, width, 3), device=points.device, dtype=colors.dtype))
            continue
        point_index = torch.arange(len(cam), device=points.device)[:, None].expand_as(x)[inside]
        point_depth = depth[:, None].expand_as(x)[inside]
        nearest_depth = torch.full((height * width,), float("inf"), device=points.device)
        nearest_depth.scatter_reduce_(0, pixel, point_depth, reduce="amin", include_self=True)
        front = point_depth <= nearest_depth[pixel] + 1e-5
        nearest_point = torch.full((height * width,), len(cam), device=points.device, dtype=torch.long)
        nearest_point.scatter_reduce_(0, pixel[front], point_index[front], reduce="amin", include_self=True)
        image = torch.zeros((height * width, 3), device=points.device, dtype=colors.dtype)
        filled = nearest_point < len(cam)
        image[filled] = visible_colors[nearest_point[filled]]
        outputs.append(image.reshape(height, width, 3))
    return torch.stack(outputs)


def opencv_to_pytorch3d_transform(w2c_opencv):
    """
    Convert OpenCV world-to-camera matrices to PyTorch3D camera parameters.
    """
    c2w_opencv = torch.inverse(w2c_opencv)
    R_opencv = c2w_opencv[:, :3, :3]
    T_opencv = c2w_opencv[:, :3, 3]

    coord_transform = torch.tensor(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        dtype=w2c_opencv.dtype,
        device=w2c_opencv.device,
    )

    R_pytorch3d = torch.bmm(
        torch.bmm(coord_transform.unsqueeze(0).expand(R_opencv.shape[0], -1, -1), R_opencv),
        coord_transform.T.unsqueeze(0).expand(R_opencv.shape[0], -1, -1),
    )
    T_pytorch3d = torch.bmm(
        coord_transform.unsqueeze(0).expand(T_opencv.shape[0], -1, -1),
        T_opencv.unsqueeze(-1),
    ).squeeze(-1)

    c2w_pytorch3d = torch.eye(4, device=w2c_opencv.device, dtype=w2c_opencv.dtype)[None].repeat(R_pytorch3d.shape[0], 1, 1)
    c2w_pytorch3d[:, :3, :3] = R_pytorch3d
    c2w_pytorch3d[:, :3, 3] = T_pytorch3d
    w2c_pytorch3d = torch.inverse(c2w_pytorch3d)
    return R_pytorch3d, w2c_pytorch3d[:, :3, 3]


@torch.inference_mode()
def render_multi_view_pointcloud(
    points_xyz_rgb: torch.Tensor,
    w2c_matrices: torch.Tensor,
    normalized_intrinsics: list | tuple,
    image_size: tuple,
    point_radius: float = 0.01,
    batch_size: int = 64,
    max_retry_attempts: int = 3,
    max_points_to_render: int = None,
    verbose: bool = False,
    downsample_points: bool = False,
    voxel_size: float = 0.01,
) -> torch.Tensor:
    """
    Render a point cloud from multiple views with PyTorch3D.
    """
    del downsample_points, voxel_size

    if w2c_matrices.shape[1] == 3:
        padding_row = torch.tensor([0, 0, 0, 1], device=w2c_matrices.device, dtype=w2c_matrices.dtype).reshape(1, 1, 4)
        w2c_matrices = torch.cat([w2c_matrices, padding_row.repeat(w2c_matrices.shape[0], 1, 1)], dim=1)
    elif w2c_matrices.shape[1] != 4:
        raise ValueError(f"Camera matrix shape is {w2c_matrices.shape}, expected (3,4) or (4,4)")

    num_views = w2c_matrices.shape[0]
    H, W = image_size
    points_xyz = points_xyz_rgb[:, :3]
    points_rgb = points_xyz_rgb[:, 3:6]

    if PointsRenderer is None:
        logger.warning("PyTorch3D unavailable (%s); using Torch point z-buffer", _pytorch3d_import_error)
        return _render_torch_zbuffer(
            points_xyz_rgb,
            w2c_matrices,
            normalized_intrinsics,
            image_size,
            point_radius,
            max_points_to_render,
        )

    if max_points_to_render is not None and max_points_to_render < points_xyz.shape[0]:
        sampled_points, sampled_indices = sample_farthest_points(points_xyz.unsqueeze(0), K=max_points_to_render)
        points_xyz = sampled_points.squeeze(0)
        points_rgb = points_rgb[sampled_indices.squeeze(0)]

    if verbose:
        logger.info("Converting coordinates from OpenCV to PyTorch3D.")

    R, T = opencv_to_pytorch3d_transform(w2c_matrices)

    coord_transform = torch.tensor(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        dtype=points_xyz.dtype,
        device=points_xyz.device,
    )
    points_xyz_pytorch3d = torch.mm(points_xyz, coord_transform.T)

    fx, fy, cx, cy = normalized_intrinsics
    s = min(H, W)
    fx_norm = fx * 2.0 / (s - 1)
    fy_norm = fy * 2.0 / (s - 1)
    cx_norm = (cx - (W - 1) / 2) * 2.0 / (s - 1)
    cy_norm = (cy - (H - 1) / 2) * 2.0 / (s - 1)

    focal_length = torch.tensor([[fx_norm, fy_norm]], device=points_xyz.device).repeat(num_views, 1)
    principal_point = torch.tensor([[cx_norm, cy_norm]], device=points_xyz.device).repeat(num_views, 1)

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=point_radius,
        points_per_pixel=5,
        bin_size=None,
        max_points_per_bin=None,
    )

    rendered_images_list = []
    current_batch_size = min(batch_size, num_views)
    retry_count = 0

    for start_idx in range(0, num_views, current_batch_size):
        end_idx = min(start_idx + current_batch_size, num_views)
        batch_num_views = end_idx - start_idx

        batch_cameras = PerspectiveCameras(
            focal_length=focal_length[start_idx:end_idx],
            principal_point=principal_point[start_idx:end_idx],
            R=R[start_idx:end_idx],
            T=T[start_idx:end_idx],
            image_size=torch.tensor([[H, W]], device=points_xyz.device).repeat(batch_num_views, 1),
            device=points_xyz.device,
        )

        batch_points_list = [points_xyz_pytorch3d for _ in range(batch_num_views)]
        batch_features_list = [points_rgb for _ in range(batch_num_views)]
        batch_point_clouds = Pointclouds(points=batch_points_list, features=batch_features_list).to(points_xyz.device)

        batch_renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=batch_cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0)),
        )

        success = False
        while not success and retry_count < max_retry_attempts:
            try:
                rendered_images_list.append(batch_renderer(batch_point_clouds, cameras=batch_cameras))
                success = True
                retry_count = 0
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                retry_count += 1
                if retry_count >= max_retry_attempts:
                    current_batch_size = max(1, current_batch_size // 2)
                    retry_count = 0
                    if current_batch_size == 1:
                        raise RuntimeError(f"Rendering failed even with batch size 1: {exc}") from exc
                    continue
            except Exception as exc:
                logger.error(f"Rendering failed: {exc}")
                raise

        del batch_point_clouds, batch_cameras, batch_renderer
        torch.cuda.empty_cache()

    rendered_images = torch.cat(rendered_images_list, dim=0)
    rendered_images_rgb = torch.clamp(rendered_images[..., :3], 0.0, 1.0)

    if verbose:
        logger.info(f"Rendering finished. Output shape: {rendered_images_rgb.shape}")

    return rendered_images_rgb
