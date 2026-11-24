import numpy as np
import torch

from .torch_libs import ensure_torch_libs_loaded

ensure_torch_libs_loaded()

from pytorch3d.renderer import (  # noqa: E402 - Loads after the torch shared libraries.
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds  # noqa: E402 - Loads after the torch shared libraries.


class PointsZbufRenderer(PointsRenderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, point_clouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf


def render_pcd(pts3d, imgs, masks, views, renderer, device, return_mask=False, return_depth=False):
    imgs_np = imgs.detach().cpu().numpy()
    pts3d_np = pts3d.detach().cpu().numpy()

    if masks is None:
        pts = torch.from_numpy(np.concatenate(list(pts3d_np))).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate(list(imgs_np))).view(-1, 3).to(device)
    else:
        masks_np = masks.detach().cpu().numpy().astype(bool)
        pts = (
            torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d_np, masks_np, strict=False)]))
            .view(-1, 3)
            .to(device)
        )
        col = (
            torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs_np, masks_np, strict=False)]))
            .view(-1, 3)
            .to(device)
        )

    point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
    images = renderer(point_cloud)[0]

    if return_depth:
        fragments = renderer.rasterizer(point_cloud)
        depth_maps = fragments.zbuf[..., 0]
        depth_maps = torch.where(
            depth_maps >= 0,
            depth_maps,
            torch.tensor(float("inf"), device=device),
        )
    else:
        fragments = None
        depth_maps = None

    if return_mask:
        if fragments is None:
            fragments = renderer.rasterizer(point_cloud)
        view_masks = (fragments.zbuf[..., 0] > 0).float()
    else:
        view_masks = None

    return images, view_masks, depth_maps


def run_render(pcd, imgs, masks, H, W, c2ws, K, num_views, return_mask=False, return_depth=False, device="cuda:0"):
    return_depth = True

    c2ws = c2ws.clone()
    c2ws[:, :, 0] = -c2ws[:, :, 0]
    c2ws[:, :, 1] = -c2ws[:, :, 1]
    w2cs = c2ws.inverse()

    background_color = [0, 0, 0]
    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
    principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
    image_shapes = torch.tensor([[H, W]]).repeat(num_views, 1)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=c2ws[:, :3, :3],
        T=w2cs[:, :3, -1],
        in_ndc=False,
        image_size=image_shapes,
        device=device,
    )

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.01,
        points_per_pixel=10,
        bin_size=100,
    )

    renderer = PointsZbufRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return render_pcd(pcd, imgs, masks, num_views, renderer, device, return_mask, return_depth)
