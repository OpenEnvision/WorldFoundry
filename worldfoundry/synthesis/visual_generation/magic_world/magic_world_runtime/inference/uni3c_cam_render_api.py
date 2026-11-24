# uni3c_cam_render.py
import os
import warnings
from collections.abc import Sequence

import einops
import numpy as np
import torch
import trimesh
from diffusers.utils import export_to_video
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, ToTensor

try:
    from pytorch3d.renderer import PointsRasterizationSettings
except ImportError:

    class PointsRasterizationSettings:
        def __init__(self, *, image_size, radius, points_per_pixel) -> None:
            self.image_size = image_size
            self.radius = radius
            self.points_per_pixel = points_per_pixel


from worldfoundry.base_models.three_dimensions.depth import depth_pro
from worldfoundry.base_models.three_dimensions.general_3d.uni3c.pointcloud import point_rendering

warnings.filterwarnings("ignore")


def _to_torch_image_0_1(img: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    """
    Convert a PIL image, NumPy array or tensor to an RGB PIL.Image.
    """
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, np.ndarray):
        # HxW or HxWxC with C >= 3, uint8 or float in [0, 1]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        pil = Image.fromarray(img[:, :, :3], mode="RGB")
    elif isinstance(img, torch.Tensor):
        # CHW or HWC, range [0, 1] or [0, 255]
        ten = img.detach().cpu()
        if ten.ndim == 3 and ten.shape[0] in (1, 3):  # CHW
            ten = ten.permute(1, 2, 0)
        if ten.max() <= 1.0:
            ten = (ten * 255.0).round().clamp(0, 255).to(torch.uint8)
        else:
            ten = ten.clamp(0, 255).to(torch.uint8)
        npimg = ten.numpy()
        pil = Image.fromarray(npimg[:, :, :3], mode="RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    return pil


def _camera_from_param(cam_param: Sequence[float]) -> torch.Tensor:
    """
    cam_param:
      - length 25: K (3x3) in the first 9 values, then the flattened w2c (4x4) in the last 16.
      - length 19: the flattened w2c (3x4) in the last 12 values.
    Returns the w2c matrix as a torch.Tensor of shape (4, 4).
    """
    n = len(cam_param)
    if n == 25:
        w2c_flat = np.array(cam_param[9:], dtype=np.float32)  # 16
        w2c = w2c_flat.reshape(4, 4)
        return torch.from_numpy(w2c)
    elif n == 19:
        w2c_flat = np.array(cam_param[-12:], dtype=np.float32)  # 12
        w2c_3x4 = w2c_flat.reshape(3, 4)
        # Extend to a 4x4 homogeneous matrix.
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        w2c_4x4 = np.vstack([w2c_3x4, bottom])
        return torch.from_numpy(w2c_4x4)
    else:
        raise ValueError(f"Unsupported cam_param length: {n}. Expected 25 or 19.")


def _points_padding_torch(points2d_xy: torch.Tensor) -> torch.Tensor:
    """
    Map (x, y) of shape [H,W,2] to homogeneous (x, y, 1) of shape [H,W,3] (torch).
    """
    ones = torch.ones_like(points2d_xy[..., :1])
    return torch.cat([points2d_xy, ones], dim=-1)


def _points_padding_numpy(points3d_hw3: np.ndarray) -> np.ndarray:
    """
    Map [N,3] points to homogeneous coordinates (x, y, z, 1) of shape [N,4] (NumPy).
    """
    ones = np.ones((points3d_hw3.shape[0], 1), dtype=points3d_hw3.dtype)
    return np.concatenate([points3d_hw3, ones], axis=1)


_MODELS = {}


def get_models(device):
    if device not in _MODELS:
        print("[cam_render] Init depth model (singleton)")
        depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
        depth_model = depth_model.eval()  # Kept in FP32 for stability
        _MODELS[device] = (depth_model, depth_transform)
    return _MODELS[device]


@torch.no_grad()
def render_from_image_and_traj(
    reference_image: Image.Image | np.ndarray | torch.Tensor,
    traj_params: list[Sequence[float]] | np.ndarray,
    output_path: str,
    *,
    traj_type: str = "custom",
    nframe: int = 199,
    d_r: float = 1.0,
    d_theta: float = 0.0,
    d_phi: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    focal_length: float = 1.0,
    start_elevation: float = 5.0,
    device: str = "cuda",
    depth_radius: float = 0.008,
    ppp: int = 8,
    fps: int = 30,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """
    Render the point cloud of a reference image along a camera trajectory (the cam_render.py pipeline).

    - reference_image: image object (PIL, NumPy or tensor)
    - traj_params: trajectory, a list of 25 floats per frame or numpy[N,25]
    - the other parameters match the cam_render.py command line

    Writes output_path/render.mp4, output_path/render_mask.mp4 and output_path/pcd.ply and returns the
    rendered frames and mask frames.
    """
    os.makedirs(output_path, exist_ok=True)

    # === 1) Preprocess the image ===
    image = _to_torch_image_0_1(reference_image)
    image = ImageOps.exif_transpose(image)
    w_origin, h_origin = image.size

    # Resolution is fixed to hw_list[7] (480x832).
    hw_list = [[480, 768], [512, 720], [608, 608], [720, 512], [768, 480], [720, 1280], [512, 896], [480, 832]]
    height, width = hw_list[7]
    print(f"[cam_render] Resolution: {h_origin}x{w_origin} -> {height}x{width}")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    # === 2) Trajectory ===
    if isinstance(traj_params, np.ndarray):
        cam_params = traj_params.tolist()
    else:
        cam_params = [list(map(float, fr)) for fr in traj_params]

    if nframe is None or nframe <= 0:
        nframe = len(cam_params)
    else:
        # Truncate the trajectory to nframe.
        cam_params = cam_params[:nframe]

    w2cs = [_camera_from_param(p) for p in cam_params]
    w2cs = torch.stack(w2cs, dim=0)  # [F,4,4]
    c2ws = w2cs.inverse()  # [F,4,4]

    # === 3) Depth estimation ===
    depth_model, depth_transform = get_models(device)

    depth_image_np = np.array(image)
    depth_image = depth_transform(depth_image_np)  # depth_pro transform
    prediction = depth_model.infer(depth_image, f_px=None)
    depth = prediction["depth"]  # [H,W] in meters
    depth = depth[None, None]  # [1,1,H,W]

    focallength_px = prediction["focallength_px"].item()
    K = torch.tensor([[focallength_px, 0, width / 2], [0, focallength_px, height / 2], [0, 0, 1]], dtype=torch.float32)
    K_inv = K.inverse()
    intrinsic = K[None].repeat(nframe, 1, 1)  # [F,3,3]

    # === 4) Point cloud in the world frame of camera 0 ===
    # Pixel grid
    xs = torch.arange(width, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32)
    points2d = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)  # [H,W,2]
    points3d_cam = _points_padding_torch(points2d).reshape(height * width, 3)  # [HW,3] (x,y,1)
    points3d_cam = (K_inv @ points3d_cam.T * depth.reshape(1, height * width).cpu()).T  # -> [HW,3]

    colors = ((depth_image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(height * width, 3)

    c2w_0 = c2ws[0].to(torch.float32)
    # Homogeneous [HW,4] points multiplied by c2w_0[:3,:]
    points3d_world = (c2w_0[:3, :] @ _points_padding_numpy(points3d_cam.cpu().numpy()).T).T  # [HW,3]
    pcd = trimesh.PointCloud(vertices=points3d_world, colors=colors.cpu().numpy())
    _ = pcd.export(os.path.join(output_path, "pcd.ply"))

    # === 5) Render the sequence ===
    # with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.startswith("cuda"))):
    with torch.no_grad():  # noqa: SIM117 - Branches carry per-case comments.
        # Disable autocast so rendering stays in FP32.
        with torch.autocast("cuda", enabled=False):
            control_imgs, render_masks = point_rendering(
                K=intrinsic.float(),
                w2cs=w2cs.float(),
                depth=depth.float(),
                image=ToTensor()(image)[None].float() * 2 - 1,  # [-1,1]
                raster_settings=PointsRasterizationSettings(
                    image_size=(height, width), radius=depth_radius, points_per_pixel=ppp
                ),
                device=device,
                background_color=[0, 0, 0],
                sobel_threshold=0.35,
                sam_mask=None,
            )

    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=nframe)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=nframe)

    render_video, mask_video = [], []
    control_imgs = control_imgs.to(torch.float32)
    to_pil = ToPILImage()
    for i in range(nframe):
        img = to_pil((control_imgs[0][:, i] + 1) / 2)  # [0,1]
        render_video.append(img)
        mask = to_pil(render_masks[0][:, i])
        mask_video.append(mask)

    export_to_video(render_video, os.path.join(output_path, "render.mp4"), fps=fps)
    export_to_video(mask_video, os.path.join(output_path, "render_mask.mp4"), fps=fps)

    print(f"[cam_render] Done. Saved to: {output_path}")

    return render_video, mask_video
