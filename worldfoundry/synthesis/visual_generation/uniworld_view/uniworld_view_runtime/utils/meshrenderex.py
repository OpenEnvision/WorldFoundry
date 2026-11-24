import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .torch_libs import ensure_torch_libs_loaded

ensure_torch_libs_loaded()

# --- PyTorch3D Imports ---
from pytorch3d.renderer import (  # noqa: E402 - Loads after the torch shared libraries.
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    TexturesUV,
    hard_rgb_blend,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments  # noqa: E402 - Loads after the torch shared libraries.
from pytorch3d.structures import Meshes  # noqa: E402 - Loads after the torch shared libraries.


def point_to_mesh_cuda(pts, rgbs, faces, min_angle_deg=2.5):
    h, w = rgbs.shape[:2]
    vertices = pts.reshape(-1, 3)
    # colors = rgbs.reshape(-1, 3)
    masks = torch.ones((h, w, 1), dtype=torch.uint8).to(rgbs.device) * 255
    rgbs = torch.cat([rgbs, masks], axis=-1)
    colors = rgbs.reshape(-1, 4)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = F.normalize(face_normals, dim=1)

    def angle_between(v1, v2):
        cos_theta = torch.sum(v1 * v2, -1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12)
        return torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    a = angle_between(v1 - v0, v2 - v0)
    b = angle_between(v2 - v1, v0 - v1)
    c = angle_between(v0 - v2, v1 - v2)
    min_angles = torch.minimum(torch.minimum(a, b), c)

    # Filter faces based on minimum angle
    valid_faces = min_angles >= min_angle_deg
    invalid_faces = faces[~valid_faces]
    colors.index_put_((invalid_faces,), torch.zeros(4).to(colors.device))

    return vertices, faces, colors


# --- 自定义网格着色器，用于输出深度 ---
class UnlitShaderWithDepth(nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> dict[str, torch.Tensor]:
        texels = meshes.sample_textures(fragments)
        colors = hard_rgb_blend(texels, fragments, self.blend_params)
        depth = fragments.zbuf[..., 0:1]

        return {"color": colors, "depth": depth}


# --- 主功能类 ---
class MeshWarperEx:
    def __init__(self, resolution: tuple | None = None, device: str = "cpu"):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.dtype = torch.float32

    def get_device(self, device: str):
        if torch.cuda.is_available() and "cuda" in device:
            return torch.device(device)
        else:
            return torch.device("cpu")

    def create_mesh_from_depth(
        self, depth: torch.Tensor, intrinsic: torch.Tensor, frame: torch.Tensor, transformation1: torch.Tensor
    ) -> Meshes:
        frame_normalized = (frame + 1.0) / 2.0

        b, _, h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
        x, y = x.float(), y.float()

        fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
        cx, cy = intrinsic[:, 0, 2], intrinsic[:, 1, 2]

        z_cam = depth.squeeze(1)
        x_cam = (x - cx.view(b, 1, 1)) * z_cam / fx.view(b, 1, 1)
        y_cam = (y - cy.view(b, 1, 1)) * z_cam / fy.view(b, 1, 1)

        verts = torch.stack([x_cam, y_cam, z_cam], dim=-1).view(b, -1, 3)
        T_cam1_world = torch.inverse(transformation1)
        verts = torch.bmm(verts, T_cam1_world[:, :3, :3].transpose(1, 2)) + T_cam1_world[:, None, :3, 3]

        # point_to_mesh_cuda(verts, frame_normalized, faces)

        x_coords = torch.arange(w - 1, device=self.device)
        y_coords = torch.arange(h - 1, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        xx, yy = xx.reshape(-1), yy.reshape(-1)
        idx = yy * w + xx

        # Wind triangles so their normals face the camera.
        faces1 = torch.stack([idx, idx + w, idx + 1], dim=1)
        faces2 = torch.stack([idx + 1, idx + w, idx + w + 1], dim=1)
        faces = torch.cat([faces1, faces2], dim=0).unsqueeze(0).repeat(b, 1, 1)

        texture_map = frame_normalized.permute(0, 2, 3, 1)

        verts, faces, texture_map = point_to_mesh_cuda(verts[0], texture_map[0], faces[0])
        verts = verts.unsqueeze(0).repeat(b, 1, 1)
        faces = faces.unsqueeze(0).repeat(b, 1, 1)
        texture_map = texture_map[..., :3].reshape(1, h, w, 3).repeat(b, 1, 1, 1)

        verts_uvs = torch.stack([x.view(-1) / (w - 1), 1.0 - y.view(-1) / (h - 1)], dim=-1).unsqueeze(0).repeat(b, 1, 1)

        textures = TexturesUV(maps=texture_map, faces_uvs=faces.clone(), verts_uvs=verts_uvs)

        return Meshes(verts=verts, faces=faces, textures=textures)

    def forward_warp(
        self,
        frame1: torch.Tensor,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution

        b, _, h, w = frame1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        mesh_world = self.create_mesh_from_depth(depth1, intrinsic1, frame1, transformation1)

        # transformation1 是 world-to-cam1 的变换矩阵，所以 inverse(transformation1) 是 cam1-to-world
        # mesh_world = Meshes(verts=verts_world, faces=mesh.faces_padded(), textures=mesh.textures)

        # transformation2 是 world-to-cam2 的变换矩阵
        R_cv, T_cv = transformation2[:, :3, :3], transformation2[:, :3, 3]

        # 转换到 PyTorch3D 的相机坐标系 (NDC, z朝后)
        R_p3d = R_cv.clone().transpose(1, 2)
        T_p3d = T_cv.clone()
        R_p3d[:, :, 0] *= -1
        R_p3d[:, :, 1] *= -1
        T_p3d[:, 0] *= -1
        T_p3d[:, 1] *= -1

        # prepare camera intrinsics for PyTorch3D
        fx = intrinsic2[:, 0, 0].to(device=self.device, dtype=self.dtype)
        fy = intrinsic2[:, 1, 1].to(device=self.device, dtype=self.dtype)
        cx = intrinsic2[:, 0, 2].to(device=self.device, dtype=self.dtype)
        cy = intrinsic2[:, 1, 2].to(device=self.device, dtype=self.dtype)

        cameras = PerspectiveCameras(
            focal_length=torch.stack([fx, fy], dim=1),
            principal_point=torch.stack([cx, cy], dim=1),
            R=R_p3d,
            T=T_p3d,
            image_size=((h, w),),
            in_ndc=False,
            device=self.device,
        )

        raster_settings = RasterizationSettings(
            image_size=(h, w), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True
        )
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=UnlitShaderWithDepth(device=self.device, cameras=cameras, blend_params=blend_params),
        )

        rendered_output_dict = renderer(mesh_world)
        rendered_color = rendered_output_dict["color"]
        rendered_depth = rendered_output_dict["depth"]

        rendered_output_rescaled = rendered_color * 2.0 - 1.0
        rendered_output_permuted = rendered_output_rescaled.permute(0, 3, 1, 2)

        warped_frame2 = rendered_output_permuted[:, :3, :, :]
        mask2 = (rendered_output_permuted[:, 3:, :, :] > -1.0).float()
        warped_depth2 = rendered_depth.permute(0, 3, 1, 2)

        return warped_frame2, mask2, warped_depth2
