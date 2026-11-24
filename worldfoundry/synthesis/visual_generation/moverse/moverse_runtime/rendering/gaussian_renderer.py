"""Minimal Gaussian loading and rendering used by MoVerse inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.point_clouds.ply_io import read_ply_vertex


def _numbered_properties(names: tuple[str, ...], prefix: str) -> list[str]:
    return sorted(
        (name for name in names if name.startswith(prefix)),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )


def _load_ply(path: Path, device: str | torch.device) -> dict[str, torch.Tensor | int]:
    vertex = read_ply_vertex(path)
    names = tuple(vertex.dtype.names or ())
    count = len(vertex)

    def stack(properties: list[str]) -> torch.Tensor:
        values = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in properties], axis=-1)
        return torch.from_numpy(values).to(device=device)

    means = stack(["x", "y", "z"])
    quats = F.normalize(stack(_numbered_properties(names, "rot_")), p=2, dim=-1)
    scales = stack(_numbered_properties(names, "scale_")).exp()
    opacities = torch.from_numpy(np.asarray(vertex["opacity"], dtype=np.float32)).to(device).sigmoid()

    dc = stack(["f_dc_0", "f_dc_1", "f_dc_2"]).unsqueeze(1)
    rest_names = _numbered_properties(names, "f_rest_")
    if rest_names:
        rest = stack(rest_names)
        if rest.shape[1] % 3:
            raise ValueError(f"Invalid spherical-harmonic property count in {path}: {rest.shape[1]}")
        rest = rest.reshape(count, -1, 3)
    else:
        rest = dc.new_zeros((count, 0, 3))
    colors = torch.cat([dc, rest], dim=1)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": int(math.sqrt(colors.shape[1]) - 1),
    }


def _load_checkpoint(path: Path, device: str | torch.device) -> dict[str, torch.Tensor | int]:
    payload = torch.load(path, map_location=device, weights_only=True)
    splats = payload.get("splats", payload)
    means = splats["means"].float().to(device)
    quats = F.normalize(splats["quats"].float().to(device), p=2, dim=-1)
    scales = splats["scales"].float().to(device).exp()
    opacities = splats["opacities"].float().to(device).sigmoid()
    sh0 = splats["sh0"].float().to(device)
    shn = splats.get("shN")
    shn = sh0.new_zeros((*sh0.shape[:-2], 0, 3)) if shn is None else shn.float().to(device)
    colors = torch.cat([sh0, shn], dim=-2)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": int(math.sqrt(colors.shape[-2]) - 1),
    }


def load_gaussians(path: str | Path, device: str | torch.device = "cuda") -> dict:
    """Load activated Gaussian parameters from PLY or a gsplat checkpoint."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return _load_ply(path, device)
    if path.suffix.lower() in {".pt", ".pth"}:
        return _load_checkpoint(path, device)
    raise ValueError(f"Unsupported Gaussian file format: {path.suffix}")


class GaussianRenderer:
    """Small inference-only wrapper around gsplat rasterization."""

    def __init__(
        self,
        *,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        sh_degree: int,
        width: int,
        height: int,
        device: str = "cuda",
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.colors = colors
        self.sh_degree = sh_degree
        self.width = width
        self.height = height
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.background = torch.tensor([bg_color], dtype=torch.float32, device=device)

    def render_batch(self, viewmats: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        from gsplat.rendering import rasterization

        colors, _alphas, _metadata = rasterization(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.colors,
            viewmats,
            intrinsics,
            self.width,
            self.height,
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            backgrounds=self.background.expand(viewmats.shape[0], -1),
            render_mode="RGB",
            packed=False,
        )
        return colors.clamp_(0.0, 1.0)
