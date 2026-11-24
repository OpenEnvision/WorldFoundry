"""PLY export for panorama SHARP inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.ply_io import write_ply

from .color_space import linear_to_srgb
from .types import Gaussians3D


def _rgb_to_spherical_harmonics(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / np.sqrt(1.0 / (4.0 * np.pi))


@torch.no_grad()
def save_panorama_ply(
    gaussians: Gaussians3D,
    image_shape: tuple[int, int],
    output_path: str | Path,
) -> Path:
    """Write SHARP output in the standard degree-zero 3DGS PLY layout."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = gaussians.mean_vectors.flatten(0, 1)
    scales = torch.log(gaussians.singular_values.clamp_min(1e-8)).flatten(0, 1)
    rotations = gaussians.quaternions.flatten(0, 1)
    colors = _rgb_to_spherical_harmonics(linear_to_srgb(gaussians.colors.flatten(0, 1)))
    opacities = gaussians.opacities.flatten(0, 1).clamp(1e-4, 1.0 - 1e-4)
    opacity_logits = torch.log(opacities / (1.0 - opacities)).unsqueeze(-1)
    attributes = torch.cat((positions, colors, opacity_logits, scales, rotations), dim=1)
    names = (
        ["x", "y", "z"]
        + [f"f_dc_{index}" for index in range(3)]
        + ["opacity"]
        + [f"scale_{index}" for index in range(3)]
        + [f"rot_{index}" for index in range(4)]
    )
    values = np.empty(len(positions), dtype=[(name, "f4") for name in names])
    values[:] = list(map(tuple, attributes.cpu().numpy()))
    image_height, image_width = image_shape

    def element(name: str, dtype: str, values: np.ndarray) -> np.ndarray:
        array = np.empty(len(values), dtype=[(name, dtype)])
        array[:] = values
        return array

    focal = float(np.hypot(image_width, image_height)) / 2.0
    metadata = [
        ("extrinsic", element("extrinsic", "f4", np.eye(4, dtype=np.float32).ravel())),
        (
            "intrinsic",
            element(
                "intrinsic",
                "f4",
                np.asarray(
                    [
                        focal,
                        0,
                        image_width * 0.5,
                        0,
                        focal,
                        image_height * 0.5,
                        0,
                        0,
                        1,
                    ],
                    dtype=np.float32,
                ),
            ),
        ),
        ("image_size", element("image_size", "u4", np.asarray([image_width, image_height], dtype=np.uint32))),
        ("frame", element("frame", "i4", np.asarray([1, len(positions)], dtype=np.int32))),
    ]
    radius = torch.linalg.vector_norm(gaussians.mean_vectors[0], dim=-1)
    disparity = 1.0 / radius.clamp_min(1e-4)
    quantiles = (
        torch.quantile(
            disparity,
            torch.tensor([0.1, 0.9], device=disparity.device),
        )
        .float()
        .cpu()
        .numpy()
    )
    metadata.extend(
        [
            ("disparity", element("disparity", "f4", quantiles)),
            ("color_space", element("color_space", "u1", np.asarray([0], dtype=np.uint8))),
            ("version", element("version", "u1", np.asarray([1, 5, 0], dtype=np.uint8))),
        ]
    )
    return write_ply(output_path, [("vertex", values), *metadata])
