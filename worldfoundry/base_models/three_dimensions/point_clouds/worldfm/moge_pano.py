"""
MoGe panorama depth inference helpers.

Constants and camera-generation utilities extracted from MoGe/moge/scripts/pipeline_pano.py.

External pip dependency: utils3d.
MoGe itself is imported from the canonical WorldFoundry base model package.

This helper intentionally loads ``moge.model.v2`` only, because WorldFM follows the
official MoGe-2 panorama path. MoGe-1 callers should keep using their own entrypoint.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fixed MoGe inference parameters
# ---------------------------------------------------------------------------
RESOLUTION_LEVEL = 30
FOV_DEG = 45.0
NUM_VIEWS = 42
MERGE_MAX_WIDTH = 4096
MERGE_MAX_HEIGHT = 2048

# ---------------------------------------------------------------------------
# Resolution tiers (sorted ascending by width)
# ---------------------------------------------------------------------------
TIERS = [
    {"name": "4k", "width": 4096, "height": 2048, "split_res": 512},
    {"name": "8k", "width": 8192, "height": 4096, "split_res": 1024},
]


def select_tier(img_width: int) -> dict:
    """Select resolution tier based on input image width."""
    if img_width < 6000:
        return TIERS[0]
    return TIERS[1]


# ---------------------------------------------------------------------------
# Panorama camera helpers (ported from infer_panorama_fov.py)
# ---------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    phi = (1 + 5 ** 0.5) / 2
    y = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y ** 2))
    theta = 2 * np.pi * i / phi
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _get_panorama_cameras(n_views: int, fov: float):
    import utils3d
    targets = _fibonacci_sphere(n_views)
    intr = utils3d.numpy.intrinsics_from_fov(
        fov_x=np.deg2rad(fov), fov_y=np.deg2rad(fov),
    ).astype(np.float32)
    extr = utils3d.numpy.extrinsics_look_at(
        [0, 0, 0], targets, [0, 0, 1],
    ).astype(np.float32)
    return extr, [intr] * n_views


MoGeModel = None
get_panorama_cameras = None
split_panorama_image = None
merge_panorama_depth = None


def ensure_moge(path: str | None = None) -> None:
    """
    Import MoGe from the canonical WorldFoundry base model package.

    Args:
        path: Deprecated compatibility argument; external source trees are rejected.
    """
    global MoGeModel, get_panorama_cameras, split_panorama_image, merge_panorama_depth

    if MoGeModel is not None:
        return

    if path is not None:
        raise RuntimeError(
            "WorldFM no longer accepts external MoGe source checkouts at runtime. "
            "Use the WorldFoundry base MoGe model package."
        )

    from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import (
        MoGeModel as _M,
    )
    from worldfoundry.base_models.three_dimensions.depth.moge.utils.panorama import (
        get_panorama_cameras as _gpc,
    )
    from worldfoundry.base_models.three_dimensions.depth.moge.utils.panorama import (
        merge_panorama_depth as _mpd,
    )
    from worldfoundry.base_models.three_dimensions.depth.moge.utils.panorama import (
        split_panorama_image as _spi,
    )

    MoGeModel = _M
    get_panorama_cameras = _gpc
    split_panorama_image = _spi
    merge_panorama_depth = _mpd
