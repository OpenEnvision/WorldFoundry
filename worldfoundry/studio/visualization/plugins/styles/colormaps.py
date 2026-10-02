"""Depth and normal colour maps used by the perception renderer."""

import matplotlib
import numpy as np


def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = "Spectral") -> np.ndarray:
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / max(float(max_depth - min_depth), 1e-8)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal
