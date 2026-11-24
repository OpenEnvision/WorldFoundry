from __future__ import annotations

import numpy as np


def get_closest_ratio(height: float, width: float, ratios: list, buckets: list) -> tuple:
    aspect_ratio = float(height) / float(width)
    ratios_array = np.array(ratios)
    closest_ratio_id = np.abs(ratios_array - aspect_ratio).argmin()
    closest_size = buckets[closest_ratio_id]
    closest_ratio = ratios_array[closest_ratio_id]
    return closest_size, closest_ratio


def generate_crop_size_list(
    base_size: int = 256, patch_size: int = 16, max_ratio: float = 4.0
) -> list[tuple[int, int]]:
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list
