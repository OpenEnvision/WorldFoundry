import importlib

from .general_utils import (
    batch_nearest_dot,
    cube_to_pano,
    depth_match,
    fill_mask_from_contour,
    map_image_to_pano,
    pano_to_cube,
    pano_unit_rays,
    resize_img,
    resize_img_and_rays,
)

_SPLAT_EXPORTS = {
    "SplatFile",
    "convert_rgbd_to_gs",
    "mask_splat",
    "merge_splats",
}


def __getattr__(name: str):
    if name in _SPLAT_EXPORTS:
        module = importlib.import_module(f"{__name__}.splat_utils")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SplatFile",
    "batch_nearest_dot",
    "convert_rgbd_to_gs",
    "cube_to_pano",
    "depth_match",
    "fill_mask_from_contour",
    "map_image_to_pano",
    "mask_splat",
    "merge_splats",
    "pano_to_cube",
    "pano_unit_rays",
    "resize_img",
    "resize_img_and_rays",
]
