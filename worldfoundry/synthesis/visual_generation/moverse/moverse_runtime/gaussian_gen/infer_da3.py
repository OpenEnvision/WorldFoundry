"""Panoramic depth inference with the in-tree Depth Anything 3 model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from worldfoundry.base_models.three_dimensions.depth.moge.utils.panorama import (
    get_cubemap_cameras,
    get_panorama_cameras,
    merge_cubemap_blended_to_panorama,
    merge_panorama_depth,
    split_panorama_image,
    zdepth_to_distance,
)
from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist.utils3d.numpy.transforms import (
    denormalize_intrinsics,
)

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def load_da3_model(model_dir: str):
    """Load the canonical in-tree DA3 implementation and its checkpoint."""
    import torch

    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
        DepthAnything3,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_dir)
    return model.to(device=device).eval()


def run_da3_inference(
    images: list[np.ndarray],
    extrinsics: np.ndarray,
    normalized_intrinsics: np.ndarray,
    model_dir: str,
    view_resolution: int,
    process_res: int = 504,
) -> np.ndarray:
    """Run pose-conditioned DA3 inference without degenerate pose alignment."""
    model = load_da3_model(model_dir)
    pixel_intrinsics = np.stack(
        [
            denormalize_intrinsics(
                intrinsic,
                (view_resolution, view_resolution),
            )
            for intrinsic in normalized_intrinsics
        ]
    ).astype(np.float32)
    prediction = model.inference(
        images,
        extrinsics=extrinsics.astype(np.float32),
        intrinsics=pixel_intrinsics,
        align_to_input_pose=False,
        process_res=process_res,
        process_res_method="upper_bound_resize",
    )
    return prediction.depth


def infer_panorama_depth(
    image_path: str | Path,
    output_dir: str | Path,
    model_dir: str = DEFAULT_MODEL,
    view_resolution: int = 512,
    fov_deg: float = 90.0,
    process_res: int = 504,
    split_mode: str = "cubemap_overlap",
) -> Path:
    """Estimate an equirectangular depth map and save it as ``depth.npy``."""
    image_path = Path(image_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read panorama: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pano_height, pano_width = image_rgb.shape[:2]

    if split_mode == "icosahedron":
        extrinsics, intrinsics = get_panorama_cameras(fov_deg=90.0)
    elif split_mode == "cubemap_overlap":
        extrinsics, intrinsics = get_cubemap_cameras(fov_deg=max(fov_deg, 100.0))
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    images = split_panorama_image(
        image_rgb,
        extrinsics,
        intrinsics,
        view_resolution,
    )
    depth_maps = run_da3_inference(
        images,
        extrinsics,
        intrinsics,
        model_dir,
        view_resolution,
        process_res,
    )
    distance_maps = [zdepth_to_distance(depth, intrinsics[index]) for index, depth in enumerate(depth_maps)]
    pred_masks = [depth > 0 for depth in depth_maps]

    merge_height = min(1024, pano_height)
    merge_width = merge_height * 2
    if split_mode == "cubemap_overlap":
        panorama_depth, _ = merge_cubemap_blended_to_panorama(
            merge_width,
            merge_height,
            distance_maps,
            pred_masks,
            extrinsics,
            intrinsics,
        )
    else:
        panorama_depth, _ = merge_panorama_depth(
            merge_width,
            merge_height,
            distance_maps,
            pred_masks,
            extrinsics,
            intrinsics,
        )

    if panorama_depth.shape != (pano_height, pano_width):
        panorama_depth = cv2.resize(
            panorama_depth,
            (pano_width, pano_height),
            interpolation=cv2.INTER_LINEAR,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "depth.npy"
    np.save(output_path, panorama_depth)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panoramic depth inference with DA3")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output_dir", default="./infer_results")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL)
    parser.add_argument("--view_resolution", type=int, default=512)
    parser.add_argument("--fov", type=float, default=120.0)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument(
        "--split_mode",
        choices=("cubemap_overlap", "icosahedron"),
        default="cubemap_overlap",
    )
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    output = infer_panorama_depth(
        image_path=args.path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        view_resolution=args.view_resolution,
        fov_deg=args.fov,
        process_res=args.process_res,
        split_mode=args.split_mode,
    )
    print(output)


if __name__ == "__main__":
    main()
