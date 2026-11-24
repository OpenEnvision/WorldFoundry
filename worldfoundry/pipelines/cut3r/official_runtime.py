"""Official Runtime visual generation pipeline module."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.cut3r import (
    ARCroco3DStereo,
    estimate_focal_knowing_depth,
    geotrf,
    inference,
    load_images,
    pose_encoding_to_camera,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_image_path(path: str | Path) -> bool:
    """Check if the given value represents a valid image file path."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _is_video_path(path: str | Path) -> bool:
    """Check if the given value represents a valid video file path."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def parse_seq_path(input_source: str | Path | Sequence[str | Path]) -> tuple[list[str], str | None]:
    """Resolve a CUT3R input source to a list of image paths.

    Accepts a list or tuple of paths or a directory of image files, both returned sorted; an image file; a
    ``.txt`` file with one path per line; or a video file whose frames are extracted to a temporary
    directory.

    Returns:
        ``(image_paths, temp_dir)``; ``temp_dir`` is the extracted-frame directory for videos, else None.

    Raises:
        ValueError: The source is unsupported or the video cannot be read.
    """
    if isinstance(input_source, (list, tuple)):
        return sorted(str(Path(p)) for p in input_source), None

    source = Path(cast("str | Path", input_source))
    if source.is_dir():
        return sorted(str(p) for p in source.iterdir() if p.is_file() and _is_image_path(p)), None
    if source.is_file() and _is_image_path(source):
        return [str(source)], None
    if source.is_file() and source.suffix.lower() == ".txt":
        paths = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        return paths, None
    if source.is_file() and _is_video_path(source):
        return _extract_video_frames(source)
    raise ValueError(f"Unsupported CUT3R input source: {input_source}")


def _extract_video_frames(video_path: Path) -> tuple[list[str], str]:
    """Extract individual frames from a video file."""
    # Initialize video capture stream to extract frame data and FPS properties
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps == 0:
        capture.release()
        raise ValueError(f"Error: Video FPS is 0 for {video_path}")

    frame_indices = list(range(0, total_frames, 1))
    temp_dir = tempfile.mkdtemp()
    img_paths: list[str] = []
    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        frame_path = os.path.join(temp_dir, f"frame_{index}.jpg")
        cv2.imwrite(frame_path, frame)
        img_paths.append(frame_path)
    capture.release()
    return img_paths, temp_dir


def prepare_input(
    img_paths: Sequence[str],
    img_mask: Sequence[bool],
    size: int,
    raymaps: Sequence[torch.Tensor] | None = None,
    raymap_mask: Sequence[bool] | None = None,
    revisit: int = 1,
    update: bool = True,
) -> list[dict[str, Any]]:
    """Build CUT3R view dicts from images and optional ray maps.

    Without ray maps every view is an image view with a NaN ray map. With ``raymaps`` and
    ``raymap_mask``, ``img_mask`` and ``raymap_mask`` select, per combined view, which image or ray map
    is consumed. With ``revisit > 1`` the views are repeated ``revisit`` times; when ``update`` is False
    the repeats do not update the state.

    Raises:
        ValueError: Only one of ``raymaps`` and ``raymap_mask`` is given, or the masks do not match the
            combined view count.
    """
    images = load_images(list(img_paths), size=size)
    views: list[dict[str, Any]] = []

    if raymaps is None and raymap_mask is None:
        for i, image in enumerate(images):
            view = {
                "img": image["img"],
                "ray_map": torch.full(
                    (
                        image["img"].shape[0],
                        6,
                        image["img"].shape[-2],
                        image["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        if raymaps is None or raymap_mask is None:
            raise ValueError("raymaps and raymap_mask must be provided together.")
        num_views = len(images) + len(raymaps)
        if len(img_mask) != num_views or len(raymap_mask) != num_views:
            raise ValueError("img_mask and raymap_mask must match the combined view length.")

        image_index = 0
        ray_index = 0
        for i in range(num_views):
            view = {
                "img": (images[image_index]["img"] if img_mask[i] else torch.full_like(images[0]["img"], torch.nan)),
                "ray_map": (raymaps[ray_index] if raymap_mask[i] else torch.full_like(raymaps[0], torch.nan)),
                "true_shape": (
                    torch.from_numpy(images[image_index]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.array([raymaps[ray_index].shape[1:-1][::-1]], dtype=np.int32))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                image_index += 1
            if raymap_mask[i]:
                ray_index += 1
            views.append(view)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(
    outputs: dict[str, Any],
    outdir: str | Path,
    *,
    revisit: int = 1,
    use_pose: bool = True,
) -> dict[str, Any]:
    """Convert CUT3R predictions of the last revisit pass to points, colors and cameras and save per-frame files.

    Writes ``depth/`` and ``conf/`` ``.npy`` files, ``color/`` PNGs and ``camera/`` ``.npz`` files (pose,
    intrinsics) under ``outdir``. With ``use_pose`` the points are the self-view points transformed by
    the predicted camera poses.

    Returns:
        Dict with ``points``, ``colors``, ``confidence``, ``camera`` (focal, principal point, R, t) and
        ``num_frames``.
    """
    outdir = Path(outdir)
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    pr_poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in outputs["pred"]]
    r_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self, strict=False):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    batch_size, height, width, _ = pts3ds_self.shape
    pp = torch.tensor([width // 2, height // 2], device=pts3ds_self.device).float().repeat(batch_size, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": r_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    depths_tosave = pts3ds_self[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)
    conf_self_tosave = torch.cat(conf_self)
    conf_other_tosave = torch.cat(conf_other)
    colors_tosave = torch.cat(colors)
    cam2world_tosave = torch.cat(pr_poses)
    intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    for name in ["depth", "conf", "color", "camera"]:
        (outdir / name).mkdir(parents=True, exist_ok=True)

    for frame_id in range(len(pts3ds_self)):
        depth = depths_tosave[frame_id].cpu().numpy()
        conf = conf_self_tosave[frame_id].cpu().numpy()
        color = colors_tosave[frame_id].cpu().numpy()
        c2w = cam2world_tosave[frame_id].cpu().numpy()
        intrinsic = intrinsics_tosave[frame_id].cpu().numpy()
        np.save(outdir / "depth" / f"{frame_id:06d}.npy", depth)
        np.save(outdir / "conf" / f"{frame_id:06d}.npy", conf)
        iio.imwrite(outdir / "color" / f"{frame_id:06d}.png", (color * 255).astype(np.uint8))
        np.savez(outdir / "camera" / f"{frame_id:06d}.npz", pose=c2w, intrinsics=intrinsic)

    return {
        "points": pts3ds_other_tosave,
        "colors": colors_tosave,
        "confidence": conf_other_tosave,
        "camera": cam_dict,
        "num_frames": len(pts3ds_self),
    }


def build_rerun_recording(
    output_dir: str | Path,
    *,
    vis_threshold: float = 1.5,
) -> str | None:
    """Build the official-style point-cloud, camera, RGB, and depth timeline."""

    try:
        import rerun as rr
    except ImportError:
        return None

    # Deferred import: keeps `import worldfoundry.pipelines.cut3r` working in
    # deployments without the studio package (only this visualization helper needs it).
    from worldfoundry.core.geometry.transforms import depth_to_world_points

    root = Path(output_dir)
    camera_paths = sorted((root / "camera").glob("*.npz"))
    if not camera_paths:
        return None

    recording_path = root / "cut3r_sequence.rrd"
    rr.init("worldfoundry_cut3r", recording_id=root.name, spawn=False)
    rr.save(recording_path)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    for frame_id, camera_path in enumerate(camera_paths):
        if hasattr(rr, "set_time"):
            rr.set_time("frame", sequence=frame_id)
        else:
            rr.set_time_sequence("frame", frame_id)
        camera = np.load(camera_path)
        pose = camera["pose"]
        intrinsics = camera["intrinsics"]
        frame_name = f"{frame_id:06d}"
        color = iio.imread(root / "color" / f"{frame_name}.png")
        depth = np.load(root / "depth" / f"{frame_name}.npy")
        confidence = np.load(root / "conf" / f"{frame_name}.npy")
        world_points = depth_to_world_points(depth, intrinsics, pose)
        valid = np.isfinite(world_points).all(axis=-1) & np.isfinite(confidence) & (confidence > vis_threshold)
        frame_root = f"world/frames/{frame_name}"
        rr.log(
            f"{frame_root}/points",
            rr.Points3D(world_points[valid], colors=color[valid], radii=0.005),
        )
        rr.log(
            f"{frame_root}/camera",
            rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], from_parent=True),
        )
        rr.log(
            f"{frame_root}/camera",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=[color.shape[1], color.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.15,
            ),
        )
        rr.log(
            "world/current_camera",
            rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], from_parent=True),
        )
        rr.log(
            "world/current_camera",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=[color.shape[1], color.shape[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.15,
            ),
        )
        rr.log("world/current_camera/rgb", rr.Image(color))
        rr.log(
            "observations/depth",
            rr.DepthImage(
                depth,
                meter=1.0,
                depth_range=[float(np.percentile(depth, 1)), float(np.percentile(depth, 99))],
            ),
        )
    rr.disconnect()
    return str(recording_path)


def run_official_export(
    input_source: str | Path | Sequence[str | Path],
    model: ARCroco3DStereo,
    *,
    output_dir: str | Path,
    device: str | None = None,
    size: int = 512,
    revisit: int = 1,
    update: bool = True,
    use_pose: bool = True,
    vis_threshold: float = 1.5,
) -> dict[str, str]:
    """Run CUT3R on an input source and export depth, confidence, color and camera files.

    Also writes ``official_summary.npz`` and, when rerun is installed, a ``.rrd`` recording.

    Returns:
        Dict of output paths: ``output_dir``, ``summary_path``, ``depth_dir``, ``conf_dir``,
        ``color_dir``, ``camera_dir`` and, when recorded, ``rrd_path``.

    Raises:
        ValueError: CUDA is requested but unavailable, or the source yields no images.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths, temp_dir = parse_seq_path(input_source)
    if not img_paths:
        raise ValueError(f"No images found in {input_source}.")

    try:
        img_mask = [True] * len(img_paths)
        views = prepare_input(
            img_paths=img_paths,
            img_mask=img_mask,
            size=size,
            revisit=revisit,
            update=update,
        )

        # PreTrainedModel.to is wrapped with functools.wraps, which type checkers cannot bind as a method.
        network = cast(torch.nn.Module, model).to(device).eval()
        with torch.no_grad():
            outputs, state_args = inference(views, network, device)
        prepared = prepare_output(outputs, output_dir, revisit=revisit, use_pose=use_pose)

        summary_path = output_dir / "official_summary.npz"
        np.savez(
            summary_path,
            num_frames=np.array([prepared["num_frames"]], dtype=np.int32),
            focal=prepared["camera"]["focal"],
            pp=prepared["camera"]["pp"],
            R=prepared["camera"]["R"],
            t=prepared["camera"]["t"],
        )
        rerun_path = build_rerun_recording(output_dir, vis_threshold=vis_threshold)
    finally:
        if temp_dir is not None:
            # Remove the frames extracted from a video input.
            shutil.rmtree(temp_dir, ignore_errors=True)

    if device_type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "depth_dir": str(output_dir / "depth"),
        "conf_dir": str(output_dir / "conf"),
        "color_dir": str(output_dir / "color"),
        "camera_dir": str(output_dir / "camera"),
    }
    if rerun_path is not None:
        result["rrd_path"] = rerun_path
    return result


__all__ = [
    "build_rerun_recording",
    "parse_seq_path",
    "prepare_input",
    "prepare_output",
    "run_official_export",
]
