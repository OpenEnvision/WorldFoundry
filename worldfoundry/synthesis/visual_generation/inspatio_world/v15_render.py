# SPDX-License-Identifier: Apache-2.0
"""Official v1.5 source-view selection and RGBD condition composition.

Adapted from inspatio-world-v1.5 dd3561f544053fe22d739b6b2f8461c9c97bf8cb.
The depth-weighted splat is shared through the 3D base-model layer.
"""
from contextlib import ExitStack
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.depth_warper import DepthWarper

from .v15_io import decode_image_depth, decode_video_depth_frame, image_depth_paths, read_range, video_writer
from .v15_scene import latent_frame_groups, source_type_for


def load_matrix(path, size):
    return np.loadtxt(path, ndmin=2).astype(np.float32).reshape(-1, size, size)


def load_scene(record):
    import cv2

    scene = Path(record["path"])
    meta = record
    captures = []
    source_k = load_matrix(scene / "depth" / "source_intrinsics.txt", 3)
    source_tcw = load_matrix(scene / "depth" / "source_tcw.txt", 4)
    target_tcw = load_matrix(Path(record.get("target_traj_path", scene / "input" / "target_tcw.txt")), 4)
    if len(target_tcw) != meta["frames"] or len(source_k) != meta["views"]:
        raise ValueError(f"Camera count mismatch: {scene}")
    if meta["kind"] == "image":
        image_paths = sorted((scene / "input").glob("view_*.png"))
        depth_paths = image_depth_paths(scene / "depth", meta["views"])
        if len(image_paths) != len(depth_paths) or not all(path.is_file() for path in depth_paths):
            raise ValueError(f"Image/depth mismatch: {scene}")
        def read(index):
            return (np.asarray(Image.open(image_paths[index]).convert("RGB")),
                    decode_image_depth(depth_paths[index], scene / "depth" / "metadata.txt"))
    else:
        input_video = cv2.VideoCapture(str(scene / "input" / "video.mp4"))
        depth_video = cv2.VideoCapture(str(scene / "depth" / "depth.mp4"))
        captures.extend([input_video, depth_video])
        depth_minimum, depth_maximum = read_range(scene / "depth" / "metadata.txt")
        if (not input_video.isOpened() or not depth_video.isOpened()
                or int(input_video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(target_tcw)
                or int(depth_video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(target_tcw)):
            for capture in captures:
                capture.release()
            raise ValueError(f"Video/depth/camera count mismatch: {scene}")
        def read(index):
            if int(input_video.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                input_video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image_bgr = input_video.read()
            if not ok:
                raise ValueError(f"Cannot read input frame {index}: {scene}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if int(depth_video.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                depth_video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = depth_video.read()
            if not ok:
                raise ValueError(f"Cannot read depth frame {index}: {scene}")
            depth = decode_video_depth_frame(frame, depth_minimum, depth_maximum)
            return image, depth
    return meta, source_k, source_tcw, target_tcw, read, captures


def prepare_source(warper, image, depth, source_tcw, source_k, device):
    image_tensor = torch.from_numpy(np.array(image, copy=True, order="C")).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
    depth_tensor = torch.from_numpy(np.array(depth, copy=True, order="C")).to(device)[None, None].float()
    tcw = torch.from_numpy(source_tcw.copy()).to(device)[None]
    intrinsic = torch.from_numpy(source_k.copy()).to(device)[None]
    height, width = image.shape[:2]
    grid = warper.create_grid(1, height, width, device=device)
    positions = torch.cat([grid, torch.ones_like(grid[:, :1])], dim=1).permute(0, 2, 3, 1)[..., None]
    rays = torch.matmul(torch.linalg.inv(intrinsic)[:, None, None], positions)
    local = rays * depth_tensor[:, 0, :, :, None, None]
    local = torch.cat([local, torch.ones_like(local[..., :1, :])], dim=3)
    return image_tensor, depth_tensor, torch.linalg.inv(tcw), local


def project_source(warper, source, target_tcw, target_k):
    image, depth, source_inverse, local_points = source
    transform = torch.bmm(target_tcw, source_inverse)[:, None, None]
    transformed = torch.matmul(transform, local_points)[..., :3, :]
    points = torch.matmul(target_k[:, None, None], transformed)
    z = points[..., 2, 0]
    coordinates = points[..., :2, 0] / z.clamp_min(1e-6).unsqueeze(-1)
    grid = warper.create_grid(1, image.shape[-2], image.shape[-1], device=image.device)
    flow = coordinates.permute(0, 3, 1, 2) - grid
    valid = ((depth[:, 0] > 0) & (z > 0)).unsqueeze(1).float()
    values = torch.cat([image, z.unsqueeze(1)], dim=1)
    projected, mask = warper.bilinear_splatting(values, valid, z, flow, None, is_image=False)
    return projected[0, :3].clamp(-1, 1), projected[0, 3], mask[0, 0].bool()


def select_chunk_top3(overlap_pixels):
    """Pick three views per three-latent chunk, then assign one view per latent."""
    if len(overlap_pixels) % 3 or any(len(scores) != 4 for scores in overlap_pixels):
        raise ValueError("chunk_top3 requires three latents per chunk and four source views")
    choices, chunk_views = [], []
    for start in range(0, len(overlap_pixels), 3):
        scores = overlap_pixels[start:start + 3]
        totals = [sum(row[view] for row in scores) for view in range(4)]
        top3 = sorted(range(4), key=lambda view: (-totals[view], view))[:3]
        assignment = min(permutations(top3),
                         key=lambda views: (-sum(scores[latent][view]
                                                  for latent, view in enumerate(views)), views))
        choices.extend(assignment)
        chunk_views.append(top3)
    return choices, chunk_views


@torch.no_grad()
def render_one(record, device, output_root):
    meta, source_k, source_tcw, target_tcw, read, captures = load_scene(record)
    source_type = source_type_for(meta)
    destination = output_root / record["id"]
    destination.mkdir(parents=True, exist_ok=True)
    warper = DepthWarper()
    target_k = torch.from_numpy(source_k[0].copy()).to(device)[None]
    target_poses = torch.from_numpy(target_tcw.copy()).to(device)
    source_indices, overlap_pixels, chunk_views = [], [], []
    try:
        sources = [read(index) for index in range(len(source_k))] if meta["kind"] == "image" else None
        prepared = [prepare_source(warper, image, depth, source_tcw[index], source_k[index], device)
                    for index, (image, depth) in enumerate(sources)] if sources is not None else None
        if source_type == "multi-image":
            groups = list(latent_frame_groups(meta["frames"]))
            if len(groups) % 3:
                raise ValueError("Multi-image frames must cover complete three-latent chunks")
            chunks = [groups[index:index + 3] for index in range(0, len(groups), 3)]
        else:
            chunks = [[range(meta["frames"])]]
        height, width = meta["resolution"]
        with ExitStack() as stack:
            video = stack.enter_context(video_writer(destination / "render.mp4", meta["fps"], width, height))
            masks = stack.enter_context(video_writer(destination / "mask.mp4", meta["fps"], width, height, lossless=True))
            source_video = stack.enter_context(video_writer(destination / "source.mp4", meta["fps"], width, height))
            for chunk in chunks:
                chunk_scores = []
                for frames in chunk:
                    scores = torch.zeros(len(source_k), device=device, dtype=torch.int64) if source_type == "multi-image" else None
                    for frame in frames:
                        indices = range(len(source_k)) if sources is not None else (frame,)
                        fused = fused_depth = fused_mask = first_image = None
                        for index in indices:
                            image, depth = sources[index] if sources is not None else read(index)
                            if first_image is None:
                                first_image = image
                            projected_source = prepared[index] if prepared is not None else prepare_source(
                                warper, image, depth, source_tcw[index], source_k[index], device)
                            rgb, distance, known = project_source(warper, projected_source, target_poses[frame:frame + 1], target_k)
                            if scores is not None:
                                scores[index] += known.sum()
                            if fused is None:
                                fused = rgb
                                fused_depth = torch.where(known, distance, torch.inf)
                                fused_mask = known
                            else:
                                closer = known & (distance < fused_depth)
                                fused = torch.where(closer[None], rgb, fused)
                                fused_depth = torch.where(closer, distance, fused_depth)
                                fused_mask |= known
                        fused = torch.where(fused_mask[None], fused, -torch.ones_like(fused))
                        pixels = ((fused.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        video.write(pixels.tobytes())
                        mask_rgb = (fused_mask.cpu().numpy().astype(np.uint8) * 255)[..., None].repeat(3, axis=2)
                        masks.write(mask_rgb.tobytes())
                        if source_type != "multi-image":
                            source_video.write(first_image.tobytes())
                        if frame % 50 == 0:
                            print(f"{record['id']}: {frame + 1}/{meta['frames']}", flush=True)
                    if scores is not None:
                        chunk_scores.append(scores.cpu().tolist())
                if source_type == "multi-image":
                    choices, views = select_chunk_top3(chunk_scores)
                    source_indices.extend(choices)
                    chunk_views.extend(views)
                    overlap_pixels.extend(chunk_scores)
                    for frames, selected in zip(chunk, choices):
                        pixels = sources[selected][0].tobytes()
                        for _ in frames:
                            source_video.write(pixels)
        extra = ({"source_indices": source_indices, "overlap_pixels": overlap_pixels, "chunk_views": chunk_views}
                 if source_type == "multi-image" else None)
        return extra
    finally:
        for capture in captures:
            capture.release()
