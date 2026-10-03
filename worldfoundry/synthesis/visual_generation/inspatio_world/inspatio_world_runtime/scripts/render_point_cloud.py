#!/usr/bin/env python3
"""Offline point cloud rendering using DA3 output + traj_txt camera poses.

Loads PLY point clouds from DA3 output (frames_pcd/*.ply), generates target
camera poses from traj_txt, renders each frame using point splatting with
z-buffer, and outputs render_offline.mp4 + mask_offline.mp4.

Usage:
    python render_point_cloud.py \
        --da3_dir /path/to/da3_output \
        --traj_txt_path "$WORLDFOUNDRY_INSPATIO_WORLD_TRAJECTORY_ROOT/x_y_circle_cycle.txt" \
        --output_dir /path/to/output \
        --width 832 --height 480
"""

import argparse
import glob
import logging
import os
import subprocess
import shutil

import cv2
import numpy as np
import open3d as o3d
import torch

from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.trajectory import (
    generate_traj_txt,
)
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.reproducibility import (
    initialize_reproducibility,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──
MIN_DEPTH_THRESHOLD = 0.1
DEPTH_EPSILON = 1e-4
_splat_offset_cache = {}


# ── Core rendering (from reference render_utils.py) ──

def load_ply_data(ply_path, device):
    """Load point cloud from PLY file. Returns (points, colors) tensors."""
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        logger.warning(f"Point cloud has no points: {ply_path}")
        return None, None
    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    return torch.from_numpy(pts).to(device), torch.from_numpy(colors).to(device)


def render_batch(points, colors, c2w, K, width, height, point_size=2,
                 ss_ratio=2.0, bg_color=0):
    """Render point cloud from given camera pose with supersampling.

    Returns (rgb_bgr, mask) as numpy arrays.
    """
    if points is None or colors is None:
        return (np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.uint8))

    H_high = int(height * ss_ratio)
    W_high = int(width * ss_ratio)

    K_high = K.clone()
    K_high[0, :] *= ss_ratio
    K_high[1, :] *= ss_ratio

    p_size_high = int(point_size * ss_ratio)
    if p_size_high % 2 == 0:
        p_size_high += 1

    # Transform points to camera space
    w2c = torch.linalg.inv(c2w)
    N = points.shape[0]
    points_h = torch.cat([points, torch.ones((N, 1), device=points.device)], dim=1)
    cam_xyz = (points_h @ w2c.T)[:, :3]
    z = cam_xyz[:, 2]

    mask_z = z > MIN_DEPTH_THRESHOLD
    if mask_z.sum() == 0:
        return (np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.uint8))

    xyz = cam_xyz[mask_z]
    rgb = colors[mask_z]
    z = z[mask_z]

    # Project to image plane
    u_float = (K_high[0, 0] * xyz[:, 0] / z) + K_high[0, 2]
    v_float = (K_high[1, 1] * xyz[:, 1] / z) + K_high[1, 2]

    # Point splatting
    if p_size_high > 1:
        cache_key = (p_size_high, points.device)
        if cache_key not in _splat_offset_cache:
            radius = p_size_high // 2
            offset_range = torch.arange(-radius, radius + 1, device=points.device)
            dy, dx = torch.meshgrid(offset_range, offset_range, indexing='ij')
            _splat_offset_cache[cache_key] = (dx.flatten(), dy.flatten())
        dx, dy = _splat_offset_cache[cache_key]

        u_final = torch.round(u_float.unsqueeze(1) + dx.unsqueeze(0)).long().view(-1)
        v_final = torch.round(v_float.unsqueeze(1) + dy.unsqueeze(0)).long().view(-1)
        z_final = z.unsqueeze(1).expand(-1, dx.shape[0]).reshape(-1)
        rgb_final = rgb.unsqueeze(1).expand(-1, dx.shape[0], 3).reshape(-1, 3)
    else:
        u_final = torch.round(u_float).long()
        v_final = torch.round(v_float).long()
        z_final = z
        rgb_final = rgb

    # Filter out-of-bounds
    valid = (u_final >= 0) & (u_final < W_high) & (v_final >= 0) & (v_final < H_high)
    u = u_final[valid]
    v = v_final[valid]
    rgb = rgb_final[valid]
    z = z_final[valid]

    # Z-buffer depth test
    indices = v * W_high + u
    depth_buffer = torch.full((H_high * W_high,), float('inf'), device=points.device)
    depth_buffer.scatter_reduce_(0, indices, z, reduce='min', include_self=True)
    is_closest = z <= depth_buffer[indices] + DEPTH_EPSILON

    final_u = u[is_closest]
    final_v = v[is_closest]
    final_rgb = rgb[is_closest]

    # Canvas
    canvas = torch.full((3, H_high, W_high), bg_color / 255.0, device=points.device)
    canvas[:, final_v, final_u] = final_rgb.permute(1, 0)

    mask_canvas = torch.zeros((H_high, W_high), dtype=torch.uint8, device=points.device)
    mask_canvas[final_v, final_u] = 255

    # Downsample
    img_high = (canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_final = cv2.resize(img_high, (width, height), interpolation=cv2.INTER_AREA)
    mask_final = cv2.resize(mask_canvas.cpu().numpy(), (width, height),
                            interpolation=cv2.INTER_NEAREST)

    img_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
    return img_bgr, mask_final


# ── Data loading ──

def load_intrinsic(da3_dir, device):
    """Load first-frame intrinsic (3x3) from DA3 intrinsic.txt."""
    path = os.path.join(da3_dir, "intrinsic.txt")
    data = np.loadtxt(path)
    K = data[0:3, :3].astype(np.float32)
    return torch.tensor(K, device=device, dtype=torch.float32)


def load_extrinsic_c2w(da3_dir, device):
    """Load extrinsics from DA3 extrinsic.txt.

    The file contains N frames, each stored as 3 rows of a 3x4 w2c matrix.

    Returns:
        initial_c2w: (4, 4) tensor, first-frame c2w
        source_c2ws: list of (4, 4) tensors, all-frame c2ws
    """
    path = os.path.join(da3_dir, "extrinsic.txt")
    data = np.loadtxt(path)
    num_frames = data.shape[0] // 3

    source_c2ws = []
    for i in range(num_frames):
        w2c_34 = data[i * 3:(i + 1) * 3, :4].astype(np.float32)
        w2c = np.vstack([w2c_34, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)
        c2w = torch.linalg.inv(w2c_t)
        source_c2ws.append(c2w)

    initial_c2w = source_c2ws[0]
    return initial_c2w, source_c2ws


def load_ply_sequence(da3_dir, device, max_frames=None):
    """Load PLY sequence from frames_pcd/. Returns lists of (points, colors)."""
    ply_folder = os.path.join(da3_dir, "frames_pcd")
    ply_files = sorted(glob.glob(os.path.join(ply_folder, "*.ply")))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files in {ply_folder}")
    if max_frames is not None:
        ply_files = ply_files[:max_frames]

    logger.info(f"Loading {len(ply_files)} point clouds from {ply_folder}")
    points_list, colors_list = [], []
    for pf in ply_files:
        pts, cols = load_ply_data(pf, device)
        points_list.append(pts)
        colors_list.append(cols)
    return points_list, colors_list


def scale_intrinsic(K, target_width, target_height):
    """Scale intrinsic matrix to target resolution based on principal point."""
    orig_cx = K[0, 2].item()
    orig_cy = K[1, 2].item()
    scale_x = target_width / (orig_cx * 2)
    scale_y = target_height / (orig_cy * 2)
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] = target_width / 2.0
    K_scaled[1, 2] = target_height / 2.0
    return K_scaled


# ── Camera pose generation ──

def generate_target_c2ws(traj_txt_path, initial_c2w, source_c2ws, num_frames, device,
                         relative_to_source=False, rotation_only=False):
    """Generate target c2w poses from traj_txt.

    1. Read traj_txt -> generate_traj_txt() -> relative c2w offsets (N, 4, 4)
    2. Optionally zero out translation (rotation_only)
    3. Compose with initial_c2w (relative_to_source) or use as absolute poses

    Args:
        traj_txt_path: path to traj txt file (3 lines: x_up, y_left, r)
        initial_c2w: (4, 4) first-frame c2w from DA3
        num_frames: how many frames to generate
        device: torch device
        relative_to_source: if True, compose relative poses on top of initial_c2w;
                            if False, treat generated poses as absolute world coords
        rotation_only: if True, zero out translation in relative poses (tripod pan/tilt)

    Returns:
        list of (4, 4) c2w tensors, length = num_frames
    """
    with open(traj_txt_path, 'r') as f:
        lines = f.readlines()
    x_up_angle = [float(i) for i in lines[0].split()]
    y_left_angle = [float(i) for i in lines[1].split()]
    r_raw = [float(i) for i in lines[2].split()]

    # generate_traj_txt returns relative c2w offsets (identity at frame 0)
    relative_c2ws = generate_traj_txt(
        x_up_angle, y_left_angle, r_raw, r_raw, num_frames, is_translation=rotation_only
    )  # (N, 4, 4) numpy, these are relative c2w transforms # Twc

    target_c2ws = []
    abs_source_c2ws = []
    for i in range(num_frames):
        rel_source = source_c2ws[i]  # already a torch tensor (4,4) on device, Twc
        abs_source_c2w = initial_c2w.inverse() @ rel_source # Tc0w @ Twc = Tc0c
        abs_source_c2ws.append(abs_source_c2w)

        rel = torch.tensor(relative_c2ws[i], dtype=torch.float32, device=device) #Twc
        abs_target_c2w = initial_c2w.inverse() @ rel # Tc0w @ Twc = Tc0c
        if relative_to_source:
            abs_target_c2w = (abs_target_c2w.inverse() @ abs_source_c2w.inverse()).inverse() # (new_Tcw_tgt = Tcw_tgt @ Tcw_src).inverse

        target_c2ws.append(abs_target_c2w)
    return target_c2ws


# ── ffmpeg streaming ──

def open_ffmpeg_writer(output_path, width, height, fps=24):
    """Open ffmpeg subprocess for streaming raw RGB24 frames to mp4."""
    ffmpeg_binary = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg_binary:
        try:
            import imageio_ffmpeg
            ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError(
                "ffmpeg was not found on PATH and imageio_ffmpeg is unavailable. "
                "Install ffmpeg or set FFMPEG_BINARY to an ffmpeg executable."
            ) from exc

    cmd = [
        ffmpeg_binary, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-loglevel", "warning",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ── Main rendering pipeline ──

def render_point_cloud(da3_dir, traj_txt_path, output_dir, width=832, height=480,
                       point_size=2, fps=24, relative_to_source=False, rotation_only=False,
                       freeze_repeat=0, freeze_frame=None):
    """Main entry: load data, generate poses, render, save mp4s."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load camera params
    K_orig = load_intrinsic(da3_dir, device)
    K_render = scale_intrinsic(K_orig, width, height)
    initial_c2w, source_c2ws = load_extrinsic_c2w(da3_dir, device)
    logger.info(f"Intrinsic (scaled to {width}x{height}):\n{K_render}")
    logger.info(f"Initial c2w:\n{initial_c2w}")
    logger.info(f"Loaded {len(source_c2ws)} source extrinsics")

    # Load PLY sequence
    points_list, colors_list = load_ply_sequence(da3_dir, device)
    num_pcds = len(points_list)
    logger.info(f"Loaded {num_pcds} point clouds")

    # Time-freeze: repeat a specific frame to create a pause effect
    if freeze_repeat > 0:
        if freeze_frame is None:
            freeze_frame = num_pcds // 2
        freeze_frame = max(0, min(freeze_frame, num_pcds - 1))
        logger.info(f"Time-freeze: repeating frame {freeze_frame} x{freeze_repeat} "
                     f"(inserting {freeze_repeat} extra frames)")
        insert_pos = freeze_frame + 1
        frozen_pts = points_list[freeze_frame]
        frozen_cols = colors_list[freeze_frame]
        frozen_c2w = source_c2ws[freeze_frame]
        points_list = points_list[:insert_pos] + [frozen_pts] * freeze_repeat + points_list[insert_pos:]
        colors_list = colors_list[:insert_pos] + [frozen_cols] * freeze_repeat + colors_list[insert_pos:]
        source_c2ws = source_c2ws[:insert_pos] + [frozen_c2w] * freeze_repeat + source_c2ws[insert_pos:]
        num_pcds = len(points_list)
        logger.info(f"After freeze: {num_pcds} total frames")

    # Generate target camera poses
    target_c2ws = generate_target_c2ws(traj_txt_path, initial_c2w, source_c2ws, num_pcds, device,
                                       relative_to_source=relative_to_source,
                                       rotation_only=rotation_only)
    num_frames = len(target_c2ws)
    logger.info(f"Generated {num_frames} target camera poses")

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "render_offline.mp4")
    mask_path = os.path.join(output_dir, "mask_offline.mp4")

    video_proc = open_ffmpeg_writer(video_path, width, height, fps)
    mask_proc = open_ffmpeg_writer(mask_path, width, height, fps)

    try:
        for idx in range(num_frames):
            # Cyclic access to point clouds (same as reference)
            pcd_idx = idx % num_pcds
            pts = points_list[pcd_idx]
            cols = colors_list[pcd_idx]
            c2w = target_c2ws[idx]

            img_bgr, mask_gray = render_batch(
                pts, cols, c2w, K_render, width, height,
                point_size=point_size, ss_ratio=2.0, bg_color=0
            )

            # BGR -> RGB for ffmpeg
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

            video_proc.stdin.write(img_rgb.tobytes())
            mask_proc.stdin.write(mask_rgb.tobytes())

            if idx % 50 == 0:
                pos = c2w[:3, 3]
                logger.info(f"  Frame {idx}/{num_frames} | "
                            f"Pose: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    finally:
        video_proc.stdin.close()
        mask_proc.stdin.close()
        video_proc.wait()
        mask_proc.wait()

    if video_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")
    if mask_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mask_path}")

    logger.info(f"Saved: {video_path}")
    logger.info(f"Saved: {mask_path}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Offline point cloud rendering using DA3 output + traj_txt")
    parser.add_argument("--da3_dir", type=str, required=True,
                        help="DA3 output directory (contains frames_pcd/, intrinsic.txt, extrinsic.txt)")
    parser.add_argument("--traj_txt_path", type=str, required=True,
                        help="Trajectory txt file (3 lines: x_up, y_left, r)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for render_offline.mp4 and mask_offline.mp4")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--point_size", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--relative_to_source", action="store_true",
                        help="Compose trajectory poses relative to initial view (default: off, use absolute poses)")
    parser.add_argument("--rotation_only", action="store_true",
                        help="Only apply rotation from the trajectory, ignore translation (tripod pan/tilt)")
    parser.add_argument("--freeze_repeat", type=int, default=0,
                        help="Number of times to repeat the freeze frame (0 = disabled)")
    parser.add_argument("--freeze_frame", type=int, default=None,
                        help="Frame index to freeze (default: middle frame)")
    args = parser.parse_args()
    initialize_reproducibility()

    render_point_cloud(
        da3_dir=args.da3_dir,
        traj_txt_path=args.traj_txt_path,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
        fps=args.fps,
        relative_to_source=args.relative_to_source,
        rotation_only=args.rotation_only,
        freeze_repeat=args.freeze_repeat,
        freeze_frame=args.freeze_frame,
    )


if __name__ == "__main__":
    main()
