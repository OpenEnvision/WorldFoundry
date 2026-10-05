"""Inference video, depth, mask and camera-trajectory input assembly.

Restored from the bundled InSpatio-World inference input reader. The upstream
``TestDataset`` name denotes inference inputs, not a test suite or training.
"""

import ast
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .utils import generate_traj_txt, read_frames


def compute_traj_total_angle(x_up_angles, y_left_angles):
    """Compute total angular change of trajectory (max of x and y cumulative absolute changes)."""
    x_arr = np.array(x_up_angles)
    y_arr = np.array(y_left_angles)
    total_x = np.sum(np.abs(np.diff(x_arr)))
    total_y = np.sum(np.abs(np.diff(y_arr)))
    return max(total_x, total_y)


def bounce_indices(n_source, n_target):
    """
    Generate a forward-backward-forward... index sequence of length n_target.
    e.g. n_source=5: 0,1,2,3,4,3,2,1,0,1,2,3,4,...
    """
    if n_source < 1 or n_target < 0:
        raise ValueError("Source frames must be positive and target frames nonnegative")
    if n_source == 1:
        return [0] * n_target
    if n_target <= n_source:
        return np.linspace(0, n_source - 1, n_target).astype(int).tolist()

    cycle_len = 2 * (n_source - 1)
    indices = []
    for i in range(n_target):
        pos = i % cycle_len
        if pos < n_source:
            indices.append(pos)
        else:
            indices.append(cycle_len - pos)
    return indices


class SceneInputs:
    # Allowed angle change per frame (degrees/frame)
    MIN_ANGLE_PER_FRAME = 0.3
    MAX_ANGLE_PER_FRAME = 0.8

    def __init__(self, sample_size, sample_n_frames, cam_idx=1, traj_txt_path=None, relative_to_source=False, rotation_only=False, adaptive_frame=True, freeze_repeat=0, freeze_frame=None):
        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size   # h, w
        self.cam_idx = cam_idx
        self.traj_txt_path = traj_txt_path
        self.relative_to_source = relative_to_source
        self.rotation_only = rotation_only
        self.adaptive_frame = adaptive_frame
        self.freeze_repeat = freeze_repeat
        self.freeze_frame = freeze_frame

    def read_matrix(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()

        parsed_data = []
        for line in data.strip().split('\n'):
            if line.strip():
                parsed_data.append(ast.literal_eval(line.strip()))

        return np.array(parsed_data, dtype=np.float32)

    def get_data(self, source_conf):
        data = {}

        data['text'] = source_conf['text']
        source_video = read_frames(source_conf['video_path']) # t c h w
        if source_video.shape[0] > 1000:
            source_video = source_video[:1000]
        source_video_name = Path(source_conf['video_path']).stem
        data['source_video_name'] = source_video_name
        n_frames, c, h, w = source_video.shape

        # ===================== source_video: resize + normalize to [-1, 1] =====================
        target_height, target_width = self.sample_size
        pixel_transforms = transforms.Compose([
            transforms.Resize((target_height, target_width)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        source_video = pixel_transforms(source_video)

        # ===================== load pre-rendered render/mask videos =====================
        render_dir = os.path.join(source_conf["vggt_depth_path"], 'render')
        render_video_path = os.path.join(render_dir, 'render_offline.mp4')
        mask_video_path = os.path.join(render_dir, 'mask_offline.mp4')

        # render_video: [T, C, H, W] in [0, 1] -> normalized to [-1, 1]
        render_video = read_frames(render_video_path)  # t c h w, [0, 1]
        render_video = render_video * 2.0 - 1.0  # normalize to [-1, 1]

        # mask_video: [T, C, H, W] in [0, 1] -> binary mask then to [-1, 1]
        mask_video = read_frames(mask_video_path)  # t c h w, [0, 1]
        mask_video = (mask_video > 0.5).float()
        mask_video = mask_video * 2.0 - 1.0  # to [-1, 1] range

        # Time-freeze: repeat a frame in source_video to match render/mask freeze
        if self.freeze_repeat > 0:
            n_src = source_video.shape[0]
            freeze_idx = self.freeze_frame if self.freeze_frame is not None else n_src // 2
            freeze_idx = max(0, min(freeze_idx, n_src - 1))
            insert_pos = freeze_idx + 1
            frozen_frame = source_video[freeze_idx:freeze_idx+1]  # (1, C, H, W)
            source_video = torch.cat([
                source_video[:insert_pos],
                frozen_frame.expand(self.freeze_repeat, -1, -1, -1),
                source_video[insert_pos:],
            ], dim=0)
            print(f'[Time-freeze] source_video: repeated frame {freeze_idx} x{self.freeze_repeat}, '
                  f'{n_src} -> {source_video.shape[0]} frames')

        # Align frame counts: render/mask may differ slightly from source_video
        min_frames = min(source_video.shape[0], render_video.shape[0], mask_video.shape[0])
        source_video = source_video[:min_frames]
        render_video = render_video[:min_frames]
        mask_video = mask_video[:min_frames]
        n_frames = min_frames

        # ===================== load depth for radius computation =====================
        depth_path = os.path.join(source_conf["vggt_depth_path"], 'depth')
        depth_files_list = os.listdir(depth_path)
        depth_files = sorted(depth_files_list, key=lambda x: int(os.path.splitext(x)[0]))
        # Only need the first depth frame for radius estimation
        first_depth = np.array(Image.open(os.path.join(depth_path, depth_files[0]))).astype(np.uint16)
        with open(os.path.join(source_conf["vggt_depth_path"], 'metadata.txt'), 'r') as f:
            depths_min, depths_max = tuple([float(t) for t in f.readline().strip().split(' ')])
        first_depth = (first_depth / 65535.0) * (depths_max - depths_min) + depths_min
        first_depth_min = float(first_depth.min())

        # ===================== frame-adaptive traj logic =====================
        if self.traj_txt_path is not None:
            radius = first_depth_min * source_conf['radius_ratio']
            print('Foreground mean (radius):', radius)
            with open(self.traj_txt_path, 'r') as file:
                lines = file.readlines()
                x_up_angle = [float(i) for i in lines[0].split()]
                y_left_angle = [float(i) for i in lines[1].split()]
                r_raw = [float(i) for i in lines[2].split()]
                r = [v * radius for v in r_raw]
                r_zoom = [v * first_depth_min for v in r_raw]
                print("r", r)
                print("r_zoom", r_zoom)

            # ---- Adaptive frame count: determine needed frames based on trajectory angle range ----
            if self.adaptive_frame:
                total_angle = compute_traj_total_angle(x_up_angle, y_left_angle)
                if total_angle > 1e-3:
                    min_needed = max(2, int(np.ceil(total_angle / self.MAX_ANGLE_PER_FRAME)) + 1)
                    max_needed = max(2, int(np.floor(total_angle / self.MIN_ANGLE_PER_FRAME)) + 1)
                else:
                    # Pure zoom trajectory (no angular change), no frame count adjustment needed
                    min_needed = n_frames
                    max_needed = n_frames

                angle_per_frame = total_angle / max(n_frames - 1, 1) if total_angle > 1e-3 else 0
                print(f'[Traj adaptive] total_angle={total_angle:.1f}, n_frames={n_frames}, '
                      f'angle_per_frame={angle_per_frame:.3f}, '
                      f'needed_range=[{min_needed}, {max_needed}]')

                # Expand target: at least min_needed and at least 121 frames
                expand_target = max(min_needed, 121)

                if n_frames < expand_target:
                    # Not enough source frames: too fast or below 121 frames minimum
                    target_n_frames = expand_target
                    reason = []
                    if n_frames < min_needed:
                        reason.append(f'too fast ({angle_per_frame:.2f}/frame > {self.MAX_ANGLE_PER_FRAME}/frame)')
                    if n_frames < 121:
                        reason.append('below minimum 121 frames')
                    print(f'[Traj adaptive] {", ".join(reason)}, '
                          f'expanding source from {n_frames} to {target_n_frames} frames (bounce)')
                    expand_indices = bounce_indices(n_frames, target_n_frames)
                    source_video = source_video[expand_indices]
                    render_video = render_video[expand_indices]
                    mask_video = mask_video[expand_indices]
                    n_frames = target_n_frames
                elif n_frames > max_needed:
                    target_n_frames = max_needed
                    print(f'[Traj adaptive] Too slow ({angle_per_frame:.2f}/frame < {self.MIN_ANGLE_PER_FRAME}/frame), '
                          f'subsampling source from {n_frames} to {target_n_frames} frames')
                    subsample_indices = np.linspace(0, n_frames - 1, target_n_frames).astype(int).tolist()
                    source_video = source_video[subsample_indices]
                    render_video = render_video[subsample_indices]
                    mask_video = mask_video[subsample_indices]
                    n_frames = target_n_frames
                else:
                    print('[Traj adaptive] Frame count OK, no adjustment needed')
            else:
                print(f'[Traj adaptive] Disabled, using original {n_frames} frames')

            # Generate target extrinsics from trajectory (for saving/reference)
            target_extrinsics = generate_traj_txt(x_up_angle, y_left_angle, r, r_zoom, n_frames, is_translation=self.rotation_only)  # Twc
            target_extrinsics = torch.tensor(target_extrinsics).inverse()  # Tcw
            data['target_extrinsics'] = target_extrinsics
        else:
            print("no traj txt path")

        data['source_video'] = source_video
        data['render_video'] = render_video
        data['mask_video'] = mask_video

        print(f"data['source_video'].shape {data['source_video'].shape}, "
              f"data['render_video'].shape {data['render_video'].shape}, "
              f"data['mask_video'].shape {data['mask_video'].shape}")

        return data
