from __future__ import annotations
from random import randint
from typing import Dict, Optional, List
import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import h5py
import torch.nn.functional as F
from einops import rearrange

# Lazy import torchcodec to avoid environment issues at module load
_VideoDecoder = None

_LEFT_NODES_ORDER: List[str] = [
    # thumb (4 joints)
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbTip",
    # index (4 joints, no metacarpal)
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    # middle (4 joints, no metacarpal)
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    # ring (4 joints, no metacarpal)
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    # little (4 joints, no metacarpal)
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]

_RIGHT_NODES_ORDER: List[str] = [
    # thumb (4 joints)
    "rightThumbKnuckle",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbTip",
    # index (4 joints, no metacarpal)
    "rightIndexFingerKnuckle",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerTip",
    # middle (4 joints, no metacarpal)
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip",
    # ring (4 joints, no metacarpal)
    "rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerTip",
    # little (4 joints, no metacarpal)
    "rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
]

to_tensor = transforms.ToTensor()


class MANODataset(Dataset):
    """
    Args:
      - eval_mode: If True, only scans directories starting with "test/".
                   If False (default), only scans directories starting with "part".
      - rotation_repr: "rot6d" (default) or "axis_angle" for rotation delta representation.

    Returns:
      - "action": (T-1, D) float32 (computed as deltas with block baseline)
                  If rotation_repr="rot6d": D = 3 + 6 + 3 + 6 + 6*NL + 6*NR
                  If rotation_repr="axis_angle": D = 3 + 3 + 3 + 3 + 3*NL + 3*NR
                  Layout: [Lw_xyz(3), Lw_rot, Rw_xyz(3), Rw_rot, Lnodes, Rnodes]
      - "video":  list[PIL.Image] of length T-1 (first frame skipped as baseline)
      - "num_frames": int (= T) # original decoded frames before skipping baseline
      - "fps": int

    Where NL=20 (left hand nodes), NR=20 (right hand nodes).
    """
    def __init__(
        self,
        randomize: bool = True,
        num_frames: int = 13,
        episode_pairs: Optional[list] = None,
        fps: int = 30,
        *,
        seek_mode: str = "exact",
        ffmpeg_threads: int = 1,
        converted_root: Optional[str] = None,
        egodex_translation_stats_path: Optional[str] = None,
        normalize_translation: bool = False,
        time_division_factor: int = 4,
        eval_mode: bool = False,
        rotation_repr: str = "axis_angle",
    ) -> None:
        super().__init__()
        self.randomize = randomize
        self.num_frames = num_frames
        self.fps = fps
        self.seek_mode = seek_mode
        self.ffmpeg_threads = ffmpeg_threads
        self.converted_root = converted_root
        self.normalize_translation = normalize_translation
        self.time_division_factor = time_division_factor
        self.eval_mode = eval_mode
        if rotation_repr not in ("rot6d", "axis_angle"):
            raise ValueError(f"rotation_repr must be 'rot6d' or 'axis_angle', got {rotation_repr}")
        self.rotation_repr = rotation_repr
        
        self._left_xyz_mean_egodex: Optional[np.ndarray] = None
        self._left_xyz_invstd_egodex: Optional[np.ndarray] = None
        self._right_xyz_mean_egodex: Optional[np.ndarray] = None
        self._right_xyz_invstd_egodex: Optional[np.ndarray] = None

        if self.normalize_translation:
            eps = 1e-8

            def load_into(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                st = np.load(path)
                lm = np.asarray(st.get("left_xyz_mean"), dtype=np.float32)
                lv = np.asarray(st.get("left_xyz_var"), dtype=np.float32)
                rm = np.asarray(st.get("right_xyz_mean"), dtype=np.float32)
                rv = np.asarray(st.get("right_xyz_var"), dtype=np.float32)
                if lm.shape != (3,) or lv.shape != (3,) or rm.shape != (3,) or rv.shape != (3,):
                    raise ValueError("translation_stats npz must contain left_xyz_mean/var and right_xyz_mean/var of shape (3,)")
                return lm, lv, rm, rv

            if egodex_translation_stats_path and os.path.isfile(egodex_translation_stats_path):
                lm, lv, rm, rv = load_into(egodex_translation_stats_path)
                self._left_xyz_mean_egodex = lm
                self._left_xyz_invstd_egodex = 1.0 / np.sqrt(lv + eps)
                self._right_xyz_mean_egodex = rm
                self._right_xyz_invstd_egodex = 1.0 / np.sqrt(rv + eps)

            # Require at least one to be present if normalization is requested
            if self._left_xyz_mean_egodex is None:
                raise FileNotFoundError(
                    "normalize_translation=True but no dataset-specific stats provided (egodex)"
                )

        if episode_pairs is not None:
            self.episode_pairs = episode_pairs
        else:
            pairs: list[tuple[str, str]] = []
            if self.converted_root:
                pairs.extend(self._discover_pairs_from_converted(self.converted_root))
            if not pairs:
                raise FileNotFoundError(
                    f"No paired .mp4 and converted .hdf5 found (provide converted_root for EgoDex-21)"
                )
            self.episode_pairs = sorted(pairs)

    # (FOR EGODEX) map converted HDF5 to the corresponding MP4 in the configured 480p dataset
    _VIDEO_ROOT = os.getenv("EGODEX_VIDEO_ROOT", "data/datasets/egodex_480p")

    def _video_path_from_converted(self, converted_path: str) -> Optional[str]:
        # find subpath starting at the first directory named like "part*"
        parts = os.path.normpath(converted_path).split(os.sep)
        start_idx = 0
        for i, p in enumerate(parts):
            if p.startswith("part"):
                start_idx = i
                break
        rel = os.path.join(*parts[start_idx:]) if start_idx < len(parts) else os.path.basename(converted_path)
        stem, _ = os.path.splitext(rel)
        video_rel = f"{stem}.mp4"
        video_path = os.path.join(self._VIDEO_ROOT, video_rel)
        return video_path if os.path.isfile(video_path) else None

    def _discover_pairs_from_converted(self, converted_root: str) -> list[tuple[str, str]]:
        # pass 1: build a set of video stems under the fixed video root using scandir DFS
        video_stems: set[str] = set()
        vr = os.path.abspath(self._VIDEO_ROOT)
        vrl = len(vr) + 1
        dir_prefix = "test" if self.eval_mode else "part"
        stack = [vr]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False):
                            dir_name = os.path.basename(e.path)
                            if d == vr:
                                if dir_name.startswith(dir_prefix):
                                    stack.append(e.path)
                            else:
                                stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            name = e.name
                            if name.endswith(".mp4"):
                                rel = e.path[vrl:] if e.path.startswith(vr) else os.path.relpath(e.path, vr)
                                stem, _ = os.path.splitext(os.path.normpath(rel))
                                video_stems.add(stem)
            except PermissionError:
                continue

        # pass 2: walk converted_root and include only those with matching video stems
        pairs: list[tuple[str, str]] = []
        cr = os.path.abspath(converted_root)
        crl = len(cr) + 1
        stack = [cr]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False):
                            dir_name = os.path.basename(e.path)
                            if d == cr:
                                if dir_name.startswith(dir_prefix):
                                    stack.append(e.path)
                            else:
                                stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            n = e.name
                            if n.endswith(".hdf5"):
                                rel = e.path[crl:] if e.path.startswith(cr) else os.path.relpath(e.path, cr)
                                stem, _ = os.path.splitext(os.path.normpath(rel))
                                if stem in video_stems:
                                    vpath = os.path.join(vr, stem + ".mp4")
                                    pairs.append((vpath, e.path))
            except PermissionError:
                continue

        if not pairs:
            raise FileNotFoundError(f"No converted .hdf5 mapped to videos under {converted_root}")
        return sorted(pairs)

    def __len__(self) -> int:
        return len(self.episode_pairs)

    def __getitem__(self, idx: int) -> Dict:
        while True:
            video_path, conv_h5_path = self.episode_pairs[idx]
            try:
                video_pil, start_idx = self._load_video_slice_torchcodec(
                    video_path,
                    num_frames=self.num_frames,
                )

                # build flat feature vector Z_t = [left_wrist(9), right_wrist(9), rot6d(left nodes...), rot6d(right nodes...)]
                with h5py.File(conv_h5_path, "r", locking=False) as f_conv:
                    left_group = f_conv["rot6d/left"]
                    right_group = f_conv["rot6d/right"]
                    left_nodes_order = _LEFT_NODES_ORDER
                    right_nodes_order = _RIGHT_NODES_ORDER

                    T = len(video_pil)
                    s0 = start_idx
                    s_end = s0 + T  # exclusive

                    Lw = np.asarray(f_conv["wrist/left_pose_cam_rot6d"][s0:s_end], dtype=np.float32)   # (T,9)
                    Rw = np.asarray(f_conv["wrist/right_pose_cam_rot6d"][s0:s_end], dtype=np.float32)  # (T,9)
                    if self.normalize_translation:
                        conv_abs = os.path.abspath(conv_h5_path)
                        conv_low = conv_abs.lower()
                        use_left_mean = None
                        use_left_inv = None
                        use_right_mean = None
                        use_right_inv = None

                        if "egodex" in conv_low and self._left_xyz_mean_egodex is not None:
                            use_left_mean = self._left_xyz_mean_egodex
                            use_left_inv = self._left_xyz_invstd_egodex
                            use_right_mean = self._right_xyz_mean_egodex
                            use_right_inv = self._right_xyz_invstd_egodex
                        else:
                            try:
                                if self.converted_root and os.path.commonpath([conv_abs, os.path.abspath(self.converted_root)]) == os.path.abspath(self.converted_root) and self._left_xyz_mean_egodex is not None:
                                    use_left_mean = self._left_xyz_mean_egodex
                                    use_left_inv = self._left_xyz_invstd_egodex
                                    use_right_mean = self._right_xyz_mean_egodex
                                    use_right_inv = self._right_xyz_invstd_egodex
                            except Exception:
                                pass

                        if use_left_mean is None or use_left_inv is None or use_right_mean is None or use_right_inv is None:
                            raise RuntimeError("normalize_translation=True but matching dataset stats not loaded for this sample")

                        Lw[:, :3] = (Lw[:, :3] - use_left_mean) * use_left_inv
                        Rw[:, :3] = (Rw[:, :3] - use_right_mean) * use_right_inv
                    L_list = [np.asarray(left_group[name][s0:s_end], dtype=np.float32) for name in left_nodes_order]
                    R_list = [np.asarray(right_group[name][s0:s_end], dtype=np.float32) for name in right_nodes_order]
                    Lcat = np.concatenate(L_list, axis=1)
                    Rcat = np.concatenate(R_list, axis=1)
                    Z = np.concatenate([Lw, Rw, Lcat, Rcat], axis=1)  # (T, D)

                    # compute delta actions:
                    # - for wrist XYZ (first 3 of each wrist): differences vs baseline
                    # - For all rot6d: relative rotations R_t @ R_{t-1}^T, encoded back to 6D
                    actions_tensor = torch.from_numpy(Z).contiguous()

                    def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
                        a1 = d6[..., 0:3]
                        a2 = d6[..., 3:6]
                        b1 = F.normalize(a1, dim=-1)
                        proj = (b1 * a2).sum(dim=-1, keepdim=True)
                        b2 = F.normalize(a2 - proj * b1, dim=-1)
                        b3 = torch.cross(b1, b2, dim=-1)
                        R = torch.stack((b1, b2, b3), dim=-1)  # columns
                        return R

                    def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
                        return R[..., :, 0:2].transpose(-2, -1).flatten(start_dim=-2)

                    def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
                        """
                        Convert rotation matrix to axis-angle (Rodrigues) vector w, where ||w|| = angle.
                        R: (..., 3, 3)
                        returns w: (..., 3)
                        """
                        # angle from trace
                        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...,)
                        cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
                        angle = torch.acos(cos_theta)  # (...,)

                        # v = [R32 - R23, R13 - R31, R21 - R12]
                        v = torch.stack([
                            R[..., 2, 1] - R[..., 1, 2],
                            R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1],
                        ], dim=-1)  # (...,3)

                        # Masks
                        small = (angle < 1e-4).unsqueeze(-1)          # (...,1)
                        near_pi = (angle > (torch.pi - 0.05)).unsqueeze(-1)  # (...,1) ~177°

                        # ---- Generic case (not small, not pi): w = (angle / (2 sin angle)) * v
                        sin_theta = torch.sin(angle)  # (...,)
                        denom = 2.0 * sin_theta
                        # Avoid dividing by ~0: replace denom with 1 in small/near_pi branches (we'll overwrite anyway)
                        safe_denom = torch.where((small | near_pi).squeeze(-1), torch.ones_like(denom), denom)
                        factor_generic = (angle / safe_denom).unsqueeze(-1)  # (...,1)
                        w_generic = factor_generic * v  # (...,3)

                        # ---- Small-angle: R ≈ I + [w]_x  => v ≈ 2 w  => w ≈ v/2
                        w_small = 0.5 * v

                        # ---- Near-π: get axis from diagonal, then multiply by angle
                        # u_i^2 = (R_ii + 1)/2; choose the largest to stabilize, fix signs with off-diagonals
                        r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
                        ux = torch.sqrt(torch.clamp((r00 + 1.0) * 0.5, min=0.0))
                        uy = torch.sqrt(torch.clamp((r11 + 1.0) * 0.5, min=0.0))
                        uz = torch.sqrt(torch.clamp((r22 + 1.0) * 0.5, min=0.0))

                        # Fix signs using off-diagonals (only meaningful when not tiny)
                        ux = torch.where((R[..., 2, 1] - R[..., 1, 2]) >= 0, ux, -ux)
                        uy = torch.where((R[..., 0, 2] - R[..., 2, 0]) >= 0, uy, -uy)
                        uz = torch.where((R[..., 1, 0] - R[..., 0, 1]) >= 0, uz, -uz)

                        u = torch.stack([ux, uy, uz], dim=-1)  # (...,3)
                        # Normalize to be safe
                        u = F.normalize(u, dim=-1, eps=1e-12)
                        w_pi = u * angle.unsqueeze(-1)

                        # Blend branches
                        w = torch.where(near_pi, w_pi, w_generic)
                        w = torch.where(small, w_small, w)
                        return w

                    D_total = actions_tensor.shape[1]
                    num_left_nodes = len(left_nodes_order)
                    num_right_nodes = len(right_nodes_order)

                    # index layout in Z (input): [Lw(9) | Rw(9) | Lnodes(6*NL) | Rnodes(6*NR)]
                    idx_lw_xyz = slice(0, 3)
                    idx_lw_rot = slice(3, 9)
                    idx_rw_xyz = slice(9, 12)
                    idx_rw_rot = slice(12, 18)
                    idx_l_nodes = slice(18, 18 + 6 * num_left_nodes)
                    idx_r_nodes = slice(18 + 6 * num_left_nodes, 18 + 6 * (num_left_nodes + num_right_nodes))

                    # Compute output dimension based on rotation representation
                    rot_dim = 3 if self.rotation_repr == "axis_angle" else 6
                    # Output layout: [Lw_xyz(3) | Lw_rot(rot_dim) | Rw_xyz(3) | Rw_rot(rot_dim) |
                    #                 Lnodes(rot_dim*NL) | Rnodes(rot_dim*NR)]
                    D_out = 3 + rot_dim + 3 + rot_dim + rot_dim * num_left_nodes + rot_dim * num_right_nodes

                    # Output index layout
                    out_lw_xyz = slice(0, 3)
                    out_lw_rot = slice(3, 3 + rot_dim)
                    out_rw_xyz = slice(3 + rot_dim, 6 + rot_dim)
                    out_rw_rot = slice(6 + rot_dim, 6 + 2 * rot_dim)
                    out_l_nodes = slice(6 + 2 * rot_dim, 6 + 2 * rot_dim + rot_dim * num_left_nodes)
                    out_r_nodes = slice(6 + 2 * rot_dim + rot_dim * num_left_nodes, D_out)

                    # Choose conversion function
                    matrix_to_repr = matrix_to_axis_angle if self.rotation_repr == "axis_angle" else matrix_to_rot6d

                    delta_actions_blocks: list[torch.Tensor] = []
                    T_total = actions_tensor.shape[0]
                    k = int(self.time_division_factor)
                    for t in range(1, T_total - 1, k):
                        t_end = min(t + k, T_total)
                        B = t_end - t  # batch size
                        block_out = torch.empty((B, D_out), dtype=torch.float32)  # Pre-set dtype

                        # wrist translations (xyz) differences
                        block_out[:, out_lw_xyz] = actions_tensor[t:t_end, idx_lw_xyz] - actions_tensor[t-1, idx_lw_xyz]
                        block_out[:, out_rw_xyz] = actions_tensor[t:t_end, idx_rw_xyz] - actions_tensor[t-1, idx_rw_xyz]

                        # wrist rotations (rot6d -> R, matmul, back to chosen repr)
                        R_block_lw = rot6d_to_matrix(actions_tensor[t:t_end, idx_lw_rot])  # (B, 3, 3)
                        R_base_lw = rot6d_to_matrix(actions_tensor[t-1, idx_lw_rot].unsqueeze(0))[0]  # (3, 3)
                        R_rel_lw = R_block_lw @ R_base_lw.T
                        block_out[:, out_lw_rot] = matrix_to_repr(R_rel_lw)

                        R_block_rw = rot6d_to_matrix(actions_tensor[t:t_end, idx_rw_rot])  # (B, 3, 3)
                        R_base_rw = rot6d_to_matrix(actions_tensor[t-1, idx_rw_rot].unsqueeze(0))[0]  # (3, 3)
                        R_rel_rw = R_block_rw @ R_base_rw.transpose(0, 1)
                        block_out[:, out_rw_rot] = matrix_to_repr(R_rel_rw)

                        # left nodes (rot_dim per node)
                        if num_left_nodes > 0:
                            block_L = actions_tensor[t:t_end, idx_l_nodes].reshape(B * num_left_nodes, 6)
                            base_L = actions_tensor[t-1, idx_l_nodes].reshape(num_left_nodes, 6)
                            R_block_L = rot6d_to_matrix(block_L)  # (B*NL, 3, 3)
                            R_base_L = rot6d_to_matrix(base_L)  # (NL, 3, 3)
                            R_base_L_rep = R_base_L.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_left_nodes, 3, 3)
                            R_rel_L = R_block_L @ R_base_L_rep.transpose(-1, -2)
                            block_out[:, out_l_nodes] = matrix_to_repr(R_rel_L).reshape(B, rot_dim * num_left_nodes)

                        # right nodes (rot_dim per node)
                        if num_right_nodes > 0:
                            block_R = actions_tensor[t:t_end, idx_r_nodes].reshape(B * num_right_nodes, 6)
                            base_R = actions_tensor[t-1, idx_r_nodes].reshape(num_right_nodes, 6)
                            R_block_R = rot6d_to_matrix(block_R)  # (B*NR, 3, 3)
                            R_base_R = rot6d_to_matrix(base_R)  # (NR, 3, 3)
                            R_base_R_rep = R_base_R.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_right_nodes, 3, 3)
                            R_rel_R = R_block_R @ R_base_R_rep.transpose(-1, -2)
                            block_out[:, out_r_nodes] = matrix_to_repr(R_rel_R).reshape(B, rot_dim * num_right_nodes)

                        delta_actions_blocks.append(block_out)

                    action_window = torch.cat(delta_actions_blocks, dim=0)  # (T-1, D)

                # video_frames = video_pil
                video_frames = [to_tensor(img) for img in video_pil]
                video_tensor = torch.stack(video_frames, dim=0)
                video_tensor = torch.clamp(video_tensor * 255.0, 0, 255).to(torch.uint8)
                video_tensor = video_tensor.transpose(0, 1)  # (T, C, H, W) -> (C, T, H, W)
                if video_tensor.shape[1] != action_window.shape[0] + 1:
                    raise ValueError(
                        f"Window length mismatch: video (without baseline) {video_tensor.shape[1]} vs action {action_window.shape[0]}"
                    )

                target_ratio = 640 / 480
                if video_tensor.shape[3] / video_tensor.shape[2] > target_ratio:
                    target_height = video_tensor.shape[2]
                    target_width = int(video_tensor.shape[2] * target_ratio)
                elif video_tensor.shape[3] / video_tensor.shape[2] < target_ratio:
                    target_height = int(video_tensor.shape[3] / target_ratio)
                    target_width = video_tensor.shape[3]
                else:
                    target_height = video_tensor.shape[2]
                    target_width = video_tensor.shape[3]
                h_crop = (video_tensor.shape[2] - target_height) // 2
                w_crop = (video_tensor.shape[3] - target_width) // 2
                video_tensor = video_tensor[:, :, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
                video_tensor = F.interpolate(video_tensor, (480, 640), mode="bilinear")

                lam_frames = F.interpolate(video_tensor, (240, 320), mode="bilinear")
                lam_frames = torch.clamp(lam_frames / 255.0, 0, 1)
                lam_frames = torch.repeat_interleave(lam_frames, 2, dim=1)[:, 1:-1, :, :]
                lam_frames = rearrange(lam_frames, "c t h w -> t h w c")

                action_tensor = action_window.to(dtype=torch.float32)

                key = action_tensor[0:1, :29]

                # gt_actions = torch.zeros(self.num_frames - 1, 220, dtype=torch.float32)
                # mano_actions = action_tensor
                # latent_actions = torch.zeros(self.num_frames - 1, 32, dtype=torch.float32)
                # # latent_actions = torch.ones(self.num_frames - 1, 32, dtype=torch.float32)
                # # latent_actions = latent_actions * torch.bernoulli(0.8 * torch.ones(1)).type_as(latent_actions)
                # action_seq = torch.cat([gt_actions, mano_actions, latent_actions], dim=1)
                gt_actions = torch.zeros(self.num_frames - 1, 352, dtype=torch.float32)
                latent_actions = torch.ones(self.num_frames - 1, 32, dtype=torch.float32)
                action_seq = torch.cat([gt_actions, latent_actions], dim=-1)

                out: Dict[str, object] = {
                    "video": video_tensor,
                    "lam_video": lam_frames,
                    "action": action_seq,
                    "dataset": "egodex",
                    "fps": self.fps,
                    "num_frames": self.num_frames,

                    "__key__": key,
                    "padding_mask": torch.zeros(1, 256, 256).cuda(),
                    "image_size": 256 * torch.ones(4).cuda(),
                    "ai_caption": "",
                }
                return out
            except Exception as e:
                print("Error loading sample, retrying with a different index...")
                print(e)
                idx = randint(0, len(self) - 1)

    def _load_video_slice_torchcodec(self, video_path: str, num_frames: int) -> tuple[list[Image.Image], int]:
        """
        Decode exactly `num_frames` contiguous frames using TorchCodec.
        Returns (list[PIL.Image] length T, start_index).
        Optimized for high resolution and long videos.
        """
        global _VideoDecoder
        if _VideoDecoder is None:
            from torchcodec.decoders import VideoDecoder as _VD
            _VideoDecoder = _VD
        
        # Use more threads for better performance (default was 1, which is too low)
        # For high-res videos, more threads significantly improve decoding speed
        num_threads = max(self.ffmpeg_threads, 4) if self.ffmpeg_threads == 1 else self.ffmpeg_threads
        
        decoder = _VideoDecoder(
            video_path,
            dimension_order="NHWC",
            num_ffmpeg_threads=num_threads,
            seek_mode=self.seek_mode,
        )

        total = len(decoder)
        if total < num_frames:
            raise ValueError(f"Video shorter than requested window: {total} < {num_frames}, {video_path}")

        start = randint(0, total - num_frames - 1) if self.randomize else 0
        stop = start + num_frames

        batch = decoder.get_frames_in_range(start, stop).data
        if batch.device.type != "cpu":
            batch = batch.to("cpu", non_blocking=True)

        # Optimized batch conversion: convert entire tensor to numpy once, then to PIL
        # This is much faster than converting frame-by-frame, especially for high-res videos
        batch_np = batch.numpy()  # Shape: (T, H, W, C)
        
        # Ensure uint8 format for PIL (handle both uint8 and float inputs)
        if batch_np.dtype != "uint8":
            # Clamp values to [0, 255] range if float
            batch_np = batch_np.clip(0, 255).astype("uint8")
        
        # Batch convert all frames - much faster than individual conversions
        # Pre-allocate list and use efficient numpy slicing
        pil_list = [Image.fromarray(batch_np[i], mode="RGB") for i in range(batch_np.shape[0])]
        return pil_list, start
