"""Evoke camera-pose transforms (AlayaLab/Evoke, Apache-2.0)."""

import torch


def SE3_inverse_torch(T):
    """Batched SE3 inverse in torch. Args: T [B,4,4]. Returns T_inv [B,4,4]."""
    Rot = T[:, :3, :3]       # [B, 3, 3]
    trans = T[:, :3, 3:]     # [B, 3, 1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).expand(T.shape[0], -1, -1).clone()
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses_lingbot(c2ws_mat, framewise=True, normalize_trans=True):
    """Compute relative poses from absolute c2w matrices, optionally framewise and translation-normalized."""
    # step 1: all frames relative to frame 0
    ref_w2cs = SE3_inverse_torch(c2ws_mat[0:1])  # [1, 4, 4]
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)  # [F, 4, 4]
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)

    # step 2: convert to per-frame relative poses
    if framewise:
        relative_poses_framewise = torch.bmm(
            SE3_inverse_torch(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise

    # step 3: normalize translations to max-norm 1
    if normalize_trans:
        translations = relative_poses[:, :3, 3]  # [F, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm

    return relative_poses
