"""Module for base_models -> three_dimensions -> point_clouds -> loger -> layers -> camera_head.py functionality."""

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.camera_head import ResConvBlock

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'

class CameraHead(nn.Module):
    """Camera head implementation."""
    def __init__(self, dim=512, output_quat=False):
        """Init.

        Args:
            dim: The dim.
            output_quat: The output quat.
        """
        super().__init__()
        output_dim = dim
        self.output_quat = output_quat
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        if self.output_quat:
            self.fc_rot_qvec = nn.Linear(output_dim, 4)
        else:
            self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):
        """Forward.

        Args:
            feat: The feat.
            patch_h: The patch h.
            patch_w: The patch w.
        """
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            if self.output_quat:
                out_r = self.fc_rot_qvec(feat.float())  # [B,4]
                pose = self.convert_quat_to_4x4(BN, out_r, out_t, feat.device)
                return pose, out_r
            else:
                out_r = self.fc_rot(feat.float())  # [B,9] or [B,4]
                pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device)
                return pose

    def convert_quat_to_4x4(self, B, q, t, device):
        """Convert quat to 4x4.

        Args:
            B: The b.
            q: The q.
            t: The t.
            device: The device.
        """
        # q: [B, 4] (w, x, y, z)
        # t: [B, 3]
        
        q = torch.nn.functional.normalize(q, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Rotation matrix elements
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w
        
        row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=-1)
        row1 = torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=-1)
        row2 = torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
        
        R = torch.stack([row0, row1, row2], dim=1) # [B, 3, 3]
        
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = R
        pose[:, :3, 3] = t
        pose[:, 3, 3] = 1.0
        
        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        """Convert pose to 4x4.

        Args:
            B: The b.
            out_r: The out r.
            out_t: The out t.
            device: The device.
        """
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize_old(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

    def svd_orthogonalize(self, m):
        """
        Convert 9D representation to SO(3) using SVD orthogonalization.
        This is a more stable implementation using torch.linalg.svd.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        
        B = m.shape[0]

        # 1. 和原来一样: 归一化 m 的每一行，然后转置
        # m_transpose 的列向量是单位向量
        m_norm_rows = torch.nn.functional.normalize(m, p=2, dim=-1)
        m_transpose = m_norm_rows.transpose(-1, -2)

        # 2. 使用 torch.linalg.svd 替换 torch.svd
        # A = U S Vh (其中 Vh = V^T)
        try:
            u, s, vh = torch.linalg.svd(m_transpose)
        except torch.linalg.LinAlgError as e:
            # SVD 失败的罕见情况 (例如，如果输入是全零或NaN)
            print(f"SVD failed: {e}. Returning identity.")
            # 返回一个 batch 的单位矩阵
            return torch.eye(3, device=m.device, dtype=m.dtype).unsqueeze(0).expand(B, 3, 3)

        # 3. 计算 R = U @ Vh (这是正交矩阵，但可能 det(R) = -1)
        R_ortho = u @ vh

        # 4. 计算行列式 det(R)
        det = torch.det(R_ortho)

        # 5. 创建修正矩阵 D = diag(1, 1, det(R))
        # 这会处理反射(reflection)情况 (det = -1)
        # 我们需要为 batch 中的每个元素单独创建 D
        D_vec = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1)
        D = torch.diag_embed(D_vec)

        # 6. 计算 R_so3 = U @ D @ Vh
        # 这是最终的 SO(3) 旋转矩阵 R
        R = u @ D @ vh
        
        # 7. 你的原始实现返回的是 R 的转置 (R^T)
        # R^T = (U @ D @ Vh)^T = Vh.T @ D.T @ U.T = V @ D @ U^T
        # 为了作为你原始代码的“直接替换”，我们返回 R^T
        R_T = R.transpose(-1, -2)

        return R_T
    
