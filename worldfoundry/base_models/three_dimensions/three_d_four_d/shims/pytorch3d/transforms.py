from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions to rotation matrices.

    This mirrors the small PyTorch3D transform subset WorldGen imports at
    inference time, avoiding a full compiled PyTorch3D dependency for demo runs.
    """
    q = quaternions
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion_to_matrix expects last dimension 4, got {q.shape}")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to real-first quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix_to_quaternion expects shape (..., 3, 3), got {matrix.shape}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))
    qx = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0))
    qy = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0))
    qz = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0))
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(quat.dtype).eps)
