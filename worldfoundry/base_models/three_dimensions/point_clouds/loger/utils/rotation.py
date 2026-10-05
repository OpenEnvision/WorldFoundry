# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quaternion operations used by the LoGeR runtime."""

import torch

from worldfoundry.core.geometry.transforms import (
    quaternion_xyzw_to_rotation_matrix as quat_to_mat,
    rotation_matrix_to_quaternion_xyzw as mat_to_quat,
    standardize_quaternion_xyzw as standardize_quaternion,
)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two XYZW quaternions."""

    x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
    x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Return the multiplicative inverse of an XYZW quaternion."""

    conjugate = q.clone()
    conjugate[..., :3] = -conjugate[..., :3]
    norm_squared = (q * q).sum(dim=-1, keepdim=True)
    return conjugate / norm_squared


__all__ = [
    "mat_to_quat",
    "quat_inverse",
    "quat_multiply",
    "quat_to_mat",
    "standardize_quaternion",
]
