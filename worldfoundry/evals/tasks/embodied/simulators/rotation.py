"""Rotation conversion utilities for VLA action spaces.

Two rot6d memory layouts coexist in this project:

  - **Interleaved**: ``mat[:, :2].reshape(6)`` → ``[r00, r01, r10, r11, r20, r21]``
    Used by all benchmark code and the official X-VLA evaluation scripts.
    Columns are recovered with stride-2: ``col0 = v[0::2]``, ``col1 = v[1::2]``.

  - **Contiguous**: ``mat[:, :2].T.flatten()`` → ``[r00, r10, r20, r01, r11, r21]``
    Used by the X-VLA model server (``xvla.py``) for internal
    axis-angle ↔ rot6d conversion.

All functions are pure NumPy with no mandatory external dependencies.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Core: Gram-Schmidt orthonormalization
# ---------------------------------------------------------------------------


def gram_schmidt(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Compute a 3x3 orthonormal rotation matrix from two 3D vectors via Gram-Schmidt process.

    ``a1`` becomes the first column (after normalisation), ``a2`` is
    orthogonalised against it to form the second column, and the third
    column is their cross product.

    Args:
        a1: A 3D numpy array.
        a2: A 3D numpy array.

    Returns:
        A 3x3 orthonormal rotation matrix.
    """
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


# ---------------------------------------------------------------------------
# Interleaved rot6d  (benchmark convention)
# ---------------------------------------------------------------------------


def rot6d_interleaved_to_matrix(v6: np.ndarray) -> np.ndarray:
    """Convert an interleaved 6D rotation vector into a 3x3 rotation matrix.

    Args:
        v6: Interleaved 6-D rotation array (length 6).

    Returns:
        The reconstructed 3x3 rotation matrix.
    """
    return gram_schmidt(v6[0::2], v6[1::2])


def matrix_to_rot6d_interleaved(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into an interleaved 6D rotation representation.

    Args:
        mat: A 3x3 rotation matrix.

    Returns:
        The interleaved 6D rotation array (length 6).
    """
    return mat[:, :2].reshape(6).copy()


# ---------------------------------------------------------------------------
# Contiguous rot6d  (model-server convention)
# ---------------------------------------------------------------------------


def matrix_to_rot6d_contiguous(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → contiguous 6-D rotation.

    Args:
        mat: A 3x3 rotation matrix.

    Returns:
        The contiguous 6D rotation array.
    """
    return mat[:, :2].T.flatten().copy()


# ---------------------------------------------------------------------------
# Quaternion ↔ matrix  (scalar-last: [x, y, z, w])
# ---------------------------------------------------------------------------


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into a quaternion [x, y, z, w] using Shepperd's method.

    Args:
        mat: A 3x3 rotation matrix array.

    Returns:
        A quaternion array in [x, y, z, w] layout.
    """
    m = np.asarray(mat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] into a 3x3 rotation matrix.

    Args:
        q: A quaternion array in [x, y, z, w] layout.

    Returns:
        A 3x3 rotation matrix.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# Quaternion ↔ axis-angle
# ---------------------------------------------------------------------------


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] into axis-angle representation (3D).

    Args:
        quat: A quaternion array in [x, y, z, w] layout.

    Returns:
        An axis-angle 3D array (rotation vector).
    """
    q = -quat if quat[3] < 0 else quat.copy()
    w = float(np.clip(q[3], -1.0, 1.0))
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (q[:3] / sin_half * angle).astype(np.float32)


def axisangle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert an axis-angle representation (3D) into a 3x3 rotation matrix via Rodrigues' formula.

    Args:
        aa: An axis-angle 3D array (rotation vector).

    Returns:
        A 3x3 rotation matrix.
    """
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Euler XYZ ↔ matrix  (extrinsic XYZ, radians)
# ---------------------------------------------------------------------------


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert extrinsic XYZ Euler angles (radians) into a 3x3 rotation matrix.

    Equivalent to ``scipy.spatial.transform.Rotation.from_euler('xyz', euler).as_matrix()``.

    Args:
        euler: An array of 3 Euler angles [x, y, z] in radians.

    Returns:
        The corresponding 3x3 rotation matrix.
    """
    x, y, z = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    # R = Rz @ Ry @ Rx  (extrinsic XYZ = intrinsic ZYX)
    return np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ]
    )


def matrix_to_euler_xyz(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into extrinsic XYZ Euler angles (radians).

    Equivalent to ``scipy.spatial.transform.Rotation.from_matrix(m).as_euler('xyz')``.

    Args:
        mat: A 3x3 rotation matrix array.

    Returns:
        An array of 3 Euler angles [x, y, z] in radians.
    """
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(mat[2, 1], mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])


# ── Quaternion convention helpers ──────────────────────────────────────
# ManiSkill2/transforms3d use wxyz; our rotation utilities use xyzw.


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from ``[w, x, y, z]`` to ``[x, y, z, w]``.

    Args:
        q: A quaternion array in WXYZ layout.

    Returns:
        A quaternion array in XYZW layout.
    """
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from ``[x, y, z, w]`` to ``[w, x, y, z]``.

    Args:
        q: A quaternion array in XYZW layout.

    Returns:
        A quaternion array in WXYZ layout.
    """
    return np.array([q[3], q[0], q[1], q[2]])


def pose7_wxyz_to_mat4(p: np.ndarray) -> np.ndarray:
    """Convert a 7D pose ``[x, y, z, qw, qx, qy, qz]`` to a 4x4 matrix.

    Args:
        p: A 7D pose array containing translation and a wxyz quaternion.

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(quat_wxyz_to_xyzw(p[3:7]))
    m[:3, 3] = p[:3]
    return m
