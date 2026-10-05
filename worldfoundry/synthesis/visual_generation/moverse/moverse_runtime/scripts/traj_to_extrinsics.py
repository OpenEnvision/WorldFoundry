#!/usr/bin/env python3
"""Trajectory keypoint sequence → world-to-camera (w2c) matrices.

Reads a 3-line trajectory file:
  Line 1: theta sequence (degrees, positive=up)
  Line 2: phi sequence (degrees, positive=right)
  Line 3: r sequence (× scene radius, positive=forward)

Interpolates keypoint sequences to num_frames using cubic spline,
then converts each (theta, phi, r) to a 4×4 w2c matrix.

Coordinate system:
  - World Y up, Camera Z forward, Camera X right
  - R = R_pitch @ R_yaw (global rotation: yaw around world Y, pitch around local X)
  - C = R^T @ [0, 0, r * radius]  (camera world position)
  - w2c = [R | -R @ C]

Usage:
  python scripts/traj_to_extrinsics.py --traj examples/alcove/trajectory.txt --num_frames 80
"""

import argparse
import numpy as np
import torch
from scipy.interpolate import CubicSpline


def load_trajectory(file_path: str) -> tuple:
    """Load trajectory file.

    Format: 3 lines of whitespace-separated floats, with optional comment lines
    starting with '#'. Lines are read in order:
      1. theta (degrees, positive=up)
      2. phi   (degrees, positive=right)
      3. r     (× scene radius, positive=forward)

    Returns:
        (theta_seq, phi_seq, r_seq) as numpy float64 arrays
    """
    sequences = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if values:
                sequences.append(np.array(values, dtype=np.float64))

    if len(sequences) < 3:
        raise ValueError(
            f"Trajectory file must have 3 sequences (theta, phi, r), "
            f"got {len(sequences)}"
        )

    theta_seq, phi_seq, r_seq = sequences[0], sequences[1], sequences[2]

    if not (len(theta_seq) == len(phi_seq) == len(r_seq)):
        raise ValueError(
            f"Sequence length mismatch: theta={len(theta_seq)}, "
            f"phi={len(phi_seq)}, r={len(r_seq)}"
        )

    return theta_seq, phi_seq, r_seq


def trajectory_to_w2cs(
    theta_seq: np.ndarray,
    phi_seq: np.ndarray,
    r_seq: np.ndarray,
    num_frames: int = 80,
    radius: float = 1.0,
) -> torch.Tensor:
    """Interpolate trajectory keypoint sequences and convert to w2c matrices.

    Args:
        theta_seq: pitch keypoint sequence (degrees, positive=up)
        phi_seq: yaw keypoint sequence (degrees, positive=right)
        r_seq: forward distance keypoint sequence (× radius, positive=forward)
        num_frames: number of output frames
        radius: scene radius (world units)

    Returns:
        w2c_matrices: (num_frames, 4, 4) float32 tensor
    """
    n_keypoints = len(theta_seq)
    if n_keypoints < 2:
        raise ValueError("Need at least 2 keypoint values per sequence")

    # Normalized keyframe positions
    t_keypoints = np.linspace(0, 1, n_keypoints)
    t_frames = np.linspace(0, 1, num_frames)

    # Cubic spline interpolation
    cs_theta = CubicSpline(t_keypoints, theta_seq)
    cs_phi = CubicSpline(t_keypoints, phi_seq)
    cs_r = CubicSpline(t_keypoints, r_seq)

    theta_interp = cs_theta(t_frames)
    phi_interp = cs_phi(t_frames)
    r_interp = cs_r(t_frames)

    # Convert to w2c matrices
    w2c_list = []
    for i in range(num_frames):
        theta_deg = float(theta_interp[i])   # positive = up
        phi_deg = float(phi_interp[i])        # positive = right
        r = float(r_interp[i]) * radius       # forward distance (world units)

        # Yaw is positive right; pitch is positive down, so negate theta.
        theta_yaw = np.radians(phi_deg)
        theta_pitch = np.radians(-theta_deg)

        cy, sy = np.cos(theta_yaw), np.sin(theta_yaw)
        cp, sp = np.cos(theta_pitch), np.sin(theta_pitch)

        # R = R_pitch @ R_yaw
        R = np.array([
            [ cy,      0.0,  -sy     ],
            [-sp * sy,  cp,  -sp * cy],
            [ cp * sy,  sp,   cp * cy],
        ], dtype=np.float64)

        # Camera world position: C = R^T @ [0, 0, r]
        # (move forward by r in camera's local +Z direction)
        C = R.T @ np.array([0.0, 0.0, r], dtype=np.float64)

        # w2c = [R | -R @ C]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ C

        w2c_list.append(w2c)

    return torch.from_numpy(np.stack(w2c_list))


def main():
    p = argparse.ArgumentParser(
        description="Convert trajectory file to w2c extrinsic matrices"
    )
    p.add_argument("--traj", type=str, required=True, help="Trajectory file path")
    p.add_argument("--num_frames", type=int, default=80, help="Number of output frames")
    p.add_argument("--radius", type=float, default=1.0, help="Scene radius (world units)")
    p.add_argument("--output", type=str, default=None, help="Output .npy file (optional)")
    args = p.parse_args()

    theta_seq, phi_seq, r_seq = load_trajectory(args.traj)
    print(f"Loaded trajectory: {len(theta_seq)} keypoints")
    print(f"  theta: {theta_seq}")
    print(f"  phi:   {phi_seq}")
    print(f"  r:     {r_seq}")

    w2cs = trajectory_to_w2cs(theta_seq, phi_seq, r_seq, args.num_frames, args.radius)
    print(f"\nGenerated {w2cs.shape[0]} w2c matrices, shape: {list(w2cs.shape)}")

    if args.output:
        np.save(args.output, w2cs.numpy())
        print(f"Saved to: {args.output}")
    else:
        for i in range(min(5, w2cs.shape[0])):
            print(f"\nFrame {i}:")
            print(w2cs[i])


if __name__ == "__main__":
    main()
