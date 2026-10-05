"""
Convert HY-WorldPlay-New action string to LingBot-World poses.npy format.

HY-WorldPlay-New uses one pose per latent step.
LingBot-World expects one pose per video frame (typically 81+ frames).

This script generates frame-level poses by expanding each latent step
into multiple per-frame motions with the same per-step speeds used by
HY-WorldPlay-New (forward_speed=0.08, yaw_speed=3°, pitch_speed=3°).
"""
import numpy as np
import sys
import os

# Add HY-WorldPlay-New to path
sys.path.insert(0, '<path_to_your_HY-WorldPlay_repo>/hyvideo')
from generate_custom_trajectory import generate_camera_trajectory_local


# Per-frame motion speeds (same as HY-WorldPlay-New's parse_pose_string)
FORWARD_SPEED = 0.08
YAW_SPEED = np.deg2rad(3)
PITCH_SPEED = np.deg2rad(3)
FRAMES_PER_LATENT = 4


def parse_action_string(action_string):
    """
    Parse HY-WorldPlay-New action string and expand to frame-level motions.

    In HY-WorldPlay-New, the number after the dash is the number of latent
    steps.  Each latent step covers FRAMES_PER_LATENT video frames.
    Total video frames = total_latents * 4 + 1 (matching HY-WorldPlay-New).

    Args:
        action_string: e.g. 'right-11', 'w-4, d-4', 'right-11, w-5'

    Returns:
        tuple: (motions list, total_latents)
    """
    commands = [cmd.strip() for cmd in action_string.split(",")]
    motions = []
    total_latents = 0

    for cmd in commands:
        if not cmd:
            continue
        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid command: {cmd}. Expected 'action-duration'")

        action = parts[0].strip()
        num_latents = int(parts[1].strip())
        total_latents += num_latents
        # Each latent step = FRAMES_PER_LATENT video frames of motion
        num_frames = num_latents * FRAMES_PER_LATENT

        motion = {}
        # Translations
        if action == "w":
            motion = {"forward": FORWARD_SPEED}
        elif action == "s":
            motion = {"forward": -FORWARD_SPEED}
        elif action == "a":
            motion = {"right": -FORWARD_SPEED}
        elif action == "d":
            motion = {"right": FORWARD_SPEED}
        # Rotations
        elif action == "up":
            motion = {"pitch": PITCH_SPEED}
        elif action == "down":
            motion = {"pitch": -PITCH_SPEED}
        elif action == "left":
            motion = {"yaw": -YAW_SPEED}
        elif action == "right":
            motion = {"yaw": YAW_SPEED}
        # Combined rotations
        elif action == "rightup":
            motion = {"yaw": YAW_SPEED, "pitch": PITCH_SPEED}
        elif action == "rightdown":
            motion = {"yaw": YAW_SPEED, "pitch": -PITCH_SPEED}
        elif action == "leftup":
            motion = {"yaw": -YAW_SPEED, "pitch": PITCH_SPEED}
        elif action == "leftdown":
            motion = {"yaw": -YAW_SPEED, "pitch": -PITCH_SPEED}
        # Combined translations
        elif action in ("wd", "dw"):
            motion = {"forward": FORWARD_SPEED, "right": FORWARD_SPEED}
        elif action in ("wa", "aw"):
            motion = {"forward": FORWARD_SPEED, "right": -FORWARD_SPEED}
        elif action in ("sd", "ds"):
            motion = {"forward": -FORWARD_SPEED, "right": FORWARD_SPEED}
        elif action in ("sa", "as"):
            motion = {"forward": -FORWARD_SPEED, "right": -FORWARD_SPEED}
        # Combined translation + rotation
        elif action == "wright":
            motion = {"forward": FORWARD_SPEED, "yaw": YAW_SPEED}
        elif action == "wleft":
            motion = {"forward": FORWARD_SPEED, "yaw": -YAW_SPEED}
        elif action == "sright":
            motion = {"forward": -FORWARD_SPEED, "yaw": YAW_SPEED}
        elif action == "sleft":
            motion = {"forward": -FORWARD_SPEED, "yaw": -YAW_SPEED}
        elif action == "dright":
            motion = {"right": FORWARD_SPEED, "yaw": YAW_SPEED}
        elif action == "dleft":
            motion = {"right": FORWARD_SPEED, "yaw": -YAW_SPEED}
        elif action == "aright":
            motion = {"right": -FORWARD_SPEED, "yaw": YAW_SPEED}
        elif action == "aleft":
            motion = {"right": -FORWARD_SPEED, "yaw": -YAW_SPEED}
        elif action == "wup":
            motion = {"forward": FORWARD_SPEED, "pitch": PITCH_SPEED}
        elif action == "wdown":
            motion = {"forward": FORWARD_SPEED, "pitch": -PITCH_SPEED}
        elif action == "sup":
            motion = {"forward": -FORWARD_SPEED, "pitch": PITCH_SPEED}
        elif action == "sdown":
            motion = {"forward": -FORWARD_SPEED, "pitch": -PITCH_SPEED}
        else:
            raise ValueError(f"Unknown action: {action}")

        for _ in range(num_frames):
            motions.append(motion.copy())

    return motions, total_latents


def convert_action_to_lingbot(action_string, output_dir='./'):
    """
    Convert HY-WorldPlay-New action string to LingBot-World format.

    Video length is determined by the action string: total_latents * 4 + 1,
    matching HY-WorldPlay-New's convention.

    No coordinate convention conversion is applied: LingBot's
    compute_relative_poses normalises everything relative to the first frame,
    making the result coordinate-convention-independent.  Applying a Y-flip
    (det = -1) would break the SLERP quaternion interpolation in
    interpolate_camera_poses.

    Args:
        action_string: HY-WorldPlay action string (e.g. 'right-11', 'w-4, d-4')
        output_dir: Directory to save poses.npy and intrinsics.npy
    """
    motions, total_latents = parse_action_string(action_string)
    target_frames = total_latents * FRAMES_PER_LATENT + 1
    print(f"Action '{action_string}': {total_latents} latents -> {target_frames} frames")
    print(f"Generated {len(motions)} frame-level motions")

    # Generate camera trajectory (c2w matrices) — returns len(motions)+1 poses
    poses_c2w = generate_camera_trajectory_local(motions)
    poses_c2w = np.array(poses_c2w, dtype=np.float32)
    # Truncate or pad to exactly target_frames
    if len(poses_c2w) > target_frames:
        poses_c2w = poses_c2w[:target_frames]
    elif len(poses_c2w) < target_frames:
        extra = target_frames - len(poses_c2w)
        poses_c2w = np.concatenate([poses_c2w, np.tile(poses_c2w[-1:], (extra, 1, 1))])
    print(f"Trajectory: {len(poses_c2w)} poses for {target_frames} frames")

    # Intrinsics for 832x480 (LingBot default resolution)
    fx, fy, cx, cy = 502.9, 503.1, 415.8, 239.8
    intrinsics = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (len(poses_c2w), 1))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'poses.npy'), poses_c2w)
    np.save(os.path.join(output_dir, 'intrinsics.npy'), intrinsics)

    print(f"\nSaved to {output_dir}:")
    print(f"  poses.npy: shape {poses_c2w.shape}")
    print(f"  intrinsics.npy: shape {intrinsics.shape}")
    print(f"\nFirst pose:\n{poses_c2w[0]}")
    print(f"\nLast pose:\n{poses_c2w[-1]}")

    # Sanity check: what LingBot will do with these poses
    len_c2ws = ((len(poses_c2w) - 1) // 4) * 4 + 1
    frame_num = min(target_frames, len_c2ws)
    n_keyframes = (frame_num - 1) // 4 + 1
    print(f"\nLingBot will use: {frame_num} frames -> {n_keyframes} keyframes")

    return poses_c2w, intrinsics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HY-WorldPlay-New actions to LingBot-World poses")
    parser.add_argument("action", nargs="?", default="right-11",
                        help="Action string (default: 'right-11')")
    parser.add_argument("--output", default="./examples/hyworld_converted",
                        help="Output directory")
    args = parser.parse_args()

    convert_action_to_lingbot(args.action, output_dir=args.output)