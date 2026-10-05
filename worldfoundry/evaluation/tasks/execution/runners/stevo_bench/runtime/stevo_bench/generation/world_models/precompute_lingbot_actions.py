"""Pre-compute LingBot-World action folders for all unique pose strings found in task YAMLs.

Run this once on the target server before running LingBot generation:

    python -m generation.world_models.precompute_lingbot_actions \\
        --tasks_root benchmark/tasks_v2 \\
        --output_dir generation/world_models/lingbot_actions \\
        --model_default "s-3, right-20, left-20"

One sub-folder is created per unique pose string (including the model default),
named by sanitizing the pose string (', ' → '_', ' ' → '_').
Each folder contains poses.npy and intrinsics.npy for the LingBot run script.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml


def _sanitize(pose_string: str) -> str:
    """Convert a pose string to a safe directory name."""
    return pose_string.replace(", ", "_").replace(" ", "_")


def _collect_pose_strings(tasks_root: Path, camera_control_field: str) -> set:
    """Walk task YAMLs and collect all unique non-null pose strings for the given field."""
    pose_strings = set()
    for d in sorted(tasks_root.rglob("*")):
        if not d.is_dir():
            continue
        yaml_path = d / f"{d.name}.yaml"
        if not yaml_path.exists():
            continue
        with yaml_path.open(encoding="utf-8") as f:
            task = yaml.safe_load(f) or {}
        cc = (task.get("camera_control") or {}).get(camera_control_field)
        if cc and str(cc).strip().lower() not in ("null", "none", ""):
            pose_strings.add(str(cc).strip())
    return pose_strings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compute LingBot action folders for all unique pose strings in task YAMLs."
    )
    parser.add_argument(
        "--tasks_root", default="benchmark/tasks_v2",
        help="Root directory of task YAML folders.",
    )
    parser.add_argument(
        "--output_dir", default="generation/world_models/lingbot_actions",
        help="Directory to write action sub-folders into.",
    )
    parser.add_argument(
        "--model_default", default="s-3, right-10, left-20, right-10",
        help="The camera_control_default from models.yaml (always pre-computed).",
    )
    parser.add_argument(
        "--hyworld_repo", default="<path_to_your_HY-WorldPlay_repo>/hyvideo",
        help="Path to HY-WorldPlay-New hyvideo package (added to sys.path).",
    )
    args = parser.parse_args()

    # Add HY-WorldPlay-New to path for generate_camera_trajectory_local
    sys.path.insert(0, args.hyworld_repo)

    # Import here so the path insertion above takes effect first
    from .generate_lingbot_poses import convert_action_to_lingbot

    tasks_root = Path(args.tasks_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Collect unique pose strings from YAMLs + always include the model default
    pose_strings = _collect_pose_strings(tasks_root, camera_control_field="lingbot")
    if args.model_default:
        pose_strings.add(args.model_default.strip())

    print(f"Found {len(pose_strings)} unique pose string(s) to pre-compute:")
    for ps in sorted(pose_strings):
        print(f"  '{ps}' -> {_sanitize(ps)}/")

    for pose_string in sorted(pose_strings):
        folder_name = _sanitize(pose_string)
        action_dir = output_dir / folder_name
        if (action_dir / "poses.npy").exists() and (action_dir / "intrinsics.npy").exists():
            print(f"\n[SKIP] Already computed: {action_dir}")
            continue
        print(f"\n[COMPUTE] '{pose_string}' -> {action_dir}")
        convert_action_to_lingbot(pose_string, output_dir=str(action_dir))

    print("\nDone. Action folders ready for LingBot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
