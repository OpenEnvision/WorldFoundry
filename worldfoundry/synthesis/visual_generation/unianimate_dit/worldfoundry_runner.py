from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import resolve_data_path


def _demo_asset_root() -> Path:
    override = os.getenv("WORLDFOUNDRY_UNIANIMATE_DIT_DEMO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return resolve_data_path("test_cases", "unianimate_dit").resolve()


DEFAULT_REF_IMAGE = "data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg"
DEFAULT_SOURCE_VIDEO = "data/videos/source_video.mp4"
DEFAULT_POSE_DIR = "data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full"


def _ensure_link(link_path: Path, target_path: Path) -> None:
    target_path = target_path.expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"required UniAnimate asset path does not exist: {target_path}")
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        raise FileExistsError(f"refusing to replace existing path: {link_path}")
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _as_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _newest_mp4(root: Path, *, since: float) -> Path | None:
    candidates = [path for path in root.rglob("*.mp4") if path.is_file() and path.stat().st_mtime >= since]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_generated_panel(source_path: Path, output_path: Path) -> bool:
    """Crop UniAnimate's official three-panel comparison video to the generated panel."""
    try:
        import cv2
    except Exception:
        return False

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if width < 3 or height <= 0:
        cap.release()
        return False
    panel_width = width // 3
    if panel_width <= 0:
        cap.release()
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width, height),
    )
    if not writer.isOpened():
        cap.release()
        return False
    wrote = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame[:, width - panel_width : width])
            wrote = True
    finally:
        cap.release()
        writer.release()
    if not wrote:
        tmp_path.unlink(missing_ok=True)
        return False
    tmp_path.replace(output_path)
    return True


def _ensure_demo_assets(workspace_root: Path) -> None:
    _ensure_link(workspace_root / "data", _demo_asset_root())


def _ensure_demo_pose(workspace_root: Path, reference_image: str) -> None:
    pose_dir = workspace_root / DEFAULT_POSE_DIR
    if (pose_dir / "pose.jpg").is_file() and any(
        path.name != "pose.jpg" and path.suffix.lower() in {".jpg", ".png"} for path in pose_dir.iterdir()
    ):
        return
    pose_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        package_module_command(
            "worldfoundry.synthesis.visual_generation.unianimate_dit.unianimate_dit_runtime.run_align_pose",
            [
                "--ref_name",
                reference_image,
                "--source_video_paths",
                DEFAULT_SOURCE_VIDEO,
                "--saved_pose_dir",
                DEFAULT_POSE_DIR,
            ],
        ),
        cwd=str(workspace_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "UniAnimate pose alignment failed before inference:\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry UniAnimate-DiT official-demo runner")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-usp", default="false")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root = output_path.parent / "unianimate_dit_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    _ensure_link(workspace_root / "Wan2.1-I2V-14B-720P", Path(args.wan_root))
    _ensure_link(workspace_root / "checkpoints", Path(args.checkpoint_dir))
    _ensure_demo_assets(workspace_root)
    reference_image = str(Path(args.image_path).expanduser().resolve()) if args.image_path else DEFAULT_REF_IMAGE
    if args.image_path and not Path(reference_image).is_file():
        raise FileNotFoundError(f"UniAnimate reference image not found: {reference_image}")
    _ensure_demo_pose(workspace_root, reference_image)

    before = time.time()
    env = os.environ.copy()
    use_usp = _as_bool(args.use_usp)
    env["WORLDFOUNDRY_UNIANIMATE_USE_USP"] = "1" if use_usp else "0"
    command = package_module_command(
        "worldfoundry.synthesis.visual_generation.unianimate_dit.unianimate_dit_runtime.examples."
        "unianimate_wan.inference_unianimate_wan_480p",
        [
            "--prompt",
            args.prompt or "a person is dancing",
            "--image-path",
            reference_image,
            "--max-frames",
            args.max_frames,
            "--num-inference-steps",
            args.num_inference_steps,
            "--height",
            args.height,
            "--width",
            args.width,
            "--seed",
            args.seed,
        ],
        nproc_per_node=max(int(args.nproc_per_node), 1) if use_usp else None,
    )
    completed = subprocess.run(
        command,
        cwd=str(workspace_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        log_path = output_path.with_suffix(output_path.suffix + ".runner.log")
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        raise RuntimeError(f"UniAnimate official demo failed with code {completed.returncode}; see {log_path}")

    produced = _newest_mp4(workspace_root / "outputs", since=before)
    if produced is None:
        produced = _newest_mp4(workspace_root, since=before)
    if produced is None:
        raise RuntimeError("UniAnimate official demo finished but no mp4 artifact was produced.")
    composite_path = output_path.with_name(f"{output_path.stem}_official_composite{output_path.suffix}")
    shutil.copy2(produced, composite_path)
    if not _extract_generated_panel(produced, output_path):
        shutil.copy2(produced, output_path)


if __name__ == "__main__":
    main()
