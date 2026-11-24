#!/usr/bin/env python3

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
from worldfoundry.evals.tasks.execution.runners.vbench.vbench_official_impl import (
    IN_TREE_VBENCH_ROOT,
    VBENCH_FULL_INFO_ASSET,
)

SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".flv", ".webm")
METRICS_REQUIRING_REFERENCE_NPZ = {
    "all",
    "trajectory",
    "traj_tolerance",
    "trajectory_following",
    "action_control",
    "camera_following",
}
METRICS_REQUIRING_VBENCH = {
    "all",
    "vbench",
    "imaging_quality",
    "motion_smoothness",
    "generation_quality",
    "trajectory_following",
    "action_control",
    "camera_following",
}
METRICS_REQUIRING_CAMERA_TXT = {"all", "trajectory", "traj_accuracy", "trajectory_following", "action_control"}
CAMERA_TRAJECTORIES_ASSET_ROOT = bundled_benchmark_asset("iworld-bench", "camera_trajectories")
DEFAULT_CAMERA_TXT_DIR = CAMERA_TRAJECTORIES_ASSET_ROOT / "inference_txt"
DEFAULT_REFERENCE_NPZ_DIR = CAMERA_TRAJECTORIES_ASSET_ROOT / "reference_npz"
DEFAULT_SOURCE_REFERENCE_NPZ_DIR = CAMERA_TRAJECTORIES_ASSET_ROOT / "source_reference_npz"


def _has_video_files(video_dir: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS for p in video_dir.rglob("*"))


def _require_dir(path: str, name: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{name} not found or not a directory: {p}")
    return p


def _prepend_pythonpath(paths: Iterable[Path]) -> None:
    valid_paths = [str(p) for p in paths if p and p.exists()]
    for p in reversed(valid_paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    if valid_paths:
        old = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(valid_paths + ([old] if old else []))


def _default_vbench_root() -> str:
    return str(IN_TREE_VBENCH_ROOT.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click iWorldBench video metric runner with preflight checks.")
    parser.add_argument("video_dir", help="Directory containing generated videos")
    parser.add_argument("save_dir", help="Output directory for reports and caches")
    parser.add_argument(
        "--metric",
        default="all",
        choices=[
            "all",
            "brightness",
            "color_temperature",
            "noise",
            "sharpness",
            "memory",
            "traj_accuracy",
            "traj_alignment",
            "traj_tolerance",
            "imaging_quality",
            "motion_smoothness",
            "generation_quality",
            "trajectory_following",
            "memory_ability",
            "action_control",
            "camera_following",
            "video_quality",
            "trajectory",
            "vbench",
        ],
    )
    parser.add_argument(
        "--source-npz-dir",
        default=None,
        help="Target/GT/reference trajectory NPZ directory for trajectory tolerance; defaults to source_reference_npz for camera_following and reference_npz otherwise",
    )
    parser.add_argument(
        "--camera-txt-dir",
        default=str(DEFAULT_CAMERA_TXT_DIR),
        help="Directory containing packaged camera_*.txt and memory_*.txt controls for trajectory accuracy",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--vbench-gpu", default="0")
    parser.add_argument("--vbench-overwrite", action="store_true")
    parser.add_argument("--skip-failed", dest="retry_failed", action="store_false", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit VBench video count")
    parser.add_argument(
        "--iworld-root", default=os.environ.get("IWORLD_BENCH_ROOT", str(Path(__file__).resolve().parent))
    )
    parser.add_argument("--vipe-root", default=os.environ.get("VIPE_ROOT"))
    parser.add_argument("--vbench-root", default=os.environ.get("VBENCH_ROOT") or _default_vbench_root())
    parser.add_argument(
        "--vbench-site-packages", default=None, help="Optional site-packages path containing pyiqa/other VBench deps"
    )
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    video_dir = _require_dir(args.video_dir, "video_dir")
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    iworld_root = _require_dir(args.iworld_root, "iworld-root")

    if not _has_video_files(video_dir):
        raise FileNotFoundError(f"No supported video files found under {video_dir}")

    source_npz_dir = None
    if args.metric in METRICS_REQUIRING_REFERENCE_NPZ:
        source_npz_dir_arg = args.source_npz_dir or str(
            DEFAULT_SOURCE_REFERENCE_NPZ_DIR if args.metric == "camera_following" else DEFAULT_REFERENCE_NPZ_DIR
        )
        if not source_npz_dir_arg:
            raise ValueError(
                f"--metric {args.metric} includes trajectory tolerance, so --source-npz-dir is required. "
                "Use traj_accuracy/traj_alignment if you only want non-reference trajectory metrics."
            )
        source_npz_dir = _require_dir(source_npz_dir_arg, "source-npz-dir")
        if not any(source_npz_dir.glob("*.npz")):
            raise FileNotFoundError(f"No .npz files found under source-npz-dir: {source_npz_dir}")

    camera_txt_dir = None
    if args.metric in METRICS_REQUIRING_CAMERA_TXT:
        camera_txt_dir = _require_dir(args.camera_txt_dir, "camera-txt-dir")
        if not any(camera_txt_dir.glob("camera_*.txt")) and not any(camera_txt_dir.glob("memory_*.txt")):
            raise FileNotFoundError(
                f"No camera_*.txt or memory_*.txt files found under camera-txt-dir: {camera_txt_dir}"
            )

    if args.vipe_root:
        vipe_root = _require_dir(args.vipe_root, "vipe-root")
        os.environ["VIPE_ROOT"] = str(vipe_root)

    vbench_root = None
    if args.vbench_root:
        vbench_root = _require_dir(args.vbench_root, "vbench-root")
        os.environ["VBENCH_ROOT"] = str(vbench_root)
    if args.metric in METRICS_REQUIRING_VBENCH:
        vbench_manifest = (
            Path(os.environ["WORLDFOUNDRY_VBENCH_FULL_INFO"]).expanduser()
            if os.environ.get("WORLDFOUNDRY_VBENCH_FULL_INFO")
            else VBENCH_FULL_INFO_ASSET
        )
        if (
            vbench_root
            and not (vbench_root / "vbench" / "VBench_full_info.json").is_file()
            and not vbench_manifest.is_file()
        ):
            raise FileNotFoundError(f"WorldFoundry VBench runtime metadata not found: {vbench_root}")

    extra_pythonpaths = [iworld_root]
    if args.vbench_site_packages:
        extra_pythonpaths.append(_require_dir(args.vbench_site_packages, "vbench-site-packages"))
    if vbench_root:
        extra_pythonpaths.append(vbench_root)
    _prepend_pythonpath(extra_pythonpaths)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("Evaluation config:")
    print(f"  metric: {args.metric}")
    print(f"  video_dir: {video_dir}")
    print(f"  save_dir: {save_dir}")
    print(f"  source_npz_dir: {source_npz_dir}")
    print(f"  camera_txt_dir: {camera_txt_dir}")
    print(f"  iworld_root: {iworld_root}")
    print(f"  vipe_root: {os.environ.get('VIPE_ROOT')}")
    print(f"  vbench_root: {os.environ.get('VBENCH_ROOT')}")
    print(f"  vbench_gpu: {args.vbench_gpu}")

    from unified_video_metrics import (
        calculate_action_control,
        calculate_all,
        calculate_all_trajectory,
        calculate_all_vbench,
        calculate_all_video_quality,
        calculate_brightness_consistency,
        calculate_camera_following,
        calculate_color_temperature,
        calculate_generation_quality,
        calculate_imaging_quality,
        calculate_memory_ability,
        calculate_memory_symmetry,
        calculate_motion_smoothness,
        calculate_sharpness_retention,
        calculate_trajectory_accuracy,
        calculate_trajectory_alignment,
        calculate_trajectory_following,
        calculate_trajectory_tolerance,
        calculate_video_noise,
    )

    dispatch = {
        "all": lambda: calculate_all(
            str(video_dir),
            str(save_dir),
            source_npz_dir=str(source_npz_dir),
            max_workers=args.max_workers,
            vbench_gpu=args.vbench_gpu,
            vbench_overwrite=args.vbench_overwrite,
            retry_failed=args.retry_failed,
            limit=args.limit,
            camera_txt_dir=str(camera_txt_dir),
        ),
        "generation_quality": lambda: calculate_generation_quality(
            str(video_dir),
            str(save_dir),
            max_workers=args.max_workers,
            vbench_gpu=args.vbench_gpu,
            vbench_overwrite=args.vbench_overwrite,
            retry_failed=args.retry_failed,
            limit=args.limit,
        ),
        "trajectory_following": lambda: calculate_trajectory_following(
            str(video_dir),
            str(save_dir),
            source_npz_dir=str(source_npz_dir),
            max_workers=args.max_workers,
            camera_txt_dir=str(camera_txt_dir),
            vbench_gpu=args.vbench_gpu,
            vbench_overwrite=args.vbench_overwrite,
            retry_failed=args.retry_failed,
            limit=args.limit,
        ),
        "memory_ability": lambda: calculate_memory_ability(str(video_dir), str(save_dir), args.max_workers),
        "action_control": lambda: calculate_action_control(
            str(video_dir),
            str(save_dir),
            source_npz_dir=str(source_npz_dir),
            max_workers=args.max_workers,
            camera_txt_dir=str(camera_txt_dir),
            vbench_gpu=args.vbench_gpu,
            vbench_overwrite=args.vbench_overwrite,
            retry_failed=args.retry_failed,
            limit=args.limit,
        ),
        "camera_following": lambda: calculate_camera_following(
            str(video_dir),
            str(save_dir),
            source_npz_dir=str(source_npz_dir),
            max_workers=args.max_workers,
            vbench_gpu=args.vbench_gpu,
            vbench_overwrite=args.vbench_overwrite,
            retry_failed=args.retry_failed,
            limit=args.limit,
        ),
        "video_quality": lambda: calculate_all_video_quality(str(video_dir), str(save_dir), args.max_workers),
        "trajectory": lambda: calculate_all_trajectory(
            str(video_dir), str(save_dir), str(source_npz_dir), args.max_workers, str(camera_txt_dir)
        ),
        "vbench": lambda: calculate_all_vbench(
            str(video_dir), str(save_dir), args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit
        ),
        "brightness": lambda: calculate_brightness_consistency(str(video_dir), str(save_dir), args.max_workers),
        "color_temperature": lambda: calculate_color_temperature(str(video_dir), str(save_dir), args.max_workers),
        "noise": lambda: calculate_video_noise(str(video_dir), str(save_dir), args.max_workers),
        "sharpness": lambda: calculate_sharpness_retention(str(video_dir), str(save_dir), args.max_workers),
        "memory": lambda: calculate_memory_symmetry(str(video_dir), str(save_dir), args.max_workers),
        "traj_accuracy": lambda: calculate_trajectory_accuracy(
            str(video_dir), str(save_dir), args.max_workers, str(camera_txt_dir)
        ),
        "traj_alignment": lambda: calculate_trajectory_alignment(str(video_dir), str(save_dir), args.max_workers),
        "traj_tolerance": lambda: calculate_trajectory_tolerance(
            str(video_dir), str(save_dir), str(source_npz_dir), args.max_workers
        ),
        "imaging_quality": lambda: calculate_imaging_quality(
            str(video_dir), str(save_dir), args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit
        ),
        "motion_smoothness": lambda: calculate_motion_smoothness(
            str(video_dir), str(save_dir), args.vbench_gpu, args.vbench_overwrite, args.retry_failed, args.limit
        ),
    }
    dispatch[args.metric]()
    print(f"Done. CSV reports are under: {save_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
