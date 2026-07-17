#!/usr/bin/env python3
"""Materialize Workspace demo inputs from pinned official source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOS_ROOT = REPO_ROOT / ".upstream_sources"
DEFAULT_CKPT_ROOT = REPO_ROOT.parent / "ckpt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "worldfoundry" / "data" / "test_cases"

# target path, official checkout, source path at that checkout's current HEAD
REPO_ASSETS = (
    ("astra/condition_images/garden_1.png", "Astra", "examples/condition_images/garden_1.png"),
    ("cameractrl/pose_files/0f47577ab3441480.txt", "CameraCtrl", "assets/pose_files/0f47577ab3441480.txt"),
    ("cosmos-predict2p5/base/robot_pouring.jsonl", "cosmos-predict2.5", "assets/base/robot_pouring.jsonl"),
    ("cut3r/examples/001", "CUT3R", "examples/001"),
    ("depth_anything_v3/examples/SOH", "Depth-Anything-3", "assets/examples/SOH"),
    ("dualcamctrl/demo_pic", "DualCamCtrl", "demo_pic"),
    ("fantasyworld/camera_forward.json", "fantasy-world", "examples/cameras/camera_data.json"),
    ("gen3c/image.png", "GEN3C", "assets/diffusion/000000.png"),
    ("hunyuan_game_craft/village.png", "Hunyuan-GameCraft-1.0", "asset/village.png"),
    ("hunyuan_world_voyager/case1", "HunyuanWorld-Voyager", "examples/case1"),
    ("hunyuan_worldplay/test.png", "HY-WorldPlay", "assets/img/test.png"),
    ("hunyuanvideo_i2v/0.jpg", "HunyuanVideo-I2V", "assets/demo/i2v/imgs/0.jpg"),
    ("images/000.png", "Depth-Anything-3", "assets/examples/SOH/000.png"),
    ("lingbot_world/00", "lingbot-world", "examples/00"),
    ("longcat_video/motorcycle.mp4", "LongCat-Video", "assets/motorcycle.mp4"),
    ("longvie/dense_control.mp4", "LongVie", "example/ride_horse/depth_00.mp4"),
    ("longvie/sparse_control.mp4", "LongVie", "example/ride_horse/track_00.mp4"),
    ("lyra/Lyra-1/00172.png", "lyra", "Lyra-1/assets/demo/static/diffusion_input/images/00172.png"),
    ("lyra/Lyra-2/00.png", "lyra", "Lyra-2/assets/samples/00.png"),
    ("lyra/Lyra-2/00.txt", "lyra", "Lyra-2/assets/samples/00.txt"),
    (
        "matrix-game-2/configs/inference_universal.yaml",
        "Matrix-Game",
        "Matrix-Game-2/configs/inference_yaml/inference_universal.yaml",
    ),
    ("matrix-game-2/universal/0000.png", "Matrix-Game", "Matrix-Game-2/demo_images/universal/0000.png"),
    ("matrix-game-3/001", "Matrix-Game", "Matrix-Game-3/demo_images/001"),
    ("motionctrl_conditions/camera_poses", "MotionCtrl", "examples/camera_poses"),
    ("mvdiffusion/outpaint_example.png", "MVDiffusion", "assets/outpaint_example.png"),
    ("neoverse/videos", "NeoVerse", "examples/videos"),
    ("sama/1526909-hd_1920_1080_24fps.mp4", "SAMA", "inference_example/1526909-hd_1920_1080_24fps.mp4"),
    ("scope/example_0", "SCOPE", "examples/example_0"),
    ("stable_virtual_camera/basic/blue-car.jpg", "stable-virtual-camera", "assets/basic/blue-car.jpg"),
    ("studio_demo/00/image.jpg", "HunyuanVideo-I2V", "assets/demo/i2v/imgs/0.jpg"),
    ("vggt/examples/kitchen/images", "vggt", "examples/kitchen/images"),
    ("videocrafter/i2v_prompts/horse.png", "VideoCrafter", "prompts/i2v_prompts/horse.png"),
    ("worldcam/0.mp4", "WorldCam", "data/0.mp4"),
    ("worldcam/0_intrinsics_palindrome.npy", "WorldCam", "data/0_intrinsics_palindrome.npy"),
    ("worldcam/0_poses_palindrome.npy", "WorldCam", "data/0_poses_palindrome.npy"),
    ("worldfm", "worldfm", "demo"),
)

# Assets intentionally removed from a later upstream revision but still used by
# the released model's official inference/evaluation flow.
HISTORICAL_REPO_ASSETS = (
    (
        "matrix-game-1/official_initial_image/forest_00.jpg",
        "Matrix-Game",
        "Matrix-Game-1/GameWorldScore/asset/init_image/forest/00.jpg",
        "8ed02bd612df7dcb9df252a445569359b54f1b30",
    ),
)

# target path, path relative to the shared checkpoint root
CHECKPOINT_ASSETS = (
    ("test_vla_case1/droid/exterior_image_1_left.png", "MolmoAct2-DROID/assets/sample_exterior_1_left_rgb.png"),
    ("test_vla_case1/droid/wrist_image_left.png", "MolmoAct2-DROID/assets/sample_wrist_left_rgb.png"),
    ("test_vla_case1/libero/main_view.png", "hfd_models/allenai--MolmoAct2-LIBERO/assets/sample_agentview_rgb.png"),
    ("test_vla_case1/libero/wrist_view.png", "hfd_models/allenai--MolmoAct2-LIBERO/assets/sample_wrist_rgb.png"),
)

# Catalog inputs acquired from official dataset releases rather than source repos.
EXTERNAL_ASSETS = (
    (
        "multiworld_ittakestwo/action.csv",
        "hf://datasets/Haoyuwu/MultiWorldData/480P_eval_chunk0001.tar#000100_f564185_564266.csv",
    ),
    (
        "multiworld_ittakestwo/input.png",
        "hf://datasets/Haoyuwu/MultiWorldData/480P_eval_chunk0001.tar#000100_f564185_564266.mp4:frame=0",
    ),
    (
        "test_vla_case1/aloha/observation_images_cam_high.png",
        "hf://datasets/lerobot/aloha_static_vinh_cup/videos/observation.images.cam_high/chunk-000/file-000.mp4:frame=0",
    ),
    (
        "test_vla_case1/aloha/observation_images_cam_left_wrist.png",
        "hf://datasets/lerobot/aloha_static_vinh_cup/videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4:frame=0",
    ),
    (
        "test_vla_case1/aloha/observation_images_cam_right_wrist.png",
        "hf://datasets/lerobot/aloha_static_vinh_cup/videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4:frame=0",
    ),
    (
        "test_vla_image_case1/init_frame.png",
        "hf://datasets/lerobot/aloha_static_vinh_cup/videos/observation.images.cam_high/chunk-000/file-000.mp4:frame=0",
    ),
)


def _replace_target(target: Path, source: Path, force: bool) -> str:
    if target.exists() and not force:
        return "ready"
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return "materialized"


def _repo_revision(repo: Path, revision: str = "HEAD") -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", revision], text=True).strip()


def _repo_remote(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip()


def _extract_repo_path(repo: Path, source_path: str, destination: Path, revision: str = "HEAD") -> Path:
    worktree_source = repo / source_path
    if revision == "HEAD" and worktree_source.exists():
        return worktree_source
    archive_path = destination / "source.tar"
    with archive_path.open("wb") as archive:
        subprocess.run(
            ["git", "-C", str(repo), "archive", revision, source_path],
            stdout=archive,
            check=True,
        )
    extracted_root = destination / "tree"
    extracted_root.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted_root, filter="data")
    extracted = extracted_root / source_path
    if not extracted.exists():
        raise FileNotFoundError(f"git archive did not contain {source_path}")
    return extracted


def _hash_path(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    size = 0
    for item in files:
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode())
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), size, len(files)


def _record(target: Path, status: str, source: dict[str, str]) -> dict[str, object]:
    row: dict[str, object] = {"target": str(target), "status": status, "source": source}
    if target.exists():
        sha256, size, file_count = _hash_path(target)
        row.update(sha256=sha256, size_bytes=size, file_count=file_count)
        if target.is_file() and target.stat().st_size < 1024:
            text = target.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("version https://git-lfs.github.com/spec/v1"):
                row["status"] = "lfs_pointer"
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, default=DEFAULT_REPOS_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only report current readiness.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, object]] = []
    repo_assets = tuple((*asset, "HEAD") for asset in REPO_ASSETS) + HISTORICAL_REPO_ASSETS
    for target_name, repo_name, source_name, revision in repo_assets:
        target = args.output_root / target_name
        repo = args.repos_root / repo_name
        source_meta = {"kind": "official_git", "repo": repo_name, "path": source_name}
        try:
            source_meta.update(revision=_repo_revision(repo, revision), remote=_repo_remote(repo))
            if args.check:
                status = "ready" if target.exists() else "missing"
            else:
                with tempfile.TemporaryDirectory(prefix="worldfoundry-demo-") as temp:
                    source = _extract_repo_path(repo, source_name, Path(temp), revision=revision)
                    status = _replace_target(target, source, args.force)
            rows.append(_record(target, status, source_meta))
        except (FileNotFoundError, subprocess.CalledProcessError, tarfile.TarError) as exc:
            rows.append({"target": str(target), "status": "source_missing", "source": source_meta, "error": str(exc)})

    for target_name, source_name in CHECKPOINT_ASSETS:
        target = args.output_root / target_name
        source = args.ckpt_root / source_name
        source_meta = {"kind": "official_checkpoint_asset", "path": str(source)}
        if args.check:
            status = "ready" if target.exists() else "missing"
        elif source.exists():
            status = _replace_target(target, source, args.force)
        else:
            status = "source_missing"
        rows.append(_record(target, status, source_meta))

    for target_name, source_uri in EXTERNAL_ASSETS:
        target = args.output_root / target_name
        rows.append(
            _record(
                target,
                "ready" if target.exists() else "external_pending",
                {"kind": "official_dataset", "uri": source_uri},
            )
        )

    summary = {
        "schema_version": "worldfoundry-workspace-demo-assets-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root),
        "asset_count": len(rows),
        "ready_count": sum(row["status"] in {"ready", "materialized"} for row in rows),
        "pending_count": sum(row["status"] not in {"ready", "materialized"} for row in rows),
    }
    payload = {"summary": summary, "assets": rows}
    if not args.check:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / ".workspace_demo_assets.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['status']}\t{row['target']}")
        print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pending_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
