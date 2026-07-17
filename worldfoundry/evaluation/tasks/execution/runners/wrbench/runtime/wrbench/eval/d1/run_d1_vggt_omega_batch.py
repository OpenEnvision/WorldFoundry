#!/usr/bin/env python3
"""Run VGGT-Omega pose export for a filtered D1 candidate shard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from wrbench.eval.d1.geometry import safe_video_id
    from wrbench.eval.d1.hydra_segments import resolve_eval_video_for_row
    from wrbench.eval.d1.run_vggt_omega_pose import (
        _load_state_dict,
        extract_frames,
        safe_scene_name,
        write_pose_outputs,
    )
except ImportError:  # pragma: no cover
    from .geometry import safe_video_id
    from .hydra_segments import resolve_eval_video_for_row
    from .run_vggt_omega_pose import _load_state_dict, extract_frames, safe_scene_name, write_pose_outputs


class InprocessVGGTOmegaRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        vggt_repo = Path(args.vggt_repo).resolve()
        if str(vggt_repo) not in sys.path:
            sys.path.insert(0, str(vggt_repo))
        import torch  # type: ignore
        from vggt_omega.models import VGGTOmega  # type: ignore
        from vggt_omega.utils.load_fn import load_and_preprocess_images  # type: ignore
        from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore

        self.torch = torch
        self.load_and_preprocess_images = load_and_preprocess_images
        self.encoding_to_camera = encoding_to_camera
        self.device = "cuda"
        self.model = VGGTOmega().to(self.device).eval()
        self.model.load_state_dict(_load_state_dict(torch, Path(args.checkpoint)), strict=True)

    def run_one(self, *, video_path: Path, scene_name: str, output_dir: Path) -> Path:
        frames_dir = output_dir / "frames" / safe_scene_name(scene_name)
        frame_paths = extract_frames(video_path, frames_dir, max_frames=self.args.max_frames)
        images = self.load_and_preprocess_images(
            [str(path) for path in frame_paths],
            mode=self.args.preprocess_mode,
            image_resolution=self.args.image_resolution,
        ).to(self.device)
        with self.torch.inference_mode():
            predictions = self.model(images)
            extrinsics, intrinsics = self.encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )
        return write_pose_outputs(
            extrinsics,
            output_dir=output_dir,
            expected_frames=len(frame_paths),
            intrinsics=intrinsics,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _eligible_ids(audit_jsonl: Path | None) -> set[str] | None:
    if audit_jsonl is None:
        return None
    rows = _read_jsonl(audit_jsonl)
    return {str(row.get("video_id")) for row in rows if row.get("status") == "ok" and row.get("video_id")}


def select_rows(
    rows: list[dict[str, Any]],
    *,
    eligible_ids: set[str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    eligible_idx = 0
    for row in rows:
        video_id = str(row.get("video_id") or "")
        if eligible_ids is not None and video_id not in eligible_ids:
            continue
        if eligible_idx % num_shards != shard_index:
            eligible_idx += 1
            continue
        selected.append(row)
        eligible_idx += 1
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _pose_ok(path: Path, *, expected_frames: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return False
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4) or len(arr) == 0:
        return False
    if expected_frames is not None and len(arr) != int(expected_frames):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=arr.dtype)
    if not np.allclose(arr[:, 3, :], expected_bottom, atol=1e-5):
        return False
    det = np.linalg.det(arr[:, :3, :3])
    return bool(np.all(np.isfinite(det)) and np.all(np.abs(det) > 1e-8))


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _read_jsonl(args.input_jsonl)
    rows = select_rows(
        candidates,
        eligible_ids=_eligible_ids(args.audit_jsonl),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    results_path = args.results_jsonl
    if results_path and results_path.exists() and not args.append_results:
        results_path.unlink()

    script = Path(__file__).resolve().with_name("run_vggt_omega_pose.py")
    cache_root = args.cache_root or args.output_root / "cache"
    log_root = args.output_root / "logs" / f"vggt_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    log_root.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    row_reports: list[dict[str, Any]] = []
    runner = None
    if args.execution_mode == "inprocess" and not args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        runner = InprocessVGGTOmegaRunner(args)
    for order, row in enumerate(rows):
        video_id = str(row.get("video_id") or "")
        input_video_path = row.get("path") or row.get("video_path")
        eval_video_path, eval_metadata = resolve_eval_video_for_row(
            row,
            clip_root=cache_root / "eval_clips" / "hydra_generated_only",
            materialize=not args.dry_run,
        )
        video_path = str(eval_video_path) if str(eval_video_path) else input_video_path
        pose_expected_frames = eval_metadata.get("generated_frame_count")
        try:
            pose_expected_frames = int(pose_expected_frames) if pose_expected_frames is not None else None
        except (TypeError, ValueError):
            pose_expected_frames = None
        safe = safe_video_id(video_id)
        output_dir = cache_root / "pose" / safe
        pose_path = output_dir / args.poses_file
        report = {
            "video_id": video_id,
            "model": row.get("model"),
            "camera_type": row.get("camera_type") or row.get("camera"),
            "input_video_path": input_video_path,
            "path": video_path,
            "pose_path": str(pose_path),
            "pose_expected_frames": pose_expected_frames,
            "status": None,
            "pose_backend": "vggt_omega",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "order": order,
        }
        report.update(eval_metadata)
        if not video_path:
            report["status"] = "missing_video_path"
        elif report.get("hydra_segment_status") == "unresolved":
            report["status"] = "hydra_concat_unresolved"
        elif _pose_ok(pose_path, expected_frames=pose_expected_frames) and not args.force:
            report["status"] = "skipped_existing_pose"
        elif args.dry_run:
            report["status"] = "dry_run"
        elif args.execution_mode == "inprocess":
            assert runner is not None
            log_path = log_root / f"{safe}.log"
            report["log_path"] = str(log_path)
            try:
                runner.run_one(video_path=Path(str(video_path)), scene_name=safe, output_dir=output_dir)
                report["returncode"] = 0
                report["status"] = "ok" if _pose_ok(pose_path, expected_frames=pose_expected_frames) else "error"
            except Exception as exc:
                log_path.write_text(str(exc) + "\n", encoding="utf-8")
                report["returncode"] = 1
                report["status"] = "error"
            if report["status"] == "error" and args.stop_on_error:
                status_counts[report["status"]] += 1
                row_reports.append(report)
                if results_path:
                    _write_jsonl(results_path, [report], append=True)
                break
        else:
            command = [
                args.python,
                str(script),
                "--video_path",
                str(video_path),
                "--scene_name",
                safe,
                "--output_dir",
                str(output_dir),
                "--vggt_repo",
                str(args.vggt_repo),
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                "cuda",
                "--image-resolution",
                str(args.image_resolution),
                "--preprocess-mode",
                str(args.preprocess_mode),
            ]
            if args.max_frames:
                command.extend(["--max_frames", str(args.max_frames)])
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            log_path = log_root / f"{safe}.log"
            report["log_path"] = str(log_path)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=args.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["returncode"] = proc.returncode
            report["status"] = "ok" if proc.returncode == 0 and _pose_ok(pose_path, expected_frames=pose_expected_frames) else "error"
            if report["status"] == "error" and args.stop_on_error:
                status_counts[report["status"]] += 1
                row_reports.append(report)
                if results_path:
                    _write_jsonl(results_path, [report], append=True)
                break
        status_counts[str(report["status"])] += 1
        row_reports.append(report)
        if results_path:
            _write_jsonl(results_path, [report], append=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_rows": len(candidates),
        "selected_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "output_root": str(args.output_root),
        "cache_root": str(cache_root),
        "pose_backend": "vggt_omega",
        "vggt_repo": str(args.vggt_repo),
        "checkpoint": str(args.checkpoint),
        "image_resolution": int(args.image_resolution),
        "preprocess_mode": str(args.preprocess_mode),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "gpu_id": str(args.gpu_id),
    }
    return {"summary": summary, "rows": row_reports}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, help="Only run rows with audit status=ok")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True, help="Pose cache root.")
    parser.add_argument("--vggt-repo", "--vggt_repo", dest="vggt_repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames", "--max_frames", dest="max_frames", type=int)
    parser.add_argument("--image-resolution", "--image_resolution", dest="image_resolution", type=int, required=True)
    parser.add_argument("--preprocess-mode", "--preprocess_mode", dest="preprocess_mode", choices=("balanced", "max_size"), required=True)
    parser.add_argument("--execution-mode", choices=("subprocess", "inprocess"), required=True)
    parser.add_argument("--poses-file", "--poses_file", dest="poses_file", required=True)
    parser.add_argument("--results-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--append-results", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num-shards)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_batch(args)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 1 if report["summary"]["status_counts"].get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
