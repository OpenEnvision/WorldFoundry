#!/usr/bin/env python3
"""Score D1 prompt-camera alignment (CamAlign) from cached pose stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wrbench.eval.d1.camera_intent import score_camera_intent_row
from wrbench.eval.d1.geometry import safe_video_id


def _camera_type(row: dict[str, Any]) -> str:
    return str(row.get("camera_type") or row.get("camera") or "").strip()


def _pose_file(row: dict[str, Any], cache_root: Path, poses_file: str) -> Path:
    return cache_root / "pose" / safe_video_id(row.get("video_id")) / poses_file


def score_camalign_row(row: dict[str, Any], *, cache_root: Path, poses_file: str) -> dict[str, Any]:
    camera = _camera_type(row)
    if camera == "uncontrolled":
        out = dict(row)
        out.update(
            {
                "d1_camalign_score": None,
                "d1_camalign_status": "excluded_uncontrolled",
                "d1_camalign_metric_scope": "excluded",
            }
        )
        return out

    pose_file = _pose_file(row, cache_root, poses_file)
    if not pose_file.exists():
        out = dict(row)
        out.update(
            {
                "d1_camalign_score": None,
                "d1_camalign_status": "missing_output",
                "d1_camalign_metric_scope": "missing_pose",
            }
        )
        return out

    try:
        predicted = np.load(pose_file, allow_pickle=False)
        return score_camera_intent_row(row, predicted)
    except Exception:
        out = dict(row)
        out.update(
            {
                "d1_camalign_score": None,
                "d1_camalign_status": "error",
                "d1_camalign_metric_scope": "error",
            }
        )
        return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score D1 prompt-camera alignment (CamAlign).")
    parser.add_argument("--input-jsonl", "--input_jsonl", dest="input_jsonl", required=True)
    parser.add_argument("--output-jsonl", "--output_jsonl", dest="output_jsonl", required=True)
    parser.add_argument(
        "--pose-cache-root",
        "--pose_cache_root",
        dest="pose_cache_root",
        required=True,
    )
    parser.add_argument("--poses-file", "--poses_file", dest="poses_file", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = Path(args.pose_cache_root)
    rows = [
        score_camalign_row(row, cache_root=cache_root, poses_file=args.poses_file)
        for row in read_jsonl(Path(args.input_jsonl))
    ]
    write_jsonl(rows, Path(args.output_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
