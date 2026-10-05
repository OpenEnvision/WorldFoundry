"""Shared helpers for video benchmark runner manifest discovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from worldfoundry.evaluation.tasks.execution.framework.runner_common import VIDEO_EXTENSIONS
MINIMAL_GIF_BYTES = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff"
    b"!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01"
    b"\x00\x00\x02\x02D\x01\x00;"
)


def _stem(path: Path) -> str:
    return path.stem


def discover_metadata_records(dataset_root: Path | None) -> list[dict[str, Any]]:
    if dataset_root is None or not dataset_root.exists():
        return []
    for suffix in (".jsonl", ".json", ".csv"):
        matches = sorted(dataset_root.rglob(f"*{suffix}"))
        for path in matches:
            if path.name.lower() in {"readme.json", "package.json"}:
                continue
            try:
                if suffix == ".csv":
                    with path.open(newline="", encoding="utf-8-sig") as handle:
                        return [dict(row) for row in csv.DictReader(handle)]
                if suffix == ".jsonl":
                    rows = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
                    return rows
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [row for row in payload if isinstance(row, dict)]
                if isinstance(payload, dict):
                    for key in ("data", "records", "samples", "items"):
                        nested = payload.get(key)
                        if isinstance(nested, list):
                            return [row for row in nested if isinstance(row, dict)]
            except (OSError, json.JSONDecodeError, csv.Error):
                continue
    return []


def expected_stems_from_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> set[str]:
    stems: set[str] = set()
    for row in records:
        for key in keys:
            value = row.get(key)
            if value in (None, ""):
                continue
            text = str(value).strip()
            if not text:
                continue
            normalized = text.split("?", 1)[0].split("#", 1)[0]
            path_like = (
                "://" in normalized
                or "/" in normalized
                or "\\" in normalized
                or Path(normalized).suffix.lower() in VIDEO_EXTENSIONS
            )
            stems.add(Path(normalized).stem if path_like else normalized)
    return stems


def build_generated_video_manifest(
    video_root: Path | None,
    *,
    expected_count: int | None = None,
    expected_stems: set[str] | None = None,
    category_from_path: Callable[[Path], str | None] | None = None,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if video_root is not None and video_root.exists():
        for path in sorted(video_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            entry: dict[str, Any] = {
                "path": str(path.resolve()),
                "stem": _stem(path),
                "size_bytes": path.stat().st_size,
            }
            if category_from_path is not None:
                category = category_from_path(path)
                if category:
                    entry["category"] = category
            files.append(entry)
    matched_stems = {item["stem"] for item in files}
    missing_stems = sorted(expected_stems - matched_stems) if expected_stems else []
    coverage_complete = False
    if expected_stems:
        coverage_complete = not missing_stems
    elif expected_count not in (None, 0):
        coverage_complete = len(files) >= expected_count
    by_category: dict[str, int] = {}
    for item in files:
        category = item.get("category")
        if isinstance(category, str) and category:
            by_category[category] = by_category.get(category, 0) + 1
    return {
        "video_root": None if video_root is None else str(video_root.resolve()),
        "file_count": len(files),
        "video_file_count": len(files),
        "expected_count": expected_count,
        "expected_file_count": expected_count,
        "expected_stems": sorted(expected_stems) if expected_stems else [],
        "missing_stems": missing_stems,
        "by_category": by_category,
        "coverage_complete": coverage_complete,
        "coverage_ratio": None if expected_count in (None, 0) else len(files) / expected_count,
        "files": files,
    }


def build_local_dataset_manifest(
    dataset_root: Path | None,
    *,
    dataset_id: str | None = None,
    config: str | None = None,
    split: str | None = None,
    expected_rows: int | None = None,
    media_extensions: tuple[str, ...] = (".mp4", ".webm", ".mov", ".mkv", ".avi"),
) -> dict[str, Any]:
    media_files: list[str] = []
    data_files: list[str] = []
    exists = bool(dataset_root is not None and dataset_root.exists())
    if dataset_root is not None and dataset_root.exists():
        data_files = [str(path.resolve()) for path in sorted(dataset_root.rglob("*")) if path.is_file()]
        extensions = {item.lower() for item in media_extensions}
        media_files = [
            str(path.resolve())
            for path in sorted(dataset_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in extensions
        ]
    records = discover_metadata_records(dataset_root)
    return {
        "dataset_root": None if dataset_root is None else str(dataset_root.resolve()),
        "exists": exists,
        "file_count": len(data_files),
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "record_count": len(records),
        "expected_rows": expected_rows,
        "media_file_count": len(media_files),
        "media_files": media_files[:100],
    }


def _tiny_rgb_frames(*, frames: int = 8, size: int = 64) -> list[Any]:
    from PIL import Image, ImageDraw

    rendered = []
    for index in range(frames):
        image = Image.new("RGB", (size, size), ((index * 31) % 255, 48, 128))
        draw = ImageDraw.Draw(image)
        offset = (index * 5) % max(1, size - 20)
        draw.rectangle((offset, offset, offset + 18, offset + 18), fill=(245, 245, 245))
        draw.line((0, size - 1 - offset, size - 1, offset), fill=(24, 196, 180), width=2)
        rendered.append(image)
    return rendered


def _write_tiny_video(path: Path, *, frames: int = 32, size: int = 224, fps: int = 8) -> str:
    try:
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (size, size),
        )
        if writer.isOpened():
            for frame in _tiny_rgb_frames(frames=frames, size=size):
                writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            writer.release()
            return "opencv_mp4v"
        writer.release()
    except Exception:
        pass

    try:
        import numpy as np
        import imageio.v2 as imageio

        arrays = [np.asarray(frame) for frame in _tiny_rgb_frames(frames=frames, size=size)]
        imageio.mimsave(path, arrays, fps=fps)
        return "imageio"
    except Exception:
        pass

    try:
        pil_frames = _tiny_rgb_frames(frames=frames, size=size)
        duration_ms = max(1, int(1000 / max(1, fps)))
        pil_frames[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return "pillow_gif"
    except Exception:
        path.write_bytes(MINIMAL_GIF_BYTES)
        return "minimal_gif_bytes"


def write_placeholder_video(path: Path, *, frames: int = 32, size: int = 224, fps: int = 8) -> Path:
    """Write a tiny generated-video artifact for contract/sample runs."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_tiny_video(path, frames=frames, size=size, fps=fps)
    return path


def materialize_sample_generated_videos(output_dir: Path, *, count: int = 1, frames: int = 32, size: int = 224, fps: int = 8) -> Path:
    generated_dir = output_dir / "sample_generated_videos"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        write_placeholder_video(generated_dir / f"sample-{index}.mp4", frames=frames, size=size, fps=fps)
    return generated_dir


def main(argv: list[str] | None = None) -> int:
    """CLI compatibility for legacy ``create_tiny_video`` tests."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Write a placeholder video artifact.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        payload = {"ok": True, "status": "skipped", "output": str(args.output)}
    else:
        path = write_placeholder_video(args.output, frames=args.frames, size=args.size, fps=args.fps)
        payload = {
            "ok": True,
            "status": "placeholder",
            "output": str(path),
            "frames": args.frames,
            "size": args.size,
            "fps": args.fps,
            "bytes": path.stat().st_size,
        }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
