"""Small, dependency-light I/O helpers shared by the PAI-Bench tracks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def read_video(path: Path, *, max_frames: int | None = None, rgb: bool = True) -> np.ndarray:
    """Decode a video into ``[T,H,W,C]`` uint8 frames using OpenCV."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not decode video: {path}")
    frames: list[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, CSV, or TSV records without benchmark-specific assumptions."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rows", "samples", "results", "predictions", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
        return [dict(payload)]
    raise ValueError(f"expected a record collection in {path}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def resolve_dataset_path(root: Path, value: Any, *fallback_parts: str) -> Path:
    """Resolve an HF metadata path and tolerate the release's directory aliases."""

    if value not in (None, "", "nan"):
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        rooted = root / candidate
        if rooted.exists():
            return rooted
    fallback = root.joinpath(*fallback_parts)
    return fallback


def find_video(root: Path, name: str) -> Path:
    candidate = root / name
    if candidate.is_file():
        return candidate
    candidate = root / "videos" / name
    if candidate.is_file():
        return candidate
    stem = Path(name).stem
    matches = sorted(path for path in root.rglob(f"{stem}.*") if path.suffix.lower() in VIDEO_SUFFIXES)
    if not matches:
        raise FileNotFoundError(f"generated video not found for {name!r} below {root}")
    return matches[0]


def find_sidecar(root: Path | None, stem: str, suffixes: Iterable[str]) -> Path | None:
    if root is None:
        return None
    for suffix in suffixes:
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in root.rglob(f"{stem}.*") if path.suffix.lower() in set(suffixes))
    return matches[0] if matches else None


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if not payload.files:
                raise ValueError(f"empty npz: {path}")
            for key in ("depth", "depths", "arr_0"):
                if key in payload:
                    return np.asarray(payload[key])
            return np.asarray(payload[payload.files[0]])
    raise ValueError(f"unsupported array file: {path}")


def prediction_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in records:
        keys = (
            row.get("uid"),
            row.get("id"),
            row.get("doc_id"),
            row.get("sample_id"),
            row.get("question_id"),
            row.get("video_id"),
            row.get("video_path"),
        )
        for value in keys:
            if value not in (None, ""):
                text = str(value)
                index[text] = row
                index[Path(text).stem] = row
    return index
