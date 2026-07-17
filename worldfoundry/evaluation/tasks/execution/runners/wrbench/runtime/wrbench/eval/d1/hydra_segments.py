"""Hydra generated-only video segment helpers.

HyDRA writes benchmark outputs as condition/source frames followed by generated
frames. Metric runners should score only the generated continuation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from wrbench.eval.d1.geometry import safe_video_id


def is_hydra_row(row: dict[str, Any]) -> bool:
    return str(row.get("model") or "").lower() == "hydra"


def load_camera_sidecar(video_path: Path) -> dict[str, Any] | None:
    sidecar_path = Path(str(video_path) + ".camera.json")
    if not sidecar_path.is_file():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _positive_int(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def sidecar_condition_frames(sidecar: dict[str, Any] | None) -> int | None:
    if not isinstance(sidecar, dict):
        return None
    for key in ("condition_frame_count", "source_frame_count", "model_frame_count", "num_frames"):
        count = _positive_int(sidecar.get(key))
        if count is not None:
            return count
    return None


def _sidecar_generated_frames(sidecar: dict[str, Any] | None) -> int | None:
    if not isinstance(sidecar, dict):
        return None
    return _positive_int(sidecar.get("generated_frame_count"))


def video_frame_count(video_path: Path) -> int:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count > 0:
        cap.release()
        return count
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    return decoded


def _safe_stem(video_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem).strip("._")
    return stem or "video"


def materialize_generated_only_clip(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    frame_count: int,
) -> Path:
    import cv2  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 8.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"could not decode first frame: {video_path}")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    written = 0
    try:
        while written < int(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written != int(frame_count):
        raise ValueError(
            f"expected to write {frame_count} generated frames from {video_path}, wrote {written}"
        )
    return out_path


def _existing_eval_path(row: dict[str, Any]) -> Path | None:
    value = row.get("eval_video_path")
    if value in (None, ""):
        return None
    return Path(str(value))


def _metadata(
    *,
    eval_video_path: Path,
    original_concat_path: Path,
    status: str,
    condition_frames: int | None = None,
    generated_frames: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    source_video_path: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eval_video_path": str(eval_video_path),
        "original_concat_path": str(original_concat_path),
        "evaluated_segment": "generated_only" if status != "unresolved" else "unresolved",
        "hydra_segment_status": status,
    }
    if condition_frames is not None:
        out["condition_frame_count"] = int(condition_frames)
    if generated_frames is not None:
        out["generated_frame_count"] = int(generated_frames)
    if start_frame is not None:
        out["eval_video_frame_start"] = int(start_frame)
    if end_frame is not None:
        out["eval_video_frame_end_exclusive"] = int(end_frame)
    if source_video_path not in (None, ""):
        out["source_video_path"] = str(source_video_path)
    return out


def resolve_eval_video_for_row(
    row: dict[str, Any],
    clip_root: Path,
    materialize: bool = True,
) -> tuple[Path, dict[str, Any]]:
    input_value = row.get("path") or row.get("video_path")
    if not input_value:
        return Path(""), {"hydra_segment_status": "missing_video_path"}
    original_path = Path(str(input_value))
    if not is_hydra_row(row):
        return original_path, {}

    sidecar = load_camera_sidecar(original_path)
    condition_frames = sidecar_condition_frames(sidecar)
    source_video_path = row.get("source_video_path")
    if isinstance(sidecar, dict) and sidecar.get("source_video_path") not in (None, ""):
        source_video_path = sidecar.get("source_video_path")
    generated_frames = _positive_int(row.get("generated_frame_count")) or _sidecar_generated_frames(sidecar)

    existing = _existing_eval_path(row)
    if existing is not None:
        expected = generated_frames or condition_frames
        if existing.exists() and (expected is None or video_frame_count(existing) == expected):
            return existing, _metadata(
                eval_video_path=existing,
                original_concat_path=Path(str(row.get("original_concat_path") or original_path)),
                status="generated_only_existing",
                condition_frames=condition_frames,
                generated_frames=expected,
                start_frame=0,
                end_frame=expected,
                source_video_path=source_video_path,
            )

    if sidecar is None or condition_frames is None or not original_path.exists():
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="unresolved",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            source_video_path=source_video_path,
        )

    total_frames = video_frame_count(original_path)
    if total_frames == condition_frames and (
        row.get("evaluated_segment") == "generated_only"
        or sidecar.get("evaluated_segment") == "generated_only"
        or sidecar.get("hydra_segment_status") in {"already_generated_only", "generated_only_materialized"}
    ):
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=Path(str(row.get("original_concat_path") or original_path)),
            status="already_generated_only",
            condition_frames=condition_frames,
            generated_frames=condition_frames,
            start_frame=0,
            end_frame=condition_frames,
            source_video_path=source_video_path,
        )

    if generated_frames is None:
        generated_frames = condition_frames

    if generated_frames is None or total_frames != condition_frames + generated_frames:
        return original_path, _metadata(
            eval_video_path=original_path,
            original_concat_path=original_path,
            status="unresolved",
            condition_frames=condition_frames,
            generated_frames=generated_frames,
            source_video_path=source_video_path,
        )

    video_id = safe_video_id(row.get("video_id") or _safe_stem(original_path))
    out_path = clip_root / f"{video_id}.generated_only.mp4"
    if materialize and not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        materialize_generated_only_clip(
            original_path,
            out_path,
            start_frame=condition_frames,
            frame_count=generated_frames,
        )
        generated_sidecar = dict(sidecar)
        generated_sidecar.update(
            {
                "evaluated_segment": "generated_only",
                "hydra_segment_status": "generated_only_materialized",
                "original_concat_path": str(original_path),
                "eval_video_path": str(out_path),
                "condition_frame_count": int(condition_frames),
                "generated_frame_count": int(generated_frames),
                "eval_video_frame_start": int(condition_frames),
                "eval_video_frame_end_exclusive": int(condition_frames + generated_frames),
            }
        )
        out_path.with_suffix(out_path.suffix + ".camera.json").write_text(
            json.dumps(generated_sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return out_path, _metadata(
        eval_video_path=out_path,
        original_concat_path=original_path,
        status="generated_only_materialized",
        condition_frames=condition_frames,
        generated_frames=generated_frames,
        start_frame=condition_frames,
        end_frame=condition_frames + generated_frames,
        source_video_path=source_video_path,
    )
