#!/usr/bin/env python3
"""Extract DINOv2 temporal consistency features for D2 subject/render collapse.

This is a lightweight feature probe: it compares global-frame and center-crop
DINO embeddings across sampled frames. Center crop is only a proxy for the
prompt-critical subject, but it is closer to subject integrity than generic
quality or VideoAlign-MQ.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    return [payload] if isinstance(payload, dict) else []


def _video_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    for row in _read_records(path):
        vid = row.get("video_id")
        video_path = row.get("video_path") or row.get("path")
        if isinstance(vid, str) and vid and isinstance(video_path, str) and video_path:
            copied = dict(row)
            copied["video_path"] = video_path
            out.append(copied)
    return out


def _unique_sorted(indices: Iterable[int], source_frame_count: int) -> list[int]:
    return sorted({max(0, min(source_frame_count - 1, int(idx))) for idx in indices})


def _fixed_count_indices(source_frame_count: int, max_frames: int) -> list[int]:
    if source_frame_count <= 0 or max_frames <= 0:
        return []
    count = min(int(max_frames), int(source_frame_count))
    return [int(i) for i in np.linspace(0, source_frame_count - 1, count, dtype=np.int64)]


def _frame_indices_for_sampling(
    *,
    source_frame_count: int,
    source_fps: float | None,
    sample_policy: str,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> list[int]:
    if source_frame_count <= 0:
        return []
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    min_frames = max(1, min(int(min_frames), int(max_frames), int(source_frame_count)))
    max_frames = min(int(max_frames), int(source_frame_count))
    if sample_policy == "fixed_count":
        return _fixed_count_indices(source_frame_count, max_frames)
    if sample_policy != "time_fps":
        raise ValueError(f"unsupported sample_policy: {sample_policy}")
    if source_fps is None or source_fps <= 0 or sample_fps <= 0:
        return _fixed_count_indices(source_frame_count, max_frames)

    step = max(float(source_fps) / float(sample_fps), 1.0)
    indices = _unique_sorted(
        [0, source_frame_count - 1, *np.arange(0, source_frame_count, step).round().astype(int).tolist()],
        source_frame_count,
    )
    if len(indices) < min_frames:
        indices = _unique_sorted(
            [*indices, *_fixed_count_indices(source_frame_count, min_frames)],
            source_frame_count,
        )
    if len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames, dtype=np.int64)
        indices = _unique_sorted([indices[int(pos)] for pos in positions], source_frame_count)
        if indices[0] != 0:
            indices = _unique_sorted([0, *indices], source_frame_count)
        if indices[-1] != source_frame_count - 1:
            indices = _unique_sorted([*indices, source_frame_count - 1], source_frame_count)
        if len(indices) > max_frames:
            middle = indices[1:-1]
            keep_middle = max_frames - 2
            if keep_middle > 0:
                pos = np.linspace(0, len(middle) - 1, keep_middle, dtype=np.int64)
                indices = [0, *[middle[int(i)] for i in pos], source_frame_count - 1]
            else:
                indices = [0, source_frame_count - 1][:max_frames]
    return indices


def _sample_frames(
    video_path: str,
    *,
    sample_policy: str,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total <= 0:
            return None, {"decode_backend": "decord", "decode_error": "empty_video"}
        source_fps = float(vr.get_avg_fps() or 0.0)
        idx = _frame_indices_for_sampling(
            source_frame_count=total,
            source_fps=source_fps,
            sample_policy=sample_policy,
            sample_fps=sample_fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        frames = vr.get_batch(idx).asnumpy()
        return frames, {
            "decode_backend": "decord",
            "decode_error": "",
            "source_fps": source_fps if source_fps > 0 else None,
            "source_frame_count": int(total),
            "sampled_frame_indices": idx,
            "sampled_frame_count": int(len(idx)),
            "sample_policy": sample_policy,
        }
    except Exception as exc:
        import cv2

        cap = cv2.VideoCapture(video_path)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None, {"decode_backend": "opencv", "decode_error": f"empty_video_after_decord:{exc}"}
        idx = _frame_indices_for_sampling(
            source_frame_count=len(frames),
            source_fps=source_fps if source_fps > 0 else None,
            sample_policy=sample_policy,
            sample_fps=sample_fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        return np.stack([frames[int(i)] for i in idx], axis=0), {
            "decode_backend": "opencv",
            "decode_error": "",
            "source_fps": source_fps if source_fps > 0 else None,
            "source_frame_count": int(reported_total or len(frames)),
            "decoded_frame_count": int(len(frames)),
            "sampled_frame_indices": idx,
            "sampled_frame_count": int(len(idx)),
            "sample_policy": sample_policy,
        }


def _center_crop(frame: np.ndarray, crop_fraction: float) -> np.ndarray:
    h, w = frame.shape[:2]
    side_h = max(1, int(round(h * crop_fraction)))
    side_w = max(1, int(round(w * crop_fraction)))
    y0 = max(0, (h - side_h) // 2)
    x0 = max(0, (w - side_w) // 2)
    return frame[y0 : y0 + side_h, x0 : x0 + side_w]


def _motion_bbox(frames: np.ndarray, *, min_area_fraction: float = 0.04, pad_fraction: float = 0.18) -> tuple[int, int, int, int] | None:
    if len(frames) < 2:
        return None
    import cv2

    h, w = frames.shape[1:3]
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames], axis=0)
    diffs = np.abs(np.diff(gray.astype(np.float32), axis=0))
    motion = np.percentile(diffs, 85, axis=0)
    threshold = max(6.0, float(np.percentile(motion, 90)))
    mask = motion >= threshold
    ys, xs = np.where(mask)
    if len(xs) < max(16, int(h * w * min_area_fraction * 0.05)):
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if (x1 - x0 + 1) * (y1 - y0 + 1) < h * w * min_area_fraction:
        return None
    pad_x = int(round((x1 - x0 + 1) * pad_fraction))
    pad_y = int(round((y1 - y0 + 1) * pad_fraction))
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return x0, y0, x1 + 1, y1 + 1


def _crop_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return frame[y0:y1, x0:x1]


def _cosine_rows(features: np.ndarray) -> tuple[np.ndarray, float | None, float | None]:
    if len(features) < 2:
        return np.array([], dtype=np.float64), None, None
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm = np.where(norm <= 1e-12, 1.0, norm)
    unit = features / norm
    adjacent = np.sum(unit[1:] * unit[:-1], axis=1)
    first_last = float(np.sum(unit[0] * unit[-1]))
    first_mid = float(np.sum(unit[0] * unit[len(unit) // 2]))
    return adjacent.astype(np.float64), first_last, first_mid


def _stats(prefix: str, feats: np.ndarray) -> dict[str, float | None]:
    adjacent, first_last, first_mid = _cosine_rows(feats)
    if adjacent.size:
        return {
            f"{prefix}_adjacent_cos_mean": float(np.mean(adjacent)),
            f"{prefix}_adjacent_cos_min": float(np.min(adjacent)),
            f"{prefix}_adjacent_cos_std": float(np.std(adjacent)),
            f"{prefix}_first_last_cos": first_last,
            f"{prefix}_first_mid_cos": first_mid,
        }
    return {
        f"{prefix}_adjacent_cos_mean": None,
        f"{prefix}_adjacent_cos_min": None,
        f"{prefix}_adjacent_cos_std": None,
        f"{prefix}_first_last_cos": first_last,
        f"{prefix}_first_mid_cos": first_mid,
    }


def _score_from_cosines(values: Iterable[Any]) -> float | None:
    cleaned = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            cleaned.append(min(1.0, max(1e-6, (f + 1.0) / 2.0)))
    if not cleaned:
        return None
    return float(math.exp(sum(math.log(value) for value in cleaned) / len(cleaned)))


def _feature_tensor(frames: list[np.ndarray], processor, model, device: str, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(frames), batch_size):
        batch = [Image.fromarray(frame) for frame in frames[start : start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            else:
                feat = out.last_hidden_state[:, 0]
        outputs.append(feat.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _extract_one(
    row: Mapping[str, Any],
    processor,
    model,
    device: str,
    sample_policy: str,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
    batch_size: int,
    crop_fraction: float,
    crop_modes: set[str],
) -> dict[str, Any]:
    vid = str(row["video_id"])
    video_path = str(row["video_path"])
    frames, sample_meta = _sample_frames(
        video_path,
        sample_policy=sample_policy,
        sample_fps=sample_fps,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    out: dict[str, Any] = {
        "video_id": vid,
        "video_path": video_path,
        "path": video_path,
        "feature_status": "ok" if frames is not None else "decode_failed",
        "model": row.get("model", ""),
        "family_id": row.get("family_id", ""),
        "variant_id": row.get("variant_id", ""),
        "camera_type": row.get("camera_type", ""),
        "event_tier": row.get("event_tier", ""),
        **sample_meta,
    }
    if frames is None:
        return out
    out["dinov2_frame_count"] = int(len(frames))
    if "global" in crop_modes:
        global_feats = _feature_tensor([frame for frame in frames], processor, model, device, batch_size)
        out.update(_stats("dinov2_global", global_feats))
    if "center" in crop_modes:
        crop_feats = _feature_tensor([_center_crop(frame, crop_fraction) for frame in frames], processor, model, device, batch_size)
        out.update(_stats("dinov2_center", crop_feats))
    if "motion" in crop_modes:
        bbox = _motion_bbox(frames)
        out["dinov2_motion_bbox"] = list(bbox) if bbox is not None else None
        out["dinov2_motion_bbox_area_fraction"] = (
            ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(frames.shape[1] * frames.shape[2])
            if bbox is not None
            else None
        )
        if bbox is not None:
            motion_feats = _feature_tensor([_crop_bbox(frame, bbox) for frame in frames], processor, model, device, batch_size)
            out.update(_stats("dinov2_motion", motion_feats))
    # Collapse proxy: low center consistency is treated as riskier than low
    # global consistency because the subject often lives near the action center.
    center_min = out.get("dinov2_center_adjacent_cos_min")
    global_min = out.get("dinov2_global_adjacent_cos_min")
    out["dinov2_center_minus_global_min_cos"] = (
        float(center_min) - float(global_min)
        if isinstance(center_min, (int, float)) and isinstance(global_min, (int, float))
        else None
    )
    out["d2_dino_score"] = _score_from_cosines(
        [
            out.get("dinov2_center_adjacent_cos_min"),
            out.get("dinov2_center_first_last_cos"),
            out.get("dinov2_global_adjacent_cos_min"),
            out.get("dinov2_global_first_last_cos"),
        ]
    )
    return out


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = row.get("video_id") if isinstance(row, dict) else None
            if isinstance(vid, str) and vid:
                done.add(vid)
    return done


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Video manifest JSON/JSONL.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local DINOv2 model directory.")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-policy", choices=("fixed_count", "time_fps"), required=True)
    parser.add_argument("--sample-fps", type=float, required=True)
    parser.add_argument("--min-frames", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--center-crop-fraction", type=float, required=True)
    parser.add_argument("--crop-mode", action="append", choices=("global", "center", "motion"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be > 0")
    if args.min_frames <= 0:
        parser.error("--min-frames must be > 0")
    if args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    if args.min_frames > args.max_frames:
        parser.error("--min-frames must be <= --max-frames")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
    model.eval()
    crop_modes = set(args.crop_mode)
    rows = _video_rows(args.videos)
    if args.limit:
        rows = rows[: args.limit]
    done = _done_ids(args.out_jsonl) if args.resume else set()
    rows_to_run = [row for row in rows if row.get("video_id") not in done]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    scored = len(done)
    errored = 0
    with args.out_jsonl.open(mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows_to_run, start=1):
            out = _extract_one(
                    row,
                    processor,
                    model,
                    args.device,
                    args.sample_policy,
                    args.sample_fps,
                    args.min_frames,
                    args.max_frames,
                    args.batch_size,
                    args.center_crop_fraction,
                    crop_modes,
                )
            if out.get("feature_status") == "ok":
                scored += 1
            else:
                errored += 1
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            if idx % 25 == 0:
                print(f"processed {idx}/{len(rows_to_run)} this_run, total_done={scored + errored}/{len(rows)}", flush=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos": str(args.videos),
        "model_dir": str(args.model_dir),
        "out_jsonl": str(args.out_jsonl),
        "requested": len(rows),
        "already_done_before_run": len(done),
        "ran_this_time": len(rows_to_run),
        "scored": scored,
        "errored": errored,
        "device": args.device,
        "sample_policy": args.sample_policy,
        "sample_fps": args.sample_fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "center_crop_fraction": args.center_crop_fraction,
        "crop_modes": sorted(crop_modes),
    }
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["errored"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
