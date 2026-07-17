#!/usr/bin/env python3
"""Extract a non-learned DINOv2 local-global D2 candidate score.

The candidate uses a pretrained DINOv2 backbone but no WRBench/D2-trained head,
no fitted feature weights, and no normalization from the evaluated benchmark
pool. It is an ablation/candidate field and must not replace the reviewed D2
mainline without separate validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]

from wrbench.eval.d2 import extract_d2_dinov2_consistency_features as d2_base

METHOD_NAME = "dinov2_local_global_patch_bestmatch_v1"
FORMULA_FAMILY = "fixed_nonlearned_scalar_variants_v1"
LEARNING_STATUS = (
    "The candidate uses a pretrained DINOv2 backbone but no WRBench/D2-trained head, "
    "no fitted feature weights, and no normalization from the evaluated benchmark pool."
)

HARD_CUT_THRESHOLD = 0.25
LOCAL_LOW_PERCENTILE = 20.0
VIDEO_LOW_PERCENTILE = 20.0
PAIR_MEDIAN_WEIGHT = 0.60
PAIR_LOW_WEIGHT = 0.40
CANDIDATE_FIELD = "d2_dinov2_local_global_candidate"
SELECTED_VISUAL_INTEGRITY_FIELD = "d2_selected_visual_integrity_score"
PROTECTED_MAINLINE_FIELD = "d2_dinov2_temporal_consistency"


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def _l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.where(norm <= 1e-12, 1.0, norm)
    return arr / norm


def _positive_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return _clip01(float(np.sum(a * b)))


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return _clip01(float(np.percentile(values.astype(np.float64), q)))


def split_dinov2_tokens(last_hidden_state: np.ndarray, num_register_tokens: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split HF DINOv2 hidden states into CLS/global and patch-token arrays."""

    hidden = np.asarray(last_hidden_state, dtype=np.float32)
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden state shape [batch, tokens, dim], got {hidden.shape}")
    patch_start = 1 + max(0, int(num_register_tokens))
    if hidden.shape[1] <= patch_start:
        raise ValueError(f"DINOv2 output does not contain patch tokens: shape={hidden.shape}")
    global_tokens = hidden[:, 0, :]
    patch_tokens = hidden[:, patch_start:, :]
    if patch_tokens.shape[1] <= 0:
        raise ValueError(f"DINOv2 output does not contain patch tokens: shape={hidden.shape}")
    return _l2_normalize(global_tokens), _l2_normalize(patch_tokens)


def _local_bestmatch_stats(patches_a: np.ndarray, patches_b: np.ndarray) -> tuple[float, float]:
    if patches_a.size == 0 or patches_b.size == 0:
        return 0.0, 0.0
    sims = np.matmul(patches_a.astype(np.float32), patches_b.astype(np.float32).T)
    sims = np.clip(sims, 0.0, 1.0)
    matches = np.concatenate([np.max(sims, axis=1), np.max(sims, axis=0)], axis=0)
    return _clip01(float(np.mean(matches))), _percentile(matches, LOCAL_LOW_PERCENTILE)


def compute_local_global_candidate(global_features: np.ndarray, patch_features: np.ndarray) -> dict[str, Any]:
    """Compute fixed-formula local-global D2 candidate and audit fields."""

    globals_ = _l2_normalize(global_features)
    patches = _l2_normalize(patch_features)
    if globals_.ndim != 2:
        raise ValueError(f"expected global feature shape [frames, dim], got {globals_.shape}")
    if patches.ndim != 3:
        raise ValueError(f"expected patch feature shape [frames, patches, dim], got {patches.shape}")
    if len(globals_) != len(patches):
        raise ValueError("global and patch feature frame counts differ")
    frame_count = int(len(globals_))
    if frame_count < 2:
        empty = {
            CANDIDATE_FIELD: None,
            "d2_lg_pair_median_candidate": None,
            "d2_lg_pair_low_candidate": None,
            "d2_lg_local_mean_candidate": None,
            "d2_lg_local_low_candidate": None,
            "d2_lg_global_adjacent_candidate": None,
            "d2_lg_global_first_last_candidate": None,
            "d2_lg_pair_median": None,
            "d2_lg_pair_p20": None,
            "d2_lg_long_score": None,
            "d2_lg_local_mean_median": None,
            "d2_lg_local_low_p20": None,
            "d2_lg_hard_cut_min": None,
            "d2_lg_sampled_frame_count": frame_count,
            "d2_lg_lowest_pair_indices": [],
            "d2_lg_method": METHOD_NAME,
            "d2_lg_formula_family": FORMULA_FAMILY,
            "d2_lg_learned_head": False,
            "d2_vbench_like_global_adjacent_mean": None,
            "d2_vbench_like_global_first_last": None,
            "d2_lg_min_global_first_local_low_relief05_candidate": None,
        }
        return empty

    pair_scores: list[float] = []
    hard_cut_scores: list[float] = []
    local_means: list[float] = []
    local_lows: list[float] = []
    global_adjacent: list[float] = []
    for idx in range(frame_count - 1):
        global_score = _positive_cosine(globals_[idx], globals_[idx + 1])
        local_mean, local_low = _local_bestmatch_stats(patches[idx], patches[idx + 1])
        hard_cut = _clip01(global_score / HARD_CUT_THRESHOLD)
        pair_score = _clip01(hard_cut * math.sqrt(max(0.0, local_mean) * max(0.0, local_low)))
        global_adjacent.append(global_score)
        hard_cut_scores.append(hard_cut)
        local_means.append(local_mean)
        local_lows.append(local_low)
        pair_scores.append(pair_score)

    pair_arr = np.asarray(pair_scores, dtype=np.float64)
    low_pair = _percentile(pair_arr, VIDEO_LOW_PERCENTILE)
    pair_median = _clip01(float(np.median(pair_arr)))
    candidate = _clip01(PAIR_MEDIAN_WEIGHT * pair_median + PAIR_LOW_WEIGHT * low_pair)
    lowest_pair = int(np.argmin(pair_arr))
    first_last = _positive_cosine(globals_[0], globals_[-1])
    return {
        CANDIDATE_FIELD: candidate,
        "d2_lg_pair_median_candidate": pair_median,
        "d2_lg_pair_low_candidate": low_pair,
        "d2_lg_local_mean_candidate": _clip01(float(np.median(np.asarray(local_means, dtype=np.float64)))),
        "d2_lg_local_low_candidate": _percentile(np.asarray(local_lows, dtype=np.float64), VIDEO_LOW_PERCENTILE),
        "d2_lg_global_adjacent_candidate": _clip01(float(np.mean(np.asarray(global_adjacent, dtype=np.float64)))),
        "d2_lg_global_first_last_candidate": first_last,
        "d2_lg_pair_median": pair_median,
        "d2_lg_pair_p20": low_pair,
        "d2_lg_long_score": first_last,
        "d2_lg_local_mean_median": _clip01(float(np.median(np.asarray(local_means, dtype=np.float64)))),
        "d2_lg_local_low_p20": _percentile(np.asarray(local_lows, dtype=np.float64), VIDEO_LOW_PERCENTILE),
        "d2_lg_hard_cut_min": _clip01(float(np.min(np.asarray(hard_cut_scores, dtype=np.float64)))),
        "d2_lg_sampled_frame_count": frame_count,
        "d2_lg_lowest_pair_indices": [lowest_pair, lowest_pair + 1],
        "d2_lg_method": METHOD_NAME,
        "d2_lg_formula_family": FORMULA_FAMILY,
        "d2_lg_learned_head": False,
        "d2_vbench_like_global_adjacent_mean": _clip01(float(np.mean(np.asarray(global_adjacent, dtype=np.float64)))),
        "d2_vbench_like_global_first_last": first_last,
    }


def _current_mainline_score(row: Mapping[str, Any] | None) -> float | None:
    if not row:
        return None
    for key in ("d2_dino_score", PROTECTED_MAINLINE_FIELD, "d2_current_mainline_score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def merge_candidate_fields(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    current_mainline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    if PROTECTED_MAINLINE_FIELD in candidate:
        raise ValueError(f"candidate attempted to overwrite protected field: {PROTECTED_MAINLINE_FIELD}")
    out.update(candidate)
    out["d2_current_mainline_score"] = _current_mainline_score(current_mainline or row)
    out["d2_current_mainline_raw"] = (
        {
            key: current_mainline.get(key)
            for key in (
                "d2_dino_score",
                PROTECTED_MAINLINE_FIELD,
                "dinov2_center_adjacent_cos_min",
                "dinov2_center_first_last_cos",
                "dinov2_global_adjacent_cos_min",
                "dinov2_global_first_last_cos",
            )
            if key in current_mainline
        }
        if current_mainline
        else None
    )
    current = _finite_score(out.get("d2_current_mainline_score"))
    local_low = _finite_score(out.get("d2_lg_local_low_candidate"))
    pair_low = _finite_score(out.get("d2_lg_pair_low_candidate"))
    global_first_last = _finite_score(out.get("d2_lg_global_first_last_candidate"))
    if local_low is not None and global_first_last is not None:
        min_global_local_low = min(global_first_last, local_low)
        out["d2_lg_min_global_first_local_low_candidate"] = min_global_local_low
        out[SELECTED_VISUAL_INTEGRITY_FIELD] = min_global_local_low
        local_mean = _finite_score(out.get("d2_lg_local_mean_candidate"))
        if local_mean is not None:
            out["d2_lg_min_global_first_local_low_relief05_candidate"] = _clip01(
                min_global_local_low + 0.05 * max(0.0, local_mean - local_low)
            )
        out["d2_lg_geom_global_first_local_low_candidate"] = math.sqrt(max(0.0, global_first_last) * max(0.0, local_low))
        out["d2_lg_blend_global_first_local_low_50_50_candidate"] = _clip01(0.50 * global_first_last + 0.50 * local_low)
    if current is not None and local_low is not None:
        out["d2_lg_blend_mainline_local_low_75_25_candidate"] = _clip01(0.75 * current + 0.25 * local_low)
        out["d2_lg_blend_mainline_local_low_50_50_candidate"] = _clip01(0.50 * current + 0.50 * local_low)
    if current is not None and pair_low is not None:
        out["d2_lg_blend_mainline_pair_low_75_25_candidate"] = _clip01(0.75 * current + 0.25 * pair_low)
    return out


def _finite_score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return _clip01(float(value))
    return None


def _load_by_video_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out = {}
    for row in d2_base._read_records(path):
        vid = row.get("video_id") if isinstance(row, Mapping) else None
        if isinstance(vid, str) and vid:
            out[vid] = dict(row)
    return out


def _done_ids(path: Path) -> set[str]:
    return d2_base._done_ids(path)


def _extract_feature_tensors(
    frames: list[np.ndarray],
    processor,
    model,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    global_chunks = []
    patch_chunks = []
    patch_grid_shape: list[int] | None = None
    token_shape: list[int] | None = None
    processor_image_shape: list[int] | None = None
    patch_size = int(getattr(model.config, "patch_size", 14))
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)
    for start in range(0, len(frames), batch_size):
        batch = [Image.fromarray(frame) for frame in frames[start : start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]
        processor_image_shape = [int(pixel_values.shape[-2]), int(pixel_values.shape[-1])]
        batch_grid_shape = [int(pixel_values.shape[-2] // patch_size), int(pixel_values.shape[-1] // patch_size)]
        with torch.inference_mode():
            out = model(**inputs)
        hidden = out.last_hidden_state.detach().float().cpu().numpy()
        global_feats, patch_feats = split_dinov2_tokens(hidden, num_register_tokens=num_register_tokens)
        expected_patches = batch_grid_shape[0] * batch_grid_shape[1]
        if patch_feats.shape[1] != expected_patches:
            raise ValueError(
                "DINOv2 patch-token count does not match processor grid: "
                f"patches={patch_feats.shape[1]} grid={batch_grid_shape}"
            )
        patch_grid_shape = batch_grid_shape
        token_shape = [int(value) for value in hidden.shape[1:]]
        global_chunks.append(global_feats)
        patch_chunks.append(patch_feats)
    meta = {
        "d2_lg_processor_image_shape": processor_image_shape,
        "d2_lg_patch_grid_shape": patch_grid_shape,
        "d2_lg_hidden_token_shape": token_shape,
        "d2_lg_patch_size": patch_size,
        "d2_lg_num_register_tokens": num_register_tokens,
        "d2_lg_processor_contract": "HF AutoImageProcessor local config; resize/center-crop before DINOv2 patch tokens",
    }
    return np.concatenate(global_chunks, axis=0), np.concatenate(patch_chunks, axis=0), meta


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
    current_mainline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    video_path = str(row["video_path"])
    frames, sample_meta = d2_base._sample_frames(
        video_path,
        sample_policy=sample_policy,
        sample_fps=sample_fps,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    out: dict[str, Any] = {
        "video_id": str(row["video_id"]),
        "video_path": video_path,
        "path": video_path,
        "feature_status": "ok" if frames is not None else "decode_failed",
        "model": row.get("model", ""),
        "family_id": row.get("family_id", ""),
        "variant_id": row.get("variant_id", ""),
        "camera_type": row.get("camera_type", ""),
        "event_tier": row.get("event_tier") or row.get("event") or "",
        "d2_lg_learning_status": LEARNING_STATUS,
        "d2_lg_formula_family": FORMULA_FAMILY,
        "d2_lg_constants": {
            "hard_cut_threshold": HARD_CUT_THRESHOLD,
            "local_low_percentile": LOCAL_LOW_PERCENTILE,
            "video_low_percentile": VIDEO_LOW_PERCENTILE,
            "pair_median_weight": PAIR_MEDIAN_WEIGHT,
            "pair_low_weight": PAIR_LOW_WEIGHT,
        },
        **sample_meta,
    }
    if frames is None:
        return merge_candidate_fields(out, compute_local_global_candidate(np.empty((0, 1)), np.empty((0, 1, 1))), current_mainline=current_mainline)

    out["d2_lg_original_frame_shape"] = [int(value) for value in frames.shape[1:]]
    global_features, patch_features, feature_meta = _extract_feature_tensors(
        [frame for frame in frames],
        processor,
        model,
        device,
        batch_size,
    )
    candidate = compute_local_global_candidate(global_features, patch_features)
    candidate.update(feature_meta)
    return merge_candidate_fields({**out, **dict(row)}, candidate, current_mainline=current_mainline)


def _field_present(row: Mapping[str, Any], key: str) -> bool:
    return key in row and row.get(key) is not None


def qa_rows(rows: list[Mapping[str, Any]], requested: int) -> dict[str, Any]:
    candidate_keys = [
        CANDIDATE_FIELD,
        "d2_lg_pair_median",
        "d2_lg_pair_p20",
        "d2_lg_long_score",
        "d2_lg_local_mean_median",
        "d2_lg_local_low_p20",
        "d2_lg_hard_cut_min",
        "d2_lg_sampled_frame_count",
        "d2_lg_lowest_pair_indices",
        "d2_lg_method",
        "d2_lg_learned_head",
        "d2_lg_formula_family",
        "d2_vbench_like_global_adjacent_mean",
        "d2_vbench_like_global_first_last",
        "d2_lg_min_global_first_local_low_candidate",
        SELECTED_VISUAL_INTEGRITY_FIELD,
        "d2_lg_min_global_first_local_low_relief05_candidate",
        "d2_lg_geom_global_first_local_low_candidate",
        "d2_lg_blend_global_first_local_low_50_50_candidate",
    ]
    finite_candidate_count = 0
    out_of_bounds = []
    missing_fields = []
    for row in rows:
        for key in candidate_keys:
            if key not in row:
                missing_fields.append({"video_id": row.get("video_id"), "field": key})
        candidate_value = row.get(CANDIDATE_FIELD)
        if isinstance(candidate_value, (int, float)) and math.isfinite(float(candidate_value)):
            finite_candidate_count += 1
        for key in (
            CANDIDATE_FIELD,
            SELECTED_VISUAL_INTEGRITY_FIELD,
            "d2_lg_pair_median",
            "d2_lg_pair_p20",
            "d2_lg_hard_cut_min",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                if not (0.0 <= float(value) <= 1.0):
                    out_of_bounds.append({"video_id": row.get("video_id"), "field": key, "value": value})
    rows_written = len(rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested": int(requested),
        "rows_written": rows_written,
        "rows_requested_equals_written": int(requested) == rows_written,
        "candidate_field": CANDIDATE_FIELD,
        "selected_visual_integrity_field": SELECTED_VISUAL_INTEGRITY_FIELD,
        "selected_alias_of": "d2_lg_min_global_first_local_low_candidate",
        "protected_mainline_field": PROTECTED_MAINLINE_FIELD,
        "protected_mainline_overwritten": False,
        "all_candidate_scores_finite": finite_candidate_count == rows_written if rows_written else False,
        "all_candidate_scores_in_0_1": not out_of_bounds,
        "out_of_bounds": out_of_bounds[:20],
        "missing_audit_fields": missing_fields[:50],
        "camera_type_counts": _counts(row.get("camera_type", "") for row in rows),
        "event_tier_counts": _counts(row.get("event_tier", "") for row in rows),
        "model_family_counts": _counts(row.get("model", row.get("family_id", "")) for row in rows),
        "mainline_joined_count": sum(1 for row in rows if _field_present(row, "d2_current_mainline_score")),
        "method": METHOD_NAME,
        "learning_status": LEARNING_STATUS,
        "constants": {
            "hard_cut_threshold": HARD_CUT_THRESHOLD,
            "local_low_percentile": LOCAL_LOW_PERCENTILE,
            "video_low_percentile": VIDEO_LOW_PERCENTILE,
            "pair_median_weight": PAIR_MEDIAN_WEIGHT,
            "pair_low_weight": PAIR_LOW_WEIGHT,
        },
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Video manifest JSON/JSONL.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local DINOv2 model directory.")
    parser.add_argument("--current-mainline-jsonl", type=Path)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-policy", choices=("fixed_count", "time_fps"), required=True)
    parser.add_argument("--sample-fps", type=float, required=True)
    parser.add_argument("--min-frames", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
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
    rows = d2_base._video_rows(args.videos)
    if args.limit:
        rows = rows[: args.limit]

    if args.qa_only:
        existing_rows = _read_jsonl(args.out_jsonl)
        summary = qa_rows(existing_rows, requested=len(rows) or len(existing_rows))
        args.out_jsonl.with_suffix(".qa.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["all_candidate_scores_in_0_1"] and not summary["missing_audit_fields"] else 2

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
    model.eval()

    current_by_id = _load_by_video_id(args.current_mainline_jsonl)
    done = _done_ids(args.out_jsonl) if args.resume else set()
    rows_to_run = [row for row in rows if row.get("video_id") not in done]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    errored = 0
    with args.out_jsonl.open(mode, encoding="utf-8") as f:
        for idx, row in enumerate(rows_to_run, start=1):
            try:
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
                    current_by_id.get(str(row.get("video_id"))),
                )
            except Exception as exc:
                errored += 1
                out = {
                    "video_id": str(row.get("video_id", "")),
                    "video_path": str(row.get("video_path") or row.get("path") or ""),
                    "feature_status": "error",
                    "error": repr(exc),
                    "d2_lg_method": METHOD_NAME,
                    "d2_lg_learned_head": False,
                }
            f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            if idx % 10 == 0:
                print(f"processed {idx}/{len(rows_to_run)} this_run, total_done={len(done) + idx}/{len(rows)}", flush=True)

    all_rows = _read_jsonl(args.out_jsonl)
    summary = qa_rows(all_rows, requested=len(rows))
    summary.update(
        {
            "videos": str(args.videos),
            "model_dir": str(args.model_dir),
            "current_mainline_jsonl": str(args.current_mainline_jsonl),
            "out_jsonl": str(args.out_jsonl),
            "already_done_before_run": len(done),
            "ran_this_time": len(rows_to_run),
            "errored_this_time": errored,
            "device": args.device,
            "sample_policy": args.sample_policy,
            "sample_fps": args.sample_fps,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "batch_size": args.batch_size,
        }
    )
    args.out_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    ok = (
        errored == 0
        and summary["rows_requested_equals_written"]
        and summary["all_candidate_scores_in_0_1"]
        and not summary["missing_audit_fields"]
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
