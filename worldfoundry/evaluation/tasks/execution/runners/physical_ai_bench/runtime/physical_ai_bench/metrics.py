"""Source-compatible CPU metrics for PAI-Bench-C and answer metrics for G/U."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def _resize_spatial(array: np.ndarray, height: int, width: int, *, nearest: bool) -> np.ndarray:
    import cv2

    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return np.stack([cv2.resize(frame, (width, height), interpolation=interpolation) for frame in array])


def align_video(pred: np.ndarray, gt: np.ndarray, *, nearest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Trim to the common duration and resize prediction to the reference resolution."""

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    frame_count = min(len(pred), len(gt))
    if frame_count == 0:
        raise ValueError("cannot score an empty video")
    pred = pred[:frame_count]
    gt = gt[:frame_count]
    if pred.shape[1:3] != gt.shape[1:3]:
        pred = _resize_spatial(pred, gt.shape[1], gt.shape[2], nearest=nearest)
    return pred, gt


def canny_edges(frames: np.ndarray, *, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    import cv2

    edges: list[np.ndarray] = []
    for frame in np.asarray(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        edges.append(cv2.Canny(gray.astype(np.uint8), low_threshold, high_threshold))
    return np.stack(edges)


def canny_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Global binary precision/recall/F1, matching the public PAI-Bench release."""

    pred, gt = align_video(np.asarray(pred), np.asarray(gt), nearest=True)
    pred_binary = pred > 0
    gt_binary = gt > 0
    tp = int(np.logical_and(pred_binary, gt_binary).sum())
    fp = int(np.logical_and(pred_binary, ~gt_binary).sum())
    fn = int(np.logical_and(~pred_binary, gt_binary).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "canny_f1_score": float(f1),
        "canny_precision": float(precision),
        "canny_recall": float(recall),
    }


def bilateral_blur(
    frames: np.ndarray,
    *,
    diameter: int = 30,
    sigma_color: float = 150.0,
    sigma_space: float = 100.0,
) -> np.ndarray:
    import cv2

    return np.stack(
        [cv2.bilateralFilter(frame.astype(np.uint8), diameter, sigma_color, sigma_space) for frame in frames]
    )


def blur_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean full-map RGB SSIM used by the conditional blur task."""

    from skimage.metrics import structural_similarity

    pred, gt = align_video(np.asarray(pred), np.asarray(gt), nearest=False)
    scores: list[float] = []
    for pred_frame, gt_frame in zip(pred, gt, strict=True):
        data_range = float(gt_frame.max()) - float(gt_frame.min())
        if data_range <= 0:
            scores.append(float(np.array_equal(pred_frame, gt_frame)))
            continue
        _, score_map = structural_similarity(
            gt_frame,
            pred_frame,
            data_range=data_range,
            full=True,
            channel_axis=2 if gt_frame.ndim == 3 else None,
        )
        scores.append(float(np.asarray(score_map).mean()))
    return float(np.mean(scores))


def depth_si_rmse(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    per_pixel_error_cap: float | None = 10.0,
) -> float:
    """Median-scale-aligned per-frame RMSE from the public PAI-Bench implementation."""

    pred, gt = align_video(
        np.asarray(pred_depth, dtype=np.float64),
        np.asarray(gt_depth, dtype=np.float64),
        nearest=True,
    )
    valid_region = np.ones_like(gt, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    valid_region, gt = align_video(valid_region, gt, nearest=True)
    pred = pred[: len(gt)]
    frame_scores: list[float] = []
    for pred_frame, gt_frame, region in zip(pred, gt, valid_region, strict=True):
        valid = region & (gt_frame > 0) & (pred_frame > 0)
        if not np.any(valid):
            continue
        pred_values = pred_frame[valid]
        gt_values = gt_frame[valid]
        median = float(np.median(pred_values))
        if median <= 0:
            continue
        residual = gt_values - pred_values * (float(np.median(gt_values)) / median)
        if per_pixel_error_cap is not None:
            residual = np.clip(residual, -per_pixel_error_cap, per_pixel_error_cap)
        frame_scores.append(float(np.sqrt(np.mean(np.square(residual)))))
    return float(np.mean(frame_scores)) if frame_scores else 0.0


def _decode_rle(rle: Any, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    if isinstance(rle, np.ndarray):
        return rle.astype(bool)
    if isinstance(rle, list):
        return np.asarray(rle, dtype=bool)
    if not isinstance(rle, Mapping):
        raise TypeError(f"unsupported segmentation payload: {type(rle).__name__}")
    if "mask" in rle:
        return np.asarray(rle["mask"], dtype=bool)
    data = rle.get("data", rle.get("_data", rle))
    if isinstance(data, Mapping) and isinstance(data.get("counts"), str):
        data = dict(data)
        data["counts"] = data["counts"].encode("ascii")
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required to decode PAI-Bench segmentation RLE") from exc
    decoded = np.asarray(mask_utils.decode(data), dtype=bool)
    if shape is not None and decoded.size == int(np.prod(shape)):
        decoded = decoded.reshape(shape)
    elif decoded.ndim == 2:
        decoded = decoded[None]
    elif decoded.ndim == 3 and shape is not None and decoded.shape[:2] == shape[1:]:
        decoded = np.moveaxis(decoded, -1, 0)
    return decoded


def _phrase_masks(records: Iterable[Mapping[str, Any]]) -> list[np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        phrase = str(record.get("phrase", "")).strip().lower()
        payload = record.get("segmentation_mask_rle", record.get("mask", record.get("masks")))
        shape_value = payload.get("mask_shape") if isinstance(payload, Mapping) else None
        shape = tuple(int(item) for item in shape_value) if shape_value is not None else None
        grouped[phrase].append(_decode_rle(payload, shape=shape))
    unions: list[np.ndarray] = []
    for masks in grouped.values():
        frame_count = min(len(mask) for mask in masks)
        target_h, target_w = masks[0].shape[1:3]
        aligned = []
        for mask in masks:
            mask = mask[:frame_count]
            if mask.shape[1:3] != (target_h, target_w):
                mask = _resize_spatial(mask.astype(np.uint8), target_h, target_w, nearest=True) > 0
            aligned.append(mask)
        unions.append(np.logical_or.reduce(aligned))
    return unions


def segmentation_scores(
    gt_records: Iterable[Mapping[str, Any]],
    pred_records: Iterable[Mapping[str, Any]],
    *,
    threshold: float = 0.1,
) -> dict[str, float]:
    """Phrase-union mask matching with Hungarian assignment and 0.1 IoU acceptance."""

    from scipy.optimize import linear_sum_assignment

    gt_masks = _phrase_masks(gt_records)
    pred_masks = _phrase_masks(pred_records)
    if not gt_masks or not pred_masks:
        return {"seg_m_iou": 0.0, "seg_recall": 0.0}
    matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    for row, gt_mask in enumerate(gt_masks):
        for column, pred_mask in enumerate(pred_masks):
            frame_count = min(len(gt_mask), len(pred_mask))
            candidate = pred_mask[:frame_count]
            reference = gt_mask[:frame_count]
            if candidate.shape[1:3] != reference.shape[1:3]:
                candidate = (
                    _resize_spatial(candidate.astype(np.uint8), reference.shape[1], reference.shape[2], nearest=True)
                    > 0
                )
            union = np.logical_or(reference, candidate).sum()
            matrix[row, column] = np.logical_and(reference, candidate).sum() / union if union else 0.0
    rows, columns = linear_sum_assignment(1.0 - matrix)
    matched = matrix[rows, columns]
    accepted = matched[matched > threshold]
    return {
        "seg_m_iou": float(accepted.mean()) if accepted.size else 0.0,
        "seg_recall": float(accepted.size / matched.size) if matched.size else 0.0,
    }


_OPTION_RE = re.compile(r"(?<![A-Z0-9])([A-E])(?![A-Z0-9])", re.IGNORECASE)


def parse_option_answer(value: Any) -> str | None:
    """Parse a multiple-choice label used by generation VQA answer mappings."""

    if value is None:
        return None
    text = str(value).strip().upper()
    text = re.sub(
        r"^(?:THE\s+)?(?:BEST|CORRECT)?\s*(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*",
        "",
        text,
    )
    match = _OPTION_RE.search(text)
    return match.group(1).upper() if match else None


def normalize_binary_answer(value: Any, index2ans: Mapping[str, Any] | None = None) -> str | None:
    if index2ans:
        text_key = str(value).strip() if value is not None else ""
        mapped = index2ans.get(text_key, index2ans.get(text_key.upper(), index2ans.get(text_key.lower())))
        option = parse_option_answer(value)
        if mapped is None and option:
            mapped = index2ans.get(option, index2ans.get(option.lower()))
        if mapped is not None:
            value = mapped
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "yes"
    if text in {"no", "false", "0", "n"}:
        return "no"
    contains_yes = re.search(r"\byes\b", text) is not None
    contains_no = re.search(r"\bno\b", text) is not None
    if contains_yes != contains_no:
        return "yes" if contains_yes else "no"
    return None


def aggregate_generation_vqa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average seed accuracy per question, then questions per video, then videos globally."""

    question_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    categories: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("correct") is None:
            continue
        video_id = str(row.get("video_id") or row.get("uid") or "unknown")
        question_id = str(row.get("question_id") or row.get("uid") or row.get("question") or "unknown")
        question_groups[(video_id, question_id)].append(bool(row["correct"]))
        category = str(row.get("category") or row.get("task") or "misc")
        categories[category].add(video_id)
    video_questions: dict[str, list[float]] = defaultdict(list)
    for (video_id, _), values in question_groups.items():
        video_questions[video_id].append(sum(values) / len(values))
    video_scores = {video_id: sum(values) / len(values) for video_id, values in video_questions.items()}
    overall = sum(video_scores.values()) / len(video_scores) if video_scores else None
    category_scores: dict[str, float] = {}
    for category, video_ids in sorted(categories.items()):
        values = [video_scores[video_id] for video_id in video_ids if video_id in video_scores]
        if values:
            category_scores[f"{category}_score"] = sum(values) / len(values)
    return {
        "vqa_accuracy": overall,
        "category_scores": category_scores,
        "video_scores": video_scores,
        "sample_count": len(rows),
    }
