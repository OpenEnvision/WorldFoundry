"""Numerical metrics used by the in-tree WorldBench runtime.

The public WorldBench release calls its foreground overlap metric ``mIoU`` but
the released implementation computes a per-class Dice coefficient.  We keep
the two values separate: ``foreground_miou`` is the conventional IoU and
``foreground_dice`` is the source-compatible diagnostic.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image


def _as_integer_labels(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape HxW, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"{name} contains non-integer labels")
        array = rounded
    array = array.astype(np.int64, copy=False)
    if array.size and int(array.min()) < 0:
        raise ValueError(f"{name} contains negative labels")
    return array


def resize_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a label image with nearest-neighbour interpolation.

    ``size`` is ``(height, width)``.
    """

    labels = _as_integer_labels(labels, name="labels")
    height, width = size
    if labels.shape == (height, width):
        return labels.copy()
    image = Image.fromarray(labels.astype(np.int32), mode="I")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def normalize_dataset_labels(labels: np.ndarray, *, background_label: int = 1) -> np.ndarray:
    """Map the dataset's background label to zero while preserving object IDs."""

    labels = _as_integer_labels(labels, name="ground-truth labels")
    if background_label < 0:
        raise ValueError("background_label must be non-negative")
    if np.any(labels < background_label):
        unique = np.unique(labels).tolist()
        raise ValueError(f"ground-truth labels {unique} cannot be shifted by background label {background_label}")
    return labels - background_label


def object_boxes(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Return inclusive XYXY boxes for every non-background object label."""

    labels = _as_integer_labels(labels, name="labels")
    boxes: dict[int, np.ndarray] = {}
    for object_id in sorted(int(item) for item in np.unique(labels) if int(item) > 0):
        ys, xs = np.nonzero(labels == object_id)
        if not len(xs):
            continue
        boxes[object_id] = np.asarray(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            dtype=np.float32,
        )
    return boxes


def first_visible_object_boxes(label_frames: Sequence[np.ndarray]) -> dict[int, tuple[int, np.ndarray]]:
    """Find the first visible frame and box for every object in a sequence."""

    result: dict[int, tuple[int, np.ndarray]] = {}
    for frame_index, frame in enumerate(label_frames):
        for object_id, box in object_boxes(frame).items():
            result.setdefault(object_id, (frame_index, box))
    return result


def overlap_by_frame(
    predicted: Sequence[np.ndarray],
    target: Sequence[np.ndarray],
    *,
    object_ids: Sequence[int] | None = None,
    smooth: float = 1e-6,
) -> list[dict[str, Any]]:
    """Compute per-object IoU and Dice for aligned label sequences."""

    if len(predicted) != len(target):
        raise ValueError(f"predicted/target frame counts differ: {len(predicted)} != {len(target)}")
    if smooth <= 0:
        raise ValueError("smooth must be positive")

    if object_ids is None:
        ids = sorted(
            {
                int(label)
                for frame in target
                for label in np.unique(_as_integer_labels(frame, name="target frame"))
                if int(label) > 0
            }
        )
    else:
        ids = sorted({int(item) for item in object_ids if int(item) > 0})

    rows: list[dict[str, Any]] = []
    for frame_index, (predicted_frame, target_frame) in enumerate(zip(predicted, target, strict=True)):
        pred = _as_integer_labels(predicted_frame, name="predicted frame")
        gt = _as_integer_labels(target_frame, name="target frame")
        if pred.shape != gt.shape:
            raise ValueError(f"frame {frame_index} shapes differ: {pred.shape} != {gt.shape}")

        per_class_iou: dict[str, float] = {}
        per_class_dice: dict[str, float] = {}
        for object_id in ids:
            pred_mask = pred == object_id
            gt_mask = gt == object_id
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())
            pred_area = int(pred_mask.sum())
            gt_area = int(gt_mask.sum())
            union = pred_area + gt_area - intersection
            iou = 1.0 if union == 0 else intersection / union
            dice = (2.0 * intersection + smooth) / (pred_area + gt_area + smooth)
            per_class_iou[str(object_id)] = float(iou)
            per_class_dice[str(object_id)] = float(dice)

        rows.append(
            {
                "frame_index": frame_index,
                "per_class_iou": per_class_iou,
                "per_class_dice": per_class_dice,
                "foreground_miou": (float(np.mean(list(per_class_iou.values()))) if per_class_iou else None),
                "foreground_dice": (float(np.mean(list(per_class_dice.values()))) if per_class_dice else None),
            }
        )
    return rows


def _rgba_float(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"image must have HxWx3 or HxWx4 shape, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(array.max()) <= 1.0:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    pil = Image.fromarray(array).convert("RGBA")
    height, width = size
    if pil.size != (width, height):
        pil = pil.resize((width, height), resample=Image.Resampling.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def background_rmse_by_frame(
    generated_frames: Sequence[np.ndarray],
    target_rgba_frames: Sequence[np.ndarray],
    target_labels: Sequence[np.ndarray],
) -> list[float | None]:
    """Compute source-compatible RGBA RMSE over ground-truth background pixels."""

    if not (len(generated_frames) == len(target_rgba_frames) == len(target_labels)):
        raise ValueError("generated RGB, target RGBA, and target label frame counts must match")
    values: list[float | None] = []
    for frame_index, (generated, target, labels) in enumerate(
        zip(generated_frames, target_rgba_frames, target_labels, strict=True)
    ):
        labels = _as_integer_labels(labels, name=f"target labels at frame {frame_index}")
        generated_rgba = _rgba_float(generated, labels.shape)
        target_rgba = _rgba_float(target, labels.shape)
        background = labels == 0
        if not np.any(background):
            values.append(None)
            continue
        difference = generated_rgba[background] - target_rgba[background]
        values.append(float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))))
    return values


def mean_available(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(clean)) if clean else None


def question_type(question: str) -> str:
    return "multiple_choice" if re.search(r"\boptions\s*:", question, flags=re.IGNORECASE) else "binary"


def _canonical_answer(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(text.split())


def _question_options(question: str) -> list[str]:
    match = re.search(r"\boptions\s*:\s*(.+)$", question, flags=re.IGNORECASE)
    if not match:
        return []
    return [item.strip().rstrip(".") for item in match.group(1).split(",") if item.strip()]


def answer_is_correct(prediction: Any, answer: Any, question: str) -> bool:
    """Score a WorldBench binary or multiple-choice answer without an LLM judge."""

    predicted = _canonical_answer(prediction)
    expected = _canonical_answer(answer)
    if not predicted:
        return False
    if predicted == expected:
        return True

    kind = question_type(question)
    if kind == "binary" and expected in {"yes", "no"}:
        tokens = re.findall(r"[a-z]+", predicted)
        explicit = [token for token in tokens if token in {"yes", "no"}]
        return bool(explicit) and explicit[0] == expected

    options = _question_options(question)
    if options:
        option_keys = [_canonical_answer(item) for item in options]
        label_match = re.fullmatch(r"(?:option\s*)?([a-z])", predicted)
        if label_match:
            index = ord(label_match.group(1)) - ord("a")
            if 0 <= index < len(option_keys):
                predicted = option_keys[index]
        if predicted == expected:
            return True
        matches = [item for item in option_keys if re.search(rf"\b{re.escape(item)}\b", predicted)]
        return len(matches) == 1 and matches[0] == expected
    return False
