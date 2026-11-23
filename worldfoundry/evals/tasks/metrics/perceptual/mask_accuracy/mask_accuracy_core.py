"""Mask accuracy metric utilities."""

from __future__ import annotations

import numpy as np


def compute_mask_accuracy(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Pixel-wise mask accuracy between binary/label masks (higher is better)."""
    pred = np.asarray(predicted_mask)
    gt = np.asarray(ground_truth_mask)
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: {pred.shape} vs {gt.shape}")
    return float(np.mean(pred == gt))


def compute_mask_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Intersection-over-union for binary masks."""
    pred = np.asarray(predicted_mask).astype(bool)
    gt = np.asarray(ground_truth_mask).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
