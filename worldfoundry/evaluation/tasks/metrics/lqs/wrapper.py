"""Layout Quality Score (LQS) from arXiv:2208.06162."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.bbox import bbox_xyxy


def _bbox_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox_xyxy(box)
    return abs(x2 - x1) * abs(y2 - y1)


def _match_boxes(
    groundtruth: Sequence[dict[str, Any]],
    predicted: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    from scipy.optimize import linear_sum_assignment

    gt_by_label: dict[str, list[dict[str, Any]]] = {}
    pred_by_label: dict[str, list[dict[str, Any]]] = {}
    for item in groundtruth:
        gt_by_label.setdefault(str(item["label"]), []).append(item)
    for item in predicted:
        pred_by_label.setdefault(str(item["label"]), []).append(item)
    shared = set(gt_by_label) & set(pred_by_label)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for label in sorted(shared):
        gt_items = gt_by_label[label]
        pred_items = pred_by_label[label]
        if len(gt_items) == 1 and len(pred_items) == 1:
            pairs.append((gt_items[0], pred_items[0]))
            continue
        gt_centers = np.array([_bbox_center(item["bbox"]) for item in gt_items])
        pred_centers = np.array([_bbox_center(item["bbox"]) for item in pred_items])
        costs = np.linalg.norm(gt_centers[:, None] - pred_centers[None, :], axis=-1)
        gt_indices, pred_indices = linear_sum_assignment(costs)
        pairs.extend((gt_items[i], pred_items[j]) for i, j in zip(gt_indices, pred_indices))
    return pairs


def compute_lqs(
    groundtruth_layout: Sequence[dict[str, Any]],
    predicted_layout: Sequence[dict[str, Any]],
    *,
    image_area: float = 80.0,
    sigma_l: float = 1.0,
    gamma_lc: float = 0.25,
    gamma_ac: float = 0.25,
) -> dict[str, float]:
    """Compute LQS using ``(x1, y1, x2, y2)`` or ``(x, y, w, h, extra)`` boxes."""
    if not groundtruth_layout:
        raise ValueError("groundtruth_layout must be non-empty")
    pairs = _match_boxes(groundtruth_layout, predicted_layout)
    lr = len(pairs) / len(groundtruth_layout)
    lp = len(pairs) / len(predicted_layout) if predicted_layout else 0.0
    if not pairs:
        return {"lqs": lr + lp, "lr": lr, "lp": lp, "lc": 0.0, "ac": 0.0}
    alc_values = []
    rlc_values = []
    gt_centers = []
    pred_centers = []
    gt_areas = []
    pred_areas = []
    for gt_item, pred_item in pairs:
        gt_center = np.array(_bbox_center(gt_item["bbox"]))
        pred_center = np.array(_bbox_center(pred_item["bbox"]))
        gt_centers.append(gt_center)
        pred_centers.append(pred_center)
        gt_areas.append(_bbox_area(gt_item["bbox"]))
        pred_areas.append(_bbox_area(pred_item["bbox"]))
        alc_values.append(float(np.linalg.norm(gt_center - pred_center)))
    alc = float(np.mean(alc_values))
    rel_terms = []
    for i, j in combinations(range(len(pairs)), 2):
        gt_rel = gt_centers[i] - gt_centers[j]
        pred_rel = pred_centers[i] - pred_centers[j]
        rel_terms.append(float(np.linalg.norm(gt_rel - pred_rel)))
    rlc = float(np.mean(rel_terms)) if rel_terms else 0.0
    lc = gamma_lc * np.exp(-alc / (2.0 * sigma_l**2)) + (1.0 - gamma_lc) * np.exp(-rlc / (2.0 * sigma_l**2))
    aac = 1.0 - float(np.mean([abs(p - g) / image_area for g, p in zip(gt_areas, pred_areas)]))
    rac_terms = []
    for i, j in combinations(range(len(pairs)), 2):
        gt_cmp = gt_areas[i] > gt_areas[j]
        pred_cmp = pred_areas[i] > pred_areas[j]
        rac_terms.append(1.0 - abs(int(gt_cmp) - int(pred_cmp)))
    rac = float(np.mean(rac_terms)) if rac_terms else 1.0
    ac = gamma_ac * aac + (1.0 - gamma_ac) * rac
    lqs = lr + lp + float(lc) + float(ac)
    return {
        "lqs": float(lqs),
        "lr": float(lr),
        "lp": float(lp),
        "lc": float(lc),
        "ac": float(ac),
        "alc": alc,
        "rlc": rlc,
        "aac": float(aac),
        "rac": rac,
    }


__all__ = ["compute_lqs"]
