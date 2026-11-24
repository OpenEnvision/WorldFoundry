"""Bounding-box helpers shared by layout and detection metrics."""

from __future__ import annotations

from collections.abc import Sequence


def bbox_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert four-value xyxy or five-value ``(x, y, w, h, extra)`` boxes to xyxy."""
    values = [float(v) for v in box]
    if len(values) == 4:
        x1, y1, x2, y2 = values
        return x1, y1, x2, y2
    if len(values) == 5:
        x, y, w, h, _ = values
        return x, y, x + w, y + h
    raise ValueError(f"expected bbox with 4 or 5 values, got {box!r}")


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union for four-value xyxy or five-value xywh boxes."""
    ax1, ay1, ax2, ay2 = bbox_xyxy(box_a)
    bx1, by1, bx2, by2 = bbox_xyxy(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


__all__ = ["bbox_iou", "bbox_xyxy"]
