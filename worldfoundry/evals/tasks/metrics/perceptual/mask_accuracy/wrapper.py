"""WorldFoundry facade for mask accuracy."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evals.tasks.metrics.perceptual.mask_accuracy.mask_accuracy_core import (
    compute_mask_accuracy,
    compute_mask_iou,
)

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


__all__ = ["compute_mask_accuracy", "compute_mask_iou", "package_root"]
