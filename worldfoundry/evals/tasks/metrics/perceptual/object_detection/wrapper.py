"""WorldFoundry facade for WorldScore object detection metric."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evals.tasks.metrics.perceptual.object_detection.object_detection_core import (
    compute_object_detection_success_rate,
)

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


__all__ = ["compute_object_detection_success_rate", "package_root"]
