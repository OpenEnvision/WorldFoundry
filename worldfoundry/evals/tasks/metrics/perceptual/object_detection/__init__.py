"""WorldScore object detection success-rate metric."""

from __future__ import annotations

from worldfoundry.evals.tasks.metrics.perceptual.object_detection.wrapper import (
    compute_object_detection_success_rate,
    package_root,
)

__all__ = ["compute_object_detection_success_rate", "package_root"]
