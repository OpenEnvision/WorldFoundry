"""LPIPS pairwise perceptual distance (torchmetrics backend)."""

from __future__ import annotations

import numpy as np

from worldfoundry.evals.tasks.metrics._shared.perceptual import default_data_range, resolve_device, to_tensor
from worldfoundry.evals.tasks.metrics.perceptual.lpips import NetType


def compute_lpips(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    net_type: NetType = "alex",
    device: str | None = None,
) -> float:
    LearnedPerceptualImagePatchSimilarity = __import__(
        "torchmetrics.image.lpip", fromlist=["LearnedPerceptualImagePatchSimilarity"]
    ).LearnedPerceptualImagePatchSimilarity

    device_t = resolve_device(device)
    metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True).to(device_t)
    data_range = default_data_range(reference, generated)
    ref = to_tensor(reference, device_t) / data_range
    gen = to_tensor(generated, device_t) / data_range
    with __import__("torch").no_grad():
        return float(metric(ref, gen).item())


__all__ = ["NetType", "compute_lpips"]
