"""PSNR pairwise image quality (torchmetrics backend)."""

from __future__ import annotations

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.perceptual import default_data_range, resolve_device, to_tensor


def compute_psnr(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    data_range: float | None = None,
    device: str | None = None,
) -> float:
    from torchmetrics.image import PeakSignalNoiseRatio

    device_t = resolve_device(device)
    data_range = default_data_range(reference, generated) if data_range is None else data_range
    metric = PeakSignalNoiseRatio(data_range=data_range).to(device_t)
    ref = to_tensor(reference, device_t)
    gen = to_tensor(generated, device_t)
    with __import__("torch").no_grad():
        return float(metric(ref, gen).item())


__all__ = ["compute_psnr"]
