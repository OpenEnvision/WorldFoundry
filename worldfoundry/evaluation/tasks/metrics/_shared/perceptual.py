"""Shared tensor helpers for pairwise perceptual metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert image pixels to a float NCHW tensor, preserving their original scale."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HxWxC image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor.unsqueeze(0).to(device)


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_data_range(reference: np.ndarray, generated: np.ndarray) -> float:
    if reference.dtype == np.uint8 or generated.dtype == np.uint8:
        return 255.0
    return 1.0 if max(reference.max(), generated.max()) <= 1.0 else 255.0


def compute_perceptual_bundle(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
    device: str | None = None,
) -> dict[str, float]:
    from worldfoundry.evaluation.tasks.metrics.lpips import compute_lpips
    from worldfoundry.evaluation.tasks.metrics.ms_ssim import compute_ms_ssim
    from worldfoundry.evaluation.tasks.metrics.psnr import compute_psnr
    from worldfoundry.evaluation.tasks.metrics.ssim import compute_ssim

    return {
        "lpips": compute_lpips(reference, generated, net_type=net_type, device=device),
        "ssim": compute_ssim(reference, generated, device=device),
        "ms_ssim": compute_ms_ssim(reference, generated, device=device),
        "psnr": compute_psnr(reference, generated, device=device),
    }


__all__ = ["compute_perceptual_bundle", "default_data_range", "resolve_device", "to_tensor"]
