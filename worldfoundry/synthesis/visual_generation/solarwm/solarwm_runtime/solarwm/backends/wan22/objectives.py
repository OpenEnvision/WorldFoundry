"""Allocation-light objective helpers shared by Wan stages.

Torch is imported inside calls so configuration and CLI discovery work in a
minimal installation. The formulas preserve the configured velocity convention:
``x_t = (1-sigma) * clean + sigma * noise`` and target ``noise-clean``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("Wan objective math requires the 'wan' dependency extra") from exc
    return torch


def apply_timestep_shift_array(normalized: Any, shift: float) -> np.ndarray:
    """NumPy reference for fixture tests without a torch runtime."""

    if not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError(f"shift must be finite, got {shift!r}")
    if float(shift) <= 0:
        raise ValueError("shift must be positive")
    value = np.asarray(normalized)
    dtype = np.float64 if value.dtype == np.float64 else np.float32
    value = value.astype(dtype, copy=False)
    return float(shift) * value / (1.0 + (float(shift) - 1.0) * value)


def rectified_interpolate_array(clean: Any, noise: Any, sigma: Any) -> np.ndarray:
    """NumPy reference for ``(1-sigma)*clean + sigma*noise`` in FP32."""

    clean_value = np.asarray(clean)
    noise_value = np.asarray(noise)
    if clean_value.shape != noise_value.shape:
        raise ValueError("clean and noise must have identical shapes")
    sigma_value = np.asarray(sigma, dtype=np.float32)
    while sigma_value.ndim < clean_value.ndim:
        sigma_value = np.expand_dims(sigma_value, -1)
    try:
        sigma_value = np.broadcast_to(sigma_value, clean_value.shape)
    except ValueError as exc:
        raise ValueError("sigma is not prefix-broadcastable to clean") from exc
    result = (1.0 - sigma_value) * clean_value.astype(np.float32) + sigma_value * noise_value.astype(np.float32)
    return result.astype(clean_value.dtype, copy=False)


def apply_timestep_shift(normalized: Any, shift: float) -> Any:
    """Apply the rational FlowMatch timestep shift."""

    torch = _torch()
    value = torch.as_tensor(normalized)
    if not torch.is_floating_point(value):
        value = value.to(torch.float32)
    if not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError(f"shift must be finite, got {shift!r}")
    if float(shift) <= 0:
        raise ValueError("shift must be positive")
    if float(shift) == 1.0:
        return value
    return float(shift) * value / (1.0 + (float(shift) - 1.0) * value)


def rectified_interpolate(clean: Any, noise: Any, sigma: Any) -> Any:
    """Return the rectified-flow point while performing arithmetic in FP32."""

    torch = _torch()
    clean_value = torch.as_tensor(clean)
    noise_value = torch.as_tensor(noise, device=clean_value.device)
    if clean_value.shape != noise_value.shape:
        raise ValueError("clean and noise must have identical shapes")
    sigma_value = torch.as_tensor(sigma, device=clean_value.device, dtype=torch.float32)
    while sigma_value.ndim < clean_value.ndim:
        sigma_value = sigma_value.unsqueeze(-1)
    try:
        sigma_value = torch.broadcast_to(sigma_value, clean_value.shape)
    except RuntimeError as exc:
        raise ValueError("sigma is not prefix-broadcastable to clean") from exc
    result = (1.0 - sigma_value) * clean_value.float() + sigma_value * noise_value.float()
    return result.to(clean_value.dtype)


__all__ = [
    "apply_timestep_shift",
    "apply_timestep_shift_array",
    "rectified_interpolate",
    "rectified_interpolate_array",
]
