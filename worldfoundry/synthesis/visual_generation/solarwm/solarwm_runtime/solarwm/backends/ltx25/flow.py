"""Dependency-light LTX native rectified-flow math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import BackendContractError
from .geometry import STABLE_GEOMETRY


def _floating_pair(clean: object, noise: object) -> tuple[np.ndarray, np.ndarray]:
    clean_array = np.asarray(clean)
    noise_array = np.asarray(noise)
    if clean_array.shape != noise_array.shape:
        raise BackendContractError("clean and noise shapes must match")
    if not np.issubdtype(clean_array.dtype, np.floating) or not np.issubdtype(noise_array.dtype, np.floating):
        raise BackendContractError("clean and noise must use floating dtypes")
    return clean_array, noise_array


def _sigma(value: object, sample: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise BackendContractError("sigma must be finite and in [0,1]")
    while result.ndim < sample.ndim:
        result = np.expand_dims(result, -1)
    try:
        np.broadcast_shapes(result.shape, sample.shape)
    except ValueError as exc:
        raise BackendContractError("sigma is not broadcastable to the sample") from exc
    return result


def scale_noise(clean: object, noise: object, sigma: object) -> np.ndarray:
    """Return ``x_sigma=(1-sigma)*x0 + sigma*eps`` in FP32."""

    clean_array, noise_array = _floating_pair(clean, noise)
    expanded = _sigma(sigma, clean_array)
    return (1.0 - expanded) * clean_array.astype(np.float32) + expanded * noise_array.astype(np.float32)


def predict_clean(noisy: object, velocity: object, sigma: object) -> np.ndarray:
    noisy_array, velocity_array = _floating_pair(noisy, velocity)
    return noisy_array.astype(np.float32) - _sigma(sigma, noisy_array) * velocity_array.astype(np.float32)


def restore_clean_first_latent(noisy: object, first_frame: object) -> np.ndarray:
    """Copy the clean first latent slice into one ``[B,128,20,16,24]`` sample."""

    sample = np.asarray(noisy)
    first = np.asarray(first_frame)
    expected = (
        STABLE_GEOMETRY.latent_channels,
        STABLE_GEOMETRY.latent_frames,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
    )
    if sample.ndim != 5 or sample.shape[1:] != expected:
        raise BackendContractError(f"noisy sample must be [B,{expected}]")
    if not np.issubdtype(sample.dtype, np.floating) or not np.issubdtype(first.dtype, np.floating):
        raise BackendContractError("latent inputs must use floating dtypes")
    if first.shape != (sample.shape[0], *STABLE_GEOMETRY.first_frame_latent_shape):
        raise BackendContractError("first_frame must be [B,128,1,16,24]")
    result = sample.copy()
    result[:, :, :1] = first
    return result


@dataclass(frozen=True)
class NativeFlowContract:
    interpolation: str = "(1-sigma)*x0+sigma*eps"
    target: str = "eps-x0"
    clean_first_sigma: float = 0.0
    loss_latent_slice: str = "1:20"


NATIVE_FLOW_CONTRACT = NativeFlowContract()


__all__ = [
    "NATIVE_FLOW_CONTRACT",
    "NativeFlowContract",
    "predict_clean",
    "restore_clean_first_latent",
    "scale_noise",
]
