"""KID computation via in-tree torch-fidelity (Inception-v3 kernel MMD)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics, prepare_image_input


def _parse_numeric_result(result: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def _polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    gamma: float | None,
    coef0: float,
) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef0) ** degree


def polynomial_mmd(
    features_x: np.ndarray,
    features_y: np.ndarray,
    *,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
) -> float:
    """Unbiased polynomial-kernel MMD^2 between two feature sets.

    This is the kernel distance behind KID (Inception features) and KVD
    (I3D video features); benchmark runtimes such as MiraBench delegate here
    instead of vendoring their own kernel code. Defaults match
    ``sklearn.metrics.pairwise.polynomial_kernel``.
    """
    x = np.asarray(features_x, dtype=np.float64)
    y = np.asarray(features_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("polynomial_mmd expects 2-D feature arrays (N, D)")
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("polynomial_mmd needs at least two samples per set")
    k_xx = _polynomial_kernel(x, x, degree=degree, gamma=gamma, coef0=coef0)
    k_yy = _polynomial_kernel(y, y, degree=degree, gamma=gamma, coef0=coef0)
    k_xy = _polynomial_kernel(x, y, degree=degree, gamma=gamma, coef0=coef0)
    k_xx_sum = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    k_yy_sum = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    k_xy_sum = k_xy.sum() / (m * n)
    return float(k_xx_sum + k_yy_sum - 2.0 * k_xy_sum)


def compute_kid(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> dict[str, float]:
    result = calculate_metrics()(
        input1=prepare_image_input(reference, **kwargs),
        input2=prepare_image_input(generated, **kwargs),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        kid=True,
        **kwargs,
    )
    return _parse_numeric_result(result)


__all__ = ["compute_kid", "polynomial_mmd"]
