"""WorldFoundry facade for Fréchet Joint Distance (FJD)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _fjd_module() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    import fjd_core as mod

    return mod


def compute_fjd_from_joint_embeddings(
    reference_joint: np.ndarray,
    generated_joint: np.ndarray,
    *,
    alpha: float | None = None,
    image_dim: int = 2048,
    eps: float = 1e-6,
) -> float:
    """Compute FJD from precomputed joint (image+condition) embedding matrices."""
    mod = _fjd_module()
    mu_real, sigma_real = mod.get_embedding_statistics(reference_joint, cuda=False)
    mu_fake, sigma_fake = mod.get_embedding_statistics(generated_joint, cuda=False)
    if alpha is None:
        image_ref = reference_joint[:, :image_dim]
        cond_ref = reference_joint[:, image_dim:]
        alpha = mod.calculate_alpha(image_ref, cond_ref, cuda=False)
    m1, s1, m2, s2 = _scale_joint_stats(
        mu_real, sigma_real, mu_fake, sigma_fake, alpha, image_dim=image_dim
    )
    return float(mod.calculate_fd(m1, s1, m2, s2, cuda=False, eps=eps))


def _scale_joint_stats(mu1, sigma1, mu2, sigma2, alpha, *, image_dim: int):
    mu1 = np.copy(mu1)
    mu2 = np.copy(mu2)
    sigma1 = np.copy(sigma1)
    sigma2 = np.copy(sigma2)
    mu1[image_dim:] = mu1[image_dim:] * alpha
    mu2[image_dim:] = mu2[image_dim:] * alpha
    sigma1[image_dim:, image_dim:] = sigma1[image_dim:, image_dim:] * alpha**2
    sigma1[image_dim:, :image_dim] = sigma1[image_dim:, :image_dim] * alpha
    sigma1[:image_dim, image_dim:] = sigma1[:image_dim, image_dim:] * alpha
    sigma2[image_dim:, image_dim:] = sigma2[image_dim:, image_dim:] * alpha**2
    sigma2[image_dim:, :image_dim] = sigma2[image_dim:, :image_dim] * alpha
    sigma2[:image_dim, image_dim:] = sigma2[:image_dim, image_dim:] * alpha
    return mu1, sigma1, mu2, sigma2


__all__ = ["compute_fjd_from_joint_embeddings", "package_root"]
