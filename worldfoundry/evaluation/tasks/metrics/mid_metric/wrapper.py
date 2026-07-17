"""WorldFoundry facade for Mutual Information Divergence (multimodal MID)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _compute_pmi_fn() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from mid_core import _compute_pmi

    return _compute_pmi


def compute_multimodal_mid(
    real_image_features: np.ndarray | torch.Tensor,
    text_features: np.ndarray | torch.Tensor,
    fake_image_features: np.ndarray | torch.Tensor,
    *,
    limit: int = 30000,
) -> float:
    """Compute MID from CLIP (or similar) image/text feature tensors."""
    x = torch.as_tensor(real_image_features, dtype=torch.float64)
    y = torch.as_tensor(text_features, dtype=torch.float64)
    x0 = torch.as_tensor(fake_image_features, dtype=torch.float64)
    value = _compute_pmi_fn()(x, y, x0, limit=limit, reduction=True)
    return float(value.item() if hasattr(value, "item") else value)


__all__ = ["compute_multimodal_mid", "package_root"]
