"""WorldFoundry facade for Feature Likelihood Divergence (FLD)."""

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
def _fld_class() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from fld.metrics.FLD import FLD

    return FLD


def _to_tensor(features: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    return torch.as_tensor(features, dtype=torch.float32)


def compute_fld(
    train_features: np.ndarray | torch.Tensor,
    test_features: np.ndarray | torch.Tensor,
    generated_features: np.ndarray | torch.Tensor,
    *,
    eval_feat: str = "test",
) -> float:
    """Compute FLD from precomputed feature tensors (lower is better)."""
    fld = _fld_class()(eval_feat=eval_feat)
    return float(
        fld.compute_metric(
            _to_tensor(train_features),
            _to_tensor(test_features),
            _to_tensor(generated_features),
        )
    )


__all__ = ["compute_fld", "package_root"]
