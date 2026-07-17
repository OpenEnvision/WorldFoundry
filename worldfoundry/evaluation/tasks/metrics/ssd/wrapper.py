"""WorldFoundry facade for Semantic Similarity Distance (SSD)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _ssd_fn() -> Any:
    from .ssd_core import ssd

    return ssd


def compute_ssd(
    real_outputs: np.ndarray,
    generated_outputs: np.ndarray,
    conditions: np.ndarray,
) -> float:
    """Compute SSD from paired condition/output embedding arrays (TensorFlow backend)."""
    import tensorflow as tf

    y_true = tf.constant(real_outputs, dtype=tf.float32)
    y_predict = tf.constant(generated_outputs, dtype=tf.float32)
    x_true = tf.constant(conditions, dtype=tf.float32)
    total, _, _, _ = _ssd_fn()(y_true, y_predict, x_true)
    return float(total.numpy())


__all__ = ["compute_ssd", "package_root"]
