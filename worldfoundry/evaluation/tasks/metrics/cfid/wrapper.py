"""WorldFoundry facade for Conditional FID (CFID)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _cfid_fn() -> Any:
    from .cfid_core import cfid

    return cfid


def compute_cfid(
    real_outputs: np.ndarray,
    generated_outputs: np.ndarray,
    conditions: np.ndarray,
) -> float:
    """Compute CFID from paired condition/output embedding arrays (TensorFlow backend)."""
    import tensorflow as tf

    y_true = tf.constant(real_outputs, dtype=tf.float32)
    y_predict = tf.constant(generated_outputs, dtype=tf.float32)
    x_true = tf.constant(conditions, dtype=tf.float32)
    value = _cfid_fn()(y_true, y_predict, x_true)
    return float(value.numpy())


__all__ = ["compute_cfid", "package_root"]
