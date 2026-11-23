"""Conditional Inception Score (CIS: BCIS / WCIS) from class-conditional softmax predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, 1e-12, None)
    return arr / arr.sum(axis=-1, keepdims=True)


def _inception_score(probs: np.ndarray, *, splits: int = 10) -> float:
    probs = _normalize_probs(probs)
    scores: list[float] = []
    n = probs.shape[0]
    if n == 0:
        raise ValueError("CIS requires at least one prediction vector")
    split_size = max(1, n // splits)
    for index in range(splits):
        start = index * split_size
        end = n if index == splits - 1 else min(n, (index + 1) * split_size)
        part = probs[start:end]
        if part.size == 0:
            continue
        py = np.mean(part, axis=0)
        kl = np.sum(part * (np.log(part) - np.log(py)), axis=1)
        scores.append(float(np.exp(np.mean(kl))))
    return float(np.mean(scores)) if scores else float("nan")


def compute_bcis(
    class_probs: Mapping[Any, np.ndarray] | Sequence[tuple[Any, np.ndarray]],
    *,
    splits: int = 10,
) -> float:
    """Between-class CIS component: exp(E_c[KL(p(y|c) || p(y))])."""
    items = list(class_probs.items()) if isinstance(class_probs, Mapping) else list(class_probs)
    if not items:
        raise ValueError("class_probs must be non-empty")
    stacked = np.stack([_normalize_probs(probs) for _, probs in items], axis=0)
    marginal = np.mean(stacked, axis=0)
    kl_terms = []
    for probs in stacked:
        kl_terms.append(np.sum(probs * (np.log(probs) - np.log(marginal))))
    bcis = float(np.exp(np.mean(kl_terms)))
    return bcis


def compute_wcis(
    class_probs: Mapping[Any, np.ndarray] | Sequence[tuple[Any, np.ndarray]],
    *,
    splits: int = 10,
) -> float:
    """Within-class CIS component: exp(E_c[IS(X_c; Y_c)])."""
    items = list(class_probs.items()) if isinstance(class_probs, Mapping) else list(class_probs)
    within_scores = [_inception_score(probs, splits=splits) for _, probs in items if len(probs) > 0]
    if not within_scores:
        raise ValueError("WCIS requires at least one non-empty class bucket")
    return float(np.exp(np.mean(np.log(np.clip(within_scores, 1e-12, None)))))


def compute_cis(
    class_probs: Mapping[Any, np.ndarray] | Sequence[tuple[Any, np.ndarray]],
    *,
    splits: int = 10,
) -> dict[str, float]:
    """Compute CIS = BCIS * WCIS and component scores (Benny et al., IJCV 2020)."""
    bcis = compute_bcis(class_probs, splits=splits)
    wcis = compute_wcis(class_probs, splits=splits)
    return {
        "cis": float(bcis * wcis),
        "bcis": bcis,
        "wcis": wcis,
    }


def compute_cis_from_predictions(
    predictions: np.ndarray,
    labels: Sequence[Any],
    *,
    splits: int = 10,
) -> dict[str, float]:
    """Bucket softmax predictions by class label and compute CIS."""
    preds = _normalize_probs(predictions)
    if preds.shape[0] != len(labels):
        raise ValueError("predictions and labels must have the same length")
    buckets: dict[Any, list[np.ndarray]] = {}
    for label, prob in zip(labels, preds, strict=True):
        buckets.setdefault(label, []).append(prob)
    class_probs = {label: np.stack(values, axis=0) for label, values in buckets.items()}
    return compute_cis(class_probs, splits=splits)


__all__ = [
    "compute_bcis",
    "compute_cis",
    "compute_cis_from_predictions",
    "compute_wcis",
]
