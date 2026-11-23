"""Visual-Semantic (VS) Similarity from HDGAN (arXiv:1802.09178)."""

from __future__ import annotations

import numpy as np


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def compute_vs_similarity_matrix(
    image_features: np.ndarray,
    text_features: np.ndarray,
) -> np.ndarray:
    """Pairwise cosine similarity matrix (HDGAN evaluation uses im @ sent.T)."""
    images = _normalize_rows(image_features)
    texts = _normalize_rows(text_features)
    return images @ texts.T


def compute_vs_similarity(
    image_features: np.ndarray,
    text_features: np.ndarray,
    *,
    paired: bool = True,
) -> float:
    """Mean VS similarity for paired image/text embeddings."""
    if paired:
        images = _normalize_rows(image_features)
        texts = _normalize_rows(text_features)
        if images.shape != texts.shape:
            raise ValueError("paired VS similarity requires matching image/text feature shapes")
        return float(np.mean(np.einsum("ij,ij->i", images, texts)))
    return float(np.mean(compute_vs_similarity_matrix(image_features, text_features)))


def compute_vs_similarity_from_scores(scores: np.ndarray, *, paired: bool = True) -> float:
    """Compute VS similarity from a precomputed score matrix."""
    matrix = np.asarray(scores, dtype=np.float64)
    if paired:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("paired VS similarity requires a square score matrix")
        return float(np.mean(np.diag(matrix)))
    return float(np.mean(matrix))


__all__ = [
    "compute_vs_similarity",
    "compute_vs_similarity_from_scores",
    "compute_vs_similarity_matrix",
]
