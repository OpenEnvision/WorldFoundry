"""WorldFoundry facade for CMMD (CLIP Maximum Mean Discrepancy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldfoundry.evals.tasks.metrics.distribution.cmmd import distance, embedding, io_util

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def compute_cmmd(
    reference_dir: str | Path,
    eval_dir: str | Path,
    *,
    batch_size: int = 32,
    max_count: int = -1,
    ref_embed_file: str | Path | None = None,
    device: str | None = None,
) -> float:
    """Compute CMMD between two image directories.

    Args:
        reference_dir: Directory of reference (.jpg/.png) images.
        eval_dir: Directory of generated images to evaluate.
        batch_size: CLIP embedding batch size.
        max_count: Max images per directory (-1 for all).
        ref_embed_file: Optional precomputed reference embeddings (.npy).
        device: Torch device string (``cuda`` / ``cpu``). Defaults to auto.

    Returns:
        CMMD distance (lower is better).
    """
    if ref_embed_file is not None and reference_dir:
        raise ValueError("ref_embed_file and reference_dir are mutually exclusive")
    model = embedding.ClipEmbeddingModel(device=device)
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(str(reference_dir), model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = io_util.compute_embeddings_for_dir(str(eval_dir), model, batch_size, max_count).astype("float32")
    value = distance.mmd(ref_embs, eval_embs)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def compute_cmmd_from_embeddings(
    reference_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
) -> float:
    """Compute CMMD directly from precomputed CLIP embeddings."""
    value = distance.mmd(
        np.asarray(reference_embeddings, dtype=np.float32),
        np.asarray(eval_embeddings, dtype=np.float32),
    )
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


__all__ = [
    "compute_cmmd",
    "compute_cmmd_from_embeddings",
    "package_root",
]
