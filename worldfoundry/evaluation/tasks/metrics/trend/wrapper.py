"""WorldFoundry facade for TREND (Truncated Generalized Normal Density Estimation)."""

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
def _trend_module() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    import trend_core as mod

    return mod


def compute_trend_jsd(
    reference_activations: np.ndarray,
    generated_activations: np.ndarray,
) -> float:
    """Compute mean TREND JSD between TGND parameter distributions."""
    mod = _trend_module()
    params_ref = mod.estimate_params(reference_activations)
    params_gen = mod.estimate_params(generated_activations)
    jsd_vals = mod.compute_jsd(params_ref, params_gen)
    return float(np.mean(jsd_vals))


def compute_trend(
    reference_dir: str | Path,
    generated_dir: str | Path,
    *,
    batch_size: int = 50,
    n_images: int = 50000,
    cuda: bool | None = None,
) -> float:
    """Compute TREND between two image directories via Inception embeddings."""
    import torch

    mod = _trend_module()
    use_cuda = cuda if cuda is not None else torch.cuda.is_available()
    ref_act = mod.extract_embeddings(str(reference_dir), batch_size=batch_size, cuda=use_cuda, n_images=n_images)
    gen_act = mod.extract_embeddings(str(generated_dir), batch_size=batch_size, cuda=use_cuda, n_images=n_images)
    return compute_trend_jsd(ref_act, gen_act)


__all__ = ["compute_trend", "compute_trend_jsd", "package_root"]
