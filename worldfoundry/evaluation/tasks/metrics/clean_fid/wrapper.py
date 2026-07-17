"""WorldFoundry facade for Clean-FID / Improved FID."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _compute_fid_fn() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from cleanfid.fid import compute_fid

    return compute_fid


def compute_clean_fid(
    reference: str | Path,
    generated: str | Path,
    *,
    mode: str = "clean",
    model_name: str = "inception_v3",
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> float:
    """Compute Clean-FID between two image directories (lower is better)."""
    import torch

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    compute_fid = _compute_fid_fn()
    return float(
        compute_fid(
            fdir1=str(reference),
            fdir2=str(generated),
            mode=mode,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device_t,
            verbose=verbose,
            **kwargs,
        )
    )


__all__ = [
    "compute_clean_fid",
    "package_root",
]
