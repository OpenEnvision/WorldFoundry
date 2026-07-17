"""SwAV ResNet50 FID backend (self-supervised-gan-eval)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "swav"


@lru_cache(maxsize=1)
def _swav_fid_module() -> Any:
    prepend_import_path(VENDOR_ROOT)
    import fid_score as mod

    return mod


def compute_swav_fid(
    reference_dir: str | Path,
    generated_dir: str | Path,
    *,
    batch_size: int = 50,
    max_size: str = "all",
    device: str | None = None,
) -> float:
    """Compute SwAV ResNet50 FID between two image directories."""
    import torch

    mod = _swav_fid_module()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return float(
        mod.calculate_fid_given_paths(
            [str(reference_dir), str(generated_dir)],
            batch_size,
            max_size,
            dev,
            2048,
        )
    )


__all__ = ["compute_swav_fid"]
