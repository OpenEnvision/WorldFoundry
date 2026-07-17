"""WorldFoundry facade for FSIM (Feature Similarity Index Measure)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path
from worldfoundry.evaluation.tasks.metrics._shared.perceptual import default_data_range, resolve_device, to_tensor

PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor"


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _fsim_fn() -> Any:
    prepend_import_path(VENDOR_ROOT)
    from piq.fsim import fsim

    return fsim


def compute_fsim(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    data_range: float | None = None,
    chromatic: bool = True,
    device: str | None = None,
) -> float:
    """Compute FSIM/FSIMc between two HxWxC images (higher is better)."""
    device_t = resolve_device(device)
    if data_range is None:
        data_range = default_data_range(reference, generated)
    ref = to_tensor(reference, device_t)
    gen = to_tensor(generated, device_t)
    with __import__("torch").no_grad():
        return float(_fsim_fn()(ref, gen, data_range=data_range, chromatic=chromatic).item())


__all__ = ["compute_fsim", "package_root"]
