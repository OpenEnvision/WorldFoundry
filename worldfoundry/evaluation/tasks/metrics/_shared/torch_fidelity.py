"""Lazy torch-fidelity loader with vendored package on ``sys.path``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .imports import prepend_import_path

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_torch_fidelity() -> None:
    prepend_import_path(_VENDOR_ROOT)


@lru_cache(maxsize=1)
def calculate_metrics() -> Any:
    ensure_torch_fidelity()
    from torch_fidelity.metrics import calculate_metrics as _calculate_metrics

    return _calculate_metrics


__all__ = ["calculate_metrics", "ensure_torch_fidelity", "vendor_root"]
