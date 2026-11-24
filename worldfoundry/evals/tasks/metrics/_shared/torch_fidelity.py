"""Lazy access to torch-fidelity under WorldFoundry's own package namespace."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


def vendor_root() -> Path:
    return _VENDOR_ROOT


@lru_cache(maxsize=1)
def calculate_metrics() -> Any:
    from .vendor.torch_fidelity.metrics import calculate_metrics as _calculate_metrics

    return _calculate_metrics


def prepare_image_input(value: str | Path | Sequence[str | Path], **kwargs: Any) -> Any:
    """Adapt explicit file lists using the backend's directory preprocessing rules."""
    if isinstance(value, (str, Path)):
        return str(value)

    from torchvision import transforms

    from .vendor.torch_fidelity.datasets import ImagesPathDataset, TransformPILtoRGBTensor
    from .vendor.torch_fidelity.helpers import get_kwarg

    resize = get_kwarg("samples_resize_and_crop", kwargs)
    steps = []
    if resize > 0:
        steps.extend((transforms.Resize(resize), transforms.CenterCrop(resize)))
    steps.append(TransformPILtoRGBTensor())
    return ImagesPathDataset(list(value), transforms.Compose(steps))


__all__ = ["calculate_metrics", "prepare_image_input", "vendor_root"]
