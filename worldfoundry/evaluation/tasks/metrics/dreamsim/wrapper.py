"""WorldFoundry facade for DreamSim perceptual image similarity."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from worldfoundry.evaluation.tasks.metrics._shared.images import ImageInput, load_rgb_image
from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path
from worldfoundry.evaluation.tasks.metrics._shared.perceptual import resolve_device

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=4)
def _load_dreamsim_model(
    cache_dir: str,
    dreamsim_type: str,
    device: str,
) -> tuple[Any, Any]:
    prepend_import_path(PACKAGE_ROOT)
    from dreamsim.model import dreamsim

    model, preprocess = dreamsim(
        pretrained=True,
        cache_dir=cache_dir,
        device=device,
        dreamsim_type=dreamsim_type,
    )
    return model, preprocess


def compute_dreamsim(
    reference: ImageInput,
    generated: ImageInput,
    *,
    cache_dir: str | Path | None = None,
    dreamsim_type: str = "ensemble",
    device: str | None = None,
) -> float:
    """Compute DreamSim perceptual distance between two images (lower is more similar)."""
    device_t = resolve_device(device)
    model_dir = str(
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path("cache/hfd") / "dreamsim"
    )
    model, preprocess = _load_dreamsim_model(model_dir, dreamsim_type, str(device_t))
    ref = preprocess(load_rgb_image(reference)).to(device_t)
    gen = preprocess(load_rgb_image(generated)).to(device_t)
    with torch.no_grad():
        return float(model(ref, gen).item())


__all__ = [
    "compute_dreamsim",
    "package_root",
]
