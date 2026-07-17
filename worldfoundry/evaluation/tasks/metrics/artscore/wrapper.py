"""WorldFoundry facade for ArtScore artness metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _get_resnet() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from models import get_resnet

    return get_resnet


def load_artscore_model(checkpoint_path: str | Path, *, device: str = "cuda", **model_kwargs: Any) -> Any:
    """Load ArtScore ResNet checkpoint."""
    import torch

    class _Args:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    args = _Args(no_dense_layer=model_kwargs.get("no_dense_layer", False), **model_kwargs)
    model = _get_resnet()(args)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()
    return model


__all__ = ["load_artscore_model", "package_root"]
