"""WorldFoundry facade for ArtScore artness metric."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def load_artscore_model(checkpoint_path: str | Path, *, device: str = "cuda", **model_kwargs: Any) -> Any:
    """Load ArtScore ResNet checkpoint."""
    import torch

    from .models import get_resnet

    args = SimpleNamespace(**{"backbone_config": "resnet50", "no_dense_layer": False, "dropout": 0.0, **model_kwargs})
    model = get_resnet(args)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()
    return model


__all__ = ["load_artscore_model", "package_root"]
