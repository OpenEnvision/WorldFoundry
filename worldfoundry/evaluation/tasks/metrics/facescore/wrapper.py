"""WorldFoundry facade for FaceScore face quality metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _facescore_class() -> Any:
    prepend_import_path(PACKAGE_ROOT / "facescore_pkg")
    from FaceScore import FaceScore

    return FaceScore


def FaceScoreModel(
    model_name: str = "FaceScore",
    *,
    med_config: str | None = None,
    device: str = "cuda",
) -> Any:
    """Construct a FaceScore model (requires ImageReward + RetinaFace checkpoints)."""
    return _facescore_class()(model_name, med_config=med_config, device=device)


def compute_facescore(image_path: str | Path, *, model_name: str = "FaceScore", device: str = "cuda") -> float:
    """Score a single image path with FaceScore (higher is better)."""
    model = FaceScoreModel(model_name=model_name, device=device)
    score, _, _ = model.get_reward(str(image_path))
    return float(score)


__all__ = ["FaceScoreModel", "compute_facescore", "package_root"]
