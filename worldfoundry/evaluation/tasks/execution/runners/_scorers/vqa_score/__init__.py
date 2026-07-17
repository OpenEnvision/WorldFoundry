"""VQAScore metric for image/video-text alignment scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.runners._scorers.scoring import resolve_cache_dir


def list_all_vqascore_models() -> list[str]:
    from worldfoundry.base_models.perception_core.video_text.vqa_score.models.vqascore_models import (
        list_all_vqascore_models as _list_all_vqascore_models,
    )

    return list(_list_all_vqascore_models())


def VQAScore(
    model: str = "clip-flant5-xxl",
    *,
    device: str = "cuda",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    from worldfoundry.evaluation.tasks.execution.runners._scorers.vqa_score.vqascore import VQAScore as _VQAScore

    return _VQAScore(
        model=model,
        device=device,
        cache_dir=cache_dir or resolve_cache_dir(),
        **kwargs,
    )


def package_root() -> Path:
    """Return the in-tree VQAScore implementation root."""
    return Path(__file__).resolve().parent


__all__ = ["VQAScore", "list_all_vqascore_models", "package_root"]
