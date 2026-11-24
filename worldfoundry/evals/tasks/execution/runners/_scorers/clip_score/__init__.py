"""CLIPScore metric for image/video-text similarity scoring."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.runners._scorers.scoring import resolve_cache_dir

FALLBACK_CLIPSCORE_MODELS: tuple[str, ...] = (
    "blip2-itc",
    "blip2-itc-coco",
    "hpsv2",
    "pickscore-v1",
    "internvideo2-1b-stage2-clip",
    "languagebind-video-v1.5",
    "umt-b16-25m-clip",
)


@lru_cache(maxsize=1)
def _backend_list_all_clipscore_models() -> tuple[str, ...] | None:
    try:
        from worldfoundry.evals.tasks.execution.runners._scorers.clip_score.models.clipscore_models import (
            list_all_clipscore_models as _list_all_clipscore_models,
        )
    except ImportError:
        return None
    return tuple(_list_all_clipscore_models())


def list_all_clipscore_models() -> list[str]:
    backend = _backend_list_all_clipscore_models()
    if backend is not None:
        return list(backend)
    return list(FALLBACK_CLIPSCORE_MODELS)


def CLIPScore(
    model: str,
    *,
    device: str = "cuda",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    from worldfoundry.evals.tasks.execution.runners._scorers.clip_score.clipscore import CLIPScore as _CLIPScore

    return _CLIPScore(
        model=model,
        device=device,
        cache_dir=cache_dir or resolve_cache_dir(),
        **kwargs,
    )


def package_root() -> Path:
    return Path(__file__).resolve().parent


__all__ = ["FALLBACK_CLIPSCORE_MODELS", "CLIPScore", "list_all_clipscore_models", "package_root"]
