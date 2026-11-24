"""ITMScore metric for image/video-text matching scoring."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.runners._scorers.scoring import resolve_cache_dir

FALLBACK_ITMSCORE_MODELS: tuple[str, ...] = (
    "blip2-itm",
    "blip2-itm-coco",
    "blip2-itm-vitL",
    "internvideo2-1b-stage2-itm",
    "umt-b16-25m-itm",
    "umt-l16-25m-itm",
)


@lru_cache(maxsize=1)
def _backend_list_all_itmscore_models() -> tuple[str, ...] | None:
    try:
        from worldfoundry.evals.tasks.execution.runners._scorers.itm_score.models.itmscore_models import (
            list_all_itmscore_models as _list_all_itmscore_models,
        )
    except ImportError:
        return None
    return tuple(_list_all_itmscore_models())


def list_all_itmscore_models() -> list[str]:
    backend = _backend_list_all_itmscore_models()
    if backend is not None:
        return list(backend)
    return list(FALLBACK_ITMSCORE_MODELS)


def ITMScore(
    model: str = "blip2-itm",
    *,
    device: str = "cuda",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    from worldfoundry.evals.tasks.execution.runners._scorers.itm_score.itmscore import ITMScore as _ITMScore

    return _ITMScore(
        model=model,
        device=device,
        cache_dir=cache_dir or resolve_cache_dir(),
        **kwargs,
    )


def package_root() -> Path:
    return Path(__file__).resolve().parent


__all__ = ["FALLBACK_ITMSCORE_MODELS", "ITMScore", "list_all_itmscore_models", "package_root"]
