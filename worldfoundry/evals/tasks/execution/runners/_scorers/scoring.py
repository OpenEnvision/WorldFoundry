"""Shared helpers for selecting and configuring multimodal score models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def package_root() -> Path:
    from worldfoundry.base_models.perception_core.video_text.vqa_score import constants

    return Path(constants.__file__).resolve().parent


def resolve_cache_dir(explicit: str | Path | None = None) -> str:
    if explicit is not None:
        return str(Path(explicit).expanduser())
    return (
        os.environ.get("WORLDFOUNDRY_T2V_METRICS_CACHE_DIR")
        or os.environ.get("WORLDFOUNDRY_HFD_ROOT")
        or str(package_root() / "hf_cache")
    )


def ensure_ffmpeg() -> None:
    from worldfoundry.evals.tasks.execution.runners._scorers.vqa_score._backend import (
        ensure_ffmpeg as _ensure_ffmpeg,
    )

    _ensure_ffmpeg()


def list_all_models() -> list[str]:
    from worldfoundry.evals.tasks.execution.runners._scorers.clip_score import list_all_clipscore_models
    from worldfoundry.evals.tasks.execution.runners._scorers.itm_score import list_all_itmscore_models
    from worldfoundry.evals.tasks.execution.runners._scorers.vqa_score import list_all_vqascore_models

    return list_all_vqascore_models() + list_all_clipscore_models() + list_all_itmscore_models()


def get_score_model(
    model: str = "clip-flant5-xxl",
    *,
    device: str = "cuda",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    from worldfoundry.evals.tasks.execution.runners._scorers.clip_score import CLIPScore, list_all_clipscore_models
    from worldfoundry.evals.tasks.execution.runners._scorers.itm_score import ITMScore, list_all_itmscore_models
    from worldfoundry.evals.tasks.execution.runners._scorers.vqa_score import VQAScore, list_all_vqascore_models

    resolved_cache_dir = cache_dir or resolve_cache_dir()
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=resolved_cache_dir, **kwargs)
    if model in list_all_clipscore_models():
        return CLIPScore(model, device=device, cache_dir=resolved_cache_dir, **kwargs)
    if model in list_all_itmscore_models():
        return ITMScore(model, device=device, cache_dir=resolved_cache_dir, **kwargs)
    raise NotImplementedError(f"unsupported score model: {model!r}")


__all__ = [
    "ensure_ffmpeg",
    "get_score_model",
    "list_all_models",
    "package_root",
    "resolve_cache_dir",
]
