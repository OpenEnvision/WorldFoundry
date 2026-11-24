"""Shared t2v_metrics multimodal scorers for benchmark runners.

Vendored CLIPScore, VQAScore, and ITMScore backends used by GenAI-Bench,
PhyGenBench, CameraBench, and other runners that need image/video-text alignment
scoring. Prefer importing from here rather than ``evaluation.tasks.metrics``.
"""

from __future__ import annotations

from .clip_score import CLIPScore, list_all_clipscore_models
from .itm_score import ITMScore, list_all_itmscore_models
from .scoring import ensure_ffmpeg, get_score_model, list_all_models, package_root, resolve_cache_dir
from .vqa_score import VQAScore, list_all_vqascore_models

__all__ = [
    "CLIPScore",
    "ITMScore",
    "VQAScore",
    "ensure_ffmpeg",
    "get_score_model",
    "list_all_clipscore_models",
    "list_all_itmscore_models",
    "list_all_models",
    "list_all_vqascore_models",
    "package_root",
    "resolve_cache_dir",
]
