"""OpenS2V-Nexus evaluation metrics (NexusScore, NaturalScore, GmeScore)."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_gme_score = lazy_export(f"{__name__}.gme", "compute_gme_score", owner=__name__)
compute_gme_score_from_results = lazy_export(f"{__name__}.gme", "compute_gme_score_from_results", owner=__name__)
compute_natural_score = lazy_export(f"{__name__}.natural", "compute_natural_score", owner=__name__)
compute_natural_score_from_results = lazy_export(f"{__name__}.natural", "compute_natural_score_from_results", owner=__name__)
compute_nexus_score = lazy_export(f"{__name__}.nexus", "compute_nexus_score", owner=__name__)
compute_nexus_score_from_results = lazy_export(f"{__name__}.nexus", "compute_nexus_score_from_results", owner=__name__)

METRIC_MODULES = (
    metric_module_from_globals(
        metric_id="nexus_score",
        aliases=("nexusscore", "nexus-score"),
        description=(
            "NexusScore: YOLO-World + GME subject-consistency metric from OpenS2V-Nexus eval."
        ),
        family="scorer",
        higher_is_better=True,
        tags=("scorer", "video_generation", "subject_consistency", "opens2v"),
        implementation="worldfoundry.evaluation.tasks.metrics.opens2v:compute_nexus_score",
    ),
    metric_module_from_globals(
        metric_id="natural_score",
        aliases=("naturalscore", "natural-score"),
        description=(
            "NaturalScore: GPT-4o video naturalness rating (1-5) from OpenS2V-Nexus eval "
            "(requires OPENAI_API_KEY)."
        ),
        family="scorer",
        higher_is_better=True,
        tags=("scorer", "video_generation", "naturalness", "opens2v"),
        implementation="worldfoundry.evaluation.tasks.metrics.opens2v:compute_natural_score",
    ),
    metric_module_from_globals(
        metric_id="gme_score",
        aliases=("gmescore", "gme-score"),
        description=(
            "GmeScore: GME-Qwen2-VL text-video relevance for OpenS2V subject-to-video evaluation."
        ),
        family="scorer",
        higher_is_better=True,
        tags=("scorer", "video_generation", "text_relevance", "opens2v"),
        implementation="worldfoundry.evaluation.tasks.metrics.opens2v:compute_gme_score",
    ),
)

METRIC_MODULE = METRIC_MODULES[0]


def compute(**kwargs: Any) -> dict[str, Any]:
    return compute_nexus_score(**kwargs)


__all__ = [
    "METRIC_MODULE",
    "METRIC_MODULES",
    "compute",
    "compute_gme_score",
    "compute_gme_score_from_results",
    "compute_natural_score",
    "compute_natural_score_from_results",
    "compute_nexus_score",
    "compute_nexus_score_from_results",
]
