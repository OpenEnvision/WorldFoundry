"""FaceSim-Cur subject-consistency metric (OpenS2V-Eval)."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_facesim_cur = lazy_export(f"{__name__}.wrapper", "compute_facesim_cur", owner=__name__)
compute_facesim_cur_from_results = lazy_export(f"{__name__}.wrapper", "compute_facesim_cur_from_results", owner=__name__)

METRIC_ID = "facesim_cur"
ALIASES = ("face-sim-cur", "facesim-curricular", "face_sim_cur")
HIGHER_IS_BETTER = True
FAMILY = "scorer"
TAGS = ("scorer", "video_generation", "subject_consistency", "opens2v")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "FaceSim-Cur: CurricularFace cosine similarity between subject image and video frames "
        "(OpenS2V-Nexus eval; requires InsightFace + OpenS2V weights)."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(**kwargs: Any) -> dict[str, Any]:
    return compute_facesim_cur(**kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_facesim_cur",
    "compute_facesim_cur_from_results",
]
