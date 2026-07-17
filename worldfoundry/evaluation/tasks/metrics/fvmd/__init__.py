"""Fréchet Video Motion Distance (FVMD) metric."""

from __future__ import annotations

import tempfile
from pathlib import Path

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"

METRIC_ID = "fvmd"
ALIASES = ("frechet-video-motion-distance",)
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = ("distribution", "video_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Fréchet Video Motion Distance between reference and generated video sets.",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute_fvmd(
    reference_videos: str | Path,
    generated_videos: str | Path,
    *,
    log_dir: str | Path | None = None,
) -> float:
    prepend_import_path(_VENDOR_ROOT)
    from fvmd.fvmd import fvmd as _fvmd

    work_dir = Path(log_dir) if log_dir is not None else Path(tempfile.mkdtemp(prefix="worldfoundry-fvmd-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    return float(_fvmd(str(work_dir), str(generated_videos), str(reference_videos)))


compute = compute_fvmd

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "compute", "compute_fvmd"]
