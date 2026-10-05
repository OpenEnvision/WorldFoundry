"""Perceptual Path Length (PPL) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_ppl = lazy_export(f"{__name__}.compute", "compute_ppl", owner=__name__)

METRIC_ID = "ppl"
ALIASES = ("perceptual-path-length", "perceptual_path_length")
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = ("distribution", "image_generation", "generative_model")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Perceptual Path Length for generative model latent-space smoothness (lower is better).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_ppl

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "compute", "compute_ppl"]
