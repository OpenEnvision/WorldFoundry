"""Local backend for Render Quality."""

from __future__ import annotations

from ..io import value_digest
from ..skills.skill_render_quality import (
    BACKEND_METRICS,
    BACKEND_OUTPUT_SCHEMA,
    COMPONENTS,
    SAMPLING,
)
from .common import LocalMetricBackend


BACKEND_ID = "harnesseval.render_quality"
BACKEND_VERSION = "harnesseval.render_quality.four_metrics"
CONFIG_DIGEST = value_digest(
    {
        "implementation": "harnesseval",
        "metrics": list(BACKEND_METRICS.values()),
        "models": {
            "aesthetic_quality": "clip-vit-l14-laion-aesthetic-head",
            "imaging_quality": "musiq",
            "hpsv3_quality": "hpsv3-qwen2-vl-7b",
            "temporal_flickering": "rgb-frame-mae",
        },
    }
)
METRIC_SAMPLING = {
    BACKEND_METRICS[component]: SAMPLING[component] for component in COMPONENTS
}

__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "CONFIG_DIGEST",
    "LocalBackend",
]


class LocalBackend(LocalMetricBackend):
    backend_id = BACKEND_ID
    version = BACKEND_VERSION
    config_digest = CONFIG_DIGEST
    output_schema = BACKEND_OUTPUT_SCHEMA
    metric_sampling = METRIC_SAMPLING
