"""Local backend for Motion Quality."""

from __future__ import annotations

from ..io import value_digest
from ..skills.skill_motion_quality import (
    BACKEND_METRICS,
    BACKEND_OUTPUT_SCHEMA,
    COMPONENTS,
    SAMPLING,
)
from .common import LocalMetricBackend


BACKEND_ID = "harnesseval.motion_quality"
BACKEND_VERSION = "harnesseval.motion_quality.two_metrics"
CONFIG_DIGEST = value_digest(
    {
        "implementation": "harnesseval",
        "metrics": list(BACKEND_METRICS.values()),
        "models": {
            "dynamic_degree": "worldfoundry-raft-things-required",
            "motion_smoothness": "amt-s",
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
