"""Embodied-action evaluation (VLA / VA / VAM / WAM).

Closed-loop embodied benchmarks use in-tree simulators and ``worldfoundry-eval embodied run``.
Upstream harness code will be vendored under ``thirdparty/vla-evaluation-harness/``.
"""


# ruff: noqa: F822 - rollout exports are loaded by __getattr__.

from __future__ import annotations

from importlib import import_module
from typing import Any

from .contracts import (
    CAPABILITY_MULTIMODAL_OBSERVATION,
    CAPABILITY_SESSION_CONTROL,
    CAPABILITY_TAGS,
    CAPABILITY_VA_VIDEO_ACTION,
    CAPABILITY_VAM_VIDEO_ACTION_MODELING,
    CAPABILITY_VLA_ACTION_PREDICTION,
    CAPABILITY_VLA_POLICY_ROLLOUT,
    CAPABILITY_WAM_BRANCH,
    CAPABILITY_WAM_RESET,
    CAPABILITY_WAM_WORLD_ACTION_MODELING,
    VLA_VA_WAM_CONTRACT_SCHEMA_VERSION,
    ActionSpaceKind,
    ActionSpaceSpec,
    EmbodiedGenerationSpec,
    EvaluationTrack,
    RequestKind,
    RunnerCapabilities,
    SessionControl,
    VlaVaWamRunnerContract,
    default_capabilities_for_track,
)
from .evaluate import (
    VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION,
    VlaVaWamRunRequest,
    build_vla_va_wam_evaluate_request,
    execute_vla_va_wam_run,
    run_vla_va_wam,
)
from .materialize import (
    DEFAULT_OBSERVATION_FALLBACK_KEYS,
    DEFAULT_SAMPLE_CONTROL_KEYS,
    VLA_VA_WAM_REQUESTS_SCHEMA_VERSION,
    VlaVaWamMaterializedRequests,
    materialize_vla_va_wam_requests,
    request_from_vla_va_wam_sample,
)
from .metrics import (
    DEFAULT_METRIC_IDS,
    ResultFieldMetric,
    default_metric_ids_for_track,
    metric_from_id,
    metric_suite,
)
from .normalizer import normalize_results as normalize_vla_va_wam_results

# Rollout imports require simulator dependencies; offline scoring does not.
_ROLLOUT_EXPORTS = {
    "EmbodiedClosedLoopRunner",
    "build_ai2thor_closed_loop_runner",
    "build_embodied_closed_loop_runner",
    "build_libero_closed_loop_runner",
    "validate_policy_simulator_specs",
}


def __getattr__(name: str) -> Any:
    if name not in _ROLLOUT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.rollout_runner"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _ROLLOUT_EXPORTS)


__all__ = [
    "CAPABILITY_MULTIMODAL_OBSERVATION",
    "CAPABILITY_SESSION_CONTROL",
    "CAPABILITY_TAGS",
    "CAPABILITY_VA_VIDEO_ACTION",
    "CAPABILITY_VAM_VIDEO_ACTION_MODELING",
    "CAPABILITY_VLA_ACTION_PREDICTION",
    "CAPABILITY_VLA_POLICY_ROLLOUT",
    "CAPABILITY_WAM_BRANCH",
    "CAPABILITY_WAM_RESET",
    "CAPABILITY_WAM_WORLD_ACTION_MODELING",
    "DEFAULT_METRIC_IDS",
    "DEFAULT_OBSERVATION_FALLBACK_KEYS",
    "DEFAULT_SAMPLE_CONTROL_KEYS",
    "VLA_VA_WAM_CONTRACT_SCHEMA_VERSION",
    "VLA_VA_WAM_REQUESTS_SCHEMA_VERSION",
    "VLA_VA_WAM_RUN_REQUEST_SCHEMA_VERSION",
    "ActionSpaceKind",
    "ActionSpaceSpec",
    "EmbodiedClosedLoopRunner",
    "EmbodiedGenerationSpec",
    "EvaluationTrack",
    "RequestKind",
    "ResultFieldMetric",
    "RunnerCapabilities",
    "SessionControl",
    "VlaVaWamMaterializedRequests",
    "VlaVaWamRunRequest",
    "VlaVaWamRunnerContract",
    "build_ai2thor_closed_loop_runner",
    "build_embodied_closed_loop_runner",
    "build_libero_closed_loop_runner",
    "build_vla_va_wam_evaluate_request",
    "default_capabilities_for_track",
    "default_metric_ids_for_track",
    "execute_vla_va_wam_run",
    "materialize_vla_va_wam_requests",
    "metric_from_id",
    "metric_suite",
    "normalize_vla_va_wam_results",
    "request_from_vla_va_wam_sample",
    "run_vla_va_wam",
    "validate_policy_simulator_specs",
]
