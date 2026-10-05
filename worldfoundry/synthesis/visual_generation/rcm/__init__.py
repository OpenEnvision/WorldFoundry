"""Causal-rCM streaming video / interactive world-model runtime.

The vendored NVlabs rCM inference stack lives under ``rcm_runtime`` (Apache-2.0,
see its ``THIRD_PARTY_NOTICES.md``). This package exposes the WorldFoundry
adapter that gates on checkpoints and builds the official rollout command.
"""

from .worldfoundry_runtime import (
    BLOCKED_REASON,
    DEFAULT_CHECKPOINT_DIR,
    INFERENCE_ENTRYPOINT,
    OFFICIAL_REPO_URL,
    RUNTIME_DIR,
    build_command,
    missing_requirements,
)

__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_CHECKPOINT_DIR",
    "INFERENCE_ENTRYPOINT",
    "OFFICIAL_REPO_URL",
    "RUNTIME_DIR",
    "build_command",
    "missing_requirements",
]
