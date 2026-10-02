"""WorldFoundry runtime adapter for the Open Dreamer (Dreamer 4) world model.

Open Dreamer ships under an all-rights-reserved notice, so no upstream source is
vendored here. This package only contains WorldFoundry-authored glue that binds a
user-staged official checkout to the shared world-model runtime manifest.
"""

from .worldfoundry_runtime import (
    BLOCKED_REASON,
    OFFICIAL_INFERENCE_REPO_URL,
    OFFICIAL_REPO_URL,
    RUNTIME_DIR,
    SOURCE_ENV_VAR,
    build_command,
    missing_requirements,
    resolved_runtime_report,
    runtime_root,
)

__all__ = [
    "BLOCKED_REASON",
    "OFFICIAL_INFERENCE_REPO_URL",
    "OFFICIAL_REPO_URL",
    "RUNTIME_DIR",
    "SOURCE_ENV_VAR",
    "build_command",
    "missing_requirements",
    "resolved_runtime_report",
    "runtime_root",
]
