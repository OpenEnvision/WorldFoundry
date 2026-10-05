"""Synthesis facade for the SCOPE runtime.

Re-exports ``DEFAULT_MODEL_DIR`` and ``runtime_root`` so callers can reach the
runtime's asset locations without importing the runtime module directly.
"""

from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import DEFAULT_MODEL_DIR, SCOPERuntime, runtime_root


class SCOPESynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the base-model SCOPE runtime."""

    RUNTIME_CLS = SCOPERuntime
    MODEL_ID = SCOPERuntime.MODEL_ID
    DISPLAY_NAME = SCOPERuntime.DISPLAY_NAME


__all__ = [
    "DEFAULT_MODEL_DIR",
    "SCOPESynthesis",
    "runtime_root",
]
