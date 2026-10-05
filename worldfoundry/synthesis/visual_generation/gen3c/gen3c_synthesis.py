from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import Gen3CRuntime


class Gen3CSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the base-model GEN3C runtime."""

    RUNTIME_CLS = Gen3CRuntime


__all__ = ["Gen3CSynthesis"]
