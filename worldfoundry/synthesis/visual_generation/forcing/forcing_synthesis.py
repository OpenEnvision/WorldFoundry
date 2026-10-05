"""Synthesis facades for the official forcing-family runtimes."""

from __future__ import annotations

from typing import Any

from ..runtime_facade import RuntimeAdapterSynthesis
from .runtime import CausalForcingRuntime, SelfForcingRuntime


class _BaseForcingSynthesis(RuntimeAdapterSynthesis):
    """Shared synthesis facade over an official forcing-family runtime."""

    MODEL_ID = ""
    DISPLAY_NAME = ""
    RUNTIME_CLS = SelfForcingRuntime

    def runtime_plan(self) -> dict[str, Any]:
        """Return the runtime's execution plan without running it."""
        return self.runtime.runtime_plan()


class SelfForcingSynthesis(_BaseForcingSynthesis):
    """Synthesis facade for Self-Forcing."""

    MODEL_ID = "self-forcing"
    DISPLAY_NAME = "Self-Forcing"
    RUNTIME_CLS = SelfForcingRuntime


class CausalForcingSynthesis(_BaseForcingSynthesis):
    """Synthesis facade for Causal-Forcing."""

    MODEL_ID = "causal-forcing"
    DISPLAY_NAME = "Causal-Forcing"
    RUNTIME_CLS = CausalForcingRuntime


__all__ = ["CausalForcingSynthesis", "SelfForcingSynthesis"]
