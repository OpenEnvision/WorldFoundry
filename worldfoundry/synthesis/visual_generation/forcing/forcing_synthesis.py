"""Synthesis facade shared by the official forcing-family runtimes."""

from __future__ import annotations

from typing import Any

from ..runtime_facade import RuntimeAdapterSynthesis


class _BaseForcingSynthesis(RuntimeAdapterSynthesis):
    """Shared synthesis facade over an official forcing-family runtime."""

    MODEL_ID = ""
    DISPLAY_NAME = ""

    def runtime_plan(self) -> dict[str, Any]:
        """Return the runtime's execution plan without running it."""
        return self.runtime.runtime_plan()
