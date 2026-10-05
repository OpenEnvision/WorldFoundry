"""Independent Echo-Memory synthesis models with lazy Wan runtime imports."""

from __future__ import annotations

from typing import Any

from .echo_memory_synthesis import (
    EchoMemoryContextK1Synthesis,
    EchoMemorySynthesis,
)
from .memory import EchoRolloutMemory

__all__ = [
    "EchoMemoryContextK1Synthesis",
    "EchoMemoryRuntime",
    "EchoMemorySynthesis",
    "EchoRolloutMemory",
]


def __getattr__(name: str) -> Any:
    if name == "EchoMemoryRuntime":
        from .runtime import EchoMemoryRuntime

        return EchoMemoryRuntime
    raise AttributeError(name)
