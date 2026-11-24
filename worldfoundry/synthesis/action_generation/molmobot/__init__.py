"""Public MolmoBot API with lazy inference dependency loading."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MolmoBotSynthesis":
        from .molmobot_synthesis import MolmoBotSynthesis

        return MolmoBotSynthesis
    raise AttributeError(name)


__all__ = ["MolmoBotSynthesis"]
