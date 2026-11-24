"""Public LingBot-VLA API with lazy inference dependency loading."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "LingBotVLASynthesis":
        from .lingbot_vla_synthesis import LingBotVLASynthesis

        return LingBotVLASynthesis
    raise AttributeError(name)


__all__ = ["LingBotVLASynthesis"]
