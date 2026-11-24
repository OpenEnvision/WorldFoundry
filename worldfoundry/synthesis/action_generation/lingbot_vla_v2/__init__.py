"""Public LingBot-VLA-v2 API with lazy inference dependency loading."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "LingBotVLAV2Synthesis":
        from .lingbot_vla_v2_synthesis import LingBotVLAV2Synthesis

        return LingBotVLAV2Synthesis
    raise AttributeError(name)


__all__ = ["LingBotVLAV2Synthesis"]
