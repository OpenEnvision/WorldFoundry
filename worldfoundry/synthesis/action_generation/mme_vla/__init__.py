"""Public MME-VLA API with lazy model dependency loading."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MMEVLASynthesis":
        from .mme_vla_synthesis import MMEVLASynthesis

        return MMEVLASynthesis
    raise AttributeError(name)


__all__ = ["MMEVLASynthesis"]
