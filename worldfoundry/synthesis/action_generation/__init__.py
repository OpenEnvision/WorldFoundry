"""Action, policy, and latent-action synthesis packages.

Concrete synthesis classes are exposed from their model-specific packages.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ActionModelSynthesis":
        from .base_action_synthesis import ActionModelSynthesis

        return ActionModelSynthesis
    if name == "DreamDojoSynthesis":
        from .dreamdojo import DreamDojoSynthesis

        return DreamDojoSynthesis
    raise AttributeError(name)

__all__ = ["ActionModelSynthesis", "DreamDojoSynthesis"]
