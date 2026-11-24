"""H3-World action-conditioned interactive world model over the canonical MiniMax-H3 runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .action_script import (
    H3_WORLD_ACTION_PRESETS,
    H3_WORLD_KEYS9,
    annotate_from_keys9,
    null_action_script,
    resolve_action_script,
)

if TYPE_CHECKING:
    from .pipeline_h3_world import H3WorldPipeline

__all__ = [
    "H3_WORLD_ACTION_PRESETS",
    "H3_WORLD_KEYS9",
    "H3WorldPipeline",
    "annotate_from_keys9",
    "null_action_script",
    "resolve_action_script",
]


def __getattr__(name: str) -> Any:
    if name == "H3WorldPipeline":
        from .pipeline_h3_world import H3WorldPipeline

        return H3WorldPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
