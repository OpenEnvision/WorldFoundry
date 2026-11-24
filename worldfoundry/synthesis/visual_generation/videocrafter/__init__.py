"""VideoCrafter visual-synthesis models and runtime."""

from __future__ import annotations

from .components import VideoCrafterComponents, load_videocrafter_components
from .videocrafter1_i2v_synthesis import VideoCrafter1I2VSynthesis
from .videocrafter1_t2v_synthesis import VideoCrafter1T2VSynthesis
from .videocrafter2_t2v_synthesis import VideoCrafter2T2VSynthesis

__all__ = [
    "VideoCrafter",
    "VideoCrafter1I2VSynthesis",
    "VideoCrafter1T2VSynthesis",
    "VideoCrafter2T2VSynthesis",
    "VideoCrafterComponents",
    "load_videocrafter_components",
    "resolve_runtime_config",
]


def __getattr__(name: str):
    if name in {"VideoCrafter", "resolve_runtime_config"}:
        from .worldfoundry_runtime import VideoCrafter, resolve_runtime_config

        return {"VideoCrafter": VideoCrafter, "resolve_runtime_config": resolve_runtime_config}[name]
    raise AttributeError(name)
