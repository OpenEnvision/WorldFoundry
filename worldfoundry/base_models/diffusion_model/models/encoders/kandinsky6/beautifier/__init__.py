"""Named prompt beautifiers. The pipeline picks one by ``beautifier.name``.

Public names: ``none``, ``qwen25``.
``qwen35_9b``, ``qwen38_27b``, and ``gigachat`` are registered by the
``kandinsky.beautifiers`` entry point from the internal package.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from worldfoundry.base_models.diffusion_model.models.encoders.kandinsky6.beautifier.base import NullBeautifier, PromptBeautifier
from worldfoundry.base_models.diffusion_model.models.encoders.kandinsky6.beautifier.qwen25 import Qwen25Beautifier

__all__ = (
    "NullBeautifier",
    "PromptBeautifier",
    "Qwen25Beautifier",
    "build_beautifier",
    "register_beautifier",
)

_Builder = Callable[[str | None], PromptBeautifier]
_BUILDERS: dict[str, _Builder] = {}


class _PluginState:
    loaded = False


def register_beautifier(name: str, builder: _Builder) -> None:
    """Add or replace a beautifier. ``builder`` receives ``model_path``."""
    _BUILDERS[name] = builder


def build_beautifier(name: str, *, model_path: str | None = None) -> PromptBeautifier:
    """Construct the named beautifier. Heavy models load on the first ``expand``."""
    _load_plugins()
    builder = _BUILDERS.get(name)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"Unknown beautifier {name!r}. Known: {known}.")
    return builder(model_path)


def _load_plugins() -> None:
    if _PluginState.loaded:
        return
    _PluginState.loaded = True
    for ep in entry_points(group="kandinsky.beautifiers"):
        ep.load()


register_beautifier("none", lambda model_path: NullBeautifier())
register_beautifier("qwen25", lambda model_path: Qwen25Beautifier())
