"""WorldFoundry synthesis wrapper for NeoVerse navigation video generation."""

from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import DEFAULT_PROMPT, NeoVerseOfficialRuntime


class NeoVerseSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the official NeoVerse runtime."""

    RUNTIME_CLS = NeoVerseOfficialRuntime

    @classmethod
    def bundled_runtime_root(cls) -> str:
        """Return the directory the NeoVerse runtime is bundled under."""
        return NeoVerseOfficialRuntime.bundled_runtime_root()


__all__ = ["DEFAULT_PROMPT", "NeoVerseSynthesis"]
