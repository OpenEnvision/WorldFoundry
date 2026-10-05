"""WorldFoundry synthesis wrapper for the Yume visual-generation runtime."""

from __future__ import annotations

from ._facade import YumeFacadeSynthesis


class YumeSynthesis(YumeFacadeSynthesis):
    """Thin WorldFoundry synthesis wrapper around the Yume runtime."""

    @classmethod
    def _runtime_cls(cls) -> type:
        from .worldfoundry_runtime import YumeRuntime

        return YumeRuntime


__all__ = ["YumeSynthesis"]
