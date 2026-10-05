"""WorldFoundry synthesis wrapper for the Yume 1.5 visual-generation runtime."""

from __future__ import annotations

from ._facade import YumeFacadeSynthesis


class Yume1p5Synthesis(YumeFacadeSynthesis):
    """Thin WorldFoundry synthesis wrapper around the Yume 1.5 runtime."""

    @classmethod
    def _runtime_cls(cls) -> type:
        from .worldfoundry_runtime import Yume1p5Runtime

        return Yume1p5Runtime


__all__ = ["Yume1p5Synthesis"]
