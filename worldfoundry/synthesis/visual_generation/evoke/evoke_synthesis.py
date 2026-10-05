"""Synthesis adapter for the bundled Evoke runtime."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .worldfoundry_runtime import EvokeRuntime


class EvokeSynthesis(BaseSynthesis):
    """Expose :class:`EvokeRuntime` through the synthesis contract."""

    def __init__(self, runtime: EvokeRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> "EvokeSynthesis":
        return cls(EvokeRuntime.from_pretrained(pretrained_model_path, **kwargs))

    def predict(self, **kwargs: Any) -> Any:
        return self.runtime.predict(**kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.plan(**kwargs)

    def preflight(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.preflight(**kwargs)


__all__ = ["EvokeSynthesis"]
