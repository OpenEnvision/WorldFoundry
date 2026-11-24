"""LiveWorld synthesis adapter."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .worldfoundry_runtime import LiveWorldRuntime


class LiveWorldSynthesis(BaseSynthesis):
    def __init__(self, runtime: LiveWorldRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> LiveWorldSynthesis:
        return cls(LiveWorldRuntime.from_pretrained(pretrained_model_path, **kwargs))

    def predict(self, **kwargs: Any) -> Any:
        return self.runtime.predict(**kwargs)

    def plan(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.plan(**kwargs)


__all__ = ["LiveWorldSynthesis"]
