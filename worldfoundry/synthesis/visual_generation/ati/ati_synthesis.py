"""ATI synthesis facade."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .worldfoundry_runtime import ATIRuntime


class ATISynthesis(BaseSynthesis):
    def __init__(self, runtime: ATIRuntime) -> None:
        super().__init__()
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Any = None, **kwargs: Any) -> ATISynthesis:
        return cls(ATIRuntime.from_pretrained(pretrained_model_path, **kwargs))

    def predict(self, **kwargs: Any) -> Any:
        return self.runtime.predict(**kwargs)

    def plan(self) -> dict[str, Any]:
        return self.runtime.plan()


__all__ = ["ATISynthesis"]
