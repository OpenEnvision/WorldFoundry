"""Unified synthesis facade for native MultiWorld inference."""

from __future__ import annotations

from typing import Any

from worldfoundry.synthesis.base_synthesis import BaseSynthesis

from .ittakestwo_runtime import MultiWorldItTakesTwoRuntime


class MultiWorldItTakesTwoSynthesis(BaseSynthesis):
    """Expose MultiWorld through the standard WorldFoundry synthesis contract."""

    def __init__(
        self,
        runtime_root: str | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        *,
        runtime: MultiWorldItTakesTwoRuntime | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.runtime = runtime or MultiWorldItTakesTwoRuntime.from_pretrained(
            runtime_root=runtime_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

    @property
    def runtime_root(self) -> str:
        return self.runtime.runtime_root

    @property
    def config_path(self) -> str:
        return self.runtime.config_path

    @property
    def checkpoint_path(self) -> str:
        return self.runtime.checkpoint_path

    @property
    def python_executable(self) -> str:
        return self.runtime.python_executable

    @property
    def device(self) -> str:
        return self.runtime.device

    @property
    def defaults(self) -> dict[str, Any]:
        return self.runtime.defaults

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "MultiWorldItTakesTwoSynthesis":
        return cls(runtime=MultiWorldItTakesTwoRuntime.from_pretrained(*args, **kwargs))

    def predict(self, *args: Any, **kwargs: Any):
        return self.runtime.predict(*args, **kwargs)


__all__ = ["MultiWorldItTakesTwoSynthesis"]
