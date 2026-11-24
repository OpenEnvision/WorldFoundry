"""Synthesis adapter for LingBot-World-V2."""

from __future__ import annotations

from typing import Any

from ...base_synthesis import BaseSynthesis
from .runtime import LingBotWorldV2Runtime


class LingBotWorldV2Synthesis(BaseSynthesis):
    """Thin synthesis facade over the distributed runtime."""

    def __init__(self, runtime: LingBotWorldV2Runtime) -> None:
        super().__init__()
        self.runtime = runtime
        self.model_id = runtime.model_id
        self.model_name = runtime.model_name
        self.generation_type = runtime.generation_type
        self.device = runtime.device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> LingBotWorldV2Synthesis:
        del args
        return cls(
            LingBotWorldV2Runtime.from_pretrained(
                pretrained_model_path,
                device=device,
                **kwargs,
            )
        )

    def runtime_plan(self, **overrides: Any) -> dict[str, Any]:
        return self.runtime.runtime_plan(**overrides)

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.predict(*args, **kwargs)


__all__ = ["LingBotWorldV2Synthesis"]
